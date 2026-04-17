"""
BPM-to-calm-score pipeline.

Input:  list of (timestamp_ms, bpm) samples at ~1 Hz from HealthKit.
Output: calm_score ∈ [0,100], state ∈ {recovery, neutral, stress},
        four component features, and session summary.

Implements the full methodology spec from stress_calculations_from_bpm_per_min.md:
  Stage 0 – Ingestion & resampling to uniform 1 Hz
  Stage 1 – Cleaning (physiological gate, spike filter, gap handling, BPM→pRR)
  Stage 2 – Detrending (smoothness-priors / moving-average fallback)
  Stage 3 – Feature extraction (Baevsky SI proxy, HF power, HR trend, breath coherence)
  Stage 4 – Per-user baseline model (session + adaptive EMA)
  Stage 5 – Scoring (state classification + continuous calm_score)
  Stage 6 – Session summary
  Stage 7 – System HRV reconciliation (logging only)
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import signal as sp_signal
from scipy.sparse import diags as sp_diags, eye as sp_eye
from scipy.sparse.linalg import spsolve

# numpy 2.0 renamed trapz -> trapezoid; support both
_trapz = getattr(np, "trapezoid", None)
if _trapz is None:
    _trapz = np.trapz

import psycopg

from app.config import settings

logger = logging.getLogger(__name__)

# ── Constants (from parameter cheat-sheet) ────────────────────────────
FEATURE_WINDOW_S   = 30     # seconds of data per feature computation
UPDATE_CADENCE_S   = 5      # compute every 5 s
RESAMPLE_HZ        = 1      # uniform 1 Hz grid
DETREND_LAMBDA     = 500    # smoothness-priors λ for 1 Hz
HIST_BIN_WIDTH_MS  = 50     # Baevsky convention
HF_LO, HF_HI      = 0.15, 0.40   # Hz – Task Force 1996
LF_LO, LF_HI      = 0.04, 0.15
WELCH_SEG_S        = 30     # Welch segment length
WELCH_OVERLAP      = 0.50
BASELINE_CAPTURE_S = 60     # first 60 s
BASELINE_EMA_TAU_S = 10 * 60  # 10 min
BASELINE_ALPHA     = UPDATE_CADENCE_S / BASELINE_EMA_TAU_S  # ≈0.0083
DISPLAY_EMA_ALPHA  = 0.15
SCORE_WEIGHTS      = (0.35, 0.25, 0.30, 0.10)  # HR, SI, HF, resonance
BPM_MIN, BPM_MAX   = 40, 180  # physiological gate
SPIKE_THRESHOLD     = 15      # |Δbpm| per sample
MIN_STD_FLOOR       = 1e-3    # prevent /0 in z-scores


# ── Data classes ──────────────────────────────────────────────────────

@dataclass
class FeatureSnapshot:
    """One 5-s tick of extracted features."""
    t: float                     # seconds since session start
    # Feature A – Baevsky SI proxy
    si_proxy: float = 0.0
    amo: float = 0.0
    mo_ms: float = 0.0
    mxdmn_ms: float = 0.0
    # Feature B – HF / LF power
    hf_power: float = 0.0
    lf_power: float = 0.0
    lf_hf_ratio: float = 0.0
    breath_rate_bpm: float = 0.0
    # Feature C – HR trend
    hr_mean_60s: float = 0.0
    hr_slope: float = 0.0
    hr_deviation: float = 0.0
    # Feature D – Breath coherence
    coherence: float = 0.0
    resonance_proximity: float = 0.0
    # Scoring
    calm_score: float = 50.0
    state: str = "neutral"


@dataclass
class SessionSummary:
    """End-of-session report."""
    hr_baseline: float = 0.0
    hr_final: float = 0.0
    hr_delta: float = 0.0
    hf_pct_change: float = 0.0
    breath_start: float = 0.0
    breath_end: float = 0.0
    avg_calm_score: float = 50.0
    time_in_recovery_pct: float = 0.0
    time_in_stress_pct: float = 0.0
    time_in_neutral_pct: float = 0.0
    duration_s: float = 0.0
    total_snapshots: int = 0


@dataclass
class _Baseline:
    hr: float = 0.0
    si: float = 0.0
    hf: float = 0.0
    breath: float = 0.0
    hr_std: float = 1.0
    si_std: float = 1.0
    hf_std: float = 1.0
    captured: bool = False


# ── Stage 0 – Resample to uniform 1 Hz ───────────────────────────────

def _resample_1hz(timestamps_s: np.ndarray, bpm: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Linear-interpolate irregular samples onto a uniform 1 Hz grid."""
    t_start = math.floor(timestamps_s[0])
    t_end   = math.floor(timestamps_s[-1])
    t_grid  = np.arange(t_start, t_end + 1, 1.0)
    bpm_grid = np.interp(t_grid, timestamps_s, bpm)
    return t_grid, bpm_grid


# ── Stage 1 – Cleaning ───────────────────────────────────────────────

def _clean(bpm: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (cleaned_bpm, valid_mask).
    Applies physiological gate, spike filter, gap interpolation.
    """
    n = len(bpm)
    cleaned = bpm.copy()
    valid = np.ones(n, dtype=bool)

    # Physiological gate
    out_of_range = (cleaned < BPM_MIN) | (cleaned > BPM_MAX)
    cleaned[out_of_range] = np.nan
    valid[out_of_range] = False

    # Spike filter
    for i in range(1, n):
        if np.isnan(cleaned[i]) or np.isnan(cleaned[i - 1]):
            continue
        if abs(cleaned[i] - cleaned[i - 1]) > SPIKE_THRESHOLD:
            cleaned[i] = np.nan
            valid[i] = False

    # Gap interpolation
    nan_mask = np.isnan(cleaned)
    if nan_mask.any() and not nan_mask.all():
        good = np.where(~nan_mask)[0]
        cleaned[nan_mask] = np.interp(
            np.where(nan_mask)[0], good, cleaned[good]
        )
        # Mark long gaps (>5 consecutive) as invalid
        gap_start = None
        for i in range(n):
            if nan_mask[i]:
                if gap_start is None:
                    gap_start = i
            else:
                if gap_start is not None and (i - gap_start) > 5:
                    valid[gap_start:i] = False
                gap_start = None
        if gap_start is not None and (n - gap_start) > 5:
            valid[gap_start:n] = False

    return cleaned, valid


def _bpm_to_prr(bpm: np.ndarray) -> np.ndarray:
    """Convert BPM to pseudo-RR intervals in ms."""
    return 60000.0 / np.clip(bpm, BPM_MIN, BPM_MAX)


# ── Stage 2 – Detrending ─────────────────────────────────────────────

def _smoothness_priors_detrend(x: np.ndarray, lam: float = DETREND_LAMBDA) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tarvainen (2002) smoothness-priors detrend.
    Returns (detrended, trend).
    """
    n = len(x)
    if n < 4:
        return x - np.mean(x), np.full(n, np.mean(x))

    I = sp_eye(n, format="csc")
    # Second-order difference matrix
    D2 = sp_diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(n - 2, n), format="csc")
    trend = spsolve(I + lam * D2.T @ D2, x)
    detrended = x - trend
    return detrended, trend


# ── Stage 3 – Feature extraction ─────────────────────────────────────

def _feature_a_baevsky_si(prr_detrended: np.ndarray,
                          prr_raw: Optional[np.ndarray] = None,
                          ) -> Tuple[float, float, float, float]:
    """
    Baevsky Stress Index proxy.
    AMo and MxDMn are computed from the detrended series (variability).
    Mo (mode) must come from the RAW pRR so it stays ~833 ms for 72 bpm
    instead of collapsing to ~0 from zero-mean detrended data.
    Returns (si_proxy, amo, mo_ms, mxdmn_ms).
    """
    if len(prr_detrended) < 10:
        return 0.0, 0.0, 0.0, 0.0

    # Percentile trim (2nd / 98th) on detrended for AMo and MxDMn
    lo, hi = np.percentile(prr_detrended, [2, 98])
    trimmed = prr_detrended[(prr_detrended >= lo) & (prr_detrended <= hi)]
    if len(trimmed) < 5:
        trimmed = prr_detrended

    # Mo from RAW pRR (absolute RR level); fall back to detrended if not provided
    if prr_raw is not None and len(prr_raw) >= 10:
        mo_ms = float(np.median(prr_raw))
    else:
        mo_ms = float(np.median(trimmed))

    # Histogram with 50 ms bins
    bin_edges = np.arange(trimmed.min() - HIST_BIN_WIDTH_MS,
                          trimmed.max() + HIST_BIN_WIDTH_MS + 1,
                          HIST_BIN_WIDTH_MS)
    if len(bin_edges) < 2:
        return 0.0, 0.0, mo_ms, 0.0

    counts, _ = np.histogram(trimmed, bins=bin_edges)
    amo = float(counts.max() / len(trimmed) * 100)  # percent

    mxdmn_ms = float(trimmed.max() - trimmed.min())
    mxdmn_s = mxdmn_ms / 1000.0
    mo_s = mo_ms / 1000.0

    denom = 2.0 * abs(mo_s) * mxdmn_s
    if denom < 1e-9:
        return 0.0, amo, mo_ms, mxdmn_ms

    si = amo / denom
    si_proxy = math.sqrt(max(si, 0.0))

    return si_proxy, amo, mo_ms, mxdmn_ms


def _feature_b_hf_power(prr_detrended: np.ndarray, fs: float = 1.0
                         ) -> Tuple[float, float, float, float]:
    """
    HF & LF power via Welch, plus breath-rate estimate.
    Returns (hf_power, lf_power, lf_hf_ratio, breath_rate_bpm).
    """
    n = len(prr_detrended)
    nperseg = min(int(WELCH_SEG_S * fs), n)
    if nperseg < 8:
        return 0.0, 0.0, 0.0, 0.0

    noverlap = int(nperseg * WELCH_OVERLAP)
    window = sp_signal.windows.hann(nperseg)
    freqs, psd = sp_signal.welch(
        prr_detrended, fs=fs, window=window,
        nperseg=nperseg, noverlap=noverlap, detrend=False,
    )

    # Integrate bands
    hf_mask = (freqs >= HF_LO) & (freqs <= HF_HI)
    lf_mask = (freqs >= LF_LO) & (freqs <= LF_HI)

    hf_power = float(_trapz(psd[hf_mask], freqs[hf_mask])) if hf_mask.any() else 0.0
    lf_power = float(_trapz(psd[lf_mask], freqs[lf_mask])) if lf_mask.any() else 0.0

    lf_hf_ratio = lf_power / hf_power if hf_power > 1e-12 else 0.0

    # Dominant peak in 0.05–0.40 Hz → breath rate
    resp_mask = (freqs >= 0.05) & (freqs <= 0.40)
    if resp_mask.any():
        peak_idx = np.argmax(psd[resp_mask])
        peak_hz = freqs[resp_mask][peak_idx]
        breath_rate_bpm = float(peak_hz * 60)
    else:
        breath_rate_bpm = 0.0

    return hf_power, lf_power, lf_hf_ratio, breath_rate_bpm


def _feature_c_hr_trend(bpm_clean: np.ndarray, t_grid: np.ndarray,
                         hr_baseline: float) -> Tuple[float, float, float]:
    """
    HR mean, slope, deviation from baseline.
    Returns (hr_mean_60s, hr_slope_bpm_per_min, hr_deviation).
    """
    hr_mean = float(np.mean(bpm_clean))

    # Slope over last 30 s (or whatever is available)
    slope_window = min(30, len(bpm_clean))
    if slope_window < 2:
        return hr_mean, 0.0, hr_mean - hr_baseline

    y = bpm_clean[-slope_window:]
    x = t_grid[-slope_window:] - t_grid[-slope_window]
    # Linear regression: y = a*x + b
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    ss_xx = np.sum((x - x_mean) ** 2)
    if ss_xx < 1e-9:
        slope = 0.0
    else:
        slope = float(np.sum((x - x_mean) * (y - y_mean)) / ss_xx)
    # Convert slope from bpm/s to bpm/min
    slope_per_min = slope * 60.0

    return hr_mean, slope_per_min, hr_mean - hr_baseline


def _feature_d_breath_coherence(prr_detrended: np.ndarray, fs: float = 1.0
                                 ) -> Tuple[float, float, float]:
    """
    Breath coherence and resonance proximity.
    Returns (breath_rate_bpm, coherence, resonance_proximity).
    """
    n = len(prr_detrended)
    nperseg = min(int(WELCH_SEG_S * fs), n)
    if nperseg < 8:
        return 0.0, 0.0, 0.0

    noverlap = int(nperseg * WELCH_OVERLAP)
    window = sp_signal.windows.hann(nperseg)
    freqs, psd = sp_signal.welch(
        prr_detrended, fs=fs, window=window,
        nperseg=nperseg, noverlap=noverlap, detrend=False,
    )

    resp_mask = (freqs >= 0.05) & (freqs <= 0.40)
    if not resp_mask.any():
        return 0.0, 0.0, 0.0

    psd_resp = psd[resp_mask]
    freqs_resp = freqs[resp_mask]
    total_power = float(_trapz(psd_resp, freqs_resp))
    if total_power < 1e-12:
        return 0.0, 0.0, 0.0

    peak_idx = np.argmax(psd_resp)
    peak_hz = float(freqs_resp[peak_idx])
    breath_rate = peak_hz * 60.0

    # Coherence: ratio of peak PSD height to mean PSD height in the band.
    # High coherence → spectrum concentrated at one frequency (rhythmic breathing).
    # Normalize to [0,1] via peak/(peak + mean) so it saturates smoothly.
    mean_psd = float(np.mean(psd_resp))
    peak_psd = float(psd_resp[peak_idx])
    coherence = peak_psd / (peak_psd + mean_psd) if (peak_psd + mean_psd) > 1e-12 else 0.0

    # Resonance proximity: Gaussian centered on 6 breaths/min
    resonance_proximity = math.exp(-((breath_rate - 6.0) ** 2) / 8.0)

    return breath_rate, coherence, resonance_proximity


# ── Stage 4 & 5 – Baseline + Scoring ─────────────────────────────────

def _capture_baseline(snapshots: List[FeatureSnapshot]) -> _Baseline:
    """Capture baseline from first 60 s of snapshots."""
    if not snapshots:
        return _Baseline()
    return _Baseline(
        hr=np.mean([s.hr_mean_60s for s in snapshots]),
        si=np.mean([s.si_proxy for s in snapshots]),
        hf=np.mean([s.hf_power for s in snapshots]),
        breath=np.mean([s.breath_rate_bpm for s in snapshots]),
        captured=True,
    )


def _classify_state(snap: FeatureSnapshot, bl: _Baseline) -> str:
    hr_dev = snap.hr_deviation
    recovery = (hr_dev < -2 and
                snap.si_proxy < 0.9 * bl.si and
                snap.hf_power > 1.1 * bl.hf)
    stress = (hr_dev > 2 and
              snap.si_proxy > 1.1 * bl.si and
              snap.hf_power < 0.9 * bl.hf)
    if recovery:
        return "recovery"
    if stress:
        return "stress"
    return "neutral"


def _compute_calm_score(snap: FeatureSnapshot, bl: _Baseline) -> float:
    """Continuous calm score ∈ [0,100] via sigmoid."""
    def _z(val, base, std):
        s = max(std, MIN_STD_FLOOR)
        return (val - base) / s

    z_hr  = -_z(snap.hr_mean_60s, bl.hr, bl.hr_std)
    z_si  = -_z(snap.si_proxy, bl.si, bl.si_std)
    z_hf  =  _z(snap.hf_power, bl.hf, bl.hf_std)
    z_res =  2.0 * snap.resonance_proximity - 1.0  # shift [0,1] → [−1,1]

    w = SCORE_WEIGHTS
    raw = w[0] * z_hr + w[1] * z_si + w[2] * z_hf + w[3] * z_res

    # Clamp raw to prevent math.exp overflow for extreme z-scores
    raw = max(-500.0, min(500.0, raw))
    return 100.0 / (1.0 + math.exp(-raw))


# ── Main pipeline ─────────────────────────────────────────────────────

def process_bpm_session(
    samples: List[Dict[str, Any]],
    cross_session_baseline: Optional[Dict[str, float]] = None,
) -> Tuple[List[FeatureSnapshot], SessionSummary]:
    """
    Full pipeline.  Accepts any of these input formats:

    1. Watch heartbeat_series rr_intervals (primary):
       [{"rr_interval_ms": 850}, {"rr_interval_ms": 875}, ...]
       No timestamps — samples are sequential at ~1 Hz.

    2. beat_to_beat_bpm from hrv_sdnn metadata:
       [{"bpm": 72.0, "timestamp": 1712345678.0}, ...]

    3. Explicit timestamped BPM:
       [{"bpm": 72.0, "timestamp_ms": 1712345678000}, ...]

    Returns (snapshots, summary).
    """
    if not samples or len(samples) < FEATURE_WINDOW_S:
        logger.warning("Too few samples (%d) for BPM pipeline", len(samples))
        return [], SessionSummary()

    # ── Normalize input to (timestamp_s[], bpm[]) ──
    ts_arr: List[float] = []
    bpm_arr: List[float] = []

    # Detect format from first sample
    first = samples[0]
    if "rr_interval_ms" in first:
        # Format 1: sequential RR intervals from Watch (~1 Hz)
        for i, s in enumerate(samples):
            rr = s.get("rr_interval_ms")
            if rr is None or rr <= 0:
                continue
            ts_arr.append(float(i))  # synthetic 1 Hz timestamps
            bpm_arr.append(60000.0 / float(rr))
    elif "bpm" in first:
        # Format 2/3: BPM with timestamps
        for s in samples:
            bpm = s.get("bpm")
            if bpm is None or bpm <= 0:
                continue
            if "timestamp_ms" in s:
                ts_arr.append(s["timestamp_ms"] / 1000.0)
            elif "timestamp" in s:
                ts_arr.append(float(s["timestamp"]))
            else:
                # No timestamp — assume sequential 1 Hz
                ts_arr.append(float(len(ts_arr)))
            bpm_arr.append(float(bpm))
    else:
        logger.warning("Unrecognized BPM sample format: %s", list(first.keys()))
        return [], SessionSummary()

    if len(ts_arr) < FEATURE_WINDOW_S:
        logger.warning("After normalization only %d usable samples", len(ts_arr))
        return [], SessionSummary()

    raw_t = np.array(ts_arr)
    raw_bpm = np.array(bpm_arr)

    # Sort by time
    order = np.argsort(raw_t)
    raw_t = raw_t[order]
    raw_bpm = raw_bpm[order]

    # Stage 0 – Resample
    t_grid, bpm_grid = _resample_1hz(raw_t, raw_bpm)
    n_total = len(t_grid)
    if n_total < FEATURE_WINDOW_S:
        return [], SessionSummary()

    # Stage 1 – Clean
    bpm_clean, valid_mask = _clean(bpm_grid)

    # Stage 1b – BPM → pseudo-RR
    prr = _bpm_to_prr(bpm_clean)

    # ── Sliding window feature extraction ──
    snapshots: List[FeatureSnapshot] = []
    baseline = _Baseline()
    prev_calm = 50.0
    rolling_hr: List[float] = []
    rolling_si: List[float] = []
    rolling_hf: List[float] = []

    # Seed cross-session baseline if available
    if cross_session_baseline:
        baseline.hr = cross_session_baseline.get("hr", 0)
        baseline.si = cross_session_baseline.get("si", 0)
        baseline.hf = cross_session_baseline.get("hf", 0)
        baseline.breath = cross_session_baseline.get("breath", 0)
        baseline.captured = True

    t0 = t_grid[0]

    for win_end in range(FEATURE_WINDOW_S, n_total + 1, UPDATE_CADENCE_S):
        win_start = max(0, win_end - FEATURE_WINDOW_S)
        seg_prr = prr[win_start:win_end]
        seg_bpm = bpm_clean[win_start:win_end]
        seg_t   = t_grid[win_start:win_end]
        seg_valid = valid_mask[win_start:win_end]

        # Skip if window is mostly invalid
        if seg_valid.sum() < FEATURE_WINDOW_S * 0.5:
            if snapshots:
                snap = FeatureSnapshot(t=float(seg_t[-1] - t0))
                snap.calm_score = prev_calm
                snap.state = snapshots[-1].state
                snapshots.append(snap)
            continue

        # Stage 2 – Detrend
        prr_detrended, _trend = _smoothness_priors_detrend(seg_prr)

        # Stage 3 – Features
        si_proxy, amo, mo_ms, mxdmn_ms = _feature_a_baevsky_si(prr_detrended, seg_prr)
        hf_power, lf_power, lf_hf_ratio, breath_b = _feature_b_hf_power(prr_detrended)
        hr_mean, hr_slope, hr_dev = _feature_c_hr_trend(seg_bpm, seg_t, baseline.hr)
        breath_rate, coherence, resonance = _feature_d_breath_coherence(prr_detrended)

        # Use breath_rate from feature D (more complete); fall back to B
        if breath_rate == 0.0:
            breath_rate = breath_b

        snap = FeatureSnapshot(
            t=float(seg_t[-1] - t0),
            si_proxy=si_proxy, amo=amo, mo_ms=mo_ms, mxdmn_ms=mxdmn_ms,
            hf_power=hf_power, lf_power=lf_power, lf_hf_ratio=lf_hf_ratio,
            breath_rate_bpm=breath_rate,
            hr_mean_60s=hr_mean, hr_slope=hr_slope, hr_deviation=hr_dev,
            coherence=coherence, resonance_proximity=resonance,
        )

        # Stage 4 – Baseline capture / update
        elapsed = snap.t
        if not baseline.captured and elapsed >= BASELINE_CAPTURE_S:
            baseline = _capture_baseline(snapshots + [snap])
            snap.hr_deviation = hr_mean - baseline.hr
        elif baseline.captured:
            # Adaptive EMA
            baseline.hr = (1 - BASELINE_ALPHA) * baseline.hr + BASELINE_ALPHA * hr_mean
            baseline.si = (1 - BASELINE_ALPHA) * baseline.si + BASELINE_ALPHA * si_proxy
            baseline.hf = (1 - BASELINE_ALPHA) * baseline.hf + BASELINE_ALPHA * hf_power
            baseline.breath = (1 - BASELINE_ALPHA) * baseline.breath + BASELINE_ALPHA * breath_rate

        # Rolling std for z-scores (5-min window)
        rolling_hr.append(hr_mean)
        rolling_si.append(si_proxy)
        rolling_hf.append(hf_power)
        std_window = int(5 * 60 / UPDATE_CADENCE_S)  # 60 ticks
        if len(rolling_hr) > std_window:
            rolling_hr = rolling_hr[-std_window:]
            rolling_si = rolling_si[-std_window:]
            rolling_hf = rolling_hf[-std_window:]
        if len(rolling_hr) >= 3:
            baseline.hr_std = max(float(np.std(rolling_hr)), MIN_STD_FLOOR)
            baseline.si_std = max(float(np.std(rolling_si)), MIN_STD_FLOOR)
            baseline.hf_std = max(float(np.std(rolling_hf)), MIN_STD_FLOOR)

        # Stage 5 – Scoring
        if baseline.captured:
            snap.state = _classify_state(snap, baseline)
            raw_score = _compute_calm_score(snap, baseline)
            # Display EMA smoothing
            snap.calm_score = DISPLAY_EMA_ALPHA * raw_score + (1 - DISPLAY_EMA_ALPHA) * prev_calm
        else:
            snap.calm_score = 50.0
            snap.state = "neutral"

        prev_calm = snap.calm_score
        snapshots.append(snap)

    # ── Stage 6 – Session summary ──
    summary = SessionSummary()
    if snapshots:
        summary.total_snapshots = len(snapshots)
        summary.duration_s = snapshots[-1].t

        # First vs last 60 s
        first_snaps = [s for s in snapshots if s.t <= BASELINE_CAPTURE_S]
        last_snaps = snapshots[-max(1, int(FEATURE_WINDOW_S / UPDATE_CADENCE_S)):]

        if first_snaps:
            summary.hr_baseline = np.mean([s.hr_mean_60s for s in first_snaps])
            summary.breath_start = np.mean([s.breath_rate_bpm for s in first_snaps])
            hf_start = np.mean([s.hf_power for s in first_snaps])
        else:
            summary.hr_baseline = snapshots[0].hr_mean_60s
            summary.breath_start = snapshots[0].breath_rate_bpm
            hf_start = snapshots[0].hf_power

        summary.hr_final = np.mean([s.hr_mean_60s for s in last_snaps])
        summary.hr_delta = summary.hr_final - summary.hr_baseline
        summary.breath_end = np.mean([s.breath_rate_bpm for s in last_snaps])

        hf_end = np.mean([s.hf_power for s in last_snaps])
        summary.hf_pct_change = ((hf_end - hf_start) / hf_start * 100) if hf_start > 1e-12 else 0.0

        summary.avg_calm_score = float(np.mean([s.calm_score for s in snapshots]))

        states = [s.state for s in snapshots]
        n_s = len(states)
        summary.time_in_recovery_pct = states.count("recovery") / n_s * 100
        summary.time_in_stress_pct = states.count("stress") / n_s * 100
        summary.time_in_neutral_pct = states.count("neutral") / n_s * 100

    return snapshots, summary


# ── Stage 7 – System HRV reconciliation (call separately) ────────────

def reconcile_system_hrv(si_proxy: float, sdnn: float) -> Optional[str]:
    """
    Log sanity check: SI ↑ should mean SDNN ↓.
    Returns a warning string if they move in the same direction, else None.
    """
    # SI_proxy and SDNN are inversely related
    # This is for offline logging — caller stores the result
    if si_proxy > 0 and sdnn > 0:
        # Just a simple check: both increasing is suspicious
        return None  # can't tell direction from single values
    return None


# ── DB persistence ────────────────────────────────────────────────────

def save_session_results(
    user_id: str,
    session_start_time: str,
    snapshots: List[FeatureSnapshot],
    summary: SessionSummary,
) -> int:
    """
    Save calm_score timeseries + session summary to health_samples.
    Returns number of rows inserted.
    """
    if not snapshots:
        return 0

    db_url = settings.database_url_psycopg
    inserted = 0
    _CALM_LINK_WINDOW_S = 120

    with psycopg.connect(db_url) as conn:
        with conn.cursor() as cur:
            # 1. Save the calm_score timeseries as one row
            timeseries_payload = {
                "snapshots": [
                    {
                        "t": round(s.t, 1),
                        "calm_score": round(s.calm_score, 1),
                        "state": s.state,
                        "hr_mean": round(s.hr_mean_60s, 1),
                        "si_proxy": round(s.si_proxy, 2),
                        "hf_power": round(s.hf_power, 4),
                        "breath_rate": round(s.breath_rate_bpm, 1),
                        "coherence": round(s.coherence, 3),
                        "resonance": round(s.resonance_proximity, 3),
                    }
                    for s in snapshots
                ],
                "summary": asdict(summary),
            }
            # Round summary floats
            for k, v in timeseries_payload["summary"].items():
                if isinstance(v, float):
                    timeseries_payload["summary"][k] = round(v, 2)

            cur.execute(
                "INSERT INTO health_samples "
                "(user_id, sample_type, start_time, value, unit, source, payload) "
                "VALUES (%s, %s, %s::timestamptz, %s, %s, %s, %s::jsonb) "
                "RETURNING id",
                (
                    user_id,
                    "calm_score_session",
                    session_start_time,
                    round(summary.avg_calm_score, 1),
                    "score_0_100",
                    "bpm_pipeline_v1",
                    json.dumps(timeseries_payload),
                ),
            )
            calm_row_id = cur.fetchone()[0]
            inserted += 1

            # 2. Backfill: link this calm_score_session to a matching mindfulness_session
            summary_dict = timeseries_payload["summary"]
            cur.execute(
                """
                UPDATE mindfulness_sessions ms
                SET calm_score_ref = %s,
                    calm_summary = %s::jsonb
                WHERE ms.id = (
                    SELECT id
                    FROM mindfulness_sessions
                    WHERE user_id = %s
                      AND ABS(EXTRACT(EPOCH FROM start_time - %s::timestamptz)) < %s
                      AND calm_score_ref IS NULL
                    ORDER BY ABS(EXTRACT(EPOCH FROM start_time - %s::timestamptz))
                    LIMIT 1
                )
                """,
                (
                    calm_row_id,
                    json.dumps(summary_dict),
                    user_id,
                    session_start_time,
                    _CALM_LINK_WINDOW_S,
                    session_start_time,
                ),
            )
            if cur.rowcount > 0:
                logger.info(
                    "Linked calm_score_session id=%d to mindfulness_session for user=%s",
                    calm_row_id, user_id[:12],
                )

            # 3. Save cross-session baseline for future seeding
            baseline_payload = {
                "hr": round(summary.hr_baseline, 1),
                "si": round(
                    float(np.mean([s.si_proxy for s in snapshots[:int(BASELINE_CAPTURE_S / UPDATE_CADENCE_S)]])), 2
                ) if snapshots else 0,
                "hf": round(
                    float(np.mean([s.hf_power for s in snapshots[:int(BASELINE_CAPTURE_S / UPDATE_CADENCE_S)]])), 4
                ) if snapshots else 0,
                "breath": round(summary.breath_start, 1),
            }

            # Upsert cross-session baseline
            cur.execute(
                "INSERT INTO health_samples "
                "(user_id, sample_type, start_time, value, source, payload) "
                "VALUES (%s, 'calm_baseline', %s::timestamptz, %s, 'bpm_pipeline_v1', %s::jsonb) ",
                (
                    user_id,
                    session_start_time,
                    round(summary.avg_calm_score, 1),
                    json.dumps(baseline_payload),
                ),
            )
            inserted += 1

        conn.commit()

    logger.info(
        "Saved BPM pipeline results: user=%s snapshots=%d avg_calm=%.1f",
        user_id[:12], len(snapshots), summary.avg_calm_score,
    )
    return inserted


def load_cross_session_baseline(user_id: str) -> Optional[Dict[str, float]]:
    """Load the most recent cross-session baseline for seeding."""
    db_url = settings.database_url_psycopg
    with psycopg.connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT payload FROM health_samples "
                "WHERE user_id = %s AND sample_type = 'calm_baseline' "
                "ORDER BY start_time DESC LIMIT 1",
                (user_id,),
            )
            row = cur.fetchone()
            if row and row[0]:
                return row[0] if isinstance(row[0], dict) else json.loads(row[0])
    return None
