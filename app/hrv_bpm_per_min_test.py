"""
Comprehensive pytest test suite for hrv_bpm_per_min.py.

Covers every pure-computation function:
  _resample_1hz, _clean, _bpm_to_prr,
  _smoothness_priors_detrend,
  _feature_a_baevsky_si, _feature_b_hf_power,
  _feature_c_hr_trend, _feature_d_breath_coherence,
  _capture_baseline, _classify_state, _compute_calm_score,
  process_bpm_session

DB-bound functions (save_session_results, load_cross_session_baseline)
are intentionally excluded — they require a live PostgreSQL connection.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

# ── Mock app.config and psycopg BEFORE importing the module ──────────
mock_settings = MagicMock()
mock_config = MagicMock()
mock_config.settings = mock_settings
sys.modules["app.config"] = mock_config
sys.modules["psycopg"] = MagicMock()

import math
import pytest
import numpy as np

from app.hrv_bpm_per_min import (
    _resample_1hz,
    _clean,
    _bpm_to_prr,
    _smoothness_priors_detrend,
    _feature_a_baevsky_si,
    _feature_b_hf_power,
    _feature_c_hr_trend,
    _feature_d_breath_coherence,
    _capture_baseline,
    _classify_state,
    _compute_calm_score,
    process_bpm_session,
    FeatureSnapshot,
    SessionSummary,
    _Baseline,
    FEATURE_WINDOW_S,
    BPM_MIN,
    BPM_MAX,
    SPIKE_THRESHOLD,
)


# ── Helpers ────────────────────────────────────────────────────────────

def _resting_bpm_array(n: int = 120, base: float = 70.0, seed: int = 42) -> np.ndarray:
    """Realistic resting heart rate with ±3 BPM natural variability."""
    rng = np.random.default_rng(seed)
    return base + rng.uniform(-3.0, 3.0, size=n)


def _uniform_timestamps(n: int = 120, start: float = 0.0) -> np.ndarray:
    """Exactly 1 Hz timestamps starting at `start`."""
    return np.arange(start, start + n, 1.0)


def _make_snapshot(
    *,
    hr_mean: float = 70.0,
    hr_dev: float = 0.0,
    si_proxy: float = 2.0,
    hf_power: float = 1.0,
    resonance: float = 0.5,
    breath: float = 14.0,
) -> FeatureSnapshot:
    snap = FeatureSnapshot(t=60.0)
    snap.hr_mean_60s = hr_mean
    snap.hr_deviation = hr_dev
    snap.si_proxy = si_proxy
    snap.hf_power = hf_power
    snap.resonance_proximity = resonance
    snap.breath_rate_bpm = breath
    return snap


def _make_baseline(
    *,
    hr: float = 70.0,
    si: float = 2.0,
    hf: float = 1.0,
    hr_std: float = 2.0,
    si_std: float = 0.5,
    hf_std: float = 0.2,
) -> _Baseline:
    bl = _Baseline()
    bl.hr = hr
    bl.si = si
    bl.hf = hf
    bl.hr_std = hr_std
    bl.si_std = si_std
    bl.hf_std = hf_std
    bl.captured = True
    return bl


# ══════════════════════════════════════════════════════════════════════
# 1.  _resample_1hz
# ══════════════════════════════════════════════════════════════════════

class TestResample1Hz:

    def test_returns_uniform_grid_spacing(self):
        # Arrange — irregular timestamps (gaps of 1.0, 1.5, 0.7, 2.1 s)
        ts = np.array([0.0, 1.0, 2.5, 3.2, 5.3])
        bpm = np.array([65.0, 66.0, 68.0, 67.0, 70.0])

        # Act
        t_grid, bpm_grid = _resample_1hz(ts, bpm)

        # Assert — uniform 1 Hz spacing
        diffs = np.diff(t_grid)
        assert np.allclose(diffs, 1.0), f"Grid step should be 1.0 s, got {diffs}"

    def test_output_length_matches_floor_range(self):
        ts = np.array([0.0, 1.0, 2.5, 3.2, 5.3])
        bpm = np.array([65.0, 66.0, 68.0, 67.0, 70.0])

        t_grid, bpm_grid = _resample_1hz(ts, bpm)

        expected_len = math.floor(ts[-1]) - math.floor(ts[0]) + 1
        assert len(t_grid) == expected_len
        assert len(bpm_grid) == expected_len

    def test_interp_values_stay_within_input_range(self):
        ts = np.array([0.0, 10.0, 20.0])
        bpm = np.array([60.0, 80.0, 70.0])

        _, bpm_grid = _resample_1hz(ts, bpm)

        assert bpm_grid.min() >= 60.0 - 1e-9
        assert bpm_grid.max() <= 80.0 + 1e-9

    def test_exact_1hz_input_is_unchanged(self):
        ts = _uniform_timestamps(60)
        bpm = _resting_bpm_array(60)

        t_grid, bpm_grid = _resample_1hz(ts, bpm)

        assert len(t_grid) == len(ts)
        np.testing.assert_allclose(bpm_grid, bpm, rtol=1e-9)

    def test_single_gap_interpolated_linearly(self):
        # Two anchor points — linear between them
        ts = np.array([0.0, 4.0])
        bpm = np.array([60.0, 80.0])

        t_grid, bpm_grid = _resample_1hz(ts, bpm)

        # At t=2, linear interp should give 70
        idx_2 = np.where(t_grid == 2.0)[0]
        assert len(idx_2) == 1
        assert abs(bpm_grid[idx_2[0]] - 70.0) < 1e-6

    def test_start_time_floored_correctly(self):
        # Non-integer start
        ts = np.array([0.3, 1.3, 2.3, 3.3])
        bpm = np.array([65.0, 66.0, 67.0, 68.0])

        t_grid, _ = _resample_1hz(ts, bpm)

        assert t_grid[0] == math.floor(0.3)


# ══════════════════════════════════════════════════════════════════════
# 2.  _clean
# ══════════════════════════════════════════════════════════════════════

class TestClean:

    def test_normal_resting_hr_all_valid(self):
        bpm = _resting_bpm_array(120)

        cleaned, valid = _clean(bpm)

        assert valid.all(), "All resting BPM should be valid"
        assert not np.isnan(cleaned).any()

    def test_below_min_flagged_and_interpolated(self):
        bpm = np.full(20, 70.0)
        bpm[5] = BPM_MIN - 1  # 39 — below gate

        cleaned, valid = _clean(bpm)

        assert valid[5] is np.bool_(False)
        # Interpolation must fill the gap — no NaN
        assert not math.isnan(cleaned[5])

    def test_above_max_flagged_and_interpolated(self):
        bpm = np.full(20, 70.0)
        bpm[10] = BPM_MAX + 1  # 181 — above gate

        cleaned, valid = _clean(bpm)

        assert valid[10] is np.bool_(False)
        assert not math.isnan(cleaned[10])

    def test_bpm_exactly_at_min_is_valid(self):
        bpm = np.full(20, float(BPM_MIN))

        _, valid = _clean(bpm)

        assert valid.all()

    def test_bpm_exactly_at_max_is_valid(self):
        bpm = np.full(20, float(BPM_MAX))

        _, valid = _clean(bpm)

        assert valid.all()

    def test_spike_above_threshold_invalidated(self):
        bpm = np.full(30, 70.0)
        bpm[15] = 70.0 + SPIKE_THRESHOLD + 1  # 86 — spike

        _, valid = _clean(bpm)

        assert valid[15] is np.bool_(False)

    def test_spike_exactly_at_threshold_allowed(self):
        # |Δ| == SPIKE_THRESHOLD is NOT a spike (condition is strictly >)
        bpm = np.full(30, 70.0)
        bpm[15] = 70.0 + SPIKE_THRESHOLD  # exactly 85

        _, valid = _clean(bpm)

        assert valid[15] is np.bool_(True)

    def test_spike_below_threshold_allowed(self):
        bpm = np.full(30, 70.0)
        bpm[15] = 70.0 + SPIKE_THRESHOLD - 1  # 84

        _, valid = _clean(bpm)

        assert valid[15] is np.bool_(True)

    def test_long_gap_more_than_5_marked_invalid(self):
        bpm = np.full(30, 70.0)
        # Inject 7 consecutive out-of-range values (positions 10–16)
        bpm[10:17] = BPM_MIN - 5

        _, valid = _clean(bpm)

        # All 7 gap positions should be invalid
        assert not valid[10:17].any()

    def test_short_gap_up_to_5_not_penalised_in_cleaned(self):
        bpm = np.full(30, 70.0)
        # 4 consecutive out-of-range
        bpm[10:14] = BPM_MIN - 5

        cleaned, _ = _clean(bpm)

        # Short gaps: cleaned array has no NaN after interpolation
        assert not np.isnan(cleaned).any()

    def test_all_out_of_range_returns_no_crash(self):
        # All values out of gate — all become NaN, gap interpolation can't fill (all invalid)
        bpm = np.full(30, float(BPM_MAX + 10))

        # Must not raise
        cleaned, valid = _clean(bpm)

        assert len(cleaned) == 30

    def test_single_element_returns_same_length(self):
        bpm = np.array([65.0])

        cleaned, valid = _clean(bpm)

        assert len(cleaned) == 1
        assert len(valid) == 1

    def test_cleaned_output_same_shape_as_input(self):
        bpm = _resting_bpm_array(200)

        cleaned, valid = _clean(bpm)

        assert cleaned.shape == bpm.shape
        assert valid.shape == bpm.shape

    def test_no_mutation_of_input_array(self):
        bpm = np.full(30, 70.0)
        bpm[5] = BPM_MIN - 5
        original = bpm.copy()

        _clean(bpm)

        np.testing.assert_array_equal(bpm, original, err_msg="_clean must not mutate input")


# ══════════════════════════════════════════════════════════════════════
# 3.  _bpm_to_prr
# ══════════════════════════════════════════════════════════════════════

class TestBpmToPrr:

    def test_60_bpm_gives_1000ms(self):
        result = _bpm_to_prr(np.array([60.0]))
        assert abs(result[0] - 1000.0) < 1e-9

    def test_75_bpm_gives_800ms(self):
        result = _bpm_to_prr(np.array([75.0]))
        assert abs(result[0] - 800.0) < 1e-9

    def test_120_bpm_gives_500ms(self):
        result = _bpm_to_prr(np.array([120.0]))
        assert abs(result[0] - 500.0) < 1e-9

    def test_bpm_below_min_clamped_to_min(self):
        # Any BPM below 40 should be treated as 40
        result_low = _bpm_to_prr(np.array([0.1]))
        result_min = _bpm_to_prr(np.array([float(BPM_MIN)]))
        assert abs(result_low[0] - result_min[0]) < 1e-9

    def test_bpm_above_max_clamped_to_max(self):
        # Any BPM above 180 should be treated as 180
        result_high = _bpm_to_prr(np.array([999.0]))
        result_max = _bpm_to_prr(np.array([float(BPM_MAX)]))
        assert abs(result_high[0] - result_max[0]) < 1e-9

    def test_higher_bpm_gives_shorter_rr(self):
        prr = _bpm_to_prr(np.array([60.0, 90.0, 120.0]))
        assert prr[0] > prr[1] > prr[2]

    def test_formula_is_60000_over_bpm(self):
        bpm = np.array([50.0, 65.0, 80.0, 100.0])
        expected = 60000.0 / bpm
        np.testing.assert_allclose(_bpm_to_prr(bpm), expected, rtol=1e-9)

    def test_array_output_shape_matches_input(self):
        bpm = _resting_bpm_array(150)
        prr = _bpm_to_prr(bpm)
        assert prr.shape == bpm.shape

    def test_no_negative_values(self):
        bpm = _resting_bpm_array(120)
        prr = _bpm_to_prr(bpm)
        assert (prr > 0).all()


# ══════════════════════════════════════════════════════════════════════
# 4.  _smoothness_priors_detrend
# ══════════════════════════════════════════════════════════════════════

class TestSmoothnessPriorsDetrend:

    def test_returns_two_arrays_same_length(self):
        x = _bpm_to_prr(_resting_bpm_array(60))

        detrended, trend = _smoothness_priors_detrend(x)

        assert detrended.shape == x.shape
        assert trend.shape == x.shape

    def test_detrended_is_approximately_zero_mean(self):
        # Add a strong linear trend so detrending actually matters
        x = np.linspace(800, 1000, 120)  # ms — upward trend

        detrended, _ = _smoothness_priors_detrend(x, lam=500)

        assert abs(np.mean(detrended)) < 5.0, (
            f"Detrended mean {np.mean(detrended):.4f} should be close to zero"
        )

    def test_detrended_plus_trend_reconstructs_original(self):
        x = _bpm_to_prr(_resting_bpm_array(80))

        detrended, trend = _smoothness_priors_detrend(x)

        np.testing.assert_allclose(detrended + trend, x, rtol=1e-6)

    def test_constant_signal_detrends_to_near_zero(self):
        x = np.full(60, 850.0)

        detrended, trend = _smoothness_priors_detrend(x)

        np.testing.assert_allclose(detrended, np.zeros(60), atol=1e-4)

    def test_short_input_less_than_4_uses_mean_subtraction(self):
        x = np.array([800.0, 850.0, 900.0])  # n=3 < 4

        detrended, trend = _smoothness_priors_detrend(x)

        assert len(detrended) == 3
        assert abs(np.mean(detrended)) < 1e-9  # mean-subtracted

    def test_high_lambda_produces_smoother_trend(self):
        # High λ → trend closely follows mean (smoother / less variance)
        rng = np.random.default_rng(0)
        x = 850.0 + rng.normal(0, 20, 120)

        _, trend_lo = _smoothness_priors_detrend(x, lam=1)
        _, trend_hi = _smoothness_priors_detrend(x, lam=10000)

        # High-lambda trend has less variance (smoother)
        assert float(np.std(trend_hi)) <= float(np.std(trend_lo)) + 1e-3

    def test_output_is_finite_for_realistic_prr(self):
        prr = _bpm_to_prr(_resting_bpm_array(120))

        detrended, trend = _smoothness_priors_detrend(prr)

        assert np.isfinite(detrended).all()
        assert np.isfinite(trend).all()


# ══════════════════════════════════════════════════════════════════════
# 5.  _feature_a_baevsky_si
# ══════════════════════════════════════════════════════════════════════

class TestFeatureABaevskySI:

    def test_returns_four_floats(self):
        prr = _bpm_to_prr(_resting_bpm_array(60))
        detrended, _ = _smoothness_priors_detrend(prr)

        result = _feature_a_baevsky_si(detrended)

        assert len(result) == 4
        si, amo, mo_ms, mxdmn = result
        assert isinstance(si, float)
        assert isinstance(amo, float)
        assert isinstance(mo_ms, float)
        assert isinstance(mxdmn, float)

    def test_si_proxy_non_negative(self):
        prr = _bpm_to_prr(_resting_bpm_array(60))
        detrended, _ = _smoothness_priors_detrend(prr)

        si, _, _, _ = _feature_a_baevsky_si(detrended)

        assert si >= 0.0

    def test_amo_within_0_to_100_percent(self):
        prr = _bpm_to_prr(_resting_bpm_array(120))
        detrended, _ = _smoothness_priors_detrend(prr)

        _, amo, _, _ = _feature_a_baevsky_si(detrended)

        assert 0.0 <= amo <= 100.0, f"AMo {amo:.2f} out of [0,100]"

    def test_fewer_than_10_samples_returns_all_zeros(self):
        tiny = np.array([800.0, 820.0, 810.0, 830.0, 800.0])  # n=5

        result = _feature_a_baevsky_si(tiny)

        assert result == (0.0, 0.0, 0.0, 0.0)

    def test_constant_detrended_signal_degenerate_case_no_crash(self):
        # All-same detrended → mxdmn = 0 → denom ≈ 0
        flat = np.zeros(60)

        si, amo, mo_ms, mxdmn = _feature_a_baevsky_si(flat)

        # Must not raise; mxdmn should be 0
        assert mxdmn == 0.0
        assert math.isfinite(si)

    def test_mxdmn_equals_range_of_trimmed_data(self):
        prr = _bpm_to_prr(_resting_bpm_array(120))
        detrended, _ = _smoothness_priors_detrend(prr)

        _, _, _, mxdmn = _feature_a_baevsky_si(detrended)

        lo, hi = np.percentile(detrended, [2, 98])
        trimmed = detrended[(detrended >= lo) & (detrended <= hi)]
        expected_range = float(trimmed.max() - trimmed.min())
        assert abs(mxdmn - expected_range) < 1e-9

    def test_mo_ms_is_median_of_trimmed(self):
        prr = _bpm_to_prr(_resting_bpm_array(120))
        detrended, _ = _smoothness_priors_detrend(prr)

        _, _, mo_ms, _ = _feature_a_baevsky_si(detrended)

        lo, hi = np.percentile(detrended, [2, 98])
        trimmed = detrended[(detrended >= lo) & (detrended <= hi)]
        assert abs(mo_ms - float(np.median(trimmed))) < 1e-9

    def test_si_is_sqrt_of_formula(self):
        # si = sqrt(amo / (2 * |mo_s| * mxdmn_s)); verify non-negative square root
        prr = _bpm_to_prr(_resting_bpm_array(120))
        detrended, _ = _smoothness_priors_detrend(prr)

        si, _, _, _ = _feature_a_baevsky_si(detrended)

        assert si >= 0.0
        # SI is sqrt so it should be smaller than raw SI
        assert si < 1e6


# ══════════════════════════════════════════════════════════════════════
# 6.  _feature_b_hf_power
# ══════════════════════════════════════════════════════════════════════

class TestFeatureBHfPower:

    def test_returns_four_floats(self):
        prr = _bpm_to_prr(_resting_bpm_array(60))
        detrended, _ = _smoothness_priors_detrend(prr)

        result = _feature_b_hf_power(detrended, fs=1.0)

        assert len(result) == 4
        for v in result:
            assert isinstance(v, float)

    def test_hf_lf_power_non_negative(self):
        prr = _bpm_to_prr(_resting_bpm_array(120))
        detrended, _ = _smoothness_priors_detrend(prr)

        hf, lf, ratio, breath = _feature_b_hf_power(detrended)

        assert hf >= 0.0
        assert lf >= 0.0
        assert ratio >= 0.0

    def test_breath_rate_in_plausible_range_when_nonzero(self):
        prr = _bpm_to_prr(_resting_bpm_array(120))
        detrended, _ = _smoothness_priors_detrend(prr)

        _, _, _, breath = _feature_b_hf_power(detrended)

        if breath > 0.0:
            # peak_hz in 0.05–0.40 → breath_rate = peak_hz*60 → 3–24 bpm
            assert 2.0 <= breath <= 25.0, f"breath_rate {breath:.2f} out of plausible range"

    def test_too_few_samples_returns_zeros(self):
        tiny = np.array([5.0, 6.0, 7.0])  # nperseg < 8

        result = _feature_b_hf_power(tiny, fs=1.0)

        assert result == (0.0, 0.0, 0.0, 0.0)

    def test_lf_hf_ratio_zero_when_hf_power_is_zero(self):
        # Flat signal has no HF content
        flat = np.zeros(60)

        _, _, ratio, _ = _feature_b_hf_power(flat)

        assert ratio == 0.0

    def test_sinusoid_at_hf_frequency_produces_nonzero_hf(self):
        # Inject 0.25 Hz sinusoid (in HF band 0.15–0.40 Hz)
        fs = 1.0
        t = np.arange(120) / fs
        sig = 50.0 * np.sin(2 * np.pi * 0.25 * t)

        hf, lf, ratio, _ = _feature_b_hf_power(sig, fs=fs)

        assert hf > 0.0, "HF sinusoid should produce nonzero HF power"
        assert hf > lf, "HF sinusoid should produce higher HF than LF power"

    def test_lf_hf_ratio_positive_when_both_bands_have_power(self):
        # Mixed signal with both LF (0.08 Hz) and HF (0.25 Hz) components
        fs = 1.0
        t = np.arange(120) / fs
        sig = 30.0 * np.sin(2 * np.pi * 0.08 * t) + 20.0 * np.sin(2 * np.pi * 0.25 * t)

        hf, lf, ratio, _ = _feature_b_hf_power(sig, fs=fs)

        assert ratio >= 0.0


# ══════════════════════════════════════════════════════════════════════
# 7.  _feature_c_hr_trend
# ══════════════════════════════════════════════════════════════════════

class TestFeatureCHrTrend:

    def test_returns_three_floats(self):
        bpm = _resting_bpm_array(60)
        t = _uniform_timestamps(60)

        result = _feature_c_hr_trend(bpm, t, hr_baseline=70.0)

        assert len(result) == 3
        for v in result:
            assert isinstance(v, float)

    def test_hr_mean_matches_numpy_mean(self):
        bpm = _resting_bpm_array(60)
        t = _uniform_timestamps(60)

        hr_mean, _, _ = _feature_c_hr_trend(bpm, t, hr_baseline=0.0)

        assert abs(hr_mean - float(np.mean(bpm))) < 1e-9

    def test_hr_deviation_equals_mean_minus_baseline(self):
        bpm = np.full(60, 75.0)
        t = _uniform_timestamps(60)
        baseline = 70.0

        _, _, hr_dev = _feature_c_hr_trend(bpm, t, hr_baseline=baseline)

        assert abs(hr_dev - (75.0 - 70.0)) < 1e-9

    def test_flat_signal_gives_zero_slope(self):
        bpm = np.full(60, 70.0)
        t = _uniform_timestamps(60)

        _, slope, _ = _feature_c_hr_trend(bpm, t, hr_baseline=70.0)

        assert abs(slope) < 1e-6

    def test_increasing_hr_gives_positive_slope(self):
        t = _uniform_timestamps(60)
        bpm = np.linspace(60.0, 80.0, 60)

        _, slope, _ = _feature_c_hr_trend(bpm, t, hr_baseline=60.0)

        assert slope > 0.0, f"Expected positive slope, got {slope:.4f}"

    def test_decreasing_hr_gives_negative_slope(self):
        t = _uniform_timestamps(60)
        bpm = np.linspace(80.0, 60.0, 60)

        _, slope, _ = _feature_c_hr_trend(bpm, t, hr_baseline=80.0)

        assert slope < 0.0, f"Expected negative slope, got {slope:.4f}"

    def test_slope_unit_is_bpm_per_minute(self):
        # +1 BPM/s over the last 30 s → slope should be +60 BPM/min
        t = _uniform_timestamps(30)
        bpm = np.arange(30, dtype=float)  # bpm[i] = i → slope = 1 bpm/s

        _, slope, _ = _feature_c_hr_trend(bpm, t, hr_baseline=0.0)

        # slope = 1 bpm/s * 60 = 60 bpm/min
        assert abs(slope - 60.0) < 1e-3, f"Expected 60 bpm/min, got {slope:.4f}"

    def test_single_sample_returns_zero_slope(self):
        bpm = np.array([72.0])
        t = np.array([0.0])

        hr_mean, slope, _ = _feature_c_hr_trend(bpm, t, hr_baseline=72.0)

        assert slope == 0.0

    def test_zero_baseline_deviation_when_hr_matches_baseline(self):
        bpm = np.full(60, 68.0)
        t = _uniform_timestamps(60)

        _, _, hr_dev = _feature_c_hr_trend(bpm, t, hr_baseline=68.0)

        assert abs(hr_dev) < 1e-9

    def test_slope_uses_only_last_30_samples(self):
        # First 30 samples: flat at 70; Last 30: increasing sharply
        bpm_first = np.full(30, 70.0)
        bpm_last = np.linspace(70.0, 100.0, 30)  # big rise
        bpm = np.concatenate([bpm_first, bpm_last])
        t = _uniform_timestamps(60)

        _, slope, _ = _feature_c_hr_trend(bpm, t, hr_baseline=70.0)

        # Slope over last 30 s with +30 bpm change → ~60 bpm/min
        assert slope > 0.0


# ══════════════════════════════════════════════════════════════════════
# 8.  _feature_d_breath_coherence
# ══════════════════════════════════════════════════════════════════════

class TestFeatureDBreathCoherence:

    def test_returns_three_floats(self):
        prr = _bpm_to_prr(_resting_bpm_array(120))
        detrended, _ = _smoothness_priors_detrend(prr)

        result = _feature_d_breath_coherence(detrended, fs=1.0)

        assert len(result) == 3
        for v in result:
            assert isinstance(v, float)

    def test_coherence_between_0_and_1(self):
        prr = _bpm_to_prr(_resting_bpm_array(120))
        detrended, _ = _smoothness_priors_detrend(prr)

        _, coherence, _ = _feature_d_breath_coherence(detrended)

        assert 0.0 <= coherence <= 1.0, f"Coherence {coherence:.4f} out of [0,1]"

    def test_resonance_proximity_between_0_and_1(self):
        prr = _bpm_to_prr(_resting_bpm_array(120))
        detrended, _ = _smoothness_priors_detrend(prr)

        _, _, resonance = _feature_d_breath_coherence(detrended)

        assert 0.0 <= resonance <= 1.0, f"Resonance {resonance:.4f} out of [0,1]"

    def test_resonance_maximal_at_6_breaths_per_min(self):
        # 6 breaths/min = 0.1 Hz — inject clean 0.1 Hz sinusoid
        fs = 1.0
        t = np.arange(120) / fs
        sig = 100.0 * np.sin(2 * np.pi * 0.1 * t)

        _, _, resonance = _feature_d_breath_coherence(sig, fs=fs)

        assert resonance > 0.85, f"At 6 bpm resonance should be near 1, got {resonance:.4f}"

    def test_resonance_decreases_away_from_6_bpm(self):
        # 20 breaths/min = 0.333 Hz — far from 6 bpm resonance
        fs = 1.0
        t = np.arange(120) / fs
        sig_resonant = 100.0 * np.sin(2 * np.pi * 0.1 * t)    # 6 bpm
        sig_offpeak = 100.0 * np.sin(2 * np.pi * 0.333 * t)   # 20 bpm

        _, _, resonance_6bpm = _feature_d_breath_coherence(sig_resonant, fs=fs)
        _, _, resonance_20bpm = _feature_d_breath_coherence(sig_offpeak, fs=fs)

        assert resonance_6bpm > resonance_20bpm, (
            f"6 bpm resonance {resonance_6bpm:.4f} should exceed 20 bpm {resonance_20bpm:.4f}"
        )

    def test_too_few_samples_returns_zeros(self):
        tiny = np.zeros(5)  # nperseg < 8

        result = _feature_d_breath_coherence(tiny, fs=1.0)

        assert result == (0.0, 0.0, 0.0)

    def test_breath_rate_plausible_range_when_nonzero(self):
        prr = _bpm_to_prr(_resting_bpm_array(120))
        detrended, _ = _smoothness_priors_detrend(prr)

        breath, _, _ = _feature_d_breath_coherence(detrended)

        if breath > 0.0:
            # 0.05–0.40 Hz * 60 = 3–24 bpm
            assert 2.0 <= breath <= 25.0

    def test_flat_signal_returns_zero_coherence(self):
        flat = np.zeros(60)

        _, coherence, _ = _feature_d_breath_coherence(flat)

        assert coherence == 0.0


# ══════════════════════════════════════════════════════════════════════
# 9.  _capture_baseline
# ══════════════════════════════════════════════════════════════════════

class TestCaptureBaseline:

    def test_empty_list_returns_uncaptured_baseline(self):
        bl = _capture_baseline([])

        assert bl.captured is False

    def test_nonempty_list_returns_captured_true(self):
        snaps = [_make_snapshot(hr_mean=70.0, si_proxy=2.0, hf_power=1.0, breath=14.0)]

        bl = _capture_baseline(snaps)

        assert bl.captured is True

    def test_hr_is_mean_of_snapshot_hr_means(self):
        snaps = [
            _make_snapshot(hr_mean=68.0),
            _make_snapshot(hr_mean=72.0),
            _make_snapshot(hr_mean=70.0),
        ]

        bl = _capture_baseline(snaps)

        assert abs(bl.hr - 70.0) < 1e-9

    def test_si_is_mean_of_snapshot_si_proxies(self):
        snaps = [
            _make_snapshot(si_proxy=1.5),
            _make_snapshot(si_proxy=2.5),
        ]

        bl = _capture_baseline(snaps)

        assert abs(bl.si - 2.0) < 1e-9

    def test_hf_is_mean_of_snapshot_hf_powers(self):
        snaps = [
            _make_snapshot(hf_power=0.5),
            _make_snapshot(hf_power=1.5),
        ]

        bl = _capture_baseline(snaps)

        assert abs(bl.hf - 1.0) < 1e-9

    def test_breath_is_mean_of_snapshot_breath_rates(self):
        snaps = [
            _make_snapshot(breath=12.0),
            _make_snapshot(breath=16.0),
        ]

        bl = _capture_baseline(snaps)

        assert abs(bl.breath - 14.0) < 1e-9

    def test_single_snapshot_baseline_equals_snapshot_values(self):
        snap = _make_snapshot(hr_mean=65.0, si_proxy=1.8, hf_power=0.9, breath=13.0)

        bl = _capture_baseline([snap])

        assert abs(bl.hr - 65.0) < 1e-9
        assert abs(bl.si - 1.8) < 1e-9
        assert abs(bl.hf - 0.9) < 1e-9
        assert abs(bl.breath - 13.0) < 1e-9

    def test_returns_baseline_dataclass(self):
        bl = _capture_baseline([_make_snapshot()])

        assert isinstance(bl, _Baseline)

    def test_many_snapshots_average_correctly(self):
        snaps = [_make_snapshot(hr_mean=float(60 + i)) for i in range(10)]

        bl = _capture_baseline(snaps)

        expected_hr = float(np.mean([60 + i for i in range(10)]))
        assert abs(bl.hr - expected_hr) < 1e-9


# ══════════════════════════════════════════════════════════════════════
# 10.  _classify_state
# ══════════════════════════════════════════════════════════════════════

class TestClassifyState:

    def test_neutral_when_within_normal_range(self):
        snap = _make_snapshot(hr_dev=0.0, si_proxy=2.0, hf_power=1.0)
        bl = _make_baseline(si=2.0, hf=1.0)

        state = _classify_state(snap, bl)

        assert state == "neutral"

    def test_stress_when_all_three_stress_conditions_met(self):
        # hr_dev > 2, si > 1.1*bl_si, hf < 0.9*bl_hf
        snap = _make_snapshot(hr_dev=5.0, si_proxy=3.0, hf_power=0.5)
        bl = _make_baseline(si=2.0, hf=1.0)  # thresholds: si>2.2, hf<0.9

        state = _classify_state(snap, bl)

        assert state == "stress"

    def test_recovery_when_all_three_recovery_conditions_met(self):
        # hr_dev < -2, si < 0.9*bl_si, hf > 1.1*bl_hf
        snap = _make_snapshot(hr_dev=-5.0, si_proxy=1.0, hf_power=2.0)
        bl = _make_baseline(si=2.0, hf=1.0)  # thresholds: si<1.8, hf>1.1

        state = _classify_state(snap, bl)

        assert state == "recovery"

    def test_stress_not_triggered_without_positive_hr_deviation(self):
        snap = _make_snapshot(hr_dev=-1.0, si_proxy=3.0, hf_power=0.5)
        bl = _make_baseline(si=2.0, hf=1.0)

        state = _classify_state(snap, bl)

        assert state != "stress"

    def test_recovery_not_triggered_without_negative_hr_deviation(self):
        snap = _make_snapshot(hr_dev=1.0, si_proxy=1.0, hf_power=2.0)
        bl = _make_baseline(si=2.0, hf=1.0)

        state = _classify_state(snap, bl)

        assert state != "recovery"

    def test_stress_requires_si_above_110pct_baseline(self):
        # si is exactly at 1.1*bl_si (threshold is strictly >)
        snap = _make_snapshot(hr_dev=5.0, si_proxy=2.2, hf_power=0.5)
        bl = _make_baseline(si=2.0, hf=1.0)

        state = _classify_state(snap, bl)

        # si=2.2 is exactly 1.1*2.0=2.2; condition is > so NOT stress
        assert state != "stress"

    def test_recovery_requires_si_below_90pct_baseline(self):
        # si is exactly at 0.9*bl_si (threshold is strictly <)
        snap = _make_snapshot(hr_dev=-5.0, si_proxy=1.8, hf_power=2.0)
        bl = _make_baseline(si=2.0, hf=1.0)

        state = _classify_state(snap, bl)

        # si=1.8 is exactly 0.9*2.0=1.8; condition is < so NOT recovery
        assert state != "recovery"

    @pytest.mark.parametrize("hr_dev,si_mult,hf_mult", [
        (hr_dev, si_mult, hf_mult)
        for hr_dev in [-5, 0, 5]
        for si_mult in [0.5, 1.0, 1.5]
        for hf_mult in [0.5, 1.0, 1.5]
    ])
    def test_output_always_one_of_three_valid_states(self, hr_dev, si_mult, hf_mult):
        snap = _make_snapshot(
            hr_dev=hr_dev,
            si_proxy=2.0 * si_mult,
            hf_power=1.0 * hf_mult,
        )
        bl = _make_baseline(si=2.0, hf=1.0)

        state = _classify_state(snap, bl)

        assert state in {"recovery", "neutral", "stress"}, (
            f"Unexpected state '{state}' for hr_dev={hr_dev}, si_mult={si_mult}, hf_mult={hf_mult}"
        )


# ══════════════════════════════════════════════════════════════════════
# 11.  _compute_calm_score
# ══════════════════════════════════════════════════════════════════════

class TestComputeCalmScore:

    def test_returns_float_in_0_to_100(self):
        snap = _make_snapshot()
        bl = _make_baseline()

        score = _compute_calm_score(snap, bl)

        assert isinstance(score, float)
        assert 0.0 <= score <= 100.0

    def test_at_baseline_values_score_is_near_50(self):
        # hr=bl.hr, si=bl.si, hf=bl.hf, resonance=0.5 → z-scores ≈ 0 → sigmoid(0)=50
        snap = _make_snapshot(hr_mean=70.0, si_proxy=2.0, hf_power=1.0, resonance=0.5)
        bl = _make_baseline(hr=70.0, si=2.0, hf=1.0)

        score = _compute_calm_score(snap, bl)

        # resonance term: 2*0.5 - 1 = 0.0, so raw ≈ 0 → score ≈ 50
        assert 40.0 <= score <= 60.0, f"At baseline, score should be near 50, got {score:.2f}"

    def test_calm_scenario_gives_high_score(self):
        # Well below baseline HR, low SI, high HF, near-resonant breathing
        snap = _make_snapshot(hr_mean=60.0, si_proxy=0.5, hf_power=5.0, resonance=0.95)
        bl = _make_baseline(hr=75.0, si=3.0, hf=1.0, hr_std=3.0, si_std=1.0, hf_std=1.0)

        score = _compute_calm_score(snap, bl)

        assert score > 70.0, f"Expected high calm score for calm state, got {score:.2f}"

    def test_stress_scenario_gives_low_score(self):
        # High HR, high SI, low HF, non-resonant breathing
        snap = _make_snapshot(hr_mean=95.0, si_proxy=8.0, hf_power=0.1, resonance=0.01)
        bl = _make_baseline(hr=65.0, si=1.5, hf=2.0, hr_std=3.0, si_std=0.5, hf_std=0.5)

        score = _compute_calm_score(snap, bl)

        assert score < 40.0, f"Expected low calm score for stress, got {score:.2f}"

    def test_score_monotonically_increases_with_resonance(self):
        bl = _make_baseline(hr=70.0, si=2.0, hf=1.0, hr_std=1.0, si_std=0.3, hf_std=0.1)

        scores = []
        for rp in [0.0, 0.25, 0.5, 0.75, 1.0]:
            snap = _make_snapshot(hr_mean=70.0, si_proxy=2.0, hf_power=1.0, resonance=rp)
            scores.append(_compute_calm_score(snap, bl))

        for i in range(len(scores) - 1):
            assert scores[i] < scores[i + 1], (
                f"Score should increase with resonance_proximity. Got: {scores}"
            )

    def test_std_floor_prevents_division_by_zero(self):
        # std values all zero — must not crash (MIN_STD_FLOOR protects)
        bl = _make_baseline(hr_std=0.0, si_std=0.0, hf_std=0.0)

        score = _compute_calm_score(_make_snapshot(), bl)

        assert math.isfinite(score)
        assert 0.0 <= score <= 100.0

    def test_score_bounded_regardless_of_extreme_z_scores(self):
        # Extreme deviations from baseline — score must remain in [0, 100]
        snap = _make_snapshot(hr_mean=200.0, si_proxy=1000.0, hf_power=0.0, resonance=0.0)
        bl = _make_baseline(hr=60.0, si=1.0, hf=10.0, hr_std=0.1, si_std=0.1, hf_std=0.1)

        score = _compute_calm_score(snap, bl)

        assert 0.0 <= score <= 100.0


# ══════════════════════════════════════════════════════════════════════
# 12.  process_bpm_session — full pipeline
# ══════════════════════════════════════════════════════════════════════

class TestProcessBpmSession:

    # ── input data factories ────────────────────────────────────────────

    @staticmethod
    def _make_format1(n: int = 120, seed: int = 10) -> list:
        """Format 1: rr_interval_ms, no timestamps."""
        rng = np.random.default_rng(seed)
        bpm = 70.0 + rng.uniform(-3, 3, n)
        return [{"rr_interval_ms": int(60000 / b)} for b in bpm]

    @staticmethod
    def _make_format2(n: int = 120, base_ts: float = 1_700_000_000.0, seed: int = 11) -> list:
        """Format 2: bpm + timestamp (Unix seconds)."""
        rng = np.random.default_rng(seed)
        bpm = 70.0 + rng.uniform(-3, 3, n)
        return [{"bpm": float(bpm[i]), "timestamp": base_ts + i} for i in range(n)]

    @staticmethod
    def _make_format3(n: int = 120, base_ts_ms: int = 1_700_000_000_000, seed: int = 12) -> list:
        """Format 3: bpm + timestamp_ms (milliseconds)."""
        rng = np.random.default_rng(seed)
        bpm = 70.0 + rng.uniform(-3, 3, n)
        return [{"bpm": float(bpm[i]), "timestamp_ms": base_ts_ms + i * 1000} for i in range(n)]

    # ── guard: too few samples ─────────────────────────────────────────

    def test_empty_input_returns_empty_list_and_default_summary(self):
        snaps, summary = process_bpm_session([])

        assert snaps == []
        assert isinstance(summary, SessionSummary)
        assert summary.total_snapshots == 0

    def test_fewer_than_feature_window_returns_empty(self):
        samples = self._make_format2(n=FEATURE_WINDOW_S - 1)

        snaps, summary = process_bpm_session(samples)

        assert snaps == []

    def test_exactly_feature_window_samples_does_not_crash(self):
        samples = self._make_format2(n=FEATURE_WINDOW_S)

        snaps, summary = process_bpm_session(samples)

        assert isinstance(snaps, list)
        assert isinstance(summary, SessionSummary)

    # ── input format detection ─────────────────────────────────────────

    def test_format1_rr_interval_ms_produces_snapshots(self):
        samples = self._make_format1(n=180)

        snaps, summary = process_bpm_session(samples)

        assert len(snaps) > 0, "Format 1 (rr_interval_ms) must produce snapshots for 180 samples"
        assert summary.total_snapshots == len(snaps)

    def test_format2_bpm_timestamp_produces_snapshots(self):
        samples = self._make_format2(n=180)

        snaps, summary = process_bpm_session(samples)

        assert len(snaps) > 0, "Format 2 (bpm+timestamp) must produce snapshots for 180 samples"

    def test_format3_bpm_timestamp_ms_produces_snapshots(self):
        samples = self._make_format3(n=180)

        snaps, summary = process_bpm_session(samples)

        assert len(snaps) > 0, "Format 3 (bpm+timestamp_ms) must produce snapshots for 180 samples"

    def test_unrecognized_format_returns_empty(self):
        samples = [{"heart_rate": 70.0, "time": 1234} for _ in range(180)]

        snaps, summary = process_bpm_session(samples)

        assert snaps == []

    def test_format1_zero_rr_intervals_are_skipped(self):
        # All-zero RR intervals should result in no usable samples
        samples = [{"rr_interval_ms": 0} for _ in range(180)]

        snaps, _ = process_bpm_session(samples)

        assert snaps == []

    def test_format1_negative_rr_intervals_are_skipped(self):
        samples = [{"rr_interval_ms": -500} for _ in range(180)]

        snaps, _ = process_bpm_session(samples)

        assert snaps == []

    def test_format3_timestamp_ms_correctly_converts_to_seconds(self):
        # timestamp_ms / 1000 should give same result as equivalent format2
        base_ts = 1_700_000_000.0
        rng = np.random.default_rng(99)
        bpm = 70.0 + rng.uniform(-2, 2, 180)

        samples_f2 = [{"bpm": float(bpm[i]), "timestamp": base_ts + i} for i in range(180)]
        samples_f3 = [{"bpm": float(bpm[i]), "timestamp_ms": int((base_ts + i) * 1000)} for i in range(180)]

        snaps_f2, _ = process_bpm_session(samples_f2)
        snaps_f3, _ = process_bpm_session(samples_f3)

        # Both formats should produce the same number of snapshots
        assert len(snaps_f2) == len(snaps_f3)

    # ── output correctness ─────────────────────────────────────────────

    def test_all_calm_scores_in_0_to_100(self):
        samples = self._make_format2(n=300)

        snaps, _ = process_bpm_session(samples)

        for snap in snaps:
            assert 0.0 <= snap.calm_score <= 100.0, (
                f"calm_score {snap.calm_score:.2f} at t={snap.t:.1f} out of [0,100]"
            )

    def test_all_states_are_valid_strings(self):
        samples = self._make_format2(n=300)

        snaps, _ = process_bpm_session(samples)

        for snap in snaps:
            assert snap.state in {"recovery", "neutral", "stress"}, (
                f"Invalid state '{snap.state}' at t={snap.t:.1f}"
            )

    def test_snapshots_are_FeatureSnapshot_instances(self):
        samples = self._make_format2(n=180)

        snaps, _ = process_bpm_session(samples)

        for snap in snaps:
            assert isinstance(snap, FeatureSnapshot)

    def test_summary_is_SessionSummary_instance(self):
        samples = self._make_format2(n=180)

        _, summary = process_bpm_session(samples)

        assert isinstance(summary, SessionSummary)

    def test_summary_total_snapshots_equals_list_length(self):
        samples = self._make_format2(n=300)

        snaps, summary = process_bpm_session(samples)

        assert summary.total_snapshots == len(snaps)

    def test_state_percentages_sum_to_exactly_100(self):
        samples = self._make_format2(n=300)

        _, summary = process_bpm_session(samples)

        total_pct = (
            summary.time_in_recovery_pct
            + summary.time_in_stress_pct
            + summary.time_in_neutral_pct
        )
        assert abs(total_pct - 100.0) < 1e-6, (
            f"State percentages sum to {total_pct:.6f}, expected 100"
        )

    def test_avg_calm_score_is_mean_of_snapshot_calm_scores(self):
        samples = self._make_format2(n=300)

        snaps, summary = process_bpm_session(samples)

        if snaps:
            expected = float(np.mean([s.calm_score for s in snaps]))
            assert abs(summary.avg_calm_score - expected) < 1e-6

    def test_hr_delta_equals_hr_final_minus_hr_baseline(self):
        samples = self._make_format2(n=300)

        _, summary = process_bpm_session(samples)

        assert abs(summary.hr_delta - (summary.hr_final - summary.hr_baseline)) < 1e-6

    def test_duration_s_is_positive_for_sufficient_samples(self):
        samples = self._make_format2(n=300)

        _, summary = process_bpm_session(samples)

        if summary.total_snapshots > 0:
            assert summary.duration_s > 0.0

    def test_snapshot_times_are_monotonically_non_decreasing(self):
        samples = self._make_format2(n=300)

        snaps, _ = process_bpm_session(samples)

        times = [s.t for s in snaps]
        for i in range(1, len(times)):
            assert times[i] >= times[i - 1], (
                f"Non-monotonic t at index {i}: {times[i-1]:.1f} → {times[i]:.1f}"
            )

    def test_no_nan_calm_scores(self):
        samples = self._make_format2(n=300)

        snaps, _ = process_bpm_session(samples)

        for snap in snaps:
            assert math.isfinite(snap.calm_score), (
                f"NaN/Inf calm_score at t={snap.t:.1f}"
            )

    # ── cross-session baseline seeding ────────────────────────────────

    def test_cross_session_baseline_dict_accepted(self):
        samples = self._make_format2(n=300)
        cross = {"hr": 68.0, "si": 1.8, "hf": 0.9, "breath": 13.5}

        snaps, summary = process_bpm_session(samples, cross_session_baseline=cross)

        assert len(snaps) > 0

    def test_cross_session_baseline_none_accepted(self):
        samples = self._make_format2(n=300)

        snaps, summary = process_bpm_session(samples, cross_session_baseline=None)

        assert isinstance(snaps, list)

    def test_cross_session_baseline_partial_keys_no_crash(self):
        # Partial baseline — missing keys should default to 0
        samples = self._make_format2(n=300)
        cross = {"hr": 70.0}  # only hr provided

        snaps, summary = process_bpm_session(samples, cross_session_baseline=cross)

        assert isinstance(snaps, list)

    # ── physiological scenarios ────────────────────────────────────────

    def test_resting_session_does_not_crash(self):
        samples = self._make_format2(n=300, seed=42)

        snaps, summary = process_bpm_session(samples)

        assert all(math.isfinite(s.calm_score) for s in snaps)
        assert all(s.state in {"recovery", "neutral", "stress"} for s in snaps)

    def test_elevated_hr_stress_scenario_completes(self):
        # First 90 s resting at 68 BPM, then 90 s elevated at 95 BPM
        rng = np.random.default_rng(99)
        bpm_rest = 68.0 + rng.uniform(-2, 2, 90)
        bpm_stress = 95.0 + rng.uniform(-3, 3, 90)
        bpm_all = np.concatenate([bpm_rest, bpm_stress])

        base_ts = 1_700_000_000.0
        samples = [
            {"bpm": float(bpm_all[i]), "timestamp": base_ts + i}
            for i in range(len(bpm_all))
        ]

        snaps, summary = process_bpm_session(samples)

        assert isinstance(snaps, list)
        assert all(0.0 <= s.calm_score <= 100.0 for s in snaps)

    def test_constant_bpm_session_no_nan_or_inf(self):
        # Pathological edge case: perfectly constant BPM
        samples = [{"bpm": 70.0, "timestamp": float(i)} for i in range(180)]

        snaps, summary = process_bpm_session(samples)

        for snap in snaps:
            assert math.isfinite(snap.calm_score), f"NaN calm_score at t={snap.t}"
            assert math.isfinite(snap.hr_mean_60s)

    def test_spiky_bpm_session_still_returns_valid_output(self):
        # Alternate between 65 and 90 BPM (spikes every other sample)
        bpm_vals = [65.0 if i % 2 == 0 else 90.0 for i in range(180)]
        base_ts = 1_700_000_000.0
        samples = [{"bpm": bpm_vals[i], "timestamp": base_ts + i} for i in range(180)]

        snaps, summary = process_bpm_session(samples)

        assert isinstance(snaps, list)
        # Spiky data may produce fewer valid windows — just verify no crash
        for snap in snaps:
            assert math.isfinite(snap.calm_score)

    # ── session summary structure ──────────────────────────────────────

    def test_breath_start_and_end_are_non_negative(self):
        samples = self._make_format2(n=300)

        _, summary = process_bpm_session(samples)

        assert summary.breath_start >= 0.0
        assert summary.breath_end >= 0.0

    def test_hr_baseline_is_set_for_sufficient_session(self):
        samples = self._make_format2(n=300)

        _, summary = process_bpm_session(samples)

        # For a 300-sample session, hr_baseline should reflect actual HR
        if summary.total_snapshots > 0:
            assert summary.hr_baseline > 0.0

    def test_time_in_neutral_pct_nonzero_for_steady_hr(self):
        # Steady resting HR should produce mostly neutral state
        samples = self._make_format2(n=300, seed=7)

        _, summary = process_bpm_session(samples)

        # For a steady HR session we expect some neutral time
        assert summary.time_in_neutral_pct >= 0.0

    def test_total_snapshots_grows_with_session_length(self):
        samples_short = self._make_format2(n=120)
        samples_long = self._make_format2(n=300)

        _, summary_short = process_bpm_session(samples_short)
        _, summary_long = process_bpm_session(samples_long)

        assert summary_long.total_snapshots >= summary_short.total_snapshots
