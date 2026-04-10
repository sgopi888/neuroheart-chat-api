# BPM Sampling Plan: ~1s HR Collection + RR Interval Pipeline

## Context

**Problem:** The Watch app collects heart rate at ~5s intervals via `HKAnchoredObjectQuery`, but:
1. `HKHeartbeatSeriesSample` is NEVER created — Apple Watch does NOT auto-generate these for third-party apps
2. `hrv_sdnn` value is always NULL in backend (field name mismatch — already fixed)
3. `heartbeat_series` payload is always NULL (iOS race condition — already fixed on frontend)

**Goal:** Move from ~5s averaged BPM to ~1s HR sampling during mindfulness sessions. Collect real-time HR → estimate RR intervals → create proper `HKHeartbeatSeriesSample` in HealthKit → send to backend for HRV computation.

**Key insight from Apple docs:** `HKLiveWorkoutBuilder.statistics(for:).mostRecentQuantity()` provides ~1s HR updates when polled from `workoutBuilder(_:didCollectDataOf:)` delegate callback. This is the highest frequency available to third-party Watch apps.

## Architecture

```
Watch Session (2-5 min)
  │
  ├─ HKWorkoutSession + HKLiveWorkoutBuilder (existing)
  │    └─ NEW: HKLiveWorkoutBuilderDelegate.didCollectDataOf
  │         └─ statistics(for: .heartRate).mostRecentQuantity() → ~1s BPM
  │
  ├─ NEW: HKHeartbeatSeriesBuilder
  │    └─ addHeartbeatWithTimeInterval() for each estimated beat
  │    └─ finishSeries() on session end → creates HKHeartbeatSeriesSample
  │
  ├─ NEW: Accumulate (bpm, timestamp) array during session
  │    └─ Convert to RR intervals: rr_ms = 60000 / bpm
  │    └─ Package as heartbeat_series payload on session end
  │
  └─ WatchConnectivity → iPhone → /v1/ingest (existing pipeline)
```

## Implementation Steps

### Step 1: Add HKLiveWorkoutBuilderDelegate to WatchSessionManager

**File:** `CareKit_new/NeuroHeartSync/NeuroHeartSync Watch App/WatchSessionManager.swift`

- Add `HKLiveWorkoutBuilderDelegate` conformance
- Set `builder.delegate = self` in `startSession()` (line ~132)
- Implement the delegate method:
```swift
func workoutBuilder(_ builder: HKLiveWorkoutBuilder,
                    didCollectDataOf types: Set<HKSampleType>) {
    if types.contains(HKQuantityType(.heartRate)) {
        guard let bpm = builder.statistics(for: HKQuantityType(.heartRate))?
            .mostRecentQuantity()?
            .doubleValue(for: .count().unitDivided(by: .minute())) else { return }
        let timestamp = Date()
        collectedHRSamples.append((bpm: bpm, date: timestamp))
        // Also feed HKHeartbeatSeriesBuilder (Step 2)
    }
}
```

**Why this gives ~1s:** Apple calls `didCollectDataOf` each time the builder receives new data from the optical sensor. During an active workout, this is ~1s for heart rate.

### Step 2: Add HKHeartbeatSeriesBuilder

**File:** `WatchSessionManager.swift`

- Add properties:
  - `private var heartbeatSeriesBuilder: HKHeartbeatSeriesBuilder?`
  - `private var collectedHRSamples: [(bpm: Double, date: Date)] = []`

- **On session start** (in `startSession()`, after `beginCollection` succeeds):
```swift
heartbeatSeriesBuilder = HKHeartbeatSeriesBuilder(
    healthStore: healthStore,
    device: .local(),
    start: Date()
)
```

- **On each ~1s HR update** (in `didCollectDataOf`):
  - Estimate beat timestamp: `timeInterval = date.timeIntervalSince(sessionStartDate)`
  - Call `heartbeatSeriesBuilder.addHeartbeatWithTimeInterval(sinceSeriesStartDate: timeInterval, precededByGap: false)`

- **On session end** (in `stopSession()`, before workout finish):
```swift
heartbeatSeriesBuilder?.finishSeries { sample, error in
    // sample is the HKHeartbeatSeriesSample now in HealthKit
}
```

- **HealthKit write permission needed:** Add `HKSeriesType.heartbeat()` to `writeTypes` in `requestHealthKitAuth()`

**Reference:** Apple's official pattern from `HeartDogs/GameViewController.swift:179-235`

### Step 3: Package & Send RR Data on Session End

**File:** `WatchSessionManager.swift`

After session ends, convert accumulated HR samples to the backend payload format:

```swift
let rrIntervals: [[String: Any]] = collectedHRSamples.map { sample in
    ["rr_interval_ms": 60000.0 / sample.bpm]
}
let payload: [String: Any] = [
    "sample_type": "heartbeat_series",
    "start_time": iso8601(sessionStartDate),
    "end_time": iso8601(Date()),
    "value": nil,
    "payload": [
        "beat_count": collectedHRSamples.count,
        "rr_intervals": rrIntervals,
        "sampling_method": "live_hr_1s"
    ]
]
```

Send via WatchConnectivity `transferUserInfo` to iPhone, which forwards to `/v1/ingest`.

### Step 4: Remove Old Post-Session Polling

**File:** `WatchSessionManager.swift`

The `fetchPostSessionHeartbeatSeries()` method (lines 229-251) and `queryHeartbeatSeries()` (lines 253-288) are no longer needed since we now CREATE the heartbeat series ourselves during the session. Replace with a simple confirmation log after `finishSeries` succeeds.

### Step 5: Clean Up startHeartbeatMonitoring / Old HR Query

The existing `startLiveHeartRateQuery()` (lines 191-224) uses `HKAnchoredObjectQuery` for ~5s HR. This is superseded by the `didCollectDataOf` delegate approach (~1s). Remove the old query or keep it as fallback logging only.

## Files to Modify

1. **`WatchSessionManager.swift`** — Main changes:
   - Add `HKLiveWorkoutBuilderDelegate` conformance
   - Add `heartbeatSeriesBuilder` + `collectedHRSamples` properties
   - Set `builder.delegate = self` in `startSession()`
   - Implement `didCollectDataOf` delegate method
   - Create/finish `HKHeartbeatSeriesBuilder` on session start/end
   - Package RR data and send via WatchConnectivity
   - Remove old 180s polling code
   - Add `HKSeriesType.heartbeat()` to `writeTypes`

2. **`HealthKitManager.swift`** (already done) — `sdnn_value` → `value`, key alignment

3. **`app/ingest_router.py`** (already done) — model validator, key mismatch fix, logging

4. **`app/main.py`** (already done) — register ingest_router

## Key API References

| API | Purpose |
|-----|---------|
| `HKLiveWorkoutBuilderDelegate.didCollectDataOf` | Called ~1s with new HR data |
| `HKStatistics.mostRecentQuantity()` | Latest BPM value from builder |
| `HKHeartbeatSeriesBuilder(healthStore:device:start:)` | Creates beat series |
| `addHeartbeatWithTimeInterval(sinceSeriesStartDate:precededByGap:)` | Records individual beat |
| `finishSeries()` | Commits HKHeartbeatSeriesSample to HealthKit |
| `HKQuantitySeriesSampleBuilder` | Alternative for quantity series (from Apple sample) |

## Verification

1. Run a 2-min mindfulness session on Watch
2. Watch console should show: `💓 Live HR: XX bpm` every ~1s (not ~5s)
3. After session end, check HealthKit for new `HKHeartbeatSeriesSample`
4. Press Sync on iPhone, then on VPS:
```sql
SELECT start_time, jsonb_pretty(payload)
FROM health_samples
WHERE sample_type = 'heartbeat_series'
AND payload IS NOT NULL
ORDER BY start_time DESC
LIMIT 1;
```
5. Payload should contain `rr_intervals` array with `rr_interval_ms` values and `sampling_method: "live_hr_1s"`
