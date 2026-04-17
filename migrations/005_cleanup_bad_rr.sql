-- Migration 005: Clean up duplicate mindfulness sessions with bad HealthKit RR data
-- The HealthKit path produced ~5000ms inter-sample timestamps (not beat-to-beat RR),
-- resulting in hr=12 BPM and inflated SDNN. These are always the second upload
-- (has ending_hrv) with mean_hr < 30. The Watch direct upload (good data) is kept.

DELETE FROM mindfulness_sessions
WHERE id IN (
    SELECT id FROM mindfulness_sessions
    WHERE session_hrv IS NOT NULL
      AND (session_hrv->>'mean_hr')::numeric < 30
);
