-- Migration 004: Add per-track full-session HRV and calm score linking to mindfulness_sessions
-- This links the calm_score_session (from heartbeat_series ingest pipeline)
-- to the mindfulness_session (from watch session upload) for unified per-track HRV.

ALTER TABLE mindfulness_sessions
  ADD COLUMN IF NOT EXISTS session_hrv    JSONB,
  ADD COLUMN IF NOT EXISTS calm_score_ref BIGINT REFERENCES health_samples(id),
  ADD COLUMN IF NOT EXISTS calm_summary   JSONB;

-- Index for the backfill time-window lookup
CREATE INDEX IF NOT EXISTS idx_mindfulness_sessions_user_start
  ON mindfulness_sessions (user_id, start_time DESC);
