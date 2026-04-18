-- Migration 006: Add narration_id to link mindfulness sessions to audio tracks
-- Enables multiple HRV readings per track (replay history).

ALTER TABLE mindfulness_sessions
  ADD COLUMN IF NOT EXISTS narration_id TEXT;

CREATE INDEX IF NOT EXISTS idx_mindfulness_sessions_narration
  ON mindfulness_sessions (narration_id)
  WHERE narration_id IS NOT NULL;
