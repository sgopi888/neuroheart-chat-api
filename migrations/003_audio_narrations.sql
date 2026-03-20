-- Migration 003: Audio narrations table for meditation audio storage
-- Tracks voice, music, and merged audio files per user session

CREATE TABLE IF NOT EXISTS audio_narrations (
    id                UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    user_uid          TEXT        NOT NULL,
    conversation_id   UUID        NOT NULL,
    session_id        UUID        NOT NULL,
    meditation_type   TEXT        NOT NULL CHECK (meditation_type IN ('short', 'medium', 'deep')),
    audio_type        TEXT        NOT NULL CHECK (audio_type IN ('voice', 'music', 'merged')),
    file_path         TEXT        NOT NULL,
    duration_seconds  INT,
    title             TEXT,
    metadata          JSONB,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_audio_narrations_user
    ON audio_narrations(user_uid, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_audio_narrations_session
    ON audio_narrations(session_id);
