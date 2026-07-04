-- Migration 008: Async meditation generation jobs
-- Lets the client start a generation, get a job_id immediately, and poll for
-- the result. This makes generation survive the app being backgrounded, killed,
-- or the phone locked/rebooted mid-request, since the work runs server-side and
-- the client re-attaches by job_id.

CREATE TABLE IF NOT EXISTS meditation_jobs (
    id                UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    user_uid          TEXT        NOT NULL,
    conversation_id   UUID        NOT NULL,
    status            TEXT        NOT NULL DEFAULT 'pending'
                                  CHECK (status IN ('pending', 'running', 'ready', 'failed')),
    -- Request parameters (so the worker has everything it needs)
    mood              TEXT        NOT NULL,
    depth             TEXT,
    duration          INT         NOT NULL,
    session_type      TEXT        NOT NULL DEFAULT 'meditation',
    music_config      JSONB,
    -- Result (populated when status = 'ready')
    session_id        UUID,
    title             TEXT,
    audio_url         TEXT,
    meditation_type   TEXT,
    -- Failure detail (populated when status = 'failed')
    error             TEXT,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_meditation_jobs_user
    ON meditation_jobs(user_uid, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_meditation_jobs_status
    ON meditation_jobs(status);
