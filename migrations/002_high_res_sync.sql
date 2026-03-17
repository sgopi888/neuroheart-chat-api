-- Migration 002: Health Data Sync (Consolidated)
-- This migration adds the core user and health sample tables required for 
-- syncing heart rate, HRV, steps, and the new high-resolution heartbeat series.

-- 1. Users Table (Stores Apple profile data)
CREATE TABLE IF NOT EXISTS users (
    user_id      TEXT        PRIMARY KEY, -- The Apple 'sub' (User UID)
    email        TEXT,
    name         TEXT,
    age_range    TEXT,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_seen_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_users_last_seen ON users(last_seen_at DESC);

-- 2. Health Samples Table (Flexible storage for all metrics)
-- This table handles: heart_rate, hrv, hrv_sdnn, steps, sleep, and heartbeat_series.
CREATE TABLE IF NOT EXISTS health_samples (
    id           BIGSERIAL   PRIMARY KEY,
    user_id      TEXT        NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    sample_type  TEXT        NOT NULL, -- e.g., 'hrv_sdnn', 'heartbeat_series'
    start_time   TIMESTAMPTZ NOT NULL,
    end_time     TIMESTAMPTZ,
    value        NUMERIC,
    unit         TEXT,
    source       TEXT,
    payload      JSONB,               -- Stores RR intervals or computed_metrics
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Indexes for performance during AI context retrieval
CREATE INDEX IF NOT EXISTS idx_health_samples_user_time 
    ON health_samples(user_id, start_time DESC);

CREATE INDEX IF NOT EXISTS idx_health_samples_type_time
    ON health_samples(sample_type, start_time DESC);

-- Partial index for high-res data sessions
CREATE INDEX IF NOT EXISTS idx_health_samples_payload 
    ON health_samples(user_id) WHERE payload IS NOT NULL;
