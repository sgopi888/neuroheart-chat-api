-- 007_account_deletion.sql
-- Apple remediation Phase 5: in-app account deletion.
--
-- Minimal security-retention table for deleted accounts (Task 5.5), aligned with
-- the NeuroHeart AI Privacy Policy (June 2026), Section 8 "Account Deletion and
-- Data Retention".
--
-- Retains ONLY non-content security metadata for fraud / abuse / legal-compliance
-- purposes, for up to 180 days:
--   user_id, deletion timestamp, optional reason, purge-after date.
-- It MUST NOT retain chat, wellness, journal, HRV, or heart-rate content.
--
-- IMPORTANT: this table intentionally has NO foreign key to users(user_id), so
-- the retention record survives after the user row (and its CASCADE children:
-- health_samples, mindfulness_sessions) are deleted.

CREATE TABLE IF NOT EXISTS deleted_accounts (
    user_id      TEXT        NOT NULL,
    deleted_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- Per privacy policy: limited security/audit records retained up to 180 days.
    -- A separate sweep job may remove rows older than purge_after.
    purge_after  TIMESTAMPTZ NOT NULL DEFAULT (now() + INTERVAL '180 days'),
    reason       TEXT,
    PRIMARY KEY (user_id, deleted_at)
);

CREATE INDEX IF NOT EXISTS idx_deleted_accounts_purge_after
    ON deleted_accounts (purge_after);
