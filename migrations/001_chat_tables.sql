-- NeuroHeart Chat API — Initial Schema
-- Run: psql -U neuroheart_user -d neuroheart -h localhost -f 001_chat_tables.sql

CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- Conversations (threads)
CREATE TABLE IF NOT EXISTS conversations (
    id           UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    user_uid     TEXT        NOT NULL,
    title        TEXT,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    is_archived  BOOLEAN     NOT NULL DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_conversations_user_updated
    ON conversations(user_uid, updated_at DESC);

-- Messages
CREATE TABLE IF NOT EXISTS chat_messages (
    id              BIGSERIAL   PRIMARY KEY,
    conversation_id UUID        NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    user_uid        TEXT        NOT NULL,
    role            TEXT        NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content         TEXT        NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    model           TEXT,
    metadata        JSONB
);

CREATE INDEX IF NOT EXISTS idx_messages_conv_time
    ON chat_messages(conversation_id, created_at ASC);

CREATE INDEX IF NOT EXISTS idx_messages_user_time
    ON chat_messages(user_uid, created_at DESC);

-- Rolling conversation summaries (for long-context management)
CREATE TABLE IF NOT EXISTS conversation_summaries (
    conversation_id              UUID        PRIMARY KEY REFERENCES conversations(id) ON DELETE CASCADE,
    user_uid                     TEXT        NOT NULL,
    summary                      TEXT        NOT NULL DEFAULT '',
    summarized_through_message_id BIGINT,
    updated_at                   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_conv_summaries_user
    ON conversation_summaries(user_uid, updated_at DESC);
