-- Migration 005: add reviewer decision columns to processed.coding_results
-- Run on Hetzner:
--   docker exec vigilex-postgres psql -U vigilex -d vigilex \
--     -f /home/cap/vigilex/migrations/005_reviewer_columns.sql

ALTER TABLE processed.coding_results
    ADD COLUMN IF NOT EXISTS reviewer_action  TEXT,
    ADD COLUMN IF NOT EXISTS reviewer_at      TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS reviewer_note    TEXT;

-- reviewer_action values: 'accepted' | 'rejected' | 'overridden'
-- reviewer_at: timestamp of the reviewer action (NULL = not yet reviewed)
-- reviewer_note: optional free-text comment from the reviewer

COMMENT ON COLUMN processed.coding_results.reviewer_action IS
    'Reviewer decision: accepted | rejected | overridden. NULL = pending.';
COMMENT ON COLUMN processed.coding_results.reviewer_at IS
    'Timestamp when the reviewer made the decision.';
COMMENT ON COLUMN processed.coding_results.reviewer_note IS
    'Optional reviewer comment or correction note.';
