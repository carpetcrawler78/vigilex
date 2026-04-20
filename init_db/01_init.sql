-- =============================================================================
-- SentinelAI (vigilex) — PostgreSQL Initialization Script
-- Run once on a fresh database to set up extensions + full schema.
-- Compatible with: PostgreSQL 15+ on Hetzner CX33 (Ubuntu 22.04)
-- =============================================================================


-- =============================================================================
-- 1. EXTENSIONS
-- Must be run as superuser (postgres). All other DDL below can run as app user.
-- =============================================================================

-- Vector similarity search — stores PubMedBERT embeddings (dim 768)
-- Used in: Module 2 (MedDRA Coding Engine), Module 3 (RAG interface)
CREATE EXTENSION IF NOT EXISTS vector;

-- Trigram-based fuzzy text search — enables BM25-style lexical matching
-- Used in: hybrid search (BM25 side), narrative text similarity
-- Also enables GIN/GIST indexes on text columns
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Allows GIST indexes on ranges (date ranges, numeric ranges)
-- Used in: range queries on report dates, PRR/ROR threshold queries
CREATE EXTENSION IF NOT EXISTS btree_gist;

-- Tracks slow queries, execution counts, cache hit rates
-- Used in: performance monitoring — check via pg_stat_statements view
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- Native full-text search (tsvector/tsquery) is built into PostgreSQL --
-- no separate extension needed. pg_search (ParadeDB) would require a
-- different base image (paradedb/paradedb) -- not used here.
-- tsvector + GIN index (idx_maude_mdr_text_fts below) covers this use case.


-- =============================================================================
-- 2. SCHEMA SEPARATION
-- We use three schemas to separate concerns clearly.
-- =============================================================================

-- Raw ingested data (MAUDE, Recalls)
CREATE SCHEMA IF NOT EXISTS raw;

-- Processed / enriched data (MedDRA codes, embeddings, signals)
CREATE SCHEMA IF NOT EXISTS processed;

-- MLflow experiment tracking tables (MLflow creates these itself, but we
-- pre-create the schema so it lands in the right place)
CREATE SCHEMA IF NOT EXISTS mlflow;


-- =============================================================================
-- 3. RAW SCHEMA — MAUDE ADVERSE EVENT REPORTS
-- One row = one MAUDE report from the openFDA API.
-- Corresponds to: flatten_maude_record() output in vigilex/data/
-- =============================================================================

CREATE TABLE IF NOT EXISTS raw.maude_reports (
    -- Primary key: openFDA MDR report number (globally unique)
    mdr_report_key          TEXT PRIMARY KEY,

    -- Report metadata
    date_received           DATE,
    date_of_event           DATE,
    report_source_code      TEXT,         -- e.g. 'Manufacturer', 'Voluntary'
    report_to_fda           TEXT,

    -- Device fields (from device[] array — flattened to first device)
    device_name             TEXT,
    device_brand_name       TEXT,
    product_code            TEXT,         -- e.g. 'LZG' (insulin pump), 'QFG' (CGM)
    manufacturer_name       TEXT,
    model_number            TEXT,
    lot_number              TEXT,
    device_age_text         TEXT,

    -- Patient outcome
    patient_sequence_number TEXT,
    date_of_birth           TEXT,         -- stored as text — MAUDE often sends partial dates
    patient_weight          TEXT,
    patient_age             TEXT,
    patient_sex             TEXT,
    sequence_of_events      TEXT,

    -- Narrative text — the main field for NLP and MedDRA coding
    -- This is what Module 2 processes
    mdr_text                TEXT,         -- full free-text adverse event description
    mdr_text_tsv            TSVECTOR,     -- pre-computed full-text search vector (auto-updated)

    -- Recall linkage (populated by 02_recall_labels.ipynb)
    recalled_ever           BOOLEAN DEFAULT FALSE,
    recall_count            INTEGER DEFAULT 0,

    -- Ingestion bookkeeping
    ingested_at             TIMESTAMPTZ DEFAULT NOW(),
    data_source             TEXT DEFAULT 'openFDA_MAUDE',
    api_batch_id            TEXT          -- which fetch_maude_by_daterange() call produced this row
);

-- Full-text search index over narrative (used by pg_search / tsvector queries)
CREATE INDEX IF NOT EXISTS idx_maude_mdr_text_fts
    ON raw.maude_reports USING GIN (mdr_text_tsv);

-- Trigram index for fuzzy/BM25-style text search (pg_trgm)
CREATE INDEX IF NOT EXISTS idx_maude_mdr_text_trgm
    ON raw.maude_reports USING GIN (mdr_text gin_trgm_ops);

-- Product code index — most queries filter on this (device type)
CREATE INDEX IF NOT EXISTS idx_maude_product_code
    ON raw.maude_reports (product_code);

-- Date range index (btree_gist) — supports date range queries in signal detection
CREATE INDEX IF NOT EXISTS idx_maude_date_received
    ON raw.maude_reports (date_received);

-- Auto-update tsvector when mdr_text changes
CREATE OR REPLACE FUNCTION raw.update_mdr_text_tsv()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    NEW.mdr_text_tsv := to_tsvector('english', COALESCE(NEW.mdr_text, ''));
    RETURN NEW;
END;
$$;

CREATE TRIGGER trg_maude_mdr_text_tsv
    BEFORE INSERT OR UPDATE OF mdr_text
    ON raw.maude_reports
    FOR EACH ROW EXECUTE FUNCTION raw.update_mdr_text_tsv();


-- =============================================================================
-- 4. RAW SCHEMA — FDA DEVICE RECALLS
-- Sourced from FDA Recall database (used in Notebook 02: recall_labels.ipynb)
-- Joined to maude_reports on product_code to compute recalled_ever labels.
-- =============================================================================

CREATE TABLE IF NOT EXISTS raw.device_recalls (
    recall_number           TEXT PRIMARY KEY,  -- FDA recall ID (e.g. Z-1234-2021)
    product_code            TEXT,
    product_description     TEXT,
    recalling_firm          TEXT,
    recall_class            TEXT,              -- Class I / II / III
    recall_initiation_date  DATE,
    termination_date        DATE,
    voluntary_mandated      TEXT,
    reason_for_recall       TEXT,

    ingested_at             TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_recalls_product_code
    ON raw.device_recalls (product_code);


-- =============================================================================
-- 5. PROCESSED SCHEMA — MEDDRA TERMS
-- Populated after MSSO Non-Profit license is approved.
-- Stores the MedDRA hierarchy for coding: LLT → PT → HLT → HLGT → SOC
-- =============================================================================

CREATE TABLE IF NOT EXISTS processed.meddra_terms (
    -- MedDRA Lowest Level Term (most specific)
    llt_code                INTEGER,
    llt_name                TEXT,

    -- Preferred Term (standard coding level — what Module 2 outputs)
    pt_code                 INTEGER PRIMARY KEY,
    pt_name                 TEXT,

    -- Higher levels (for grouping / signal analysis)
    hlt_code                INTEGER,           -- High Level Term
    hlt_name                TEXT,
    hlgt_code               INTEGER,           -- High Level Group Term
    hlgt_name               TEXT,
    soc_code                INTEGER,           -- System Organ Class
    soc_name                TEXT,

    -- Pre-computed PubMedBERT embedding of pt_name
    -- dim=768 matches PubMedBERT output; used by FAISS in Module 2
    pt_embedding            VECTOR(768),

    meddra_version          TEXT DEFAULT '27.0'
);

-- Vector similarity index (IVFFlat — approximate, fast for large MedDRA term sets)
-- Note: populate this index AFTER bulk-inserting all MedDRA terms, not before.
-- CREATE INDEX idx_meddra_embedding ON processed.meddra_terms
--     USING ivfflat (pt_embedding vector_cosine_ops) WITH (lists = 100);
-- ↑ Uncomment and run manually after inserting MedDRA data.

-- Trigram index on pt_name for lexical matching alongside vectors
CREATE INDEX IF NOT EXISTS idx_meddra_pt_name_trgm
    ON processed.meddra_terms USING GIN (pt_name gin_trgm_ops);


-- =============================================================================
-- 6. PROCESSED SCHEMA — MEDDRA CODING RESULTS
-- Output of Module 2: each row = one coding attempt for one MAUDE report.
-- =============================================================================

CREATE TABLE IF NOT EXISTS processed.coding_results (
    id                      SERIAL PRIMARY KEY,
    mdr_report_key          TEXT REFERENCES raw.maude_reports(mdr_report_key),

    -- MedDRA output
    pt_code                 INTEGER,           -- predicted PT code
    pt_name                 TEXT,
    llt_code                INTEGER,
    llt_name                TEXT,
    soc_name                TEXT,

    -- Confidence scoring
    vector_similarity       FLOAT,             -- cosine similarity from FAISS
    crossencoder_score      FLOAT,             -- MiniLM-L-6 reranker score
    llm_confidence          FLOAT,             -- LLM-assigned confidence (0–1)
    final_confidence        FLOAT,             -- composite score

    -- Model versioning (MLflow run ID)
    mlflow_run_id           TEXT,
    model_version           TEXT,

    coded_at                TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_coding_mdr_key
    ON processed.coding_results (mdr_report_key);

CREATE INDEX IF NOT EXISTS idx_coding_pt_code
    ON processed.coding_results (pt_code);


-- =============================================================================
-- 7. PROCESSED SCHEMA — SIGNAL DETECTION RESULTS (PRR / ROR)
-- Output of Module 3: disproportionality analysis per (product_code, pt_code).
-- =============================================================================

CREATE TABLE IF NOT EXISTS processed.signal_results (
    id                      SERIAL PRIMARY KEY,

    -- What we computed the signal for
    product_code            TEXT,              -- device type (e.g. 'LZG')
    pt_code                 INTEGER,           -- MedDRA PT
    pt_name                 TEXT,
    soc_name                TEXT,

    -- Time window for this computation
    analysis_start_date     DATE,
    analysis_end_date       DATE,

    -- Raw counts
    n_reports_focal         INTEGER,           -- reports for this device × PT combo
    n_reports_total         INTEGER,           -- total reports in window
    n_pt_all_devices        INTEGER,           -- reports for this PT across all devices
    n_all_devices_total     INTEGER,           -- total reports across all devices

    -- Disproportionality metrics
    prr                     FLOAT,             -- Proportional Reporting Ratio
    prr_lower_ci            FLOAT,             -- 95% CI lower bound
    prr_upper_ci            FLOAT,             -- 95% CI upper bound
    ror                     FLOAT,             -- Reporting Odds Ratio
    ror_lower_ci            FLOAT,
    ror_upper_ci            FLOAT,

    -- Alert flag
    is_signal               BOOLEAN DEFAULT FALSE,   -- TRUE if PRR ≥ 2 AND n ≥ 3
    signal_threshold_config JSONB,             -- which thresholds triggered this

    computed_at             TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_signal_product_pt
    ON processed.signal_results (product_code, pt_code);

CREATE INDEX IF NOT EXISTS idx_signal_is_signal
    ON processed.signal_results (is_signal)
    WHERE is_signal = TRUE;   -- partial index — only index the alerts


-- =============================================================================
-- 8. PROCESSED SCHEMA — NARRATIVE EMBEDDINGS
-- Stores PubMedBERT embeddings for MAUDE report narratives.
-- Used by Module 3 RAG interface for semantic search over reports.
-- Separate table (not a column on maude_reports) — embeddings are large
-- and we want to insert them in a separate pipeline step.
-- =============================================================================

CREATE TABLE IF NOT EXISTS processed.report_embeddings (
    mdr_report_key          TEXT PRIMARY KEY
                            REFERENCES raw.maude_reports(mdr_report_key),
    embedding               VECTOR(768),
    embedded_at             TIMESTAMPTZ DEFAULT NOW(),
    model_name              TEXT DEFAULT 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract'
);

-- IVFFlat vector index for ANN search over report embeddings
-- Populate after bulk insert (same as meddra_terms index above)
-- CREATE INDEX idx_report_embedding ON processed.report_embeddings
--     USING ivfflat (embedding vector_cosine_ops) WITH (lists = 200);


-- =============================================================================
-- 9. HELPER VIEW — HYBRID SEARCH BASELINE
-- Combines BM25 (trigram) + semantic (vector) scores via RRF.
-- This is NOT a materialized view — it's a building block for queries.
-- The actual RRF fusion happens in Python (vigilex/search/hybrid.py).
-- This view just exposes the pre-joined fields needed for search.
-- =============================================================================

CREATE OR REPLACE VIEW processed.search_base AS
SELECT
    r.mdr_report_key,
    r.product_code,
    r.date_received,
    r.mdr_text,
    r.mdr_text_tsv,
    r.recalled_ever,
    e.embedding,
    c.pt_code,
    c.pt_name,
    c.soc_name,
    c.final_confidence
FROM raw.maude_reports r
LEFT JOIN processed.report_embeddings e  USING (mdr_report_key)
LEFT JOIN processed.coding_results c     USING (mdr_report_key);


-- =============================================================================
-- 10. PERMISSIONS
-- Two roles: vigilex_app (read/write for Docker services),
--            vigilex_readonly (for Grafana dashboard — read only)
-- =============================================================================

-- Application user (api, worker containers)
DO $$ BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'vigilex_app') THEN
        CREATE ROLE vigilex_app LOGIN PASSWORD 'CHANGE_ME_IN_ENV';
    END IF;
END $$;

GRANT USAGE ON SCHEMA raw, processed TO vigilex_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA raw TO vigilex_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA processed TO vigilex_app;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA processed TO vigilex_app;

-- Grafana read-only user (only needs SELECT on signal_results + search_base)
DO $$ BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'vigilex_readonly') THEN
        CREATE ROLE vigilex_readonly LOGIN PASSWORD 'CHANGE_ME_IN_ENV';
    END IF;
END $$;

GRANT USAGE ON SCHEMA raw, processed TO vigilex_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA raw TO vigilex_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA processed TO vigilex_readonly;


-- =============================================================================
-- DONE
-- Run with: psql -U postgres -d vigilex -f init.sql
-- Check extensions: SELECT * FROM pg_extension;
-- Check tables: \dt raw.* \dt processed.*
-- =============================================================================
