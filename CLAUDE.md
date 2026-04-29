# CLAUDE.md -- SentinelAI (vigilex) Project Briefing

Read this file first. It is the single source of truth for any AI assistant
working on this project. Update the "Current Status" and "Next Steps" sections
at the end of each working session.

Last updated: 2026-04-29

---

## Identity

| Field | Value |
|---|---|
| External name | SentinelAI (portfolio, LinkedIn, presentations) |
| Internal name | vigilex (repo, Python module, Docker services) |
| Concept | Adverse event signal detection for medical devices |
| Domain | Post-market surveillance, FDA MAUDE, MedDRA coding |
| Author | Thomas Heger -- Dr. sc. ETH Biochemistry, ex-DKFZ Clinical Data Manager |
| Bootcamp | neue fische AI Engineering Bootcamp 2025/2026 (Capstone II) |

---

## Architecture -- 3 Modules

### Module 1 -- MAUDE Ingestion
openFDA MAUDE API -> flatten_maude_record() -> PostgreSQL (raw.maude_reports)

Key files:
- src/vigilex/data/maude_client.py  -- fetch_maude_by_daterange(), flatten_maude_record(), upsert_maude_records()
- src/vigilex/workers/ingest.py     -- CLI, batch commits (500 records), full loop
- src/vigilex/db/connection.py      -- psycopg2 from DATABASE_URL

### Module 2 -- MedDRA Coding Engine (COMPLETE)
Stage 1: HybridSearcher    (BM25 pg_trgm + pgvector cosine, Weighted RRF) -> Top-20
Stage 2: CrossEncoderReranker (ms-marco-MiniLM-L-6-v2) -> Top-5
Stage 3: LLMCoder          (Ollama llama3.2:3b, local, no external API calls) -> PT code

Key files:
- src/vigilex/coding/hybrid_search.py   -- HybridSearcher + EmbeddingModel
- src/vigilex/coding/reranker.py        -- CrossEncoderReranker
- src/vigilex/coding/llm_coder.py       -- LLMCoder (Ollama REST client)
- src/vigilex/workers/coding.py         -- CodingWorker (SQL queue, polling loop)
- src/vigilex/data/import_meddra.py     -- MedDRA v29.0 import (PTs + LLTs)
- src/vigilex/coding/embed_meddra_terms.py -- PubMedBERT embedding generation

Confidence formula: 0.3 * sigmoid(CE logit) + 0.7 * LLM confidence
Flagged for human review if final_confidence < 0.5
LLM fallback: if Ollama unreachable, use top CE result with sigmoid(CE logit) as confidence

Hybrid search design:
- BM25: pg_trgm word_similarity(lower(pt_name), lower(narrative)) -- threshold 0.1
- Vector: pgvector cosine on PubMedBERT embeddings, IVFFlat probes=100
- Fusion: Weighted RRF k=60, w_bm25=0.4, w_vector=0.6
- Query encoding: first sentence only (full narrative dilutes embeddings -- Bug #5)

### Module 3 -- Signal Detection (NOT YET STARTED)
PRR/ROR disproportionality over processed.coding_results
Planned files:
- src/vigilex/signals/prr_ror.py
Dashboards: Grafana (primary), Streamlit (demo/portfolio)

---

## Infrastructure

| Item | Detail |
|---|---|
| Server | Hetzner CX33 x86, Nuremberg (EU/GDPR), 4 vCPU, 8 GB RAM, 80 GB SSD |
| Server IP | 46.225.109.99 |
| SSH user | cap (id_ed25519, key-only, sudo enabled) |
| Repo on server | /home/cap/vigilex |
| Autopull cron | daily 06:00, /home/cap/vigilex/hetzner_autopull.sh |
| Portainer | http://46.225.109.99:9000 (Hetzner firewall: home IP only) |
| Ollama | 127.0.0.1:11434 (local only), llama3.2:3b loaded |
| GitHub | github.com/carpetcrawler78/vigilex |
| Local dev | C:\Users\thheg\bootcamps\CAPSTONE II\vigilex (Windows) |
| Google Drive sync | G:\Meine Ablage\CAPSTONE II (nightly_push.ps1, 23:00 daily) |

Docker Compose services (9):
postgres, redis, api, worker-ingest, worker-coding, worker-signal,
mlflow, frontend (Streamlit), grafana

PostgreSQL extensions active (5):
vector, pg_trgm, btree_gist, pg_stat_statements, plpgsql

Schemas: raw, processed, mlflow, public

Key tables:
- raw.maude_reports          -- ingested MAUDE adverse event reports
- raw.device_recalls         -- FDA recall data (for label join)
- processed.meddra_terms     -- 27,361 PTs with pt_embedding (VECTOR 768)
- processed.meddra_llt       -- 81,719 LLTs (MedDRA v29.0)
- processed.coding_results   -- output of Module 2 coding pipeline
- processed.signal_results   -- output of Module 3 (not yet populated)
- processed.report_embeddings

MedDRA license:
- MSSO Non-Commercial, ID: 33549, active 2026-05-01 to 2027-04-30
- Account: thheger@web.de
- raw_data/ is in .gitignore (license data, not in version control)

---

## Current Status (as of 2026-04-29)

Block A -- Security:     DONE
Block B -- Server:       DONE (Fail2ban, SSH hardening, Hetzner firewall, autopull cron)
Block C -- Phase 2:      DONE (Docker + full DB schema live, MAUDE worker, 10k LZG records ingested)
Block D -- MedDRA:       DONE (MedDRA import, embeddings, hybrid search, reranker, LLM coder, coding worker)
Block E -- Phase 3/Demo: NOT STARTED

Last git commit: 2026-04-24 (nightly snapshot). No new code since then.
Module 2 fully complete as of ~2026-04-20 (CodingWorker, smoke test green).
Module 3 (PRR/ROR + Grafana + Streamlit) not yet started.

Data ingested so far:
- LZG (insulin pumps): 10,000 records (2024-01-01 to 2024-12-31)
  Note: openFDA hard limit is 10,000 per query -- full year is ~17,375 records.
  Full import needs monthly windowing. Other product codes (PKU, OYC, FRN, QFG) not yet pulled.

---

## Next Steps

1. Module 3 -- PRR/ROR signal detection
   - Create src/vigilex/signals/prr_ror.py
   - Proportional Reporting Ratio + Reporting Odds Ratio over processed.coding_results
   - Configurable alert thresholds per device type

2. Full MAUDE import
   - Split 2015-2024 into monthly windows for LZG
   - Then pull PKU (pacemakers), OYC (CGM sensors), FRN (defibrillators), QFG (ventilators)

3. Grafana dashboard
   - Signal alerts, confidence distribution, coding throughput

4. Streamlit frontend
   - Interactive review queue for low-confidence codings (flagged=True)
   - Signal browser

5. Stage 3 end-to-end smoke test
   - Run scripts/smoke_test_pipeline.py with both SSH tunnels open simultaneously

6. Notebook 02 (02_recall_labels.ipynb)
   - Recall label join: MAUDE x FDA Recall DB
   - Check recalled_ever distribution per product code

---

## Code Conventions

IMPORTANT -- enforced since 2026-04-10:
- ASCII only in all code files and scripts
- No Unicode special characters: use -- instead of em dash, -> instead of arrow
- No UTF-8 special characters in PowerShell scripts (BOM required if needed)
- All scripts (Python, Bash, PowerShell) must be ASCII-safe

Python style:
- Module path: vigilex.workers.coding (run as python -m vigilex.workers.coding)
- DB connections: always via psycopg2 from DATABASE_URL env var
- Workers connect fresh per batch (self-healing against SSH tunnel restarts)
- Batch size default: 500 (ingest), configurable via CLI for coding worker

SSH tunnel pattern (for local dev connecting to Hetzner):
  Port 5432 (postgres) and 11434 (Ollama) are bound to 127.0.0.1 on the server
  Access via SSH tunnel: ssh -L 5432:127.0.0.1:5432 -L 11434:127.0.0.1:11434 cap@46.225.109.99

Common gotcha: local ny-taxi-db container may conflict on port 5432 -- stop it first.

---

## Key Decisions and Differentiators

- Hybrid search (BM25 + semantic + RRF): strong interview talking point vs pure vector search
- Privacy-by-Design: all LLM inference via local Ollama, no PHI leaves EU infrastructure
- pg_search was removed (ParadeDB-specific, not in pgvector image) -- use pg_trgm instead
- word_similarity() not similarity() for BM25 arm (short PT vs long narrative -- Bug #3)
- First sentence only for query embedding to avoid narrative dilution (Bug #5)
- IVFFlat probes=100 required for deterministic results on this dataset size (Bug #6)

---

## Notebooks Overview

| Notebook | Topic | Status |
|---|---|---|
| 01_openfda_maude.ipynb | openFDA API, data pull, EDA | done |
| 02_recall_labels.ipynb | Recall label join MAUDE x FDA Recall | not yet run |
| 03_features_and_model.ipynb | Feature engineering, modeling | partial |
| 04_meddra_eda.ipynb | MedDRA file structure, SOC distribution | done |
| 05_meddra_hybrid_search.ipynb | Hybrid search walkthrough, RRF formula | done |
| 06_meddra_reranker.ipynb | CrossEncoder walkthrough, rank change analysis | done |
| 07_meddra_llm_coding.ipynb | Full pipeline: hybrid -> reranker -> LLM -> DB | done |
| 08_pipeline_debugging.ipynb | Bug diary: 6 bugs, root causes, fixes | done |
| 09_coding_worker.ipynb | CodingWorker explained walkthrough | done |

---

## How to Use This File

At the start of each session, tell your AI assistant:
  "Read CLAUDE.md first -- it has the full project context."

At the end of each session, ask your AI assistant to update:
  - "Current Status" block
  - "Next Steps" list
  - Any new conventions or decisions

This file lives at vigilex/CLAUDE.md and is tracked in git.
