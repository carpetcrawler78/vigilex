# CLAUDE.md -- SentinelAI (vigilex) Project Briefing

Read this file first. It is the single source of truth for any AI assistant
working on this project. Update the "Current Status" and "Next Steps" sections
at the end of each working session.

Last updated: 2026-05-06 (LLM Backend Strategy + Demo Plan added)

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
   - Retrospective validation: could Module 3 signals have predicted actual recalls?

7. Future work (post-Capstone, not graded)
   - EUDAMED integration when API becomes available (~May 2026 mandate)
   - MDR Art. 83-86 compliant report generation (Periodic Safety Update Report template)
   - LoRA finetuning of LLM on FDA adverse event language (RunPod A40, ~$3-5)
   - Agent layer: Claude Managed Agents or LangGraph as interactive demo frontend
   - OpenClaw + Telegram for overnight batch monitoring and alert summaries
   - Figma -> React frontend for portfolio/recruiter demo
     Rationale: Streamlit is sufficient for Capstone grading, but a polished
     React UI (Signal Browser, MAUDE report viewer, Coding Review Queue) gives
     more visual impact for HealthTech recruiters (Roche, Siemens Healthineers,
     BfArM). Workflow: design in Figma -> Claude Code generates React + Tailwind
     -> wire to existing FastAPI endpoints. Estimated effort: 1-2 days post-Capstone.

---

## ML Engineering Tooling

### CI/CD (GitHub Actions)
- ci.yml: ruff + pytest + coverage on every push to `work` branch
- Deployment: Autopull cron (hetzner_autopull.sh, daily 06:00) -- NOT GitHub Actions
  Reason: Hetzner firewall restricts inbound SSH to home IP; cron avoids inbound connections
  entirely and reduces attack surface. Interview framing: intentional architecture decision.
- Privacy note: CI runners process only infrastructure code, no data/PHI -- no GDPR conflict

### Code Quality
- ruff -- linting + formatting (replaces black + flake8 + isort)
- pre-commit hooks: ruff, nbstripout, gitleaks (prevents API key leaks in commits)

### Testing
- pytest + coverage report as CI artifact
- Key test targets: flatten_maude_record(), PRR/ROR math, confidence formula (0.3*sigmoid + 0.7*LLM)

### Experiment Tracking & Model Registry
- MLflow already in Docker Compose -- use Stage transitions actively: Staging -> Production
- Track: hybrid search weights (w_bm25, w_vector), CE model versions, LLM model + temp

### Monitoring & Drift Detection (Evidently AI)
- Run after each MAUDE batch import:
  - Confidence score distribution vs. baseline
  - flagged rate per product code (LZG, PKU, OYC, FRN, QFG)
  - PT code diversity (detect degenerate coding toward same few terms)
- Output: HTML report + JSON for Grafana
- Interview angle: LLMOps content; in regulated PMS context, drift is MDR Art. 83 relevant

### Data Versioning
- No DVC -- MAUDE is a public API with deterministic re-ingestion, no DVC use case
- Instead: data/raw/manifest.json with hash + date-range per ingest run
- Interview rationale: "DVC evaluated; manifest sufficient for reproducible public API data.
  DVC added when moving to private training data."

### Hospital-Data-Ready Extensions (not built, but architecturally prepared)
- pgcrypto column-level encryption (postgres extension, one activation step away)
- Row-Level Security in PostgreSQL per service-user
- AVV (Auftragsverarbeitungsvertrag) with Hetzner -- contractual, not technical
- VPN for hospital-to-server connections instead of SSH tunnel
- Append-only audit log for PHI access (MDR Art. 10 + DSGVO Art. 30)
- Interview framing: "Path to clinical data is evolutionary, not a redesign."

---

## LLM Backend Strategy (Tier Architecture)

Added 2026-05-06 after EU-AI-Act / DSGVO compliance review.

### Concept
Not every task needs the same model. The system uses a four-tier mental model:

| Tier | Backend | Use cases in SentinelAI |
|---|---|---|
| 0 | No LLM (SQL, regex, math) | flatten_maude_record, PRR/ROR computation, pg_trgm BM25 arm |
| 1 | Local small model (Ollama llama3.2:3b on Hetzner) | MedDRA coding (Stage 3 multiple-choice over Top-5 PT candidates) -- current production default |
| 2 | Mid-tier model (larger local OR external mid-tier) | Optional: signal narrative drafts, batch summaries |
| 3 | Frontier model via EU-resident endpoint | Optional: PMS-style report generation, regulator-facing prose |

### Compliance Constraints (DSGVO + EU AI Act)
- Tier 0/1 stays on Hetzner CX33 (Nuernberg). Zero data egress, full control.
- Tier 2/3 must NOT use the direct Anthropic API (data routes via US, workspace data US-only). 
  Approved external paths: AWS Bedrock eu-central-1 (Frankfurt) or GCP Vertex AI europe-west1 (Belgium).
  Under Bedrock the AWS DPA applies, Anthropic does not retain prompts/outputs for training.
- EU AI Act risk attaches to the use case, not the provider:
  - Pure signal detection on public MAUDE data: not automatically high-risk.
  - Output piped into MDR Art. 83-86 PSUR workflows: high-risk (Annex III).
- GPAI documentation advantage of Anthropic: ISO 42001 + ISO 27001 + SOC 2 (drop-in for our technical file). For pure self-hosted llama3.2 the operator provides this documentation.
- Deadline window: original Aug 2026 high-risk obligations may shift to Dec 2027 (standalone) / Aug 2028 (embedded in regulated products) via Digital Omnibus.

### Implementation Plan (post-Block-D, pre-demo)
- src/vigilex/coding/llm_backend.py: pluggable backend layer
  - LocalOllamaBackend (current default)
  - BedrockEUBackend (Claude Sonnet, eu-central-1)
  - DirectAnthropicBackend (dev/test only, NEVER production)
- Config via env: VIGILEX_LLM_BACKEND, AWS_REGION, fallback chain
- All backends implement same interface: complete(prompt, schema) -> dict
- Existing LLMCoder constructor takes a backend instance instead of hard-coded Ollama URL

### Architecture Decision Record (informal)
Status: ACCEPTED 2026-05-06.
Default for production demo: Tier 1 (local Ollama) for coding, Tier 0 (math + templates) for signals.
Bonus demo path: Tier 3 (Bedrock EU) only as a backend toggle for the Signal Narrative Generator step in the live demo. This is the strongest architectural talking point: same code path, three backends, explicit tradeoff visible to the audience.

### Privacy-by-Design Talking Point (revised wording)
OLD: "all LLM inference via local Ollama, no PHI leaves EU infrastructure"
NEW: "Production data and PHI never leave the Hetzner trust boundary. Optional Tier 3 backend routes through AWS Bedrock eu-central-1 with the AWS DPA in effect; data never leaves EU jurisdiction."

---

## Capstone Demo Plan

Added 2026-05-06.

Total runtime: ~12 min presentation + 5 min Q&A. Demo runs in Streamlit + Grafana, no code edits live.

| Slide | Topic | Time |
|---|---|---|
| 1 | Hook: insulin-pump hyperglycaemia case | 45 s |
| 2 | Problem: MAUDE volume vs MDR Art. 83 manual coding | 60 s |
| 3 | Architecture overview (3 modules, Postgres-native pivot) | 90 s |
| 4 | Live: Module 1 ingestion status (Streamlit tab) | 45 s |
| 5 | Live: Module 2 coding inspector (Hybrid -> Reranker -> LLM) | 3 min |
| 6 | Live: Module 3 signal alerts (Grafana dashboard) | 2 min |
| 7 | Tier-Switch wow-moment (Local Template / Ollama / Bedrock EU) | 2 min |
| 8 | Compliance + architecture justification | 75 s |
| 9 | Lessons learned (3 bugs from bug-diary) + roadmap | 45 s |
| 10 | Q&A | 5 min |

Slide 7 is optional. Without Bedrock setup, drop slide 7 and keep slides 1-6 + 8-10. The Tier-Switch demonstration requires:
- AWS account, IAM user, eu-central-1, Claude Sonnet model access (one-time, ~30-60 min)
- BedrockEUBackend implemented in vigilex.coding.llm_backend (see plan above)
- Streamlit toggle in the Signal Narrative Generator tab

Q&A preparation cards (planned):
- "Why not ChatGPT/Claude direct?" -> EU residency, DSGVO, Bedrock EU as the bridge.
- "Why FDA data for an EU project?" -> MAUDE is the largest public dataset; EUDAMED follows ~May 2026 mandate.
- "How do you test for hallucinations?" -> Confidence formula, flagged review queue, Evidently AI drift report on PT-distribution.
- "Why PostgreSQL and not a vector DB?" -> Single-DB ACID, fewer failure modes, sufficient at <100k terms.

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
- Architecture pivot: original plan (Feb-Mar 2026) included SciSpacy NER, BERTopic
  cluster detection, FAISS/Weaviate vector DB, and LangGraph agent loop.
  Final build replaced all of these with a PostgreSQL-native approach
  (pg_trgm + pgvector + CrossEncoder + Ollama). Simpler, fewer moving parts,
  all search/storage in one DB. This was an intentional simplification, not a cut.
- Embedding model: PubMedBERT chosen over allenai-specter (originally discussed).
  PubMedBERT better captures medical terminology for MedDRA PT matching;
  specter is optimized for paper-level similarity, not term-level coding.
- LLM model: llama3.2:3b chosen over originally discussed Qwen2.5:14b/7b.
  Reason: CX33 runs 9 Docker containers + PostgreSQL on 8 GB RAM.
  3b model fits comfortably; 14b would OOM under load.
- Server migration: early planning (pre-Apr 2026) assumed Hetzner CAX11 (ARM).
  Actual deployment is on CX33 (x86) -- same server as tradetest (user: trader).
  Vigilex runs under user: cap. Both projects share the CX33 hardware.

---

## Design History (from chat discussions)

This section captures ideas discussed but not (yet) implemented,
so future sessions don't re-derive them.

Original Capstone II plan (Feb 2026):
- Option A: Claude Managed Agents with custom tools (query_maude, get_signal_analysis)
- Option B: Self-hosted LangGraph agent with FAISS, FastAPI, Bedrock eu-central-1
- Recommendation was "build both" -- Managed Agents for demo, self-hosted for portfolio
- Current build is neither -- it is a PostgreSQL-native pipeline without an agent layer

Dropped components (consciously):
- SciSpacy NER -- replaced by MedDRA coding (more structured, better for PMS domain)
- BERTopic clustering -- deferred; may revisit for Module 3 signal grouping
- FAISS -- replaced by pgvector IVFFlat (fewer dependencies, single-DB architecture)
- Weaviate -- same reasoning as FAISS
- LangGraph / LangChain -- not needed; coding pipeline is deterministic, not agentic

Still viable for future work:
- LoRA finetuning on RunPod A40 (~$3-5 one-shot) for domain-adapted LLM
- OpenClaw + Telegram integration for overnight batch monitoring
- Claude Managed Agents as a demo frontend for the presentation

---

## Presentation Talking Points

From chat discussions -- useful for bootcamp final presentation and job interviews:

- "Why not pure vector search?" -- Hybrid BM25+semantic beats either alone;
  BM25 catches exact medical terms, semantic catches paraphrases.
  Weighted RRF fusion is a known IR technique (Cormack et al., 2009).
- "Why PostgreSQL and not a dedicated vector DB?" -- Single-DB architecture
  means fewer failure modes, simpler ops, ACID guarantees across search + storage.
  pgvector IVFFlat is sufficient for <100k terms.
- "Why local Ollama and not an API?" -- Privacy-by-Design for EU regulated context.
  Even though MAUDE data is public, the architecture demonstrates GDPR-readiness
  for real clinical data. Strong differentiator for HealthTech employers.
- "Why FDA data for a European project?" -- MAUDE is the largest public adverse
  event database globally. EU MDR Article 83-86 requires post-market surveillance;
  EUDAMED (mandatory ~May 2026) will be an additional data source.
  FDA-first is a feature, not a limitation -- shows cross-regulatory competence.
- Notebook 02 (recall label join) validates the system retrospectively:
  "Could our signals have predicted actual FDA recalls?"

---

## Related Projects (same infrastructure)

| Project | Repo | Server user | Purpose |
|---|---|---|---|
| vigilex / SentinelAI | carpetcrawler78/vigilex | cap | This project (Capstone II) |
| tradetest | bobschmid/tradetest | trader | Automated trading bot (private, separate) |

Both share Hetzner CX33 (46.225.109.99). Do not mix deployments.

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
