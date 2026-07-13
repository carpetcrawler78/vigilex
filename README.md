# SentinelAI Backend

> Status: Completed capstone / portfolio project.
> The original Hetzner deployment is no longer running (deliberately paused,
> not a bug -- kept offline to avoid ongoing hosting cost) -- this repository
> documents the backend architecture, the coding pipeline, and the evaluation
> results of a working system.

---

## Why this project exists

Regulated safety workflows frequently require adverse-event narratives to be
mapped to standardized medical terminology such as MedDRA. Manual coding is
time-consuming and shows inter-coder variability. SentinelAI explores
AI-assisted MedDRA coding using public MAUDE narratives as the evaluation
corpus. SentinelAI was built as Capstone II of the neue fische AI Engineering
Bootcamp 2025/2026 to test whether a retrieval + reranking + local-LLM
pipeline can narrow that search space and produce consistent, auditable
suggestions -- without sending patient-adjacent data to an external API.

The architecture is built around three EU regulations: GDPR (no patient
data outside EU), EU AI Act (Art. 14: proportionate human oversight for
high-risk systems), EU MDR (post-market surveillance + reproducible records).

## What the system does

- **Narrows the choice** -- from ~27k MedDRA PTs to a ranked top-5 with a
  written rationale per candidate
- **Same method every time** -- removes day-to-day variation between coders
- **Reviewer stays in charge** -- the system suggests, the human decides;
  every accept/correct/reject decision is persisted for regulatory records
- **Downstream demo** -- PRR/ROR disproportionality analysis on the coded
  data, visualized in Grafana (signal detection is a demo of the coded
  output, not a second product scope)

## Architecture

Three-stage local pipeline, turning a free-text MAUDE narrative into ranked
MedDRA Preferred Term suggestions:

```
27k MedDRA terms                MAUDE narrative (free text)
        |                                |
        |                                v
        |               Stage 1 -- RETRIEVE  (~0.5 sec)
        |               all-mpnet-base-v2 bi-encoder via pgvector (IVFFlat)
        |               +  pg_trgm trigram on pt_name + 88k LLT synonyms
        |               -> RRF fusion (w_lex=0.4, w_vec=0.6, k=60)
        |                                v
        |                            Top-20
        |                                v
        |               Stage 2 -- RERANK
        |               ELECTRA-base cross-encoder, joint attention on
        |               (narrative, pt_name)
        |                                v
        |                             Top-5
        |                                v
        |               Stage 3 -- SUGGEST
        |               qwen2.5:7b via Ollama, local (on-prem)
        |               -> JSON: pt_code, ordinal_rating, rationale
        |                                v
        |                ranking_index = weighted(CE score, LLM ordinal rating)
        |                                v
        |                  Reviewer UI -- accept / correct / reject
        |                                v
        +--------> processed.coding_results (audit-grade row per case)
```

Note: `all-mpnet-base-v2` (Stage 1), `ELECTRA-base` (Stage 2) and
`qwen2.5:7b` (Stage 3) are the current winning models after the Phase 4 /
evaluation runs; earlier baselines (PubMedBERT, MiniLM cross-encoder,
llama3.2:3b) are documented in `DEVLOG.md` and `EVAL_PLAN.md` for
comparison. `ranking_index` is a **heuristic ordinal value**, not a
calibrated probability -- the reviewer UI uses it as a triage signal
(higher = earlier in the review queue), never as "X% correct".

### Why hybrid retrieval?
BM25/trigram catches exact MedDRA-vocabulary matches; semantic search
catches paraphrases (`low blood sugar` ~ `Hypoglycaemia`). RRF fuses the
two ranks without depending on incompatible score scales. Index: pgvector
IVFFlat in PostgreSQL (not FAISS).

### Why a cross-encoder in Stage 2?
The bi-encoder from Stage 1 compares pre-computed vectors; the
cross-encoder reads (narrative, candidate) jointly and catches negation,
primary-vs-secondary, and relation logic. Too slow for 27k candidates, fast
enough for 20.

### Why a local LLM?
External LLM APIs can introduce additional data-protection, processing-agreement
and international-transfer requirements. The project therefore uses a local
LLM runtime as a privacy-preserving architectural default. Groq-hosted models
were benchmarked for reference only and are explicitly excluded from any
production path.

## Tech stack

- **Language / API:** Python, FastAPI (`src/vigilex/api/main.py`,
  X-API-Key auth)
- **Database:** PostgreSQL + pgvector (IVFFlat) + pg_trgm
- **Orchestration:** Docker Compose (Postgres, Ollama, ingest worker, coding
  worker, API, Grafana)
- **LLM runtime:** Ollama, local models (qwen2.5:7b / llama3.2:3b baseline)
- **Experiment tracking:** MLflow
- **Dashboards:** Grafana (provisioned datasources + dashboards)
- **Testing:** pytest

## Repository status

| Component | Can be run locally today? |
|---|---|
| Test suite (pytest) | Yes |
| DB schema / migrations | Yes |
| Reduced demo dataset (no license needed) | Not yet -- planned as a separate follow-up, not part of this snapshot |
| Full MedDRA / MAUDE pipeline end-to-end | Only with your own licensed MedDRA files (`raw_data/` is gitignored, not included) |
| Hetzner production server | Offline -- deliberately paused, not reactivated for this portfolio snapshot |

## Demo artifacts

![Grafana operations dashboard](docs/screenshots/grafana_dashboard.png)

Grafana dashboard screenshot (signal-detection demo panels + pipeline
health strip), taken from the live deployment before it was paused.

A reviewer-workflow screenshot and a standalone architecture diagram are
not yet part of this repository -- open item, to be added later.

## Local setup

```bash
# Clone and install
git clone https://github.com/carpetcrawler78/sentinelai-backend.git
cd sentinelai-backend
python -m venv .venv
source .venv/bin/activate
pip install -e .

# Configure
cp .env.example .env
# Edit .env with your DATABASE_URL, OLLAMA_BASE_URL, API_KEY

# Optional: openFDA API key (raises rate limit from 1k to 120k req/day)
# Free at https://open.fda.gov/apis/authentication/

# Run tests
pytest
```

The full stack (Postgres + pgvector, Ollama, ingest worker, coding worker,
FastAPI, Grafana) is defined in `docker-compose.yml`. It previously
ran on a Hetzner CX33 instance (Nuremberg, EU) that has since been paused.

## What is included / not included

**Included:** full source for all three pipeline stages, REST API, Grafana
provisioning, database schema/migrations, evaluation scripts and frozen
results, pytest suite, development diary (`DEVLOG.md`), architecture and
decision docs.

**Not included:** MedDRA reference data (licensed, must be obtained
separately), MAUDE raw data snapshots, `.env` secrets, a running server.

## Results

Evaluation on a 24-case golden set (Stage 1+2, `top_k_stage1=20`,
`llama3.2:3b` baseline):

- recall@5 = 0.333, soft_recall@5 = 0.500
- recall@10 = 0.500, soft_recall@10 = 0.833
- p@1 = 0.083, MRR = 0.191, R@100 = 0.750

Best on-prem stack found (ELECTRA-base reranker + qwen2.5:7b):

- recall@5 = 0.375, soft_recall@5 = 0.625, p@1_llm = 0.375
- Matches a Groq-hosted reference run on p@1_llm, without any external API
  call -- i.e. on-prem parity with a cloud LLM, while staying inside the
  GDPR/EU AI Act constraints described above.

Full breakdown, category analysis (Stage-1 miss vs. cross-encoder drop),
and MLflow run names are in `DEVLOG.md` and `EVAL_PLAN.md`.

## Known limitations

- Bottleneck is Stage-1 retrieval misses (`cat_A`), unchanged across all
  evaluated configurations -- the reranker and LLM stages cannot recover a
  candidate that Stage 1 never retrieves.
- `llm_confidence` / `final_confidence` column names are historical; the
  values are heuristic ordinal ratings, not calibrated probabilities
  (planned rename in a future migration).
- Golden set is 24 cases -- enough to compare configurations, not a
  statistically powered clinical validation.
- No reduced/synthetic demo dataset shipped yet (see "Repository status").

## Author

Built by Thomas Heger -- Dr. sc. ETH Biochemistry, former Clinical Data
Manager at DKFZ Heidelberg (REQUITE, RADprecise post-market surveillance
studies). Domain expertise in GCP documentation, MDR conformity, and EU
regulatory context informs both the architecture choices and the
evaluation framing of this project.

## License

MIT
