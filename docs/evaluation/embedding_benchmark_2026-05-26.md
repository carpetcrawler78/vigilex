# Embedding Model Benchmark -- 2026-05-26

**Status:** Complete
**Ran on:** Hetzner CX33 instance (EU), tmux session 'bench', night 2026-05-25 -> 2026-05-26
**Script:** `scripts/bench_embedding_models.py --full`
**Related:** VECTOR_MIGRATION_PLAN.md, EXECUTION_PLAN.md Phase 2

---

## Setup

| Parameter | Value |
|---|---|
| Dataset | `data/eval/golden_set_v1.jsonl` |
| n | 24 MAUDE cases (LZG + OYC + QFG) |
| Task | Stage-1 vector retrieval: find expected MedDRA PT in top-K |
| Index | pgvector IVFFlat (PubMedBERT-base, dim=768, production) |
| Metrics | R@1, R@5, R@100, not_found |
| Evaluation | exact match on pt_code only |

---

## Results

| Model | Pool | Query | R@1 | R@5 | R@100 | not_found |
|---|---|---|---|---|---|---|
| all-mpnet-base-v2 | pt_only | first_sentence | 0.208 | 0.250 | 0.667 | 8 |
| all-mpnet-base-v2 | pt_only | full_text_truncated | 0.042 | 0.042 | 0.708 | 7 |
| all-MiniLM-L6-v2 | pt_only | first_sentence | 0.167 | 0.292 | 0.500 | 12 |
| BAAI/bge-small-en-v1.5 | pt_only | full_text_truncated | 0.083 | 0.125 | 0.667 | 8 |
| all models | pt_limited_llt | * | ~0.0 | ~0.0 | ~0.0 | 19-24 |

(Full 3x2x2 matrix = 12 configurations; all pt_limited_llt rows uniformly near zero.)

---

## Key Findings

1. **Winner (precision):** all-mpnet-base-v2 + pt_only + first_sentence
   R@1=0.208, R@5=0.250, R@100=0.667 -- best top-rank precision across all configs.

2. **Winner (recall depth):** all-mpnet-base-v2 + pt_only + full_text_truncated
   R@100=0.708 -- highest deep recall, but R@1/R@5 collapse (0.042).
   Interpretation: full text retrieves correct PT somewhere in top-100 more often,
   but drowns signal in ranking. Good for candidate generation, bad as final ranker.

3. **pt_limited_llt fails completely** -- all models near zero. Restricted LLT pool
   is not viable as a candidate pool. Do not use.

4. **first_sentence consistently better for top ranks** -- makes sense: adverse event
   description usually leads the MAUDE narrative (Insulet template: "IT WAS REPORTED...").

5. **not_found=7-12 in pt_only:** 30-50% of cases not in Top-100 with vector-only.
   Embedding retrieval alone is insufficient for production. CrossEncoder + LLM mandatory.

6. **Dangerous failure mode (threshold triggers):** high cosine score does not imply
   correct retrieval. A confidently wrong first-sentence retrieval will not self-report
   as uncertain. This rules out CE_THRESHOLD / GAP_THRESHOLD trigger-based second passes.

---

## Query Fusion Implication

first_sentence and full_text_truncated are complementary:
- first_sentence: better precision in top ranks
- full_text_truncated: better coverage at depth

Running both in parallel and unioning candidates (max ~100) before CrossEncoder reranking
is the correct architectural response. No threshold trigger needed. See DECISION_query_fusion.md.

---

## Caveats

- **n=24 is small.** Results are directionally valid but not statistically robust.
  Confidence intervals would be wide. Use as engineering guidance, not scientific proof.
- **Exact match only.** acceptable_pt_codes (clinically equivalent PTs) not yet implemented.
  True recall is likely higher than reported -- some "not found" cases may have acceptable
  alternative PTs in the candidate list.
- **Stage-1 only.** This benchmark does not measure full pipeline (CrossEncoder + LLM).
  Final coding accuracy cannot be inferred from these numbers alone.
- **Status-quo baseline confirmed 2026-05-26:**
  PubMedBERT vector-only: R@1=0.0, R@5=0.0, R@100=0.0, not_found=24/24.
  Root cause: asymmetric query mismatch -- PubMedBERT cannot bridge short PT names
  ("Hyperglycaemia") and long MAUDE narratives in the same embedding space.
  Symmetric query (PT name -> PT index) works (rank 1, distance 0.000).
  Migration delta: 0.0 -> 0.667 R@100 (mpnet, mismatched index -- dedicated index expected higher).
- **Index built with PubMedBERT.** bench_embedding_models.py queried with each model
  against the PubMedBERT index -- this is a vector space mismatch for non-PubMedBERT models.
  mpnet numbers would improve further with a dedicated mpnet index. See DECISION.md.

---

## Files

| File | Description |
|---|---|
| `data/eval/bench_results_summary.csv` | One row per (model, pool, query) config |
| `data/eval/bench_results_detailed.csv` | One row per (config, case_id) |
| `data/eval/bench_run.log` | Full run log with timestamps |
| `data/eval/cache/` | Cached doc embeddings per model+pool |
| `data/eval/status_quo_baseline.json` | PENDING -- run baseline_vector_only.py |
