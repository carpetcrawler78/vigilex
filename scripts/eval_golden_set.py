"""
eval_golden_set.py -- Evaluate the 3-stage MedDRA coding pipeline on golden_set_v1.jsonl.

Metrics logged to MLflow:
    recall_at_5       : expected PT in Stage 2 top-5 output
    recall_at_10      : expected PT in Stage 1 top-10 output
    p_at_1_reranker   : top-1 Stage 2 result == expected PT
    mrr               : Mean Reciprocal Rank within Stage 2 top-5
    p_at_1_llm        : LLM final choice == expected PT  (requires --stage3-model)
    n_evaluated       : records processed without error

Per-difficulty and per-device breakdowns are logged as sub-metrics.

Usage on Hetzner:
    cd ~/vigilex
    source .env
    export PYTHONPATH=src
    export MLFLOW_TRACKING_URI=http://localhost:5000

    # Stage 1+2 only (fast, ~30s for 24 records)
    python scripts/eval_golden_set.py

    # Stage 1+2+3 with llama3.2:3b
    python scripts/eval_golden_set.py --stage3-model llama3.2:3b

    # Stage 1+2+3 with qwen2.5:7b (comparison run)
    python scripts/eval_golden_set.py --stage3-model qwen2.5:7b --run-name qwen25_7b

    # Groq reference run (throughput benchmark only -- production-excluded)
    # WARNING: narratives are sent to Groq external API.
    # GDPR Art. 44 (third-country transfer) + Art. 9 (health data).
    # Do NOT run with real patient data.
    export GROQ_API_KEY=<your_key>
    python scripts/eval_golden_set.py --groq-reference --run-name groq_lm31_8b

    # Custom eval set or tracking URI
    python scripts/eval_golden_set.py \
        --eval-set data/eval/golden_set_v1.jsonl \
        --tracking-uri http://localhost:5000 \
        --experiment sentinelai_coding
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

# Make sure src/ is on the path whether script is run from repo root or scripts/
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

try:
    import mlflow
except ImportError:
    sys.exit("mlflow not installed. Run: pip install mlflow --break-system-packages")

try:
    import psycopg2
except ImportError:
    sys.exit("psycopg2-binary not installed.")

import tempfile
import pandas as pd

from vigilex.coding.hybrid_search import EmbeddingModel, HybridSearcher, RRF_K
from vigilex.coding.reranker import CrossEncoderReranker

# ---------------------------------------------------------------------------
# Optional Stage 3
# ---------------------------------------------------------------------------

def _try_import_llm_coder():
    try:
        from vigilex.coding.llm_coder import LLMCoder
        return LLMCoder
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def reciprocal_rank(expected_pt_code: int, ranked_codes: list[int]) -> float:
    """
    1 / rank if expected appears in ranked_codes, else 0.
    rank is 1-based.
    """
    for i, code in enumerate(ranked_codes, 1):
        if code == expected_pt_code:
            return 1.0 / i
    return 0.0


def compute_metrics(results: list[dict]) -> dict:
    """
    Compute aggregate metrics from per-record result dicts.

    Each result dict must have:
        expected_pt_code, stage1_top10_codes, stage2_top5_codes,
        difficulty, product_code, llm_pt_code (optional)
    """
    n = len(results)
    if n == 0:
        return {}

    def avg(values):
        return sum(values) / len(values) if values else 0.0

    recall_at_5  = avg([int(r["expected_pt_code"] in r["stage2_top5_codes"]) for r in results])
    recall_at_10 = avg([int(r["expected_pt_code"] in r["stage1_top10_codes"]) for r in results])

    def soft_hit(r, codes_field):
        """Hit if expected PT or any acceptable PT is in the candidate list."""
        acceptable = r.get("acceptable_pt_codes", set())
        codes = set(r[codes_field])
        return int(r["expected_pt_code"] in codes or bool(codes & acceptable))

    soft_recall_at_5  = avg([soft_hit(r, "stage2_top5_codes")  for r in results])
    soft_recall_at_10 = avg([soft_hit(r, "stage1_top10_codes") for r in results])
    p_at_1_re    = avg([int(r["stage2_top5_codes"][0] == r["expected_pt_code"])
                        for r in results if r["stage2_top5_codes"]])
    mrr          = avg([reciprocal_rank(r["expected_pt_code"], r["stage2_top5_codes"])
                        for r in results])

    # Category breakdown: A/B/C/hit
    by_cat = {"A": 0, "B": 0, "C": 0, "hit": 0}
    for r in results:
        by_cat[r.get("category", "A")] += 1

    metrics = {
        "recall_at_5":       round(recall_at_5,       4),
        "soft_recall_at_5":  round(soft_recall_at_5,  4),
        "recall_at_10":      round(recall_at_10,      4),
        "soft_recall_at_10": round(soft_recall_at_10, 4),
        "p_at_1_reranker":   round(p_at_1_re,         4),
        "mrr":               round(mrr,                4),
        "n_evaluated":       n,
        # A=not in stage1, B=stage1 yes but reranker dropped, C=reranker top-k not top-5
        "cat_A_stage1_miss":      by_cat["A"],
        "cat_B_reranker_dropped": by_cat["B"],
        "cat_C_reranker_low":     by_cat["C"],
        "cat_hit":                by_cat["hit"],
    }

    # P@1 for LLM stage if available
    llm_results = [r for r in results if r.get("llm_pt_code") is not None]
    if llm_results:
        p_at_1_llm = avg([int(r["llm_pt_code"] == r["expected_pt_code"]) for r in llm_results])
        metrics["p_at_1_llm"]    = round(p_at_1_llm, 4)
        metrics["n_llm_judged"]  = len(llm_results)

    # Per-difficulty breakdown
    by_diff = defaultdict(list)
    for r in results:
        by_diff[r["difficulty"]].append(r)
    for diff, recs in by_diff.items():
        key = diff.replace(" ", "_")
        metrics[f"recall_at_5_{key}"] = round(
            avg([int(r["expected_pt_code"] in r["stage2_top5_codes"]) for r in recs]), 4
        )

    # Per-device breakdown
    by_dev = defaultdict(list)
    for r in results:
        by_dev[r["product_code"]].append(r)
    for dev, recs in by_dev.items():
        metrics[f"recall_at_5_{dev}"] = round(
            avg([int(r["expected_pt_code"] in r["stage2_top5_codes"]) for r in recs]), 4
        )

    return metrics


# ---------------------------------------------------------------------------
# Main eval loop
# ---------------------------------------------------------------------------

def evaluate(args):
    # -- MLflow setup -------------------------------------------------------
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)

    # -- DB connection -------------------------------------------------------
    db_url = args.db_url or os.environ.get("DATABASE_URL")
    if not db_url:
        sys.exit("DATABASE_URL not set. Export it or pass --db-url.")
    conn = psycopg2.connect(db_url)

    # -- Load eval set -------------------------------------------------------
    eval_path = Path(args.eval_set)
    if not eval_path.exists():
        sys.exit(f"Eval set not found: {eval_path}")
    with open(eval_path, encoding="utf-8") as f:
        cases = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(cases)} eval cases from {eval_path.name}")

    # -- Load models --------------------------------------------------------
    print("Loading EmbeddingModel (all-mpnet-base-v2)...")
    t0 = time.time()
    embedding_model = EmbeddingModel()
    print(f"  EmbeddingModel ready ({time.time()-t0:.1f}s)")

    searcher = HybridSearcher(
        conn=conn,
        embedding_model=embedding_model,
        candidate_pool=args.candidate_pool,
        rrf_k=RRF_K,
    )

    print("Loading CrossEncoderReranker...")
    t0 = time.time()
    reranker = CrossEncoderReranker()
    print(f"  CrossEncoderReranker ready ({time.time()-t0:.1f}s)")

    # Optional Stage 3
    llm_coder = None
    groq_reference = False
    stage3_model_label = "none"

    if args.groq_reference:
        # Groq branch: external API, reference-only, production-excluded
        LLMCoder = _try_import_llm_coder()
        if LLMCoder is None:
            print("WARNING: Could not import LLMCoder -- skipping Groq reference run")
        else:
            groq_key = os.environ.get("GROQ_API_KEY")
            if not groq_key:
                sys.exit("GROQ_API_KEY not set. Export it before running with --groq-reference.")
            print("WARNING: Groq backend sends narratives to external API.")
            print("         This run is for reference only. Never use in production.")
            llm_coder = LLMCoder(use_groq=True, groq_api_key=groq_key)
            groq_reference = True
            stage3_model_label = "groq:llama-3.1-8b-instant"
            print("LLMCoder ready: Groq llama-3.1-8b-instant (reference, production-excluded)")

    elif args.stage3_model:
        LLMCoder = _try_import_llm_coder()
        if LLMCoder is None:
            print("WARNING: Could not import LLMCoder -- skipping Stage 3")
        else:
            ollama_url = args.ollama_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
            llm_coder = LLMCoder(model=args.stage3_model, ollama_url=ollama_url)
            stage3_model_label = args.stage3_model
            print(f"LLMCoder ready: {args.stage3_model} @ {ollama_url}")

    # -- MLflow run ---------------------------------------------------------
    if groq_reference:
        base_name = args.run_name or "groq_lm31_8b"
        run_name = base_name + "_groq_ref"
    else:
        run_name = args.run_name or (
            f"eval_{args.stage3_model.replace(':', '_')}" if args.stage3_model
            else "eval_stage1_2_only"
        )

    with mlflow.start_run(run_name=run_name):
        # Log parameters
        mlflow.log_params({
            "eval_set":         eval_path.name,
            "n_cases":          len(cases),
            "stage1_model":     "all-mpnet-base-v2",
            "stage2_model":     "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "stage3_model":     stage3_model_label,
            "rrf_w_bm25":       0.4,
            "rrf_w_vector":     0.6,
            "rrf_k":            RRF_K,
            "top_k_stage1":     args.top_k_stage1,
            "top_k_stage2":     args.top_k_stage2,
            "candidate_pool":   args.candidate_pool,
        })
        if groq_reference:
            mlflow.set_tag("backend", "groq_reference")
            mlflow.set_tag("production_eligible", "false")
            mlflow.set_tag("exclusion_reason", "GDPR_Art44_Art9_external_api")

        if run_name.startswith("topk_sweep"):
            mlflow.set_tag("experiment_group", "topk_sweep")

        # -- Eval loop ------------------------------------------------------
        results = []
        errors  = []
        t_start = time.time()
        stage1_total_s = 0.0
        stage2_total_s = 0.0
        stage3_total_s = 0.0

        for i, case in enumerate(cases):
            mdr_key     = case["mdr_report_key"]
            mdr_text    = case["mdr_text"]
            expected_code = case["expected_pt_code"]
            expected_name = case["expected_pt_name"]

            try:
                # Stage 1: Hybrid search
                _t1 = time.time()
                stage1_results = searcher.search(mdr_text, top_k=args.top_k_stage1)
                stage1_total_s += time.time() - _t1
                stage1_codes   = [r.pt_code for r in stage1_results]

                if i < 3:
                    print(f"  DEBUG stage1 top5: {[(r.pt_code, r.pt_name) for r in stage1_results[:5]]}")

                # Stage 2: CrossEncoder rerank
                _t2 = time.time()
                stage2_results = reranker.rerank(mdr_text, stage1_results, top_k=args.top_k_stage2)
                stage2_total_s += time.time() - _t2
                stage2_codes   = [r.pt_code for r in stage2_results]

                if i < 3:
                    print(f"  DEBUG top5: {[(r.pt_code, r.pt_name) for r in stage2_results]}")
                    print(f"  DEBUG expected: {expected_code!r} (type={type(expected_code).__name__})")
                    
                # Ranks: 1-based position in each stage, None if not found
                stage1_rank = next(
                    (i + 1 for i, c in enumerate(stage1_codes) if c == expected_code),
                    None
                )
                reranker_rank = next(
                    (i + 1 for i, c in enumerate(stage2_codes) if c == expected_code),
                    None
                )
                # delta: positive = reranker improved rank, negative = reranker hurt
                rank_delta = None
                if stage1_rank is not None and reranker_rank is not None:
                    rank_delta = stage1_rank - reranker_rank

                # Category for analysis:
                # A = not in stage1 at all
                # B = in stage1 but reranker did not include in top-k
                # C = in stage1 top-k but not in reranker top-5
                # D = soft hit only (acceptable PT found, expected not found)
                acceptable = set(case.get("acceptable_pt_codes", []))
                if stage1_rank is None:
                    category = "A"
                elif reranker_rank is None:
                    category = "B"
                elif reranker_rank > 5:
                    category = "C"
                else:
                    category = "hit"

                rec = {
                    "mdr_report_key":      mdr_key,
                    "expected_pt_code":    expected_code,
                    "expected_pt_name":    expected_name,
                    "stage1_top10_codes":  stage1_codes[:10],
                    "stage2_top5_codes":   stage2_codes,
                    "stage1_rank":         stage1_rank,
                    "reranker_rank":       reranker_rank,
                    "rank_delta":          rank_delta,
                    "category":            category,
                    "difficulty":          case["difficulty"],
                    "product_code":        case["product_code"],
                    "llm_pt_code":         None,
                    "acceptable_pt_codes": acceptable,
                    "report_snippet":      mdr_text[:100],
                    "stage2_top5_names":   [r.pt_name for r in stage2_results],
                    "stage2_top5_scores":  [r.crossencoder_score for r in stage2_results],
                }

                # Stage 3: LLM (optional)
                if llm_coder is not None:
                    _t3 = time.time()
                    coding_result = llm_coder.code(mdr_text, stage2_results)
                    stage3_total_s += time.time() - _t3
                    rec["llm_pt_code"] = coding_result.pt_code
                    rec["llm_pt_name"] = coding_result.pt_name

                results.append(rec)

                in_top5 = expected_code in stage2_codes
                top1_ok = stage2_codes[0] == expected_code if stage2_codes else False
                print(
                    f"  [{i+1:2d}/{len(cases)}] {mdr_key} | "
                    f"R@5={'Y' if in_top5 else 'N'} P@1={'Y' if top1_ok else 'N'} | "
                    f"expected={expected_name}"
                )

            except Exception as e:
                print(f"  ERROR [{i+1}] {mdr_key}: {e}")
                errors.append({"mdr_report_key": mdr_key, "error": str(e)})

        # -- Metrics --------------------------------------------------------
        metrics = compute_metrics(results)
        metrics["elapsed_seconds"] = round(time.time() - t_start, 1)
        metrics["stage1_total_s"]  = round(stage1_total_s, 2)
        metrics["stage2_total_s"]  = round(stage2_total_s, 2)
        metrics["stage3_total_s"]  = round(stage3_total_s, 2)
        metrics["n_errors"]        = len(errors)

        mlflow.set_tag("stage3_active", "true" if llm_coder is not None else "false")
        mlflow.log_metrics(metrics)

        # -- Case CSV artifact ----------------------------------------------
        if results:
            rows = []
            for rec in results:
                names  = rec.get("stage2_top5_names", [])
                scores = rec.get("stage2_top5_scores", [])
                row = {
                    "case_id":       rec["mdr_report_key"],
                    "snippet":       rec.get("report_snippet", ""),
                    "expected_pt":   rec["expected_pt_name"],
                    "stage1_rank":   rec["stage1_rank"],
                    "reranker_rank": rec["reranker_rank"],
                    "rank_delta":    rec["rank_delta"],
                    "category":      rec["category"],
                    "hit":           "yes" if rec["category"] == "hit" else "no",
                }
                for idx in range(5):
                    row[f"rank{idx+1}_pt"]    = names[idx]  if idx < len(names)  else ""
                    row[f"rank{idx+1}_score"]  = scores[idx] if idx < len(scores) else ""
                rows.append(row)

            csv_name = f"{run_name}_cases.csv"
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".csv", delete=False, prefix=csv_name
            ) as tmp:
                tmp_path = tmp.name
            pd.DataFrame(rows).to_csv(tmp_path, index=False)
            mlflow.log_artifact(tmp_path, artifact_path="case_details")

        # Log golden set as artifact
        mlflow.log_artifact(str(eval_path), artifact_path="eval_set")

        print(f"\n{'='*60}")
        print(f"Run: {run_name}")
        print(f"Evaluated: {len(results)}/{len(cases)} cases ({len(errors)} errors)")
        for k, v in sorted(metrics.items()):
            print(f"  {k:<30} {v}")
        print(f"MLflow UI: {args.tracking_uri}/#/experiments")
        print(f"{'='*60}")

    conn.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval-set",      default="data/eval/golden_set_v1.jsonl",
                   help="Path to JSONL eval set (default: data/eval/golden_set_v1.jsonl)")
    p.add_argument("--stage3-model",  default=None,
                   help="Ollama model tag for Stage 3, e.g. 'llama3.2:3b' or 'qwen2.5:7b'. "
                        "Omit to run Stage 1+2 only.")
    p.add_argument("--run-name",      default=None,
                   help="MLflow run name (auto-generated from stage3-model if omitted)")
    p.add_argument("--experiment",    default="sentinelai_coding",
                   help="MLflow experiment name (default: sentinelai_coding)")
    p.add_argument("--tracking-uri",  default=os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"),
                   help="MLflow tracking URI (default: $MLFLOW_TRACKING_URI or http://localhost:5000)")
    p.add_argument("--db-url",        default=None,
                   help="PostgreSQL URL (default: $DATABASE_URL)")
    p.add_argument("--groq-reference", action="store_true", default=False,
                   help="Use Groq llama-3.1-8b-instant as Stage 3 (reference only, "
                        "production-excluded: GDPR Art.44 + Art.9). "
                        "Requires GROQ_API_KEY env var.")
    p.add_argument("--ollama-url",    default=None,
                   help="Ollama base URL (default: $OLLAMA_BASE_URL or http://localhost:11434)")
    p.add_argument("--top-k-stage1",  type=int, default=20,
                   help="Top-K from Stage 1 hybrid search (default: 20)")
    p.add_argument("--top-k-stage2",  type=int, default=5,
                   help="Top-K from Stage 2 reranker (default: 5)")
    p.add_argument("--candidate-pool", type=int, default=100,
                   help="Candidate pool size for hybrid search (default: 100)")
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
