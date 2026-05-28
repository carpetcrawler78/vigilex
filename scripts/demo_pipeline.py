"""
demo_pipeline.py -- Transparent single-record pipeline inspector.

Runs the full 3-stage MedDRA coding pipeline on arbitrary input text
and prints each stage's output. No DB writes. No worker process.

Usage:
    python scripts/demo_pipeline.py --text "Patient experienced hypoglycaemia..."
    python scripts/demo_pipeline.py --text "..." --product-code LZG
    python scripts/demo_pipeline.py --demo
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path

# Load .env before any vigilex imports that read env vars
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass  # dotenv optional; env vars may already be set

import psycopg2
from vigilex.coding.hybrid_search import EmbeddingModel, HybridSearcher
from vigilex.coding.reranker import CrossEncoderReranker
from vigilex.coding.llm_coder import LLMCoder


# ---------------------------------------------------------------------------
# Demo cases -- used by --demo and --live (fallback keys)
# Source: LZG records fc>=0.97, Diabetic ketoacidosis, confirmed 2026-05-27
# ---------------------------------------------------------------------------

DEMO_KEYS = [
    "3004464228-2024-22834",
    "3004464228-2024-30992",
    "3004464228-2024-17250",
]

DEMO_CASES = [
    # 3004464228-2024-22834 | LZG | fc=0.976 | Diabetic ketoacidosis
    (
        "LZG",
        "IT WAS REPORTED THAT THE PATIENT HAD BEEN HOSPITALIZED WITH DIABETIC "
        "KETOACIDOSIS (DKA). THE PATIENT'S BLOOD GLUCOSE (BG) LEVELS READ HIGH "
        "(> 27.8 MMOL/L) (> 500 MG/DL) WHILE WEARING THE POD ON THE ABDOMEN FOR "
        "BETWEEN 24 AND 36 HOURS. SYMPTOMS REPORTED INCLUDE HYPERGLYCEMIA, VOMITING "
        "AND NAUSEA. LAB WORK WAS PERFORMED AND THE PATIENT WAS TREATED WITH INSULIN "
        "AND FLUIDS INTRAVENOUSLY. THE PATIENT TRANSITIONED TO INSULIN INJECTIONS "
        "DURING THE VISIT. THE PATIENT WAS RELEASED AFTER SEVEN DAYS. THE POD WAS "
        "DISCARDED.",
    ),
    # 3004464228-2024-30992 | LZG | fc=0.971 | Diabetic ketoacidosis
    (
        "LZG",
        "IT WAS REPORTED THAT THE PATIENT HAD BEEN HOSPITALIZED WITH DIABETIC "
        "KETOACIDOSIS (DKA). THE PATIENT'S BLOOD GLUCOSE LEVELS ROSE ABOVE 13.9 "
        "MMOL/L (>250 MG/DL) WHILE WEARING THE POD ON THE ABDOMEN FOR BETWEEN 25 "
        "AND 36 HOURS. THE PATIENT WAS VOMITING AND WENT TO THE HOSPITAL FOR "
        "TREATMENT. THE PATIENT WAS PLACED ON AN INSULIN DRIP AND WAS DISCHARGED "
        "FROM THE HOSPITAL AFTER 24 HOURS.",
    ),
    # 3004464228-2024-17250 | LZG | fc=0.970 | Diabetic ketoacidosis
    (
        "LZG",
        "IT WAS REPORTED THAT THE PATIENT HAD BEEN HOSPITALIZED WITH DIABETIC "
        "KETOACIDOSIS (DKA). THE PATIENT'S BLOOD GLUCOSE LEVELS ROSE ABOVE 250 "
        "MG/DL WHILE WEARING THE POD. THE PATIENT HAD A STOMACH VIRUS CAUSING "
        "DIARRHEA AND VOMITING. THE PATIENT WAS TREATED WITH FLUIDS AND INSULIN "
        "INTRAVENOUSLY. THE POD WAS DISCARDED AND THE PATIENT WAS RELEASED FROM "
        "THE HOSPITAL AFTER APPROXIMATELY THREE HOURS. THE PATIENT WAS ADVISED BY "
        "THE DOCTOR TO RETURN TO USING MULTIPLE DAILY INJECTIONS FOR INSULIN THERAPY.",
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fetch_cases_from_db(conn, keys: list) -> list:
    """Fetch (product_code, mdr_text) pairs from DB for given mdr_report_keys."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT mdr_report_key, product_code, mdr_text "
            "FROM raw.maude_reports "
            "WHERE mdr_report_key = ANY(%s)",
            (keys,)
        )
        rows = {r[0]: (r[1], r[2]) for r in cur.fetchall()}
    result = []
    for key in keys:
        if key not in rows:
            print(f"  WARNING: mdr_report_key {key} not found in DB -- skipping")
            continue
        product_code, mdr_text = rows[key]
        if not mdr_text:
            print(f"  WARNING: mdr_text is NULL for {key} -- skipping")
            continue
        result.append((product_code, mdr_text, key))
    return result


def fetch_random_uncoded(conn, n: int) -> list:
    """
    Fetch N random records from raw.maude_reports that have not yet been
    processed by the coding pipeline (no entry in processed.coding_results).

    Filters:
      - mdr_text IS NOT NULL
      - LENGTH(mdr_text) > 100   (skip near-empty boilerplate)
      - NOT IN coding_results    (truly unseen by pipeline)

    Returns list of (product_code, mdr_text, mdr_report_key) tuples.
    """
    sql = """
        SELECT
            r.mdr_report_key,
            r.product_code,
            r.mdr_text
        FROM raw.maude_reports r
        WHERE r.mdr_text IS NOT NULL
          AND LENGTH(r.mdr_text) > 100
          AND r.mdr_report_key NOT IN (
              SELECT mdr_report_key FROM processed.coding_results
          )
        ORDER BY RANDOM()
        LIMIT %s
    """
    with conn.cursor() as cur:
        cur.execute(sql, (n,))
        rows = cur.fetchall()

    if not rows:
        print("  WARNING: no uncoded records found in raw.maude_reports")
        return []

    result = []
    for mdr_report_key, product_code, mdr_text in rows:
        result.append((product_code or "???", mdr_text, mdr_report_key))

    print(f"  Fetched {len(result)} random uncoded records from DB")
    return result


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def compute_fc(ce_score: float, llm_confidence) -> float:
    """Replicate workers/coding.py final_confidence formula."""
    llm = llm_confidence if llm_confidence is not None else 0.0
    return 0.3 * sigmoid(ce_score) + 0.7 * llm


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def run_pipeline(text, conn, embedding_model, reranker, llm_coder,
                 top_k_stage1=20, skip_llm=False):
    t0 = time.time()

    searcher = HybridSearcher(conn, embedding_model=embedding_model)

    t1s = time.time()
    stage1 = searcher.search(text, top_k=top_k_stage1)
    t1e = time.time()

    t2s = time.time()
    stage2 = reranker.rerank(text, stage1, top_k=5)
    t2e = time.time()

    if skip_llm or llm_coder is None:
        result = None
        t3e = t3s = time.time()
    else:
        t3s = time.time()
        result = llm_coder.code(text, stage2)
        t3e = time.time()

    return {
        "stage1":    stage1,
        "stage2":    stage2,
        "result":    result,
        "t1":        t1e - t1s,
        "t2":        t2e - t2s,
        "t3":        t3e - t3s,
        "elapsed":   time.time() - t0,
        "skip_llm":  skip_llm,
    }


# ---------------------------------------------------------------------------
# Output formatter
# ---------------------------------------------------------------------------

def print_result(text, out, case_num=None, key=None):
    SEP = "-" * 72

    print(f"\n{'=' * 72}")
    if case_num is not None:
        label = f"  CASE {case_num}"
        if key:
            label += f"  [{key}]"
        print(label)
    print(f"INPUT:   {text[:200]}")
    print(SEP)

    # Stage 1
    s1 = out["stage1"]
    print(f"STAGE 1  (Hybrid Search, top_k={len(s1)})   [{out['t1']:.1f}s]")
    for i, r in enumerate(s1[:10], 1):
        bm25 = f"bm25={r.trgm_sim:.3f}" if r.trgm_sim is not None else "bm25=  n/a"
        vec  = f"vec={r.cosine_sim:.3f}"  if r.cosine_sim is not None else "vec=  n/a"
        print(f"  {i:2d}. {r.pt_name:<40s}  rrf={r.rrf_score:.4f}  {bm25}  {vec}")
    if len(s1) > 10:
        print(f"       ... ({len(s1) - 10} more not shown)")
    print(SEP)

    # Stage 2
    s2 = out["stage2"]
    print(f"STAGE 2  (CrossEncoder rerank)   [{out['t2']:.1f}s]")
    for i, r in enumerate(s2, 1):
        delta = r.rrf_rank - i
        arrow = f"(+{delta})" if delta > 0 else (f"({delta})" if delta < 0 else "(=)")
        print(f"  {i}. {r.pt_name:<40s}  ce={r.crossencoder_score:+.3f}  was rank {r.rrf_rank} {arrow}")
    print(SEP)

    # Stage 3
    if out.get("skip_llm") or out["result"] is None:
        print(f"STAGE 3  (LLM: skipped)")
    else:
        res = out["result"]
        fallback = res.fallback_reason or "No"
        backend  = res.llm_backend or "ollama"
        model_lbl = "groq" if "groq" in backend else "llama3.2:3b"
        print(f"STAGE 3  (LLM: {model_lbl})   [{out['t3']:.1f}s]")
        print(f"  selected:  {res.pt_name} ({res.pt_code})")
        conf_str = str(res.confidence) if res.confidence is not None else "n/a (fallback)"
        print(f"  ordinal:   {conf_str}")
        print(f"  rationale: \"{(res.rationale or '')[:200]}\"")
    print(SEP)

    # Final
    if not out.get("skip_llm") and out["result"] is not None:
        res = out["result"]
        ce_top = s2[0].crossencoder_score if s2 else 0.0
        fc = compute_fc(ce_top, res.confidence)
        print("FINAL:")
        print(f"  PT       = {res.pt_name} ({res.pt_code})")
        print(f"  SOC      = {res.soc_name}")
        print(f"  fc       = {fc:.3f}")
        print(f"  fallback = {res.fallback_reason or 'No'}")
        print(f"  elapsed  = {out['elapsed']:.1f}s")
    else:
        print(f"FINAL:  Stage 2 top pick: {s2[0].pt_name if s2 else 'n/a'}")
        print(f"  elapsed  = {out['elapsed']:.1f}s")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="SentinelAI -- transparent pipeline inspector (no DB write)"
    )
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--text", type=str, help="Narrative text to code")
    grp.add_argument("--demo", action="store_true",
                     help="Run 3 hardcoded demo cases (texts embedded in script)")
    grp.add_argument("--live", nargs="*", metavar="MDR_KEY",
                     help="Fetch text from DB by mdr_report_key. "
                          "Pass one or more keys, or no keys to use the 3 default demo keys.")
    grp.add_argument(
        "--random",
        type=int,
        metavar="N",
        help=(
            "Fetch N random records from raw.maude_reports that are NOT yet "
            "in processed.coding_results. Runs full pipeline on each."
        ),
    )
    parser.add_argument("--product-code", type=str, default="QFG",
                        help="Product code label (display only, default: QFG)")
    parser.add_argument("--top-k", type=int, default=20,
                        help="Stage 1 top_k (default: 20)")
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip Stage 3 (Ollama LLM). Only Stage 1 + Stage 2 run. Faster.",
    )
    args = parser.parse_args()

    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        sys.exit("DATABASE_URL not set in environment / .env")

    ollama_url = os.environ.get("OLLAMA_BASE_URL")
    if not ollama_url and not args.skip_llm:
        sys.exit("OLLAMA_BASE_URL not set. Use --skip-llm to run without Ollama.")

    print("Loading EmbeddingModel (all-mpnet-base-v2)...")
    t0 = time.time()
    em = EmbeddingModel()
    print(f"  ready ({time.time()-t0:.1f}s)")

    print("Loading CrossEncoderReranker...")
    t0 = time.time()
    reranker = CrossEncoderReranker()
    print(f"  ready ({time.time()-t0:.1f}s)")

    if args.skip_llm:
        llm_coder = None
        print("LLM stage skipped (--skip-llm)")
    else:
        print("Connecting to LLM (Ollama)...")
        t0 = time.time()
        llm_coder = LLMCoder(ollama_url=ollama_url)
        print(f"  ready ({time.time()-t0:.1f}s)")

    conn = psycopg2.connect(db_url)

    if args.demo:
        cases = [(*c, None) for c in DEMO_CASES]
    elif args.live is not None:
        keys = args.live if args.live else DEMO_KEYS
        fetched = fetch_cases_from_db(conn, keys)
        if not fetched:
            conn.close()
            sys.exit("No records found in DB for the given keys.")
        cases = fetched
    elif args.random is not None:
        cases = fetch_random_uncoded(conn, args.random)
        if not cases:
            conn.close()
            sys.exit("No uncoded records available.")
    else:
        cases = [(args.product_code, args.text, None)]

    outputs = []
    for i, item in enumerate(cases, 1):
        product_code, text = item[0], item[1]
        key = item[2] if len(item) > 2 else None
        label = key or f"case {i}"
        print(f"\nRunning {label} ({i}/{len(cases)})...")
        out = run_pipeline(text, conn, em, reranker, llm_coder,
                           top_k_stage1=args.top_k,
                           skip_llm=args.skip_llm)
        outputs.append(out)
        show_num = i if (args.demo or args.live is not None or args.random is not None) else None
        print_result(text, out, case_num=show_num, key=key)

    conn.close()

    if args.demo or args.live is not None or args.random is not None:
        print(f"\n{'=' * 72}")
        print(f"SUMMARY  ({len(outputs)} cases)")
        avg_e = sum(o["elapsed"] for o in outputs) / len(outputs)
        print(f"  avg elapsed: {avg_e:.1f}s")
        if not args.skip_llm:
            n_fb = sum(1 for o in outputs if o["result"] and o["result"].fallback_reason)
            avg_fc = sum(
                compute_fc(
                    o["stage2"][0].crossencoder_score if o["stage2"] else 0.0,
                    o["result"].confidence,
                )
                for o in outputs
                if o["result"]
            ) / len(outputs)
            print(f"  fallback:    {n_fb}/{len(outputs)}")
            print(f"  avg fc:      {avg_fc:.3f}")
        print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
