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
# Hardcoded demo cases (--demo mode)
# Realistic MAUDE-style narratives for QFG (CGM / insulin pump) devices.
# ---------------------------------------------------------------------------

DEMO_CASES = [
    (
        "QFG",
        "Patient with type 1 diabetes using an insulin pump developed diabetic "
        "ketoacidosis. Blood glucose was 480 mg/dL, blood ketones 5.2 mmol/L, "
        "pH 7.12. The insulin pump had delivered insufficient basal insulin over "
        "approximately 8 hours due to an occlusion alarm that was not addressed.",
    ),
    (
        "QFG",
        "Patient experienced a severe hypoglycemic episode with blood glucose of "
        "34 mg/dL. The patient was found unresponsive and required glucagon injection "
        "by emergency services. The insulin pump had delivered an unintended bolus "
        "during the night.",
    ),
    (
        "QFG",
        "Patient developed erythema, swelling and papules at the CGM sensor insertion "
        "site approximately 48 hours after placement. The skin reaction was consistent "
        "with allergic contact dermatitis to the adhesive. Sensor was removed and site "
        "treated topically.",
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
                 top_k_stage1=20):
    t0 = time.time()

    searcher = HybridSearcher(conn, embedding_model=embedding_model)

    t1s = time.time()
    stage1 = searcher.search(text, top_k=top_k_stage1)
    t1e = time.time()

    t2s = time.time()
    stage2 = reranker.rerank(text, stage1, top_k=5)
    t2e = time.time()

    t3s = time.time()
    result = llm_coder.code(text, stage2)
    t3e = time.time()

    return {
        "stage1":   stage1,
        "stage2":   stage2,
        "result":   result,
        "t1":       t1e - t1s,
        "t2":       t2e - t2s,
        "t3":       t3e - t3s,
        "elapsed":  time.time() - t0,
    }


# ---------------------------------------------------------------------------
# Output formatter
# ---------------------------------------------------------------------------

def print_result(text, out, case_num=None):
    SEP = "-" * 72

    print(f"\n{'=' * 72}")
    if case_num is not None:
        print(f"  CASE {case_num}")
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
    ce_top = s2[0].crossencoder_score if s2 else 0.0
    fc = compute_fc(ce_top, res.confidence)
    print("FINAL:")
    print(f"  PT       = {res.pt_name} ({res.pt_code})")
    print(f"  SOC      = {res.soc_name}")
    print(f"  fc       = {fc:.3f}")
    print(f"  fallback = {fallback}")
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
                     help="Run 3 hardcoded demo cases")
    parser.add_argument("--product-code", type=str, default="QFG",
                        help="Product code label (display only, default: QFG)")
    parser.add_argument("--top-k", type=int, default=20,
                        help="Stage 1 top_k (default: 20)")
    args = parser.parse_args()

    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        sys.exit("DATABASE_URL not set in environment / .env")

    ollama_url = os.environ.get("OLLAMA_BASE_URL")
    if not ollama_url:
        sys.exit("OLLAMA_BASE_URL not set in environment / .env")

    print("Loading EmbeddingModel (all-mpnet-base-v2)...")
    t0 = time.time()
    em = EmbeddingModel()
    print(f"  ready ({time.time()-t0:.1f}s)")

    print("Loading CrossEncoderReranker...")
    t0 = time.time()
    reranker = CrossEncoderReranker()
    print(f"  ready ({time.time()-t0:.1f}s)")

    print("Connecting to LLM (Ollama)...")
    t0 = time.time()
    llm_coder = LLMCoder(ollama_url=ollama_url)
    print(f"  ready ({time.time()-t0:.1f}s)")

    conn = psycopg2.connect(db_url)

    cases = DEMO_CASES if args.demo else [(args.product_code, args.text)]

    outputs = []
    for i, (product_code, text) in enumerate(cases, 1):
        print(f"\nRunning case {i}/{len(cases)}...")
        out = run_pipeline(text, conn, em, reranker, llm_coder,
                           top_k_stage1=args.top_k)
        outputs.append(out)
        print_result(text, out, case_num=i if args.demo else None)

    conn.close()

    if args.demo:
        print(f"\n{'=' * 72}")
        print(f"SUMMARY  ({len(outputs)} cases)")
        n_fb  = sum(1 for o in outputs if o["result"].fallback_reason)
        avg_e = sum(o["elapsed"] for o in outputs) / len(outputs)
        avg_fc = sum(
            compute_fc(
                o["stage2"][0].crossencoder_score if o["stage2"] else 0.0,
                o["result"].confidence
            )
            for o in outputs
        ) / len(outputs)
        print(f"  fallback:    {n_fb}/{len(outputs)}")
        print(f"  avg fc:      {avg_fc:.3f}")
        print(f"  avg elapsed: {avg_e:.1f}s")
        print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
