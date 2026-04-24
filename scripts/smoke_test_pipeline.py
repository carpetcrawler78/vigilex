"""
smoke_test_pipeline.py -- Quick end-to-end test of the MedDRA coding pipeline.

Tests all three stages with a single narrative:
  Stage 1: HybridSearcher  (BM25 + Vector + RRF)
  Stage 2: CrossEncoder    (MiniLM-L-6 reranker)
  Stage 3: LLM Coder       (Ollama llama3.2)

Prerequisites:
  - SSH tunnels open:
      ssh -L 5432:localhost:5432 cap@46.225.109.99
      ssh -L 11434:localhost:11434 cap@46.225.109.99
  - .env file in vigilex root with POSTGRES_* credentials
  - pip install sentence-transformers psycopg2-binary python-dotenv torch transformers

Usage (from vigilex root):
  python scripts/smoke_test_pipeline.py
  python scripts/smoke_test_pipeline.py --skip-llm   # skip Ollama stage
  python scripts/smoke_test_pipeline.py --verbose     # show full outputs
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Allow imports from src/
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import psycopg2

PASS = "[PASS]"
FAIL = "[FAIL]"
SKIP = "[SKIP]"
INFO = "[INFO]"

def get_db_url():
    url = os.getenv("DATABASE_URL")
    if url:
        return url
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db   = os.getenv("POSTGRES_DB",   "vigilex")
    user = os.getenv("POSTGRES_USER", "vigilex")
    pw   = os.getenv("POSTGRES_PASSWORD", "")
    return f"postgresql://{user}:{pw}@{host}:{port}/{db}"

TEST_NARRATIVE = (
    "Patient experienced hypoglycaemia after insulin pump delivered unexpected bolus. "
    "Blood glucose dropped to 2.1 mmol/L. Patient became confused and required assistance."
)

EXPECTED_PT_FRAGMENT = "hypoglycaem"   # case-insensitive substring expected in top result


def check(label, condition, detail=""):
    status = PASS if condition else FAIL
    print(f"  {status}  {label}")
    if detail:
        print(f"         {detail}")
    if not condition:
        sys.exit(f"\nSmoke test failed at: {label}. Aborting.")


def separator(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def test_db(verbose):
    separator("Stage 0: Database connection")
    t0 = time.time()
    try:
        conn = psycopg2.connect(get_db_url(), connect_timeout=5)
        ms = (time.time() - t0) * 1000
        print(f"  {PASS}  Connected in {ms:.0f}ms")
    except Exception as e:
        print(f"  {FAIL}  {e}")
        print("  Hint: is the SSH tunnel open?  ssh -L 5432:localhost:5432 cap@46.225.109.99")
        sys.exit(1)

    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM processed.meddra_terms WHERE pt_embedding IS NOT NULL")
        n_pts = cur.fetchone()[0]
        check("processed.meddra_terms has embeddings", n_pts > 27000,
              f"{n_pts:,} rows with pt_embedding")

        cur.execute("SELECT COUNT(*) FROM processed.meddra_llt")
        n_llts = cur.fetchone()[0]
        check("processed.meddra_llt populated", n_llts > 50000,
              f"{n_llts:,} LLT rows")

    return conn


def test_hybrid(conn, verbose):
    separator("Stage 1: Hybrid Search (BM25 + Vector + RRF)")

    from vigilex.coding.hybrid_search import HybridSearcher, EmbeddingModel

    print(f"  {INFO}  Loading PubMedBERT (may take ~10s first time)...")
    t0 = time.time()
    model = EmbeddingModel()
    print(f"  {INFO}  Model loaded on: {model.device} ({time.time()-t0:.1f}s)")

    searcher = HybridSearcher(conn, embedding_model=model, candidate_pool=100)

    t0 = time.time()
    results = searcher.search(TEST_NARRATIVE, top_k=10)
    ms = (time.time() - t0) * 1000

    check("Returns 10 results", len(results) == 10, f"got {len(results)}")
    check("All results have pt_code", all(r.pt_code for r in results))
    check("All results have rrf_score", all(r.rrf_score > 0 for r in results))

    top = results[0]
    # Always show top 10 for diagnosis
    print(f"\n  Top 10 hybrid results:")
    for i, r in enumerate(results[:10], 1):
        bm25 = f"bm25={r.bm25_rank}" if r.bm25_rank else "bm25=--"
        vec  = f"vec={r.vector_rank}" if r.vector_rank else "vec=--"
        print(f"    {i}. {r.pt_name:<50} {bm25:<10} {vec}")

    # Check if expected PT appears anywhere in top 10
    top10_names = [r.pt_name.lower() for r in results[:10]]
    found_in_top10 = any(EXPECTED_PT_FRAGMENT.lower() in n for n in top10_names)
    found_in_top1  = EXPECTED_PT_FRAGMENT.lower() in top.pt_name.lower()

    check(
        f"Expected PT in top 10",
        found_in_top10,
        f"Searched for '{EXPECTED_PT_FRAGMENT}' in top 10 PT names"
    )
    if not found_in_top1:
        print(f"  {INFO}  Note: expected PT not at rank 1 (rank 1: {top.pt_name})")

    print(f"  {INFO}  Query time: {ms:.0f}ms")

    return searcher, model, results


def test_reranker(results, verbose):
    separator("Stage 2: CrossEncoder Reranker (MiniLM-L-6)")

    from vigilex.coding.reranker import CrossEncoderReranker

    print(f"  {INFO}  Loading CrossEncoder...")
    t0 = time.time()
    reranker = CrossEncoderReranker()
    print(f"  {INFO}  Loaded in {time.time()-t0:.1f}s")

    t0 = time.time()
    reranked = reranker.rerank(TEST_NARRATIVE, results, top_k=5)
    ms = (time.time() - t0) * 1000

    check("Returns 5 reranked results", len(reranked) == 5, f"got {len(reranked)}")
    check("All results have crossencoder_score", all(hasattr(r, "crossencoder_score") for r in reranked))
    check("Scores are sorted descending",
          all(reranked[i].crossencoder_score >= reranked[i+1].crossencoder_score
              for i in range(len(reranked)-1)))

    top = reranked[0]
    check(
        "Top reranked result is clinically relevant",
        EXPECTED_PT_FRAGMENT.lower() in top.pt_name.lower(),
        f"Top PT: {top.pt_name} (CE score: {top.crossencoder_score:.3f})"
    )
    print(f"  {INFO}  Reranker time: {ms:.0f}ms")

    if verbose:
        print(f"\n  Top 5 reranked results:")
        for i, r in enumerate(reranked, 1):
            print(f"    {i}. {r.pt_name:<45} CE={r.crossencoder_score:.3f}  was_rank={r.rrf_rank}")

    return reranker, reranked


def test_llm(reranked, verbose):
    separator("Stage 3: LLM Coder (Ollama llama3.2)")

    from vigilex.coding.llm_coder import LLMCoder

    ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    print(f"  {INFO}  Ollama URL: {ollama_url}")
    print(f"  {INFO}  Sending request (may take 5-20s on CPU)...")

    coder = LLMCoder(ollama_url=ollama_url, confidence_threshold=0.5)

    t0 = time.time()
    try:
        result = coder.code(TEST_NARRATIVE, reranked)
        ms = (time.time() - t0) * 1000
    except Exception as e:
        print(f"  {FAIL}  LLM call failed: {e}")
        print("  Hint: is the SSH tunnel open?  ssh -L 11434:localhost:11434 cap@46.225.109.99")
        sys.exit(1)

    check("Result has pt_code", isinstance(result.pt_code, int), f"pt_code={result.pt_code}")
    check("Result has pt_name", bool(result.pt_name), f"pt_name={result.pt_name}")
    check("Confidence in [0, 1]", 0.0 <= result.confidence <= 1.0,
          f"confidence={result.confidence}")
    check("Result has rationale", bool(result.rationale))
    check(
        "LLM selected a relevant PT",
        EXPECTED_PT_FRAGMENT.lower() in result.pt_name.lower(),
        f"PT: {result.pt_name}  confidence={result.confidence:.2f}"
    )
    print(f"  {INFO}  LLM time: {ms:.0f}ms")

    if verbose:
        print(f"\n  LLM raw response:\n  {result.raw_response[:300]}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Smoke test for SentinelAI coding pipeline")
    parser.add_argument("--skip-llm",  action="store_true", help="Skip Ollama stage (Stage 3)")
    parser.add_argument("--verbose",   action="store_true", help="Show detailed outputs")
    args = parser.parse_args()

    print("\nSentinelAI -- Coding Pipeline Smoke Test")
    print(f"Narrative: \"{TEST_NARRATIVE[:80]}...\"")

    t_start = time.time()

    conn                      = test_db(args.verbose)
    searcher, model, results  = test_hybrid(conn, args.verbose)
    reranker, reranked        = test_reranker(results, args.verbose)

    if args.skip_llm:
        separator("Stage 3: LLM Coder")
        print(f"  {SKIP}  --skip-llm flag set")
    else:
        test_llm(reranked, args.verbose)

    elapsed = time.time() - t_start
    separator("Result")
    print(f"  All checks passed in {elapsed:.1f}s")
    if args.skip_llm:
        print("  (LLM stage skipped)")
    print()


if __name__ == "__main__":
    main()
