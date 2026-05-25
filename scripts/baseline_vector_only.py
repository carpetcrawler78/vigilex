"""
baseline_vector_only.py -- Status-quo baseline: vector-arm only (PubMedBERT-base).

Uses the identical EmbeddingModel and vector search SQL as the production
hybrid_search.py -- no BM25, no CrossEncoder, no RRF fusion.

Metrics written to data/eval/status_quo_baseline.json.

Usage on Hetzner:
    cd ~/vigilex
    source .env
    export PYTHONPATH=src
    python scripts/baseline_vector_only.py
"""

import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

try:
    import psycopg2
except ImportError:
    sys.exit("psycopg2-binary not installed.")

from vigilex.coding.hybrid_search import EmbeddingModel

TOP_K     = 100
EVAL_PATH = ROOT / "data/eval/golden_set_v1.jsonl"
OUT_PATH  = ROOT / "data/eval/status_quo_baseline.json"


def vector_search(conn, embedding: list, top_k: int) -> list:
    """Return pt_codes ordered by cosine similarity, best first."""
    sql = """
        SELECT pt_code
        FROM processed.meddra_terms
        WHERE pt_embedding IS NOT NULL
        ORDER BY pt_embedding <=> %(emb)s::vector
        LIMIT %(k)s
    """
    with conn.cursor() as cur:
        cur.execute("SET ivfflat.probes = 100")
        cur.execute(sql, {"emb": str(embedding), "k": top_k})
        return [row[0] for row in cur.fetchall()]


def find_rank(expected_code: int, ranked_codes: list) -> Optional[int]:
    """1-based rank of expected_code in ranked_codes, None if not present."""
    for i, code in enumerate(ranked_codes, 1):
        if code == expected_code:
            return i
    return None


def main():
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        sys.exit("DATABASE_URL not set. Run: source .env")

    if not EVAL_PATH.exists():
        sys.exit(f"Golden set not found: {EVAL_PATH}")

    with open(EVAL_PATH, encoding="utf-8") as f:
        cases = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(cases)} cases from {EVAL_PATH.name}")

    print("Loading EmbeddingModel (PubMedBERT)...")
    t0 = time.time()
    model = EmbeddingModel()
    print(f"  ready in {time.time() - t0:.1f}s")

    conn = psycopg2.connect(db_url)
    ranks = []
    t_start = time.time()

    for i, case in enumerate(cases):
        expected_code = int(case["expected_pt_code"])
        mdr_text      = case["mdr_text"]

        # Identical first-sentence extraction as production hybrid_search.py
        first_sentence = mdr_text.split(".")[0].strip()
        query_text     = first_sentence if first_sentence else mdr_text

        embedding    = model.encode(query_text)
        result_codes = vector_search(conn, embedding, TOP_K)
        rank         = find_rank(expected_code, result_codes)
        ranks.append(rank)

        status = f"rank={rank}" if rank is not None else "NOT FOUND"
        print(
            f"  [{i+1:2d}/{len(cases)}] {case['mdr_report_key']} | "
            f"{status:<12} | {case['expected_pt_name']}"
        )

    conn.close()

    n          = len(cases)
    found      = [r for r in ranks if r is not None]
    not_found  = ranks.count(None)

    def recall_at_k(k: int) -> float:
        return round(sum(1 for r in ranks if r is not None and r <= k) / n, 4)

    median_rank = round(statistics.median(found), 1) if found else None

    result = {
        "model":             "PubMedBERT-base (Produktion)",
        "recall_at_5":       recall_at_k(5),
        "recall_at_20":      recall_at_k(20),
        "recall_at_100":     recall_at_k(100),
        "median_rank_found": median_rank,
        "not_found_count":   not_found,
        "n":                 n,
        "measured_at":       "2026-05-25",
        "method":            "vector-only, scripts/baseline_vector_only.py",
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    elapsed = round(time.time() - t_start, 1)
    print(f"\n=== Status-quo Baseline (vector-only, top_k={TOP_K}) ===")
    for k, v in result.items():
        print(f"  {k:<25} {v}")
    print(f"  elapsed_seconds:          {elapsed}s")
    print(f"\nWritten to: {OUT_PATH}")


if __name__ == "__main__":
    main()
