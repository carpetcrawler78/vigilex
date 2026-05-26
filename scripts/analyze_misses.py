"""
Stage-1 Miss Analysis -- analyze_misses.py
Outputs top-20 candidates, expected rank, and q1/q2 hit flags for miss cases.
Run: python3 scripts/analyze_misses.py
Requires: DATABASE_URL env var
"""
import json
import os
import sys

sys.path.insert(0, "src")

try:
    import psycopg2
except ImportError:
    sys.exit("psycopg2-binary not installed.")

from vigilex.coding.hybrid_search import HybridSearcher

# ---------------------------------------------------------------------------
# Miss case IDs from R@100 run 2026-05-26
# Update this set if golden set changes.
# ---------------------------------------------------------------------------
MISS_IDS = {
    "3004464228-2024-03085",
    "3013756811-2024-09995",
    "3013756811-2024-17732",
    "3004464228-2024-27443",
    "3004464228-2024-04816",
    "2032227-2024-227041",
    "3004464228-2024-04262",
    "3004464228-2024-29565",
}

GOLDEN_SET = "data/eval/golden_set_v1.jsonl"
OUT_FILE   = "data/eval/miss_analysis.json"
TOP_K      = 100


def main():
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        sys.exit("DATABASE_URL not set.")

    conn = psycopg2.connect(db_url)
    searcher = HybridSearcher(conn)

    cases = [json.loads(l) for l in open(GOLDEN_SET)]
    miss_cases = [c for c in cases if c["mdr_report_key"] in MISS_IDS]

    if not miss_cases:
        print("No miss cases found -- check MISS_IDS against golden set IDs.")
        return

    results_out = []

    for c in miss_cases:
        mdr_text     = c["mdr_text"]
        expected_code = c["expected_pt_code"]
        expected_name = c["expected_pt_name"]

        # Full fusion search top-100
        res   = searcher.search(mdr_text, top_k=TOP_K)
        codes = [r.pt_code for r in res]
        names = [r.pt_name for r in res]

        expected_rank = next(
            (i + 1 for i, code in enumerate(codes) if code == expected_code),
            None
        )

        # q1 only (first sentence)
        q1    = mdr_text.split(".")[0].strip() or mdr_text
        r1    = searcher.search(q1, top_k=TOP_K)
        q1_hit = expected_code in [r.pt_code for r in r1]

        # q2 only (full text truncated)
        q2    = mdr_text[:512]
        r2    = searcher.search(q2, top_k=TOP_K)
        q2_hit = expected_code in [r.pt_code for r in r2]

        row = {
            "case_id":       c["mdr_report_key"],
            "expected_pt":   expected_name,
            "expected_rank": expected_rank,
            "top1":          names[0] if names else None,
            "top5":          names[:5],
            "top20":         names[:20],
            "q1_hit":        q1_hit,
            "q2_hit":        q2_hit,
        }
        results_out.append(row)

        print(f"\n{'=' * 60}")
        print(f"CASE:          {c['mdr_report_key']}")
        print(f"EXPECTED PT:   {expected_name}  (rank={expected_rank})")
        print(f"Q1_HIT: {q1_hit}   Q2_HIT: {q2_hit}")
        print(f"TOP-5:")
        for i, n in enumerate(names[:5], 1):
            marker = " <-- EXPECTED" if codes[i - 1] == expected_code else ""
            print(f"  {i:2d}. {n}{marker}")
        print(f"TOP 6-20:")
        for i, n in enumerate(names[5:20], 6):
            marker = " <-- EXPECTED" if codes[i - 1] == expected_code else ""
            print(f"  {i:2d}. {n}{marker}")

    json.dump(results_out, open(OUT_FILE, "w"), indent=2)
    print(f"\nSaved: {OUT_FILE}")

    # Summary table
    print(f"\n{'=' * 60}")
    print(f"{'CASE_ID':<30} {'EXPECTED PT':<35} {'RANK':>5} {'Q1':>4} {'Q2':>4}")
    print("-" * 80)
    for r in results_out:
        rank_str = str(r["expected_rank"]) if r["expected_rank"] else "NONE"
        print(f"{r['case_id']:<30} {r['expected_pt']:<35} {rank_str:>5} "
              f"{'Y' if r['q1_hit'] else 'N':>4} {'Y' if r['q2_hit'] else 'N':>4}")


if __name__ == "__main__":
    main()
