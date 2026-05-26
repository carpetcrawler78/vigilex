"""
Stage-1 Miss Analysis -- analyze_misses.py
Outputs q1_rank, q2_rank, fusion_rank, acceptable_candidate_present for miss cases.

IMPORTANT: q1/q2 are tested with limit=50 (matching actual fusion in hybrid_search.py),
not limit=100. A q1_rank > 50 means it would NOT appear in the fusion candidate pool.

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

# acceptable_pt_codes: case-specific (NOT universal synonyms).
# These are clinically plausible alternative PTs for a given report,
# not blanket synonyms for the PT name.
# Format: case_id -> set of acceptable pt_codes (ints)
ACCEPTABLE = {
    # BG 600+ report -- "Blood glucose increased" plausible but not DKA territory
    "3004464228-2024-03085": {10005557},   # Blood glucose increased
    # BG 700, ICU, saline+insulin -- could also be hyperglycaemic coma territory
    "3013756811-2024-09995": {10005557, 10016750},  # Blood glucose increased, Hyperglycaemic unconsciousness
    # BG 212, device fill issue -- BG elevation is secondary
    "3013756811-2024-17732": {10005557},   # Blood glucose increased
    # BG 250+, cannula dislodged -- BG increase plausible
    "3004464228-2024-27443": {10005557},   # Blood glucose increased
    # Infusion site bleeding -- haemorrhage variants plausible
    "3004464228-2024-04816": {10022034, 10055322},  # Infusion site haemorrhage, Medical device site haemorrhage
    # No-delivery + hemorrhage report
    "2032227-2024-227041":   {10022034, 10055322},  # Infusion site haemorrhage, Medical device site haemorrhage
    # Skin irritation / allergic to adhesive -- dermatitis variants plausible
    "3004464228-2024-04262": {10061519, 10040914},  # Medical device site dermatitis, Skin reaction
    "3004464228-2024-29565": {10061519, 10040880},  # Medical device site dermatitis, Skin irritation
}

GOLDEN_SET = "data/eval/golden_set_v1.jsonl"
OUT_FILE   = "data/eval/miss_analysis_v2.json"
FUSION_LIMIT = 50   # must match _vector_search limit in hybrid_search.py
TOP_K_FULL   = 100


def get_rank(results, pt_code):
    """Return 1-based rank of pt_code in results, or None if not found."""
    for i, r in enumerate(results):
        if r.pt_code == pt_code:
            return i + 1
    return None


def acceptable_present(results, acceptable_codes, top_n=20):
    """Check if any acceptable PT appears in top_n results."""
    top_codes = {r.pt_code for r in results[:top_n]}
    found = top_codes & acceptable_codes
    return bool(found), found


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
        mdr_text      = c["mdr_text"]
        expected_code = c["expected_pt_code"]
        expected_name = c["expected_pt_name"]
        case_id       = c["mdr_report_key"]
        acc_codes     = ACCEPTABLE.get(case_id, set())

        # --- fusion (q1+q2 union, top-100) ---
        fusion_res  = searcher.search(mdr_text, top_k=TOP_K_FULL)
        fusion_rank = get_rank(fusion_res, expected_code)

        # --- q1 only, limit=50 (matching actual fusion) ---
        q1      = mdr_text.split(".")[0].strip() or mdr_text
        # Use internal _vector_search to isolate vector-only signal per query
        # Fallback: full searcher.search with q1 as text (includes BM25)
        # For simplicity: re-run searcher with q1 text, top_k=FUSION_LIMIT
        r1      = searcher.search(q1, top_k=FUSION_LIMIT)
        q1_rank = get_rank(r1, expected_code)

        # --- q2 only, limit=50 ---
        q2      = mdr_text[:512]
        r2      = searcher.search(q2, top_k=FUSION_LIMIT)
        q2_rank = get_rank(r2, expected_code)

        # --- acceptable candidates in top-20 fusion results ---
        acc_hit, acc_found = acceptable_present(fusion_res, acc_codes, top_n=20)

        names = [r.pt_name for r in fusion_res]
        codes = [r.pt_code for r in fusion_res]

        row = {
            "case_id":                   case_id,
            "expected_pt":               expected_name,
            "fusion_rank":               fusion_rank,
            "q1_rank":                   q1_rank,
            "q2_rank":                   q2_rank,
            "acceptable_present_top20":  acc_hit,
            "acceptable_found":          list(acc_found),
            "top1":                      names[0] if names else None,
            "top5":                      names[:5],
            "top20":                     names[:20],
        }
        results_out.append(row)

        print(f"\n{'=' * 65}")
        print(f"CASE:         {case_id}")
        print(f"EXPECTED:     {expected_name}")
        print(f"fusion_rank:  {fusion_rank}   q1_rank: {q1_rank}   q2_rank: {q2_rank}")
        print(f"acceptable_present_top20: {acc_hit}  -> {acc_found}")
        print(f"TOP-5:")
        for i, (n, code) in enumerate(zip(names[:5], codes[:5]), 1):
            tag = " <-- EXPECTED" if code == expected_code else \
                  " <-- ACCEPTABLE" if code in acc_codes else ""
            print(f"  {i:2d}. {n}{tag}")
        print(f"TOP 6-20:")
        for i, (n, code) in enumerate(zip(names[5:20], codes[5:20]), 6):
            tag = " <-- EXPECTED" if code == expected_code else \
                  " <-- ACCEPTABLE" if code in acc_codes else ""
            print(f"  {i:2d}. {n}{tag}")

    json.dump(results_out, open(OUT_FILE, "w"), indent=2)
    print(f"\nSaved: {OUT_FILE}")

    # Summary table
    print(f"\n{'=' * 90}")
    print(f"{'CASE_ID':<30} {'EXPECTED PT':<35} {'FUS':>4} {'Q1':>4} {'Q2':>4} {'ACC_TOP20':>9}")
    print("-" * 90)
    for r in results_out:
        def fmt(v): return str(v) if v else "NONE"
        print(f"{r['case_id']:<30} {r['expected_pt']:<35} "
              f"{fmt(r['fusion_rank']):>4} {fmt(r['q1_rank']):>4} "
              f"{fmt(r['q2_rank']):>4} {str(r['acceptable_present_top20']):>9}")

    # Soft recall estimate
    strict_hits = sum(1 for r in results_out if r["fusion_rank"] is not None)
    soft_hits   = sum(1 for r in results_out
                      if r["fusion_rank"] is not None or r["acceptable_present_top20"])
    total = len(results_out)
    print(f"\nMiss set ({total} cases):")
    print(f"  strict hits (expected PT in top-100): {strict_hits}/{total}")
    print(f"  soft hits   (+ acceptable in top-20): {soft_hits}/{total}")


if __name__ == "__main__":
    main()
