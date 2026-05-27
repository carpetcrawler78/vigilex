"""
show_stage2_detail.py -- Print Stage 1+2 breakdown for all golden set cases.

No LLM, no MLflow, no DB writes. Output: per-case table with CE scores.

Usage:
    python scripts/show_stage2_detail.py
    python scripts/show_stage2_detail.py --eval-path data/eval/golden_set_v1.jsonl
    python scripts/show_stage2_detail.py --reranker-model cross-encoder/ms-marco-electra-base
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

import psycopg2
from vigilex.coding.hybrid_search import EmbeddingModel, HybridSearcher
from vigilex.coding.reranker import CrossEncoderReranker


def main():
    parser = argparse.ArgumentParser(description="Stage 1+2 detail view for golden set")
    parser.add_argument("--eval-path", default="data/eval/golden_set_v1.jsonl")
    parser.add_argument("--reranker-model",
                        default="cross-encoder/ms-marco-electra-base")
    parser.add_argument("--top-k-stage1", type=int, default=20)
    parser.add_argument("--top-k-stage2", type=int, default=5)
    args = parser.parse_args()

    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        sys.exit("DATABASE_URL not set")

    eval_path = ROOT / args.eval_path
    cases = [json.loads(l) for l in eval_path.read_text().splitlines() if l.strip()]
    print(f"Loaded {len(cases)} cases from {eval_path.name}")

    print(f"Loading EmbeddingModel...")
    em = EmbeddingModel()
    print(f"Loading CrossEncoderReranker ({args.reranker_model})...")
    reranker = CrossEncoderReranker(model_name=args.reranker_model)

    conn = psycopg2.connect(db_url)
    searcher = HybridSearcher(conn, embedding_model=em)

    SEP = "-" * 80
    hits = 0

    for i, case in enumerate(cases, 1):
        key          = case["mdr_report_key"]
        mdr_text     = case["mdr_text"]
        expected_code = case["expected_pt_code"]
        expected_name = case["expected_pt_name"]

        t0 = time.time()
        stage1 = searcher.search(mdr_text, top_k=args.top_k_stage1)
        stage2 = reranker.rerank(mdr_text, stage1, top_k=args.top_k_stage2)
        elapsed = time.time() - t0

        stage2_codes = [r.pt_code for r in stage2]
        hit = expected_code in stage2_codes
        rank = next((j + 1 for j, c in enumerate(stage2_codes) if c == expected_code), None)
        if hit:
            hits += 1

        print(f"\n{'=' * 80}")
        print(f"[{i:2d}/24]  {key}  |  R@5={'Y' if hit else 'N'}  |  [{elapsed:.1f}s]")
        print(f"  Expected : {expected_name} ({expected_code})")
        print(f"  Snippet  : {mdr_text[:120].strip()}")
        print(SEP)
        print(f"  {'Rank':<4}  {'PT Name':<45}  {'CE Score':>10}  {'Stage1 Rank':>11}  {'Note'}")
        print(SEP)
        for j, r in enumerate(stage2, 1):
            note = ""
            if r.pt_code == expected_code:
                note = "<-- EXPECTED"
            s1_rank = next((k + 1 for k, c in enumerate([x.pt_code for x in stage1])
                            if c == r.pt_code), None)
            s1_str = str(s1_rank) if s1_rank else "n/a"
            print(f"  {j:<4}  {r.pt_name:<45}  {r.crossencoder_score:>+10.4f}  {s1_str:>11}  {note}")

        if not hit:
            # Show where expected PT ended up in stage1
            s1_rank_exp = next((k + 1 for k, c in enumerate([x.pt_code for x in stage1])
                                if c == expected_code), None)
            if s1_rank_exp:
                print(f"  ** Expected was stage1 rank {s1_rank_exp} -- dropped by CE reranker (cat_B)")
            else:
                print(f"  ** Expected NOT in stage1 top-{args.top_k_stage1} (cat_A miss)")

    conn.close()
    print(f"\n{'=' * 80}")
    print(f"SUMMARY: recall@5 = {hits}/{len(cases)} = {hits/len(cases):.3f}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
