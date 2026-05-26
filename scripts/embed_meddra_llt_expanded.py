"""
embed_meddra_llt_expanded.py
Embeds processed.meddra_terms_llt_expanded.embedding_text using all-mpnet-base-v2.
Writes results into embedding_mpnet_llt column.

Usage:
    python3 scripts/embed_meddra_llt_expanded.py [--dry-run] [--batch-size 64]
"""
import argparse
import logging
import os
import sys
import time
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

try:
    import psycopg2
    import psycopg2.extras
except ImportError:
    sys.exit("psycopg2-binary not installed.")

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    sys.exit("sentence-transformers not installed.")

MODEL_NAME  = "sentence-transformers/all-mpnet-base-v2"
TABLE       = "processed.meddra_terms_llt_expanded"
TEXT_COL    = "embedding_text"
EMBED_COL   = "embedding_mpnet_llt"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run",    action="store_true", help="Load model + fetch rows, no DB writes")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--force",      action="store_true", help="Re-embed already embedded rows")
    return p.parse_args()


def main():
    args = parse_args()
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        sys.exit("DATABASE_URL not set.")

    log.info(f"Model: {MODEL_NAME}")
    log.info(f"Table: {TABLE} | text_col: {TEXT_COL} | embed_col: {EMBED_COL}")
    log.info(f"Started: {datetime.now().isoformat()}")

    log.info("Connecting to database...")
    conn = psycopg2.connect(db_url)

    where = "" if args.force else f"WHERE {EMBED_COL} IS NULL"
    with conn.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) FROM {TABLE} {where}")
        total = cur.fetchone()[0]
    log.info(f"Rows to embed: {total}")

    if args.dry_run:
        log.info("--dry-run: fetching first batch to verify shape")
    else:
        log.info(f"Loading model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)
    log.info(f"Model ready")

    if args.dry_run:
        with conn.cursor() as cur:
            cur.execute(f"SELECT {TEXT_COL} FROM {TABLE} LIMIT {args.batch_size}")
            rows = cur.fetchall()
        texts = [r[0] for r in rows]
        emb = model.encode(texts, normalize_embeddings=True)
        log.info(f"Dry-run OK: shape={emb.shape}, norm={float((emb**2).sum(axis=1).mean()**0.5):.6f}")
        conn.close()
        return

    # Full run
    with conn.cursor() as cur:
        cur.execute(f"SELECT pt_code, {TEXT_COL} FROM {TABLE} {where} ORDER BY pt_code")
        all_rows = cur.fetchall()

    start = time.time()
    done  = 0
    for batch_start in range(0, len(all_rows), args.batch_size):
        batch = all_rows[batch_start:batch_start + args.batch_size]
        codes = [r[0] for r in batch]
        texts = [r[1] for r in batch]

        embeddings = model.encode(texts, normalize_embeddings=True)

        with conn.cursor() as cur:
            for code, emb in zip(codes, embeddings):
                cur.execute(
                    f"UPDATE {TABLE} SET {EMBED_COL} = %s WHERE pt_code = %s",
                    (emb.tolist(), code)
                )
        conn.commit()
        done += len(batch)

        elapsed = time.time() - start
        eta     = (elapsed / done) * (total - done) if done > 0 else 0
        print(f"\r  [{done:6d}/{total:6d}] {100*done/total:5.1f}%"
              f"  elapsed: {elapsed:5.0f}s  ETA: {eta:5.0f}s", end="", flush=True)

    print()
    elapsed = time.time() - start
    log.info(f"Done. {done} embeddings written in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    conn.close()


if __name__ == "__main__":
    main()
