"""
import_meddra.py -- Load MedDRA v29.0 hierarchy into processed.meddra_terms

Reads MedAscii files from the official MSSO release package and inserts
one row per Preferred Term (PT) with its full SOC hierarchy.

Source files used:
  mdhier.asc  -- pt -> hlt -> hlgt -> soc (primary SOC only, flag='Y')
  llt.asc     -- llt -> pt mapping (stored in separate table, see below)

Usage:
  python -m vigilex.data.import_meddra --meddra-dir raw_data/MedDRA_29_0_English/MedAscii

  Or with explicit DB URL:
  python -m vigilex.data.import_meddra \
      --meddra-dir raw_data/MedDRA_29_0_English/MedAscii \
      --db-url postgresql://vigilex:yourpassword@localhost:5432/vigilex

Prerequisites:
  - processed.meddra_terms table must exist (created by init_db/01_init.sql)
  - pip install psycopg2-binary python-dotenv
"""

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Optional

try:
    import psycopg2
    import psycopg2.extras
except ImportError:
    sys.exit("psycopg2-binary not installed. Run: pip install psycopg2-binary")

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv optional -- fall back to env vars


MEDDRA_VERSION = "29.0"

# ---------------------------------------------------------------------------
# mdhier.asc column indices (0-based, $ delimiter, no header)
# ---------------------------------------------------------------------------
# 0  pt_code
# 1  hlt_code
# 2  hlgt_code
# 3  soc_code
# 4  pt_name
# 5  hlt_name
# 6  hlgt_name
# 7  soc_name
# 8  soc_abbrev
# 9  (null)
# 10 primary_soc_code
# 11 primary_soc_flag  (Y = primary SOC for this PT)
# 12 (trailing)
MDHIER_COLS = {
    "pt_code": 0,
    "hlt_code": 1,
    "hlgt_code": 2,
    "soc_code": 3,
    "pt_name": 4,
    "hlt_name": 5,
    "hlgt_name": 6,
    "soc_name": 7,
    "primary_soc_flag": 11,
}

# ---------------------------------------------------------------------------
# llt.asc column indices
# ---------------------------------------------------------------------------
# 0  llt_code
# 1  llt_name
# 2  pt_code
# 9  llt_currency  (Y=current, N=retired)
LLT_COLS = {
    "llt_code": 0,
    "llt_name": 1,
    "pt_code": 2,
    "currency": 9,
}


def _read_asc(path: Path) -> list[list[str]]:
    """Read a $ delimited .asc file, strip trailing empty fields."""
    rows = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n").rstrip("$")
            if line:
                rows.append(line.split("$"))
    return rows


def load_hierarchy(meddra_dir: Path) -> list[dict]:
    """
    Parse mdhier.asc, keep only primary-SOC rows (flag='Y').
    Returns list of dicts ready for DB insert.
    """
    path = meddra_dir / "mdhier.asc"
    if not path.exists():
        sys.exit(f"File not found: {path}")

    rows = _read_asc(path)
    terms = []
    seen_pt = set()

    for r in rows:
        if len(r) <= MDHIER_COLS["primary_soc_flag"]:
            continue  # malformed row

        flag = r[MDHIER_COLS["primary_soc_flag"]].strip()
        if flag != "Y":
            continue  # skip non-primary SOC assignments

        pt_code = int(r[MDHIER_COLS["pt_code"]].strip())
        if pt_code in seen_pt:
            continue  # guard against duplicate primary flags
        seen_pt.add(pt_code)

        terms.append({
            "pt_code": pt_code,
            "pt_name": r[MDHIER_COLS["pt_name"]].strip(),
            "hlt_code": int(r[MDHIER_COLS["hlt_code"]].strip()),
            "hlt_name": r[MDHIER_COLS["hlt_name"]].strip(),
            "hlgt_code": int(r[MDHIER_COLS["hlgt_code"]].strip()),
            "hlgt_name": r[MDHIER_COLS["hlgt_name"]].strip(),
            "soc_code": int(r[MDHIER_COLS["soc_code"]].strip()),
            "soc_name": r[MDHIER_COLS["soc_name"]].strip(),
            "llt_code": None,
            "llt_name": None,
            "meddra_version": MEDDRA_VERSION,
        })

    print(f"[mdhier.asc] {len(terms)} primary-SOC PT rows parsed")
    return terms


def load_llt_map(meddra_dir: Path) -> dict[int, list[tuple[int, str]]]:
    """
    Parse llt.asc, return dict: pt_code -> [(llt_code, llt_name), ...]
    Only includes current LLTs (currency flag = 'Y').
    """
    path = meddra_dir / "llt.asc"
    if not path.exists():
        print("[llt.asc] Not found -- skipping LLT import")
        return {}

    rows = _read_asc(path)
    llt_map: dict[int, list[tuple[int, str]]] = {}

    for r in rows:
        if len(r) <= LLT_COLS["currency"]:
            continue
        currency = r[LLT_COLS["currency"]].strip() if len(r) > LLT_COLS["currency"] else "Y"
        if currency != "Y":
            continue  # skip retired LLTs

        try:
            llt_code = int(r[LLT_COLS["llt_code"]].strip())
            llt_name = r[LLT_COLS["llt_name"]].strip()
            pt_code = int(r[LLT_COLS["pt_code"]].strip())
        except (ValueError, IndexError):
            continue

        llt_map.setdefault(pt_code, []).append((llt_code, llt_name))

    print(f"[llt.asc] {sum(len(v) for v in llt_map.values())} current LLTs mapped to {len(llt_map)} PTs")
    return llt_map


def get_db_url() -> str:
    url = os.getenv("DATABASE_URL")
    if url:
        return url
    # Build from individual components (matches vigilex docker-compose .env)
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "vigilex")
    user = os.getenv("POSTGRES_USER", "vigilex")
    pw = os.getenv("POSTGRES_PASSWORD", "")
    return f"postgresql://{user}:{pw}@{host}:{port}/{db}"


def insert_terms(conn, terms: list[dict]) -> int:
    """
    Upsert MedDRA terms into processed.meddra_terms.
    ON CONFLICT (pt_code) DO UPDATE -- safe to re-run after version upgrade.
    """
    sql = """
        INSERT INTO processed.meddra_terms
            (llt_code, llt_name, pt_code, pt_name,
             hlt_code, hlt_name, hlgt_code, hlgt_name,
             soc_code, soc_name, meddra_version)
        VALUES
            (%(llt_code)s, %(llt_name)s, %(pt_code)s, %(pt_name)s,
             %(hlt_code)s, %(hlt_name)s, %(hlgt_code)s, %(hlgt_name)s,
             %(soc_code)s, %(soc_name)s, %(meddra_version)s)
        ON CONFLICT (pt_code) DO UPDATE SET
            pt_name        = EXCLUDED.pt_name,
            hlt_code       = EXCLUDED.hlt_code,
            hlt_name       = EXCLUDED.hlt_name,
            hlgt_code      = EXCLUDED.hlgt_code,
            hlgt_name      = EXCLUDED.hlgt_name,
            soc_code       = EXCLUDED.soc_code,
            soc_name       = EXCLUDED.soc_name,
            meddra_version = EXCLUDED.meddra_version
    """
    with conn.cursor() as cur:
        psycopg2.extras.execute_batch(cur, sql, terms, page_size=500)
    conn.commit()
    return len(terms)


def create_llt_table_if_missing(conn) -> None:
    """
    Create a dedicated meddra_llt table if it doesn't exist.
    This is cleaner than stuffing LLTs into the PT table (many LLTs per PT).
    """
    ddl = """
        CREATE TABLE IF NOT EXISTS processed.meddra_llt (
            llt_code    INTEGER PRIMARY KEY,
            llt_name    TEXT NOT NULL,
            pt_code     INTEGER NOT NULL
                        REFERENCES processed.meddra_terms(pt_code),
            meddra_version TEXT DEFAULT '29.0'
        );
        CREATE INDEX IF NOT EXISTS idx_meddra_llt_pt_code
            ON processed.meddra_llt (pt_code);
        CREATE INDEX IF NOT EXISTS idx_meddra_llt_name_trgm
            ON processed.meddra_llt USING GIN (llt_name gin_trgm_ops);
    """
    with conn.cursor() as cur:
        cur.execute(ddl)
    conn.commit()


def insert_llts(conn, llt_map: dict[int, list[tuple[int, str]]]) -> int:
    """Upsert LLTs into processed.meddra_llt."""
    rows = []
    for pt_code, llts in llt_map.items():
        for llt_code, llt_name in llts:
            rows.append({
                "llt_code": llt_code,
                "llt_name": llt_name,
                "pt_code": pt_code,
                "meddra_version": MEDDRA_VERSION,
            })

    sql = """
        INSERT INTO processed.meddra_llt (llt_code, llt_name, pt_code, meddra_version)
        VALUES (%(llt_code)s, %(llt_name)s, %(pt_code)s, %(meddra_version)s)
        ON CONFLICT (llt_code) DO UPDATE SET
            llt_name       = EXCLUDED.llt_name,
            pt_code        = EXCLUDED.pt_code,
            meddra_version = EXCLUDED.meddra_version
    """
    with conn.cursor() as cur:
        psycopg2.extras.execute_batch(cur, sql, rows, page_size=500)
    conn.commit()
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import MedDRA v29.0 hierarchy into PostgreSQL"
    )
    parser.add_argument(
        "--meddra-dir",
        default="raw_data/MedDRA_29_0_English/MedAscii",
        help="Path to MedAscii folder (default: raw_data/MedDRA_29_0_English/MedAscii)",
    )
    parser.add_argument(
        "--db-url",
        default=None,
        help="PostgreSQL connection URL (default: reads from DATABASE_URL or .env)",
    )
    parser.add_argument(
        "--skip-llt",
        action="store_true",
        help="Skip LLT import (faster, useful for initial testing)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse files and report counts without writing to DB",
    )
    args = parser.parse_args()

    meddra_dir = Path(args.meddra_dir)
    if not meddra_dir.exists():
        sys.exit(f"MedDRA directory not found: {meddra_dir}")

    # Parse files
    terms = load_hierarchy(meddra_dir)
    llt_map = {} if args.skip_llt else load_llt_map(meddra_dir)

    if args.dry_run:
        print(f"\n[dry-run] Would insert {len(terms)} PT rows")
        if llt_map:
            total_llts = sum(len(v) for v in llt_map.values())
            print(f"[dry-run] Would insert {total_llts} LLT rows")
        print("[dry-run] No DB writes performed.")
        return

    db_url = args.db_url or get_db_url()
    print(f"\nConnecting to DB...")
    try:
        conn = psycopg2.connect(db_url)
    except psycopg2.OperationalError as e:
        sys.exit(f"DB connection failed: {e}")

    print("Inserting MedDRA terms into processed.meddra_terms ...")
    n_terms = insert_terms(conn, terms)
    print(f"  -> {n_terms} PT rows upserted")

    if llt_map:
        print("Creating processed.meddra_llt table if missing ...")
        create_llt_table_if_missing(conn)
        print("Inserting LLTs into processed.meddra_llt ...")
        n_llts = insert_llts(conn, llt_map)
        print(f"  -> {n_llts} LLT rows upserted")

    conn.close()
    print(f"\nDone. MedDRA v{MEDDRA_VERSION} successfully imported.")
    print("Next step: run the embedding pipeline to populate pt_embedding column.")
    print("  python -m vigilex.coding.embed_meddra_terms")


if __name__ == "__main__":
   main()