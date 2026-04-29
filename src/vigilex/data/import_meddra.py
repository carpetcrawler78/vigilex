"""
import_meddra.py -- Load MedDRA v29.0 hierarchy into PostgreSQL.

What is MedDRA?
    The Medical Dictionary for Regulatory Activities is a standardised,
    multilingual medical terminology used by pharmaceutical and device companies
    worldwide to classify adverse events. Every serious adverse event report
    (to FDA, EMA, etc.) requires a MedDRA code.

    MedDRA has a 5-level hierarchy:
        SOC   -- System Organ Class      (27 top-level categories)
                 e.g. "Cardiac disorders"
        HLGT  -- High Level Group Term   (334 groups)
                 e.g. "Cardiac arrhythmias"
        HLT   -- High Level Term         (1,737 terms)
                 e.g. "Rate and rhythm disorders NEC"
        PT    -- Preferred Term          (27,361 terms in v29.0)
                 e.g. "Ventricular tachycardia"  <-- this is what we code to
        LLT   -- Lowest Level Term       (81,719 terms in v29.0)
                 Synonyms and near-synonyms for PTs

    We code adverse event narratives to PT level (Preferred Terms).
    Each PT belongs to exactly one primary SOC.

What does this script do?
    1. Reads the mdhier.asc file (PT hierarchy: PT -> HLT -> HLGT -> SOC)
    2. Reads the llt.asc file (LLT synonyms, each pointing to a PT)
    3. Inserts the PTs into processed.meddra_terms
    4. Inserts the LLTs into processed.meddra_llt

Why does this matter for Module 2?
    Module 2 (the MedDRA Coding Engine) needs all PT names in the database
    so it can search for the best match. The PubMedBERT embedding script
    runs after this import and adds a 768-dimensional vector to each PT row.

Source files:
    mdhier.asc -- PT hierarchy (one row per PT x SOC assignment)
    llt.asc    -- LLT synonyms (many rows per PT)

File format: pipe-character-delimited ASCII, dollar sign ($) as separator.
No CSV header -- column positions are defined by the MedDRA specification.

Usage:
    python -m vigilex.data.import_meddra --meddra-dir raw_data/MedDRA_29_0_English/MedAscii
    python -m vigilex.data.import_meddra --meddra-dir ... --dry-run  # check counts only
"""

import argparse
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
    pass  # dotenv is optional -- fall back to environment variables set elsewhere


MEDDRA_VERSION = "29.0"


# ---------------------------------------------------------------------------
# Column index maps for the MedAscii flat files
# ---------------------------------------------------------------------------
# MedDRA distributes data as fixed-format ASCII files with $ as the delimiter
# and no header row. The column positions are defined by the MedDRA specification
# and do not change within a version.

# mdhier.asc -- the "medical hierarchy" file.
# Each row represents one PT-to-SOC assignment. A PT can appear in multiple
# SOCs (e.g. as a secondary classification), but each PT has exactly one
# PRIMARY SOC (identified by primary_soc_flag == "Y"). We only import primary rows.
MDHIER_COLS = {
    "pt_code":          0,   # MedDRA numeric code for the Preferred Term
    "hlt_code":         1,   # numeric code for the High Level Term
    "hlgt_code":        2,   # numeric code for the High Level Group Term
    "soc_code":         3,   # numeric code for the System Organ Class
    "pt_name":          4,   # full text name, e.g. "Hypoglycaemia"
    "hlt_name":         5,   # e.g. "Glucose metabolism disorders"
    "hlgt_name":        6,   # e.g. "Nutrition and metabolic disorders"
    "soc_name":         7,   # e.g. "Metabolism and nutrition disorders"
    "primary_soc_flag": 11,  # "Y" if this SOC is the primary classification
}

# llt.asc -- the "lowest level term" file.
# Each row is one LLT (synonym or near-synonym) that maps to a PT.
# We only import LLTs where currency == "Y" (active, not retired/deprecated).
LLT_COLS = {
    "llt_code": 0,   # numeric LLT code
    "llt_name": 1,   # LLT text, e.g. "Low blood glucose" (synonym for Hypoglycaemia)
    "pt_code":  2,   # the parent PT this LLT maps to
    "currency": 9,   # "Y" = current/active, "N" = retired
}


# ---------------------------------------------------------------------------
# File reader
# ---------------------------------------------------------------------------

def _read_asc(path: Path) -> list[list[str]]:
    """
    Read a MedDRA .asc file and return it as a list of rows.

    Each row is a list of field strings. The file uses $ as the delimiter
    and may have a trailing $ at the end of each line (which we strip).

    We use errors="replace" to handle any non-UTF-8 bytes gracefully --
    some versions of MedDRA contain latin-1 characters in rare term names.
    """
    rows = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n").rstrip("$")
            if line:
                rows.append(line.split("$"))
    return rows


# ---------------------------------------------------------------------------
# Parse mdhier.asc (PT hierarchy)
# ---------------------------------------------------------------------------

def load_hierarchy(meddra_dir: Path) -> list[dict]:
    """
    Parse mdhier.asc and return one dict per Preferred Term (primary SOC only).

    Why filter to primary_soc_flag == "Y"?
        A single PT can be associated with multiple SOCs (e.g. "Pyrexia" appears
        under both "General disorders" and "Infections"). The primary SOC flag
        marks the authoritative classification. We store only the primary to avoid
        duplicate PT rows in our database.

    Returns:
        List of dicts with keys: pt_code, pt_name, hlt_code, hlt_name,
        hlgt_code, hlgt_name, soc_code, soc_name, meddra_version.
    """
    path = meddra_dir / "mdhier.asc"
    if not path.exists():
        sys.exit(f"mdhier.asc not found at: {path}")

    rows = _read_asc(path)
    terms = []
    seen_pt = set()  # guard against duplicate pt_codes with multiple "Y" flags

    for r in rows:
        # Skip malformed rows (too few columns to read the flag)
        if len(r) <= MDHIER_COLS["primary_soc_flag"]:
            continue

        flag = r[MDHIER_COLS["primary_soc_flag"]].strip()
        if flag != "Y":
            continue  # skip non-primary SOC assignments

        pt_code = int(r[MDHIER_COLS["pt_code"]].strip())
        if pt_code in seen_pt:
            continue  # safety check: skip if we already have this PT
        seen_pt.add(pt_code)

        terms.append({
            "pt_code":        pt_code,
            "pt_name":        r[MDHIER_COLS["pt_name"]].strip(),
            "hlt_code":       int(r[MDHIER_COLS["hlt_code"]].strip()),
            "hlt_name":       r[MDHIER_COLS["hlt_name"]].strip(),
            "hlgt_code":      int(r[MDHIER_COLS["hlgt_code"]].strip()),
            "hlgt_name":      r[MDHIER_COLS["hlgt_name"]].strip(),
            "soc_code":       int(r[MDHIER_COLS["soc_code"]].strip()),
            "soc_name":       r[MDHIER_COLS["soc_name"]].strip(),
            "llt_code":       None,  # not stored at PT level (separate LLT table)
            "llt_name":       None,
            "meddra_version": MEDDRA_VERSION,
        })

    print(f"[mdhier.asc] {len(terms)} primary-SOC PT rows parsed")
    return terms


# ---------------------------------------------------------------------------
# Parse llt.asc (synonym LLTs)
# ---------------------------------------------------------------------------

def load_llt_map(meddra_dir: Path) -> dict[int, list[tuple[int, str]]]:
    """
    Parse llt.asc and return a dict mapping pt_code -> [(llt_code, llt_name)].

    Why do we need LLTs?
        MAUDE narratives use everyday clinical language, not necessarily the
        exact MedDRA PT name. For example, a reporter might write "low blood glucose"
        instead of the PT "Hypoglycaemia". The LLT "Low blood glucose" is a synonym
        that maps to that PT. Searching LLT names improves the BM25 arm of the
        hybrid search pipeline.

    We only import current LLTs (currency == "Y"). Retired LLTs (currency == "N")
    were active in older MedDRA versions but are no longer maintained.

    Returns:
        dict: {pt_code: [(llt_code, llt_name), ...], ...}
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
            continue  # skip retired (deprecated) LLTs

        try:
            llt_code = int(r[LLT_COLS["llt_code"]].strip())
            llt_name = r[LLT_COLS["llt_name"]].strip()
            pt_code  = int(r[LLT_COLS["pt_code"]].strip())
        except (ValueError, IndexError):
            continue  # skip malformed rows

        # Group LLTs by their parent PT code
        llt_map.setdefault(pt_code, []).append((llt_code, llt_name))

    total_llts = sum(len(v) for v in llt_map.values())
    print(f"[llt.asc] {total_llts} current LLTs mapped to {len(llt_map)} PTs")
    return llt_map


# ---------------------------------------------------------------------------
# Database connection helper
# ---------------------------------------------------------------------------

def get_db_url() -> str:
    """
    Return the PostgreSQL connection URL.

    Checks DATABASE_URL first (standard docker-compose convention).
    Falls back to building a URL from individual components (POSTGRES_HOST, etc.)
    which may be useful during local development or manual testing.
    """
    url = os.getenv("DATABASE_URL")
    if url:
        return url
    # Build from individual components -- matches the variable names in .env
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db   = os.getenv("POSTGRES_DB", "vigilex")
    user = os.getenv("POSTGRES_USER", "vigilex")
    pw   = os.getenv("POSTGRES_PASSWORD", "")
    return f"postgresql://{user}:{pw}@{host}:{port}/{db}"


# ---------------------------------------------------------------------------
# Database write functions
# ---------------------------------------------------------------------------

def insert_terms(conn, terms: list[dict]) -> int:
    """
    Insert (or update) MedDRA Preferred Terms into processed.meddra_terms.

    Uses ON CONFLICT (pt_code) DO UPDATE so the script is safe to re-run
    after a MedDRA version upgrade -- existing rows are updated with new names
    if the terminology changed, rather than causing a duplicate key error.

    After this function completes, the embed_meddra_terms.py script must be
    run to populate the pt_embedding column with PubMedBERT vectors.

    Args:
        conn:  open psycopg2 connection
        terms: list of dicts from load_hierarchy()

    Returns:
        Number of rows processed (inserts + updates combined).
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
    Create the processed.meddra_llt table if it does not already exist.

    Design decision: LLTs are stored in a separate table (not as an array
    in the PT table) because each PT has an average of 3 LLTs, and we need
    to search LLT names efficiently with a GIN trigram index.
    Storing them separately allows the index to cover each LLT name individually.

    The GIN index (gin_trgm_ops) powers the BM25 synonym search arm in
    hybrid_search.py -- it enables fast substring/similarity queries over
    all 81,719 LLT names.
    """
    ddl = """
        CREATE TABLE IF NOT EXISTS processed.meddra_llt (
            llt_code    INTEGER PRIMARY KEY,
            llt_name    TEXT NOT NULL,
            pt_code     INTEGER NOT NULL
                        REFERENCES processed.meddra_terms(pt_code),
            meddra_version TEXT DEFAULT '29.0'
        );
        -- Index for joining LLTs back to their parent PT
        CREATE INDEX IF NOT EXISTS idx_meddra_llt_pt_code
            ON processed.meddra_llt (pt_code);
        -- GIN trigram index: enables fast word_similarity() queries over LLT names.
        -- GIN = Generalized Inverted Index; gin_trgm_ops = trigram operator class.
        -- This is what makes the BM25 synonym search in hybrid_search.py fast.
        CREATE INDEX IF NOT EXISTS idx_meddra_llt_name_trgm
            ON processed.meddra_llt USING GIN (llt_name gin_trgm_ops);
    """
    with conn.cursor() as cur:
        cur.execute(ddl)
    conn.commit()


def insert_llts(conn, llt_map: dict[int, list[tuple[int, str]]]) -> int:
    """
    Insert (or update) LLT synonym rows into processed.meddra_llt.

    Flattens the dict structure (pt_code -> [(llt_code, llt_name), ...])
    into individual rows for batch insertion.

    Uses ON CONFLICT (llt_code) DO UPDATE so the script is safe to re-run.

    Args:
        conn:    open psycopg2 connection
        llt_map: output from load_llt_map()

    Returns:
        Total number of LLT rows processed.
    """
    # Flatten the nested dict into a flat list of row dicts for execute_batch
    rows = []
    for pt_code, llts in llt_map.items():
        for llt_code, llt_name in llts:
            rows.append({
                "llt_code":       llt_code,
                "llt_name":       llt_name,
                "pt_code":        pt_code,
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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Parse arguments and run the MedDRA import pipeline.

    Order of operations:
        1. Parse mdhier.asc -> list of PT dicts
        2. Parse llt.asc    -> dict of LLT synonyms (optional)
        3. Insert PTs into processed.meddra_terms
        4. Create processed.meddra_llt table (if needed)
        5. Insert LLTs into processed.meddra_llt

    After this script completes, run:
        python -m vigilex.coding.embed_meddra_terms
    to generate PubMedBERT embeddings for all PT names.
    """
    parser = argparse.ArgumentParser(
        description="Import MedDRA v29.0 hierarchy into PostgreSQL"
    )
    parser.add_argument(
        "--meddra-dir",
        default="raw_data/MedDRA_29_0_English/MedAscii",
        help="Path to the MedAscii folder inside the MSSO release package",
    )
    parser.add_argument(
        "--db-url",
        default=None,
        help="PostgreSQL connection URL (default: reads DATABASE_URL from environment)",
    )
    parser.add_argument(
        "--skip-llt",
        action="store_true",
        help="Skip LLT synonym import (faster; useful for initial testing)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse files and print counts, but do not write to the database",
    )
    args = parser.parse_args()

    meddra_dir = Path(args.meddra_dir)
    if not meddra_dir.exists():
        sys.exit(f"MedDRA directory not found: {meddra_dir}")

    # Parse the source files first (this works offline, no DB needed)
    terms   = load_hierarchy(meddra_dir)
    llt_map = {} if args.skip_llt else load_llt_map(meddra_dir)

    if args.dry_run:
        # Dry-run: report what would be inserted without actually writing
        print(f"\n[dry-run] Would insert {len(terms)} PT rows into processed.meddra_terms")
        if llt_map:
            total_llts = sum(len(v) for v in llt_map.values())
            print(f"[dry-run] Would insert {total_llts} LLT rows into processed.meddra_llt")
        print("[dry-run] No database writes performed.")
        return

    # Connect to PostgreSQL
    db_url = args.db_url or get_db_url()
    print(f"\nConnecting to database...")
    try:
        conn = psycopg2.connect(db_url)
    except psycopg2.OperationalError as e:
        sys.exit(f"Database connection failed: {e}")

    # Insert PTs
    print("Inserting Preferred Terms into processed.meddra_terms ...")
    n_terms = insert_terms(conn, terms)
    print(f"  -> {n_terms} PT rows upserted")

    # Insert LLTs (if requested)
    if llt_map:
        print("Creating processed.meddra_llt table (if not exists) ...")
        create_llt_table_if_missing(conn)
        print("Inserting LLT synonyms into processed.meddra_llt ...")
        n_llts = insert_llts(conn, llt_map)
        print(f"  -> {n_llts} LLT rows upserted")

    conn.close()
    print(f"\nDone. MedDRA v{MEDDRA_VERSION} successfully imported.")
    print("\nNext step: generate PubMedBERT embeddings for all PT names:")
    print("  python -m vigilex.coding.embed_meddra_terms")


if __name__ == "__main__":
    main()
