"""
SentinelAI (vigilex) -- MAUDE Ingestion Worker.

Holt Adverse Event Reports von der openFDA API und schreibt sie
in raw.maude_reports auf der PostgreSQL-Datenbank.

Aufruf (lokal oder im Docker Container):
    python -m vigilex.workers.ingest
    python -m vigilex.workers.ingest --product-code LZG --year 2024
    python -m vigilex.workers.ingest --product-code LZG --start 20230101 --end 20231231

Alle Produkt-Codes auf einmal (Phase-2-Vollimport):
    python -m vigilex.workers.ingest --all-products --year 2024

Umgebungsvariablen (aus .env via docker-compose):
    DATABASE_URL       -- PostgreSQL Connection String
    OPENFDA_API_KEY    -- optional, erhoeht Rate Limit von 1k auf 120k/Tag
"""

import argparse
import logging
import os
import sys
from datetime import date

from vigilex.data.maude_client import fetch_maude_by_daterange, upsert_maude_records
from vigilex.db.connection import get_connection

# ── Logging Setup ─────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("vigilex.workers.ingest")


# ── Produktcode-Konfiguration ─────────────────────────────────────────────────
#
# Die fuenf Produktcodes aus Notebook 01 / Capstone-Planung:
#   LZG -- Insulinpumpen (insulin pump)
#   QFG -- CGM Sensoren (continuous glucose monitor)
#   OYC -- Herzschrittmacher (pacemaker)
#   PKU -- Defibrillatoren (defibrillator)
#   FRN -- Beatmungsgeraete (ventilator)
#
# Fuer Phase 1 (Test-Pull): nur LZG, Jahr 2024.
# Fuer Phase 2 (Vollimport): alle fuenf, 2015-2024.

PRODUCT_CODES = {
    "LZG": "Insulinpumpen",
    "QFG": "CGM Sensoren",
    "OYC": "Herzschrittmacher",
    "PKU": "Defibrillatoren",
    "FRN": "Beatmungsgeraete",
}

# Batch-Groesse fuer DB-Commits
BATCH_SIZE = 500


# ── Kern-Logik ────────────────────────────────────────────────────────────────

def run_ingest(
    product_code: str,
    start_date: str,
    end_date: str,
    api_key: str = "",
) -> int:
    """
    Holt Records fuer einen Produktcode + Zeitraum und schreibt sie in die DB.

    Returns:
        Anzahl eingefuegter Records (Duplikate nicht gezaehlt)
    """
    logger.info(
        "Starte Ingestion | product_code=%s | %s bis %s",
        product_code, start_date, end_date,
    )

    batch_id = f"{product_code}_{start_date}_{end_date}"
    total_inserted = 0
    buffer: list[dict] = []

    conn = get_connection()

    try:
        for row in fetch_maude_by_daterange(
            product_code=product_code,
            start_date=start_date,
            end_date=end_date,
            api_key=api_key,
            batch_id=batch_id,
        ):
            buffer.append(row)

            # Batch-Commit wenn Puffer voll
            if len(buffer) >= BATCH_SIZE:
                inserted = upsert_maude_records(conn, buffer)
                total_inserted += inserted
                logger.info("Batch committed: %d Records | Total: %d", inserted, total_inserted)
                buffer.clear()

        # Rest-Puffer
        if buffer:
            inserted = upsert_maude_records(conn, buffer)
            total_inserted += inserted
            logger.info("Abschluss-Batch: %d Records | Total: %d", inserted, total_inserted)

    except Exception as exc:
        logger.error("Fehler waehrend Ingestion: %s", exc, exc_info=True)
        conn.rollback()
        raise
    finally:
        conn.close()

    logger.info(
        "Ingestion abgeschlossen | product_code=%s | %d Records eingefuegt",
        product_code, total_inserted,
    )
    return total_inserted


def run_full_ingest(years: list[int], api_key: str = "") -> None:
    """Vollimport aller 5 Produktcodes fuer eine Liste von Jahren."""
    for product_code, description in PRODUCT_CODES.items():
        for year in years:
            start = f"{year}0101"
            end   = f"{year}1231"
            logger.info("--- %s (%s) | Jahr %d ---", product_code, description, year)
            try:
                n = run_ingest(product_code, start, end, api_key)
                logger.info("OK: %d Records", n)
            except Exception as exc:
                logger.error("FEHLER bei %s/%d: %s", product_code, year, exc)
                continue


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SentinelAI MAUDE Ingestion Worker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  # Test-Pull: Insulinpumpen 2024
  python -m vigilex.workers.ingest --product-code LZG --year 2024

  # Eigener Zeitraum
  python -m vigilex.workers.ingest --product-code QFG --start 20230101 --end 20231231

  # Vollimport alle Produkte, 2020-2024
  python -m vigilex.workers.ingest --all-products --years 2020 2021 2022 2023 2024
        """
    )

    # Welche Produktcodes?
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument(
        "--product-code",
        choices=list(PRODUCT_CODES.keys()),
        help=f"Einzelner Produktcode: {', '.join(PRODUCT_CODES.keys())}",
    )
    grp.add_argument(
        "--all-products",
        action="store_true",
        help="Alle 5 Produktcodes importieren",
    )

    # Zeitraum
    time_grp = parser.add_mutually_exclusive_group()
    time_grp.add_argument(
        "--year",
        type=int,
        help="Einzelnes Jahr, z.B. 2024",
    )
    time_grp.add_argument(
        "--years",
        type=int,
        nargs="+",
        help="Mehrere Jahre, z.B. 2022 2023 2024",
    )
    time_grp.add_argument(
        "--start",
        help="Startdatum YYYYMMDD (zusammen mit --end)",
    )

    parser.add_argument(
        "--end",
        help="Enddatum YYYYMMDD (zusammen mit --start)",
    )

    args = parser.parse_args()

    api_key = os.environ.get("OPENFDA_API_KEY", "")
    current_year = date.today().year

    # Zeitraum bestimmen
    if args.year:
        start = f"{args.year}0101"
        end   = f"{args.year}1231"
        years = [args.year]
    elif args.years:
        start = f"{min(args.years)}0101"
        end   = f"{max(args.years)}1231"
        years = sorted(args.years)
    elif args.start and args.end:
        start = args.start
        end   = args.end
        years = None
    else:
        # Default: aktuelles Jahr
        logger.info("Kein Zeitraum angegeben -- nutze aktuelles Jahr %d", current_year)
        start = f"{current_year}0101"
        end   = f"{current_year}1231"
        years = [current_year]

    # Ausfuehren
    if args.all_products:
        target_years = years or [current_year]
        run_full_ingest(target_years, api_key)
    else:
        run_ingest(args.product_code, start, end, api_key)


if __name__ == "__main__":
    main()
