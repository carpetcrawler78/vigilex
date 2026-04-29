"""
SentinelAI (vigilex) -- MAUDE Ingestion Worker.

What is a "worker"?
    In software architecture, a worker is a long-running process that pulls
    work from a queue (or a data source) and processes it. This worker's job is
    to download MAUDE adverse event reports from the openFDA API and insert them
    into the PostgreSQL database.

What does this worker do, step by step?
    1. Reads configuration (which device type, which date range) from CLI arguments.
    2. Calls fetch_maude_by_daterange() to stream records from the openFDA API.
    3. Collects records into batches of 500.
    4. Calls upsert_maude_records() to write each batch to the database.
    5. Logs progress to stdout.

Why batches of 500?
    Inserting one record at a time would require 10,000 separate round-trips
    to the database for 10,000 records. Batching 500 at a time means only 20
    round-trips -- roughly 50x faster. 500 is small enough to not overload
    memory, and large enough to be efficient.

How to run this worker:
    python -m vigilex.workers.ingest --product-code LZG --year 2024
    python -m vigilex.workers.ingest --product-code LZG --start 20230101 --end 20231231
    python -m vigilex.workers.ingest --all-products --years 2020 2021 2022 2023 2024

Required environment variables (from .env via docker-compose):
    DATABASE_URL       -- PostgreSQL connection string
    OPENFDA_API_KEY    -- optional but strongly recommended; without it the API
                         rate limit is ~240 requests/day
"""

import argparse
import logging
import os
import sys
from datetime import date

from vigilex.data.maude_client import fetch_maude_by_daterange, upsert_maude_records
from vigilex.db.connection import get_connection


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
# We configure logging here (instead of each module doing it separately)
# because this is the entry point of the process. The format includes:
#   - timestamp (for understanding timing and performance)
#   - log level (INFO for normal operation, ERROR for problems)
#   - logger name (which module produced the message)
#   - the message itself

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("vigilex.workers.ingest")


# ---------------------------------------------------------------------------
# Device product code configuration
# ---------------------------------------------------------------------------
# The five FDA product codes targeted by SentinelAI.
# These codes identify specific device categories in the MAUDE database.
# Each code was selected because it represents a high-risk implantable or
# life-critical medical device category.
#
# To find other product codes, see: https://www.fda.gov/medical-devices/
# or use Notebook 01 (scripts/explore_product_codes.py for the full list).

PRODUCT_CODES = {
    "LZG": "Insulin pumps",
    "QFG": "CGM sensors (continuous glucose monitors)",
    "OYC": "Pacemakers",
    "PKU": "Defibrillators (implantable cardioverter-defibrillators)",
    "FRN": "Ventilators",
}

# Number of records to accumulate before writing to the database.
# 500 is a balance between memory efficiency and database round-trip overhead.
BATCH_SIZE = 500


# ---------------------------------------------------------------------------
# Core ingestion logic
# ---------------------------------------------------------------------------

def run_ingest(
    product_code: str,
    start_date: str,
    end_date: str,
    api_key: str = "",
) -> int:
    """
    Fetch all MAUDE records for one product code and date range, write to DB.

    This function ties together the two main building blocks:
        fetch_maude_by_daterange() -- downloads records from openFDA (streaming)
        upsert_maude_records()     -- writes batches of records to PostgreSQL

    The streaming approach (generator + buffer) means memory usage stays flat
    even when downloading 10,000 records. We never hold the entire dataset
    in memory -- only the current batch of 500 records.

    Args:
        product_code: e.g. "LZG" for insulin pumps
        start_date:   YYYYMMDD string, e.g. "20240101"
        end_date:     YYYYMMDD string, e.g. "20241231"
        api_key:      openFDA API key (read from env if not passed)

    Returns:
        Total number of new records inserted (duplicates not counted).
    """
    logger.info(
        "Starting ingestion | product_code=%s | %s to %s",
        product_code, start_date, end_date,
    )

    # Create a human-readable batch identifier for traceability.
    # This gets stored in the api_batch_id column of raw.maude_reports,
    # so we can later identify which run produced which records.
    batch_id = f"{product_code}_{start_date}_{end_date}"
    total_inserted = 0
    buffer: list[dict] = []

    # Open a single database connection for the entire ingestion run.
    # The connection is closed in the finally block, even if an error occurs.
    conn = get_connection()

    try:
        # Iterate over records as they arrive from the API (streaming generator).
        # fetch_maude_by_daterange() yields one record at a time -- it does not
        # load all records into memory before returning.
        for row in fetch_maude_by_daterange(
            product_code=product_code,
            start_date=start_date,
            end_date=end_date,
            api_key=api_key,
            batch_id=batch_id,
        ):
            buffer.append(row)

            # Once the buffer is full, flush it to the database.
            # This is a "mini-transaction" -- 500 records committed at a time.
            if len(buffer) >= BATCH_SIZE:
                inserted = upsert_maude_records(conn, buffer)
                total_inserted += inserted
                logger.info(
                    "Batch committed: %d records | Running total: %d",
                    inserted, total_inserted
                )
                buffer.clear()  # reset buffer for next batch

        # After the loop, flush any remaining records that did not fill a full batch.
        # (e.g. if total records = 1050, the last 50 would still be in the buffer)
        if buffer:
            inserted = upsert_maude_records(conn, buffer)
            total_inserted += inserted
            logger.info(
                "Final batch committed: %d records | Total: %d",
                inserted, total_inserted
            )

    except Exception as exc:
        # If anything goes wrong, roll back any uncommitted changes and re-raise.
        # The rollback ensures we do not leave the database in a partial state.
        logger.error("Error during ingestion: %s", exc, exc_info=True)
        conn.rollback()
        raise
    finally:
        # Always close the connection, whether we succeeded or failed.
        conn.close()

    logger.info(
        "Ingestion complete | product_code=%s | %d records inserted",
        product_code, total_inserted,
    )
    return total_inserted


def run_full_ingest(years: list[int], api_key: str = "") -> None:
    """
    Run the full import for all 5 device product codes across multiple years.

    Used for the complete historical import (2015-2024) that will provide
    enough data for meaningful signal detection in Module 3. Errors on
    individual product code / year combinations are logged and skipped --
    the rest of the import continues.

    Args:
        years:   List of years to import, e.g. [2020, 2021, 2022, 2023, 2024]
        api_key: openFDA API key (passed through to run_ingest)
    """
    for product_code, description in PRODUCT_CODES.items():
        for year in years:
            start = f"{year}0101"  # January 1st of the year
            end   = f"{year}1231"  # December 31st of the year
            logger.info(
                "--- %s (%s) | Year %d ---",
                product_code, description, year
            )
            try:
                n = run_ingest(product_code, start, end, api_key)
                logger.info("OK: %d records imported", n)
            except Exception as exc:
                # Log the error but continue with the next product code / year.
                # A single failure should not abort the entire multi-year import.
                logger.error(
                    "FAILED for %s/%d: %s -- continuing with next.",
                    product_code, year, exc
                )
                continue


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

def main():
    """
    Parse command-line arguments and run the appropriate ingestion mode.

    The CLI is designed around two questions:
        1. Which device type(s)? (--product-code LZG | --all-products)
        2. Which time period?    (--year 2024 | --years 2020 2021 | --start/--end)
    """
    parser = argparse.ArgumentParser(
        description="SentinelAI MAUDE Ingestion Worker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test pull: insulin pumps, full year 2024
  python -m vigilex.workers.ingest --product-code LZG --year 2024

  # Custom date range for CGM sensors
  python -m vigilex.workers.ingest --product-code QFG --start 20230101 --end 20231231

  # Full historical import: all 5 device types, 2020-2024
  python -m vigilex.workers.ingest --all-products --years 2020 2021 2022 2023 2024
        """
    )

    # Mutually exclusive group: either specify one product code OR request all five.
    # "mutually exclusive" means you cannot use both flags at the same time.
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument(
        "--product-code",
        choices=list(PRODUCT_CODES.keys()),
        help=f"Single device product code: {', '.join(PRODUCT_CODES.keys())}",
    )
    grp.add_argument(
        "--all-products",
        action="store_true",
        help="Import all 5 device product codes",
    )

    # Time range: three options, mutually exclusive
    time_grp = parser.add_mutually_exclusive_group()
    time_grp.add_argument(
        "--year",
        type=int,
        help="Single calendar year, e.g. 2024",
    )
    time_grp.add_argument(
        "--years",
        type=int,
        nargs="+",  # "+" means: one or more values
        help="Multiple years, e.g. --years 2022 2023 2024",
    )
    time_grp.add_argument(
        "--start",
        help="Start date in YYYYMMDD format (use together with --end)",
    )

    parser.add_argument(
        "--end",
        help="End date in YYYYMMDD format (use together with --start)",
    )

    args = parser.parse_args()

    # Read API key from environment (set in .env / docker-compose)
    api_key = os.environ.get("OPENFDA_API_KEY", "")
    current_year = date.today().year

    # Resolve the time range from the CLI arguments
    if args.year:
        # Single year: full January 1 to December 31
        start = f"{args.year}0101"
        end   = f"{args.year}1231"
        years = [args.year]
    elif args.years:
        # Multiple specific years
        start = f"{min(args.years)}0101"
        end   = f"{max(args.years)}1231"
        years = sorted(args.years)
    elif args.start and args.end:
        # Custom date range
        start = args.start
        end   = args.end
        years = None
    else:
        # Default: current year (useful for scheduled daily/weekly runs)
        logger.info(
            "No time range specified -- defaulting to current year %d",
            current_year
        )
        start = f"{current_year}0101"
        end   = f"{current_year}1231"
        years = [current_year]

    # Execute the ingestion
    if args.all_products:
        target_years = years or [current_year]
        run_full_ingest(target_years, api_key)
    else:
        run_ingest(args.product_code, start, end, api_key)


if __name__ == "__main__":
    main()
