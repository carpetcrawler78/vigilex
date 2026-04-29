"""
SentinelAI (vigilex) -- openFDA MAUDE API client.

What is MAUDE?
    The FDA's Manufacturer and User Facility Device Experience (MAUDE) database
    contains reports of adverse events (problems, injuries, malfunctions) involving
    medical devices. It is publicly accessible via the openFDA REST API.

What does this module do?
    It provides two public functions:
        fetch_maude_by_daterange()  -- downloads records from the API (handles pagination)
        flatten_maude_record()      -- converts a raw API record to a flat database row

    A third function, upsert_maude_records(), writes those rows to PostgreSQL.

Background: why "flatten"?
    The MAUDE API returns deeply nested JSON objects. For example, device information
    is buried inside rec["device"][0]["brand_name"]. Flattening means extracting all
    the fields we care about and putting them in a simple key-value dictionary that
    maps directly to columns in the raw.maude_reports database table.

This module was derived from the exploration done in Notebook 01_openfda_maude.ipynb
and made production-ready by adding:
    - Logging instead of print() (so we can monitor it in docker)
    - Automatic retry on network errors (robust to temporary API failures)
    - Proper date parsing (API returns dates as "YYYYMMDD" strings)
    - Field mapping exactly matching the raw.maude_reports table schema
"""

import logging
import os
import time
from datetime import date, datetime
from typing import Any, Iterator

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# The base URL for all openFDA device adverse event queries.
BASE_URL = "https://api.fda.gov/device/event.json"

# openFDA limits each query to a maximum of 10,000 total results.
# We fetch them in pages of 100 (smaller than the maximum of 1000) because
# smaller pages are more resilient to timeouts on slow connections.
# MAX_SKIP + PAGE_SIZE must not exceed 10,000 (API hard limit).
PAGE_SIZE = 100   # records per HTTP request
MAX_SKIP  = 9900  # maximum value of the "skip" (offset) parameter

# Rate limiting:
# Without an API key: 1 request/second is safe.
# With an API key:    up to 240 requests/minute.
# We use a conservative 0.15s delay (about 6 req/s) which works with a key
# and is still polite without one.
SLEEP_BETWEEN_REQUESTS = 0.15  # seconds between API calls


# ---------------------------------------------------------------------------
# HTTP Session with automatic retry
# ---------------------------------------------------------------------------

def _make_session() -> requests.Session:
    """
    Create an HTTP session that automatically retries on transient errors.

    What is an HTTP session?
        A Session object reuses the same underlying TCP connection for multiple
        requests, which is faster than opening a new connection each time.

    What is a Retry adapter?
        It configures the session to automatically re-try failed requests.
        Parameters:
            total=5             -- retry up to 5 times before giving up
            backoff_factor=1.0  -- wait 1s, then 2s, 4s, 8s, 16s between retries
            status_forcelist    -- retry on these HTTP error codes:
                429 = Too Many Requests (we sent too many in a row)
                500 = Internal Server Error (FDA server glitch)
                502 = Bad Gateway
                503 = Service Unavailable (FDA server temporarily down)
                504 = Gateway Timeout
    """
    session = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    return session


# ---------------------------------------------------------------------------
# Date parsing helper
# ---------------------------------------------------------------------------

def _parse_fda_date(value: str | None) -> date | None:
    """
    Convert an openFDA date string to a Python date object.

    The openFDA API returns all dates as plain 8-digit strings in YYYYMMDD format,
    e.g. "20240315" means March 15, 2024. We parse these into proper Python date
    objects so PostgreSQL can store them in DATE columns.

    Returns None if the value is missing or malformed (rather than crashing),
    because many MAUDE records have incomplete date fields.
    """
    if not value:
        return None
    try:
        return datetime.strptime(str(value).strip(), "%Y%m%d").date()
    except ValueError:
        return None  # silently ignore unparseable dates


# ---------------------------------------------------------------------------
# flatten_maude_record
# ---------------------------------------------------------------------------

def flatten_maude_record(rec: dict[str, Any]) -> dict[str, Any]:
    """
    Convert a raw openFDA MAUDE API record into a flat dictionary.

    The flat dictionary has exactly the keys that match columns in the
    raw.maude_reports PostgreSQL table, so it can be inserted directly.

    Why is flattening necessary?
        The raw API record is a deeply nested structure. For example:
            rec["device"][0]["brand_name"]          -- device brand
            rec["patient"][0]["patient_age"]         -- patient age
            rec["mdr_text"][0]["text"]              -- the narrative text

        Each of these is a list (some records have multiple devices or patients).
        We always take the first element [0], which covers the vast majority of
        adverse event reports.

        Missing fields return None instead of raising a KeyError -- this is
        important because MAUDE data quality is highly variable.

    The key field for NLP (Module 2):
        "mdr_text" is the free-text adverse event narrative written by the reporter.
        This is what the MedDRA Coding Engine (Module 2) reads to assign a PT code.

    Args:
        rec: raw dict from the openFDA API (one element of the "results" list)

    Returns:
        Flat dict ready for insertion into raw.maude_reports.
    """
    # device[0] -- extract the first (primary) device from the devices list.
    # Most reports involve a single device, so index 0 is the right choice.
    devices = rec.get("device") or [{}]
    dev = devices[0] if devices else {}

    # patient[0] -- extract the first patient record.
    patients = rec.get("patient") or [{}]
    pat = patients[0] if patients else {}

    # mdr_text[0]["text"] -- the free-text narrative. This is the most
    # important field for our NLP pipeline. It describes what happened
    # in the reporter's own words.
    mdr_list = rec.get("mdr_text") or [{}]
    mdr = mdr_list[0] if mdr_list else {}

    # sequence_of_events -- a structured description of event sequence.
    # Sometimes found at the top level, sometimes nested inside patient.
    soe = rec.get("sequence_of_events_text") or pat.get("sequence_of_events_text")

    return {
        # Primary key -- openFDA's unique report number (e.g. "3012345678-2024-00001")
        "mdr_report_key":       rec.get("report_number"),

        # Report metadata -- dates and source information
        "date_received":        _parse_fda_date(rec.get("date_received")),
        "date_of_event":        _parse_fda_date(rec.get("date_of_event")),
        "report_source_code":   rec.get("report_source_code"),   # e.g. "M" = manufacturer
        "report_to_fda":        rec.get("report_to_fda"),

        # Device fields -- what device was involved?
        "device_name":          dev.get("generic_name"),          # e.g. "INSULIN PUMP"
        "device_brand_name":    dev.get("brand_name"),            # e.g. "OMNIPOD 5"
        "product_code":         dev.get("device_report_product_code"),  # e.g. "LZG"
        "manufacturer_name":    dev.get("manufacturer_d_name"),
        "model_number":         dev.get("model_number"),
        "lot_number":           dev.get("lot_number"),
        "device_age_text":      dev.get("device_age_text"),       # how old was the device?

        # Patient fields -- who was affected?
        "patient_sequence_number": pat.get("patient_sequence_number"),
        "date_of_birth":           pat.get("date_of_birth"),
        "patient_weight":          pat.get("weight"),
        "patient_age":             pat.get("patient_age"),
        "patient_sex":             pat.get("patient_sex"),
        "sequence_of_events":      soe,

        # The narrative text -- PRIMARY INPUT for Module 2 (MedDRA Coding Engine).
        # Default to empty string rather than None so we can filter easily with
        # "mdr_text <> ''" in SQL.
        "mdr_text":             mdr.get("text") or "",

        # Recall flags -- these start as False/0 and get updated later
        # when Notebook 02 joins the MAUDE data with the FDA Recall Database.
        "recalled_ever":        False,
        "recall_count":         0,

        # Bookkeeping fields set by the ingestion worker
        "data_source":          "openFDA_MAUDE",
        "api_batch_id":         None,   # filled in by fetch_maude_by_daterange()
    }


# ---------------------------------------------------------------------------
# fetch_maude_by_daterange
# ---------------------------------------------------------------------------

def fetch_maude_by_daterange(
    product_code: str,
    start_date: str,
    end_date: str,
    api_key: str = "",
    batch_id: str = "",
) -> Iterator[dict[str, Any]]:
    """
    Download MAUDE adverse event reports for a specific device type and date range.

    This is a generator function (uses "yield" instead of "return"), which means
    it produces records one at a time without loading all 10,000 into memory at once.
    The calling code (the ingest worker) can process each record as it arrives.

    What is pagination?
        The API cannot return 10,000 records in a single response -- it would be
        too slow. Instead it returns them in pages of up to 1,000 records each.
        We iterate through these pages using the "skip" parameter (like SQL OFFSET):
            Page 1: skip=0,   limit=100  -> records 1-100
            Page 2: skip=100, limit=100  -> records 101-200
            Page 3: skip=200, limit=100  -> records 201-300
            ...

    The 10,000 record limit:
        openFDA enforces skip + limit <= 10,000 regardless of how many records
        actually match your query. If a product code has 17,000 reports in a year,
        you can only get the first 10,000 per query. The solution is to split
        the time window into monthly chunks (each of which stays under 10,000).
        This is a known limitation documented in Notebook 01.

    Args:
        product_code: FDA 3-letter device code, e.g. "LZG" (insulin pumps).
                      Full list: LZG, QFG, OYC, PKU, FRN.
        start_date:   Start of date range in YYYYMMDD format, e.g. "20240101".
        end_date:     End of date range in YYYYMMDD format, e.g. "20241231".
        api_key:      Optional openFDA API key. Without a key, rate limit is
                      ~240 requests/day. With a key, it is ~120,000/day.
        batch_id:     Free-text label stored in the database for traceability,
                      e.g. "LZG_2024_run1". Useful for debugging and auditing.

    Yields:
        Dicts in the format of flatten_maude_record() -- ready for DB insertion.

    Example:
        for row in fetch_maude_by_daterange("LZG", "20240101", "20241231"):
            print(row["mdr_report_key"], row["mdr_text"][:80])
    """
    session = _make_session()
    # Prefer the passed-in key; fall back to environment variable; finally empty string.
    api_key = api_key or os.environ.get("OPENFDA_API_KEY", "")

    # Build the openFDA search query using Lucene syntax.
    # Spaces between terms mean AND in openFDA's query language.
    # The date range filter [start TO end] is an openFDA-specific syntax.
    search = (
        f"device.device_report_product_code:{product_code}"
        f" AND date_of_event:[{start_date} TO {end_date}]"
    )

    # Step 1: Send a probe request to find out how many records match the query.
    # We only request 1 record -- we just want the "meta.results.total" count.
    params: dict[str, Any] = {"search": search, "limit": 1}
    if api_key:
        params["api_key"] = api_key

    try:
        r = session.get(BASE_URL, params=params, timeout=30)
        r.raise_for_status()
        total = r.json().get("meta", {}).get("results", {}).get("total", 0)
    except Exception as exc:
        logger.error("Could not fetch total record count from openFDA: %s", exc)
        return  # exit the generator -- yields nothing

    logger.info(
        "fetch_maude | product_code=%s | %s to %s | total=%d records",
        product_code, start_date, end_date, total,
    )

    if total == 0:
        logger.info("No records found for this query.")
        return

    # Cap at the openFDA hard limit (skip + limit <= 10,000).
    # If total > 10,000, we log a warning so the operator knows to use
    # smaller date windows for a complete import.
    effective_total = min(total, MAX_SKIP + PAGE_SIZE)
    if total > effective_total:
        logger.warning(
            "openFDA limits retrieval to %d records (total in DB is %d). "
            "To import all records, split the date range into monthly windows.",
            effective_total, total,
        )

    fetched = 0
    skip = 0

    # Step 2: Paginate through all available records.
    while skip < effective_total:
        params = {
            "search": search,
            "limit": PAGE_SIZE,
            "skip":  skip,
        }
        if api_key:
            params["api_key"] = api_key

        try:
            r = session.get(BASE_URL, params=params, timeout=30)
            r.raise_for_status()
            batch = r.json().get("results", [])
        except requests.exceptions.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                # A 404 at this point means we have gone past the last page.
                logger.debug("404 at skip=%d -- end of pagination reached", skip)
                break
            logger.error("HTTP error at skip=%d: %s", skip, exc)
            break
        except Exception as exc:
            logger.error("Unexpected error at skip=%d: %s", skip, exc)
            break

        if not batch:
            # Empty page -- we have retrieved everything available.
            break

        # Yield each record in the page after flattening it.
        for raw_rec in batch:
            row = flatten_maude_record(raw_rec)
            row["api_batch_id"] = batch_id or f"{product_code}_{start_date}_{end_date}"
            yield row

        fetched += len(batch)
        skip += len(batch)
        logger.debug("Fetched %d / %d records so far", fetched, effective_total)
        time.sleep(SLEEP_BETWEEN_REQUESTS)  # be polite to the API

    logger.info("fetch_maude complete | %d records delivered", fetched)


# ---------------------------------------------------------------------------
# upsert_maude_records
# ---------------------------------------------------------------------------

def upsert_maude_records(conn, rows: list[dict[str, Any]]) -> int:
    """
    Insert a list of MAUDE records into the raw.maude_reports table.

    What is an upsert?
        "Upsert" = INSERT + UPDATE. In this case we use:
            INSERT INTO ... ON CONFLICT (mdr_report_key) DO NOTHING

        This means:
        - If the report does not exist yet: INSERT it normally.
        - If a report with the same mdr_report_key already exists: do nothing.

        This makes the operation idempotent -- you can run the same import
        twice without creating duplicate rows or raising errors. This is
        important for robustness: if the import crashes halfway through, you
        can restart it safely.

    What is execute_batch?
        psycopg2's execute_batch() sends multiple INSERT statements to the
        database in a single round-trip (in groups of page_size=200), which
        is much faster than inserting one row at a time.

    Args:
        conn: open psycopg2 connection (caller is responsible for closing it)
        rows: list of dicts from flatten_maude_record()

    Returns:
        Approximate number of rows inserted (duplicates count as 0, but
        psycopg2's execute_batch does not give an exact per-row count,
        so we return len(rows) as an approximation).
    """
    if not rows:
        return 0

    sql = """
        INSERT INTO raw.maude_reports (
            mdr_report_key,
            date_received,
            date_of_event,
            report_source_code,
            report_to_fda,
            device_name,
            device_brand_name,
            product_code,
            manufacturer_name,
            model_number,
            lot_number,
            device_age_text,
            patient_sequence_number,
            date_of_birth,
            patient_weight,
            patient_age,
            patient_sex,
            sequence_of_events,
            mdr_text,
            recalled_ever,
            recall_count,
            data_source,
            api_batch_id
        ) VALUES (
            %(mdr_report_key)s,
            %(date_received)s,
            %(date_of_event)s,
            %(report_source_code)s,
            %(report_to_fda)s,
            %(device_name)s,
            %(device_brand_name)s,
            %(product_code)s,
            %(manufacturer_name)s,
            %(model_number)s,
            %(lot_number)s,
            %(device_age_text)s,
            %(patient_sequence_number)s,
            %(date_of_birth)s,
            %(patient_weight)s,
            %(patient_age)s,
            %(patient_sex)s,
            %(sequence_of_events)s,
            %(mdr_text)s,
            %(recalled_ever)s,
            %(recall_count)s,
            %(data_source)s,
            %(api_batch_id)s
        )
        ON CONFLICT (mdr_report_key) DO NOTHING
    """

    with conn.cursor() as cur:
        import psycopg2.extras
        psycopg2.extras.execute_batch(cur, sql, rows, page_size=200)
        inserted = cur.rowcount  # -1 when execute_batch does not report exact count

    conn.commit()

    # execute_batch reports rowcount=-1 (unknown) when multiple statements were batched.
    # Fall back to len(rows) as an approximation.
    inserted = inserted if inserted >= 0 else len(rows)
    return inserted
