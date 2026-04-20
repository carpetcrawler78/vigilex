"""
SentinelAI (vigilex) -- openFDA MAUDE API client.

Zwei public Funktionen:
    fetch_maude_by_daterange()  -- holt Records von der API (paginiert)
    flatten_maude_record()      -- wandelt rohen API-Record in DB-Row um

Basiert auf Notebook 01_openfda_maude.ipynb, produktionsreif gemacht:
    - Logging statt print
    - Retry bei Netzwerkfehlern
    - Datum-Parsing (YYYYMMDD -> date)
    - Felder exakt auf raw.maude_reports gemappt
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

# ── Konstanten ────────────────────────────────────────────────────────────────

BASE_URL = "https://api.fda.gov/device/event.json"

# openFDA: max 1000 pro Request, max 10000 total per search via skip
# Quelle: https://open.fda.gov/apis/query-syntax/
PAGE_SIZE = 100   # kleiner als 1000 = robuster bei Timeouts
MAX_SKIP  = 9900  # openFDA limitiert skip+limit <= 10000

# Rate limiting: ohne Key 1 req/s, mit Key ~10 req/s
SLEEP_BETWEEN_REQUESTS = 0.15   # Sekunden


# ── HTTP Session mit Retry ────────────────────────────────────────────────────

def _make_session() -> requests.Session:
    """HTTP Session mit automatischem Retry bei transienten Fehlern."""
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


# ── Datum-Parsing ─────────────────────────────────────────────────────────────

def _parse_fda_date(value: str | None) -> date | None:
    """Wandelt openFDA Datums-Strings (YYYYMMDD) in Python date um."""
    if not value:
        return None
    try:
        return datetime.strptime(str(value).strip(), "%Y%m%d").date()
    except ValueError:
        return None


# ── flatten_maude_record ──────────────────────────────────────────────────────

def flatten_maude_record(rec: dict[str, Any]) -> dict[str, Any]:
    """
    Wandelt einen rohen openFDA MAUDE API-Record in ein flaches Dict um,
    das 1:1 auf die Spalten von raw.maude_reports passt.

    openFDA-Records sind tief verschachtelt:
        rec['device'][0]['brand_name']
        rec['patient'][0]['sequence_number_outcome'][0]
        rec['mdr_text'][0]['text']

    Wir nehmen immer das erste Element der Listen (Index 0).
    Fehlende Felder ergeben None -- kein KeyError, kein Crash.
    """
    # device[0] -- Geraetefeldner
    devices = rec.get("device") or [{}]
    dev = devices[0] if devices else {}

    # patient[0] -- Patientenfelder
    patients = rec.get("patient") or [{}]
    pat = patients[0] if patients else {}

    # mdr_text[0]['text'] -- das Freitext-Narrativ (Hauptfeld fuer NLP)
    mdr_list = rec.get("mdr_text") or [{}]
    mdr = mdr_list[0] if mdr_list else {}

    # sequence_of_events -- steht manchmal direkt im Record, manchmal im patient
    soe = rec.get("sequence_of_events_text") or pat.get("sequence_of_events_text")

    return {
        # Primaerschluessel -- openFDA report_number
        "mdr_report_key":       rec.get("report_number"),

        # Report-Metadaten
        "date_received":        _parse_fda_date(rec.get("date_received")),
        "date_of_event":        _parse_fda_date(rec.get("date_of_event")),
        "report_source_code":   rec.get("report_source_code"),
        "report_to_fda":        rec.get("report_to_fda"),

        # Geraetefelder
        "device_name":          dev.get("generic_name"),
        "device_brand_name":    dev.get("brand_name"),
        "product_code":         dev.get("device_report_product_code"),
        "manufacturer_name":    dev.get("manufacturer_d_name"),
        "model_number":         dev.get("model_number"),
        "lot_number":           dev.get("lot_number"),
        "device_age_text":      dev.get("device_age_text"),

        # Patientenfelder
        "patient_sequence_number": pat.get("patient_sequence_number"),
        "date_of_birth":           pat.get("date_of_birth"),
        "patient_weight":          pat.get("weight"),
        "patient_age":             pat.get("patient_age"),
        "patient_sex":             pat.get("patient_sex"),
        "sequence_of_events":      soe,

        # Narrativ-Text -- das ist was Module 2 (MedDRA Coding) verarbeitet
        "mdr_text":             mdr.get("text") or "",

        # Ingestion-Bookkeeping (wird vom Worker gesetzt)
        "recalled_ever":        False,
        "recall_count":         0,
        "data_source":          "openFDA_MAUDE",
        "api_batch_id":         None,   # Worker setzt das vor dem Insert
    }


# ── fetch_maude_by_daterange ──────────────────────────────────────────────────

def fetch_maude_by_daterange(
    product_code: str,
    start_date: str,
    end_date: str,
    api_key: str = "",
    batch_id: str = "",
) -> Iterator[dict[str, Any]]:
    """
    Holt MAUDE Adverse Event Reports fuer einen Produktcode + Zeitraum.

    Gibt Records als Iterator zurueck (kein grosses In-Memory-Array).
    Der Worker kann so direkt per Batch in die DB schreiben.

    Args:
        product_code:  FDA Produktcode, z.B. 'LZG' (Insulinpumpen), 'QFG' (CGM)
        start_date:    Format 'YYYYMMDD', z.B. '20240101'
        end_date:      Format 'YYYYMMDD', z.B. '20241231'
        api_key:       openFDA API Key (optional, erhoeht Rate Limit stark)
        batch_id:      Freitext-ID fuer Nachverfolgung (z.B. 'LZG_2024_run1')

    Yields:
        Dicts im Format von flatten_maude_record() -- DB-ready

    Beispiel:
        for row in fetch_maude_by_daterange('LZG', '20240101', '20241231'):
            print(row['mdr_report_key'], row['mdr_text'][:80])
    """
    session = _make_session()
    api_key = api_key or os.environ.get("OPENFDA_API_KEY", "")

    # openFDA Lucene-Syntax: device.device_report_product_code = LZG
    # AND date_of_event im Zeitfenster
    search = (
        f"device.device_report_product_code.exact:{product_code}"
        f"+AND+date_of_event:[{start_date}+TO+{end_date}]"
    )

    # Erst Total abfragen (1 Record reicht)
    params: dict[str, Any] = {"search": search, "limit": 1}
    if api_key:
        params["api_key"] = api_key

    try:
        r = session.get(BASE_URL, params=params, timeout=30)
        r.raise_for_status()
        total = r.json().get("meta", {}).get("results", {}).get("total", 0)
    except Exception as exc:
        logger.error("Konnte Total nicht abfragen: %s", exc)
        return

    logger.info(
        "fetch_maude | product_code=%s | %s bis %s | total=%d",
        product_code, start_date, end_date, total,
    )

    if total == 0:
        logger.info("Keine Records gefunden.")
        return

    # openFDA erlaubt max skip+limit = 10000
    effective_total = min(total, MAX_SKIP + PAGE_SIZE)
    if total > effective_total:
        logger.warning(
            "openFDA limitiert auf %d Records (total=%d). "
            "Fuer mehr Records: Zeitraum in kleinere Fenster aufteilen.",
            effective_total, total,
        )

    fetched = 0
    skip = 0

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
                # 404 = keine weiteren Seiten
                logger.debug("404 bei skip=%d -- Ende der Pagination", skip)
                break
            logger.error("HTTP-Fehler bei skip=%d: %s", skip, exc)
            break
        except Exception as exc:
            logger.error("Fehler bei skip=%d: %s", skip, exc)
            break

        if not batch:
            break

        for raw_rec in batch:
            row = flatten_maude_record(raw_rec)
            row["api_batch_id"] = batch_id or f"{product_code}_{start_date}_{end_date}"
            yield row

        fetched += len(batch)
        skip += len(batch)
        logger.debug("Fetched %d / %d", fetched, effective_total)
        time.sleep(SLEEP_BETWEEN_REQUESTS)

    logger.info("fetch_maude fertig | %d Records geliefert", fetched)


# ── upsert_maude_records ──────────────────────────────────────────────────────

def upsert_maude_records(conn, rows: list[dict[str, Any]]) -> int:
    """
    Schreibt eine Liste von flatten_maude_record()-Dicts in raw.maude_reports.

    Verwendet INSERT ... ON CONFLICT (mdr_report_key) DO NOTHING:
    -> Idempotent: Doppelter Import eines Records erzeugt kein Duplikat,
       keinen Fehler. Sicher zum wiederholten Ausfuehren.

    Returns:
        Anzahl der tatsaechlich eingefuegten Zeilen (Duplikate = 0)
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
        psycopg2 = __import__("psycopg2.extras", fromlist=["execute_batch"])
        import psycopg2.extras
        psycopg2.extras.execute_batch(cur, sql, rows, page_size=200)
        inserted = cur.rowcount  # -1 wenn execute_batch keine genaue Zahl liefert

    conn.commit()

    # execute_batch liefert rowcount=-1 (kein exakter Wert) -- Naherung:
    inserted = inserted if inserted >= 0 else len(rows)
    return inserted
