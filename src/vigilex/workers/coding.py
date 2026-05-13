"""
SentinelAI (vigilex) -- MedDRA Coding Worker.

This is the orchestrator: it pulls MAUDE reports from the database queue,
runs each one through the three-stage coding pipeline, and writes the results
back to processed.coding_results.

The three-stage pipeline:
    Stage 1 -- HybridSearcher (BM25 + Vector + RRF)
        Input:  mdr_text (free-text adverse event narrative)
        Output: Top-20 candidate MedDRA PTs (by fused rank score)
        Time:   ~500ms per report (mostly the vector embedding step)

    Stage 2 -- CrossEncoderReranker (MiniLM-L-6)
        Input:  Top-20 candidates from Stage 1
        Output: Top-5 candidates (reranked by joint (query, PT) relevance)
        Time:   ~20ms per report (fast because only 20 pairs)

    Stage 3 -- LLMCoder (Ollama llama3.2:3b)
        Input:  Top-5 candidates + full narrative
        Output: Final PT code + confidence score + rationale text
        Time:   ~5-15s per report (LLM inference on CPU)

The confidence formula:
    final_confidence = 0.3 * sigmoid(crossencoder_score)
                     + 0.7 * llm_confidence

    Why a weighted combination?
        - sigmoid(crossencoder_score) normalises the raw CrossEncoder logit
          (which has no fixed range) to a 0-1 value. Weight 0.3 gives it
          30% influence on the final score.
        - llm_confidence is the LLM's self-reported certainty (0-1). Weight 0.7
          gives it 70% influence. The LLM has access to the full narrative and
          clinical reasoning, so its estimate is more informative.
        - Cases with final_confidence < 0.5 are flagged for human review.

Worker design:
    - Models are loaded ONCE at startup (PubMedBERT + CrossEncoder = several seconds)
    - A fresh DB connection is opened per batch (self-healing against SSH tunnel restarts)
    - Each report is processed independently: one failure does not abort the batch
    - A polling loop checks for uncoded reports every 60s when the queue is empty

Usage (from vigilex repo root, with SSH tunnels open):
    python -m vigilex.workers.coding                       # continuous polling loop
    python -m vigilex.workers.coding --once                # single pass, then exit
    python -m vigilex.workers.coding --batch-size 50       # 50 reports per batch
    python -m vigilex.workers.coding --product-code LZG    # only insulin pump reports
    python -m vigilex.workers.coding --skip-llm --once     # Stage 1+2 only (no Ollama)

Required SSH tunnels for local development:
    ssh -L 5432:localhost:5432 -L 11434:localhost:11434 cap@46.225.109.99
"""

import argparse
import logging
import math
import os
import sys
import time
from collections import deque        # Rolling-Window fuer ntfy-Statistik
from dataclasses import dataclass
from typing import Optional

import psycopg2
import psycopg2.extras
import requests                       # ntfy.sh Push-Notifications

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from vigilex.coding.hybrid_search import HybridSearcher, EmbeddingModel
from vigilex.coding.reranker import CrossEncoderReranker
from vigilex.coding.llm_coder import LLMCoder
from vigilex.db.connection import get_connection


# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("vigilex.workers.coding")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_BATCH_SIZE   = 25    # reports fetched from DB per round-trip
DEFAULT_POLL_SECS    = 60    # seconds to sleep when the queue is empty
HYBRID_TOP_K         = 20    # candidates from Stage 1 -> Stage 2
RERANK_TOP_K         = 5     # candidates from Stage 2 -> Stage 3
CONFIDENCE_THRESHOLD = 0.5   # final_confidence < 0.5 -> flagged for human review

# Model version tag stored in each coding result row.
# Increment this (e.g. "pipeline_v2") when the pipeline components change,
# so you can identify which version produced which results in the database.
MODEL_VERSION = os.getenv("MODEL_VERSION", "pipeline_v1")

# ---------------------------------------------------------------------------
# Strict-Mode -- "fail-fast in dev, fail-soft in prod"
# Bei VIGILEX_STRICT=true wird der Worker bei LLM-Fehler abgebrochen statt
# stillschweigend in den Fallback-Pfad zu fallen. Konsistent mit
# llm_coder.py STRICT_MODE. Siehe CLAUDE.md "Kritischer Befund 2026-05-13".
# ---------------------------------------------------------------------------
STRICT_MODE = os.environ.get("VIGILEX_STRICT", "false").lower() == "true"

# ---------------------------------------------------------------------------
# ntfy.sh Push-Notifications fuer Worker-Monitoring
# Topic z.B. "sentinelai-cap-progress" -- Push-Notifications aufs Handy.
# Alle NTFY_BATCH_SIZE Records eine Summary; Sonder-Alert bei Non-Konformitaet.
# Leer = ntfy deaktiviert.
# ---------------------------------------------------------------------------
NTFY_TOPIC = os.environ.get("NTFY_TOPIC", "")
NTFY_BATCH_SIZE = int(os.environ.get("NTFY_BATCH_SIZE", "100"))
NTFY_FALLBACK_THRESHOLD_PCT = 10    # in Prod-Mode: Alert wenn >10% Fallback in letzten N


# ---------------------------------------------------------------------------
# Sigmoid helper
# ---------------------------------------------------------------------------

def sigmoid(x: float) -> float:
    """
    Map any real number to the range (0, 1).

    sigmoid(x) = 1 / (1 + e^(-x))

    Why do we need this?
        The CrossEncoder outputs a raw relevance logit -- a number that can be
        any real value (e.g. -5.2, 2.3, 8.1). There is no fixed range.
        We apply sigmoid to convert it to a probability-like value in (0, 1)
        so it can be combined with the LLM's confidence (which is already 0-1)
        in the final confidence formula.

    Examples:
        sigmoid(-10) = 0.000045   (very low confidence)
        sigmoid(0)   = 0.5        (uncertain)
        sigmoid(5)   = 0.993      (high confidence)
        sigmoid(10)  = 0.999955   (very high confidence)
    """
    return 1.0 / (1.0 + math.exp(-x))


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def fetch_uncoded_reports(
    conn,
    batch_size: int,
    product_code: Optional[str] = None,
    limit: Optional[int] = None,
) -> list[dict]:
    """
    Fetch up to batch_size MAUDE reports that have not been coded yet.

    How do we know which reports are uncoded?
        We use a NOT EXISTS subquery: a report is "uncoded" if there is no row
        in processed.coding_results with the same mdr_report_key.

        This is the standard SQL pattern for "anti-join" -- finding rows in
        table A that have no corresponding row in table B.

    Why newest-first (ORDER BY date_received DESC)?
        We prioritise recent reports because they are more clinically relevant
        for signal detection. Older reports will be processed later.
        NULLS LAST ensures reports with no date are processed last.

    Why NOT EXISTS instead of a LEFT JOIN ... WHERE NULL?
        NOT EXISTS short-circuits: as soon as it finds a matching coding_result,
        it stops searching. This is typically more efficient than a LEFT JOIN
        which must compute the full join before filtering.

    Args:
        conn:         open psycopg2 connection
        batch_size:   maximum number of reports to return
        product_code: optional filter (e.g. "LZG" for insulin pumps only)
        limit:        caller-side cap on total reports; not applied here --
                      the caller is responsible for respecting this limit

    Returns:
        List of dicts with keys: mdr_report_key, mdr_text, product_code, date_received.
    """
    # Conditionally add a product code filter to the query
    product_filter = "AND r.product_code = %(product_code)s" if product_code else ""

    sql = f"""
        SELECT r.mdr_report_key, r.mdr_text, r.product_code, r.date_received
        FROM raw.maude_reports r
        WHERE r.mdr_text IS NOT NULL
          AND r.mdr_text <> ''
          AND NOT EXISTS (
              SELECT 1
              FROM processed.coding_results c
              WHERE c.mdr_report_key = r.mdr_report_key
          )
          {product_filter}
        ORDER BY r.date_received DESC NULLS LAST
        LIMIT %(batch_size)s
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, {"batch_size": batch_size, "product_code": product_code})
        return [dict(row) for row in cur.fetchall()]


def write_coding_result(conn, result: dict) -> None:
    """
    Insert one coding result row into processed.coding_results.

    ON CONFLICT DO NOTHING: if a report was already coded (e.g. by a parallel
    worker or a previous run), we silently skip it. This makes the worker safe
    to run in multiple instances simultaneously.

    The caller is responsible for calling conn.commit() after this function.
    We commit per-report (not per-batch) so that a failure midway through a batch
    does not lose all preceding work.

    Args:
        conn:   open psycopg2 connection (not yet committed)
        result: dict with exactly the keys listed in the SQL below
    """
    sql = """
        INSERT INTO processed.coding_results (
            mdr_report_key,
            pt_code, pt_name, soc_name,
            vector_similarity, crossencoder_score, llm_confidence, final_confidence,
            model_version, coded_at
        )
        VALUES (
            %(mdr_report_key)s,
            %(pt_code)s, %(pt_name)s, %(soc_name)s,
            %(vector_similarity)s, %(crossencoder_score)s,
            %(llm_confidence)s, %(final_confidence)s,
            %(model_version)s, NOW()
        )
        ON CONFLICT DO NOTHING
    """
    with conn.cursor() as cur:
        cur.execute(sql, result)


def count_pending(conn, product_code: Optional[str] = None) -> int:
    """
    Return the number of uncoded MAUDE reports currently in the queue.

    Used by the polling loop to decide whether to process or sleep.
    Uses the same NOT EXISTS logic as fetch_uncoded_reports() for consistency.

    Args:
        conn:         open psycopg2 connection
        product_code: optional filter to count only one device type

    Returns:
        Integer count of uncoded reports.
    """
    product_filter = "AND r.product_code = %(product_code)s" if product_code else ""
    sql = f"""
        SELECT COUNT(*)
        FROM raw.maude_reports r
        WHERE r.mdr_text IS NOT NULL
          AND r.mdr_text <> ''
          AND NOT EXISTS (
              SELECT 1 FROM processed.coding_results c
              WHERE c.mdr_report_key = r.mdr_report_key
          )
          {product_filter}
    """
    with conn.cursor() as cur:
        cur.execute(sql, {"product_code": product_code})
        return cur.fetchone()[0]


# ---------------------------------------------------------------------------
# Core pipeline: one report through all three stages
# ---------------------------------------------------------------------------

def code_report(
    report: dict,
    searcher: HybridSearcher,
    reranker: CrossEncoderReranker,
    coder: Optional[LLMCoder],
) -> dict:
    """
    Run one MAUDE report through the complete three-stage coding pipeline.

    Returns a dict ready for insertion into processed.coding_results.
    Never raises -- errors at any stage result in a fallback/partial result.

    Stage 1 -- Hybrid Search:
        Takes the narrative text, encodes the first sentence with PubMedBERT,
        runs BM25 and vector search in parallel, fuses with RRF -> Top-20 candidates.

    Stage 2 -- CrossEncoder Reranking:
        Scores each of the 20 (narrative, PT_name) pairs with MiniLM-L-6 -> Top-5.

    Stage 3 -- LLM Coding (optional):
        Sends the narrative + Top-5 candidates to Ollama llama3.2.
        The LLM picks the best PT and assigns a confidence score.
        If skipped (--skip-llm) or failed, Stage 2's top result is used.

    Final confidence formula:
        WITH LLM:    0.3 * sigmoid(ce_score) + 0.7 * llm_confidence
        WITHOUT LLM: sigmoid(ce_score)  (sigmoid of raw CrossEncoder logit only)

    Args:
        report:   dict from fetch_uncoded_reports() -- contains mdr_report_key, mdr_text
        searcher: HybridSearcher instance (models loaded, DB connection injected per batch)
        reranker: CrossEncoderReranker instance
        coder:    LLMCoder instance, or None if --skip-llm was passed

    Returns:
        Dict with all columns needed for processed.coding_results.
    """
    narrative  = report["mdr_text"]
    report_key = report["mdr_report_key"]

    # -- Stage 1: Hybrid Search -----------------------------------------------
    candidates = searcher.search(narrative, top_k=HYBRID_TOP_K)

    if not candidates:
        # Edge case: no MedDRA terms matched at all (very short or uninformative narrative)
        logger.warning("Stage 1 returned no candidates for report %s", report_key)
        return _fallback_result(report_key, reason="no_candidates")

    # -- Stage 2: CrossEncoder Reranking --------------------------------------
    reranked = reranker.rerank(narrative, candidates, top_k=RERANK_TOP_K)

    if not reranked:
        logger.warning("Stage 2 returned no reranked results for %s", report_key)
        return _fallback_result(report_key, reason="no_reranked")

    # Top CrossEncoder candidate -- used as the final answer if Stage 3 fails or is skipped
    top_ce = reranked[0]
    ce_score_norm = sigmoid(top_ce.crossencoder_score)  # normalise to [0, 1]

    # Extract the best cosine similarity for the top CE result (for the DB record)
    vector_sim = top_ce.cosine_sim if top_ce.cosine_sim is not None else 0.0

    # -- Stage 3: LLM Coding (optional) ----------------------------------------
    llm_confidence = None
    final_pt_code  = top_ce.pt_code
    final_pt_name  = top_ce.pt_name
    final_soc_name = top_ce.soc_name

    if coder is not None:
        try:
            llm_result     = coder.code(narrative, reranked)
            llm_confidence = llm_result.confidence    # float ODER None (Prod-Fallback)
            final_pt_code  = llm_result.pt_code
            final_pt_name  = llm_result.pt_name
            # soc_name comes from the LLM prompt which sourced it from Stage 2
            # but the LLM does not reliably echo it back -- keep Stage 2's soc_name
        except Exception as exc:
            if STRICT_MODE:
                # Strict: Bug nicht in Logs verstecken, Worker bricht ab.
                # Siehe CLAUDE.md Befund 13.05 -- so haetten wir es frueher gemerkt.
                logger.error(
                    "STRICT MODE: Stage 3 raised for %s -- re-raising to abort worker.",
                    report_key,
                )
                raise
            # Prod-Mode: code() sollte normalerweise NICHT raisen (handhabt
            # interne Fehler selbst). Falls doch eine unerwartete Exception
            # durchschlaegt -> warnen und Stage-2-Pfad nehmen.
            logger.warning(
                "Stage 3 unexpected exception for %s (%s) -- using Stage 2 top result",
                report_key, exc,
            )
            # llm_confidence remains None -> "skip-llm-Pfad" weiter unten

    # -- Final confidence computation -- drei-Zweig-Logik ---------------------
    # Drei semantisch unterschiedliche Faelle:
    #   A) coder is None (--skip-llm): bewusste Entscheidung, Stage 2 only ist Ziel
    #      -> final = sigmoid(CE) ist korrekt
    #   B) coder is not None aber llm_confidence is None: LLM-Fallback
    #      (coder.code() hat None zurueckgegeben weil LLM ausfiel)
    #      -> final = None (NULL in DB, konsistent: "kein Wert ermittelt")
    #   C) llm_confidence is float: normaler LLM-Pfad
    #      -> final = gewichtete Kombination
    if coder is None:
        # Fall A -- --skip-llm Mode
        final_confidence = ce_score_norm
    elif llm_confidence is None:
        # Fall B -- LLM-Fallback (echter Failure-Pfad)
        final_confidence = None
    else:
        # Fall C -- normaler Pfad mit gewichteter Kombination
        final_confidence = 0.3 * ce_score_norm + 0.7 * llm_confidence

    return {
        "mdr_report_key":    report_key,
        "pt_code":           final_pt_code,
        "pt_name":           final_pt_name,
        "soc_name":          final_soc_name,
        "vector_similarity": round(vector_sim, 6),
        "crossencoder_score": round(top_ce.crossencoder_score, 6),
        "llm_confidence":    round(llm_confidence, 6) if llm_confidence is not None else None,
        "final_confidence":  round(final_confidence, 6) if final_confidence is not None else None,
        "model_version":     MODEL_VERSION,
    }


def _fallback_result(report_key: str, reason: str = "unknown") -> dict:
    """
    Create a minimal result row for reports where the pipeline produced no output.

    Why store a fallback instead of skipping?
        If we skip, the NOT EXISTS query in fetch_uncoded_reports() will keep
        returning the same failing report forever. Storing a fallback with
        final_confidence=0 means the report is considered "processed" and
        won't be fetched again. It will appear in quality dashboards as
        a case needing human review.

    Args:
        report_key: mdr_report_key of the failed report
        reason:     short label for the failure cause (logged for debugging)
    """
    logger.warning("Writing fallback result for %s (reason: %s)", report_key, reason)
    return {
        "mdr_report_key":    report_key,
        "pt_code":           None,   # no PT could be assigned
        "pt_name":           None,
        "soc_name":          None,
        "vector_similarity": None,
        "crossencoder_score": None,
        "llm_confidence":    None,
        "final_confidence":  None,   # NULL in DB -- konsistent mit "kein Wert ermittelt"
        "model_version":     MODEL_VERSION + "_fallback",
    }


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def load_pipeline(
    skip_llm: bool = False,
    use_groq: bool = False,
) -> tuple[HybridSearcher, CrossEncoderReranker, Optional[LLMCoder]]:
    """
    Load all three pipeline components once at startup.

    Loading PubMedBERT and the CrossEncoder from disk takes several seconds each.
    By loading them once here, we pay this startup cost only once -- all subsequent
    batches reuse the same model objects in memory.

    DB connections are NOT created here. The HybridSearcher's connection is
    replaced per batch in run_batch_loop() using _update_searcher_conn().
    This ensures we are resilient to SSH tunnel restarts between batches.

    Args:
        skip_llm: If True, skip initialising the LLMCoder (Stage 3). Useful when
                  the Ollama SSH tunnel is not open or for quick testing.

    Returns:
        (searcher, reranker, coder) -- coder is None when skip_llm=True.
    """
    logger.info("Loading PubMedBERT embedding model (used by Stage 1 vector search)...")
    t0 = time.time()
    embedding_model = EmbeddingModel()
    logger.info(
        "PubMedBERT loaded on %s in %.1fs",
        embedding_model.device, time.time() - t0
    )

    # HybridSearcher needs a DB connection to run SQL queries.
    # We open an init connection here just to satisfy the constructor.
    # The actual per-batch connections are opened in run_batch_loop().
    conn_init = get_connection()
    searcher = HybridSearcher(
        conn=conn_init,
        embedding_model=embedding_model,
        candidate_pool=100,  # retrieve 100 candidates per arm before RRF fusion
    )
    conn_init.close()  # close immediately; run_batch_loop will inject fresh connections

    logger.info("Loading CrossEncoder reranker (Stage 2, MiniLM-L-6)...")
    t0 = time.time()
    reranker = CrossEncoderReranker()
    logger.info("CrossEncoder loaded in %.1fs", time.time() - t0)

    coder = None
    if not skip_llm:
        if use_groq:
            # WARNING: Groq sends narratives to an external API.
            # Acceptable for benchmarking/capstone dev only -- NOT for production.
            groq_api_key = os.getenv("GROQ_API_KEY")
            if not groq_api_key:
                logger.error(
                    "--groq requested but GROQ_API_KEY is not set. "
                    "Export GROQ_API_KEY=<key> and retry."
                )
                sys.exit(1)
            logger.warning(
                "=== GROQ BACKEND ACTIVE (EXPERIMENTAL) ==="
                " Narratives will be sent to Groq's external API."
                " NOT for production use. ==="
            )
            coder = LLMCoder(
                confidence_threshold=CONFIDENCE_THRESHOLD,
                use_groq=True,
                groq_api_key=groq_api_key,
            )
        else:
            # LLMCoder liest OLLAMA_BASE_URL selbst aus dem env.
            # Wenn nicht gesetzt: RuntimeError (kein stiller localhost-Default mehr).
            # Vorher hatte coding.py die env-var unter falschem Namen ("OLLAMA_URL")
            # gelesen -- inkonsistent mit docker-compose.yml ("OLLAMA_BASE_URL").
            # Das war Teil des Befunds vom 13.05.
            ollama_url_for_log = os.environ.get("OLLAMA_BASE_URL", "<not set -- will raise>")
            logger.info("Initialising LLM coder (Stage 3, Ollama at %s)...", ollama_url_for_log)
            coder = LLMCoder(
                confidence_threshold=CONFIDENCE_THRESHOLD,
            )
            # Connectivity verification passiert in LLMCoder._check_connection().
            # Im Strict-Mode raised das hart -- Worker startet nicht bei kaputtem Setup.

    return searcher, reranker, coder


def notify_progress(
    total_coded: int,
    rate_per_min: float,
    rolling: list[dict],
) -> None:
    """
    Push progress summary to ntfy.sh after every NTFY_BATCH_SIZE records.

    Sendet eine Zusammenfassung der letzten N Records aufs Handy:
      - Cumulative count
      - Throughput (rec/min)
      - Fallback-Quote in der letzten N
      - Mean/Median der echten LLM-Confidence (None ueberspringen)
      - Sonder-Alert wenn non-konform:
          * Strict-Mode: jeder Fallback > 0 ist Violation
          * Prod-Mode: >NTFY_FALLBACK_THRESHOLD_PCT in letzter N

    No-op wenn NTFY_TOPIC nicht gesetzt ist.
    """
    if not NTFY_TOPIC:
        return  # ntfy deaktiviert -- nichts tun

    if not rolling:
        return  # keine Daten zum Reporten

    # Statistik ueber rolling-window berechnen
    n_total = len(rolling)
    n_fallback = sum(1 for r in rolling if r["is_fallback"])
    n_real = n_total - n_fallback
    fallback_pct = (n_fallback / n_total * 100) if n_total > 0 else 0.0

    # Mean/Median nur ueber echte LLM-Confidence (None ueberspringen)
    real_conf = [
        r["llm_confidence"] for r in rolling
        if r["llm_confidence"] is not None
    ]
    if real_conf:
        mean_conf = sum(real_conf) / len(real_conf)
        sorted_conf = sorted(real_conf)
        median_conf = sorted_conf[len(sorted_conf) // 2]
        conf_str = f"mean {mean_conf:.2f} | median {median_conf:.2f}"
    else:
        conf_str = "no real LLM data"

    # Non-Konform-Check -- Sonder-Alert-Prefix
    if STRICT_MODE and n_fallback > 0:
        # Sollte im Strict-Mode unmoeglich sein (Worker wuerde abbrechen)
        prefix = "[STRICT VIOLATION] "
        priority = "urgent"
    elif fallback_pct > NTFY_FALLBACK_THRESHOLD_PCT:
        prefix = f"[HIGH FALLBACK {fallback_pct:.0f}%] "
        priority = "high"
    else:
        prefix = ""
        priority = "default"

    msg = (
        f"{prefix}"
        f"Total: {total_coded} | "
        f"{rate_per_min:.1f} rec/min\n"
        f"Last {n_total}: {n_real} real / {n_fallback} fallback ({fallback_pct:.0f}%)\n"
        f"{conf_str}"
    )

    # Push an ntfy.sh -- 5s Timeout, kein blocking falls ntfy nicht erreichbar
    try:
        requests.post(
            f"https://ntfy.sh/{NTFY_TOPIC}",
            data=msg.encode("utf-8"),
            headers={
                "Title": f"SentinelAI Coding @ {total_coded}",
                "Priority": priority,
                "Tags": "robot,chart_with_upwards_trend",
            },
            timeout=5,
        )
    except Exception as exc:
        # ntfy-Fehler darf den Worker nicht stoppen -- nur loggen
        logger.warning("ntfy push failed: %s", exc)


def _update_searcher_conn(searcher: HybridSearcher, conn) -> None:
    """
    Inject a fresh DB connection into an existing HybridSearcher instance.

    Why do we need to do this?
        HybridSearcher stores a psycopg2 connection as self.conn and uses it
        for every BM25 and vector SQL query. If the SSH tunnel drops and comes
        back, the old connection is broken. Opening a fresh connection per batch
        and injecting it here ensures the searcher always has a live connection.

    This pattern is sometimes called "dependency injection" -- passing a
    dependency (the connection) into an object rather than having the object
    create it internally.
    """
    searcher.conn = conn


# ---------------------------------------------------------------------------
# Main polling loop
# ---------------------------------------------------------------------------

def run_batch_loop(
    searcher: HybridSearcher,
    reranker: CrossEncoderReranker,
    coder: Optional[LLMCoder],
    batch_size: int,
    product_code: Optional[str],
    limit: Optional[int],
    once: bool,
) -> int:
    """
    The main processing loop: fetch reports -> code them -> write results -> repeat.

    Loop behaviour:
        1. Open a fresh DB connection
        2. Count pending (uncoded) reports
        3. If pending == 0 and --once: exit
        4. If pending == 0 and continuous mode: sleep 60s, then loop again
        5. If pending > 0: fetch a batch, code each report, write results, commit
        6. If --once: exit after the batch completes
        7. Otherwise: loop back to step 1

    Error isolation:
        Each report is processed in a try/except block. A single report that
        causes an error (e.g. Ollama timeout) triggers a rollback for that
        report only. The loop continues with the next report in the batch.
        The problematic report is logged and will be retried in the next run.

    Args:
        searcher:     HybridSearcher (model loaded, connection refreshed per batch)
        reranker:     CrossEncoderReranker
        coder:        LLMCoder or None (--skip-llm mode)
        batch_size:   reports to fetch per DB round-trip
        product_code: optional device type filter
        limit:        maximum total reports to code in this run (None = no limit)
        once:         if True, exit when the queue is empty

    Returns:
        Total number of reports successfully coded in this run.
    """
    total_coded = 0
    total_pending_start: Optional[int] = None
    t_run_start = time.time()

    # Rolling-Window fuer ntfy-Statistik -- letzte N Records
    # Speichert pro Record: {is_fallback: bool, llm_confidence: Optional[float]}
    rolling: deque = deque(maxlen=NTFY_BATCH_SIZE)
    last_notify_at = 0    # letzter Notify-Punkt (in total_coded-Einheiten)

    while True:
        # Apply --limit: reduce batch size if we are close to the cap
        effective_batch = batch_size
        if limit is not None:
            remaining = limit - total_coded
            if remaining <= 0:
                logger.info("Reached --limit of %d. Stopping.", limit)
                break
            effective_batch = min(batch_size, remaining)

        # Open a fresh DB connection for this batch.
        # A new connection ensures we are resilient to SSH tunnel restarts
        # that may have broken the previous connection.
        conn = get_connection()
        _update_searcher_conn(searcher, conn)  # give the searcher the new connection

        try:
            pending = count_pending(conn, product_code)

            if pending == 0:
                if once:
                    logger.info("Queue empty (--once mode). Exiting.")
                    break
                logger.info(
                    "Queue empty -- sleeping %ds before next poll.",
                    DEFAULT_POLL_SECS,
                )
                conn.close()
                time.sleep(DEFAULT_POLL_SECS)
                continue

            if total_pending_start is None:
                total_pending_start = pending + total_coded

            reports = fetch_uncoded_reports(conn, effective_batch, product_code)
            logger.info(
                "Fetched %d reports for coding (%d still pending in DB).",
                len(reports), pending,
            )

            batch_ok  = 0
            batch_err = 0

            for report in reports:
                rk = report["mdr_report_key"]
                t0 = time.time()
                try:
                    # Run the full three-stage pipeline for this report
                    result = code_report(report, searcher, reranker, coder)
                    write_coding_result(conn, result)
                    conn.commit()  # commit per report for error isolation
                    batch_ok += 1
                    ms = (time.time() - t0) * 1000
                    # final_confidence kann None sein (Fallback-Pfad mit NULL-Strategie)
                    fc = result.get("final_confidence")
                    fc_str = f"{fc:.3f}" if fc is not None else "NULL "
                    logger.info(
                        "[OK] %-30s  PT=%-40s  conf=%s  %.0fms",
                        rk,
                        result.get("pt_name") or "N/A",
                        fc_str,
                        ms,
                    )

                    # -- Rolling-Window fuer ntfy ----------------------------
                    # is_fallback: LLM wurde befragt, hat aber None zurueckgegeben.
                    # NICHT-Fallback: skip-llm (coder is None) ODER echter LLM-Wert.
                    llm_conf_val = result.get("llm_confidence")
                    is_fallback = (coder is not None and llm_conf_val is None)
                    rolling.append({
                        "is_fallback": is_fallback,
                        "llm_confidence": llm_conf_val,
                    })

                    # -- Notify alle NTFY_BATCH_SIZE Records -----------------
                    if (total_coded + batch_ok) - last_notify_at >= NTFY_BATCH_SIZE:
                        elapsed_now = time.time() - t_run_start
                        rate_now = ((total_coded + batch_ok) / elapsed_now * 60
                                    if elapsed_now > 0 else 0.0)
                        notify_progress(
                            total_coded=total_coded + batch_ok,
                            rate_per_min=rate_now,
                            rolling=list(rolling),
                        )
                        last_notify_at = total_coded + batch_ok
                except Exception as exc:
                    # Roll back this report's transaction, log the error
                    try:
                        conn.rollback()
                    except Exception:
                        pass
                    batch_err += 1
                    logger.error("[ERR] %s: %s", rk, exc, exc_info=False)
                    if STRICT_MODE:
                        # Strict: Worker bricht ab beim ersten Fehler.
                        # Verbindung sauber schliessen, dann raise nach oben.
                        try:
                            conn.close()
                        except Exception:
                            pass
                        raise

            total_coded += batch_ok
            logger.info(
                "==> Batch done: %d coded / %d errors | Running total: %d",
                batch_ok, batch_err, total_coded,
            )

            # -- Progress summary with ETA ---------------------------------
            if total_pending_start and total_pending_start > 0:
                pct = total_coded / total_pending_start * 100
                elapsed = time.time() - t_run_start
                rate = total_coded / elapsed if elapsed > 0 else 0
                remaining_records = total_pending_start - total_coded
                eta_secs = remaining_records / rate if rate > 0 else 0
                eta_h = int(eta_secs // 3600)
                eta_m = int((eta_secs % 3600) // 60)
                logger.info(
                    ">>> PROGRESS: %d / %d (%.1f%%) | %.2f rec/min | ETA: %dh %02dm",
                    total_coded, total_pending_start, pct,
                    rate * 60, eta_h, eta_m,
                )

        except Exception as exc:
            # Unexpected error outside the per-report loop (e.g. DB connection lost)
            logger.error("Unexpected error in batch loop: %s", exc, exc_info=True)
            try:
                conn.rollback()
            except Exception:
                pass
            if STRICT_MODE:
                # Strict: auch unerwartete Errors duerfen nicht still bleiben
                raise

        finally:
            # Always close the connection, whether the batch succeeded or failed
            try:
                conn.close()
            except Exception:
                pass

        if once:
            break  # exit after first successful batch in --once mode

    # Final ntfy-Push am Ende -- gibt es noch ungesendete Records seit dem letzten Notify?
    if total_coded > last_notify_at and rolling:
        elapsed_final = time.time() - t_run_start
        rate_final = (total_coded / elapsed_final * 60) if elapsed_final > 0 else 0.0
        notify_progress(
            total_coded=total_coded,
            rate_per_min=rate_final,
            rolling=list(rolling),
        )

    return total_coded


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    """
    Parse command-line arguments and run the MedDRA coding worker.
    """
    parser = argparse.ArgumentParser(
        description="SentinelAI MedDRA Coding Worker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Continuous polling loop (default mode -- runs until Ctrl+C)
  python -m vigilex.workers.coding

  # Single pass: code all uncoded insulin pump reports, then exit
  python -m vigilex.workers.coding --once --product-code LZG

  # Test run: 100 reports in batches of 10, then exit
  python -m vigilex.workers.coding --limit 100 --batch-size 10 --once

  # Stage 1+2 only (skips Ollama -- useful when LLM tunnel is closed)
  python -m vigilex.workers.coding --skip-llm --once --limit 50
        """,
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help=f"Reports to process per DB fetch (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--product-code",
        help="Only code reports with this product code (e.g. LZG for insulin pumps)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Stop after coding this many total reports (default: no limit)",
    )
    parser.add_argument(
        "--once", action="store_true",
        help="Single pass -- exit when the queue is empty (no polling loop)",
    )
    parser.add_argument(
        "--skip-llm", action="store_true",
        help="Skip Stage 3 (Ollama). Use CrossEncoder top result as final answer. "
             "Useful for testing Stage 1+2, or when the Ollama tunnel is not open.",
    )
    parser.add_argument(
        "--groq", action="store_true",
        help="[EXPERIMENTAL] Use Groq API instead of Ollama for Stage 3. "
             "Requires GROQ_API_KEY env var. "
             "WARNING: sends narratives to external API -- capstone benchmarking only.",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable DEBUG-level logging (very verbose)",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("=== SentinelAI Coding Worker starting ===")
    logger.info(
        "Config: batch_size=%d | product_code=%s | limit=%s | once=%s | skip_llm=%s | groq=%s",
        args.batch_size,
        args.product_code or "ALL",
        args.limit or "none",
        args.once,
        args.skip_llm,
        args.groq,
    )
    # Mode-Status sichtbar im Startup-Log -- so siehst du sofort ob Strict-Mode
    # aktiv ist und ob ntfy-Pushes rausgehen werden.
    logger.info(
        "Mode: STRICT_MODE=%s | NTFY_TOPIC=%s | NTFY_BATCH_SIZE=%d",
        STRICT_MODE,
        NTFY_TOPIC if NTFY_TOPIC else "<disabled>",
        NTFY_BATCH_SIZE,
    )

    # Load all three pipeline components once at startup
    searcher, reranker, coder = load_pipeline(
        skip_llm=args.skip_llm,
        use_groq=args.groq,
    )

    t_start = time.time()
    total = run_batch_loop(
        searcher=searcher,
        reranker=reranker,
        coder=coder,
        batch_size=args.batch_size,
        product_code=args.product_code,
        limit=args.limit,
        once=args.once,
    )

    elapsed = time.time() - t_start
    # Compute throughput (reports per minute); avoid division by zero
    throughput = (total / elapsed * 60) if elapsed > 0 else 0
    logger.info(
        "=== Coding Worker finished | %d reports coded in %.1fs (%.1f reports/min) ===",
        total, elapsed, throughput,
    )


if __name__ == "__main__":
    main()
