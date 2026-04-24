"""
SentinelAI (vigilex) -- MedDRA Coding Worker.

Pulls uncoded MAUDE reports from raw.maude_reports and runs the full
three-stage coding pipeline:

  Stage 1: HybridSearcher        (BM25 + Vector + RRF)    -> Top-20 candidates
  Stage 2: CrossEncoderReranker  (MiniLM-L-6)             -> Top-5 candidates
  Stage 3: LLMCoder              (Ollama llama3.2)        -> Final PT code

Results are written to processed.coding_results.

Usage (from vigilex root, with SSH tunnels open):
  python -m vigilex.workers.coding                  # continuous polling loop
  python -m vigilex.workers.coding --once           # single batch, then exit
  python -m vigilex.workers.coding --batch-size 50  # process 50 reports per batch
  python -m vigilex.workers.coding --product-code LZG --limit 200
  python -m vigilex.workers.coding --skip-llm       # Stage 1+2 only (no Ollama needed)

Environment variables:
  DATABASE_URL      PostgreSQL connection string (required)
  OLLAMA_URL        Ollama API base URL (default: http://localhost:11434)
  MODEL_VERSION     Tag written to coding_results.model_version (default: pipeline_v1)

SSH tunnels required:
  ssh -L 5432:localhost:5432 cap@46.225.109.99      # PostgreSQL
  ssh -L 11434:localhost:11434 cap@46.225.109.99    # Ollama (unless --skip-llm)
"""

import argparse
import logging
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional

import psycopg2
import psycopg2.extras

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
# Logging
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

DEFAULT_BATCH_SIZE  = 25      # reports to process per DB round-trip
DEFAULT_POLL_SECS   = 60      # seconds between polls when queue is empty
HYBRID_TOP_K        = 20      # candidates from Stage 1 -> Stage 2
RERANK_TOP_K        = 5       # candidates from Stage 2 -> Stage 3
CONFIDENCE_THRESHOLD = 0.5    # below this -> flagged for human review

MODEL_VERSION = os.getenv("MODEL_VERSION", "pipeline_v1")


# ---------------------------------------------------------------------------
# Helper: sigmoid normalization for CrossEncoder logits
# ---------------------------------------------------------------------------

def sigmoid(x: float) -> float:
    """Map any real number to (0, 1). Used to normalize CrossEncoder logits."""
    return 1.0 / (1.0 + math.exp(-x))


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def fetch_uncoded_reports(
    conn,
    batch_size: int,
    product_code: Optional[str] = None,
    limit: Optional[int] = None,
) -> list[dict]:
    """
    Return up to `batch_size` MAUDE reports that have no coding result yet.

    A report is considered uncoded when there is no row in
    processed.coding_results with the same mdr_report_key.

    Parameters
    ----------
    conn         : psycopg2 connection
    batch_size   : max rows to fetch in this call
    product_code : optional filter (e.g. 'LZG' for insulin pumps)
    limit        : overall cap on total rows processed -- enforced by caller
    """
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
    Insert one row into processed.coding_results.

    On conflict (same mdr_report_key already exists), does nothing -- a report
    that was already coded by a previous run is left untouched.

    Parameters
    ----------
    conn   : psycopg2 connection (caller commits)
    result : dict with keys matching coding_results columns
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
    """Return number of uncoded MAUDE reports."""
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
# Core coding logic: one report through the full pipeline
# ---------------------------------------------------------------------------

def code_report(
    report: dict,
    searcher: HybridSearcher,
    reranker: CrossEncoderReranker,
    coder: Optional[LLMCoder],
) -> dict:
    """
    Run one MAUDE report through the three-stage pipeline and return a dict
    ready for insertion into processed.coding_results.

    Stage 1 (HybridSearcher)      -> Top-20 candidates via BM25 + Vector + RRF
    Stage 2 (CrossEncoderReranker) -> Top-5 candidates by MiniLM-L-6 score
    Stage 3 (LLMCoder)             -> Final PT selection with structured rationale

    If coder is None (--skip-llm mode) or the LLM call fails, Stage 2's top
    result is used as the final answer with a lower confidence estimate.

    Final confidence formula:
      - With LLM:    0.3 * sigmoid(ce_score) + 0.7 * llm_confidence
      - Without LLM: sigmoid(ce_score)
    """
    narrative   = report["mdr_text"]
    report_key  = report["mdr_report_key"]

    # -- Stage 1: Hybrid Search -----------------------------------------------
    candidates = searcher.search(narrative, top_k=HYBRID_TOP_K)

    if not candidates:
        # Empty result -- no MedDRA terms matched at all
        logger.warning("No hybrid search results for %s", report_key)
        return _fallback_result(report_key, reason="no_candidates")

    # -- Stage 2: CrossEncoder Reranker ----------------------------------------
    reranked = reranker.rerank(narrative, candidates, top_k=RERANK_TOP_K)

    if not reranked:
        logger.warning("Reranker returned no results for %s", report_key)
        return _fallback_result(report_key, reason="no_reranked")

    top_ce = reranked[0]
    ce_score_norm = sigmoid(top_ce.crossencoder_score)

    # Best cosine similarity from hybrid search for the top CE result
    # (trgm_sim and cosine_sim are stored on SearchResult via rrf_score lineage)
    vector_sim = top_ce.cosine_sim if top_ce.cosine_sim is not None else 0.0

    # -- Stage 3: LLM Coder (optional) -----------------------------------------
    llm_confidence  = None
    final_pt_code   = top_ce.pt_code
    final_pt_name   = top_ce.pt_name
    final_soc_name  = top_ce.soc_name

    if coder is not None:
        try:
            llm_result = coder.code(narrative, reranked)
            llm_confidence  = llm_result.confidence
            final_pt_code   = llm_result.pt_code
            final_pt_name   = llm_result.pt_name
            # soc_name is not returned by LLM -- keep top CE soc for now
        except Exception as exc:
            logger.warning(
                "LLM call failed for %s (%s), falling back to CE top result",
                report_key, exc,
            )

    # -- Compute final_confidence ----------------------------------------------
    if llm_confidence is not None:
        final_confidence = 0.3 * ce_score_norm + 0.7 * llm_confidence
    else:
        final_confidence = ce_score_norm

    return {
        "mdr_report_key":    report_key,
        "pt_code":           final_pt_code,
        "pt_name":           final_pt_name,
        "soc_name":          final_soc_name,
        "vector_similarity": round(vector_sim, 6),
        "crossencoder_score": round(top_ce.crossencoder_score, 6),
        "llm_confidence":    round(llm_confidence, 6) if llm_confidence is not None else None,
        "final_confidence":  round(final_confidence, 6),
        "model_version":     MODEL_VERSION,
    }


def _fallback_result(report_key: str, reason: str = "unknown") -> dict:
    """
    Minimal result row for reports where the pipeline produced nothing.
    Inserted with final_confidence=0 so they appear in quality queries.
    """
    logger.warning("Writing fallback result for %s (reason=%s)", report_key, reason)
    return {
        "mdr_report_key":    report_key,
        "pt_code":           None,
        "pt_name":           None,
        "soc_name":          None,
        "vector_similarity": None,
        "crossencoder_score": None,
        "llm_confidence":    None,
        "final_confidence":  0.0,
        "model_version":     MODEL_VERSION + "_fallback",
    }


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run_batch(
    searcher: HybridSearcher,
    reranker: CrossEncoderReranker,
    coder: Optional[LLMCoder],
    batch_size: int,
    product_code: Optional[str],
    limit: Optional[int] = None,
    once: bool = False,
) -> int:
    """
    Main polling loop.

    Repeatedly fetches uncoded reports, runs them through the pipeline, and
    writes results to the DB. Sleeps between polls when the queue is empty.

    Returns total number of reports coded in this run.
    """
    total_coded = 0

    while True:
        # Respect the --limit cap
        effective_batch = batch_size
        if limit is not None:
            remaining = limit - total_coded
            if remaining <= 0:
                logger.info("Reached --limit %d, stopping.", limit)
                break
            effective_batch = min(batch_size, remaining)

        conn = get_connection()
        try:
            pending = count_pending(conn, product_code)
            if pending == 0:
                if once:
                    logger.info("No uncoded reports found (--once mode). Done.")
                    break
                logger.info(
                    "No uncoded reports. Sleeping %ds before next poll...",
                    DEFAULT_POLL_SECS,
                )
                conn.close()
                time.sleep(DEFAULT_POLL_SECS)
                continue

            reports = fetch_uncoded_reports(conn, effective_batch, product_code)
            logger.info(
                "Batch: %d reports to code (%d pending total)",
                len(reports), pending,
            )

            batch_coded = 0
            batch_errors = 0

            for report in reports:
                rk = report["mdr_report_key"]
                t0 = time.time()
                try:
                    result = code_report(report, searcher, reranker, coder)
                    write_coding_result(conn, result)
                    conn.commit()
                    batch_coded += 1
                    elapsed_ms = (time.time() - t0) * 1000
                    logger.info(
                        "Coded %s -> PT=%s (conf=%.3f) in %.0fms",
                        rk,
                        result.get("pt_name", "N/A"),
                        result.get("final_confidence", 0.0),
                        elapsed_ms,
                    )
                except Exception as exc:
                    conn.rollback()
                    batch_errors += 1
                    logger.error("Failed to code %s: %s", rk, exc, exc_info=True)

            total_coded += batch_coded
            logger.info(
                "Batch complete: %d coded, %d errors | Total coded: %d",
                batch_coded, batch_errors, total_coded,
            )

        finally:
            conn.close()

        if once:
            break

    return total_coded


# ---------------------------------------------------------------------------
# Model loading (shared across all batches for efficiency)
# ---------------------------------------------------------------------------

def load_pipeline(
    skip_llm: bool = False,
) -> tuple[HybridSearcher, CrossEncoderReranker, Optional[LLMCoder]]:
    """
    Load all three pipeline components.

    Models are loaded once and reused across all batches -- loading PubMedBERT
    and the CrossEncoder each take several seconds on first call. A fresh DB
    connection is created per batch (see run_batch).

    Returns
    -------
    (searcher, reranker, coder)
    coder is None when skip_llm=True.
    """
    logger.info("Loading PubMedBERT embedding model...")
    t0 = time.time()
    embedding_model = EmbeddingModel()
    logger.info("PubMedBERT loaded on %s in %.1fs", embedding_model.device, time.time() - t0)

    # HybridSearcher gets a DB connection per batch (passed in run_batch)
    # We create a temporary connection just to pass to the constructor --
    # the actual per-batch connections are opened in run_batch.
    conn_init = get_connection()
    searcher = HybridSearcher(
        conn=conn_init,
        embedding_model=embedding_model,
        candidate_pool=100,
    )
    # We close this init connection immediately; run_batch opens fresh ones.
    conn_init.close()

    logger.info("Loading CrossEncoder reranker (MiniLM-L-6)...")
    t0 = time.time()
    reranker = CrossEncoderReranker()
    logger.info("CrossEncoder loaded in %.1fs", time.time() - t0)

    coder = None
    if not skip_llm:
        ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        logger.info("Initialising LLM coder (Ollama at %s)...", ollama_url)
        coder = LLMCoder(
            ollama_url=ollama_url,
            confidence_threshold=CONFIDENCE_THRESHOLD,
        )
        # Quick connectivity test
        try:
            import requests
            r = requests.get(f"{ollama_url}/api/tags", timeout=5)
            if r.status_code == 200:
                models = [m["name"] for m in r.json().get("models", [])]
                logger.info("Ollama reachable -- models: %s", models)
            else:
                logger.warning("Ollama responded with status %d", r.status_code)
        except Exception as exc:
            logger.warning(
                "Cannot reach Ollama at %s (%s). "
                "Stage 3 will fall back to CrossEncoder top result.",
                ollama_url, exc,
            )

    return searcher, reranker, coder


# ---------------------------------------------------------------------------
# Reload DB connection for HybridSearcher between batches
# ---------------------------------------------------------------------------

def _update_searcher_conn(searcher: HybridSearcher, conn) -> None:
    """Replace the DB connection on an existing HybridSearcher instance."""
    searcher.conn = conn


# ---------------------------------------------------------------------------
# Revised run_batch that reconnects the searcher's connection per batch
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
    Polling loop that opens a fresh DB connection per batch and injects it
    into the HybridSearcher (which holds a reference to a psycopg2 connection).
    """
    total_coded = 0

    while True:
        effective_batch = batch_size
        if limit is not None:
            remaining = limit - total_coded
            if remaining <= 0:
                logger.info("Reached --limit %d. Stopping.", limit)
                break
            effective_batch = min(batch_size, remaining)

        conn = get_connection()
        # Inject the fresh connection into the searcher for this batch
        _update_searcher_conn(searcher, conn)

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

            reports = fetch_uncoded_reports(conn, effective_batch, product_code)
            logger.info(
                "Fetched %d reports for coding (%d still pending in DB).",
                len(reports), pending,
            )

            batch_ok = 0
            batch_err = 0

            for report in reports:
                rk = report["mdr_report_key"]
                t0 = time.time()
                try:
                    result = code_report(report, searcher, reranker, coder)
                    write_coding_result(conn, result)
                    conn.commit()
                    batch_ok += 1
                    ms = (time.time() - t0) * 1000
                    logger.info(
                        "[OK] %-30s  PT=%-40s  conf=%.3f  %.0fms",
                        rk, result.get("pt_name") or "N/A",
                        result.get("final_confidence", 0.0), ms,
                    )
                except Exception as exc:
                    try:
                        conn.rollback()
                    except Exception:
                        pass
                    batch_err += 1
                    logger.error("[ERR] %s: %s", rk, exc, exc_info=False)

            total_coded += batch_ok
            logger.info(
                "==> Batch done: %d OK / %d errors | Running total: %d",
                batch_ok, batch_err, total_coded,
            )

        except Exception as exc:
            logger.error("Unexpected error in batch loop: %s", exc, exc_info=True)
            try:
                conn.rollback()
            except Exception:
                pass

        finally:
            try:
                conn.close()
            except Exception:
                pass

        if once:
            break

    return total_coded


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="SentinelAI MedDRA Coding Worker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Continuous mode (polls every 60s when queue is empty)
  python -m vigilex.workers.coding

  # Single pass: code all uncoded insulin pump reports, then exit
  python -m vigilex.workers.coding --once --product-code LZG

  # Process only 100 reports (for testing), in batches of 10
  python -m vigilex.workers.coding --limit 100 --batch-size 10 --once

  # Skip Ollama (Stage 1+2 only) -- useful when LLM tunnel is closed
  python -m vigilex.workers.coding --skip-llm --once --limit 50
        """,
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help=f"Reports to process per DB fetch (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--product-code",
        help="Only code reports with this product code (e.g. LZG)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Stop after coding this many reports (default: no limit)",
    )
    parser.add_argument(
        "--once", action="store_true",
        help="Single pass -- exit when queue is empty (no polling loop)",
    )
    parser.add_argument(
        "--skip-llm", action="store_true",
        help="Skip Stage 3 (Ollama). Use CrossEncoder top result as final answer.",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="DEBUG-level logging",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("=== SentinelAI Coding Worker starting ===")
    logger.info(
        "Config: batch_size=%d | product_code=%s | limit=%s | once=%s | skip_llm=%s",
        args.batch_size, args.product_code or "ALL",
        args.limit or "none", args.once, args.skip_llm,
    )

    # Load models once at startup
    searcher, reranker, coder = load_pipeline(skip_llm=args.skip_llm)

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
    logger.info(
        "=== Coding Worker finished | %d reports coded in %.1fs (%.1f/min) ===",
        total, elapsed, (total / elapsed * 60) if elapsed > 0 else 0,
    )


if __name__ == "__main__":
    main()
