"""
SentinelAI (vigilex) -- PRR / ROR Disproportionality Signal Detection
Module 3, first component.

What this module does:
    For every (product_code, pt_code) combination in processed.coding_results,
    it computes two standard pharmacovigilance disproportionality metrics:

    PRR  -- Proportional Reporting Ratio  (EMA standard)
    ROR  -- Reporting Odds Ratio          (more robust for small counts)

    Both are based on a 2x2 contingency table:

                     | this PT  | all other PTs |
    this device      |    a     |      b        |
    all other devices|    c     |      d        |

    PRR = (a / (a+b)) / (c / (c+d))
    ROR = (a * d) / (b * c)

    Signal threshold (EMA guideline):
        is_signal = TRUE  when  PRR >= 2.0  AND  n_reports_focal >= 3

    95% confidence intervals use the log-normal method (Evans 2001):
        SE(ln PRR) = sqrt(1/a - 1/(a+b) + 1/c - 1/(c+d))
        CI = exp(ln(PRR) +/- 1.96 * SE)

    Same approach for ROR.

Why a separate module (not inside the worker)?
    Signal computation is a batch analytics step, not a per-record pipeline step.
    It reads from processed.coding_results (Module 2 output) and writes to
    processed.signal_results. It can be run on demand or on a schedule.

References:
    Evans SJ et al. (2001). Use of proportional reporting ratios (PRRs)
    for signal generation from spontaneous adverse drug reaction reports.
    Pharmacoepidemiol Drug Saf. 10(6):483-6.
"""

import math
import logging
from datetime import date, datetime
from typing import Optional

#from vigilex.db.connection import get_connection, get_cursor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Signal threshold config (EMA standard defaults)
# ---------------------------------------------------------------------------

DEFAULT_THRESHOLDS = {
    "min_reports_focal": 3,     # minimum n for a signal to be flagged
    "prr_min":           2.0,   # PRR must be >= this value
    "ci_lower_min":      1.0,   # lower 95% CI must be > 1.0 (optional, strict mode)
}


# ---------------------------------------------------------------------------
# Core math functions -- pure Python, no DB dependency
# ---------------------------------------------------------------------------

def _compute_prr(a: int, b: int, c: int, d: int) -> tuple[float | None, float | None, float | None]:
    """
    Compute PRR and its 95% confidence interval.

    Args:
        a -- reports for focal device x focal PT
        b -- reports for focal device, all other PTs
        c -- reports for all other devices x focal PT
        d -- reports for all other devices, all other PTs

    Returns:
        (prr, lower_ci, upper_ci)
        Returns (None, None, None) if computation is not possible (zero denominators).
    """
    # Guard against division by zero
    if (a + b) == 0 or (c + d) == 0 or c == 0:
        return (None, None, None)

    prr = (a / (a + b)) / (c / (c + d))

    if prr <= 0 or a == 0:
        return (prr, None, None)

    # SE of ln(PRR) via Evans 2001
    se = math.sqrt(
        1.0 / a
        - 1.0 / (a + b)
        + 1.0 / c
        - 1.0 / (c + d)
    )
    ln_prr = math.log(prr)
    lower_ci = math.exp(ln_prr - 1.96 * se)
    upper_ci = math.exp(ln_prr + 1.96 * se)

    return (prr, lower_ci, upper_ci)


def _compute_ror(a: int, b: int, c: int, d: int) -> tuple[float | None, float | None, float | None]:
    """
    Compute ROR and its 95% confidence interval.

    Args:
        a, b, c, d -- same 2x2 table as _compute_prr

    Returns:
        (ror, lower_ci, upper_ci)
    """
    if b == 0 or c == 0:
        return (None, None, None)

    ror = (a * d) / (b * c)

    if ror <= 0 or a == 0 or d == 0:
        return (ror, None, None)

    # SE of ln(ROR)
    se = math.sqrt(1.0 / a + 1.0 / b + 1.0 / c + 1.0 / d)
    ln_ror = math.log(ror)
    lower_ci = math.exp(ln_ror - 1.96 * se)
    upper_ci = math.exp(ln_ror + 1.96 * se)

    return (ror, lower_ci, upper_ci)


def _is_signal(
    n_focal: int,
    prr: Optional[float],
    thresholds: dict,
) -> bool:
    """
    Apply signal threshold rules.

    EMA standard: PRR >= 2 AND n_focal >= 3.
    """
    if prr is None:
        return False
    return (
        n_focal >= thresholds["min_reports_focal"]
        and prr >= thresholds["prr_min"]
    )


# ---------------------------------------------------------------------------
# DB query -- build the 2x2 table for all (product_code, pt_code) combos
# ---------------------------------------------------------------------------

QUERY_CONTINGENCY = """
WITH
-- Step 1: count reports per (product_code, pt_code) -- cell a
focal AS (
    SELECT
        r.product_code,
        c.pt_code,
        c.pt_name,
        c.soc_name,
        COUNT(*) AS a
    FROM processed.coding_results c
    JOIN raw.maude_reports r USING (mdr_report_key)
    WHERE r.date_received BETWEEN %(start_date)s AND %(end_date)s
      AND c.pt_code IS NOT NULL
    GROUP BY r.product_code, c.pt_code, c.pt_name, c.soc_name
),

-- Step 2: total reports per product_code -- (a + b)
device_total AS (
    SELECT
        r.product_code,
        COUNT(*) AS ab
    FROM processed.coding_results c
    JOIN raw.maude_reports r USING (mdr_report_key)
    WHERE r.date_received BETWEEN %(start_date)s AND %(end_date)s
      AND c.pt_code IS NOT NULL
    GROUP BY r.product_code
),

-- Step 3: total reports with this PT across ALL devices -- (a + c)
pt_all_devices AS (
    SELECT
        c.pt_code,
        COUNT(*) AS ac
    FROM processed.coding_results c
    JOIN raw.maude_reports r USING (mdr_report_key)
    WHERE r.date_received BETWEEN %(start_date)s AND %(end_date)s
      AND c.pt_code IS NOT NULL
    GROUP BY c.pt_code
),

-- Step 4: grand total -- N = a + b + c + d
grand_total AS (
    SELECT COUNT(*) AS N
    FROM processed.coding_results c
    JOIN raw.maude_reports r USING (mdr_report_key)
    WHERE r.date_received BETWEEN %(start_date)s AND %(end_date)s
      AND c.pt_code IS NOT NULL
)

-- Assemble 2x2 table:
--   a = focal.a
--   b = ab - a           (focal device, other PTs)
--   c = ac - a           (other devices, this PT)
--   d = N - ab - ac + a  (other devices, other PTs)
SELECT
    f.product_code,
    f.pt_code,
    f.pt_name,
    f.soc_name,
    f.a                                          AS n_reports_focal,
    dt.ab                                        AS n_reports_device_total,
    pa.ac                                        AS n_pt_all_devices,
    gt.N                                         AS n_grand_total,
    (dt.ab - f.a)                                AS b,
    (pa.ac - f.a)                                AS c,
    (gt.N - dt.ab - pa.ac + f.a)                 AS d
FROM focal f
JOIN device_total  dt ON f.product_code = dt.product_code
JOIN pt_all_devices pa ON f.pt_code = pa.pt_code
CROSS JOIN grand_total gt
ORDER BY f.product_code, f.a DESC
"""


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_prr_ror(
    start_date: date,
    end_date: date,
    thresholds: Optional[dict] = None,
    dry_run: bool = False,
) -> list[dict]:
    """
    Compute PRR/ROR for all (product_code, pt_code) combinations in a time window
    and write results to processed.signal_results.

    Args:
        start_date  -- analysis window start (inclusive)
        end_date    -- analysis window end (inclusive)
        thresholds  -- override DEFAULT_THRESHOLDS (optional)
        dry_run     -- if True, compute but do NOT write to DB (for testing)

    Returns:
        List of result dicts (all computed rows, regardless of is_signal).
    """
 
 
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    logger.info(
        "PRR/ROR: window %s to %s | thresholds: %s | dry_run=%s",
        start_date, end_date, thresholds, dry_run,
    )
    
    from vigilex.db.connection import get_connection, get_cursor  # lazy
    conn = get_connection()
    try:
        cur = get_cursor(conn)

        # -- Fetch contingency table rows ---------------------------------
        cur.execute(QUERY_CONTINGENCY, {
            "start_date": start_date,
            "end_date":   end_date,
        })
        rows = cur.fetchall()
        logger.info("PRR/ROR: %d (product_code, pt_code) combinations found", len(rows))

        if not rows:
            logger.warning("No coding_results found for the given time window.")
            return []

        # -- Compute metrics for each row ---------------------------------
        results = []
        for row in rows:
            a = row["n_reports_focal"]
            b = row["b"]
            c = row["c"]
            d = row["d"]

            prr, prr_lo, prr_hi = _compute_prr(a, b, c, d)
            ror, ror_lo, ror_hi = _compute_ror(a, b, c, d)
            signal = _is_signal(a, prr, thresholds)

            result = {
                "product_code":         row["product_code"],
                "pt_code":              row["pt_code"],
                "pt_name":              row["pt_name"],
                "soc_name":             row["soc_name"],
                "analysis_start_date":  start_date,
                "analysis_end_date":    end_date,
                "n_reports_focal":      a,
                "n_reports_total":      row["n_reports_device_total"],
                "n_pt_all_devices":     row["n_pt_all_devices"],
                "n_all_devices_total":  row["n_grand_total"],
                "prr":                  prr,
                "prr_lower_ci":         prr_lo,
                "prr_upper_ci":         prr_hi,
                "ror":                  ror,
                "ror_lower_ci":         ror_lo,
                "ror_upper_ci":         ror_hi,
                "is_signal":            signal,
                "signal_threshold_config": thresholds,
            }
            results.append(result)

        n_signals = sum(1 for r in results if r["is_signal"])
        logger.info(
            "PRR/ROR: %d total results, %d signals detected",
            len(results), n_signals,
        )

        # -- Write to DB --------------------------------------------------
        if not dry_run:
            _upsert_signal_results(cur, results)
            conn.commit()
            logger.info("PRR/ROR: results written to processed.signal_results")
        else:
            logger.info("PRR/ROR: dry_run=True -- skipping DB write")

        return results

    except Exception as e:
        conn.rollback()
        logger.error("PRR/ROR failed: %s", e)
        raise
    finally:
        conn.close()


def _upsert_signal_results(cur, results: list[dict]) -> None:
    """
    Insert signal results into processed.signal_results.
    Uses DELETE + INSERT per (product_code, pt_code, analysis window)
    to allow re-running without creating duplicates.
    """
    if not results:
        return

    # Grab window from first result (all rows share same window)
    start_date = results[0]["analysis_start_date"]
    end_date   = results[0]["analysis_end_date"]

    # Remove existing results for this window to allow re-runs
    cur.execute("""
        DELETE FROM processed.signal_results
        WHERE analysis_start_date = %(start)s
          AND analysis_end_date   = %(end)s
    """, {"start": start_date, "end": end_date})

    deleted = cur.rowcount
    if deleted > 0:
        logger.info("PRR/ROR: removed %d existing rows for this window (re-run)", deleted)

    INSERT_SQL = """
        INSERT INTO processed.signal_results (
            product_code, pt_code, pt_name, soc_name,
            analysis_start_date, analysis_end_date,
            n_reports_focal, n_reports_total, n_pt_all_devices, n_all_devices_total,
            prr, prr_lower_ci, prr_upper_ci,
            ror, ror_lower_ci, ror_upper_ci,
            is_signal, signal_threshold_config
        ) VALUES (
            %(product_code)s, %(pt_code)s, %(pt_name)s, %(soc_name)s,
            %(analysis_start_date)s, %(analysis_end_date)s,
            %(n_reports_focal)s, %(n_reports_total)s,
            %(n_pt_all_devices)s, %(n_all_devices_total)s,
            %(prr)s, %(prr_lower_ci)s, %(prr_upper_ci)s,
            %(ror)s, %(ror_lower_ci)s, %(ror_upper_ci)s,
            %(is_signal)s, %(signal_threshold_config)s::jsonb
        )
    """
    import json
    for r in results:
        r_copy = dict(r)
        r_copy["signal_threshold_config"] = json.dumps(r_copy["signal_threshold_config"])
        cur.execute(INSERT_SQL, r_copy)

    logger.info("PRR/ROR: inserted %d rows into processed.signal_results", len(results))


# ---------------------------------------------------------------------------
# CLI entry point -- run directly for quick tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s -- %(message)s",
    )

    # Default: full 2024 window (what we have ingested)
    start = date(2024, 1, 1)
    end   = date(2024, 12, 31)

    dry = "--dry-run" in sys.argv

    results = run_prr_ror(start_date=start, end_date=end, dry_run=dry)

    print(f"\n{'='*60}")
    print(f"PRR/ROR results: {len(results)} combinations")
    signals = [r for r in results if r["is_signal"]]
    print(f"Signals detected (PRR>=2, n>=3): {len(signals)}")
    print()

    if signals:
        print("TOP SIGNALS:")
        for s in sorted(signals, key=lambda x: x["prr"] or 0, reverse=True)[:10]:
            print(
                f"  {s['product_code']:6s}  {s['pt_name'][:40]:40s}  "
                f"PRR={s['prr']:.2f}  n={s['n_reports_focal']}"
            )
    else:
        print("No signals detected (or no data -- check SSH tunnel)")
