"""
SentinelAI (vigilex) -- FastAPI Backend
========================================
Serves processed coding_results from PostgreSQL via REST API.

Endpoints:
    GET /health                  -- liveness check (no auth required)
    GET /coding-results          -- paginated list with filters
    GET /coding-results/stats    -- aggregate summary statistics
    GET /coding-results/{id}     -- single record by primary key

Authentication:
    All endpoints except /health require the header:
        X-API-Key: <value of API_KEY env var>

Run locally (outside docker, with SSH tunnel to Hetzner on port 5432):
    export DATABASE_URL=postgresql://vigilex:<pw>@localhost:5432/vigilex
    export API_KEY=dev-secret
    uvicorn vigilex.api.main:app --reload --port 8000

Inside docker-compose:
    docker compose up api
    -> accessible at http://localhost:8000
    -> Swagger UI at http://localhost:8000/docs
"""

import os
import logging
from datetime import date, datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, Security, Depends
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from vigilex.db.connection import get_connection, get_cursor

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="SentinelAI API",
    description="Query adverse event coding results from the vigilex pipeline.",
    version="1.0.0",
    docs_url="/docs",       # Swagger UI
    redoc_url="/redoc",
)

# CORS -- allow Streamlit / local frontend to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production if needed
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# API Key authentication
# ---------------------------------------------------------------------------
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(api_key: Optional[str] = Security(API_KEY_HEADER)) -> str:
    """
    Dependency that validates the X-API-Key header.

    The expected key is read from the API_KEY environment variable.
    If API_KEY is not set, the server refuses to start (see startup event).
    """
    expected = os.environ.get("API_KEY", "")
    if not api_key or api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")
    return api_key


# ---------------------------------------------------------------------------
# Pydantic response models
# ---------------------------------------------------------------------------
class CodingResult(BaseModel):
    id: int
    mdr_report_key: Optional[str]
    pt_code: Optional[int]
    pt_name: Optional[str]
    llt_code: Optional[int]
    llt_name: Optional[str]
    soc_name: Optional[str]
    vector_similarity: Optional[float]
    crossencoder_score: Optional[float]
    llm_confidence: Optional[float]
    final_confidence: Optional[float]
    model_version: Optional[str]
    coded_at: Optional[datetime]

    model_config = {"from_attributes": True}


class CodingStats(BaseModel):
    total_records: int
    records_with_llm: int       # llm_confidence is not NULL and != 0.3 (fallback)
    fallback_count: int         # llm_confidence == 0.3 (known fallback sentinel)
    avg_final_confidence: Optional[float]
    median_final_confidence: Optional[float]
    high_confidence_count: int  # final_confidence >= 0.5
    distinct_pt_codes: int
    earliest_coded_at: Optional[datetime]
    latest_coded_at: Optional[datetime]


class HealthResponse(BaseModel):
    status: str
    db: str
    version: str


class DecisionRequest(BaseModel):
    action: str          # 'accepted' | 'rejected' | 'overridden'
    note: Optional[str] = None  # free-text reviewer comment


class DecisionResponse(BaseModel):
    id: int
    reviewer_action: str
    reviewer_at: datetime
    reviewer_note: Optional[str]


# ---------------------------------------------------------------------------
# Startup check
# ---------------------------------------------------------------------------
@app.on_event("startup")
def startup_check():
    """
    Verify required environment variables on startup.
    Fail loudly rather than silently serving broken responses.
    """
    if not os.environ.get("API_KEY"):
        logger.warning("API_KEY is not set -- all authenticated endpoints will reject requests.")
    if not os.environ.get("DATABASE_URL"):
        logger.error("DATABASE_URL is not set -- database queries will fail.")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    """
    Liveness check. No authentication required.
    Attempts a lightweight DB query to confirm connectivity.
    """
    db_status = "ok"
    try:
        conn = get_connection()
        cur = get_cursor(conn)
        cur.execute("SELECT 1")
        conn.close()
    except Exception as exc:
        logger.error("Health check DB error: %s", exc)
        db_status = f"error: {exc}"

    return HealthResponse(
        status="ok",
        db=db_status,
        version=app.version,
    )


@app.get("/coding-results", response_model=list[CodingResult], tags=["Coding Results"])
def list_coding_results(
    limit: int = Query(default=50, ge=1, le=500, description="Max records to return"),
    offset: int = Query(default=0, ge=0, description="Skip N records (pagination)"),
    min_confidence: Optional[float] = Query(default=None, ge=0.0, le=1.0, description="Filter by final_confidence >="),
    max_confidence: Optional[float] = Query(default=None, ge=0.0, le=1.0, description="Filter by final_confidence <="),
    pt_code: Optional[int] = Query(default=None, description="Filter by MedDRA PT code"),
    soc_name: Optional[str] = Query(default=None, description="Filter by System Organ Class name (partial match)"),
    from_date: Optional[date] = Query(default=None, description="Filter coded_at >= this date (YYYY-MM-DD)"),
    to_date: Optional[date] = Query(default=None, description="Filter coded_at <= this date (YYYY-MM-DD)"),
    exclude_fallback: bool = Query(default=True, description="Exclude records where llm_confidence = 0.3 (known fallback)"),
    _key: str = Depends(require_api_key),
):
    """
    Return a paginated list of coding results with optional filters.

    Pagination example:
        GET /coding-results?limit=100&offset=0   -- first page
        GET /coding-results?limit=100&offset=100 -- second page

    Quality filter tip:
        Use min_confidence=0.5 to retrieve only higher-confidence codings.
        Use exclude_fallback=true (default) to skip the 0.30 fallback sentinel records.
    """
    conditions = []
    params: list = []

    if exclude_fallback:
        conditions.append("llm_confidence IS DISTINCT FROM 0.3")

    if min_confidence is not None:
        conditions.append("final_confidence >= %s")
        params.append(min_confidence)

    if max_confidence is not None:
        conditions.append("final_confidence <= %s")
        params.append(max_confidence)

    if pt_code is not None:
        conditions.append("pt_code = %s")
        params.append(pt_code)

    if soc_name is not None:
        conditions.append("soc_name ILIKE %s")
        params.append(f"%{soc_name}%")

    if from_date is not None:
        conditions.append("coded_at >= %s")
        params.append(from_date)

    if to_date is not None:
        conditions.append("coded_at <= %s")
        params.append(to_date)

    where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""

    sql = f"""
        SELECT
            id, mdr_report_key, pt_code, pt_name,
            llt_code, llt_name, soc_name,
            vector_similarity, crossencoder_score,
            llm_confidence, final_confidence,
            model_version, coded_at
        FROM processed.coding_results
        {where_clause}
        ORDER BY coded_at DESC
        LIMIT %s OFFSET %s
    """
    params += [limit, offset]

    try:
        conn = get_connection()
        cur = get_cursor(conn)
        cur.execute(sql, params)
        rows = cur.fetchall()
        conn.close()
    except Exception as exc:
        logger.error("DB error in list_coding_results: %s", exc)
        raise HTTPException(status_code=500, detail=f"Database error: {exc}")

    return [CodingResult(**dict(row)) for row in rows]


@app.get("/coding-results/stats", response_model=CodingStats, tags=["Coding Results"])
def coding_stats(
    _key: str = Depends(require_api_key),
):
    """
    Aggregate statistics over the entire coding_results table.

    Useful for dashboard overview panels, progress monitoring,
    and verifying pipeline health after a worker run.

    Note on fallback_count:
        Records where llm_confidence = 0.3 are known fallbacks
        (LLM call failed, hardcoded sentinel value used).
        High fallback_count indicates a pipeline health issue.
    """
    sql = """
        SELECT
            COUNT(*)                                        AS total_records,
            COUNT(*) FILTER (
                WHERE llm_confidence IS DISTINCT FROM 0.3
            )                                               AS records_with_llm,
            COUNT(*) FILTER (
                WHERE llm_confidence = 0.3
            )                                               AS fallback_count,
            ROUND(AVG(final_confidence)::numeric, 4)        AS avg_final_confidence,
            ROUND(PERCENTILE_CONT(0.5)
                WITHIN GROUP (ORDER BY final_confidence)
                ::numeric, 4)                               AS median_final_confidence,
            COUNT(*) FILTER (
                WHERE final_confidence >= 0.5
            )                                               AS high_confidence_count,
            COUNT(DISTINCT pt_code)                         AS distinct_pt_codes,
            MIN(coded_at)                                   AS earliest_coded_at,
            MAX(coded_at)                                   AS latest_coded_at
        FROM processed.coding_results
    """

    try:
        conn = get_connection()
        cur = get_cursor(conn)
        cur.execute(sql)
        row = cur.fetchone()
        conn.close()
    except Exception as exc:
        logger.error("DB error in coding_stats: %s", exc)
        raise HTTPException(status_code=500, detail=f"Database error: {exc}")

    return CodingStats(**dict(row))


@app.get("/coding-results/{record_id}", response_model=CodingResult, tags=["Coding Results"])
def get_coding_result(
    record_id: int,
    _key: str = Depends(require_api_key),
):
    """
    Fetch a single coding result by its primary key (id column).
    Returns 404 if the record does not exist.
    """
    sql = """
        SELECT
            id, mdr_report_key, pt_code, pt_name,
            llt_code, llt_name, soc_name,
            vector_similarity, crossencoder_score,
            llm_confidence, final_confidence,
            model_version, coded_at
        FROM processed.coding_results
        WHERE id = %s
    """

    try:
        conn = get_connection()
        cur = get_cursor(conn)
        cur.execute(sql, (record_id,))
        row = cur.fetchone()
        conn.close()
    except Exception as exc:
        logger.error("DB error in get_coding_result(%s): %s", record_id, exc)
        raise HTTPException(status_code=500, detail=f"Database error: {exc}")

    if row is None:
        raise HTTPException(status_code=404, detail=f"Record {record_id} not found.")

    return CodingResult(**dict(row))


@app.post(
    "/coding-results/{record_id}/decision",
    response_model=DecisionResponse,
    tags=["Coding Results"],
)
def save_decision(
    record_id: int,
    body: DecisionRequest,
    _key: str = Depends(require_api_key),
):
    """
    Save a reviewer decision for a coding result.

    Sets reviewer_action, reviewer_at (NOW()), and reviewer_note.
    Returns 404 if the record does not exist.
    Returns 400 if action is not one of: accepted, rejected, overridden.
    """
    valid_actions = {"accepted", "rejected", "overridden"}
    if body.action not in valid_actions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid action '{body.action}'. Must be one of: {sorted(valid_actions)}",
        )

    sql_check = "SELECT id FROM processed.coding_results WHERE id = %s"
    sql_update = """
        UPDATE processed.coding_results
        SET reviewer_action = %s,
            reviewer_at     = NOW(),
            reviewer_note   = %s
        WHERE id = %s
        RETURNING id, reviewer_action, reviewer_at, reviewer_note
    """

    try:
        conn = get_connection()
        cur = get_cursor(conn)

        cur.execute(sql_check, (record_id,))
        if cur.fetchone() is None:
            conn.close()
            raise HTTPException(status_code=404, detail=f"Record {record_id} not found.")

        cur.execute(sql_update, (body.action, body.note, record_id))
        row = cur.fetchone()
        conn.commit()
        conn.close()
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("DB error in save_decision(%s): %s", record_id, exc)
        raise HTTPException(status_code=500, detail=f"Database error: {exc}")

    return DecisionResponse(**dict(row))
