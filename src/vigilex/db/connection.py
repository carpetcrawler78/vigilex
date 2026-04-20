"""
SentinelAI (vigilex) -- Database connection helper.
Reads DATABASE_URL from environment (set by docker-compose).
"""
import os
import logging
import psycopg2
import psycopg2.extras

logger = logging.getLogger(__name__)


def get_connection():
    """Return a psycopg2 connection from DATABASE_URL env var."""
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL environment variable not set")
    return psycopg2.connect(url)


def get_cursor(conn):
    """Return a DictCursor for convenient column-name access."""
    return conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
