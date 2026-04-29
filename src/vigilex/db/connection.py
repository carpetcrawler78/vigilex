"""
SentinelAI (vigilex) -- Database connection helper.

Why this file exists:
    Every part of the system that needs to talk to PostgreSQL imports this module.
    Centralising the connection logic means we only define DATABASE_URL in one place
    (the .env file loaded by docker-compose) and everything else just calls get_connection().

What is psycopg2?
    psycopg2 is the standard Python library for talking to PostgreSQL. Think of it
    as the telephone wire between Python code and the database.

What is DATABASE_URL?
    A single string that encodes everything needed to connect:
        postgresql://username:password@hostname:port/database_name
    docker-compose reads this from the .env file and makes it available as an
    environment variable, so the code never has hardcoded passwords.
"""

import os
import logging
import psycopg2
import psycopg2.extras

logger = logging.getLogger(__name__)


def get_connection():
    """
    Open and return a new connection to the PostgreSQL database.

    Reads the connection string from the DATABASE_URL environment variable,
    which is set by docker-compose (or the .env file during local development).

    Returns:
        psycopg2.connection -- an open database connection.

    Raises:
        RuntimeError  -- if DATABASE_URL is not set (e.g. running outside docker
                         without setting the env variable manually).
        psycopg2.OperationalError -- if the database is unreachable (e.g. postgres
                         container is not running, or SSH tunnel is closed).

    Usage:
        conn = get_connection()
        # ... do database work ...
        conn.close()  # always close when done

    Design note:
        We deliberately open a fresh connection per batch rather than keeping one
        connection alive forever. Long-lived connections can go stale after SSH
        tunnel restarts, which are common during local development. A fresh
        connection per batch is self-healing.
    """
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError(
            "DATABASE_URL environment variable not set. "
            "Are you running inside docker-compose, or did you forget to "
            "source your .env file?"
        )
    return psycopg2.connect(url)


def get_cursor(conn):
    """
    Return a 'dictionary cursor' for the given connection.

    What is a cursor?
        In database terminology, a cursor is an object that lets you execute
        SQL queries and iterate over the results. Think of it as a pointer
        into the result set.

    What is a RealDictCursor?
        By default, psycopg2 returns rows as tuples: (value1, value2, ...).
        A RealDictCursor returns rows as dictionaries: {"column_name": value, ...}.
        This makes code much more readable -- you can write row["pt_name"]
        instead of row[4].

    Usage:
        conn = get_connection()
        cur = get_cursor(conn)
        cur.execute("SELECT pt_code, pt_name FROM processed.meddra_terms LIMIT 5")
        for row in cur.fetchall():
            print(row["pt_name"])  # works because of RealDictCursor
        conn.close()
    """
    return conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
