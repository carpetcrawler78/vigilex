#!/usr/bin/env bash
# =============================================================================
# SentinelAI (vigilex) -- Create application roles with passwords from env vars
# Docker runs *.sh files in docker-entrypoint-initdb.d/ automatically.
# Runs AFTER 01_init.sql (alphabetical order).
# Passwords come from .env via docker-compose environment injection.
# =============================================================================
set -e

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    DO \$\$ BEGIN
        IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'vigilex_app') THEN
            CREATE ROLE vigilex_app LOGIN PASSWORD '${POSTGRES_PASSWORD}';
        END IF;
    END \$\$;

    DO \$\$ BEGIN
        IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'vigilex_readonly') THEN
            CREATE ROLE vigilex_readonly LOGIN PASSWORD '${POSTGRES_PASSWORD}';
        END IF;
    END \$\$;

    GRANT USAGE ON SCHEMA raw, processed TO vigilex_app;
    GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA raw TO vigilex_app;
    GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA processed TO vigilex_app;
    GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA processed TO vigilex_app;

    GRANT USAGE ON SCHEMA raw, processed TO vigilex_readonly;
    GRANT SELECT ON ALL TABLES IN SCHEMA raw TO vigilex_readonly;
    GRANT SELECT ON ALL TABLES IN SCHEMA processed TO vigilex_readonly;
EOSQL
