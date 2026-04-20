#!/usr/bin/env bash
# =============================================================================
# SentinelAI (vigilex) -- Hetzner Server Migration
# Migriert von vigilex_old/ zum echten vigilex/ Repo.
#
# Ausfuehren auf dem Hetzner Server:
#   ssh cap@46.225.109.99
#   cd ~/vigilex
#   chmod +x migrate_from_old.sh
#   ./migrate_from_old.sh
#
# Was der Script tut:
#   1. Zeigt aktuellen Status (was laeuft wo?)
#   2. Stoppt Services in vigilex_old/ (sofern vorhanden)
#   3. Prueft .env im vigilex/ Repo
#   4. Startet Infrastruktur-Services: postgres, redis, grafana, ollama
#   5. Prueft ob init.sql korrekt ausgefuehrt wurde
#   6. Zeigt naechste Schritte
#
# HINWEIS: api / worker-ingest / worker-coding / worker-signal werden
# NOCH NICHT gestartet -- erst wenn src/vigilex/ Code vorhanden ist.
# =============================================================================

set -euo pipefail

# Farben fuer Output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'  # No Color

ok()   { echo -e "${GREEN}[OK]${NC}    $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC}  $*"; }
fail() { echo -e "${RED}[FAIL]${NC}  $*"; exit 1; }
info() { echo -e "        $*"; }

echo ""
echo "============================================================"
echo "  SentinelAI -- Migration vigilex_old -> vigilex repo"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
echo ""


# =============================================================================
# SCHRITT 1 -- Status pruefen
# =============================================================================
echo "--- SCHRITT 1: Status pruefen ---"
echo ""

# Welcher User bin ich?
CURRENT_USER=$(whoami)
info "User: $CURRENT_USER"
info "Hostname: $(hostname)"
info "Verzeichnis: $(pwd)"
echo ""

# Docker laufen?
if ! command -v docker &> /dev/null; then
    fail "Docker nicht gefunden. Bitte zuerst Docker installieren."
fi
ok "Docker gefunden: $(docker --version | head -1)"

# Laufende Container anzeigen
echo ""
info "Laufende Container:"
docker ps --format "  - {{.Names}} ({{.Image}}) -- {{.Status}}" 2>/dev/null || warn "Keine Container oder Docker-Daemon nicht erreichbar"
echo ""


# =============================================================================
# SCHRITT 2 -- vigilex_old stoppen (sofern vorhanden)
# =============================================================================
echo "--- SCHRITT 2: vigilex_old stoppen ---"
echo ""

OLD_PATH="$HOME/vigilex_old"

if [ -d "$OLD_PATH" ] && [ -f "$OLD_PATH/docker-compose.yml" ]; then
    warn "vigilex_old/ gefunden: $OLD_PATH"
    info "Stoppe Services aus vigilex_old..."
    cd "$OLD_PATH"
    docker compose down --remove-orphans 2>/dev/null || warn "docker compose down in vigilex_old fehlgeschlagen (evtl. liefen keine Services)"
    ok "vigilex_old Services gestoppt"
    cd - > /dev/null
else
    ok "vigilex_old/ nicht gefunden oder kein docker-compose.yml -- nichts zu stoppen"
fi
echo ""


# =============================================================================
# SCHRITT 3 -- vigilex Repo pruefen + aktualisieren
# =============================================================================
echo "--- SCHRITT 3: vigilex Repo ---"
echo ""

VIGILEX_PATH="$HOME/vigilex"

if [ ! -d "$VIGILEX_PATH" ]; then
    fail "vigilex/ Verzeichnis nicht gefunden: $VIGILEX_PATH"
fi

cd "$VIGILEX_PATH"
ok "Wechsel nach $VIGILEX_PATH"

# Git status
if [ -d ".git" ]; then
    info "Git-Branch: $(git branch --show-current 2>/dev/null || echo 'unbekannt')"
    info "Letzter Commit: $(git log --oneline -1 2>/dev/null || echo 'keine Commits')"

    # Pull neuesten Stand
    info "git pull..."
    git fetch origin 2>/dev/null && git pull origin main 2>/dev/null && ok "git pull OK" || warn "git pull fehlgeschlagen -- ggf. kein Internet oder kein Remote"
else
    warn "Kein .git Verzeichnis -- kein git pull"
fi
echo ""


# =============================================================================
# SCHRITT 4 -- .env pruefen
# =============================================================================
echo "--- SCHRITT 4: .env pruefen ---"
echo ""

if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        warn ".env fehlt -- kopiere .env.example nach .env"
        cp .env.example .env
        warn "WICHTIG: .env jetzt anpassen!"
        info "  nano .env"
        info ""
        info "  Pflicht-Felder:"
        info "    POSTGRES_PASSWORD=<sicheres-passwort>"
        info "    GRAFANA_PASSWORD=<sicheres-passwort>"
        info "    OPENFDA_API_KEY=<deinen-Key-eintragen>"
        echo ""
        echo -e "  ${RED}Bitte .env ausfuellen und dann dieses Script neu starten.${NC}"
        echo ""
        exit 1
    else
        fail ".env und .env.example fehlen -- bitte manuell erstellen"
    fi
fi

# Pflichtfelder pruefen
check_env_var() {
    local var="$1"
    local val
    val=$(grep "^${var}=" .env | cut -d'=' -f2- | tr -d '"' | tr -d "'" | xargs)
    if [ -z "$val" ] || [ "$val" = "CHANGE_ME_STRONG_PASSWORD" ] || [ "$val" = "CHANGE_ME_IN_ENV" ]; then
        fail "${var} in .env ist leer oder noch Standard-Wert -- bitte aendern"
    fi
    ok "${var} gesetzt"
}

check_env_var "POSTGRES_PASSWORD"
check_env_var "GRAFANA_PASSWORD"
echo ""


# =============================================================================
# SCHRITT 5 -- Infrastruktur-Services starten
#
# NOCH NICHT starten: api, worker-ingest, worker-coding, worker-signal
# Diese brauchen src/vigilex/ Python-Code, der noch nicht existiert.
# =============================================================================
echo "--- SCHRITT 5: Infrastruktur starten (postgres, redis, grafana, ollama) ---"
echo ""

info "docker compose up -d postgres redis grafana ollama"
docker compose up -d postgres redis grafana ollama

ok "Services gestartet -- warte auf postgres healthcheck..."
echo ""

# Warte bis postgres healthy
MAX_WAIT=60
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    STATUS=$(docker inspect --format='{{.State.Health.Status}}' vigilex-postgres 2>/dev/null || echo "not_found")
    if [ "$STATUS" = "healthy" ]; then
        ok "postgres ist healthy (nach ${WAITED}s)"
        break
    fi
    if [ "$STATUS" = "not_found" ]; then
        fail "vigilex-postgres Container nicht gefunden"
    fi
    info "postgres Status: $STATUS -- warte... (${WAITED}s / ${MAX_WAIT}s)"
    sleep 5
    WAITED=$((WAITED + 5))
done

if [ $WAITED -ge $MAX_WAIT ]; then
    fail "postgres nicht healthy nach ${MAX_WAIT}s -- pruefe: docker logs vigilex-postgres"
fi
echo ""


# =============================================================================
# SCHRITT 6 -- init.sql Ergebnis pruefen
#
# PostgreSQL fuehrt init_db/*.sql automatisch beim ersten Start aus.
# Wir pruefen ob Extensions + Schemas + Tabellen korrekt angelegt wurden.
# =============================================================================
echo "--- SCHRITT 6: Datenbank-Schema pruefen ---"
echo ""

# Postgres-User aus .env lesen
POSTGRES_USER=$(grep "^POSTGRES_USER=" .env | cut -d'=' -f2 | xargs)
POSTGRES_USER=${POSTGRES_USER:-vigilex}

run_psql() {
    docker exec vigilex-postgres psql -U "$POSTGRES_USER" -d vigilex -tAc "$1" 2>/dev/null
}

# Extensions
info "Pruefe Extensions..."
EXT_COUNT=$(run_psql "SELECT COUNT(*) FROM pg_extension WHERE extname IN ('vector','pg_trgm','btree_gist','pg_stat_statements');")
if [ "$EXT_COUNT" -ge 4 ]; then
    ok "Extensions OK ($EXT_COUNT/4 Standard-Extensions vorhanden)"
else
    warn "Nur $EXT_COUNT/4 Extensions -- pg_search evtl. nicht verfuegbar (das ist normal, siehe Hinweis unten)"
fi

# Schemas
info "Pruefe Schemas..."
SCHEMAS=$(run_psql "SELECT string_agg(schema_name, ', ' ORDER BY schema_name) FROM information_schema.schemata WHERE schema_name IN ('raw','processed','mlflow');")
if echo "$SCHEMAS" | grep -q "raw" && echo "$SCHEMAS" | grep -q "processed"; then
    ok "Schemas OK: $SCHEMAS"
else
    warn "Schemas fehlen: $SCHEMAS -- pruefe docker logs vigilex-postgres"
fi

# Tabellen
info "Pruefe Tabellen..."
TABLE_COUNT=$(run_psql "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema IN ('raw','processed');")
if [ "$TABLE_COUNT" -ge 5 ]; then
    ok "Tabellen OK ($TABLE_COUNT Tabellen in raw + processed)"
else
    warn "Nur $TABLE_COUNT Tabellen -- init.sql evtl. nicht vollstaendig gelaufen"
    info "Manuell pruefen: docker exec -it vigilex-postgres psql -U $POSTGRES_USER -d vigilex"
    info "  \\dt raw.*"
    info "  \\dt processed.*"
fi
echo ""

# Einzelne Tabellen
info "Tabellen-Uebersicht:"
run_psql "SELECT table_schema || '.' || table_name FROM information_schema.tables WHERE table_schema IN ('raw','processed') ORDER BY table_schema, table_name;" | while read -r tbl; do
    info "  $tbl"
done
echo ""


# =============================================================================
# SCHRITT 7 -- Grafana + Ollama check
# =============================================================================
echo "--- SCHRITT 7: Grafana + Ollama ---"
echo ""

# Grafana
GRAFANA_STATUS=$(docker inspect --format='{{.State.Status}}' vigilex-grafana 2>/dev/null || echo "not_found")
if [ "$GRAFANA_STATUS" = "running" ]; then
    ok "Grafana laeuft -- erreichbar unter: http://$(hostname -I | awk '{print $1}'):3000"
else
    warn "Grafana Status: $GRAFANA_STATUS"
fi

# Ollama
OLLAMA_STATUS=$(docker inspect --format='{{.State.Status}}' vigilex-ollama 2>/dev/null || echo "not_found")
if [ "$OLLAMA_STATUS" = "running" ]; then
    ok "Ollama laeuft"

    # Welche Modelle sind geladen?
    MODELS=$(docker exec vigilex-ollama ollama list 2>/dev/null | tail -n +2 | awk '{print $1}' | tr '\n' ' ' || echo "keine")
    if [ -n "$MODELS" ] && [ "$MODELS" != "keine" ]; then
        ok "Ollama Modelle: $MODELS"
    else
        warn "Ollama: kein Modell geladen"
        info "llama3.2:3b pullen:"
        info "  docker exec vigilex-ollama ollama pull llama3.2:3b"
    fi
else
    warn "Ollama Status: $OLLAMA_STATUS"
fi
echo ""


# =============================================================================
# ERGEBNIS + NAECHSTE SCHRITTE
# =============================================================================
echo "============================================================"
echo "  Migration abgeschlossen"
echo "============================================================"
echo ""
echo -e "${GREEN}Laufende Services:${NC}"
docker compose ps --format "  {{.Name}}: {{.Status}}" 2>/dev/null
echo ""
echo -e "${YELLOW}Naechste Schritte (Block C):${NC}"
echo ""
echo "  1. Falls Ollama-Modell fehlt:"
echo "       docker exec vigilex-ollama ollama pull llama3.2:3b"
echo ""
echo "  2. MAUDE Ingestion Worker schreiben:"
echo "       src/vigilex/workers/ingest.py"
echo "       src/vigilex/data/maude_client.py"
echo ""
echo "  3. Erster Test-Pull (Insulinpumpen, 2024):"
echo "       python -m vigilex.data.maude_client --product-code LZG --year 2024"
echo ""
echo "  4. API + Worker Container bauen (sobald src/ Code vorhanden):"
echo "       docker compose up -d --build api worker-ingest"
echo ""
echo "  5. Notebook 02 lokal ausfuehren (recall_labels, unabhaengig vom Server):"
echo "       jupyter lab notebooks/02_recall_labels.ipynb"
echo ""
echo "  HINWEIS pg_search:"
echo "  pg_search ist eine ParadeDB-Extension, nicht im Standard pgvector-Image."
echo "  Entweder: ParadeDB-Image verwenden (pgvector/pgvector austauschen),"
echo "  oder: pg_search aus init.sql entfernen (pg_trgm + tsvector genuegen)."
echo ""

exit 0
