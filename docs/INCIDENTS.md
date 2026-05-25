# SentinelAI (vigilex) -- Incident Log

> **Hinweis:** Diese Datei bleibt als Archiv erhalten.
> Aktuelle Kopie fuer schnellen Zugriff: `CAPSTONE II/BUG_HISTORY.md` (root)



Zusammenfassung aller kritischen Produktionsvorfaelle waehrend Capstone II.
Quellen: session_2026-05-18_prr_setup.md, session_2026-05-18_evening_diagnostics.md,
session_2026-05-19_morning_fallback_analysis.md, sentinelai_debug_summary_2026-05-19.md
(alle vier Dateien in alt/session-logs/ archiviert)

---

## Incident #1: Massen-Fallback LZG (2026-05-13)

**Zeitraum:** 2026-05-11 18:00 UTC bis 2026-05-13 09:00 UTC
**Auswirkung:** 4740 von 5832 LZG-Records (82,9%) mit Fallback-Wert statt LLM-Output

### Symptom
Spike bei final_confidence=0.21 im Pairplot. value_counts(llm_confidence):
82,9% exakt bei 0.30 -- dem hardcoded Fallback-Sentinel-Wert bei LLM-Exception.

### Diagnose-Reihenfolge (90 Min, 2026-05-13 Vormittag)
1. Pairplot -> Spike bei 0.21
2. value_counts(llm_confidence) -> 82,9% exakt 0.30
3. Code-Audit llm_coder.py -> 0.30 ist hardcoded Exception-Fallback
4. Hourly-Stratifizierung von coded_at:
   - 11.05 11:00-17:00 UTC: 0% Fallback (Healthy Window, ~890 Records)
   - 11.05 18:00-21:00 UTC: ~85% Fallback (Degradation)
   - 12.05 09:00 bis 13.05 09:00 UTC: 100% Fallback
5. Worker-Container CreatedAt: 2026-05-12 09:25:02 UTC -- exakter Cutover
6. docker logs vigilex-worker-coding:
   "HTTPConnectionPool(host='localhost', port=11434): Max retries exceeded
    Caused by NewConnectionError: [Errno 111] Connection refused"

### Root Cause
`llm_coder.py`: OLLAMA_DEFAULT_URL = "http://localhost:11434" hardcoded.
Im Container zeigt localhost auf den Container selbst, nicht auf den Ollama-Container.
Env-Var OLLAMA_BASE_URL wurde nicht gelesen.

### Fix
- OLLAMA_BASE_URL aus Env-Var lesen
- VIGILEX_STRICT=True fuer Fail-fast (kein "Erfolg" bei Fallback)
- 17 Code-Edits in 4 Dateien
- DB bereinigt (coding_results_pre_fix Backup, TRUNCATE)
- Re-Coding-Run gestartet -> 7324 echte LLM-Codings fuer LZG

### Lesson
Fail-fast in dev, fail-soft in prod hat einen Preis: Beobachtungsluecke.
Pattern: VIGILEX_STRICT=True bei manuellen Runs, False nur in echtem Prod-Mode.
Containerisierung nicht vor sauberem Code-Pfad.

---

## Incident #2: Multi-Worker-Konflikt + Zombie-Ingest (2026-05-18 Abend)

**Zeitraum:** 2026-05-18 17:00-20:05 UTC
**Auswirkung:** 100% Fallback-Rate fuer ca. 3 Stunden, ~178 Muell-Records

### Symptom
LLM coding failed (HTTPConnectionPool: Read timed out, read timeout=60).
Records mit llm_confidence=NULL. Worker brach nicht ab (VIGILEX_STRICT=False).

### Initiale Hypothesen (alle falsch, 4 Stunden Container-Diagnose)
- QFG-Narratives laenger -> Memory-Bandwidth-Bottleneck
- Ollama-Container strukturelles Problem -> Restart
- Hetzner CPU-Steal-Time -> Provider drosselt
- Postgres-Autovacuum -> Background-Indexing

### Methodische Wende
Thomas: "das ist ja alles auf hetzner, kann man da nicht schauen, was los ist?"
-> Sofortige Host-Level-Diagnose:

```
uptime: load average 7.72, 8.87, 9.50  (bei 4 CPUs -- 2x ueberlastet)

ps -ef | grep docker:
  PID 1408034 -- docker compose run --rm worker-ingest (seit 16:58 UTC, 249 Min CPU)

ps -ef | grep python3:
  PID 2728171 -- python3 -m vigilex.workers.coding (seit Fr 15.05 14:37, tmux 'coding-weekend')
```

### Root Causes (zwei parallel)
1. Zombie-Ingest: hetzner_switch_to_qfg_oyc.sh mit `docker compose run -d` --
   CLI-Prozess bleibt blocking trotz -d Flag. Brannte ~100% CPU, nicht in `docker ps` sichtbar.
2. Multi-Worker: Host-Worker (tmux) + Docker-Worker (compose) liefen gleichzeitig.
   Beide schickten Ollama-Calls auf 4-vCPU-Hardware -> 60s-Timeouts -> HTTP 500 -> Fallback.

### Timeline (UTC)
| Zeit | Event |
|------|-------|
| Fr 15.05 14:37 | Host-Worker tmux 'coding-weekend' gestartet |
| Mo 16:58 | Switch-Script, Zombie-Ingest entsteht |
| Mo 17:01 | Erste Ollama HTTP 500 (60s Timeout) |
| Mo 17:00-19:00 | 100% Fallback, ~178 Muell-Records |
| Mo 19:00 | Wende: Host-Level-Diagnose |
| Mo 19:15 | Zombie gekillt (kill -9 1408034) |
| Mo 20:05 | Host-Worker gestoppt (kill -INT 2728171) |
| Mo 20:06 | Ollama Smoke-Test: 1.19s, gesund |
| Mo 20:15 | DB-Cleanup: ~178 NULL/Fallback-Records geloescht |
| Mo 20:23 | Worker neu gestartet, Single-Instance |
| Mo 20:24 | Erste echte QFG-Codings: conf=0.561-0.593, ~35s/Record |

### Fix
- Beide Worker gestoppt
- DB bereinigt (178 Muell-Records geloescht)
- Worker als Single-Instance neu gestartet
- Single-Instance-Policy dokumentiert (CLAUDE.md + memory)

### Lessons
1. **Host zuerst:** Bei Service-Performance ZUERST uptime/top/ps, DANN Container-Logs.
2. **Single-Instance-Policy:** NIE Host-Worker und Docker-Worker gleichzeitig.
   Vor jedem Coding-Run pruefen:
   ```
   ps -ef | grep "vigilex.workers.coding" | grep -v grep
   docker ps | grep worker-coding
   tmux ls
   ```
3. **`docker compose run -d` ist ein Fallstrick:** CLI bleibt blocking trotz -d.
   Besser: `docker compose up -d worker-ingest` oder `nohup ... & disown`.

### Standard-Diagnose-Sequenz (fuer kuenftige Incidents)
```bash
uptime                                  # load avg vs nproc
top -bn1 -o %CPU | head -15            # wer brennt CPU?
ps -ef | grep <service> | grep -v grep # vergessene Prozesse?
vmstat 1 5                             # CPU-Steal (st), Swap (si/so)
free -h                                # RAM
```

---

## Incident #3: PRR/ROR lieferte nur None-Werte (2026-05-18 Vormittag)

**Zeitraum:** 2026-05-18 Vormittag
**Auswirkung:** threshold_scan.py: 430 PT-Kombinationen, alle prr=None

### Symptom
```
Total results: 430
Results with PRR not None: 0
Sample row: {'prr': None, 'is_signal': False, ...}
```

### Root Cause
Nur ein Produkt-Code (LZG = Insulinpumpen) in processed.coding_results.
PRR-Formel benoetigt Hintergrundpopulation aus anderen Device-Typen.
Mit nur LZG: n_all_devices_total == n_reports_focal -> Division by Zero -> None.

### Fix
QFG (CGM-Sensoren) Ingest gestartet: ~20.000 Records ingested in raw.maude_reports.
OYC (Pacemaker) Versuch: nur 17 Records (openFDA-API zu langsam, abgebrochen).
Nach QFG-Coding und Re-Run von threshold_scan.py: > 0 Signals.

### Lesson
PRR/ROR braucht mindestens 2 verschiedene Device-Typen als Komparator-Population.
Mit nur einem Device-Typ ist der Algorithmus mathematisch nicht ausfuehrbar.

---

## Incident #4: QFG Fallback-Verdacht (2026-05-19) -- KEIN Incident

**Zeitraum:** 2026-05-19 morgens
**Status:** Kein echter Fehler. Fehlinterpretation korrigiert.

### Initialer Verdacht
Nach 12h QFG-Overnight-Run: 68% llm_confidence=0.50 exakt, 22% llm_confidence=0.80 exakt.
Kein Wert zwischen 0.5 und 0.7. Verdacht: Rueckkehr des alten Fallback-Bugs.

### Diagnose (2026-05-19)
Checks durchgefuehrt:
1. Worker-Topology: nur Host-Worker aktiv, kein Docker-Worker -- kein Multi-Worker-Konflikt
2. Ollama-Health: 28.8s (kurz erhoeht), spaeter 1.2s -- kein persistentes Problem
3. Host-Ressourcen: Load avg 1.5-2.1, RAM OK -- kein Ueberlastungsproblem
4. Import-Pfad: PYTHONPATH korrekt, STRICT_MODE aktiv -- kein Stale-Code
5. Kontrollierter 5-Record-Run mit VIGILEX_STRICT=True: kein Abbruch, HTTP 200, echte LLM-Calls

### Korrektur
```
Alt: "92% Fallback"
Korrekt: "Prompt v1 induziert grobe ordinale LLM-Konfidenzwerte"
```

SYSTEM_PROMPT instruiert: "1.0=perfect match, 0.5=uncertain, <0.3=flag for review".
Das lokale llama3.2:3b emittiert daher bevorzugt Ankerwerte (0.5, 0.8, 1.0).
0.5 und 0.8 sind ECHTE LLM-Outputs, keine Exception-Fallback-Marker.
Gleiches Muster bereits in LZG vorhanden (55% bei 0.5, 27% bei 0.8).

### Folge-Entscheidung (dauerhaft)
Prompt, Formel, Modell bleiben unveraendert bis Capstone-Ende.
Begruendung: >7000 LZG-Records unter Prompt v1 codiert -- Aenderung wuerde
Vergleichbarkeit zwischen LZG und QFG brechen.

Sprachregelung: "ranking index" statt "confidence score". Detailliert in DECISIONS.md.

### Technisches Tech-Debt aus diesem Incident
- model_version='pipeline_v1' zu generisch -- kein Unterschied ollama-success vs fallback
- llm_confidence=0.3/0.5/0.8 als Sentinel-Werte kollidieren mit echten LLM-Outputs
  (Fix: separate Spalte fallback_reason, llm_confidence nur bei echtem LLM-Aufruf)
- Schema-Erweiterung um Trace-Spalten (coding_path, llm_backend, llm_status etc.)
  ab 19.05 implementiert -- nur neue QFG-Records haben sauberes Status-Logging
- scripts/load_host_env.sh sollte erstellt werden, um Env-Drift Host vs. Docker zu vermeiden

---

## Wiederkehrende Diagnose-Checkpoints

Vor jedem Coding-Run auf Hetzner:
```bash
# 1. Single-Instance-Check
ps -ef | grep "vigilex.workers.coding" | grep -v grep
docker ps | grep worker-coding
tmux ls

# 2. Ollama-Health
time curl -s -m 30 http://localhost:11434/api/chat \
  -d '{"model":"llama3.2:3b","messages":[{"role":"user","content":"hi"}],"stream":false}' \
  | python3 -c "import sys,json; r=json.load(sys.stdin); print(r['message']['content'])"

# 3. Worker-Status nach Run
ps -ef | grep "vigilex.workers.coding" | grep -v grep
tmux capture-pane -t coding-weekend:0 -p | tail -30
docker exec vigilex-postgres psql -U vigilex -d vigilex -c \
  "SELECT COUNT(*), MIN(coded_at), MAX(coded_at), ROUND(AVG(llm_confidence)::numeric,3)
   FROM processed.coding_results WHERE coded_at >= NOW() - INTERVAL '1 day';"
```
