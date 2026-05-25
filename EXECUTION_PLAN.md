# Execution Plan -- Bench V1 + Migration

**Datum:** 2026-05-25
**Status:** Aktiv
**Bezug:** VECTOR_MIGRATION_PLAN.md (Plan), BUG_VECTOR_SEARCH.md (Diagnose)
**Ziel:** Bench V1 + ggf. Migration spaetestens bis 2026-05-28 abend abgeschlossen, damit 29.05 Talk-Vorbereitung sauber laeuft.

---

## 0. Tooling-Verteilung (konkretisiert fuer Setup Thomas)

| Mode | Host | Wofuer | Aufrufweise |
|---|---|---|---|
| **Cowork (Sonnet)** | Anthropic-Cloud | Strategie, Review, Doku, Resultate interpretieren, ggf. Sample-Audit via Anthropic-API | Chat-Oberflaeche in der Cowork-App |
| **Claude Code** | Windows lokal (PowerShell) | Skripte schreiben, git-Ops, SSH-Sessions zu Hetzner starten, kleine lokale Tests | `claude` in PowerShell, im Repo-Folder oder ueberall |
| **Dispatch (App-Agent)** | Anthropic-Cloud | Asynchrone Reviewer-Aufgaben, LLM-Judge-Calls fuer Sample-Audit, Logfile-Monitoring | Agent-Tool aus Cowork heraus aufgerufen |
| **Hetzner-Server** | 46.225.109.99 | Lange Compute-Laeufe (Bench, Re-Embedding, Re-Eval) | SSH + tmux, ausgeloest aus Claude Code in PowerShell |

**Wichtige Klarstellung:**

- Cowork-Dispatch-Agents haben KEIN Compute fuer SentenceTransformer auf deinem PC oder Hetzner. Sie laufen in Anthropic-Sandboxes. Brauchbar fuer: API-Calls (LLM-Judge), Workspace-File-Lesen, Synthese-Aufgaben. NICHT brauchbar fuer: Bench-Lauf, Re-Embedding.
- Lange Compute-Laeufe laufen auf Hetzner. Du startest sie aus Claude Code in PowerShell via SSH+tmux. Hetzner laeuft ueber Nacht weiter, dein Windows-PC kann aus.

**Regel:** Cowork plant + reviewed, Claude Code (lokal in PS) orchestriert + entwickelt, Hetzner rechnet, Cowork-Dispatch macht API-Reviews wo keine lokale Compute noetig ist.

---

## 1. Phasen-Plan

### Phase 1 -- heute (2026-05-25 nachmittag/abend)
**Modus:** Claude Code (PowerShell, interaktiv) + SSH zu Hetzner
**Dauer:** 1-2h
**Ziel:** Skript laeuft sauber mit EINER Konfiguration durch (auf Hetzner).
**Workflow:** Claude Code editiert lokal, `git push origin work`, Hetzner zieht via `git pull`, Skript auf Hetzner testen.

Tasks:
1. Schema verifizieren: `processed.meddra_terms`, `processed.meddra_llt` Spaltennamen, Tabellen-Pfade
2. Pool-SQLs ausfuehren und Sample-Output ansehen (10 Zeilen)
3. Status-quo-Baseline messen (PubMedBERT-base via Produktionscode, vector-only, Recall@100 -> `data/eval/status_quo_baseline.json`)
4. Bench-Skelett `scripts/bench_embedding_models.py` mit nur:
   - 1 Modell: all-MiniLM-L6-v2
   - 1 Pool: pt_only
   - 1 Query-Variante: first_sentence
5. Sanity-Check assert einbauen
6. Detail-CSV + Summary-CSV-Format pruefen (eine Zeile schreiben, anschauen)
7. Caching-Logik einbauen, einmal mit Cache-Hit testen

**Sync-Point:** Bevor Phase 2 startet -- du siehst die erste Summary-Zeile, plausible Zahlen, Detail-CSV-Spalten sind alle da, `query_text_used` zeigt sinnvollen Text. Wenn nicht: hier fixen, nicht in Phase 2.

### Phase 2 -- heute abend (2026-05-25 nach Phase 1)
**Modus:** Hetzner via SSH+tmux (Background, ueber Nacht)
**Dauer:** 2-5h Laufzeit
**Ziel:** Volle Bench-Matrix mit 3 Modellen x 2 Pools x 2 Queries = 12 Konfigurationen.

**Start-Befehl (aus PowerShell heraus):**
```powershell
# 1. SSH zum Hetzner-Server
ssh cap@46.225.109.99

# 2. Auf Hetzner: tmux session starten
tmux new -s bench

# 3. In tmux: Skript starten mit Logging
cd ~/vigilex
git pull origin work
python scripts/bench_embedding_models.py --full 2>&1 | tee data/eval/bench_run.log

# 4. Detach: Ctrl+B dann d
# Damit laeuft Skript weiter, du kannst SSH-Session schliessen
```

**Status-Check am naechsten Morgen:**
```powershell
ssh cap@46.225.109.99
tmux attach -t bench       # zeigt aktuellen Stand
# oder ohne attach:
tail -100 ~/vigilex/data/eval/bench_run.log
ls -la ~/vigilex/data/eval/cache/   # Cache-Files pruefen
cat ~/vigilex/data/eval/bench_results_summary.csv
```

**Resultate-Transfer fuer Cowork-Review (Phase 3):**
- Optionen, je nachdem was schneller ist:
  - rclone-Sync nach Google Drive laeuft eh (laut CLAUDE.md) -> CSV liegt morgen frueh in Google Drive
  - `scp cap@46.225.109.99:~/vigilex/data/eval/bench_results_summary.csv ./local/path/`
  - manuell via git (CSV ins Repo committen) -- aber dann unbedingt .gitignore pruefen

Tasks (Skript laeuft selbst, autonom):
1. Modelle herunterladen (falls noch nicht im HF-Cache des Hetzner-Users)
2. Doc-Embeddings je Modell+Pool berechnen und cachen
3. Pro Modell+Pool+Query: Recall@K + Detail-CSV
4. Summary-CSV + Detail-CSV finalisieren
5. Logfile `data/eval/bench_run.log` schreiben

**Failure-Recovery (siehe Abschnitt 3):**
- Bei Crash: Skript laesst sich neu starten, Caching greift, fertige Konfigurationen werden uebersprungen.
- tmux ueberlebt SSH-Disconnect; PC kann aus.
- Hetzner-Reboot: tmux-Session weg, aber Caching macht Re-Start guenstig.

**Single-Instance-Policy beachten:** Wenn auf Hetzner schon ein Coding-Worker im tmux laeuft (siehe Memory `vigilex_worker_topology`), Bench in eigene Session (`bench`, nicht in vorhandene Session reinhaengen). Mit `tmux ls` vor Start pruefen.

### Phase 3 -- morgen frueh (2026-05-26 vormittag)
**Modus:** Cowork (interaktiv)
**Dauer:** 30 min
**Ziel:** Entscheidung -- Migration ja/nein, welche Konfiguration.

Tasks:
1. Summary-CSV ansehen, Pool-Effekt-Pivot, Query-Effekt-Pivot
2. Status-quo-Baseline gegenueberstellen
3. Entscheidungstabelle aus VECTOR_MIGRATION_PLAN.md Abschnitt 5 anwenden
4. Falls Migration: Konfiguration festziehen (Modell + Pool + Query)
5. Falls nicht: Vector-Arm-Strategie festlegen (runtergewichten oder deaktivieren)

**Sync-Point:** Klare Entscheidung schriftlich in `data/eval/DECISION.md` (1 Absatz reicht).

### Phase 4 -- morgen Tag (2026-05-26 nachmittag, falls Migration)
**Modus:** Claude Code in PowerShell (Entwicklung) + Hetzner tmux (Re-Embedding)
**Dauer:** 2-4h
**Ziel:** Pipeline auf Gewinner-Konfiguration umgestellt.

Tasks (Claude Code lokal, dann via SSH):
1. Schema-Migration falls Dim != 768: SQL-File `migrations/00X_vector_v2.sql` erstellen, lokal reviewen, dann auf Hetzner ausfuehren via `ssh cap@... 'psql ... < migration.sql'`
2. `embed_meddra_terms_v2.py` lokal entwickeln, git push
3. `hybrid_search.py` Query-Seite umstellen, identische Settings, git push
4. pgvector IVFFlat-Index-Create-SQL ebenfalls als Migration

Tasks (Hetzner tmux):
5. `tmux new -s reembed` -> `git pull && python scripts/embed_meddra_terms_v2.py` (10-30 min auf CPU)
6. CREATE INDEX in psql
7. Re-Eval `eval_golden_set.py` zur Bestaetigung

**Sync-Point:** Recall@100 auf Golden Set ist nachweislich >= Bench-Bestwert.

### Phase 5 -- 27./28.05 (Mi/Do)
**Modus:** Hetzner tmux (Sample-Export + Re-Coding) + Cowork-Dispatch-Agent (LLM-Judge via Anthropic-API) + Cowork (Interpretation)
**Dauer:** 1h aktive Arbeit, 4-6h Background
**Ziel:** Sample-Audit der 11k codierten Records + Eval-Final-Lauf.

Tasks (Hetzner tmux):
1. Sample-Export: 100 Random Records aus `processed.coding_results`, stratified by device_type (33/33/34) -> CSV nach Google Drive sync
2. Falls neue Pipeline aktiv: Re-Coding desselben Samples mit neuer Pipeline (auch auf Hetzner) -> zweite CSV

Tasks (Cowork-Dispatch-Agent, Anthropic-API):
3. Agent liest beide CSVs aus dem Workspace-Folder (Google-Drive-synced), ruft Sonnet als Judge fuer jeden Record mit Prompt "Is the MedDRA PT clinically correct for this narrative? YES/NO/PARTIALLY, brief rationale" auf, aggregiert YES%/NO%/PARTIALLY% mit 95% CI
4. Wenn beide Pipelines verglichen werden: Agreement-Matrix alt vs. neu

Tasks (Cowork, du + ich):
5. Resultate interpretieren, Caveats fuer Final Presentation festziehen

---

## 2. Prompt-Templates

### Template fuer Phase 1 (Claude Code, Schritt fuer Schritt)

```
Kontext: Lies vigilex/VECTOR_MIGRATION_PLAN.md vollstaendig, danach
vigilex/BUG_VECTOR_SEARCH.md fuer den Hintergrund.

Aufgabe Phase 1, Schritt 1 (Schema-Verification):
- Verifiziere die Spaltennamen in processed.meddra_terms und
  processed.meddra_llt (per psql oder Python).
- Wenn die Tabellennamen abweichen vom Plan, melde es bevor du
  weitermachst.
- Zeig mir das tatsaechliche Schema dieser zwei Tabellen.

Stop hier, ich review.
```

Folgeprompt nach OK:

```
Aufgabe Phase 1, Schritt 2 (Pool-SQLs):
- Implementiere die zwei Pool-Loading-Funktionen aus dem Plan
  (pt_only und pt_limited_llt).
- Fuehre beide aus, zeig mir die ersten 10 Zeilen jeder Variante.
- Achte auf den WHERE-Filter llt_name != pt_name.
```

Folgeprompt:

```
Aufgabe Phase 1, Schritt 3 (Status-quo-Baseline):
- Modifiziere eval_golden_set.py so, dass nur der Vector-Arm
  gemessen wird (BM25 + CrossEncoder skip).
- Lass das Skript einmal laufen mit top_k_stage1=100.
- Schreib das Recall@100, median_rank_found, not_found_count
  nach data/eval/status_quo_baseline.json.
- Zeig mir die Zahlen und den Patch, bevor du committest.
```

Folgeprompt:

```
Aufgabe Phase 1, Schritt 4-7 (Bench-Skelett + Caching):
- Implementiere scripts/bench_embedding_models.py gemaess
  VECTOR_MIGRATION_PLAN.md Abschnitt 3.
- Aber nur fuer eine Konfiguration: MiniLM-L6-v2, pt_only,
  first_sentence.
- Sanity-Check assert vor dem Modell-Run.
- Detail-CSV-Spalten alle wie spezifiziert.
- Caching mit Pfadschema aus Anhang 9.1.
- Lass das Skript einmal mit Cache-Miss laufen, einmal mit Cache-Hit
  (zweiter Run soll Doc-Embeddings nicht neu rechnen).
- Zeig mir Summary-Zeile + erste 3 Zeilen der Detail-CSV.
```

### Template fuer Phase 2 (Hetzner tmux, autonomes Skript)

**Im Skript am Anfang einbauen (Logging-Setup):**
```python
import logging
from pathlib import Path

log_dir = Path("data/eval")
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=str(log_dir / "bench_run.log"),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
```

**Start-Sequenz (aus PowerShell):**

```powershell
# 1. Vor dem Start: tmux-Sessions auf Hetzner pruefen
ssh cap@46.225.109.99 "tmux ls"
# Wenn 'coding-worker' o.ae. laeuft: Single-Instance-Policy beachten,
# Bench in EIGENE Session 'bench' starten, nicht in vorhandene.

# 2. SSH und tmux-Setup
ssh cap@46.225.109.99

# Ab hier auf Hetzner:
cd ~/vigilex
git pull origin work     # holt neuestes bench_embedding_models.py

tmux new -s bench
# in tmux:
python scripts/bench_embedding_models.py --full \
    2>&1 | tee data/eval/bench_stdout.log

# Detach: Ctrl+B dann d
# SSH-Session schliessen, PC kann aus.
```

**Status-Check (jederzeit, aus PowerShell):**
```powershell
# Schnellcheck ohne attach:
ssh cap@46.225.109.99 "tail -50 ~/vigilex/data/eval/bench_run.log"

# Cache-Files anschauen (welche Konfigurationen fertig sind):
ssh cap@46.225.109.99 "ls -la ~/vigilex/data/eval/cache/"

# Zwischenstand der Summary-CSV:
ssh cap@46.225.109.99 "cat ~/vigilex/data/eval/bench_results_summary.csv"

# Vollstaendiges Re-Attach falls noetig:
ssh cap@46.225.109.99
tmux attach -t bench
```

**Skript-Logik (im Code):**
- For each (model, pool_type, query_field):
  - Check ob Summary-CSV-Zeile schon existiert -> skip (Idempotenz)
  - Check ob Doc-Embedding-Cache existiert -> load, sonst encode
  - Encode queries, compute metrics, append to CSVs
  - Log progress mit Zeitstempel
- Bei Exception in einer Konfiguration: catch, log, continue mit naechster Konfig.

**Resultate nach Cowork transferieren:**
- rclone-Sync laeuft eh (CLAUDE.md): CSVs landen morgen frueh in Google Drive
- Falls schneller noetig: `scp cap@46.225.109.99:~/vigilex/data/eval/bench_results_*.csv ./local/`

### Template fuer Phase 3 (Cowork-Review)

```
Aufgabe: Bench-Resultate reviewen.

Inputs:
- data/eval/bench_results_summary.csv
- data/eval/bench_results_detailed.csv
- data/eval/status_quo_baseline.json

Bitte:
1. Pool-Effekt-Pivot anzeigen (recall@100 pro Modell x Pool-Typ)
2. Query-Effekt-Pivot anzeigen
3. Top-5 Konfigurationen nach recall@100 anzeigen
4. Gegen Status-quo-Baseline vergleichen (Delta in pp)
5. Entscheidungstabelle aus VECTOR_MIGRATION_PLAN.md Abschnitt 5
   anwenden -- welcher Befund liegt vor?
6. Konkrete Migrations-Empfehlung (Modell, Pool, Query) oder
   Begruendung warum keine Migration.
7. Schreibe data/eval/DECISION.md mit der Entscheidung.
```

### Template fuer Phase 4 (Claude Code, Migration)

```
Kontext: data/eval/DECISION.md enthaelt die Migrationsentscheidung.
VECTOR_MIGRATION_PLAN.md Abschnitt 4 Schritt 8 beschreibt die Schritte.

Aufgabe Phase 4:
1. Lies DECISION.md -- wenn keine Migration, stoppe.
2. Schema-Migration: neue Spalte embedding_v2 in processed.meddra_terms
   mit korrekter Dimension. Zeig mir die Migration-SQL bevor du sie
   ausfuehrst.
3. Skript embed_meddra_terms_v2.py erstellen, das mit
   SentenceTransformer arbeitet, Pool-Repraesentation gemaess
   DECISION.md verwendet.
4. Lass es einmal mit 100 PTs probelaufen, zeig mir die Embeddings-
   Shape und ein paar Sample-Vektoren (norm pruefen).
5. Wenn ok: voller Run fuer alle 27k PTs (DIESEN Run als Dispatch
   starten, du wartest nicht).
6. pgvector-Index neu bauen: ich gebe dir die SQL.
7. hybrid_search.py Query-Seite anpassen, identische Settings wie
   Index-Seite, Test mit DKA-Case.

Stop nach jedem Schritt fuer Review.
```

### Template fuer Phase 5 (Hetzner-Export + Cowork-Agent fuer LLM-Judge)

**Schritt 1 (Hetzner via SSH): Sample-Export**

```bash
# In tmux session 'audit' auf Hetzner
psql -d vigilex -c "
COPY (
  SELECT id, mdr_text, assigned_pt_code, assigned_pt_name, device_type
  FROM processed.coding_results
  WHERE device_type IN ('LZG','OYC','QFG')
  ORDER BY device_type, RANDOM()
  LIMIT 100
) TO '/tmp/sample_audit_100.csv' WITH (FORMAT csv, HEADER true);
"
# Stratified ueber device_type sicherstellen, evtl. via 3 separate Queries
# mit LIMIT 33/33/34 und UNION
```

CSV in Google Drive sync, dann Workspace-Folder lesbar fuer Cowork-Agent.

**Schritt 2 (Cowork, hier in der App): Agent fuer LLM-Judge spawnen**

Prompt fuer den Agent:
```
Du bist Reviewer fuer MedDRA-Coding-Qualitaet.

Lies die CSV unter C:\Users\thheg\bootcamps\CAPSTONE II\vigilex\data\eval\sample_audit_100.csv
(via Read tool, falls in Workspace gemountet).

Fuer jeden Record:
- Eingabe: mdr_text, assigned_pt_name, assigned_pt_code
- Aufgabe: bewerte, ob das zugewiesene MedDRA Preferred Term klinisch
  korrekt fuer die beschriebene Adverse-Event-Narrative ist.
- Output pro Record:
  - verdict: YES / NO / PARTIALLY
  - rationale: 1-2 Saetze begruendet

Aggregiere am Ende:
- Anzahl YES, NO, PARTIALLY
- Prozentsaetze
- 95% Konfidenzintervall fuer YES-Rate (Wilson Score)
- Aufschluesselung nach device_type

Speichere Resultate als data/eval/audit_old_pipeline.json in dem Format:
{
  "n": 100,
  "yes": ..., "no": ..., "partially": ...,
  "yes_rate": ..., "ci_95_lower": ..., "ci_95_upper": ...,
  "by_device": { "LZG": {...}, "OYC": {...}, "QFG": {...} },
  "details": [ {id, verdict, rationale}, ... ]
}
```

**Schritt 3 (falls neue Pipeline aktiv, Hetzner): Re-Coding desselben Samples**

```bash
# in tmux 'audit'
python scripts/recode_sample.py \
    --input /tmp/sample_audit_100.csv \
    --output /tmp/sample_recoded_100.csv \
    --use-new-pipeline
```

Dann Cowork-Agent wieder, mit beiden CSVs als Input, Agreement-Matrix bauen.

---

## 3. Failure Modes und Recovery

| Failure | Symptom | Recovery |
|---|---|---|
| Modell-Download schlaegt fehl | `OSError: Cannot reach huggingface.co` | Vorab `huggingface-cli download <model>` ausfuehren; alternativ in Skript: catch, log, skip dieses Modell, weiter mit naechstem |
| Bench-Skript crasht mitten in Konfig | Letzte Konfig hat keine Summary-Zeile | Re-Start: Skript checkt Summary-CSV, ueberspringt fertige Konfigs, Caching greift fuer Doc-Embeddings |
| Out of Memory bei mpnet/PubMedBERT-Varianten | `RuntimeError: out of memory` | Batch-Size reduzieren (default 64 -> 16); bei mpnet ggf. CPU only erzwingen |
| Pool-SQL fehlerhaft (z.B. doppelte PT-Codes) | Sanity-Check assert oder doppelte Zeilen in pool | Schema in Repo nachschlagen, SQL korrigieren, NICHT raten |
| Expected codes nicht im Pool | Assertion-Error in Phase 1 | Golden-Set-Code-Werte mit MedDRA-Tabelle abgleichen; entweder Pool-Loading falsch oder Golden-Set hat Tippfehler |
| Status-quo-Baseline-Modifikation bricht Produktion | eval_golden_set.py funktioniert nicht mehr | Eigenes Skript scripts/baseline_vector_only.py erstellen, Produktionscode nicht patchen |
| pgvector-Dim-Migration scheitert | Spalte existiert schon | Migration mit `ALTER TABLE ... DROP COLUMN IF EXISTS embedding_v2`; vorher Backup |
| Re-Embedding zu langsam | mpnet >2h fuer 27k | Auf MiniLM/bge-small ausweichen falls Bench die als competitive zeigt |
| Dispatch-Skript laeuft nicht zu Ende | tmux session gestorben, ueber Nacht weg | Skript idempotent gebaut -- naechster Start macht weiter; Logfile zeigt wo abgebrochen |

---

## 4. Sync-Points (was muss in Cowork reviewed werden)

| Sync-Point | Wann | Was zeigen |
|---|---|---|
| Phase 1 -> Phase 2 | nach Phase 1, vor Dispatch-Start | Schema-Verifikation, 1 Konfig durchgelaufen, Sample-Detail-CSV |
| Phase 2 -> Phase 3 | morgens nach Bench-Lauf | Summary-CSV (komplett), Logfile |
| Phase 3 -> Phase 4 | nach Entscheidung | DECISION.md |
| Phase 4 -> Phase 5 | nach Migration | Recall@100 nach Migration vs. vor Migration |

Zwischen den Sync-Points: Claude Code / Dispatch autonom, kein Cowork-Eingriff noetig.

---

## 5. Was bewusst NICHT in V1 ist

(Zur Erinnerung, damit Scope-Creep ausbleibt)

- acceptable_pt_codes-Annotation (V2)
- clinical_window Query-Variante (V2)
- Keyword-Extractor (V2)
- MLflow-Integration des Benchs (V2)
- LLT als eigenes Dokument indizieren (V2)
- 4. und 5. Modell ergaenzen (V2)
- Voller 2015-2024 Import (gestrichen)

Wenn V1 ein klares Ergebnis liefert: V2-Material wird nur bedient, falls bis 02.06 Zeit ist. Sonst: Final Presentation mit V1-Resultaten.

---

## 6. Zeitbudget realistisch

| Phase | Aktive Zeit Thomas | Background-Zeit | Wenn alles glatt |
|---|---|---|---|
| 1 | 1-2h | 0 | heute abend fertig |
| 2 | 5 min (start) | 2-5h | morgen frueh fertig |
| 3 | 30 min | 0 | 26.05 vormittag |
| 4 | 2-3h | 30 min Re-Embedding | 26.05 abend fertig |
| 5 | 30 min | 4-6h ueber Nacht | 27.05 morgen fertig |
| Puffer | 28.05 ganzer Tag | -- | falls etwas haengt |

**Worst Case:** Phase 1 zieht sich auf 4h (Schema unklar, Status-quo-Test bricht). Dann V2 streichen, V1 minimal bis 28.05 abend, 29.05 Talk-Vorbereitung normal.

**Best Case:** Phase 1 abend fertig, Phase 2 ueber Nacht durch, Phase 3+4 26.05 fertig, Phase 5 ueber Nacht. Ab 27.05 frueh ist alles abgeschlossen und du hast 2 Tage Puffer fuer Talk und Polish.

---

End of Execution Plan.
