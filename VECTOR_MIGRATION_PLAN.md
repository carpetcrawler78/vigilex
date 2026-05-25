# Vigilex Vector Migration Plan -- Bench V1

**Datum:** 2026-05-25
**Status:** Approved Plan, Implementation pending
**Bezug:** BUG_VECTOR_SEARCH.md, EVAL_PLAN.md, CLAUDE.md
**Owner:** Thomas

---

## 1. Kontext und Anlass

- Status quo: PubMedBERT-base als Embedding-Modell, geladen via raw `transformers.AutoModel` mit eigenem Pooling und unterschiedlichen `max_length`-Settings (64 indexiert vs. 128 Query).
- Diagnose: Vector-Arm liefert fuer Narrative-Queries Muell (siehe BUG_VECTOR_SEARCH.md). Root Cause: Anisotropie in PubMedBERT-base + asymmetrische Encoding-Pipeline.
- Folge: 11.161 bereits codierte Records sind mit derselben kaputten Pipeline entstanden -- Qualitaet teilweise ungewiss. BM25 und CrossEncoder konnten in einigen Cases retten, in vielen nicht.
- Eval-Befund: Recall@5 = 0 fuer alle 24 Golden-Set-Cases bei Default-Parametern. Mit Workaround (`top_k_stage1=100`, `candidate-pool=200`) rettet sich nur ein Teil.
- Final Presentation: 2026-06-03. Zeitfenster fuer empirisch begruendete Entscheidung knapp, aber vorhanden.

---

## 2. Strategische Entscheidung

- Plan B bleibt aktiv (kein Streamlit, kein voller Import, Eval-First Reihenfolge).
- **Neu:** Vector-Modell-Migration nicht improvisieren, sondern empirisch via Bench V1 entscheiden.
- **Constraint:** Alle Modelle laufen lokal. Keine Cloud-Embedding-API in Produktion. Begruendung: DSGVO + EU AI Act + Schrems II (siehe Abschnitt 6).
- **Groq bleibt ausgeschlossen** fuer Production. Nur Benchmarking, wenn ueberhaupt.
- **Status quo (PubMedBERT-base)** wird NICHT als SentenceTransformer-Bench-Teilnehmer gemessen, sondern ueber den existierenden Produktionscode separat.

---

## 3. Bench V1 Specification (final)

### 3.1 Kandidaten-Modelle

| Modell | Params | Dim | max_seq_length | Default | Quelle |
|---|---|---|---|---|---|
| sentence-transformers/all-MiniLM-L6-v2 | 22M | 384 | 256 | generic | HF Hub |
| BAAI/bge-small-en-v1.5 | 33M | 384 | 512 | generic, MTEB-stark | HF Hub |
| sentence-transformers/all-mpnet-base-v2 | 110M | 768 | 384 | generic-heavy | HF Hub |

Alle drei laufen lokal nach einmaligem Download (HuggingFace Hub liefert die Gewichte, Inferenz lokal auf CPU).

Optional V2-Erweiterung nach V1: `pritamdeka/S-PubMedBert-MS-MARCO` (bio-domain, Sentence-Transformer-finetuned).

### 3.2 Query-Varianten

| Name | Definition |
|---|---|
| `first_sentence` | `text.split(".")[0].strip()[:1000]` (entspricht aktueller Produktion) |
| `full_text_truncated` | `text.strip()[:2048]` (Tokenizer-Truncation uebernimmt das Modell) |

Kein `clinical_window` und kein `meddra_lexicon_keywords` in V1 -- bewusst weggelassen, V2-Material.

### 3.3 Pool-Typen

| Name | Document-Text |
|---|---|
| `pt_only` | `pt_name` (Status quo der bestehenden Pipeline) |
| `pt_limited_llt` | `pt_name + " | " + bis zu 10 LLT-Synonyme` (LLT != pt_name) |

Begruendung pt_limited_llt: kompensiert asymmetrisches Encoding zwischen kurzen PT-Namen und langen Narratives. Sortierung der LLTs: `ORDER BY LENGTH(llt_name), llt_name` (deterministisch, kurze zuerst).

### 3.4 Metriken

**Summary-CSV (eine Zeile pro Konfiguration):**

- `model`
- `pool_type`
- `query_field`
- `n` (Anzahl Cases)
- `max_seq_length` (vom Modell uebernommen)
- `exact_recall_at_1`
- `exact_recall_at_5`
- `exact_recall_at_20`
- `exact_recall_at_50`
- `exact_recall_at_100`
- `median_rank_found`
- `mean_rank_found_only`
- `not_found_count`

**Detail-CSV (eine Zeile pro Case + Konfiguration):**

- `model`, `pool_type`, `query_field`
- `case_id`, `mdr_report_key` (falls verfuegbar)
- `expected_pt_code`, `expected_pt_name`
- `primary_rank`
- `top1_code`, `top1_name`
- `top5_names` (Pipe-separiert)
- `top20_names` (Pipe-separiert)
- `query_text_used` (effektiv verwendeter Query-String, fuer Debugging)
- `query_length_chars`

### 3.5 Caching

Doc-Embeddings sind teuer (27k Texte), Query-Embeddings billig (24 Texte). Caching nur fuer Doc-Embeddings.

- Pfad: `data/eval/cache/{model_slug}_dim{dim}_{pool_type}_doc_embeddings.npy`
- Begleitdatei: `data/eval/cache/{model_slug}_dim{dim}_{pool_type}_meta.csv` (pt_code, pt_name, search_text)
- Begleitdatei: `data/eval/cache/{model_slug}_dim{dim}_{pool_type}_pt_codes.npy`
- Slug-Funktion: `model_name.replace("/", "__").replace("-", "_").replace(".", "_")`

Mit Caching: 6 Doc-Encodings (3 Modelle x 2 Pool-Typen), 6 Query-Encodings (3 Modelle x 2 Query-Varianten), 12 Cosine-Matrizen (3 x 2 x 2). Geschaetzte Gesamtlaufzeit auf CPU: 1-4h einmalig.

---

## 4. Schritt-fuer-Schritt-Anleitung

### Schritt 0 -- Vorbereitung (15 min)

1. Repo-Schema verifizieren: `processed.meddra_terms` und `processed.meddra_llt` Spaltennamen pruefen. Tatsaechliche Tabellen-/Spaltennamen in `vigilex/CLAUDE.md` oder im Schema-Dump nachsehen.
2. Existierende `EmbeddingModel`-Klasse oder vergleichbare Schnittstelle in `hybrid_search.py` lokalisieren (fuer Schritt 1).
3. Disk-Check: ca. 3-5 GB freier Platz fuer Modell-Caches + Embedding-Files.
4. Optional vorab `huggingface-cli download <model_name>` fuer die drei Modelle (vermeidet langes erstes Skript-Run-Up).

### Schritt 1 -- Status-quo-Baseline messen (30 min)

Variante B (pragmatisch): isolierter Vector-only-Run mit dem bestehenden Produktionscode auf demselben Golden Set. Ziel: eine Zahl fuer `recall_at_100` mit dem aktuellen PubMedBERT-base-Setup.

- Vorhandenes Skript `eval_golden_set.py` mit Flag oder Codepatch so anpassen, dass nur der Vector-Arm gemessen wird (BM25 und CrossEncoder muten / skip).
- Output dokumentieren in `data/eval/status_quo_baseline.json`:
  ```json
  {
    "model": "PubMedBERT-base (Produktion)",
    "recall_at_100": 0.XX,
    "median_rank_found": ...,
    "not_found_count": ...,
    "measured_at": "2026-05-25",
    "method": "vector-only, eval_golden_set.py modifiziert"
  }
  ```
- Dieser Wert ist die Anker-Baseline. Alle Bench-Resultate werden gegen ihn verglichen.

### Schritt 2 -- Pool-Loading isoliert testen (30 min)

SQL fuer beide Pool-Typen formulieren und Output pruefen.

**pt_only:**
```sql
SELECT pt_code, pt_name, pt_name AS search_text
FROM processed.meddra_terms
ORDER BY pt_code;
```

**pt_limited_llt:**
```sql
WITH llt_limited AS (
    SELECT
        pt_code,
        llt_name,
        ROW_NUMBER() OVER (
            PARTITION BY pt_code
            ORDER BY LENGTH(llt_name), llt_name
        ) AS rn
    FROM processed.meddra_llt
    WHERE llt_name IS NOT NULL
)
SELECT
    t.pt_code,
    t.pt_name,
    t.pt_name || ' | ' || COALESCE(string_agg(l.llt_name, ' | '), '') AS search_text
FROM processed.meddra_terms t
LEFT JOIN llt_limited l
    ON t.pt_code = l.pt_code
    AND l.rn <= 10
    AND l.llt_name <> t.pt_name
GROUP BY t.pt_code, t.pt_name
ORDER BY t.pt_code;
```

Sanity-Check VOR dem Modell-Run:
```python
expected_codes = {int(c["expected_pt_code"]) for c in golden}
pool_codes = set(map(int, pt_pool["pt_code"]))
missing = expected_codes - pool_codes
assert not missing, f"Expected PT codes missing from pool: {missing}"
```

### Schritt 3 -- Bench-Skelett mit einem Modell (1h)

End-to-end mit `MiniLM-L6-v2` + `pt_only` + `first_sentence`:

1. Golden Set laden (`data/eval/golden_set_v1.jsonl`)
2. pt_only-Pool laden
3. SentenceTransformer instanziieren, max_seq_length nicht setzen (Default lassen)
4. Doc-Embedding berechnen mit `normalize_embeddings=True`
5. Query-Embedding (first_sentence) mit `normalize_embeddings=True`
6. Cosine via Dot-Product (da normalized)
7. Top-100 indices via argsort
8. Recall@K + median_rank berechnen
9. Detail-CSV mit `query_text_used` schreiben
10. Summary-CSV-Zeile schreiben

Stop-Condition: Skript laeuft sauber durch und produziert plausible Zahlen. Wenn nicht, hier fixen, bevor Caching/Loop dazu kommt.

### Schritt 4 -- Caching einbauen (30 min)

- Vor jedem Doc-Encoding: pruefen ob `cache_path.exists()`. Wenn ja, `np.load`, sonst encode + `np.save`.
- meta.csv ebenfalls cachen.
- Re-Run mit denselben Settings darf nichts mehr encoden, nur laden.

### Schritt 5 -- Loop ueber alle Konfigurationen (3-5h Laufzeit)

```
for model in MODELS:           # 3
    for pool_type in POOL_TYPES:    # 2
        doc_emb = encode_or_load_cache(model, pool_type)
        for query_field in QUERY_FIELDS:    # 2
            q_emb = encode_queries(model, query_field)
            metrics, details = evaluate(doc_emb, q_emb, golden)
            append_to_summary(metrics)
            append_to_detail(details)
```

12 Konfigurationen insgesamt. Bei MiniLM/bge etwa 5 min pro Konfig, bei mpnet/PubMedBERT-Varianten 15-30 min.

### Schritt 6 -- Auswertung mit Pivots (15 min)

**Pool-Effekt:**
```python
pool_pivot = df.pivot_table(
    index="model", columns="pool_type",
    values="exact_recall_at_100", aggfunc="max"
)
print("Delta PT+LLT vs PT-only:")
print((pool_pivot["pt_limited_llt"] - pool_pivot["pt_only"]).sort_values(ascending=False))
```

**Query-Effekt:**
```python
query_pivot = df.pivot_table(
    index="model", columns="query_field",
    values="exact_recall_at_100", aggfunc="max"
)
print("Delta full_text_truncated vs first_sentence:")
print((query_pivot["full_text_truncated"] - query_pivot["first_sentence"]).sort_values(ascending=False))
```

**Best-Config:**
```python
best = df.sort_values("exact_recall_at_100", ascending=False).head(5)
print(best[["model", "pool_type", "query_field", "exact_recall_at_100", "exact_recall_at_5"]])
```

### Schritt 7 -- Entscheidung gegen Entscheidungstabelle (siehe Abschnitt 5)

### Schritt 8 -- Migration (falls Gewinner identifiziert) (1 Tag)

1. Falls neue Dim (384 statt 768): `processed.meddra_terms` Schema-Migration. Neue Spalte `embedding_v2 vector(384)`.
2. Re-Embedding aller 27k PTs mit Gewinner-Modell + gewinnender Pool-Repraesentation. Skript: Variante von `embed_meddra_terms.py` mit SentenceTransformer-Library.
3. pgvector IVFFlat-Index auf neuer Spalte neu bauen: `CREATE INDEX ... USING ivfflat (embedding_v2 vector_cosine_ops) WITH (lists = 100);`
4. `hybrid_search.py` umstellen: neue Embedding-Klasse via SentenceTransformer, identische Settings auf Index- und Query-Seite.
5. Erneuter `eval_golden_set.py`-Run zur Bestaetigung.
6. Optional: Sample-Re-Coding von ~500 alten Records mit neuer Pipeline, Vergleich der Code-Entscheidungen alt vs. neu. Wenn Uebereinstimmung hoch: 11k Records koennen mit Caveat weiterverwendet werden. Wenn niedrig: Bulk-Re-Coding oder klare Kennzeichnung als "alte Pipeline".

---

## 5. Entscheidungsregel (nach Bench)

| Befund im Bench | Konsequenz |
|---|---|
| PT+LLT verbessert alle Modelle stark | Hauptproblem war Target-Repraesentation. Pool-Rebuild reicht ggf. ohne Modellwechsel. |
| full_text_truncated verbessert stark | first_sentence schneidet klinische Info ab. Query-Strategie aendern. |
| Ein Modell verbessert PT-only UND PT+LLT stark | Modellwechsel lohnt sich (mit Re-Embedding). |
| Alle Kandidaten bleiben schwach | Vector-Arm im RRF auf w=0.2 runtergewichten oder deaktivieren. BM25+CE+LLM tragen Pipeline. |
| Recall@100 gut, Recall@5 schwach | CrossEncoder ist gerechtfertigt, top_k_stage1 erhoehen. |
| Status quo besser als alle Kandidaten | Keine Migration. BUG_VECTOR_SEARCH.md als Edge-Case-Befund framen. |

Mindest-Lift fuer Migration: `+10pp Recall@100` ueber Status-quo-Baseline. Darunter ist Migrations-Aufwand nicht gerechtfertigt.

---

## 6. Regulatorisches Framing

### 6.1 Drei Regulationen, die hier greifen

**DSGVO (seit 2018) + Schrems II (2020):**
Personenbezogene Gesundheitsdaten (Art. 9 "besondere Kategorien") brauchen explizite Rechtsgrundlage. Datentransfer in Drittlaender (insb. USA) seit Schrems II ohne adequaten Schutz problematisch. Blockt Cloud-LLMs (OpenAI, Anthropic, Groq) fuer echte Patientendaten faktisch aus.

**EU AI Act (2024 verabschiedet, gestaffelt in Kraft):**
AI fuer Medizinprodukte = Annex III High-Risk. Pflichten ab August 2026:
- Art. 9 Risk Management
- Art. 10 Data and Data Governance
- Art. 12 Automatic Logging
- Art. 13 Transparency
- Art. 14 Human Oversight
- Art. 15 Accuracy, Robustness, Cybersecurity

**MDR (2017/745) + GVP Module VI:**
Post-Market Surveillance und Adverse Event Coding nach MedDRA (ICH-Standard, auch EMA-genutzt) sind Pflicht.

### 6.2 SentinelAI-Architektur ↔ Regulation Mapping

| Requirement | SentinelAI-Implementierung |
|---|---|
| DSGVO Datenresidenz | PostgreSQL + Embeddings auf Hetzner DE |
| Schrems II (kein US-Transfer) | Ollama lokal, kein Cloud-LLM in Produktion |
| EU AI Act Art. 12 Logging | processed.coding_results mit coded_at, audit-trail |
| EU AI Act Art. 14 Human Oversight | Reviewer Interface |
| EU AI Act Art. 10 Data Governance | Golden Set + MLflow-Tracking |
| EU AI Act Art. 13 Transparency | "ranking index" statt "confidence" |
| EU AI Act Art. 15 Accuracy | gemessene Recall@K auf Eval-Dataset |
| MDR Post-Market Surveillance | MAUDE-aehnlicher Use Case |
| GVP Module VI MedDRA-Coding | Kernfunktion |

### 6.3 Bench-Resultat als Beleg

Egal welcher der drei moeglichen Bench-Befunde eintritt -- die regulatorische Story funktioniert:

- **Kleines lokales Modell reicht** -> "kein technischer Zwang zu Cloud-Inferenz, Compliance ist kein Tradeoff"
- **PT+LLT-Repraesentation ist Hauptgewinn** -> "Knowledge-Base-Engineering schlaegt Modellgroesse, Architektur-Argument fuer Auditoren"
- **LLM-Stage traegt Pipeline** -> "Hybrid-Architektur empirisch gerechtfertigt, LLM lokal via Ollama"

### 6.4 Caveats -- nicht verschweigen

1. MAUDE = US-Daten, keine EU-Patientendaten. Argument ist Future-Use-Case-Konformitaet, nicht aktuelle DSGVO-Anwendung.
2. "DSGVO-konform" ist kein Selbst-Stempel. Verwenden: "DSGVO-by-design" oder "Architektur unterstuetzt DSGVO-konformen Betrieb".
3. EU AI Act High-Risk-Pflichten greifen erst ab August 2026. Sprache: "prepared for", nicht "compliant with".
4. Hetzner: spezifisch deutsches Rechenzentrum nennen (Hetzner hat auch nicht-EU-RZ in Singapore).
5. Qwen ist chinesisches Modell -- laeuft lokal (keine Datenexfiltration), aber Provenienz-Audit-Fragen moeglich.
6. MedDRA ist lizenzpflichtig (MSSO-Subskription). Im Capstone-Kontext irrelevant, bei "Marktreife"-Behauptungen wichtig.

---

## 7. Memory- und Konvention-Regeln (nicht brechen)

- pgvector IVFFlat -- KEIN FAISS (Capstone I, Memory-Drift korrigiert 2026-05-14)
- "ranking index" bei Stakeholdern, NIE "confidence score"
- SentinelAI = MedDRA Coding Assistance Tool; PRR/ROR ist Downstream-Demo
- `mdr_text` (NICHT event_description), `coded_at` (NICHT created_at)
- Single-Instance-Policy: NIE Host-Worker + Docker-Worker gleichzeitig
- Groq: nur Benchmarking, NIE Production
- Schema/Code nie raten -- erst nachschlagen oder fragen
- ASCII only in Code-Files

---

## 8. Risiken (priorisiert)

1. **Schema-Migration (Dim 768 -> 384)**: pgvector-Spalte neu, Index neu. Mittlerer Aufwand, gut handhabbar.
2. **Pipeline-Konsistenz**: Index- und Query-Seite muessen identische Encoding-Settings haben. Stolperstein: derselbe Bug in anderer Form, wenn Settings divergieren.
3. **11k-Records-Qualitaet**: bleiben mit alter Pipeline kodiert. Sample-Re-Coding zeigt, wie stark sich neue Pipeline unterscheidet. Wenn stark: ehrliche Kommunikation noetig.
4. **Zeitbudget**: Bench V1 + Migration realistisch 1-2 Tage. Bei Verzoegerung: V1 als reine Diagnostik, Migration in V2 verschoben, Caveat im Eval-Report.

---

## 9. Anhang

### 9.1 Python-Helfer (Skizzen)

```python
def model_slug(model_name: str) -> str:
    return (
        model_name.replace("/", "__")
                  .replace("-", "_")
                  .replace(".", "_")
    )

def cache_path(model_name: str, dim: int, pool_type: str, kind: str) -> Path:
    slug = model_slug(model_name)
    return Path(f"data/eval/cache/{slug}_dim{dim}_{pool_type}_{kind}")

# kind in {"doc_embeddings.npy", "pt_codes.npy", "meta.csv"}
```

### 9.2 Query-Funktionen

```python
def make_first_sentence(text: str) -> str:
    return text.split(".")[0].strip()[:1000]

def make_full_text_truncated(text: str) -> str:
    return text.strip()[:2048]
```

### 9.3 Sanity-Check (im Skript pflichtig)

```python
expected_codes = {int(c["expected_pt_code"]) for c in golden}
pool_codes = set(map(int, pt_pool["pt_code"]))
missing = expected_codes - pool_codes
assert not missing, f"Expected PT codes missing from pool: {missing}"
```

### 9.4 Auswertungs-Snippet

```python
import pandas as pd

df = pd.read_csv("data/eval/bench_results_summary.csv")

pool_pivot = df.pivot_table(
    index="model", columns="pool_type",
    values="exact_recall_at_100", aggfunc="max"
)
print("=== Pool-Effekt (Recall@100) ===")
print(pool_pivot)

query_pivot = df.pivot_table(
    index="model", columns="query_field",
    values="exact_recall_at_100", aggfunc="max"
)
print("=== Query-Effekt (Recall@100) ===")
print(query_pivot)

best = df.sort_values("exact_recall_at_100", ascending=False).head(5)
print("=== Top-5 Konfigurationen ===")
print(best[["model","pool_type","query_field","exact_recall_at_100","exact_recall_at_5"]])
```

---

## 10. Folge-Schritte nach Final Presentation (V2-Material)

- `acceptable_pt_codes` im Golden Set annotieren -> acceptable_recall@K
- `clinical_window` Query-Variante implementieren
- 4. und 5. Modell ergaenzen (S-PubMedBert-MS-MARCO, NeuML/pubmedbert-base-embeddings)
- LLT pro Document als eigenes Dokument indizieren statt Konkatenation (vermeidet Token-Truncation)
- MLflow-Integration des Benchs

---

End of Plan.
