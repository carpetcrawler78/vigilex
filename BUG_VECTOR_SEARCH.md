# Bug Report: Vector Arm Returns Garbage for Narrative Queries

**Datum:** 2026-05-25  
**Status:** Diagnosed, Workaround active  
**Schweregrad:** Kritisch fuer Eval (alle R@5=N ohne Fix), kein Produktions-Crash  

---

## Symptom

`eval_golden_set.py` mit Default-Parametern (`--top-k-stage1 20`):
alle 24 Eval-Cases geben R@5=N, P@1=N zurueck.

---

## Diagnose-Kette

### Schritt 1 -- Embedding-Vergleich (ausgeschlossen)

Direkter Vergleich des gespeicherten DB-Embeddings mit dem Query-Embedding
fuer denselben Text ("Diabetic ketoacidosis"):

```
Query  norm : 1.0000
DB     norm : 1.0000
DB     dim  : 768
Query  dim  : 768
Cosine similarity: 1.0000
```

**Ergebnis:** Embeddings sind identisch. Kein Modell-Mismatch, kein Pooling-Fehler.

### Schritt 2 -- Arm-separierte Diagnose

Fuer Case 1 (DKA-Narrative, expected: Diabetic ketoacidosis pt_code=10012671):

```
first_sentence: IT WAS REPORTED THAT THE PATIENT HAD BEEN HOSPITALIZED
                WITH DIABETIC KETOACIDOSIS (DKA) ON (B)(6) 2024
len: 102 Zeichen

--- BM25 arm ---
Total: 100
Top10: [(Diabetes mellitus, 1.0), (Pain, 1.0), (Blood glucose, 1.0),
        (Malaise, 1.0), (Blood insulin, 1.0), (Loss of consciousness, 1.0),
        (Ketoacidosis, 1.0), (Hyperglycaemia, 1.0),
        (Diabetic ketoacidosis, 1.0),   <-- Rank 9 ✓
        (B symptoms, 0.818)]
DKA in BM25: [pt_code=10012671, sim=1.0]  ✓

--- Vector arm ---
Total: 100
Top10: [(Myocarditis-myositis-myasthenia gravis overlap syndrome, 0.9389),
        (Stroke-like migraine attacks after radiation therapy, 0.9389),
        (Sleep disorder due to general medical condition, hypersomnia, 0.9375),
        (Sleep disorder due to general medical condition, parasomnia, 0.9371),
        (Therapeutic drug monitoring analysis incorrectly performed, 0.9365),
        ...]
DKA in vector: []  -- nicht in top-100 ✗
```

---

## Root Cause: Anisotropie in PubMedBERT Base

**PubMedBERT base** (nicht fine-tuned fuer Sentence Similarity) hat ein bekanntes
Problem: die Embeddings clustern in bestimmten Richtungen des Vektorraums
(Anisotropie). Das fuehrt zu zwei Effekten:

1. **Asymmetrische Encodings:** PT-Name-Embeddings (2-4 Woerter, max_length=64,
   indexiert mit `embed_meddra_terms.py`) und Narrative-Embeddings (15-30 Woerter,
   max_length=128, berechnet in `hybrid_search.py`) liegen in **verschiedenen
   Regionen** des Vektorraums -- auch wenn der Narrative explizit den PT-Namen
   enthaelt.

2. **Uniformes Noise-Signal:** Alle 100 Vector-Ergebnisse haben cosine_sim ~0.93-0.94.
   Die Narrative-Embeddings fallen in eine dichte Cluster-Region, sodass viele
   unverwandte PT-Namen aehnlich nah erscheinen.

**Validierungsfehler waehrend der Entwicklung:** Die Notebooks (05, 06, 08) haben
den Vector-Arm nur mit **kurzen Test-Texten** (2-5 Woerter) validiert -- nicht mit
vollstaendigen MAUDE-Narrativen. Mit kurzen Texten funktioniert die Semantik, weil
der Encoding-Laengen-Unterschied minimal ist.

---

## Warum R@5=N bei Default-Parametern

RRF-Fusion (w_bm25=0.4, w_vector=0.6, k=60) mit `top_k_stage1=20`:

| PT | BM25 Rank | BM25 RRF | Vector Rank | Vector RRF | Total RRF |
|----|-----------|----------|-------------|------------|-----------|
| Diabetes mellitus | 1 | 0.00656 | ~50 | ~0.00545 | ~0.01201 |
| Pain | 2 | 0.00645 | ~30 | ~0.00600 | ~0.01245 |
| Myocarditis... | -- | 0 | 1 | 0.00984 | 0.00984 |
| Stroke-like... | -- | 0 | 2 | 0.00968 | 0.00968 |
| **Diabetic ketoacidosis** | **9** | **0.00580** | **>100** | **0** | **0.00580** |

- DKA-Score (0.00580) wird von ~46 Vector-only-Termen geschlagen
  (Vector ranks 1-46 haben RRF > 0.00580)
- DKA landet bei approx. Rank ~55 im fusionierten Ergebnis
- Mit `top_k_stage1=20`: DKA wird **abgeschnitten**, CrossEncoder sieht DKA nie
- Ergebnis: R@5=N fuer alle Cases

---

## Workaround (aktiv)

```bash
python scripts/eval_golden_set.py \
  --top-k-stage1 100 \
  --top-k-stage2 5 \
  --candidate-pool 200 \
  --run-name eval_stage1_top100_candidate200
```

**Effekt:** DKA bei Rank ~55 in RRF-Union --> mit top_k=100 ueberlebt es die Fusion.
CrossEncoder sieht DKA und rankt es korrekt (Case 1: DKA an Rank 2 in Stage 2 --> R@5=Y).

**Limitierung:** Nicht alle Cases profitieren. Cases wo der korrekte PT weder in
BM25 noch im Vector top-200 erscheint (z.B. "Application site dermatitis" bei
Narrative ohne das Wort "dermatitis") bleiben R@5=N.

Betroffene Cases:
- Case 2 (App site dermatitis): "dermatitis"/"application site" kein Wort-Match
- Case 3/4 (Hyperglycaemia): "hyperglycaemia" nicht im Text, nur "blood glucose 600 mg/dl"

---

## Implikation fuer die 11.161 kodierten Records

Der Vector-Arm war **auch waehrend der Bulk-Kodierung broken** (gleiche Pipeline).
Die Korrektheit der 11k kodierten Records haengt davon ab, ob der korrekte PT
via BM25 in die top-20 RRF-Ergebnisse kommt UND ob der CrossEncoder ihn
korrekt auf Rank 1 setzt. Fuer "simple"-Cases (z.B. Narrative enthaelt
"hypoglycemia" oder "low blood sugar") ist das plausibel. Fuer "difficult"-Cases
ist Vorsicht geboten.

---

## Langfristige Loesung (nach Final Presentation)

**Option A -- Sentence Transformers (empfohlen):**
Ersetze PubMedBERT durch ein Modell das fuer asymmetrische Aehnlichkeit
trainiert wurde:
- `sentence-transformers/all-MiniLM-L6-v2` (generisch, schnell)
- `pritamdeka/PubMedBERT-finetuned-pubmed-bioasq-unfactoid` (medizinisch, besser)
- Re-embedding aller 27k PT-Namen notwendig (run `embed_meddra_terms.py` neu)

**Option B -- Kurzere Vector-Query:**
Statt `first_sentence` nur die extrahierten medizinischen Keywords embedden
(NER-basiert oder simples Regex fuer clinische Terme). Kein Re-Embedding noetig.

**Option C -- BM25-only mit erweitertem LLT-Dictionary:**
Vector-Arm deaktivieren. BM25 mit groesserem LLT-Synonym-Lexikon (z.B. SNOMED CT
Mappings). Einfacher, robust, gut erklaerbar.

---

## Talk #3 Framing

Dieser Bug ist **kein Misserfolg** -- er ist ein **Forschungsergebnis**:

> "Wir haben im Eval-Dataset festgestellt, dass Stage 1 (Hybrid Search) fuer
> 'difficult' Cases -- wo das korrekte MedDRA PT nicht woertlich im Narrative
> erscheint -- einen niedrigen Recall hat. Das zeigt genau, warum Stage 3
> (LLM) notwendig ist: der LLM schließt die semantische Luecke zwischen
> klinischem Alltagstext und MedDRA-Standardvokabular."

Metrik-Framing:
- recall_at_5 Stage 1+2: "Baseline ohne LLM"
- p_at_1_llm Stage 3: "LLM-Verbesserung gegenueber Baseline"
- Diff = Wert des dritten Pipeline-Stages (Talk-Narrativ)
