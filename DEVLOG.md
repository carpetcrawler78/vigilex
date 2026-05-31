

## 2026-05-26 -- Phase 4+5: mpnet Migration + LLT Expansion Eval

- Migration: PubMedBERT -> all-mpnet-base-v2 (embed_meddra_terms_v2.py, 27361 rows, 50 min)
- Schema: processed.meddra_terms.embedding_mpnet vector(768) + idx_meddra_mpnet_ivfflat
- hybrid_search.py: EmbeddingModel -> SentenceTransformer, Query Fusion (c1+c2, first_sentence+full_text)
- Baseline (PubMedBERT vector-only): R@100=0.0 -- vector arm was contributing nothing
- Post-migration full pipeline: recall_at_5=0.333, soft_recall_at_5=0.500
- LLT-expansion eval (llt_expanded_top20): no improvement, cat_A 11->12, reverted
- IVFFlat index idx_meddra_llt_expanded_mpnet built (available for future opt-in experiments)
- DECISION.md written: data/eval/DECISION.md


## 2026-05-27 -- Qwen2.5:7b Eval + Test Suite gruen

### Qwen2.5:7b vs. llama3.2:3b (Stage 3 Model Comparison)
- Run: qwen25_7b | MLflow Experiment 1 | Run ID: 6549c88cdd57455caa7325b4dc303ee5
- eval_golden_set.py Bug fix: ollama_base_url -> ollama_url (Signatur-Mismatch)
- recall_at_5=0.333 | soft_recall_at_5=0.500 | soft_recall_at_10=0.833 | mrr=0.191
- p_at_1_llm=0.292 vs. p_at_1_reranker=0.083 -- Qwen waehlt aus Top-5 3.5x haeufiger korrekt
- Recall-Metriken identisch zur llama3.2:3b Baseline (Stage 1+2 Bottleneck unveraendert)
- cat_A=11, cat_B=5, cat_hit=8 -- identisch zum Baseline-Run
- 2 LLM-Timeouts (Cases 1+19, qwen7b auf belastetem CX33) -> CE Fallback, kein Metrik-Effekt
- Elapsed: 1477s (qwen7b) vs ~468s (llama3b) -- 3.15x langsamer
- Fazit: kein Recall-Gewinn, aber P@1-Verbesserung relevant fuer Production-Top1-Quality

### Tests -- 31/31 gruen
- test_api.py (12 Tests): FastAPI TestClient, DB gemockt, httpx installiert
- test_reranker.py (8 Tests): CrossEncoder gemockt, alle Assertions gruen
- test_prr_ror.py (8 Tests) + test_smoke.py (2+1 Tests): unveraendert gruen


## 2026-05-27 -- Groq llama-3.1-8b-instant Referenz-Run

- Run: groq_lm31_8b_groq_ref | MLflow Experiment 1 | Run ID: 77731f77ce0346119044a6bee05d3582
- eval_golden_set.py: --groq-reference Flag implementiert (Stage 3 via Groq external API)
- Regulatorische Tags in MLflow: backend=groq_reference, production_eligible=false, exclusion_reason=GDPR_Art44_Art9_external_api
- recall_at_5=0.333 | soft_recall_at_5=0.500 | soft_recall_at_10=0.833 | mrr=0.191
- p_at_1_llm=0.333 (bestes P@1 der drei Modelle) | elapsed=485s (near-instant Groq-Calls)
- Stage-1-Bottleneck unveraendert: cat_A=11, cat_B=5, cat_hit=8

3-Modell-Vergleich (abgeschlossen):
  llama3.2:3b (Baseline, on-prem): p_at_1_llm=n/a, elapsed=468s
  qwen2.5:7b  (Vergleich, on-prem): p_at_1_llm=0.292, elapsed=1477s
  Groq llama-3.1-8b (Referenz, extern): p_at_1_llm=0.333, elapsed=485s

Fazit: Groq zeigt Upper-Bound der P@1-Qualitaet bei minimalem Throughput-Overhead.
Production-excluded by design (GDPR Art.44+Art.9), nicht wegen Capability.

## 2026-05-31 -- Grafana Histogram Fix (Panel id=3)
- DONE: Panel "Ranking Index Distribution" von histogram auf barchart umgestellt; Bucketing nach SQL vorgezogen (width_bucket 10 Buckets 0.0-0.9); Dashboard via Grafana API zurueckgeschrieben (version 9, status success)
- MISTAKE: histogram-Typ + PostgreSQL table-format + raw numeric column = client-seitiges Bucketing schlaegt fehl (Grafana-bekannte Inkompatibilitaet); erst nach Diagnose-Queries klar
- RULE: Fuer SQL-Datasources immer barchart + SQL-Aggregation statt histogram-Panel-Typ

## 2026-05-31 09:22 -- Grafana Dashboard sentinelai-coding-v1 Fix
- DONE: Panel 3 (Ranking Index Distribution) auf barchart umgestellt + SQL width_bucket()-Aggregation; filter coding_path='hybrid_ce_llm_success' (Pre-Fix-Records raus)
- DONE: Panel 2 (Top 15 MedDRA PT Codes) geloescht (leer nach Post-Fix-Filter)
- DONE: Panel 1 (Coding Throughput) + Panel 99 (Full Table) gefiltert auf coding_path IS NOT NULL
- DONE: Panel 4 (System Summary) -- neues Stat-Tile "Pre-Fix Records" + gefilterte "Unique PTs"
- DONE: Workflow-Fix -- Provisioning-JSON auf Hetzner aktualisiert + reload getriggert (verhindert Grafana-Revert)
- OFFEN: Eval-Panels 5+6 stehen doppelt zu Panel 4 (Soft Recall@5/@10) -- Entscheidung pending
- OFFEN: Luecke oben rechts (ex Panel 2) -- neues Panel oder leer lassen -- Entscheidung pending

## 2026-05-31 17:42 -- Datum-Fix + Slide-Rename + Pipeline-Refactor
- DONE: Datums-Footer auf 03 Jun 2026 korrigiert (6 Dateien: slide2, slide_eng_tools, slide_gold, slide_metrics, slide_pipeline_a, GrafanaSlide.tsx), routes/index.tsx Titel auf "Final Presentation", Pipeline.tsx als wiederverwendbares Component extrahiert (Slot 10, reveal-Prop "abstract"/"models"), alle 14 aktiven Slide-Dateien auf Slide0N_Name-Schema umbenannt, Deck.tsx-Imports aktualisiert, Build gruen (221 modules, 2.2s).
- MISTAKE: Kein -- tote Imports in Deck.tsx waren nicht vorhanden (slide_final existiert in data/ aber war nie importiert).

## 2026-05-31 18:00 -- Housekeeping: tote Slides nach alt/
- DONE: SlideB.tsx (identische Kopie in alt/ bestaetigt per SHA256), slide5.html, slide_final.html nach alt/ verschoben; Build gruen (221/279 modules, 1.93s).
