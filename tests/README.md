# vigilex tests

Pytest-basierte Test-Suite fuer vigilex.

## Erstmaliges Setup (nur einmal)

Aus dem vigilex-Wurzelverzeichnis (also wo CLAUDE.md liegt):

```
pip install pytest
```

## Tests ausfuehren

Aus dem vigilex-Wurzelverzeichnis:

```
pytest
```

Erwartete Ausgabe:

```
===== test session starts =====
collected 2 items

tests/test_smoke.py ..                     [100%]

===== 2 passed in 0.05s =====
```

## Was steckt aktuell drin?

- `test_smoke.py` -- Smoke-Test: prueft, dass pytest laeuft und vigilex importierbar ist.

## Was kommt als naechstes?

Schritt fuer Schritt, immer wenn ein neuer Code-Teil entsteht oder ein Bug
auftaucht:

1. `tests/unit/test_confidence.py`  -- Confidence-Formel (0.3*sigmoid(CE) + 0.7*LLM)
2. `tests/unit/test_rrf_fusion.py`  -- Weighted Reciprocal Rank Fusion
3. `tests/unit/test_maude_flatten.py` -- flatten_maude_record()
4. `tests/unit/test_prr_ror.py`     -- PRR/ROR-Math (zusammen mit Modul 3)

## Konvention

- ASCII only (keine Sonderzeichen, keine Umlaute in Code/Docstrings)
- Test-Dateien beginnen mit `test_`
- Test-Funktionen beginnen mit `test_`
- Ein Test prueft genau eine Sache
