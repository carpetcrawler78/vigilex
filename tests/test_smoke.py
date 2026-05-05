# tests/test_smoke.py
#
# Allererster pytest-Test fuer vigilex.
# Zweck: pruefen, dass pytest ueberhaupt laeuft und Tests gefunden werden.
# Sobald dieser Test gruen wird, ist die Test-Infrastruktur einsatzbereit.
# Ab dann koennen echte Tests (Confidence-Formel, RRF-Fusion, PRR/ROR, ...) folgen.


def test_python_works():
    """Sanity-Check: 1 + 1 ist immer noch 2."""
    assert 1 + 1 == 2


def test_vigilex_package_importable():
    """Stellt sicher, dass das Package vigilex auffindbar ist.

    Faellt dieser Test, dann ist der Python-Pfad nicht korrekt und
    spaetere Tests koennen die echten Module nicht importieren.
    """
    import vigilex  # noqa: F401
