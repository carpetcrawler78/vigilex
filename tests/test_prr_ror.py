from vigilex.signals.prr_ror import _compute_prr, _compute_ror, _is_signal

def test_compute_prr_normal():
    prr, lo, hi = _compute_prr(a=10, b=90, c=5, d=895)
    expected = (10/100) / (5/900)
    assert abs(prr - expected) < 1e-9 
    assert lo < prr < hi

def test_compute_prr_zero_c():
    prr, lo, hi = _compute_prr(a=10, b=90, c=0, d=895)
    assert (prr, lo, hi) == (None, None, None)


def test_compute_prr_a_zero():
    # a=0 -- PRR berechenbar (= 0.0), aber CI nicht (log(0) undefiniert)
    prr, lo, hi = _compute_prr(a=0, b=100, c=5, d=895)
    assert prr == 0.0
    assert lo is None
    assert hi is None


def test_compute_ror_normal():
    ror, lo, hi = _compute_ror(a=10, b=90, c=5, d=895)
    expected = (10 * 895) / (90 * 5)
    assert abs(ror - expected) < 1e-9
    assert lo < ror < hi


def test_compute_ror_zero_b():
    # b=0 -- Division by zero, kein ROR moeglich
    ror, lo, hi = _compute_ror(a=10, b=0, c=5, d=895)
    assert (ror, lo, hi) == (None, None, None)


def test_is_signal_true():
    assert _is_signal(n_focal=5, prr=3.0, thresholds={"min_reports_focal": 3, "prr_min": 2.0}) is True


def test_is_signal_false_low_prr():
    assert _is_signal(n_focal=5, prr=1.5, thresholds={"min_reports_focal": 3, "prr_min": 2.0}) is False


def test_is_signal_false_low_n():
    assert _is_signal(n_focal=2, prr=3.0, thresholds={"min_reports_focal": 3, "prr_min": 2.0}) is False


def test_is_signal_none_prr():
    assert _is_signal(n_focal=5, prr=None, thresholds={"min_reports_focal": 3, "prr_min": 2.0}) is False
