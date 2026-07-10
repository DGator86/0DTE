"""
Score-before-update (PR 2 of Prediction Engine V2): the current observation
must be standardized against the scale state learned from HISTORY only, and
enter that state only after it has been scored — in the RobustScaleBook, in
mtf_matrix.build_matrix, and in RegimeClassifier.classify.
"""
import pytest

from mtf_matrix import MTFInput, TIMEFRAMES, build_matrix
from prediction.scalers import RobustScaleBook


def test_std_is_read_only():
    book = RobustScaleBook()
    for x in (1.0, 2.0, 3.0):
        book.update("k:1m", x)
    before = [list(v) for v in book.to_dict()["stats"].values()]
    book.std("k:1m", 0.5)
    book.reliability("k:1m")
    after = [list(v) for v in book.to_dict()["stats"].values()]
    assert before == after


def test_observation_does_not_influence_its_own_score():
    """An extreme new print must be judged against the OLD scale. Under the
    legacy update-then-score order, the outlier first widens the scale and
    then gets scored against the widened version — self-normalizing away
    exactly the anomaly the matrix exists to flag."""
    hist = [0.01, -0.012, 0.008, 0.011, -0.009, 0.0095, -0.0105] * 6

    lagged_book = RobustScaleBook(n_min=10)
    for x in hist:
        lagged_book.update("v:1m", x)
    scale_before = lagged_book.std("v:1m", 999.0)

    outlier = 0.5
    # correct order: score first ...
    scale_used = lagged_book.std("v:1m", 999.0)
    assert scale_used == pytest.approx(scale_before)
    # ... then update
    lagged_book.update("v:1m", outlier)
    assert lagged_book.std("v:1m", 999.0) > scale_used  # outlier entered after


def test_build_matrix_scores_against_pre_update_state():
    """First-ever tick: the book has no history, so the score must come from
    the fixed prior even though the observation itself enters the book during
    the same call (legacy order produced the same n=1 state but conceptually
    scored 'after update'; with n=1 std() falls back to prior either way, so
    we assert on the SECOND tick where the orders genuinely diverge)."""
    def one_tick(book, x):
        inp = MTFInput(native={"ema_slope": {"1m": x}}, snapshot={})
        rows = build_matrix(inp, book)
        row = next(r for r in rows if r.variable == "ema_slope")
        return row.scores["1m"]

    # V2 lagged book: tick 2 is scored with only tick 1 in the state (n=1 →
    # prior scale), so the score equals the fixed-prior transform exactly.
    book = RobustScaleBook()
    one_tick(book, 0.02)
    prior_score = one_tick(RobustScaleBook(), 0.04)   # fresh book → prior
    lagged_score = one_tick(book, 0.04)
    assert lagged_score == prior_score

    # After the call, tick 2 HAS entered the state (n=2)
    assert book.to_dict()["stats"]["ema_slope:1m"][0] == 2


def test_classifier_standardizes_before_updating_scales():
    from regime_classifier import RegimeClassifier

    class SpyScales:
        """Records the order of read (std) vs write (update) calls."""
        def __init__(self):
            self.calls = []

        def std(self, name, default):
            self.calls.append(("std", name))
            return default

        def update(self, name, x):
            self.calls.append(("update", name))

        def reliability(self, name):
            return 1.0

        def to_dict(self):
            return {}

        def load_dict(self, d):
            pass

    import datetime as dt
    from zoneinfo import ZoneInfo
    from regime_classifier import ClassifierContext
    from gate_scorer import MarketSnapshot

    spot = 600.0
    market = MarketSnapshot(
        spot=spot, net_gex=4e9, gamma_flip=spot - 7,
        call_wall=spot + 5, put_wall=spot - 5, gex_pct_rank=0.85,
        vix9d=12.0, vix=13.0, vix3m=15.0, vvix=92.0, vvix_baseline=95.0,
        straddle_breakeven=4.0, expected_range=3.2,
        adx=12.0, rsi=51.0, bb_width=1.4, bb_width_baseline=2.0,
        vwap=spot, vwap_reversion_count=5,
        tick_abs_mean=450.0, cvd_slope=0.05,
        now=dt.datetime(2026, 6, 25, 11, 30,
                        tzinfo=ZoneInfo("America/New_York")),
        has_catalyst=False, catalyst_label="",
    )

    clf = RegimeClassifier()
    clf.scales = SpyScales()
    clf.classify(ClassifierContext(market=market))

    reads = [i for i, c in enumerate(clf.scales.calls) if c[0] == "std"]
    writes = [i for i, c in enumerate(clf.scales.calls) if c[0] == "update"]
    assert reads and writes
    # every standardization read happens before the first scale update
    assert max(reads) < min(writes)
