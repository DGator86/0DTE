"""
Per-feature-and-timeframe scale keying (prediction/scalers.py + mtf_matrix,
PR 2 of Prediction Engine V2): a 1-minute feature update must never alter the
1-day feature's scale state, and snapshot features update once per tick.
"""
import pytest

from mtf_matrix import (MTFInput, SNAPSHOT, TIMEFRAMES, VARS, build_matrix)
from prediction.scalers import RobustScaleBook, scale_key
from regime_classifier import ScaleBook


def test_scale_key_construction():
    assert scale_key("ema_slope", "1m") == "ema_slope:1m"
    assert scale_key("net_gex") == "net_gex"


def test_one_minute_updates_never_touch_daily_state():
    book = RobustScaleBook()
    for x in (0.01, -0.02, 0.03, -0.01, 0.02):
        book.update(scale_key("ema_slope", "1m"), x)
    stats = book.to_dict()["stats"]
    assert "ema_slope:1m" in stats
    assert "ema_slope:1d" not in stats
    assert "ema_slope" not in stats
    # the daily key is still stone cold: std falls back to the prior
    assert book.std("ema_slope:1d", 0.05) == 0.05
    assert book.reliability("ema_slope:1d") == pytest.approx(0.4)


def test_different_timeframes_learn_different_scales():
    import random
    rng = random.Random(3)
    book = RobustScaleBook(n_min=10)
    for _ in range(400):
        book.update("ema_slope:1m", rng.gauss(0.0, 0.05))
        book.update("ema_slope:1d", rng.gauss(0.0, 1.5))
    s1m = book.std("ema_slope:1m", 999.0)
    s1d = book.std("ema_slope:1d", 999.0)
    assert s1m == pytest.approx(0.05, rel=0.35)
    assert s1d == pytest.approx(1.5, rel=0.35)
    # a pooled (legacy) book would have blended these into one wrong scale
    assert s1d / s1m > 10


def test_build_matrix_keys_native_by_timeframe():
    book = RobustScaleBook()
    inp = MTFInput(
        native={"ema_slope": {tf: 0.01 * (i + 1)
                              for i, tf in enumerate(TIMEFRAMES)}},
        snapshot={"gamma_sign": 2.5e9},
    )
    build_matrix(inp, book)
    stats = book.to_dict()["stats"]
    for tf in TIMEFRAMES:
        assert f"ema_slope:{tf}" in stats
        assert stats[f"ema_slope:{tf}"][0] == 1
    assert "ema_slope" not in stats


def test_build_matrix_snapshot_updates_once_per_tick():
    book = RobustScaleBook()
    inp = MTFInput(native={}, snapshot={"gamma_sign": 2.5e9})
    build_matrix(inp, book)
    stats = book.to_dict()["stats"]
    # broadcast across 7 columns but only ONE sample entered the book
    assert stats["gamma_sign"][0] == 1


def test_legacy_book_keeps_legacy_keying():
    book = ScaleBook()
    inp = MTFInput(
        native={"ema_slope": {tf: 0.01 for tf in TIMEFRAMES}},
        snapshot={"gamma_sign": 2.5e9},
    )
    build_matrix(inp, book)
    stats = book.to_dict()
    assert "ema_slope" in stats            # name-only pooling, unchanged
    assert all(":" not in k for k in stats)
    assert stats["ema_slope"][0] == len(TIMEFRAMES)


def test_snapshot_vars_use_bare_name_keys():
    book = RobustScaleBook()
    snap_names = {v.name for v in VARS if v.kind == SNAPSHOT and v.adapt_fn}
    inp = MTFInput(native={}, snapshot={n: 0.001 for n in snap_names})
    build_matrix(inp, book)
    stats = book.to_dict()["stats"]
    assert set(stats) <= snap_names
    assert all(":" not in k for k in stats)
