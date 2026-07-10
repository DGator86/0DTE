"""
RobustScaleBook persistence (PR 2 of Prediction Engine V2): restart
reproduces the same next-tick score; incompatible or corrupt state is
rejected (cold re-warm), never silently reinterpreted.
"""
import pytest

from prediction.scalers import RobustScaleBook, STATE_VERSION


def _warmed(n_min=10):
    book = RobustScaleBook(n_min=n_min)
    xs = [0.01, -0.02, 0.015, 0.005, -0.012, 0.018, -0.007, 0.011,
          -0.016, 0.009, 0.013, -0.01]
    for x in xs:
        book.update("ema_slope:1m", x)
        book.update("gamma_sign", x * 1e11)
    return book


def test_roundtrip_reproduces_next_tick_score():
    a = _warmed()
    data = a.to_dict()

    b = RobustScaleBook(n_min=10)
    assert b.load_dict(data) is True

    # identical read state after "restart"
    assert b.std("ema_slope:1m", 999.0) == pytest.approx(
        a.std("ema_slope:1m", 999.0))
    assert b.reliability("ema_slope:1m") == pytest.approx(
        a.reliability("ema_slope:1m"))

    # and the NEXT tick evolves identically on both sides
    a.update("ema_slope:1m", 0.02)
    b.update("ema_slope:1m", 0.02)
    assert b.std("ema_slope:1m", 999.0) == pytest.approx(
        a.std("ema_slope:1m", 999.0))
    assert a.to_dict() == b.to_dict()


def test_incompatible_config_rejected():
    data = _warmed(n_min=10).to_dict()
    other = RobustScaleBook(n_min=50)          # different config hash
    assert other.load_dict(data) is False
    assert other.std("ema_slope:1m", 0.05) == 0.05   # cold: prior fallback
    assert other.to_dict()["stats"] == {}


def test_wrong_state_version_rejected():
    data = _warmed().to_dict()
    data["meta"]["state_version"] = "rsb-0"
    b = RobustScaleBook(n_min=10)
    assert b.load_dict(data) is False
    assert b.to_dict()["stats"] == {}


def test_legacy_shaped_state_rejected():
    # a legacy ScaleBook dump (name -> [n, mean, M2], no meta) must never be
    # reinterpreted as per-timeframe exponentially-decayed state
    legacy = {"ema_slope": [120, 0.01, 0.4]}
    b = RobustScaleBook()
    assert b.load_dict(legacy) is False
    assert b.std("ema_slope", 0.05) == 0.05


def test_corrupt_state_rewarns():
    b = RobustScaleBook()
    assert b.load_dict({"meta": {"state_version": STATE_VERSION,
                                 "config_hash": b.config_hash()},
                        "stats": {"k": "garbage"}}) is False
    assert b.to_dict()["stats"] == {}
    assert b.load_dict("not even a dict") is False


def test_metadata_present():
    d = RobustScaleBook().to_dict()
    assert d["meta"]["state_version"] == STATE_VERSION
    assert len(d["meta"]["config_hash"]) == 16
