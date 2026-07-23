"""
tests/test_spy_der_predict.py
=============================
SPY-DER deterministic "trader" price prediction.
"""
from __future__ import annotations

from types import SimpleNamespace

from spy_der_predict import deterministic_prediction, predict_spy_der_tick


def _mkt(**kw):
    base = dict(spot=602.0, vwap=601.0, call_wall=606.0, put_wall=596.0, gamma_flip=598.0,
                net_gex=3e9, gex_pct_rank=0.8, rsi=58.0, adx=20.0, cvd_slope=0.03,
                expected_range=3.0, straddle_breakeven=4.0)
    base.update(kw)
    return SimpleNamespace(**base)


def test_prediction_has_ordered_cone_and_confidence():
    p = deterministic_prediction(_mkt())
    assert p is not None
    assert p["target_low"] <= p["target"] <= p["target_high"]
    assert 0.0 <= p["confidence"] <= 1.0
    assert p["source"] == "deterministic"
    labels = {lv["label"] for lv in p["key_levels"]}
    assert {"Call wall", "Put wall", "γ-flip", "VWAP"} <= labels


def test_bullish_when_above_vwap_and_flip_with_momentum():
    p = deterministic_prediction(_mkt(spot=603.5, vwap=601.0, gamma_flip=598.0, rsi=64.0, cvd_slope=0.05))
    assert p["bias"] == "bullish"
    assert p["target"] >= p["spot_at_pred"]


def test_bearish_when_below_vwap_and_flip():
    p = deterministic_prediction(_mkt(spot=597.0, vwap=600.0, gamma_flip=599.0, rsi=38.0, cvd_slope=-0.05))
    assert p["bias"] == "bearish"
    assert p["target"] <= p["spot_at_pred"]


def test_target_is_capped_by_walls():
    # Extreme momentum should still not project beyond the wall magnets.
    p = deterministic_prediction(_mkt(spot=605.5, call_wall=606.0, rsi=90.0, cvd_slope=1.0,
                                      expected_range=8.0))
    assert p["target"] <= 606.0 + 1e-6


def test_returns_none_without_spot():
    assert deterministic_prediction(SimpleNamespace(spot=None)) is None


def test_predict_falls_back_to_deterministic_without_package():
    # No spy_der package installed in CI -> deterministic model.
    p = predict_spy_der_tick(_mkt(), now_iso="2026-07-22T11:00:00-04:00")
    assert p is not None and p["source"] == "deterministic"
    assert p["generated_at"] == "2026-07-22T11:00:00-04:00"
