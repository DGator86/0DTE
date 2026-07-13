"""
tests/test_channel_features.py
==============================
Unit tests for the Bollinger / Keltner / Donchian channel features:
resample._channel_features math, matrix registration, the classifier
feature path, and the decision-matrix conviction adjustment.
"""
from __future__ import annotations

import numpy as np
import pytest

from resample import (
    BB_P, CHANNEL_KEYS, DONCH_P, KC_P, NATIVE_KEYS,
    _atr_series, _channel_features, compute_tf_features,
)
from mtf_matrix import VARS, MatrixRow


# --------------------------------------------------------------------------- #
# Synthetic bar builders                                                       #
# --------------------------------------------------------------------------- #
def _bars_flat(n=200, base=100.0, wiggle=0.05, spread=0.5, seed=1):
    """Tight closes inside wide bar ranges: Bollinger inside Keltner (squeeze)."""
    rng = np.random.default_rng(seed)
    c = base + wiggle * rng.standard_normal(n)
    h = c + spread
    l = c - spread
    return h, l, c


def _bars_breakout_up(n=200, base=100.0, jump_bars=5, jump=1.0, seed=2):
    """Flat coil then a hard ramp up through the prior Donchian high."""
    h, l, c = _bars_flat(n, base, seed=seed)
    ramp = jump * np.arange(1, jump_bars + 1)
    c[-jump_bars:] = base + ramp
    h[-jump_bars:] = c[-jump_bars:] + 0.2
    l[-jump_bars:] = c[-jump_bars:] - 0.2
    return h, l, c


def _bars_trend(n=200, base=100.0, drift=0.10, spread=0.3, seed=3):
    """Steady uptrend: price should ride the upper Keltner half."""
    rng = np.random.default_rng(seed)
    c = base + drift * np.arange(n) + 0.02 * rng.standard_normal(n)
    h = c + spread
    l = c - spread
    return h, l, c


# --------------------------------------------------------------------------- #
# resample._channel_features math                                              #
# --------------------------------------------------------------------------- #
def test_short_history_returns_all_none():
    h, l, c = _bars_flat(n=5)
    out = _channel_features(h, l, c)
    assert set(out) == set(CHANNEL_KEYS)
    assert all(v is None for v in out.values())


def test_all_keys_present_with_full_history():
    h, l, c = _bars_flat(n=300)
    out = _channel_features(h, l, c)
    assert set(out) == set(CHANNEL_KEYS)
    assert all(v is not None for v in out.values())


def test_squeeze_detected_in_tight_coil():
    # closes barely move but true ranges are wide -> BB deep inside Keltner
    h, l, c = _bars_flat(n=300, wiggle=0.05, spread=0.5)
    out = _channel_features(h, l, c)
    assert out["bb_squeeze"] == pytest.approx(1.0)
    # no breakout while coiling
    assert out["donchian_breakout_up"] == 0.0
    assert out["donchian_breakout_down"] == 0.0
    # positions near mid-channel
    assert 0.2 < out["bb_position"] < 0.8
    assert 0.2 < out["keltner_position"] < 0.8


def test_no_squeeze_when_closes_span_the_range():
    # closes as volatile as the bar ranges -> BB at/outside Keltner
    rng = np.random.default_rng(7)
    c = 100.0 + np.cumsum(0.8 * rng.standard_normal(300))
    h = c + 0.1
    l = c - 0.1
    out = _channel_features(h, l, c)
    assert out["bb_squeeze"] == pytest.approx(0.0)


def test_donchian_breakout_up_graded_in_atrs():
    h, l, c = _bars_breakout_up(jump=1.0)
    out = _channel_features(h, l, c)
    assert out["donchian_breakout_up"] > 0.0
    assert out["donchian_breakout_down"] == 0.0
    # close beyond both channels' upper region
    assert out["donchian_position"] > 0.9
    assert out["bb_position"] > 0.9
    # width expanding through the break
    assert out["bb_expansion"] > 0.0
    # stronger jump -> stronger breakout reading
    h2, l2, c2 = _bars_breakout_up(jump=2.0)
    out2 = _channel_features(h2, l2, c2)
    assert out2["donchian_breakout_up"] > out["donchian_breakout_up"]


def test_donchian_breakout_down_mirror():
    h, l, c = _bars_breakout_up(jump=1.0)
    # mirror the series around the base to get a breakdown
    base = 100.0
    c2 = 2 * base - c
    h2 = 2 * base - l
    l2 = 2 * base - h
    out = _channel_features(h2, l2, c2)
    assert out["donchian_breakout_down"] > 0.0
    assert out["donchian_breakout_up"] == 0.0
    assert out["donchian_position"] < 0.1


def test_keltner_trend_strength_signed():
    h, l, c = _bars_trend(drift=0.10)
    up = _channel_features(h, l, c)
    assert up["keltner_trend_strength"] > 0.1          # riding upper half
    h2, l2, c2 = _bars_trend(drift=-0.10)
    dn = _channel_features(h2, l2, c2)
    assert dn["keltner_trend_strength"] < -0.1         # riding lower half


def test_width_ranks_are_percentiles():
    h, l, c = _bars_breakout_up()
    out = _channel_features(h, l, c)
    for k in ("bb_width", "keltner_width", "donchian_width"):
        assert 0.0 <= out[k] <= 1.0
    # a fresh breakout puts current widths near the top of their history
    assert out["bb_width"] > 0.8
    assert out["donchian_width"] > 0.8


def test_atr_series_alignment_and_positivity():
    h, l, c = _bars_flat(n=100)
    atr = _atr_series(h, l, c, KC_P)
    assert atr is not None and len(atr) == len(c) - 1
    assert np.isfinite(atr[-1]) and atr[-1] > 0
    assert _atr_series(h[:5], l[:5], c[:5], KC_P) is None


# --------------------------------------------------------------------------- #
# Wiring: NATIVE_KEYS, compute_tf_features, matrix registry                     #
# --------------------------------------------------------------------------- #
def test_channel_keys_in_native_keys_and_vars():
    assert set(CHANNEL_KEYS) <= set(NATIVE_KEYS)
    channel_vars = {v.name for v in VARS if v.domain == "channel"}
    assert channel_vars == set(CHANNEL_KEYS)
    assert all(v.kind == "native" for v in VARS if v.domain == "channel")


def test_compute_tf_features_returns_channel_keys():
    import pandas as pd
    h, l, c = _bars_flat(n=300)
    idx = pd.date_range("2026-06-01 09:30", periods=len(c), freq="1min")
    rs = pd.DataFrame({"open": c, "high": h, "low": l, "close": c,
                       "volume": np.full(len(c), 1e4),
                       "svol": np.zeros(len(c))}, index=idx)
    feats = compute_tf_features(rs)
    for k in CHANNEL_KEYS:
        assert k in feats


def test_matrix_standardization_semantics():
    """No-breakout must standardize to the 50-neutral; squeeze 0..1 -> 0..100."""
    by = {v.name: v for v in VARS if v.domain == "channel"}
    assert by["donchian_breakout_up"].std(0.0) == 50.0
    assert by["donchian_breakout_up"].std(2.0) > 90.0
    assert by["bb_squeeze"].std(0.0) == 0.0
    assert by["bb_squeeze"].std(1.0) == 100.0
    assert by["keltner_position"].std(0.5) == 50.0
    assert by["keltner_position"].std(1.5) == 100.0    # clipped outside channel


# --------------------------------------------------------------------------- #
# Classifier feature path                                                      #
# --------------------------------------------------------------------------- #
def _classifier_market():
    import datetime as dt
    from zoneinfo import ZoneInfo
    from gate_scorer import MarketSnapshot
    spot = 600.0
    return MarketSnapshot(
        spot=spot, net_gex=4e9, gamma_flip=spot - 7,
        call_wall=spot + 5, put_wall=spot - 5, gex_pct_rank=0.85,
        vix9d=12.0, vix=13.0, vix3m=15.0, vvix=92.0, vvix_baseline=95.0,
        straddle_breakeven=4.0, expected_range=3.2,
        adx=12.0, rsi=51.0, bb_width=1.4, bb_width_baseline=2.0,
        vwap=spot, vwap_reversion_count=5,
        tick_abs_mean=450.0, cvd_slope=0.05,
        now=dt.datetime(2026, 6, 25, 11, 30, tzinfo=ZoneInfo("America/New_York")),
        has_catalyst=False,
    )


def test_classifier_consumes_channel_dict():
    from regime_classifier import ClassifierContext, RegimeClassifier
    chan = {"bb_squeeze": 0.9, "bb_expansion": 0.0,
            "keltner_position": 0.5, "keltner_trend_strength": 0.0,
            "donchian_width": 0.2,
            "donchian_breakout_up": 0.0, "donchian_breakout_down": 0.0}
    clf = RegimeClassifier()
    st = clf.classify(ClassifierContext(market=_classifier_market(), channel=chan))
    v, rel = st.standardized["bb_squeeze"]
    assert v == pytest.approx(90.0) and rel > 0
    v, rel = st.standardized["donchian_breakout"]
    assert v == pytest.approx(50.0) and rel > 0          # no breakout = neutral


def test_classifier_degrades_without_channel_dict():
    from regime_classifier import ClassifierContext, RegimeClassifier
    clf = RegimeClassifier()
    st = clf.classify(ClassifierContext(market=_classifier_market()))
    for name in ("bb_squeeze", "donchian_breakout", "keltner_position"):
        v, rel = st.standardized[name]
        assert v is None and rel == 0.0


def test_channel_features_from_bars_handles_thin_input():
    from volatility_channel_features import channel_features_from_bars
    assert channel_features_from_bars(None) == {}


def test_channel_features_from_bars_full_stream():
    from resample import _synth_bars
    from volatility_channel_features import (
        channel_features_from_bars, donchian_breakout_strength,
    )
    chan = channel_features_from_bars(_synth_bars(days=5))
    assert set(chan) == set(CHANNEL_KEYS)
    assert chan["bb_squeeze"] is not None
    assert donchian_breakout_strength(chan) is not None
    assert donchian_breakout_strength({}) is None


# --------------------------------------------------------------------------- #
# Decision-matrix conviction adjustment                                        #
# --------------------------------------------------------------------------- #
def _rows(squeeze=None, brk_up=None, brk_dn=None):
    from mtf_matrix import TIMEFRAMES
    rows = []
    for name, val in (("bb_squeeze", squeeze),
                      ("donchian_breakout_up", brk_up),
                      ("donchian_breakout_down", brk_dn)):
        if val is None:
            continue
        rows.append(MatrixRow("channel", name, "native",
                              {tf: val for tf in TIMEFRAMES}))
    return rows


def test_size_adjust_boosts_credit_in_squeeze():
    from decision_matrix import CH_BOOST, _channel_size_adjust
    mult, note = _channel_size_adjust(_rows(squeeze=80, brk_up=50, brk_dn=50), "IC")
    assert mult == CH_BOOST and "boost" in note


def test_size_adjust_trims_on_opposing_breakout():
    from decision_matrix import CH_TRIM, _channel_size_adjust
    # bull-exposed PCS vs a downside break
    mult, note = _channel_size_adjust(_rows(squeeze=20, brk_dn=80), "PCS")
    assert mult == CH_TRIM and "down" in note
    # bear-exposed CCS vs an upside break
    mult, _ = _channel_size_adjust(_rows(brk_up=80), "CCS")
    assert mult == CH_TRIM
    # neutral premium trimmed on either side
    mult, _ = _channel_size_adjust(_rows(brk_up=80), "IC")
    assert mult == CH_TRIM
    # supportive break does NOT trim (PCS vs upside break)
    mult, _ = _channel_size_adjust(_rows(brk_up=80), "PCS")
    assert mult == 1.0


def test_size_adjust_neutral_without_channel_rows():
    from decision_matrix import _channel_size_adjust
    assert _channel_size_adjust([], "IC") == (1.0, "channels unavailable")


def test_size_adjust_never_boosts_debit_structures():
    from decision_matrix import _channel_size_adjust
    mult, _ = _channel_size_adjust(_rows(squeeze=90, brk_up=50, brk_dn=50), "LCS")
    assert mult == 1.0


def test_decide_from_matrix_applies_channel_note():
    """End-to-end: a squeeze-with-no-breakout matrix boosts a credit intent."""
    from decision_matrix import decide_from_matrix
    from mtf_matrix import TIMEFRAMES

    def row(name, val, domain="trend"):
        return MatrixRow(domain, name, "native", {tf: val for tf in TIMEFRAMES})

    rows = [
        row("adx_strength", 20), row("bb_compression", 80),
        row("rv_expansion", 40), row("di_spread", 50), row("ema_slope", 50),
        row("cvd_persistence", 50), row("vwap_slope", 50), row("rsi", 50),
        row("trend_cleanliness", 30),
        row("bb_squeeze", 85, "channel"),
        row("donchian_breakout_up", 50, "channel"),
        row("donchian_breakout_down", 50, "channel"),
        row("donchian_width", 20, "channel"),
        row("keltner_trend_strength", 50, "channel"),
        row("bb_expansion", 45, "channel"),
    ]
    regimes = {
        "compression": {tf: 80.0 for tf in TIMEFRAMES},
        "trend": {tf: 30.0 for tf in TIMEFRAMES},
        "breakout": {tf: 20.0 for tf in TIMEFRAMES},
    }
    intent = decide_from_matrix(rows, regimes, vetoes=[])
    assert intent.decision.structure == "IC"
    assert intent.size_mult == pytest.approx(1.15)
    assert "channel boost" in intent.note
