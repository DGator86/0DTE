"""
volatility_channel_features.py
==============================
Bridge between the raw bar stream and the regime classifier for the
volatility-channel indicators (Bollinger / Keltner / Donchian).

The channel math itself lives in resample._channel_features (single source of
truth, shared with the multi-timeframe matrix). This module resamples the
classifier's bar stream to one working timeframe (5m, matching the
classifier's existing bar-derived technicals in massive_feed._bar_technicals)
and exposes the raw channel dict, plus a couple of derived higher-level
signals used by regime scoring.

Raw feature semantics (see resample._channel_features for full definitions):
    bb_width / keltner_width / donchian_width   percentile rank 0..1
    bb_position / keltner_position /
        donchian_position                       0..1 inside the channel
    bb_squeeze                                  graded TTM squeeze 0..1
    bb_expansion                                width rate of change (signed)
    keltner_trend_strength                      signed midline persistence
    donchian_breakout_up / _down                ATRs of penetration (>= 0)

NOT financial advice.
"""
from __future__ import annotations

from typing import Optional

from resample import TF_RULE, _channel_features, resample_ohlcv

# Working timeframe for the classifier path. The classifier is a point-in-time
# view; 5m matches its other bar-derived technicals (adx/rsi/cvd in
# massive_feed._bar_technicals). The multi-timeframe view lives in mtf_matrix.
CLASSIFIER_TF = "5m"


def channel_features_from_bars(bars, tf: str = CLASSIFIER_TF) -> dict:
    """Raw channel feature dict from a resample.RawBars stream at one TF.

    Returns {} when bars are missing/too thin or on any computation error --
    the classifier treats absent keys as feature None (reliability 0), which
    is the honest-degradation contract everywhere else.
    """
    if bars is None or getattr(bars, "close", None) is None or len(bars.close) < 2:
        return {}
    try:
        rs = resample_ohlcv(bars.to_frame(), TF_RULE[tf])
        h = rs["high"].to_numpy(float)
        l = rs["low"].to_numpy(float)
        c = rs["close"].to_numpy(float)
        return _channel_features(h, l, c)
    except Exception as exc:
        import logging
        logging.getLogger("volatility_channel_features").warning(
            "channel_features_from_bars failed: %s", exc)
        return {}


def donchian_breakout_strength(chan: dict) -> Optional[float]:
    """Direction-agnostic breakout strength: max penetration (in ATRs) of
    either Donchian extreme. 0 = inside the channel. None when unavailable."""
    up = chan.get("donchian_breakout_up")
    dn = chan.get("donchian_breakout_down")
    if up is None and dn is None:
        return None
    return max(up or 0.0, dn or 0.0)
