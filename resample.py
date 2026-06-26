"""
resample.py
===========
The feed layer under mtf_matrix.py. Takes a raw bar stream (base resolution,
e.g. 1-minute OHLCV), resamples to the seven timeframes, computes the standard
indicators at each resolution, and emits the `native` dict that
mtf_matrix.MTFInput expects -- so the matrix, regime classifier, and decision
table run on real data instead of a hand-built dict.

What it produces (per timeframe), matching mtf_matrix's NATIVE variables:
    dist_to_vwap, vwap_slope, range_position, realized_vol, rv_expansion,
    adx_strength, di_spread, ema_slope, rsi, bb_compression,
    trend_cleanliness, cvd_persistence, tick_two_sided

Honest degradation:
  * Higher timeframes need history. A 14-period ADX on the 1d bar needs ~14
    days of base bars; if the series is too short for an indicator at a given
    TF, that cell is None and drops out downstream (reliability 0).
  * cvd_persistence uses a signed-volume series if you supply one; otherwise it
    falls back to a close-location proxy. tick_two_sided needs a real $TICK
    feed -- without it the cell is None (the matrix shows it absent, by design).

Indicators are implemented from OHLCV directly (no TA-Lib dependency).
NOT financial advice.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from mtf_matrix import MTFInput, TIMEFRAMES

# pandas resample rules for each target timeframe
TF_RULE = {"1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min",
           "1h": "60min", "4h": "240min", "1d": "1D"}

# indicator periods (in TF bars)
ADX_P = RSI_P = ATR_P = 14
EMA_P = 20
BB_P = 20
SLOPE_LB = 5          # bars back for slope measures
RV_W = 20             # realized-vol window
R2_W = 20             # trend-cleanliness window
VWAP_W = 20           # rolling VWAP window (per TF)
RV_RANK_W = 100       # window to percentile-rank realized vol


# --------------------------------------------------------------------------- #
# Raw input                                                                    #
# --------------------------------------------------------------------------- #
@dataclass
class RawBars:
    ts: np.ndarray            # datetime64[ns], base resolution (e.g. 1-minute)
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    signed_volume: Optional[np.ndarray] = None   # for CVD; else proxy is used
    tick: Optional[np.ndarray] = None            # NYSE $TICK per base bar; else None

    def to_frame(self) -> pd.DataFrame:
        d = {"open": self.open, "high": self.high, "low": self.low,
             "close": self.close, "volume": self.volume}
        if self.signed_volume is not None:
            d["svol"] = self.signed_volume
        else:
            # close-location proxy: where in the bar's range it closed, * volume
            rng = np.where(self.high > self.low, self.high - self.low, np.nan)
            clv = (2 * (self.close - self.low) / rng - 1)
            d["svol"] = np.nan_to_num(clv) * self.volume
        if self.tick is not None:
            d["tick"] = self.tick
        return pd.DataFrame(d, index=pd.DatetimeIndex(self.ts))


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {"open": "first", "high": "max", "low": "min", "close": "last",
           "volume": "sum", "svol": "sum"}
    if "tick" in df.columns:
        agg["tick"] = "mean"
    out = df.resample(rule, label="right", closed="right").agg(agg).dropna(subset=["close"])
    return out


# --------------------------------------------------------------------------- #
# Indicators (numpy, last-value)                                               #
# --------------------------------------------------------------------------- #
def _wilder_rma(x: np.ndarray, p: int) -> np.ndarray:
    """Wilder's smoothing (RMA)."""
    out = np.full_like(x, np.nan, dtype=float)
    if len(x) < p:
        return out
    out[p - 1] = np.mean(x[:p])
    for i in range(p, len(x)):
        out[i] = (out[i - 1] * (p - 1) + x[i]) / p
    return out


def _adx_di(h, l, c, p=ADX_P):
    n = len(c)
    if n < 2 * p:
        return None, None, None
    up = h[1:] - h[:-1]
    dn = l[:-1] - l[1:]
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = np.maximum.reduce([h[1:] - l[1:], np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])])
    atr = _wilder_rma(tr, p)
    pdi = 100 * _wilder_rma(plus_dm, p) / atr
    mdi = 100 * _wilder_rma(minus_dm, p) / atr
    dx = 100 * np.abs(pdi - mdi) / (pdi + mdi)
    adx = _wilder_rma(np.nan_to_num(dx), p)
    return (float(adx[-1]) if np.isfinite(adx[-1]) else None,
            float(pdi[-1]) if np.isfinite(pdi[-1]) else None,
            float(mdi[-1]) if np.isfinite(mdi[-1]) else None)


def _rsi(c, p=RSI_P):
    if len(c) < p + 1:
        return None
    d = np.diff(c)
    gain = _wilder_rma(np.where(d > 0, d, 0.0), p)
    loss = _wilder_rma(np.where(d < 0, -d, 0.0), p)
    rs = gain[-1] / loss[-1] if loss[-1] > 0 else np.inf
    return float(100 - 100 / (1 + rs))


def _ema(c, p=EMA_P):
    if len(c) < p:
        return None
    k = 2 / (p + 1)
    e = c[0]
    for v in c[1:]:
        e = v * k + e * (1 - k)
    return e


def _ema_series(c, p=EMA_P):
    k = 2 / (p + 1)
    e = np.empty_like(c, dtype=float)
    e[0] = c[0]
    for i in range(1, len(c)):
        e[i] = c[i] * k + e[i - 1] * (1 - k)
    return e


def _ema_slope(c, p=EMA_P, lb=SLOPE_LB):
    if len(c) < p + lb:
        return None
    e = _ema_series(c, p)
    return float((e[-1] - e[-1 - lb]) / e[-1 - lb] * 100 / lb)   # %/bar


def _bb_ratio(c, p=BB_P):
    """Current Bollinger width / its own trailing median (compression ratio)."""
    if len(c) < p * 2:
        return None
    s = pd.Series(c)
    mid = s.rolling(p).mean()
    sd = s.rolling(p).std(ddof=0)
    width = (4 * sd) / mid                      # (upper-lower)/mid, 2σ bands
    w = width.dropna()
    if len(w) < 2 or not np.isfinite(w.iloc[-1]):
        return None
    base = np.median(w.iloc[-p:]) if len(w) >= p else np.median(w)
    return float(w.iloc[-1] / base) if base > 0 else None


def _realized_vol_rank(c, w=RV_W, rank_w=RV_RANK_W):
    if len(c) < w + 2:
        return None
    r = np.diff(np.log(c))
    rv = pd.Series(r).rolling(w).std(ddof=0) * np.sqrt(252 * 390)   # annualized-ish
    rv = rv.dropna()
    if len(rv) < 2:
        return None
    recent = rv.iloc[-rank_w:] if len(rv) >= rank_w else rv
    last = rv.iloc[-1]
    return float((recent < last).mean())        # percentile rank 0..1


def _rv_expansion(c, short_w=5, long_w=RV_W):
    if len(c) < long_w + 2:
        return None
    r = np.diff(np.log(c))
    s = pd.Series(r)
    rv_s = s.rolling(short_w).std(ddof=0).iloc[-1]
    rv_l = s.rolling(long_w).std(ddof=0).iloc[-1]
    if not (np.isfinite(rv_s) and np.isfinite(rv_l)) or rv_l <= 0:
        return None
    return float(rv_s / rv_l - 1.0)             # >0 expanding


def _r2(c, w=R2_W):
    if len(c) < w:
        return None
    y = c[-w:]
    x = np.arange(w)
    xm, ym = x.mean(), y.mean()
    sxx = np.sum((x - xm) ** 2)
    syy = np.sum((y - ym) ** 2)
    if sxx <= 0 or syy <= 0:
        return None
    sxy = np.sum((x - xm) * (y - ym))
    return float((sxy ** 2) / (sxx * syy))      # R^2 0..1


def _vwap_roll(c, v, w=VWAP_W):
    if len(c) < 2:
        return None, None
    n = min(w, len(c))
    cc, vv = c[-n:], v[-n:]
    tot = vv.sum()
    if tot <= 0:
        return None, None
    vwap = float((cc * vv).sum() / tot)
    dist = (c[-1] - vwap) / c[-1] * 100          # % from vwap
    # slope: vwap now vs one bar back
    if n >= 2 and vv[:-1].sum() > 0:
        vwap_prev = (cc[:-1] * vv[:-1]).sum() / vv[:-1].sum()
        slope = (vwap - vwap_prev) / vwap_prev * 100
    else:
        slope = 0.0
    return float(dist), float(slope)


def _range_position(h, l, c, w=VWAP_W):
    n = min(w, len(c))
    hi, lo = h[-n:].max(), l[-n:].min()
    if hi <= lo:
        return None
    return float((c[-1] - lo) / (hi - lo))


def _cvd_slope(svol, vol, lb=SLOPE_LB):
    if len(svol) < lb + 1:
        return None
    cvd = np.cumsum(svol)
    avg_v = np.mean(vol[-lb:]) if np.mean(vol[-lb:]) > 0 else 1.0
    return float((cvd[-1] - cvd[-1 - lb]) / (avg_v * lb))   # normalized per bar


def _tick_two_sided(tick):
    if tick is None:
        return None
    t = pd.Series(tick).dropna()
    if len(t) < 3:
        return None
    return float(t.abs().tail(VWAP_W).mean())


# --------------------------------------------------------------------------- #
# Per-timeframe feature computation                                            #
# --------------------------------------------------------------------------- #
def compute_tf_features(rs: pd.DataFrame) -> dict:
    h = rs["high"].to_numpy(float)
    l = rs["low"].to_numpy(float)
    c = rs["close"].to_numpy(float)
    v = rs["volume"].to_numpy(float)
    sv = rs["svol"].to_numpy(float)
    tick = rs["tick"].to_numpy(float) if "tick" in rs.columns else None

    adx, pdi, mdi = _adx_di(h, l, c)
    dist_vwap, vwap_slope = _vwap_roll(c, v)

    return {
        "dist_to_vwap": dist_vwap,
        "vwap_slope": vwap_slope,
        "range_position": _range_position(h, l, c),
        "realized_vol": _realized_vol_rank(c),
        "rv_expansion": _rv_expansion(c),
        "adx_strength": adx,
        "di_spread": (pdi - mdi) if (pdi is not None and mdi is not None) else None,
        "ema_slope": _ema_slope(c),
        "rsi": _rsi(c),
        "bb_compression": _bb_ratio(c),
        "trend_cleanliness": _r2(c),
        "cvd_persistence": _cvd_slope(sv, v),
        "tick_two_sided": _tick_two_sided(tick),
    }


NATIVE_KEYS = ["dist_to_vwap", "vwap_slope", "range_position", "realized_vol",
               "rv_expansion", "adx_strength", "di_spread", "ema_slope", "rsi",
               "bb_compression", "trend_cleanliness", "cvd_persistence", "tick_two_sided"]


def build_mtf_input(raw: RawBars, snapshot: dict) -> MTFInput:
    """Resample raw bars to every timeframe and assemble the MTFInput.native dict."""
    df = raw.to_frame()
    native = {k: {} for k in NATIVE_KEYS}
    for tf in TIMEFRAMES:
        rs = resample_ohlcv(df, TF_RULE[tf])
        feats = compute_tf_features(rs)
        for k in NATIVE_KEYS:
            native[k][tf] = feats[k]            # may be None for short history
    return MTFInput(native=native, snapshot=snapshot)


# --------------------------------------------------------------------------- #
# Demo: synthesize a coiling multi-day 1m stream, resample, run the matrix      #
# --------------------------------------------------------------------------- #
def _synth_bars(days=20, seed=7) -> RawBars:
    rng = np.random.default_rng(seed)
    per_day = 390
    n = days * per_day
    start = np.datetime64("2026-05-01T13:30:00")  # ~09:30 ET in UTC-ish; spacing only matters
    ts = start + np.arange(n) * np.timedelta64(1, "m")

    # slow daily uptrend + intraday Ornstein-Uhlenbeck mean reversion (=> low-TF
    # compression, high-TF trend), plus a vol term.
    price = np.empty(n)
    p = 600.0
    daily_drift = 0.0008                          # ~8 bps/day trend
    ou_theta, ou_sigma = 0.05, 0.045
    dev = 0.0
    for i in range(n):
        intraday = i % per_day
        if intraday == 0:
            dev = 0.0
        dev += -ou_theta * dev + ou_sigma * rng.standard_normal()
        trend = daily_drift / per_day * i
        p = 600.0 * (1 + trend) + dev
        price[i] = p

    spread = 0.03 + 0.02 * np.abs(rng.standard_normal(n))
    high = price + spread
    low = price - spread
    openp = np.concatenate([[price[0]], price[:-1]])
    vol = 1e5 * (1 + 0.5 * np.abs(rng.standard_normal(n)))
    return RawBars(ts=ts, open=openp, high=high, low=low, close=price, volume=vol)


if __name__ == "__main__":
    from mtf_matrix import build_matrix, regime_rows, render_text
    from decision_matrix import decide_from_matrix

    raw = _synth_bars()
    snapshot = {                                  # dealer/vol state would come from your GEX+RND modules
        "gamma_sign": 4.0e9, "gamma_magnitude": 0.86, "flip_cushion": 0.006,
        "channel_tightness": 0.010, "wall_proximity": 0.0025,
        "term_structure": 0.16, "vvix_elevation": -0.03, "richness": 0.66,
        "skew_dir": -0.17, "tail_heaviness": 0.30,
    }
    inp = build_mtf_input(raw, snapshot)
    rows = build_matrix(inp)
    regimes = regime_rows(rows)
    print(render_text(rows, regimes))
    print("\n(· = insufficient history for that indicator at that timeframe)\n")

    intent = decide_from_matrix(rows, regimes, vetoes=[])
    d = intent.decision
    print(f"DECISION: exec={intent.exec_regime} context={intent.context_regime} "
          f"bias={intent.direction_bias}({intent.bias_value}) -> "
          f"{d.structure}/{d.direction} {d.conviction} x{intent.size_mult}")
