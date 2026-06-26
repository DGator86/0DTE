"""
resample.py
===========
Raw 1-minute OHLCV bars -> per-timeframe indicators -> MTFInput.native.

Consumes a DataFrame (or numpy arrays) of 1-minute bars; resamples into
the four standard 0DTE analysis timeframes (1m, 5m, 15m, 1h); computes
a consistent indicator set on each; and packages the result into the
MTFInput struct that mtf_matrix.build_matrix() expects.

Indicators per timeframe (8):
  adx         - Average Directional Index: trend strength (0..100+)
  rsi         - Relative Strength Index: momentum
  ema_dist    - Distance from 20-period EMA as % of EMA (+/-, signed)
  bb_width    - Bollinger Band width (2-sigma) / mid as decimal fraction
  rv          - Realized variance annualized (from close-to-close log returns)
  cvd         - Cumulative Volume Delta (signed, normalised -1..+1)
                  proxy: sign(close - open) * volume, cumulative
  vwap_dist   - Distance from VWAP as % of VWAP (+/-, signed)
  tick_abs    - Mean |$TICK| (filled from `tick` column if present, else NaN)

Deps: pandas, numpy.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Input / Output types                                                         #
# --------------------------------------------------------------------------- #
@dataclass
class RawBars:
    """1-minute OHLCV bars + optional signed_volume / tick columns."""
    timestamp: np.ndarray     # UTC unix seconds (int64) or datetime64
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    signed_volume: Optional[np.ndarray] = None  # pre-computed CVD input
    tick: Optional[np.ndarray] = None            # $TICK per bar


# The MTFInput struct lives in mtf_matrix.py and is imported there.
# We re-export it here so callers can do `from resample import MTFInput, build_mtf_input`.
@dataclass
class MTFInput:
    """
    Packaged multi-timeframe input for mtf_matrix.build_matrix().
    native: {"1m": {"adx": ..., "rsi": ..., ...}, "5m": {...}, ...}
    snapshot: dealer/vol positioning fields (filled by the caller from the live feed)
    """
    native: dict[str, dict[str, float]]
    snapshot: dict[str, float] = field(default_factory=dict)


TIMEFRAMES = ["1m", "5m", "15m", "1h"]
TF_MINUTES = {"1m": 1, "5m": 5, "15m": 15, "1h": 60}


# --------------------------------------------------------------------------- #
# Indicator helpers (pure functions, no side effects)                         #
# --------------------------------------------------------------------------- #
def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0).ewm(alpha=1.0 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0.0)).ewm(alpha=1.0 / n, adjust=False).mean()
    rs = gain / loss.replace(0.0, np.nan)
    return (100.0 - 100.0 / (1.0 + rs)).fillna(50.0)


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / n, adjust=False).mean()

    up = high.diff().clip(lower=0.0)
    dn = (-low.diff()).clip(lower=0.0)
    dm_pos = np.where(up > dn, up, 0.0)
    dm_neg = np.where(dn > up, dn, 0.0)

    di_pos = 100.0 * pd.Series(dm_pos, index=close.index).ewm(alpha=1.0 / n, adjust=False).mean() / atr.replace(0.0, np.nan)
    di_neg = 100.0 * pd.Series(dm_neg, index=close.index).ewm(alpha=1.0 / n, adjust=False).mean() / atr.replace(0.0, np.nan)
    dx = 100.0 * (di_pos - di_neg).abs() / (di_pos + di_neg).replace(0.0, np.nan)
    return dx.ewm(alpha=1.0 / n, adjust=False).mean().fillna(0.0)


def _bollinger(close: pd.Series, n: int = 20, k: float = 2.0) -> pd.Series:
    """Returns BB width / mid as a fractional decimal."""
    mid = close.rolling(n).mean()
    std = close.rolling(n).std()
    return ((2.0 * k * std) / mid.replace(0.0, np.nan)).fillna(0.0)


def _rv(close: pd.Series, n: int = 20) -> pd.Series:
    """Realized variance: rolling var of log returns, annualised to 1-min periods."""
    lr = np.log(close / close.shift(1))
    var = lr.rolling(n).var(ddof=1) * (252 * 390)   # annualize from 1-min returns
    return var.fillna(0.0)


def _cvd(close: pd.Series, open_: pd.Series, volume: pd.Series,
          signed_vol: Optional[pd.Series] = None) -> pd.Series:
    """
    Cumulative volume delta, normalized by total cumulative volume so it sits
    in (-1, +1). Falls back to (sign(close-open) * volume) when signed_vol is
    absent — only valid for bar-level aggregation, not tick-by-tick.
    """
    if signed_vol is not None:
        raw = signed_vol.cumsum()
    else:
        direction = np.sign(close - open_)
        raw = (direction * volume).cumsum()
    tot = volume.cumsum().replace(0.0, np.nan)
    return (raw / tot).fillna(0.0)


def _vwap(close: pd.Series, high: pd.Series, low: pd.Series,
          volume: pd.Series) -> pd.Series:
    typ = (high + low + close) / 3.0
    cum_pv = (typ * volume).cumsum()
    cum_v = volume.cumsum().replace(0.0, np.nan)
    return cum_pv / cum_v


def _to_df(bars: RawBars) -> pd.DataFrame:
    ts = bars.timestamp
    if np.issubdtype(np.array(ts).dtype, np.integer):
        index = pd.to_datetime(ts, unit="s", utc=True)
    else:
        index = pd.DatetimeIndex(ts)
    df = pd.DataFrame({
        "open": bars.open, "high": bars.high,
        "low": bars.low, "close": bars.close,
        "volume": bars.volume,
    }, index=index)
    if bars.signed_volume is not None:
        df["signed_volume"] = bars.signed_volume
    if bars.tick is not None:
        df["tick"] = bars.tick
    return df.sort_index()


def _resample_df(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    rule = f"{minutes}min"
    agg = {
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum",
    }
    if "signed_volume" in df.columns:
        agg["signed_volume"] = "sum"
    if "tick" in df.columns:
        agg["tick"] = "mean"
    return df.resample(rule).agg(agg).dropna(subset=["close"])


def _last_indicator(s: pd.Series) -> float:
    v = s.dropna()
    return float(v.iloc[-1]) if len(v) else float("nan")


def _indicators(df: pd.DataFrame, tf_label: str) -> dict[str, float]:
    close = df["close"]
    high = df["high"]
    low = df["low"]
    open_ = df["open"]
    vol = df["volume"]
    sv = df.get("signed_volume")
    tick_col = df.get("tick")

    ema20 = _ema(close, 20)
    ema_dist = ((close - ema20) / ema20.replace(0.0, np.nan)).fillna(0.0)
    vwap = _vwap(close, high, low, vol)
    vwap_dist = ((close - vwap) / vwap.replace(0.0, np.nan)).fillna(0.0)

    adx_s = _adx(high, low, close, n=14)
    rsi_s = _rsi(close, n=14)
    bb_s = _bollinger(close, n=20, k=2.0)
    rv_s = _rv(close, n=20)
    cvd_s = _cvd(close, open_, vol, sv)

    tick_abs = float("nan")
    if tick_col is not None and not tick_col.dropna().empty:
        tick_abs = float(tick_col.abs().dropna().mean())

    return {
        "adx": _last_indicator(adx_s),
        "rsi": _last_indicator(rsi_s),
        "ema_dist": _last_indicator(ema_dist),
        "bb_width": _last_indicator(bb_s),
        "rv": _last_indicator(rv_s),
        "cvd": _last_indicator(cvd_s),
        "vwap_dist": _last_indicator(vwap_dist),
        "tick_abs": tick_abs,
    }


# --------------------------------------------------------------------------- #
# Top-level                                                                    #
# --------------------------------------------------------------------------- #
def build_mtf_input(bars: RawBars,
                    snapshot: Optional[dict] = None) -> MTFInput:
    """
    Build the MTFInput from 1-minute bars + optional dealer/vol snapshot dict.
    `snapshot` keys match gate_scorer.MarketSnapshot field names.
    """
    df1 = _to_df(bars)
    native: dict[str, dict[str, float]] = {}

    for tf in TIMEFRAMES:
        mins = TF_MINUTES[tf]
        df = df1 if mins == 1 else _resample_df(df1, mins)
        if len(df) < 2:
            native[tf] = {k: float("nan") for k in
                          ("adx", "rsi", "ema_dist", "bb_width", "rv", "cvd", "vwap_dist", "tick_abs")}
        else:
            native[tf] = _indicators(df, tf)

    return MTFInput(native=native, snapshot=snapshot or {})


# --------------------------------------------------------------------------- #
# Demo                                                                         #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n = 390   # 1 full trading day of 1m bars
    ts = np.arange(1_750_000_000, 1_750_000_000 + n * 60, 60)
    log_ret = rng.normal(0.0, 0.0008, n)
    close = 600.0 * np.exp(np.cumsum(log_ret))
    open_ = np.roll(close, 1); open_[0] = 600.0
    high = np.maximum(open_, close) * (1.0 + rng.uniform(0, 0.0005, n))
    low = np.minimum(open_, close) * (1.0 - rng.uniform(0, 0.0005, n))
    vol = rng.integers(5_000, 50_000, n).astype(float)

    bars = RawBars(timestamp=ts, open=open_, high=high, low=low, close=close, volume=vol)
    inp = build_mtf_input(bars)

    for tf, ind in inp.native.items():
        print(f"{tf:>3}: adx={ind['adx']:.1f}  rsi={ind['rsi']:.1f}  "
              f"bb={ind['bb_width']:.4f}  cvd={ind['cvd']:.3f}")
    print("resample OK")
