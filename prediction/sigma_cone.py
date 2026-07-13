"""
prediction/sigma_cone.py
========================
Outward-looking MTF sigma cones for Prediction Engine V2.

Contract (per timeframe):
  From spot *now*, emit bands at 0.5σ / 1σ / 2σ. Each band carries:
    * price interval [lo, hi]
    * endogenous horizon (minutes ahead) — wider σ looks further out
  Horizons are derived from anticipated vol (time to accumulate kσ), not
  fixed calendar buckets like "always 30m".

Journal + settlement:
  Every cone band is persisted; when wall-clock reaches settle_by, the
  realized spot is matched against [lo, hi] so coverage / error are measurable.

NOT financial advice.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import math
from dataclasses import dataclass, field
from typing import Optional, Sequence
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")

SIGMA_LEVELS: tuple[float, ...] = (0.5, 1.0, 2.0)
# Intraday MTF panes the cone journal emits (matches mtf_matrix short end).
CONE_TIMEFRAMES: tuple[str, ...] = ("1m", "5m", "15m", "30m")
TF_BAR_MINUTES = {"1m": 1.0, "5m": 5.0, "15m": 15.0, "30m": 30.0}
# Higher TFs look further out for the same σ (anticipated scale).
TF_HORIZON_SCALE = {"1m": 1.0, "5m": 1.35, "15m": 1.7, "30m": 2.1}
# Sub-quadratic stretch: 2σ must not explode to 4× the 1σ horizon.
SIGMA_HORIZON_EXP = 1.35

CONE_MODEL_VERSION = "sigma-cone-v1"


@dataclass(frozen=True)
class ConeBand:
    sigma: float
    lo: float
    hi: float
    horizon_min: float
    mid: float
    settle_by: str                 # ISO timestamp

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class ConeForecast:
    """One MTF pane's outward-looking prediction."""
    snapshot_id: str
    ts: str
    session_date: str
    timeframe: str
    spot: float
    model_version: str = CONE_MODEL_VERSION
    sigma_per_sqrt_min: float = 0.0
    drift_per_min: float = 0.0
    bands: tuple[ConeBand, ...] = ()
    diagnostics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        return d


@dataclass(frozen=True)
class ConeSettlement:
    inside: bool
    realized_spot: float
    realized_ts: str
    error_mid: float
    coverage_note: str

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def _finite(x) -> Optional[float]:
    if isinstance(x, (int, float)) and math.isfinite(float(x)):
        return float(x)
    return None


def estimate_sigma_per_sqrt_min(
    market,
    *,
    signals: Optional[dict] = None,
    minutes_to_close: Optional[float] = None,
) -> float:
    """
    Per-√minute log-return volatility.

    Preference order:
      1. EWMA / RV proxy from bb_width or tick_abs_mean when present
      2. Remaining-session implied move / sqrt(minutes_to_close)
      3. Conservative floor (so cones always emit)
    """
    signals = signals or {}
    spot = _finite(getattr(market, "spot", None)) or 0.0
    mtc = _finite(minutes_to_close)
    if mtc is None:
        mtc = _finite(signals.get("minutes_to_close"))
    if mtc is None or mtc <= 1.0:
        mtc = 120.0

    # Implied remaining-session move → session σ → per-√min
    for attr in ("expected_range", "straddle_breakeven"):
        v = _finite(getattr(market, attr, None))
        if v is not None and v > 0 and spot > 0:
            frac = v / spot
            # treat as ~1σ session move in simple-return space → log approx
            session_sigma = max(math.log1p(frac), 1e-6)
            return session_sigma / math.sqrt(mtc)

    bb = _finite(getattr(market, "bb_width", None))
    if bb is not None and bb > 0 and spot > 0:
        # bb_width is often a raw width or percentile; if > 1 treat as dollars
        frac = (bb / spot) if bb > 1.0 else max(bb, 1e-4) * 0.02
        return max(frac, 1e-5)

    tick = _finite(getattr(market, "tick_abs_mean", None))
    if tick is not None and tick > 0 and spot > 0:
        return max(tick / spot, 1e-5)

    # ~15% annualized → per-√min
    return 0.15 / math.sqrt(252.0 * 390.0)


def estimate_drift_per_min(signals: Optional[dict] = None) -> float:
    """Log-return drift per minute from matrix bias (50 = neutral)."""
    signals = signals or {}
    bias = _finite(signals.get("regime_bias_value"))
    if bias is None:
        bias = _finite(signals.get("bias_value"))
    if bias is None:
        return 0.0
    # Map 0..100 → roughly ± a few bps per minute at extremes (soft).
    centered = (bias - 50.0) / 50.0          # [-1, +1]
    return centered * 0.00015                 # 1.5 bps/min at full bias


def _horizon_minutes(
    sigma: float,
    sigma_per_sqrt_min: float,
    *,
    timeframe: str,
    minutes_to_close: float,
) -> float:
    """
    Endogenous look-ahead: wider σ looks further out; higher vol arrives sooner.

    Uses a session-relative 1σ target (vol-scaled) so bands stay ordered inside
    the remaining session instead of all clamping to the close.
    """
    sig = max(float(sigma_per_sqrt_min), 1e-8)
    mtc = max(float(minutes_to_close), 1.0)
    bar = TF_BAR_MINUTES.get(timeframe, 1.0)
    tf_scale = TF_HORIZON_SCALE.get(timeframe, 1.0)
    # Reference vol (~session implied / √120 ≈ 0.002 for a 2.5% move)
    ref = 0.002
    vol_scale = max(0.35, min(2.5, ref / sig))
    # 1σ horizon target: ~35% of remaining session, capped, TF-scaled
    target_1sig = min(60.0, 0.35 * mtc) * vol_scale * (tf_scale / TF_HORIZON_SCALE["5m"])
    raw = target_1sig * (float(sigma) ** SIGMA_HORIZON_EXP)
    return float(max(bar, min(raw, mtc)))


def build_cone_for_timeframe(
    *,
    snapshot_id: str,
    ts: dt.datetime,
    session_date: str,
    timeframe: str,
    spot: float,
    sigma_per_sqrt_min: float,
    drift_per_min: float = 0.0,
    minutes_to_close: float = 120.0,
    sigma_levels: Sequence[float] = SIGMA_LEVELS,
    model_version: str = CONE_MODEL_VERSION,
) -> ConeForecast:
    """Build one outward-looking cone pane for a timeframe."""
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=ET)
    ts_et = ts.astimezone(ET)
    spot = float(spot)
    if spot <= 0:
        raise ValueError(f"spot must be > 0, got {spot!r}")
    sig = max(float(sigma_per_sqrt_min), 1e-8)
    drift = float(drift_per_min)
    mtc = max(float(minutes_to_close), 1.0)

    bands: list[ConeBand] = []
    for k in sigma_levels:
        T = _horizon_minutes(k, sig, timeframe=timeframe, minutes_to_close=mtc)
        mu = drift * T
        half = float(k) * sig * math.sqrt(T)
        lo = spot * math.exp(mu - half)
        hi = spot * math.exp(mu + half)
        if lo > hi:
            lo, hi = hi, lo
        mid = spot * math.exp(mu)
        settle_by = (ts_et + dt.timedelta(minutes=T)).isoformat()
        bands.append(ConeBand(
            sigma=float(k), lo=lo, hi=hi, horizon_min=T, mid=mid,
            settle_by=settle_by,
        ))

    return ConeForecast(
        snapshot_id=snapshot_id,
        ts=ts_et.isoformat(),
        session_date=session_date,
        timeframe=timeframe,
        spot=spot,
        model_version=model_version,
        sigma_per_sqrt_min=sig,
        drift_per_min=drift,
        bands=tuple(bands),
        diagnostics={
            "minutes_to_close": mtc,
            "tf_scale": TF_HORIZON_SCALE.get(timeframe, 1.0),
        },
    )


def build_mtf_cones(
    *,
    snapshot_id: str,
    ts: dt.datetime,
    session_date: str,
    spot: float,
    market=None,
    signals: Optional[dict] = None,
    minutes_to_close: Optional[float] = None,
    timeframes: Sequence[str] = CONE_TIMEFRAMES,
    sigma_levels: Sequence[float] = SIGMA_LEVELS,
) -> list[ConeForecast]:
    """Emit one cone per MTF pane."""
    signals = dict(signals or {})
    mtc = _finite(minutes_to_close)
    if mtc is None and market is not None:
        try:
            from prediction.dataset import session_metadata
            meta = session_metadata(ts if getattr(ts, "tzinfo", None)
                                    else ts.replace(tzinfo=ET))
            mtc = _finite(meta.get("minutes_to_close"))
        except Exception:
            mtc = None
    if mtc is None:
        mtc = 120.0

    if market is not None:
        sig = estimate_sigma_per_sqrt_min(
            market, signals=signals, minutes_to_close=mtc)
        if spot is None or spot <= 0:
            spot = float(getattr(market, "spot", 0.0) or 0.0)
    else:
        sig = max(_finite(signals.get("sigma_per_sqrt_min")) or 1e-4, 1e-8)

    drift = estimate_drift_per_min(signals)
    out: list[ConeForecast] = []
    for tf in timeframes:
        out.append(build_cone_for_timeframe(
            snapshot_id=snapshot_id,
            ts=ts,
            session_date=session_date,
            timeframe=tf,
            spot=float(spot),
            sigma_per_sqrt_min=sig,
            drift_per_min=drift,
            minutes_to_close=mtc,
            sigma_levels=sigma_levels,
        ))
    return out


def settle_band(
    band: ConeBand,
    *,
    realized_spot: float,
    realized_ts: str,
) -> ConeSettlement:
    """Match a single predicted band against the true spot at settle_by."""
    px = float(realized_spot)
    inside = band.lo <= px <= band.hi
    err = px - band.mid
    note = "inside" if inside else ("above" if px > band.hi else "below")
    return ConeSettlement(
        inside=inside,
        realized_spot=px,
        realized_ts=realized_ts,
        error_mid=err,
        coverage_note=note,
    )


def cones_to_signals(cones: Sequence[ConeForecast]) -> dict:
    """Flat journal keys for signals_json / live_state."""
    out: dict = {
        "cone_model_version": CONE_MODEL_VERSION,
        "cone_n_timeframes": float(len(cones)),
    }
    if not cones:
        return out
    out["cone_sigma_per_sqrt_min"] = float(cones[0].sigma_per_sqrt_min)
    out["cone_drift_per_min"] = float(cones[0].drift_per_min)
    # Prefer 5m pane for compact live readout; fall back to first.
    primary = next((c for c in cones if c.timeframe == "5m"), cones[0])
    out["cone_primary_tf"] = primary.timeframe
    out["cone_spot"] = float(primary.spot)
    for b in primary.bands:
        tag = str(b.sigma).replace(".", "p")
        out[f"cone_{tag}_lo"] = float(b.lo)
        out[f"cone_{tag}_hi"] = float(b.hi)
        out[f"cone_{tag}_mid"] = float(b.mid)
        out[f"cone_{tag}_horizon_min"] = float(b.horizon_min)
    return out


def cones_live_summary(cones: Sequence[ConeForecast]) -> dict:
    """Structured payload for dashboard Prediction tab."""
    panes = []
    for c in cones:
        panes.append({
            "timeframe": c.timeframe,
            "spot": c.spot,
            "ts": c.ts,
            "session_date": c.session_date,
            "snapshot_id": c.snapshot_id,
            "model_version": c.model_version,
            "sigma_per_sqrt_min": c.sigma_per_sqrt_min,
            "drift_per_min": c.drift_per_min,
            "bands": [b.to_dict() for b in c.bands],
        })
    return {
        "model_version": CONE_MODEL_VERSION,
        "panes": panes,
    }
