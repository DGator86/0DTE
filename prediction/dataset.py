"""
prediction/dataset.py
=====================
Canonical observation construction for Prediction Engine V2
(docs/PREDICTION_ENGINE_V2_HANDOFF.md §8).

One observation = symbol + session_date + decision timestamp, identified by a
STABLE snapshot_id:

    SHA256(symbol | normalized ET timestamp | feature version | source seq)

so rebuilding the dataset from identical recordings reproduces identical ids
(and identical dataset hashes — see storage.dataset_hash).

Session identity uses exchange-local (America/New_York) session dates via
market_calendar / exchange_calendars, distinguishing regular sessions,
early closes, and non-sessions (holidays / weekends / outages).

The offline entry point `build_dataset_from_recording()` turns a
chain_store recording directory into feature_snapshots + observation_labels
rows in a PredictionStore — the leakage-safe research dataset. Features are
taken from each tick's own recorded snapshot (as-of by construction, and
defensively re-checked); labels are computed per session AFTER the session's
bars are complete, with structural levels frozen at observation time.

NOT financial advice.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import math
from dataclasses import dataclass, field
from typing import Optional
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")

# Version of the raw feature definition captured below. Bump when the set or
# meaning of recorded features changes; snapshot ids and scaler state are both
# keyed by it, so incompatible datasets can never silently mix.
FEATURE_VERSION = "v2.0.0"

LABEL_VERSION = "v2.0.0"


# --------------------------------------------------------------------------- #
# Stable snapshot identity (§8.1)                                              #
# --------------------------------------------------------------------------- #
def normalize_ts(ts: dt.datetime) -> str:
    """Canonical ET timestamp string (second precision) for hashing."""
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=ET)
    return ts.astimezone(ET).replace(microsecond=0).isoformat()


def make_snapshot_id(symbol: str, ts: dt.datetime,
                     feature_version: str = FEATURE_VERSION,
                     source_seq: int = 0) -> str:
    payload = f"{symbol}|{normalize_ts(ts)}|{feature_version}|{source_seq}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# Session identity (§8.2)                                                      #
# --------------------------------------------------------------------------- #
def session_metadata(ts: dt.datetime) -> dict:
    """
    Exchange-session metadata for an observation timestamp. Non-session dates
    (weekends, holidays) return is_session=False with None timing fields —
    the dataset builder must be able to say "this is not a session" rather
    than fabricate minutes-to-close.
    """
    from market_calendar import (_calendar, _session_open_close,
                                 _session_type, _to_et)
    ts_et = _to_et(ts)
    date_str = ts_et.date().isoformat()
    out = {
        "session_date": date_str,
        "is_session": False,
        "session_open": None,
        "session_close": None,
        "is_early_close": None,
        "minutes_since_open": None,
        "minutes_to_close": None,
        "day_of_week": ts_et.weekday(),
    }
    if not _calendar().is_session(date_str):
        return out
    open_et, close_et = _session_open_close(date_str)
    out.update({
        "is_session": True,
        "session_open": open_et.isoformat(),
        "session_close": close_et.isoformat(),
        "is_early_close": _session_type(close_et) == "early_close",
        "minutes_since_open": round((ts_et - open_et).total_seconds() / 60.0, 3),
        "minutes_to_close": round((close_et - ts_et).total_seconds() / 60.0, 3),
    })
    return out


# --------------------------------------------------------------------------- #
# Observation row (§8.4)                                                       #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ObservationRow:
    snapshot_id: str
    symbol: str
    session_date: str
    ts: str
    minutes_since_open: Optional[float]
    minutes_to_close: Optional[float]
    spot: float
    feature_version: str
    features: dict = field(default_factory=dict)      # raw values, None=missing
    standardized: dict = field(default_factory=dict)  # 0-100 matrix view (optional)
    missingness: dict = field(default_factory=dict)
    source_ages: dict = field(default_factory=dict)
    quality: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def build_observation(symbol: str, ts: dt.datetime, spot: float, *,
                      features: dict,
                      standardized: Optional[dict] = None,
                      missingness: Optional[dict] = None,
                      source_ages: Optional[dict] = None,
                      quality: Optional[dict] = None,
                      feature_version: str = FEATURE_VERSION,
                      source_seq: int = 0) -> ObservationRow:
    meta = session_metadata(ts)
    q = dict(quality or {})
    q.setdefault("is_session", meta["is_session"])
    if meta["is_early_close"] is not None:
        q.setdefault("is_early_close", meta["is_early_close"])
    return ObservationRow(
        snapshot_id=make_snapshot_id(symbol, ts, feature_version, source_seq),
        symbol=symbol,
        session_date=meta["session_date"],
        ts=normalize_ts(ts),
        minutes_since_open=meta["minutes_since_open"],
        minutes_to_close=meta["minutes_to_close"],
        spot=float(spot),
        feature_version=feature_version,
        features=dict(features),
        standardized=dict(standardized or {}),
        missingness=dict(missingness or {}),
        source_ages=dict(source_ages or {}),
        quality=q,
    )


# --------------------------------------------------------------------------- #
# Offline dataset builder from chain_store recordings                          #
# --------------------------------------------------------------------------- #
def _tick_features(market, bars, observation_ts) -> dict:
    """
    Raw feature capture for one recorded tick under the as-of rule:
    the market snapshot's own fields (recorded AT the tick — age 0) plus the
    mtf snapshot dict. Missing values (None/NaN) are recorded as missing.
    """
    from prediction.asof import AsOfFeatureBuilder, bars_asof

    b = AsOfFeatureBuilder(observation_ts=observation_ts)
    snap_dict = market.mtf_snapshot()
    for name, v in snap_dict.items():
        b.add(name, float(v) if isinstance(v, (int, float)) else v)
    for name in ("spot", "net_gex", "gamma_flip", "call_wall", "put_wall",
                 "gex_pct_rank", "vix9d", "vix", "vix3m", "vvix",
                 "straddle_breakeven", "expected_range", "adx", "rsi",
                 "bb_width", "vwap", "tick_abs_mean", "cvd_slope"):
        v = getattr(market, name, None)
        b.add(name, float(v) if isinstance(v, (int, float))
              and math.isfinite(float(v)) else None)

    # Native multi-timeframe indicators, keyed "name:tf" like the V2 scalers.
    if bars is not None and len(bars.ts):
        safe_bars = bars_asof(bars, observation_ts)   # defensive: no future bars
        if len(safe_bars.ts):
            try:
                from resample import build_mtf_input
                mtf = build_mtf_input(safe_bars, {})
                for name, per_tf in mtf.native.items():
                    for tf, v in per_tf.items():
                        b.add(f"{name}:{tf}",
                              float(v) if isinstance(v, (int, float))
                              and math.isfinite(float(v)) else None)
            except Exception:
                pass                       # thin history: native block missing
    return b.build()


def build_dataset_from_recording(directory: str, store, symbol: str = "SPY",
                                 feature_version: str = FEATURE_VERSION
                                 ) -> dict:
    """
    Rebuild the canonical dataset from a chain_store recording directory into
    a storage.PredictionStore. Deterministic: identical recordings produce
    identical snapshot ids, rows, and dataset hash (storage.dataset_hash).

    Returns {"observations": n, "labeled": n, "sessions": [...]}.
    """
    from chain_store import RecordedFeed
    from prediction.labels import SessionLabeler

    feed = RecordedFeed(directory)

    # Pass 1 — observations, with structural levels remembered for labeling.
    per_session: dict[str, list] = {}
    n_obs = 0
    for seq, ts, snap in feed.replay_ticks():
        built = _tick_features(snap.market, snap.bars, ts)
        row = build_observation(
            symbol, ts, snap.market.spot,
            features=built["features"],
            missingness=built["missingness"],
            source_ages=built["source_ages"],
            quality={"feature_coverage": built["coverage"],
                     "has_chain": snap.chain is not None},
            feature_version=feature_version,
            source_seq=seq,
        )
        store.log_feature_snapshot(row)
        n_obs += 1
        per_session.setdefault(row.session_date, []).append({
            "snapshot_id": row.snapshot_id,
            "ts": ts,
            "spot": snap.market.spot,
            "call_wall": snap.market.call_wall,
            "put_wall": snap.market.put_wall,
            "gamma_flip": snap.market.gamma_flip,
            "bars": snap.bars,
        })

    # Pass 2 — labels, from each session's COMPLETE bar path, one session at
    # a time; levels frozen at each observation's own recorded values.
    n_labeled = 0
    for session_date, obs_list in per_session.items():
        last_bars = obs_list[-1]["bars"]
        if last_bars is None or not len(last_bars.ts):
            continue
        # Restrict the label path to this session's own bars: the recording
        # carries a rolling lookback window that spans prior sessions.
        first_ts = min(o["ts"] for o in obs_list)
        import numpy as np
        from prediction.asof import _to_naive_utc
        ts_arr = np.asarray(last_bars.ts, dtype="datetime64[ns]")
        lo = int(np.searchsorted(ts_arr, np.datetime64(_to_naive_utc(first_ts)),
                                 side="left"))
        if lo >= len(ts_arr):
            continue
        labeler = SessionLabeler(ts=ts_arr[lo:], high=last_bars.high[lo:],
                                 low=last_bars.low[lo:],
                                 close=last_bars.close[lo:])
        for o in obs_list:
            labels = labeler.label_observation(
                o["ts"], o["spot"],
                call_wall=o["call_wall"], put_wall=o["put_wall"],
                gamma_flip=o["gamma_flip"])
            store.log_labels(o["snapshot_id"], labels,
                             label_version=LABEL_VERSION)
            n_labeled += 1

    return {"observations": n_obs, "labeled": n_labeled,
            "sessions": sorted(per_session)}
