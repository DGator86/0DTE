"""
prediction/canonical_snapshot.py
================================
CanonicalSnapshot — one immutable market snapshot per tick
(docs/UNIFIED_V1_V2_V3_HANDOFF.md §7.1).

Constructed exactly once. Shared snapshot_id across journal, features,
forecasts, candidates, decisions, and dashboard.

NOT financial advice.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from prediction.dataset import FEATURE_VERSION, make_snapshot_id

SNAPSHOT_SCHEMA_VERSION = "v1.0.0"

# Keys that must never appear in model feature dicts (forecast-policy separation).
_FORBIDDEN_FEATURE_PREFIXES = (
    "selected_", "gate_", "policy_", "candidate_score", "candidate_rank",
    "human_action", "final_action", "authority_",
)
_FORBIDDEN_FEATURE_KEYS = frozenset({
    "structure", "family", "direction", "size_mult", "conviction",
    "trade_decision", "ras_action", "gate_result", "selected_candidate_id",
})


class CanonicalSnapshotError(ValueError):
    """Invalid or future-leaking snapshot construction."""


@dataclass(frozen=True)
class CanonicalSnapshot:
    snapshot_id: str
    symbol: str
    ts: str
    session_date: str
    market: Any = None
    bars: Any = None
    chain: Any = None
    raw_features: dict = field(default_factory=dict)
    standardized_features: dict = field(default_factory=dict)
    missingness: dict = field(default_factory=dict)
    source_timestamps: dict = field(default_factory=dict)
    source_ages_seconds: dict = field(default_factory=dict)
    quality: dict = field(default_factory=dict)
    structural_sources: dict = field(default_factory=dict)
    structural_state: Any = None
    feature_version: str = FEATURE_VERSION
    structural_state_version: Optional[str] = None
    snapshot_schema_version: str = SNAPSHOT_SCHEMA_VERSION

    def snapshot_hash(self) -> str:
        """Deterministic content hash (excludes live object refs)."""
        payload = {
            "snapshot_id": self.snapshot_id,
            "symbol": self.symbol,
            "ts": self.ts,
            "session_date": self.session_date,
            "raw_features": self.raw_features,
            "standardized_features": self.standardized_features,
            "missingness": self.missingness,
            "source_timestamps": self.source_timestamps,
            "source_ages_seconds": self.source_ages_seconds,
            "quality": self.quality,
            "feature_version": self.feature_version,
            "structural_state_version": self.structural_state_version,
            "snapshot_schema_version": self.snapshot_schema_version,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"),
                       default=str).encode("utf-8")).hexdigest()

    def to_dict(self) -> dict:
        return {
            "snapshot_id": self.snapshot_id,
            "symbol": self.symbol,
            "ts": self.ts,
            "session_date": self.session_date,
            "raw_features": dict(self.raw_features),
            "standardized_features": dict(self.standardized_features),
            "missingness": dict(self.missingness),
            "source_timestamps": dict(self.source_timestamps),
            "source_ages_seconds": dict(self.source_ages_seconds),
            "quality": dict(self.quality),
            "structural_sources": dict(self.structural_sources),
            "feature_version": self.feature_version,
            "structural_state_version": self.structural_state_version,
            "snapshot_schema_version": self.snapshot_schema_version,
            "snapshot_hash": self.snapshot_hash(),
        }


def _assert_no_policy_fields(features: Mapping[str, Any]) -> None:
    for key in features:
        k = str(key)
        if k in _FORBIDDEN_FEATURE_KEYS:
            raise CanonicalSnapshotError(
                f"post-routing field {k!r} forbidden in model features")
        for prefix in _FORBIDDEN_FEATURE_PREFIXES:
            if k.startswith(prefix):
                raise CanonicalSnapshotError(
                    f"post-routing field {k!r} forbidden in model features")


def _parse_ts(value: Any):
    """Parse ISO timestamps to timezone-aware UTC datetimes. Reject invalid."""
    import datetime as _dt
    if value is None or value == "":
        return None
    if isinstance(value, _dt.datetime):
        ts = value
    else:
        text = str(value).replace("Z", "+00:00")
        try:
            ts = _dt.datetime.fromisoformat(text)
        except ValueError as exc:
            raise CanonicalSnapshotError(
                f"unparseable timestamp {value!r}") from exc
    if ts.tzinfo is None:
        raise CanonicalSnapshotError(
            f"timestamp {value!r} lacks timezone — refuse naive comparison")
    return ts.astimezone(_dt.timezone.utc)


def _reject_future_sources(
    prediction_ts: str,
    source_timestamps: Mapping[str, Any],
) -> None:
    """Reject any source timestamp strictly after the prediction timestamp."""
    if not prediction_ts:
        return
    pred = _parse_ts(prediction_ts)
    for name, src_ts in source_timestamps.items():
        if src_ts is None or src_ts == "":
            continue
        src = _parse_ts(src_ts)
        if src is not None and pred is not None and src > pred:
            raise CanonicalSnapshotError(
                f"future-dated source {name!r}: {src_ts!r} > {prediction_ts!r}")


def compute_source_ages_seconds(
    prediction_ts: str,
    source_timestamps: Mapping[str, Any],
) -> dict:
    """Age of each source relative to prediction_ts (seconds). Missing → omit."""
    import datetime as _dt
    pred = _parse_ts(prediction_ts)
    if pred is None:
        return {}
    ages: dict = {}
    for name, src_ts in (source_timestamps or {}).items():
        if src_ts is None or src_ts == "":
            continue
        try:
            src = _parse_ts(src_ts)
        except CanonicalSnapshotError:
            continue
        if src is None:
            continue
        ages[str(name)] = max(0.0, (pred - src).total_seconds())
    return ages


def extract_source_timestamps(
    *,
    now_iso: str,
    bars: Any = None,
    chain: Any = None,
    market: Any = None,
    signals: Optional[Mapping[str, Any]] = None,
) -> dict:
    """
    Best-effort real source timestamps (never invent now for bars/chain).

    Missing sources are omitted so ages stay honest rather than zero-age fakes.
    """
    import datetime as _dt
    out: dict = {}
    # Last completed bar
    if bars is not None and getattr(bars, "ts", None) is not None:
        try:
            import numpy as np
            arr = np.asarray(bars.ts)
            if len(arr):
                last = arr[-1]
                if isinstance(last, _dt.datetime):
                    py = last
                elif hasattr(last, "isoformat") and not isinstance(
                        last, (np.datetime64, np.generic)):
                    ts_s = last.isoformat()
                    py = None
                    try:
                        _parse_ts(ts_s)
                        out["bars"] = ts_s
                    except CanonicalSnapshotError:
                        py = None
                else:
                    # numpy datetime64 → python datetime
                    try:
                        converted = np.asarray(last).astype(
                            "datetime64[us]").astype(object)
                        py = converted.item() if hasattr(
                            converted, "item") else converted
                    except Exception:
                        py = None
                    if not isinstance(py, _dt.datetime):
                        py = None
                if isinstance(py, _dt.datetime):
                    if py.tzinfo is None:
                        try:
                            from zoneinfo import ZoneInfo
                            py = py.replace(tzinfo=ZoneInfo("America/New_York"))
                        except Exception:
                            py = py.replace(tzinfo=_dt.timezone.utc)
                    try:
                        s = py.isoformat()
                        _parse_ts(s)
                        out["bars"] = s
                    except CanonicalSnapshotError:
                        pass
        except Exception:
            pass

    # Chain quote timestamp if the feed attached one
    for attr in ("as_of", "quote_ts", "fetched_at", "ts", "timestamp"):
        val = getattr(chain, attr, None) if chain is not None else None
        if val is None and isinstance(chain, dict):
            val = chain.get(attr)
        if val is None:
            continue
        try:
            if hasattr(val, "isoformat"):
                s = val.isoformat()
            else:
                s = str(val)
            _parse_ts(s)
            out["chain"] = s
            break
        except Exception:
            continue

    # GEX / structural ages from market or signals when present
    for key, dest in (
        ("gex_as_of", "gex"),
        ("gex_ts", "gex"),
        ("feed_as_of", "feed"),
    ):
        val = None
        if market is not None:
            val = getattr(market, key, None)
        if val is None and signals:
            val = signals.get(key)
        if val is None:
            continue
        try:
            s = val.isoformat() if hasattr(val, "isoformat") else str(val)
            _parse_ts(s)
            out[dest] = s
        except Exception:
            continue

    return out


def build_canonical_snapshot(
    *,
    symbol: str,
    ts: str,
    session_date: str,
    market: Any = None,
    bars: Any = None,
    chain: Any = None,
    raw_features: Optional[Mapping[str, Any]] = None,
    standardized_features: Optional[Mapping[str, Any]] = None,
    missingness: Optional[Mapping[str, bool]] = None,
    source_timestamps: Optional[Mapping[str, Any]] = None,
    source_ages_seconds: Optional[Mapping[str, Any]] = None,
    quality: Optional[Mapping[str, Any]] = None,
    structural_sources: Optional[Mapping] = None,
    structural_state: Any = None,
    feature_version: str = FEATURE_VERSION,
    structural_state_version: Optional[str] = None,
    snapshot_id: Optional[str] = None,
    source_seq: int = 0,
) -> CanonicalSnapshot:
    """
    Build one immutable CanonicalSnapshot.

    Missing values remain missing (None). Missingness is recorded explicitly.
    Never replaces missing structural values with zero.
    """
    raw = dict(raw_features or {})
    std = dict(standardized_features or {})
    _assert_no_policy_fields(raw)
    _assert_no_policy_fields(std)

    src_ts = dict(source_timestamps or {})
    _reject_future_sources(ts, src_ts)
    ages = dict(source_ages_seconds or {})
    if not ages and src_ts:
        ages = compute_source_ages_seconds(ts, src_ts)

    miss = dict(missingness or {})
    # Derive missingness for raw features when not provided.
    for k, v in raw.items():
        if k not in miss:
            miss[k] = v is None

    if snapshot_id:
        sid = snapshot_id
    else:
        ts_obj = _parse_ts(ts)
        if ts_obj is None:
            raise CanonicalSnapshotError(
                f"cannot build snapshot_id without valid ts: {ts!r}")
        sid = make_snapshot_id(symbol, ts_obj, feature_version, source_seq)

    ss_version = structural_state_version
    if ss_version is None and structural_state is not None:
        ss_version = getattr(structural_state, "version", None) or getattr(
            structural_state, "structural_state_version", None)

    return CanonicalSnapshot(
        snapshot_id=str(sid),
        symbol=str(symbol),
        ts=str(ts),
        session_date=str(session_date),
        market=market,
        bars=bars,
        chain=chain,
        raw_features=raw,
        standardized_features=std,
        missingness=miss,
        source_timestamps=src_ts,
        source_ages_seconds=ages,
        quality=dict(quality or {}),
        structural_sources=dict(structural_sources or {}),
        structural_state=structural_state,
        feature_version=str(feature_version),
        structural_state_version=ss_version,
        snapshot_schema_version=SNAPSHOT_SCHEMA_VERSION,
    )
