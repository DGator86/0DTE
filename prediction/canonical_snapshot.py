"""
prediction/canonical_snapshot.py
================================
CanonicalSnapshot — one immutable market snapshot per tick
(unified integration handoff §4.1 / §6.2 / PR B).

Constructed exactly once per tick. Shared snapshot_id across journal,
features, forecasts, candidates, decisions, and (later) the versioned API.

No authority change in this PR: contracts + builder only. Orchestrator
wiring lands in a later bounded PR.

NOT financial advice.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from prediction.dataset import FEATURE_VERSION, make_snapshot_id
from prediction.feed_status import FeedStatus, overall_feed_status

SNAPSHOT_SCHEMA_VERSION = "canonical.v1"

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
    schema_version: str
    symbol: str
    ts: str
    session_date: str
    market: Any = None
    bars: Any = None
    chain: Any = None
    raw_features: dict = field(default_factory=dict)
    standardized_features: dict = field(default_factory=dict)
    missingness: dict = field(default_factory=dict)
    feed_statuses: dict = field(default_factory=dict)  # source -> FeedStatus
    quality: dict = field(default_factory=dict)
    structural_sources: dict = field(default_factory=dict)
    structural_state: Any = None
    feature_version: str = FEATURE_VERSION
    structural_state_version: Optional[str] = None
    configuration_hash: str = ""
    # Feature-family provenance (value/missingness live above; ages here).
    source_timestamps: dict = field(default_factory=dict)
    source_ages_seconds: dict = field(default_factory=dict)

    def overall_feed_status(self) -> str:
        return overall_feed_status(self.feed_statuses)

    def snapshot_hash(self) -> str:
        """Deterministic content hash (excludes live market/bars/chain refs)."""
        feeds = {
            k: (v.to_dict() if isinstance(v, FeedStatus) else dict(v))
            for k, v in sorted(self.feed_statuses.items())
        }
        payload = {
            "snapshot_id": self.snapshot_id,
            "schema_version": self.schema_version,
            "symbol": self.symbol,
            "ts": self.ts,
            "session_date": self.session_date,
            "raw_features": self.raw_features,
            "standardized_features": self.standardized_features,
            "missingness": self.missingness,
            "feed_statuses": feeds,
            "quality": self.quality,
            "source_timestamps": self.source_timestamps,
            "source_ages_seconds": self.source_ages_seconds,
            "feature_version": self.feature_version,
            "structural_state_version": self.structural_state_version,
            "configuration_hash": self.configuration_hash,
            "structural_sources": self.structural_sources,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"),
                       default=str).encode("utf-8")
        ).hexdigest()

    def to_dict(self) -> dict:
        feeds = {
            k: (v.to_dict() if isinstance(v, FeedStatus) else dict(v))
            for k, v in self.feed_statuses.items()
        }
        return {
            "snapshot_id": self.snapshot_id,
            "schema_version": self.schema_version,
            "symbol": self.symbol,
            "ts": self.ts,
            "session_date": self.session_date,
            "raw_features": dict(self.raw_features),
            "standardized_features": dict(self.standardized_features),
            "missingness": dict(self.missingness),
            "feed_statuses": feeds,
            "overall_feed_status": self.overall_feed_status(),
            "quality": dict(self.quality),
            "structural_sources": dict(self.structural_sources),
            "feature_version": self.feature_version,
            "structural_state_version": self.structural_state_version,
            "configuration_hash": self.configuration_hash,
            "source_timestamps": dict(self.source_timestamps),
            "source_ages_seconds": dict(self.source_ages_seconds),
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


def _reject_future_sources(
    prediction_ts: str,
    source_timestamps: Mapping[str, Any],
) -> None:
    if not prediction_ts:
        return
    for name, src_ts in source_timestamps.items():
        if src_ts is None or src_ts == "":
            continue
        if str(src_ts) > str(prediction_ts):
            raise CanonicalSnapshotError(
                f"future-dated source {name!r}: {src_ts!r} > {prediction_ts!r}")


def _freeze_mapping(m: Optional[Mapping]) -> dict:
    return dict(m or {})


def _normalize_feed_statuses(
    feed_statuses: Optional[Mapping[str, Any]],
) -> dict[str, FeedStatus]:
    out: dict[str, FeedStatus] = {}
    for k, v in (feed_statuses or {}).items():
        if isinstance(v, FeedStatus):
            out[str(k)] = v
        elif isinstance(v, Mapping):
            d = dict(v)
            d.setdefault("source", k)
            out[str(k)] = FeedStatus.from_dict(d)
        else:
            raise CanonicalSnapshotError(
                f"feed_statuses[{k!r}] must be FeedStatus or mapping")
    return out


def configuration_hash_for(
    *,
    feature_version: str,
    schema_version: str = SNAPSHOT_SCHEMA_VERSION,
    structural_state_version: Optional[str] = None,
    extras: Optional[Mapping[str, Any]] = None,
) -> str:
    """Stable hash of configuration inputs that define snapshot semantics."""
    payload = {
        "feature_version": feature_version,
        "schema_version": schema_version,
        "structural_state_version": structural_state_version,
        "extras": dict(extras or {}),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"),
                   default=str).encode("utf-8")
    ).hexdigest()


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
    feed_statuses: Optional[Mapping[str, Any]] = None,
    quality: Optional[Mapping[str, Any]] = None,
    structural_sources: Optional[Mapping] = None,
    structural_state: Any = None,
    feature_version: str = FEATURE_VERSION,
    structural_state_version: Optional[str] = None,
    configuration_hash: Optional[str] = None,
    snapshot_id: Optional[str] = None,
    source_seq: int = 0,
    source_timestamps: Optional[Mapping[str, Any]] = None,
    source_ages_seconds: Optional[Mapping[str, Any]] = None,
    schema_version: str = SNAPSHOT_SCHEMA_VERSION,
) -> CanonicalSnapshot:
    """
    Build one immutable CanonicalSnapshot.

    Missing values remain missing (None). Missingness is recorded explicitly.
    Never replaces missing structural values with zero.
    """
    raw = _freeze_mapping(raw_features)
    std = _freeze_mapping(standardized_features)
    _assert_no_policy_fields(raw)
    _assert_no_policy_fields(std)

    src_ts = _freeze_mapping(source_timestamps)
    _reject_future_sources(ts, src_ts)

    miss = _freeze_mapping(missingness)
    for k, v in raw.items():
        if k not in miss:
            miss[k] = v is None

    feeds = _normalize_feed_statuses(feed_statuses)

    if snapshot_id:
        sid = snapshot_id
    else:
        import datetime as _dt
        ts_obj: Any = ts
        if isinstance(ts, str):
            try:
                ts_obj = _dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except ValueError:
                ts_obj = _dt.datetime.now(_dt.timezone.utc)
        sid = make_snapshot_id(symbol, ts_obj, feature_version, source_seq)

    ss_version = structural_state_version
    if ss_version is None and structural_state is not None:
        ss_version = getattr(structural_state, "version", None) or getattr(
            structural_state, "structural_state_version", None)

    cfg_hash = configuration_hash or configuration_hash_for(
        feature_version=feature_version,
        schema_version=schema_version,
        structural_state_version=ss_version,
    )

    return CanonicalSnapshot(
        snapshot_id=str(sid),
        schema_version=str(schema_version),
        symbol=str(symbol),
        ts=str(ts),
        session_date=str(session_date),
        market=market,
        bars=bars,
        chain=chain,
        raw_features=raw,
        standardized_features=std,
        missingness=miss,
        feed_statuses=feeds,
        quality=_freeze_mapping(quality),
        structural_sources=_freeze_mapping(structural_sources),
        structural_state=structural_state,
        feature_version=str(feature_version),
        structural_state_version=ss_version,
        configuration_hash=str(cfg_hash),
        source_timestamps=src_ts,
        source_ages_seconds=_freeze_mapping(source_ages_seconds),
    )
