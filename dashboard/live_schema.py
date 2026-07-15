"""
dashboard/live_schema.py
========================
Versioned /api/live contract — live.v1 (unified handoff §12 / PR C–D).

The serializer emits this envelope. The dashboard (PR D) consumes only
documented live.v1 sections; flat top-level aliases are no longer emitted
(system.compat_flat_keys=False).

NOT financial advice.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from prediction.feed_status import (
    DEFAULT_REQUIRED_SOURCES,
    FeedStatus,
    build_feed_status,
    overall_feed_status,
)

LIVE_SCHEMA_VERSION = "live.v1"

# Top-level live.v1 sections (handoff §12).
LIVE_V1_SECTIONS: tuple[str, ...] = (
    "schema_version",
    "generated_at",
    "snapshot",
    "feeds",
    "market",
    "legacy",
    "forecast",
    "v3",
    "accounts",
    "risk",
    "paper",
    "system",
)

# Historical flat aliases (removed in PR D). Kept for inventory/docs only.
COMPAT_FLAT_KEYS: tuple[str, ...] = ()

_DEFAULT_LIMITS = {
    "spot": 5.0,
    "bars": 90.0,
    "option_chain": 15.0,
    "settlement": 86_400.0,
}


class LiveSchemaError(ValueError):
    """live.v1 payload failed structural validation."""


def feeds_payload_from_statuses(
    statuses: Mapping[str, FeedStatus] | Sequence[FeedStatus],
) -> dict:
    """Serialize FeedStatus map into the /api/live feeds section."""
    if isinstance(statuses, Mapping):
        items = {str(k): v for k, v in statuses.items()}
    else:
        items = {s.source: s for s in statuses}
    out: dict[str, Any] = {
        "overall_status": overall_feed_status(items),
    }
    for name in DEFAULT_REQUIRED_SOURCES:
        fs = items.get(name)
        if fs is None:
            fs = build_feed_status(
                source=name,
                freshness_limit_seconds=_DEFAULT_LIMITS.get(name, 30.0),
                present=False,
                required=True,
            )
        out[name] = fs.to_dict()
    # Include any extra non-required sources after the required set.
    for name, fs in sorted(items.items()):
        if name not in out:
            out[name] = fs.to_dict()
    return out


def synthesize_feed_statuses(
    *,
    feed_source: Optional[str] = None,
    chain_available: bool = False,
    ages_seconds: Optional[Mapping[str, Optional[float]]] = None,
    providers: Optional[Mapping[str, Optional[str]]] = None,
    observed_at: Optional[Mapping[str, Optional[str]]] = None,
    freshness_limits: Optional[Mapping[str, float]] = None,
    feed_statuses: Optional[Mapping[str, FeedStatus]] = None,
) -> dict[str, FeedStatus]:
    """
    Build required FeedStatus entries.

    When explicit feed_statuses are provided they win. Otherwise derive an
    honest degraded picture from feed_source / chain_available — never claim
    overall LIVE from a truthy provider name alone (age unknown ⇒ DELAYED).
    """
    if feed_statuses:
        out = {str(k): v for k, v in feed_statuses.items()}
        for name in DEFAULT_REQUIRED_SOURCES:
            out.setdefault(
                name,
                build_feed_status(
                    source=name,
                    freshness_limit_seconds=(freshness_limits or _DEFAULT_LIMITS)
                    .get(name, 30.0),
                    present=False,
                    required=True,
                ),
            )
        return out

    ages = dict(ages_seconds or {})
    prov = dict(providers or {})
    obs = dict(observed_at or {})
    limits = dict(_DEFAULT_LIMITS)
    limits.update(freshness_limits or {})
    provider_default = feed_source

    out: dict[str, FeedStatus] = {}
    for name in DEFAULT_REQUIRED_SOURCES:
        limit = float(limits.get(name, 30.0))
        age = ages.get(name)
        # Explicit age means the source was observed for this tick even if the
        # TickSnapshot.chain pointer is None (e.g. tests / partial snapshots).
        if name == "option_chain" and not chain_available and age is None:
            out[name] = build_feed_status(
                source=name,
                freshness_limit_seconds=limit,
                provider=prov.get(name, provider_default),
                present=False,
                required=True,
                error_code="chain_unavailable",
            )
            continue
        present = (
            bool(provider_default) or age is not None or name in prov
            or (name == "option_chain" and chain_available)
        )
        if name == "settlement" and not present:
            # Settlement may be absent intraday; still required for overall LIVE.
            out[name] = build_feed_status(
                source=name,
                freshness_limit_seconds=limit,
                present=False,
                required=True,
                error_code="settlement_unobserved",
            )
            continue
        out[name] = build_feed_status(
            source=name,
            freshness_limit_seconds=limit,
            provider=prov.get(name, provider_default if present else None),
            observed_at=obs.get(name),
            age_seconds=age,
            present=present,
            required=True,
            diagnostics=(
                {"note": "age_unknown_synthesized"} if age is None and present
                else {}
            ),
        )
    return out


def validate_live_v1(payload: Mapping[str, Any]) -> list[str]:
    """
    Structural validation for live.v1. Returns a list of error strings
    (empty = ok). Does not require a jsonschema dependency.
    """
    errors: list[str] = []
    if not isinstance(payload, Mapping):
        return ["payload must be a mapping"]
    if payload.get("schema_version") != LIVE_SCHEMA_VERSION:
        errors.append(
            f"schema_version must be {LIVE_SCHEMA_VERSION!r}, "
            f"got {payload.get('schema_version')!r}")
    for key in LIVE_V1_SECTIONS:
        if key not in payload:
            errors.append(f"missing section {key!r}")
    feeds = payload.get("feeds")
    if isinstance(feeds, Mapping):
        if "overall_status" not in feeds:
            errors.append("feeds.overall_status missing")
        for src in DEFAULT_REQUIRED_SOURCES:
            if src not in feeds:
                errors.append(f"feeds.{src} missing")
            else:
                block = feeds[src]
                if not isinstance(block, Mapping):
                    errors.append(f"feeds.{src} must be object")
                else:
                    for f in ("status", "required", "freshness_limit_seconds"):
                        if f not in block:
                            errors.append(f"feeds.{src}.{f} missing")
    else:
        errors.append("feeds must be an object")

    # Explicit sections must not mislabel V2 under v3.
    v3 = payload.get("v3")
    if isinstance(v3, Mapping):
        if v3.get("source_version") == "v2":
            errors.append("v3 section must not claim source_version=v2")
    forecast = payload.get("forecast")
    if isinstance(forecast, Mapping):
        # forecast may be empty/unavailable; if labeled v3 that's wrong
        if forecast.get("source_version") == "v3":
            errors.append("forecast section must not claim source_version=v3")
    return errors


def assert_live_v1(payload: Mapping[str, Any]) -> None:
    errs = validate_live_v1(payload)
    if errs:
        raise LiveSchemaError("; ".join(errs))
