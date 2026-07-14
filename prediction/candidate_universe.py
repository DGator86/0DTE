"""
prediction/candidate_universe.py
================================
CandidateUniverse — generate option candidates once per tick
(docs/UNIFIED_V1_V2_V3_HANDOFF.md §7.3).

Legacy and V3 must evaluate the identical candidate set.

NOT financial advice.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence

GENERATOR_VERSION = "v1.0.0"


@dataclass(frozen=True)
class CandidateUniverse:
    snapshot_id: str
    generated_at: str
    generator_version: str
    generator_configuration_hash: str
    candidates: tuple
    excluded_at_generation: tuple = ()
    chain_quality: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)

    def candidate_ids(self) -> tuple[str, ...]:
        return tuple(
            getattr(c, "candidate_id", None)
            or (c.get("candidate_id") if isinstance(c, dict) else None)
            or make_candidate_id(
                self.snapshot_id,
                family=getattr(c, "family", None) or (
                    c.get("family") if isinstance(c, dict) else "unknown"),
                legs=_legs_from(c),
            )
            for c in self.candidates
        )

    def to_dict(self) -> dict:
        return {
            "snapshot_id": self.snapshot_id,
            "generated_at": self.generated_at,
            "generator_version": self.generator_version,
            "generator_configuration_hash": self.generator_configuration_hash,
            "candidate_count": len(self.candidates),
            "excluded_count": len(self.excluded_at_generation),
            "candidate_ids": list(self.candidate_ids()),
            "excluded_at_generation": [
                dict(x) if isinstance(x, dict) else {"repr": repr(x)}
                for x in self.excluded_at_generation
            ],
            "chain_quality": dict(self.chain_quality),
            "diagnostics": dict(self.diagnostics),
        }


def _legs_from(candidate: Any) -> Sequence[dict]:
    if isinstance(candidate, dict):
        return list(candidate.get("legs") or [])
    legs = getattr(candidate, "legs", None) or ()
    out = []
    for leg in legs:
        if isinstance(leg, dict):
            out.append(leg)
        else:
            out.append({
                "right": getattr(leg, "right", None)
                or getattr(leg, "option_type", None),
                "side": getattr(leg, "side", None)
                or getattr(leg, "action", None),
                "qty": getattr(leg, "qty", None)
                or getattr(leg, "quantity", 1),
                "strike": getattr(leg, "strike", None),
                "expiration": getattr(leg, "expiration", None)
                or getattr(leg, "expiry", None),
            })
    return out


def make_candidate_id(
    snapshot_id: str,
    *,
    family: str,
    legs: Sequence[dict],
) -> str:
    """
    Deterministic candidate ID incorporating snapshot, family, and legs.
    """
    normalized_legs = []
    for leg in legs:
        normalized_legs.append({
            "right": str(leg.get("right") or leg.get("option_type") or ""),
            "side": str(leg.get("side") or leg.get("action") or ""),
            "qty": int(leg.get("qty") or leg.get("quantity") or 1),
            "strike": float(leg.get("strike") or 0.0),
            "expiration": str(
                leg.get("expiration") or leg.get("expiry") or ""),
        })
    payload = {
        "snapshot_id": snapshot_id,
        "family": str(family),
        "legs": normalized_legs,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"),
                   default=str).encode("utf-8")).hexdigest()
    return f"cand_{digest[:24]}"


def generator_configuration_hash(config: Optional[dict] = None) -> str:
    return hashlib.sha256(
        json.dumps(config or {}, sort_keys=True, separators=(",", ":"),
                   default=str).encode("utf-8")).hexdigest()


def build_candidate_universe(
    *,
    snapshot_id: str,
    generated_at: str,
    candidates: Iterable[Any],
    excluded_at_generation: Optional[Iterable[dict]] = None,
    chain_quality: Optional[dict] = None,
    diagnostics: Optional[dict] = None,
    generator_config: Optional[dict] = None,
    generator_version: str = GENERATOR_VERSION,
    assign_ids: bool = True,
) -> CandidateUniverse:
    """
    Freeze a candidate universe. Optionally stamp deterministic candidate_ids
    onto dict candidates or objects that accept setattr.
    """
    cfg = dict(generator_config or {})
    cfg_hash = generator_configuration_hash(cfg)
    frozen = []
    seen_ids: set[str] = set()
    for c in candidates:
        family = (
            c.get("family") if isinstance(c, dict)
            else getattr(c, "family", "unknown"))
        legs = _legs_from(c)
        cid = (
            (c.get("candidate_id") if isinstance(c, dict)
             else getattr(c, "candidate_id", None))
            or make_candidate_id(snapshot_id, family=str(family), legs=legs)
        )
        if cid in seen_ids:
            # Skip exact economic duplicates by id.
            continue
        seen_ids.add(cid)
        if assign_ids:
            if isinstance(c, dict):
                c = dict(c)
                c["candidate_id"] = cid
            else:
                try:
                    object.__setattr__(c, "candidate_id", cid)
                except (AttributeError, TypeError):
                    try:
                        setattr(c, "candidate_id", cid)
                    except Exception:
                        pass
        frozen.append(c)
    return CandidateUniverse(
        snapshot_id=str(snapshot_id),
        generated_at=str(generated_at),
        generator_version=str(generator_version),
        generator_configuration_hash=cfg_hash,
        candidates=tuple(frozen),
        excluded_at_generation=tuple(excluded_at_generation or ()),
        chain_quality=dict(chain_quality or {}),
        diagnostics=dict(diagnostics or {}),
    )
