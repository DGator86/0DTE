"""
decision_stack/authority.py
===========================
Authority router for deployment modes
(docs/UNIFIED_V1_V2_V3_HANDOFF.md §9.3).

Hard veto always wins. Shadow/advisory keep legacy authority.
Champion uses V3 unless fail-closed fallback applies.

NOT financial advice.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class AuthorityResult:
    authority_source: str
    final_action: str
    selected_candidate_id: Optional[str]
    final_structure: Optional[str]
    final_direction: Optional[str]
    final_size_mult: float
    fallback_used: bool
    fallback_reason: Optional[str]
    advisory_action: Optional[str] = None
    advisory_candidate_id: Optional[str] = None
    reference_account: str = "legacy"
    candidate_account: Optional[str] = None
    reasons: tuple = ()


def _action_of(decision: Any, default: str = "NO_EDGE") -> str:
    if decision is None:
        return default
    if isinstance(decision, dict):
        return str(
            decision.get("final_action")
            or decision.get("action")
            or decision.get("statistical_action")
            or default)
    return str(
        getattr(decision, "final_action", None)
        or getattr(decision, "action", None)
        or getattr(decision, "statistical_action", None)
        or default)


def _field(decision: Any, *names: str, default=None):
    if decision is None:
        return default
    for name in names:
        if isinstance(decision, dict) and name in decision:
            return decision.get(name)
        if hasattr(decision, name):
            val = getattr(decision, name)
            if val is not None:
                return val
    return default


def coerce_size_mult(value: Any, *, default: float = 1.0) -> float:
    """Preserve explicit 0.0 sizing — only substitute when value is None."""
    if value is None:
        return float(default)
    return float(value)


def resolve_authority(
    *,
    mode: str,
    legacy_decision: Any,
    v3_decision: Any,
    hard_vetoes: tuple[str, ...] = (),
    fallback_policy: str = "abstain",
    legacy_size_mult: float = 1.0,
    v3_size_mult: float = 1.0,
) -> AuthorityResult:
    """
    Route authority according to deployment mode.

    Hard veto always produces HARD_VETO regardless of V1/V3 preference.
    """
    mode = str(mode or "shadow").lower()
    vetoes = tuple(hard_vetoes or ())

    legacy_action = _action_of(legacy_decision, "NO_EDGE")
    # Normalize legacy take/skip style if present
    if legacy_action in ("take", "TAKE", "trade", "ENTER"):
        legacy_action = "TRADE"
    elif legacy_action in ("skip", "SKIP", "pass", "PASS", "no_trade"):
        legacy_action = "NO_EDGE"

    v3_stat = _action_of(v3_decision, "UNAVAILABLE")
    v3_final = str(
        _field(v3_decision, "final_action", "action", default=v3_stat)
        or v3_stat)

    leg_cand = _field(legacy_decision, "candidate_id", "selected_candidate_id")
    leg_struct = _field(legacy_decision, "structure", "family")
    leg_dir = _field(legacy_decision, "direction")
    v3_cand = _field(v3_decision, "candidate_id", "selected_candidate_id",
                     "v3_candidate_id")
    v3_struct = _field(v3_decision, "structure", "family", "v3_structure")
    v3_dir = _field(v3_decision, "direction", "v3_direction")

    if vetoes:
        return AuthorityResult(
            authority_source="legacy",
            final_action="HARD_VETO",
            selected_candidate_id=None,
            final_structure=None,
            final_direction=None,
            final_size_mult=0.0,
            fallback_used=False,
            fallback_reason=None,
            advisory_action=v3_final if mode == "advisory" else None,
            advisory_candidate_id=v3_cand if mode == "advisory" else None,
            reference_account="legacy",
            candidate_account="v3" if mode == "candidate" else None,
            reasons=("hard_veto",) + vetoes,
        )

    if mode in ("research", "shadow", "advisory"):
        return AuthorityResult(
            authority_source="legacy",
            final_action=legacy_action,
            selected_candidate_id=leg_cand,
            final_structure=leg_struct,
            final_direction=leg_dir,
            final_size_mult=float(legacy_size_mult),
            fallback_used=False,
            fallback_reason=None,
            advisory_action=v3_final if mode == "advisory" else None,
            advisory_candidate_id=v3_cand if mode == "advisory" else None,
            reference_account="legacy",
            reasons=("legacy_authority",),
        )

    if mode == "candidate":
        return AuthorityResult(
            authority_source="legacy",  # reference account
            final_action=legacy_action,
            selected_candidate_id=leg_cand,
            final_structure=leg_struct,
            final_direction=leg_dir,
            final_size_mult=float(legacy_size_mult),
            fallback_used=False,
            fallback_reason=None,
            advisory_action=v3_final,
            advisory_candidate_id=v3_cand,
            reference_account="legacy",
            candidate_account="v3",
            reasons=("dual_paper_candidate_mode",),
        )

    if mode == "champion":
        # Fail-closed fallback when V3 unavailable.
        if v3_final in ("UNAVAILABLE",) or v3_decision is None:
            if fallback_policy == "legacy":
                return AuthorityResult(
                    authority_source="legacy",
                    final_action=legacy_action,
                    selected_candidate_id=leg_cand,
                    final_structure=leg_struct,
                    final_direction=leg_dir,
                    final_size_mult=float(legacy_size_mult),
                    fallback_used=True,
                    fallback_reason="v3_unavailable_fallback_legacy",
                    reasons=("champion_fallback_legacy",),
                )
            if fallback_policy == "no_trade":
                return AuthorityResult(
                    authority_source="v3",
                    final_action="NO_EDGE",
                    selected_candidate_id=None,
                    final_structure=None,
                    final_direction=None,
                    final_size_mult=0.0,
                    fallback_used=True,
                    fallback_reason="v3_unavailable_fallback_no_trade",
                    reasons=("champion_fallback_no_trade",),
                )
            # abstain
            return AuthorityResult(
                authority_source="v3",
                final_action="ABSTAIN",
                selected_candidate_id=None,
                final_structure=None,
                final_direction=None,
                final_size_mult=0.0,
                fallback_used=True,
                fallback_reason="v3_unavailable_fallback_abstain",
                reasons=("champion_fallback_abstain",),
            )
        return AuthorityResult(
            authority_source="v3",
            final_action=v3_final,
            selected_candidate_id=v3_cand,
            final_structure=v3_struct,
            final_direction=v3_dir,
            final_size_mult=float(v3_size_mult),
            fallback_used=False,
            fallback_reason=None,
            reasons=("champion_v3_authority",),
        )

    # Unknown mode → fail closed to abstain
    return AuthorityResult(
        authority_source="legacy",
        final_action="ABSTAIN",
        selected_candidate_id=None,
        final_structure=None,
        final_direction=None,
        final_size_mult=0.0,
        fallback_used=True,
        fallback_reason=f"unknown_mode_{mode}",
        reasons=("unknown_mode",),
    )
