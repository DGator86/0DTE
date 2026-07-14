"""
prediction/fill_training.py
===========================
Dataset helpers for empirical fill models (Part 3 §13–§15).

NOT financial advice.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

from execution.fill_records import FillRecord
from prediction.models.fill import fill_fraction_for


@dataclass
class FillSupportThresholds:
    minimum_sessions: int = 40
    minimum_attempts: int = 200
    minimum_fills: int = 100
    minimum_family_attempts: int = 50
    minimum_family_fills: int = 25
    prior_equivalent_support: int = 100


def empirical_weight(
    support: int,
    *,
    prior_equivalent_support: int = 100,
) -> float:
    s = max(int(support), 0)
    pe = max(int(prior_equivalent_support), 1)
    return float(s / (s + pe))


def blend_with_prior(
    empirical: float,
    prior: float,
    support: int,
    *,
    prior_equivalent_support: int = 100,
) -> tuple:
    w = empirical_weight(
        support, prior_equivalent_support=prior_equivalent_support)
    blended = w * float(empirical) + (1.0 - w) * float(prior)
    return float(blended), float(w)


def fallback_level(
    *,
    family_support: int,
    broad_family_support: int = 0,
    leg_group_support: int = 0,
    global_support: int = 0,
    thresholds: Optional[FillSupportThresholds] = None,
) -> str:
    """Exact family → broad family → leg-count → global → deterministic prior."""
    thr = thresholds or FillSupportThresholds()
    if family_support >= thr.minimum_family_fills:
        return "exact_family"
    if broad_family_support >= thr.minimum_family_fills:
        return "broad_family"
    if leg_group_support >= thr.minimum_family_attempts:
        return "leg_count_group"
    if global_support >= thr.minimum_fills:
        return "global_empirical"
    return "deterministic_prior"


def stage1_attempts(records: Sequence[FillRecord]) -> list:
    """All attempts enter Stage 1 (probability), including unfilled."""
    return list(records)


def stage2_fills(records: Sequence[FillRecord]) -> list:
    """Only valid completed fills enter Stage 2 (concession)."""
    out = []
    for r in records:
        if not r.filled:
            continue
        if r.fill_credit is None:
            continue
        if r.mid_credit_at_submit is None or r.natural_credit_at_submit is None:
            continue
        out.append(r)
    return out


def prior_fill_fraction_for_record(rec: FillRecord) -> float:
    frac, _ = fill_fraction_for(
        rec.family,
        n_legs=rec.n_legs,
        quote_age_seconds=rec.quote_age_seconds,
        minutes_to_close=rec.minutes_to_close,
        relative_spread=rec.relative_spread,
    )
    return float(frac)
