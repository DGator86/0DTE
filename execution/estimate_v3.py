"""
execution/estimate_v3.py
========================
ExecutionEstimateV3 — fill probability, concession, fees, expected order value
(docs/PREDICTION_ENGINE_V3_PART3_DECISION_DEPLOYMENT.md §16–§17).

Shadow / research diagnostics only. Missing empirical models must not silently
fall back to midpoint-as-executable without recording fallback_level.

NOT financial advice.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

from execution_cost import (
    ExecutionCostConfig, entry_fees, exit_fees, fill_credit, np_clip,
)
from prediction.models.fill import fill_fraction_for

EXECUTION_ESTIMATE_V3_VERSION = "v3.0.0"


@dataclass(frozen=True)
class ExecutionEstimateV3:
    mid_credit: float
    natural_credit: float
    p_fill: float
    expected_fill_fraction: float
    conservative_fill_fraction: float
    expected_credit: float
    conservative_credit: float
    entry_fees: float
    expected_exit_fees: float
    expected_exit_slippage: float
    expected_stop_slippage: float
    expected_round_trip_cost: float
    conservative_round_trip_cost: float
    fill_uncertainty: float
    empirical_weight: float
    fallback_level: str
    model_versions: dict
    diagnostics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def expected_order_value(
    p_fill: float,
    expected_net_pnl_given_fill: float,
    *,
    opportunity_cost_unfilled: float = 0.0,
) -> float:
    """
    Production-facing opportunity cost defaults to 0.0 (§16.3).
    """
    p = float(np_clip(p_fill, 0.0, 1.0))
    return p * float(expected_net_pnl_given_fill) - (1.0 - p) * float(
        opportunity_cost_unfilled)


def build_execution_estimate_v3(
    *,
    mid_credit: float,
    natural_credit: float,
    family: str,
    n_legs: int,
    p_fill: Optional[float] = None,
    expected_fill_fraction: Optional[float] = None,
    conservative_fill_fraction: Optional[float] = None,
    fill_uncertainty: Optional[float] = None,
    empirical_weight: float = 0.0,
    fallback_level: Optional[str] = None,
    model_versions: Optional[dict] = None,
    quote_age_seconds: Optional[float] = None,
    minutes_to_close: Optional[float] = None,
    relative_spread: Optional[float] = None,
    realized_vol: Optional[float] = None,
    cfg: Optional[ExecutionCostConfig] = None,
    require_empirical: bool = False,
) -> ExecutionEstimateV3:
    """
    Compose V3 execution panel. If empirical fill inputs are missing, use the
    deterministic prior and set fallback_level accordingly — never pretend the
    midpoint is executable.
    """
    cfg = cfg or ExecutionCostConfig()
    mid = float(mid_credit)
    nat = float(natural_credit)
    if nat > mid + 1e-9:
        nat = mid

    versions = dict(model_versions or {})
    used_prior = False
    if expected_fill_fraction is None or conservative_fill_fraction is None:
        if require_empirical:
            raise RuntimeError(
                "empirical fill model required but missing; refusing midpoint")
        frac_exp, fill_diag = fill_fraction_for(
            family, n_legs=n_legs, quote_age_seconds=quote_age_seconds,
            minutes_to_close=minutes_to_close, relative_spread=relative_spread,
            realized_vol=realized_vol, cfg=cfg.fill)
        frac_con = np_clip(frac_exp + cfg.conservative_fill_boost, 0.0, 1.0)
        expected_fill_fraction = float(frac_exp)
        conservative_fill_fraction = float(frac_con)
        used_prior = True
        versions.setdefault("fill_prior", "v2-deterministic")
        prior_diag = fill_diag
    else:
        prior_diag = {}

    if p_fill is None:
        # Without an empirical probability model, do not assume fill=1.
        # Use a conservative prior tied to fill fraction quality.
        p_fill = float(np_clip(1.0 - 0.5 * float(expected_fill_fraction), 0.05, 0.95))
        used_prior = True
        versions.setdefault("fill_probability", "prior-proxy")

    if fill_uncertainty is None:
        fill_uncertainty = float(np_clip(
            0.5 * float(expected_fill_fraction)
            + 0.5 * (1.0 - float(p_fill)), 0.0, 1.0))

    level = fallback_level or (
        "deterministic_prior" if used_prior else "empirical")

    exp_frac = float(np_clip(expected_fill_fraction, 0.0, 1.0))
    con_frac = float(np_clip(conservative_fill_fraction, 0.0, 1.0))
    if con_frac < exp_frac:
        con_frac = exp_frac

    exp_c = fill_credit(mid, nat, exp_frac)
    con_c = fill_credit(mid, nat, con_frac)
    # Monotonicity: credit never better than mid; conservative never better
    # than expected (for credits: lower credit is worse).
    if mid >= nat:  # credit structure
        exp_c = min(exp_c, mid)
        con_c = min(con_c, exp_c)
    else:  # debit (mid < nat in magnitude sense with signed credits)
        exp_c = max(exp_c, mid) if mid < 0 else min(exp_c, mid)
        # For debit mid=-1, nat=-1.2: fill moves toward nat (more negative).
        # expected should not be cheaper than mid (exp_c <= mid for credits;
        # for debits paying more means exp_c <= mid numerically when both neg).
        if mid <= 0 and nat <= mid:
            exp_c = min(exp_c, mid)
            con_c = min(con_c, exp_c)

    e_fees = entry_fees(n_legs, cfg)
    x_fees = exit_fees(n_legs, cfg)
    hs = abs(mid - nat)
    exit_frac = np_clip(exp_frac + cfg.exit_fill_boost, 0.0, 1.0)
    stop_frac = np_clip(exp_frac + cfg.stop_exit_fill_boost, 0.0, 1.0)
    exit_slip = exit_frac * hs
    stop_slip = stop_frac * hs
    rt_exp = abs(mid - exp_c) + e_fees + x_fees + exit_slip
    rt_con = abs(mid - con_c) + e_fees + x_fees + stop_slip

    return ExecutionEstimateV3(
        mid_credit=mid,
        natural_credit=nat,
        p_fill=float(np_clip(p_fill, 0.0, 1.0)),
        expected_fill_fraction=exp_frac,
        conservative_fill_fraction=con_frac,
        expected_credit=float(exp_c),
        conservative_credit=float(con_c),
        entry_fees=float(e_fees),
        expected_exit_fees=float(x_fees),
        expected_exit_slippage=float(exit_slip),
        expected_stop_slippage=float(stop_slip),
        expected_round_trip_cost=float(rt_exp),
        conservative_round_trip_cost=float(rt_con),
        fill_uncertainty=float(fill_uncertainty),
        empirical_weight=float(empirical_weight),
        fallback_level=level,
        model_versions=versions,
        diagnostics={
            "version": EXECUTION_ESTIMATE_V3_VERSION,
            "prior_diagnostics": prior_diag,
            "used_prior": used_prior,
        },
    )
