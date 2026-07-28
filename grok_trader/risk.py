from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np

from spread_selector import Leg, SpreadCandidate, _chain_maps, _credit, _payoff_curve

from .config import GrokConfig

ET = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class FirewallResult:
    approved: bool
    reasons: tuple[str, ...]
    paper_intent: dict | None = None
    diagnostics: dict | None = None


FAMILY_RULES: dict[str, tuple[tuple[str, int], ...]] = {
    "put_credit": (("P", -1), ("P", 1)),
    "call_credit": (("C", -1), ("C", 1)),
    "long_call_spread": (("C", 1), ("C", -1)),
    "long_put_spread": (("P", 1), ("P", -1)),
    "iron_condor": (("P", 1), ("P", -1), ("C", -1), ("C", 1)),
}


def _parse_hhmm(value: Any) -> tuple[int, int]:
    text = str(value or "").strip()
    parts = text.split(":")
    if len(parts) != 2:
        raise ValueError("mandatory_exit_time must be HH:MM")
    hour, minute = int(parts[0]), int(parts[1])
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError("mandatory_exit_time must be a valid HH:MM time")
    return hour, minute


def _normalized_legs(raw: Any) -> tuple[Leg, ...]:
    if not isinstance(raw, list):
        raise ValueError("legs must be a list")
    legs: list[Leg] = []
    for row in raw:
        if not isinstance(row, dict):
            raise ValueError("each leg must be an object")
        kind = str(row.get("kind") or row.get("option_type") or "").upper()
        if kind in {"CALL", "C"}:
            kind = "C"
        elif kind in {"PUT", "P"}:
            kind = "P"
        else:
            raise ValueError("leg kind must be call/C or put/P")
        side = str(row.get("side") or "").lower()
        qty_raw = row.get("qty")
        if side:
            qty = 1 if side == "buy" else -1 if side == "sell" else 0
        else:
            qty = int(qty_raw or 0)
        if qty not in {-1, 1}:
            raise ValueError("every leg must be one long or one short contract unit")
        strike = float(row["strike"])
        legs.append(Leg(strike=strike, kind=kind, qty=qty))
    return tuple(legs)


def _validate_family_shape(family: str, legs: tuple[Leg, ...]) -> bool:
    expected = FAMILY_RULES.get(family)
    if expected is None or len(expected) != len(legs):
        return False
    actual = tuple((leg.kind, leg.qty) for leg in sorted(legs, key=lambda x: (x.kind, x.strike)))
    expected_sorted = tuple(sorted(expected))
    if tuple(sorted(actual)) != expected_sorted:
        return False
    strikes = [leg.strike for leg in legs]
    if len(set(strikes)) != len(strikes):
        return False
    by = {(leg.kind, leg.qty): leg.strike for leg in legs}
    if family == "put_credit":
        return by[("P", -1)] > by[("P", 1)]
    if family == "call_credit":
        return by[("C", -1)] < by[("C", 1)]
    if family == "long_call_spread":
        return by[("C", 1)] < by[("C", -1)]
    if family == "long_put_spread":
        return by[("P", 1)] > by[("P", -1)]
    if family == "iron_condor":
        p_long = by[("P", 1)]
        p_short = by[("P", -1)]
        c_short = by[("C", -1)]
        c_long = by[("C", 1)]
        return p_long < p_short < c_short < c_long
    return False


class RiskFirewall:
    def __init__(self, cfg: GrokConfig) -> None:
        self.cfg = cfg

    def validate_entry(self, *, now: dt.datetime, plan: dict, result: Any, broker: Any,
                       allow_new_entry: bool) -> FirewallResult:
        reasons: list[str] = []
        if not self.cfg.paper_only:
            reasons.append("paper_only_disabled")
        if not self.cfg.submission_enabled:
            reasons.append("paper_submission_disabled")
        if not allow_new_entry:
            reasons.append("budget_soft_cap_entries_disabled")
        symbol = str(plan.get("symbol") or "").upper()
        if symbol not in self.cfg.allowed_symbols or symbol != str(getattr(broker, "symbol", "")).upper():
            reasons.append("symbol_not_allowed")
        family = str(plan.get("family") or "").lower()
        if family not in self.cfg.allowed_families:
            reasons.append("family_not_allowed")

        et = now.astimezone(ET)
        current_hm = (et.hour, et.minute)
        if current_hm < (self.cfg.entry_start_hour, self.cfg.entry_start_minute):
            reasons.append("before_entry_window")
        if current_hm >= (self.cfg.entry_cutoff_hour, self.cfg.entry_cutoff_minute):
            reasons.append("after_entry_cutoff")
        expiration = str(plan.get("expiration") or "").lower()
        if expiration not in {"0dte", "same_day", et.date().isoformat()}:
            reasons.append("expiration_must_be_same_day")
        mandatory_exit: tuple[int, int] | None = None
        try:
            mandatory_exit = _parse_hhmm(plan.get("mandatory_exit_time"))
        except (TypeError, ValueError) as exc:
            reasons.append(f"invalid_mandatory_exit:{exc}")
        else:
            broker_eod = tuple(getattr(broker.cfg, "eod_close_et", (15, 55)))
            if mandatory_exit <= current_hm:
                reasons.append("mandatory_exit_must_be_future")
            if mandatory_exit > broker_eod:
                reasons.append("mandatory_exit_after_broker_eod")

        snap = getattr(result, "snapshot", None)
        chain = getattr(snap, "chain", None) if snap is not None else None
        if chain is None:
            reasons.append("chain_unavailable")
            return FirewallResult(False, tuple(reasons))
        market = getattr(snap, "market", None)
        quote_age = getattr(market, "quote_age_seconds", None)
        if quote_age is not None and float(quote_age) > self.cfg.max_quote_age_seconds:
            reasons.append("quotes_stale")
        signals = getattr(result, "signals", None) or {}
        if float(signals.get("session_warmup") or 0.0) >= 1.0:
            reasons.append("session_warmup")
        if getattr(broker, "_open_count")("grok") >= getattr(broker.cfg, "max_open_positions", 1):
            reasons.append("grok_position_already_open")

        try:
            legs = _normalized_legs(plan.get("legs"))
        except (KeyError, TypeError, ValueError) as exc:
            reasons.append(f"invalid_legs:{exc}")
            return FirewallResult(False, tuple(reasons))
        if not _validate_family_shape(family, legs):
            reasons.append("legs_do_not_match_defined_risk_family")

        cmid, pmid, spreads = _chain_maps(chain)
        available_strikes = set(cmid) | set(pmid)
        if any(leg.strike not in available_strikes for leg in legs):
            reasons.append("strike_not_in_current_chain")
        for leg in legs:
            mids = cmid if leg.kind == "C" else pmid
            pair = spreads.get(leg.strike)
            mid = mids.get(leg.strike)
            spread = pair[0 if leg.kind == "C" else 1] if pair else None
            if mid is None or spread is None:
                reasons.append("missing_leg_quote")
                continue
            if spread > self.cfg.max_leg_spread_abs:
                reasons.append(f"leg_spread_too_wide:{leg.strike:g}{leg.kind}")
            if mid > 0 and spread / mid > self.cfg.max_leg_relative_spread:
                reasons.append(f"leg_relative_spread_too_wide:{leg.strike:g}{leg.kind}")

        credit = _credit(legs, cmid, pmid)
        if credit is None:
            reasons.append("cannot_price_structure")
            return FirewallResult(False, tuple(reasons))
        is_credit = family in {"put_credit", "call_credit", "iron_condor"}
        limit_price = float(plan.get("limit_price") or 0.0)
        if limit_price <= 0:
            reasons.append("limit_price_must_be_positive")
        elif is_credit and credit + 1e-9 < limit_price:
            reasons.append("current_credit_below_minimum_limit")
        elif not is_credit and -credit - 1e-9 > limit_price:
            reasons.append("current_debit_above_maximum_limit")
        if is_credit and credit <= 0:
            reasons.append("credit_family_has_nonpositive_credit")
        if not is_credit and credit >= 0:
            reasons.append("debit_family_has_nonpositive_debit")

        lo, hi = max(float(chain.spot) * 0.75, 1.0), float(chain.spot) * 1.25
        grid = np.linspace(lo, hi, 5001)
        payoff = _payoff_curve(legs, grid, float(credit))
        max_profit = float(np.max(payoff))
        max_loss = float(-np.min(payoff))
        if not np.isfinite(max_loss) or max_loss <= 0:
            reasons.append("undefined_or_zero_max_loss")

        requested_risk = float(plan.get("risk_fraction") or self.cfg.max_risk_fraction)
        if requested_risk <= 0 or requested_risk > self.cfg.max_requested_risk_fraction:
            reasons.append("risk_fraction_out_of_bounds")
        requested_risk = min(max(requested_risk, 0.0), self.cfg.max_risk_fraction)
        broker_base_risk = float(getattr(broker.cfg, "risk_per_trade_frac", 0.0) or 0.0)
        if broker_base_risk <= 0:
            reasons.append("broker_risk_budget_disabled")
            size_mult = 0.0
        else:
            size_mult = min(1.0, requested_risk / broker_base_risk)

        if reasons:
            return FirewallResult(False, tuple(dict.fromkeys(reasons)), diagnostics={
                "mid_credit": credit,
                "max_profit_per_share": max_profit,
                "max_loss_per_share": max_loss,
            })

        short_strikes = tuple(sorted(leg.strike for leg in legs if leg.qty < 0))
        long_strikes = tuple(sorted(leg.strike for leg in legs if leg.qty > 0))
        confidence = max(0.0, min(float(plan.get("confidence") or 0.0), 1.0))
        candidate = SpreadCandidate(
            family=family,
            short_strikes=short_strikes,
            long_strikes=long_strikes,
            legs=legs,
            credit=float(credit),
            max_loss=max_loss,
            ev=0.0,
            ev_per_risk=0.0,
            theta=0.0,
            gamma=0.0,
            prob_profit=0.0,
            prob_touch_short=0.0,
            distance_to_wall=0.0,
            liquidity_score=1.0,
            wall_safety=1.0,
            gamma_safety=1.0,
            touch_safety=1.0,
            score=confidence * 100.0,
            passes_vetoes=True,
            veto_reasons=(),
        )
        direction = str(plan.get("direction") or "none").lower()
        paper_intent = {
            "track": "grok",
            "candidate": candidate,
            "size_mult": size_mult,
            "gate_kelly": 1.0,
            "gate_score": confidence * 100.0,
            "structure": family,
            "direction": direction,
            "reason": str(plan.get("thesis") or "grok_4_5_trade")[:2000],
            "grok_plan": {
                "limit_price": limit_price,
                "risk_fraction": requested_risk,
                "confidence": confidence,
                "supporting_evidence": plan.get("supporting_evidence") or [],
                "contradictory_evidence": plan.get("contradictory_evidence") or [],
                "invalidation_conditions": plan.get("invalidation_conditions") or [],
                "mandatory_exit_time": (f"{mandatory_exit[0]:02d}:{mandatory_exit[1]:02d}"
                                        if mandatory_exit is not None else None),
            },
        }
        return FirewallResult(True, (), paper_intent=paper_intent, diagnostics={
            "mid_credit": credit,
            "max_profit_per_share": max_profit,
            "max_loss_per_share": max_loss,
            "size_mult": size_mult,
        })

    def close_grok_positions(self, *, now: dt.datetime, result: Any, broker: Any,
                             reason: str) -> tuple[bool, list[str]]:
        """Execute a risk-reducing paper close using the broker's own accounting math."""
        snap = getattr(result, "snapshot", None)
        chain = getattr(snap, "chain", None) if snap is not None else None
        if chain is None:
            return False, ["chain_unavailable"]
        cmid, pmid, spreads = _chain_maps(chain)
        targets = [p for p in list(broker.open_positions) if broker._track_of(p) == "grok"]
        if not targets:
            return False, ["no_grok_position"]
        events: list[str] = []
        for pos in targets:
            credit_now = _credit(pos.legs, cmid, pmid)
            if credit_now is None:
                return False, ["missing_exit_quote"]
            gross_ps = pos.entry_credit - credit_now
            slip_exit = broker._slippage_ps(pos.legs, spreads)
            net_ps = gross_ps - slip_exit
            pnl_dollars = net_ps * broker.cfg.multiplier * pos.contracts
            track = broker._track_of(pos)
            broker.ledgers[track] = broker.ledgers.get(track, broker.cfg.starting_cash) + pnl_dollars
            day = now.astimezone(ET).date().isoformat()
            key = f"{day}|{track}"
            broker._day_realized[key] = broker._day_realized.get(key, 0.0) + pnl_dollars
            broker.open_positions.remove(pos)
            broker.position_monitor.release(pos.id)
            broker._last_exit_at[track] = now
            broker._last_exit_reason[track] = "grok_exit"
            pos.entry_ctx["grok_exit_reason"] = reason[:1000]
            broker._record(pos, now, credit_now, net_ps, pnl_dollars, "grok_exit")
            broker._notify(
                "PAPER EXIT",
                f"[grok] {pos.family} {pos.strikes_str()} grok_exit "
                f"pnl=${pnl_dollars:+.2f} equity=${broker.ledgers[track]:.2f}",
            )
            events.append(
                f"PAPER EXIT [grok] {pos.family} grok_exit pnl=${pnl_dollars:+.2f} "
                f"equity=${broker.ledgers[track]:.2f}"
            )
        return True, events
