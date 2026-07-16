from __future__ import annotations

"""Premium-selling structure support for the Grok paper trader.

This module extends the Grok firewall without changing the deterministic
Legacy/V2/V3 engines.  The public Grok mandate is exactly:

- bull put credit spread (``put_credit``)
- bear call credit spread (``call_credit``)
- iron condor (``iron_condor``)
- iron butterfly (``iron_fly``)
- broken-wing butterfly (``broken_wing``)

All structures remain same-day, paper-only, and defined-risk.  The existing
firewall still reprices every leg from the current chain and computes the full
payoff curve before approving an intent.
"""

from typing import Any

from spread_selector import Leg


PUBLIC_FAMILIES = (
    "put_credit",
    "call_credit",
    "iron_condor",
    "iron_fly",
    "broken_wing",
)


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

        quantity = int(row.get("quantity") or 1)
        if quantity not in {1, 2}:
            raise ValueError("leg quantity must be 1 or 2")
        side = str(row.get("side") or "").lower()
        if side:
            if side not in {"buy", "sell"}:
                raise ValueError("leg side must be buy or sell")
            qty = quantity if side == "buy" else -quantity
        else:
            qty = int(row.get("qty") or 0)
        if qty not in {-2, -1, 1, 2}:
            raise ValueError("leg quantity must be one or two long/short units")
        legs.append(Leg(strike=float(row["strike"]), kind=kind, qty=qty))
    return tuple(legs)


def _vertical_shape(kind: str, legs: tuple[Leg, ...], *, bullish: bool) -> bool:
    if len(legs) != 2 or any(leg.kind != kind or abs(leg.qty) != 1 for leg in legs):
        return False
    long_legs = [leg for leg in legs if leg.qty == 1]
    short_legs = [leg for leg in legs if leg.qty == -1]
    if len(long_legs) != 1 or len(short_legs) != 1:
        return False
    long_strike, short_strike = long_legs[0].strike, short_legs[0].strike
    return short_strike > long_strike if bullish else short_strike < long_strike


def _iron_shape(legs: tuple[Leg, ...], *, fly: bool) -> bool:
    if len(legs) != 4 or any(abs(leg.qty) != 1 for leg in legs):
        return False
    p_longs = [leg.strike for leg in legs if leg.kind == "P" and leg.qty == 1]
    p_shorts = [leg.strike for leg in legs if leg.kind == "P" and leg.qty == -1]
    c_shorts = [leg.strike for leg in legs if leg.kind == "C" and leg.qty == -1]
    c_longs = [leg.strike for leg in legs if leg.kind == "C" and leg.qty == 1]
    if not all(len(xs) == 1 for xs in (p_longs, p_shorts, c_shorts, c_longs)):
        return False
    p_long, p_short = p_longs[0], p_shorts[0]
    c_short, c_long = c_shorts[0], c_longs[0]
    if fly:
        return p_long < p_short == c_short < c_long
    return p_long < p_short < c_short < c_long


def _broken_wing_shape(legs: tuple[Leg, ...]) -> bool:
    if len(legs) != 3 or len({leg.kind for leg in legs}) != 1:
        return False
    ordered = sorted(legs, key=lambda leg: leg.strike)
    if [leg.qty for leg in ordered] != [1, -2, 1]:
        return False
    lower_width = ordered[1].strike - ordered[0].strike
    upper_width = ordered[2].strike - ordered[1].strike
    return lower_width > 0 and upper_width > 0 and abs(lower_width - upper_width) > 1e-9


def _family_shape(family: str, legs: tuple[Leg, ...]) -> bool:
    if family == "put_credit":
        return _vertical_shape("P", legs, bullish=True) or _broken_wing_shape(legs)
    if family == "call_credit":
        return _vertical_shape("C", legs, bullish=False) or _broken_wing_shape(legs)
    if family == "iron_condor":
        return _iron_shape(legs, fly=False) or _iron_shape(legs, fly=True)
    return False


def install_premium_structure_support() -> None:
    """Patch the Grok-only adapter to expose the five approved structures.

    The existing RiskFirewall remains responsible for all pricing, liquidity,
    payoff, risk, time-window, account, and paper-only checks.  This adapter only
    canonicalizes the two additional multi-leg shapes so they pass through that
    same audited validation path.
    """
    from . import agent as agent_module
    from . import risk as risk_module

    risk_module._normalized_legs = _normalized_legs
    risk_module._validate_family_shape = _family_shape

    original_validate = risk_module.RiskFirewall.validate_entry
    if not getattr(original_validate, "_premium_structure_wrapper", False):
        def validate_entry(self, *, now, plan, result, broker, allow_new_entry):
            incoming = dict(plan)
            original_family = str(incoming.get("family") or "").lower()
            canonical = original_family
            if original_family == "iron_fly":
                canonical = "iron_condor"
            elif original_family == "broken_wing":
                try:
                    normalized = _normalized_legs(incoming.get("legs"))
                except (KeyError, TypeError, ValueError):
                    normalized = ()
                kinds = {leg.kind for leg in normalized}
                canonical = "put_credit" if kinds == {"P"} else "call_credit"
            incoming["family"] = canonical
            outcome = original_validate(
                self,
                now=now,
                plan=incoming,
                result=result,
                broker=broker,
                allow_new_entry=allow_new_entry,
            )
            if outcome.approved and outcome.paper_intent is not None:
                candidate = outcome.paper_intent.get("candidate")
                if candidate is not None:
                    candidate.family = original_family
                outcome.paper_intent["structure"] = original_family
                grok_plan = outcome.paper_intent.setdefault("grok_plan", {})
                grok_plan["family"] = original_family
            return outcome

        validate_entry._premium_structure_wrapper = True
        risk_module.RiskFirewall.validate_entry = validate_entry

    # Keep the model-facing tool schema synchronized with the firewall mandate.
    for tool in agent_module.TOOLS:
        if tool.get("name") != "submit_paper_trade":
            continue
        props = tool["parameters"]["properties"]
        props["family"]["enum"] = list(PUBLIC_FAMILIES)
        props["legs"]["items"]["properties"]["quantity"] = {
            "type": "integer",
            "enum": [1, 2],
            "description": "Use 2 only for the short body of a broken-wing butterfly.",
        }
        break

    agent_module.SYSTEM_PROMPT += """

Approved premium-selling structures only:
- put_credit: bull put credit spread
- call_credit: bear call credit spread
- iron_condor: separated put and call short strikes
- iron_fly: put and call short legs share the same body strike
- broken_wing: three same-type legs ordered long 1 / short 2 / long 1 with unequal wings
Do not propose debit spreads or any other family.
"""
