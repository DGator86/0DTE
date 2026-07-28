from __future__ import annotations

"""Additional defined-risk structure support for the Grok paper trader.

The Grok mandate is not premium-selling-only. It may trade bullish, bearish,
neutral, volatility-contraction, or volatility-expansion setups, provided the
structure is options-only and has deterministically bounded maximum loss.

This module adds simple long calls/puts, iron butterflies, and broken-wing
butterflies to the original vertical-spread and iron-condor support.
"""

from typing import Any

from spread_selector import Leg


PUBLIC_FAMILIES = (
    "put_credit",
    "call_credit",
    "iron_condor",
    "iron_fly",
    "broken_wing",
    "long_call",
    "long_put",
    "long_call_spread",
    "long_put_spread",
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


def _single_long_shape(kind: str, legs: tuple[Leg, ...]) -> bool:
    return len(legs) == 1 and legs[0].kind == kind and legs[0].qty == 1


def _vertical_shape(kind: str, legs: tuple[Leg, ...], *, credit: bool) -> bool:
    if len(legs) != 2 or any(leg.kind != kind or abs(leg.qty) != 1 for leg in legs):
        return False
    longs = [leg for leg in legs if leg.qty == 1]
    shorts = [leg for leg in legs if leg.qty == -1]
    if len(longs) != 1 or len(shorts) != 1:
        return False
    long_strike, short_strike = longs[0].strike, shorts[0].strike
    if kind == "P":
        return short_strike > long_strike if credit else long_strike > short_strike
    return short_strike < long_strike if credit else long_strike < short_strike


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
    return p_long < p_short == c_short < c_long if fly else p_long < p_short < c_short < c_long


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
        return _vertical_shape("P", legs, credit=True) or _broken_wing_shape(legs)
    if family == "call_credit":
        return _vertical_shape("C", legs, credit=True) or _broken_wing_shape(legs)
    if family == "long_call":
        return _single_long_shape("C", legs)
    if family == "long_put":
        return _single_long_shape("P", legs)
    if family == "long_call_spread":
        return _vertical_shape("C", legs, credit=False)
    if family == "long_put_spread":
        return _vertical_shape("P", legs, credit=False)
    if family == "iron_condor":
        return _iron_shape(legs, fly=False) or _iron_shape(legs, fly=True)
    return False


def install_premium_structure_support() -> None:
    """Install the expanded options-only defined-risk structure adapter."""
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

    for tool in agent_module.TOOLS:
        if tool.get("name") != "submit_paper_trade":
            continue
        props = tool["parameters"]["properties"]
        props["family"]["enum"] = list(PUBLIC_FAMILIES)
        props["legs"]["minItems"] = 1
        props["legs"]["items"]["properties"]["quantity"] = {
            "type": "integer",
            "enum": [1, 2],
            "description": "Use 2 only when a validated ratio structure requires it.",
        }
        break

    agent_module.SYSTEM_PROMPT += """

Trading mandate:
- Seek opportunity in bullish, bearish, neutral, volatility-expansion, and volatility-contraction regimes.
- Simple directional long calls and long puts are explicitly permitted.
- Bullish and bearish debit spreads are permitted.
- The requested premium structures are permitted: bull put credit spread, bear call credit spread, iron condor, iron butterfly, and broken-wing butterfly.
- The controlling rules are options-only construction, no stock ownership requirement, and deterministically bounded maximum loss. Never propose covered-stock, naked-unlimited-risk, or undefined-risk exposure.
"""
