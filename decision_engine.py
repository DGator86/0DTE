"""
decision_engine.py
==================
The pure composition layer. Given a MarketSnapshot + ChainSnapshot, run:

    RND extractor -> EdgeReport -> SpreadSelector   (what to trade)
                                -> GateScorer        (whether to trade at all)
    -> TradeDecision

No I/O, no clock, no DB -- entirely a function of its inputs, so it is
backtestable and unit-testable in isolation. The orchestrator handles feed,
timing, and journaling.

Key design choices:
  * The selector and gate run INDEPENDENTLY. The final decision is their AND,
    but BOTH results are always captured. A no-trade still records the would-be
    candidate (the selector's pick, ignoring the gate) so settlement can score
    its hypothetical P&L -- the only way to measure whether the gate helped.
  * physical_pdf is injected once and passed to BOTH compute_edge and
    select_spreads, enforcing a single source of truth for the edge measure.

NOT financial advice.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from gate_scorer import MarketSnapshot, GateConfig, Decision, evaluate as gate_evaluate
from rnd_extractor import ChainSnapshot, RNDConfig, extract_rnd, compute_edge
from spread_selector import (
    GammaContext, SelectorConfig, select_spreads, SpreadCandidate,
    STRUCTURE_TO_FAMILIES,
)


@dataclass
class TradeDecision:
    # context
    session_date: str
    ts: str
    spot: float
    net_gex: float
    gex_regime: str
    gex_pct_rank: float
    zero_gamma_dist: float
    zero_gamma_dist_pct: float
    adx: float
    call_wall: float
    put_wall: float

    # selector / would-be candidate (present even on no-trade)
    candidate: Optional[SpreadCandidate]

    # gate
    gate_pass: bool
    gate_score: float
    gate_failed: list

    # outcome
    decision: str                 # "TRADE" | "NO_TRADE"
    no_trade_reason: str
    edge_richness: float
    direction: str = ""           # Track B direction: call|put|both|none

    def as_row(self) -> dict:
        c = self.candidate
        legs = [{"strike": lg.strike, "kind": lg.kind, "qty": lg.qty}
                for lg in (c.legs if c else ())]
        return {
            "session_date": self.session_date, "ts": self.ts, "spot": self.spot,
            "net_gex": self.net_gex, "gex_regime": self.gex_regime,
            "gex_pct_rank": self.gex_pct_rank,
            "zero_gamma_dist": self.zero_gamma_dist,
            "zero_gamma_dist_pct": self.zero_gamma_dist_pct, "adx": self.adx,
            "call_wall": self.call_wall, "put_wall": self.put_wall,
            "selected_family": c.family if c else None,
            "short_strikes": json.dumps(c.short_strikes) if c else None,
            "long_strikes": json.dumps(c.long_strikes) if c else None,
            "legs_json": json.dumps(legs) if c else None,
            "credit": c.credit if c else None,
            "candidate_score": c.score if c else None,
            "ev": c.ev if c else None,
            "max_loss": c.max_loss if c else None,
            "ev_per_risk": c.ev_per_risk if c else None,
            "theta": c.theta if c else None,
            "gamma": c.gamma if c else None,
            "prob_profit": c.prob_profit if c else None,
            "prob_touch_short": c.prob_touch_short if c else None,
            "liquidity_score": c.liquidity_score if c else None,
            "wall_safety": c.wall_safety if c else None,
            "gamma_safety": c.gamma_safety if c else None,
            "touch_safety": c.touch_safety if c else None,
            "gate_pass": 1 if self.gate_pass else 0,
            "gate_score": self.gate_score,
            "gate_failed": json.dumps(self.gate_failed),
            "veto_reasons": json.dumps(list(c.veto_reasons)) if c else json.dumps([]),
            "decision": self.decision,
            "no_trade_reason": self.no_trade_reason,
            "was_traded": 1 if self.decision == "TRADE" else 0,
            "candidate_present": 1 if c else 0,
            "regime_direction": self.direction,
        }


@dataclass
class EngineConfig:
    rnd: RNDConfig = field(default_factory=RNDConfig)
    selector: SelectorConfig = field(default_factory=SelectorConfig)
    gate: GateConfig = field(default_factory=GateConfig)


def decide(
    market: MarketSnapshot,
    chain: ChainSnapshot,
    cfg: Optional[EngineConfig] = None,
    physical_pdf: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    target_structure: Optional[str] = None,
    direction: str = "",
) -> TradeDecision:
    cfg = cfg or EngineConfig()
    session_date = market.now.astimezone().date().isoformat()
    ts = market.now.isoformat()
    regime = "long" if market.net_gex > 0 else ("short" if market.net_gex < 0 else "flat")
    zg = market.spot - market.gamma_flip

    base = dict(
        session_date=session_date, ts=ts, spot=market.spot,
        net_gex=market.net_gex, gex_regime=regime, gex_pct_rank=market.gex_pct_rank,
        zero_gamma_dist=zg, zero_gamma_dist_pct=zg / market.spot, adx=market.adx,
        call_wall=market.call_wall, put_wall=market.put_wall,
    )

    # --- gate (independent of selector) ---
    gate = gate_evaluate(market, cfg.gate)
    gate_pass = gate.decision is Decision.GO

    # --- selector: needs a usable chain ---
    candidate = None
    selector_reason = ""
    edge_rich = float("nan")
    try:
        rnd = extract_rnd(chain, cfg.rnd)
        edge = compute_edge(rnd, chain, cfg.rnd, physical_pdf=physical_pdf)
        edge_rich = edge.richness_signal
        ctx = GammaContext.from_market_snapshot(market)
        fams = STRUCTURE_TO_FAMILIES.get(target_structure) if target_structure else None
        sel = select_spreads(chain, rnd, edge, ctx, cfg.selector, physical_pdf=physical_pdf,
                             target_families=fams)
        candidate = sel.best
        if candidate is None:
            # keep the top-by-score would-be candidate for diagnostics if any exist
            if sel.ranked:
                candidate = max(sel.ranked, key=lambda c: c.score)
            selector_reason = sel.no_trade_reason or "no candidate"
    except Exception as e:  # thin/!arbitrage-free chain etc.
        selector_reason = f"chain_unavailable: {e}"

    # A genuinely tradable candidate is one that passed the SELECTOR's own vetoes.
    tradable = candidate is not None and candidate.passes_vetoes

    # --- compose ---
    if gate_pass and tradable:
        decision, reason = "TRADE", ""
    else:
        decision = "NO_TRADE"
        parts = []
        if not gate_pass:
            parts.append("gate:" + ",".join(g.split(":")[0] for g in gate.failed_gates))
        if not tradable:
            parts.append("selector:" + (selector_reason or "vetoed"))
        reason = " | ".join(parts)

    return TradeDecision(
        **base,
        candidate=candidate,
        gate_pass=gate_pass, gate_score=gate.score, gate_failed=gate.failed_gates,
        decision=decision, no_trade_reason=reason, edge_richness=edge_rich,
        direction=direction,
    )
