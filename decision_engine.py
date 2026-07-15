"""
decision_engine.py
==================
The pure composition layer. Given a MarketSnapshot + ChainSnapshot, run:

    RND extractor -> EdgeReport -> SpreadSelector   (what to trade)
                                -> GateScorer        (whether to trade at all)
    -> TradeDecision

Prediction Engine V2 / PR 10: structure/direction arrive from PolicyRouter
(legacy matrix or PredictionPolicy). This module does not choose regime —
it only enumerates and gates candidates for the routed family
(docs/PREDICTION_ENGINE_V2_HANDOFF.md §17).

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
    STRUCTURE_TO_FAMILIES, DEBIT_FAMILIES,
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
    gate_kelly: float = 1.0       # gate's score->Kelly fraction; sizes the fill
    # Physical-density provenance (Prediction Engine V2, PR 5). Observation-
    # only: which density priced the candidate, and its moments. Never fed
    # back into the density itself (independence rule §12.2).
    physical_density_mode: str = ""
    physical_moments: Optional[dict] = None
    # Prediction Engine V2 / PR 8: full evaluated candidate set from the
    # selector (for shadow ranking). Empty when the chain was unavailable.
    # Never used to override `candidate` / `decision` until promotion.
    all_candidates: list = field(default_factory=list)

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
            "signals_json": None,     # observation-only signals; unified_loop fills it
            # PR 6 executable economics (optional; journal stores when present)
            "execution_json": (json.dumps(c.execution)
                               if c is not None and getattr(c, "execution", None)
                               else None),
            "credit_expected": (c.execution.get("net_expected_credit")
                                if c is not None and getattr(c, "execution", None)
                                else None),
            "credit_conservative": (c.execution.get("net_conservative_credit")
                                    if c is not None and getattr(c, "execution", None)
                                    else None),
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
    physical_density_mode: str = "",
    physical_moments: Optional[dict] = None,
    pin_active: bool = False,
) -> TradeDecision:
    """
    Compose gate + selector into a TradeDecision.

    `physical_pdf` is the SINGLE source of truth for edge and candidate EV.
    Callers (unified_loop) are responsible for building it independently of
    `target_structure` / `direction` when V2 is active — those arguments
    select the gate class and the candidate family filter only; they must
    never be used to construct the density that prices the trade (§12.2).
    `pin_active` soft-exempts short-gamma / below-flip / trending on the
    premium gate and selector so credit can fill into a flip pin.
    """
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
    # The routed structure decides WHICH gate applies: premium credit families
    # face the full premium-selling gate; debit families face only the
    # universal stops plus the trend-quality score. Without this split, the
    # premium gate (no trend, above flip, strong long gamma) vetoes every
    # directional ticket the regime router emits — the exact tape a debit
    # trade wants is the tape the premium gate forbids.
    fams = STRUCTURE_TO_FAMILIES.get(target_structure) if target_structure else None
    structure_class = ("directional" if fams is not None and fams <= DEBIT_FAMILIES
                       else "premium")
    gate = gate_evaluate(market, cfg.gate, structure_class=structure_class,
                         direction=direction, pin_active=pin_active)
    gate_pass = gate.decision is Decision.GO

    # --- selector: needs a usable chain ---
    candidate = None
    selector_reason = ""
    edge_rich = float("nan")
    all_candidates: list = []
    try:
        rnd = extract_rnd(chain, cfg.rnd)
        edge = compute_edge(rnd, chain, cfg.rnd, physical_pdf=physical_pdf)
        edge_rich = edge.richness_signal
        ctx = GammaContext.from_market_snapshot(market, pin_active=pin_active)
        sel = select_spreads(chain, rnd, edge, ctx, cfg.selector, physical_pdf=physical_pdf,
                             target_families=fams)
        all_candidates = list(sel.all_candidates or sel.ranked or [])
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
        direction=direction, gate_kelly=gate.kelly_fraction,
        physical_density_mode=physical_density_mode or "",
        physical_moments=dict(physical_moments) if physical_moments else None,
        all_candidates=all_candidates,
    )
