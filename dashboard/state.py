"""
dashboard/state.py
==================
Serialize TickResult into live_state.json for the observability dashboard.
Read-only snapshot — no decision logic.
"""
from __future__ import annotations

import dataclasses
import json
import os
import tempfile
from typing import Any, Optional

from gate_scorer import MarketSnapshot


def _market_inputs(market: MarketSnapshot) -> dict:
    zg = market.spot - market.gamma_flip
    return {
        "spot": market.spot,
        "net_gex": market.net_gex,
        "gamma_flip": market.gamma_flip,
        "zero_gamma_dist": zg,
        "zero_gamma_dist_pct": zg / market.spot if market.spot else 0.0,
        "gex_pct_rank": market.gex_pct_rank,
        "call_wall": market.call_wall,
        "put_wall": market.put_wall,
        "vix9d": market.vix9d,
        "vix": market.vix,
        "vix3m": market.vix3m,
        "vvix": market.vvix,
        "vvix_baseline": market.vvix_baseline,
        "straddle_breakeven": market.straddle_breakeven,
        "expected_range": market.expected_range,
        "adx": market.adx,
        "rsi": market.rsi,
        "bb_width": market.bb_width,
        "bb_width_baseline": market.bb_width_baseline,
        "vwap": market.vwap,
        "vwap_reversion_count": market.vwap_reversion_count,
        "tick_abs_mean": market.tick_abs_mean,
        "cvd_slope": market.cvd_slope,
        "has_catalyst": market.has_catalyst,
    }


def serialize_tick_result(
    result,
    *,
    feed_source: Optional[str] = None,
    paper_summary: Optional[dict] = None,
    market_status: Optional[dict] = None,
) -> dict:
    """Build live_state payload from a UnifiedOrchestrator tick."""
    regime = result.regime
    intent = result.intent
    dec = result.decision
    snap = result.snapshot

    chain_available = snap is not None and snap.chain is not None
    market = snap.market if snap else None

    doing: dict[str, Any] = {
        "dominant_regime": regime.dominant_regime,
        "permitted_engine": regime.permitted_engine,
        "stand_down": regime.stand_down,
        "structure": intent.decision.structure,
        "direction": intent.decision.direction,
        "conviction": intent.decision.conviction,
        "size_mult": intent.size_mult,
        "final_size_mult": result.final_size_mult,
    }

    if dec is not None:
        doing.update({
            "gate_pass": bool(dec.gate_pass),
            "gate_score": dec.gate_score,
            "decision": dec.decision,
        })
    else:
        doing.update({
            "gate_pass": False,
            "gate_score": 0.0,
            "decision": "NO_TRADE",
        })

    stand_down_reason = None
    if regime.stand_down:
        if regime.global_information_gain >= 70.0:
            stand_down_reason = "high_information_gain"
        else:
            top_conf = max(regime.confidences.values()) if regime.confidences else 0
            stand_down_reason = "low_regime_confidence" if top_conf < 55 else "regime_vetoes"

    gate_failed: list = []
    selector_vetoes: list = []
    no_trade_reason = ""
    risk_vetoes: list = []

    if dec is not None:
        gate_failed = list(dec.gate_failed or [])
        no_trade_reason = dec.no_trade_reason or ""
        if dec.candidate is not None:
            selector_vetoes = list(dec.candidate.veto_reasons or ())
    elif intent.decision.structure == "NT":
        no_trade_reason = intent.note or "regime_nt"

    for v in result.vetoes or []:
        if str(v).startswith("risk:"):
            risk_vetoes.append(str(v))

    why: dict[str, Any] = {
        "regime_confidences": dict(regime.confidences),
        "global_information_gain": regime.global_information_gain,
        "stand_down_reason": stand_down_reason,
        "dealer_vetoes": list(regime.vetoes),
        "matrix_cell": [intent.exec_regime, intent.context_regime, intent.direction_bias],
        "intent_note": intent.note,
        "capture": intent.decision.capture,
        "strike_rule": intent.decision.strike_rule,
        "gate_failed": gate_failed,
        "selector_vetoes": selector_vetoes,
        "no_trade_reason": no_trade_reason,
        "risk_vetoes": risk_vetoes,
        "intent_vetoes": list(intent.vetoes or []),
    }

    inputs = _market_inputs(market) if market else {}

    payload = {
        "ts": result.ts.isoformat(),
        "market": market_status or {},
        "feed_source": feed_source,
        "chain_available": chain_available,
        "doing": doing,
        "inputs": inputs,
        "why": why,
        "paper": paper_summary or {},
    }
    return payload


def write_live_state(path: str, payload: dict) -> None:
    """Atomically write live_state.json."""
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    data = json.dumps(payload, indent=2, default=str)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".live_state_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def read_live_state(path: str) -> Optional[dict]:
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)
