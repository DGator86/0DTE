"""
dashboard/state.py
==================
Serialize TickResult into live_state.json for the observability dashboard.
Read-only snapshot — no decision logic.

PR C/D: payloads use schema_version=live.v1 with explicit sections (feeds,
legacy, forecast, v3, …). Flat top-level aliases are gone
(system.compat_flat_keys=False); the dashboard reads documented sections only.
"""
from __future__ import annotations

import dataclasses
import json
import math
import os
import tempfile
from typing import Any, Mapping, Optional

from dashboard.live_schema import (
    LIVE_SCHEMA_VERSION,
    feeds_payload_from_statuses,
    synthesize_feed_statuses,
)
from gate_scorer import MarketSnapshot
from prediction.feed_status import FeedStatus, build_feed_status


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
    part3: Optional[dict] = None,
    feed_statuses: Optional[Mapping[str, FeedStatus]] = None,
    feed_ages_seconds: Optional[Mapping[str, Optional[float]]] = None,
) -> dict:
    """Build live_state payload from a UnifiedOrchestrator tick (live.v1)."""
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
        # continuous direction bias (0-100, 50 = neutral) + its label, for the
        # four-way quadrant and regime shading on the chart; bias_fast is the
        # raw fast-timeframe composite (leads the blend at intraday turns)
        "direction_bias": intent.direction_bias,
        "bias_value": intent.bias_value,
        "bias_fast": getattr(intent, "bias_fast", None),
        "bias_slow": getattr(intent, "bias_slow", None),
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

    ras_results = getattr(result, "ras_results", None) or []
    if ras_results:
        # Match unified_loop worst-position selection for multi-position ticks.
        primary = min(ras_results, key=lambda r: float(getattr(r, "score", 0.0)))
        why["position_health"] = {
            "ras_score": primary.score,
            "ras_action": primary.action,
            "position_id": primary.position_id,
            "components": [
                {"name": c.name, "raw": c.raw, "contribution": c.contribution, "note": c.note}
                for c in primary.components
            ],
        }
        if len(ras_results) > 1:
            why["position_health"]["n_positions"] = len(ras_results)
    elif paper_summary and paper_summary.get("open"):
        open_pos = paper_summary["open"][0]
        ctx = (open_pos.get("entry_ctx") or {})
        if ctx.get("ras_score") is not None:
            why["position_health"] = {
                "ras_score": ctx.get("ras_score"),
                "ras_action": ctx.get("ras_action"),
                "position_id": open_pos.get("id"),
                "components": ctx.get("ras_components"),
            }

    inputs = _market_inputs(market) if market else {}

    # V2 / policy / GEX observation signals for the dashboard V2 tab.
    raw_signals = getattr(result, "signals", None) or {}
    v2_keys = (
        "policy_", "v2_", "phys_", "gex_", "legacy_policy_", "pin_", "cf_",
        "cone_",
    )
    v2_signals = {
        k: v for k, v in raw_signals.items()
        if any(k.startswith(p) for p in v2_keys) or k in (
            "gex_rank_warm", "routed_structure", "premium_flip",
        )
    }

    # Flat forecast summary for the V2 tab when prediction_store is offline.
    forecast_summary = {
        k[len("v2_fc_"):]: v for k, v in raw_signals.items()
        if k.startswith("v2_fc_")
    }

    # Parallel decision summary for Legacy / V2 / V3 / SPY-DER panels.
    # V2 side uses only v2_policy_* — never fall back to policy_* (legacy).
    parallel = {
        "legacy": {
            "structure": intent.decision.structure,
            "direction": intent.decision.direction,
            "size_mult": intent.size_mult,
            "gate_pass": doing.get("gate_pass"),
            "gate_score": doing.get("gate_score"),
            "decision": doing.get("decision"),
            "label": "Legacy",
        },
        "v2": {
            "structure": raw_signals.get("v2_policy_structure"),
            "direction": raw_signals.get("v2_policy_direction"),
            "action": raw_signals.get("v2_policy_action"),
            "confidence": raw_signals.get("v2_policy_confidence"),
            "uncertainty": raw_signals.get("v2_policy_uncertainty"),
            "size_cap": raw_signals.get("policy_size_cap"),
            "source": raw_signals.get("policy_source"),
            "mode": raw_signals.get("policy_mode"),
            "disagreement": raw_signals.get("policy_disagreement"),
            "fallback_used": raw_signals.get("policy_fallback_used"),
            "label": "V2",
        },
    }
    # V3 from Part 3 decision summary (statistical action preferred).
    part3_for_parallel = part3
    if part3_for_parallel is None:
        part3_for_parallel = getattr(result, "part3", None) or {}
    ds_p3 = (part3_for_parallel or {}).get("decision_summary") or {}
    parallel["v3"] = {
        "label": "V3",
        "structure": ds_p3.get("family"),
        "direction": ds_p3.get("direction"),
        "action": ds_p3.get("statistical_action") or ds_p3.get("action"),
        "confidence": ds_p3.get("p_positive_utility"),
        "uncertainty": ds_p3.get("uncertainty"),
        "size_cap": None,
        "candidate_id": ds_p3.get("selected_candidate_id"),
        "mode": (part3_for_parallel or {}).get("mode", "shadow"),
        "source": "part3",
    }
    spy_der_payload = getattr(result, "spy_der", None)
    if isinstance(spy_der_payload, dict) and spy_der_payload:
        parallel["spy_der"] = dict(spy_der_payload)
        parallel["spy_der"].setdefault("label", "SPY-DER")
        parallel["spy_der"].setdefault("track", "spy_der")
    else:
        parallel["spy_der"] = {
            "track": "spy_der",
            "label": "SPY-DER",
            "source": "spy_der",
            "mode": "shadow",
            "action": "UNAVAILABLE",
            "available": False,
        }

    # MTF sigma cones (live panes) — prefer orchestrator cache via result attr.
    sigma_cones = getattr(result, "sigma_cones", None)
    if sigma_cones is None and hasattr(result, "signals"):
        # Reconstruct a minimal 5m pane from flat signals when live cache absent.
        if raw_signals.get("cone_primary_tf"):
            bands = []
            for k, tag in ((0.5, "0p5"), (1.0, "1p0"), (2.0, "2p0")):
                lo = raw_signals.get(f"cone_{tag}_lo")
                hi = raw_signals.get(f"cone_{tag}_hi")
                if lo is None or hi is None:
                    continue
                bands.append({
                    "sigma": k,
                    "lo": lo,
                    "hi": hi,
                    "mid": raw_signals.get(f"cone_{tag}_mid"),
                    "horizon_min": raw_signals.get(f"cone_{tag}_horizon_min"),
                })
            if bands:
                sigma_cones = {
                    "model_version": raw_signals.get("cone_model_version"),
                    "panes": [{
                        "timeframe": raw_signals.get("cone_primary_tf"),
                        "spot": raw_signals.get("cone_spot"),
                        "bands": bands,
                    }],
                }

    part3_payload = part3
    if part3_payload is None:
        part3_payload = getattr(result, "part3", None)
    if part3_payload is None:
        part3_payload = {
            "note": "part3 decision not available",
            "shadow_label": "SHADOW — not an executed order",
            "mode": "shadow",
        }

    statuses = synthesize_feed_statuses(
        feed_source=feed_source,
        chain_available=chain_available,
        ages_seconds=feed_ages_seconds,
        feed_statuses=feed_statuses,
    )
    feeds = feeds_payload_from_statuses(statuses)

    snapshot_id = None
    if isinstance(raw_signals, dict):
        snapshot_id = raw_signals.get("_snapshot_id") or raw_signals.get("snapshot_id")
    ts_iso = result.ts.isoformat()
    market_session = market_status or {}

    # live.v1 market: session fields + nested session/inputs.
    market_section = {
        **dict(market_session),
        "session": dict(market_session),
        "inputs": inputs,
    }

    # live.v1 forecast: summary + shadow signals; never source_version=v3.
    # parallel keeps the historical V2 object; parallel_tracks is the
    # multi-system comparison map (legacy / v2 / v3 / spy_der).
    forecast_section: dict[str, Any] = {
        "source_version": "v2",
        "source_type": (
            "unavailable" if not forecast_summary and not v2_signals
            else "shadow_observation"
        ),
        "summary": forecast_summary or None,
        "v2_signals": v2_signals,
        "sigma_cones": sigma_cones,
        "parallel": parallel.get("v2"),
        "parallel_tracks": parallel,
    }
    if forecast_summary:
        forecast_section.update(forecast_summary)

    paper_section = paper_summary or {}

    return {
        "schema_version": LIVE_SCHEMA_VERSION,
        "generated_at": ts_iso,
        "snapshot": {
            "snapshot_id": snapshot_id,
            "ts": ts_iso,
            "chain_available": chain_available,
            "feed_source": feed_source,
        },
        "feeds": feeds,
        "market": market_section,
        "legacy": {
            "source_version": "v1",
            "doing": doing,
            "why": why,
            "parallel": parallel.get("legacy"),
        },
        "forecast": forecast_section,
        "v3": {
            "source_version": "v3",
            "source_type": "shadow",
            "mode": (part3_payload or {}).get("mode", "shadow"),
            "shadow_label": (part3_payload or {}).get(
                "shadow_label", "SHADOW — not an executed order"),
            "decision": part3_payload,
        },
        "accounts": {
            "reference": {
                "track": "legacy",
                "authority": "v1",
                "paper": paper_section,
            },
            "candidate": None,
            "champion": None,
        },
        "risk": {
            "vetoes": risk_vetoes,
            "scope": "global_single_risk_manager",
        },
        "paper": paper_section,
        "system": {
            "status": "live",
            "note": None,
            "compat_flat_keys": False,
            "schema_version": LIVE_SCHEMA_VERSION,
        },
    }


def heartbeat_state(
    now,
    *,
    status: str,
    note: str,
    feed_source: Optional[str] = None,
    paper_summary: Optional[dict] = None,
    market_status: Optional[dict] = None,
    feed_statuses: Optional[Mapping[str, FeedStatus]] = None,
) -> dict:
    """Build a live_state payload for a loop iteration that produced no tick.

    Lets the dashboard tell "pipeline alive but idle/feed-down" (fresh ts +
    status/note) apart from "pipeline crashed" (stale/absent file). Carries no
    decision or market inputs — only liveness and the reason there is no data.
    """
    ts_iso = now.isoformat()
    statuses = synthesize_feed_statuses(
        feed_source=feed_source,
        chain_available=False,
        feed_statuses=feed_statuses,
    )
    # Heartbeat with no tick: do not claim overall LIVE.
    if status in ("market_closed", "feed_not_ready", "feed_error"):
        if statuses["spot"].status == "LIVE":
            statuses = dict(statuses)
            statuses["spot"] = build_feed_status(
                source="spot",
                freshness_limit_seconds=statuses["spot"].freshness_limit_seconds,
                provider=feed_source,
                present=False,
                required=True,
                error_code=status,
            )
    feeds = feeds_payload_from_statuses(statuses)
    market_session = market_status or {}
    market_section = {
        **dict(market_session),
        "session": dict(market_session),
        "inputs": {},
    }
    paper_section = paper_summary or {}
    return {
        "schema_version": LIVE_SCHEMA_VERSION,
        "generated_at": ts_iso,
        "snapshot": {
            "snapshot_id": None,
            "ts": ts_iso,
            "chain_available": False,
            "feed_source": feed_source,
        },
        "feeds": feeds,
        "market": market_section,
        "legacy": {"source_version": "v1", "doing": {}, "why": {}, "parallel": None},
        "forecast": {
            "source_version": "v2",
            "source_type": "unavailable",
            "summary": None,
            "v2_signals": {},
            "sigma_cones": None,
            "parallel": None,
        },
        "v3": {
            "source_version": "v3",
            "source_type": "unavailable",
            "mode": "shadow",
            "shadow_label": "SHADOW — not an executed order",
            "decision": None,
        },
        "accounts": {
            "reference": {"track": "legacy", "authority": "v1",
                          "paper": paper_section},
            "candidate": None,
            "champion": None,
        },
        "risk": {"vetoes": [], "scope": "global_single_risk_manager"},
        "paper": paper_section,
        "system": {
            "status": status,
            "note": note,
            "compat_flat_keys": False,
            "schema_version": LIVE_SCHEMA_VERSION,
        },
    }


def _sanitize_non_finite(obj: Any) -> Any:
    """Replace inf/-inf/nan floats with None.

    json.dumps() writes these as bare Infinity/NaN tokens (not valid JSON) by
    default, so a value like ev_per_risk from a division by a near-zero
    max_loss can slip into live_state.json unnoticed on write. Starlette's
    JSONResponse correctly rejects them on read (allow_nan=False), 500-ing
    every /api/live request until the next tick happens to avoid the same
    ratio. Scrub at the single point everything funnels through instead.
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _sanitize_non_finite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_non_finite(v) for v in obj]
    return obj


def write_live_state(path: str, payload: dict) -> None:
    """Atomically write live_state.json."""
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    payload = _sanitize_non_finite(payload)
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


def serialize_part3_decision(decision, *, generated_at: str | None = None) -> dict:
    """
    Dashboard panel payload for Part 3 (§40). Always includes timestamp,
    model versions, and mode so shadow output is not mistaken for an order.
    """
    if decision is None:
        return {"note": "part3 decision not available"}
    d = decision.to_dict() if hasattr(decision, "to_dict") else dict(decision)
    return {
        "generated_at": generated_at or d.get("ts"),
        "mode": d.get("mode", "shadow"),
        "model_versions": dict(d.get("model_versions") or {}),
        "decision_summary": {
            "action": d.get("action"),
            "statistical_action": d.get("statistical_action"),
            "hard_vetoes": list(d.get("hard_vetoes") or ()),
            "selected_candidate_id": d.get("selected_candidate_id"),
            "family": d.get("family"),
            "direction": d.get("direction"),
            "expected_order_value": d.get("expected_order_value"),
            "candidate_utility": d.get("candidate_utility"),
            "p_positive_utility": d.get("p_positive_utility"),
            "fill_probability": d.get("fill_probability"),
            "uncertainty": d.get("uncertainty"),
            "ood_score": d.get("ood_score"),
            "reasons": list(d.get("reasons") or ()),
        },
        "shadow_label": "SHADOW — not an executed order",
    }
