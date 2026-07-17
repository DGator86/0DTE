from __future__ import annotations

import dataclasses
import datetime as dt
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

# Final policy/action fields are not evidence.  They are stripped from every
# Legacy/V2/V3 view before anything leaves the process.
BLOCKED_EXACT = {
    "decision",
    "decision_summary",
    "action",
    "stand_down",
    "trade_or_no_trade",
    "recommended_trade",
    "selected_legs",
    "trade_intent",
    "recommended_action",
    "selected_candidate",
    "selected_candidate_id",
    "selected_strategy",
    "selected_structure",
    "selected_strikes",
    "final_size",
    "final_size_mult",
    "paper_intent",
    "paper_intents",
    "authoritative_policy",
    "policy_source",
    "champion_trade",
    "permitted_engine",
}
BLOCKED_SUFFIXES = (
    "_decision",
    "_action",
    "_policy_structure",
    "_policy_direction",
    "_policy_confidence",
    "_selected_candidate",
    "_selected_candidate_id",
    "_selected_structure",
    "_selected_strikes",
    "_paper_intent",
    "_recommended_trade",
    "_selected_legs",
)


def _blocked_key(key: object) -> bool:
    k = str(key).strip().lower()
    return (
        k in BLOCKED_EXACT
        or "selected_candidate" in k
        or "recommended_action" in k
        or k.startswith("policy_")
        or any(k.endswith(s) for s in BLOCKED_SUFFIXES)
    )


def json_safe(value: Any, *, max_items: int | None = None, _depth: int = 0) -> Any:
    if _depth > 12:
        return "<max-depth>"
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (dt.datetime, dt.date, dt.time)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return json_safe(value.item(), max_items=max_items, _depth=_depth + 1)
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist(), max_items=max_items, _depth=_depth + 1)
    if dataclasses.is_dataclass(value):
        return json_safe(dataclasses.asdict(value), max_items=max_items, _depth=_depth + 1)
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if _blocked_key(key):
                continue
            out[str(key)] = json_safe(item, max_items=max_items, _depth=_depth + 1)
        return out
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        seq = list(value)
        limited = seq if max_items is None else seq[:max_items]
        out = [json_safe(x, max_items=max_items, _depth=_depth + 1) for x in limited]
        if max_items is not None and len(seq) > max_items:
            out.append({"_truncated": len(seq) - max_items})
        return out
    if hasattr(value, "__dict__"):
        public = {k: v for k, v in vars(value).items() if not k.startswith("_")}
        return json_safe(public, max_items=max_items, _depth=_depth + 1)
    return str(value)


def blind(value: Any) -> Any:
    return json_safe(value)


def assert_decision_blind(value: Any, path: str = "root") -> None:
    """Raise if a forbidden policy field survived serialization."""
    if isinstance(value, Mapping):
        for key, item in value.items():
            if _blocked_key(key):
                raise ValueError(f"decision leakage at {path}.{key}")
            assert_decision_blind(item, f"{path}.{key}")
    elif isinstance(value, list):
        for idx, item in enumerate(value):
            assert_decision_blind(item, f"{path}[{idx}]")


def _split_signals(signals: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {
        "legacy": {},
        "v2": {},
        "v3": {},
        "shared": {},
    }
    for key, value in signals.items():
        if _blocked_key(key):
            continue
        k = str(key).lower()
        if k.startswith(("v2_", "prediction_", "forecast_", "physical_", "ranker_")):
            groups["v2"][str(key)] = value
        elif k.startswith(("v3_", "part3_")):
            groups["v3"][str(key)] = value
        elif k.startswith(("legacy_", "policy_", "gate_", "selector_", "rnd_")):
            groups["legacy"][str(key)] = value
        else:
            groups["shared"][str(key)] = value
    return {name: blind(data) for name, data in groups.items()}


def _chain_summary(chain: Any) -> dict:
    if chain is None:
        return {"available": False}
    quotes = sorted(getattr(chain, "quotes", ()) or (), key=lambda q: float(q.strike))
    strikes = [float(q.strike) for q in quotes]
    spot = float(getattr(chain, "spot", 0.0) or 0.0)
    atm = sorted(quotes, key=lambda q: abs(float(q.strike) - spot))[:7]
    return {
        "available": True,
        "spot": spot,
        "t_years": float(getattr(chain, "t_years", 0.0) or 0.0),
        "quote_count": len(quotes),
        "strike_min": min(strikes) if strikes else None,
        "strike_max": max(strikes) if strikes else None,
        "near_atm": blind(atm),
    }


def _grok_positions(broker: Any, now: dt.datetime) -> list[dict]:
    rows: list[dict] = []
    for pos in getattr(broker, "open_positions", ()):
        track = str((getattr(pos, "entry_ctx", {}) or {}).get("fill_track") or "legacy")
        if track != "grok":
            continue
        rows.append(
            {
                "id": pos.id,
                "family": pos.family,
                "legs": blind(pos.legs),
                "contracts": pos.contracts,
                "opened_at": pos.opened_at.isoformat(),
                "hold_minutes": round((now - pos.opened_at).total_seconds() / 60.0, 2),
                "entry_credit": pos.entry_credit,
                "max_profit_per_share": pos.max_profit_ps,
                "max_loss_per_share": pos.max_loss_ps,
                "last_pnl_per_share": pos.last_pnl_ps,
                "peak_pnl_per_share": pos.peak_pnl_ps,
            }
        )
    return rows


class EvidenceTerminal:
    """Decision-blinded, read-only terminal over one immutable tick result."""

    def __init__(self, *, now: dt.datetime, result: Any, broker: Any, symbol: str,
                 max_rows: int, memory: dict | None = None) -> None:
        self.now = now
        self.result = result
        self.broker = broker
        self.symbol = symbol
        self.max_rows = max_rows
        self.memory = memory or {}
        self.snapshot = getattr(result, "snapshot", None)
        self.signals = dict(getattr(result, "signals", None) or {})
        self.groups = _split_signals(self.signals)

    def summary(self) -> dict:
        snap = self.snapshot
        market = getattr(snap, "market", None) if snap is not None else None
        chain = getattr(snap, "chain", None) if snap is not None else None
        v3 = json_safe(getattr(self.result, "part3", None) or {}, max_items=50)
        evidence = {
            "as_of": self.now.isoformat(),
            "symbol": self.symbol,
            "market": blind(market),
            "chain": _chain_summary(chain),
            "gex_feed_source": str(getattr(snap, "gex_feed_source", "") or "") if snap is not None else "",
            "regime_analysis": blind(getattr(self.result, "regime", None)),
            "legacy_analysis": self.groups["legacy"],
            "v2_analysis": self.groups["v2"],
            "v3_analysis": {"signals": self.groups["v3"], "part3": v3},
            "shared_analysis": self.groups["shared"],
            "analytical_vetoes": blind(getattr(self.result, "vetoes", ()) or ()),
            "sigma_cones": blind(getattr(self.result, "sigma_cones", None)),
            "ras_analysis": blind(getattr(self.result, "ras_results", None) or ()),
            "account": self.account_state(),
            "session_memory": blind(self.memory),
        }
        assert_decision_blind(evidence)
        return evidence

    def account_state(self) -> dict:
        ledgers = getattr(self.broker, "ledgers", {}) or {}
        return {
            "track": "grok",
            "realized_cash": float(ledgers.get("grok", getattr(self.broker.cfg, "starting_cash", 0.0))),
            "equity": float(self.broker.track_equity("grok")) if "grok" in ledgers else None,
            "open_positions": _grok_positions(self.broker, self.now),
            "paper_only": True,
        }

    def raw_section(self, section: str, *, offset: int = 0, limit: int | None = None) -> dict:
        snap = self.snapshot
        if snap is None:
            return {"error": "snapshot_unavailable"}
        limit = max(1, min(int(limit or self.max_rows), self.max_rows))
        offset = max(0, int(offset))
        if section == "market":
            return {"section": section, "data": blind(getattr(snap, "market", None))}
        if section == "chain":
            chain = getattr(snap, "chain", None)
            quotes = list(getattr(chain, "quotes", ()) or ()) if chain is not None else []
            return {
                "section": section,
                "offset": offset,
                "limit": limit,
                "total": len(quotes),
                "spot": getattr(chain, "spot", None),
                "t_years": getattr(chain, "t_years", None),
                "rows": blind(quotes[offset:offset + limit]),
            }
        if section in {"option_rows", "weekly_option_rows"}:
            rows = list(getattr(snap, section, None) or [])
            return {
                "section": section,
                "offset": offset,
                "limit": limit,
                "total": len(rows),
                "rows": blind(rows[offset:offset + limit]),
            }
        if section == "bars":
            bars = json_safe(getattr(snap, "bars", None))
            if not isinstance(bars, dict):
                return {"section": section, "data": bars}
            sliced: dict[str, Any] = {}
            for key, value in bars.items():
                if isinstance(value, list):
                    sliced[key] = value[offset:offset + limit]
                else:
                    sliced[key] = value
            return {"section": section, "offset": offset, "limit": limit, "data": sliced}
        return {"error": "unknown_section", "allowed": ["market", "bars", "chain", "option_rows", "weekly_option_rows"]}

    def chain_slice(self, *, center: float, width: float, max_rows: int = 80) -> dict:
        snap = self.snapshot
        chain = getattr(snap, "chain", None) if snap is not None else None
        if chain is None:
            return {"error": "chain_unavailable"}
        width = max(0.25, min(float(width), 50.0))
        rows = [q for q in chain.sorted_quotes() if abs(float(q.strike) - float(center)) <= width]
        rows = rows[:max(1, min(int(max_rows), self.max_rows))]
        return {"spot": chain.spot, "center": center, "width": width, "rows": blind(rows)}

    def engine_analysis(self, engine: str) -> dict:
        engine = engine.lower()
        if engine == "legacy":
            return self.groups["legacy"]
        if engine == "v2":
            return self.groups["v2"]
        if engine == "v3":
            return {"signals": self.groups["v3"], "part3": blind(getattr(self.result, "part3", None) or {})}
        if engine == "shared":
            return self.groups["shared"]
        return {"error": "unknown_engine", "allowed": ["legacy", "v2", "v3", "shared"]}
