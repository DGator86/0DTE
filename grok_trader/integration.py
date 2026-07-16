from __future__ import annotations

import datetime as dt
import logging
import os
from dataclasses import dataclass, field
from typing import Any
from zoneinfo import ZoneInfo

from .agent import GrokAgent
from .audit import GrokAuditStore
from .config import GrokConfig
from .evidence import EvidenceTerminal
from .risk import RiskFirewall

ET = ZoneInfo("America/New_York")
log = logging.getLogger("grok_trader")


def register_grok_track() -> None:
    """Register a fourth isolated ledger before PaperBroker construction."""
    import paper_broker
    if "grok" not in paper_broker.PAPER_TRACKS:
        paper_broker.PAPER_TRACKS = tuple(paper_broker.PAPER_TRACKS) + ("grok",)


@dataclass
class GrokOutcome:
    paper_intent: dict | None = None
    events: list[str] = field(default_factory=list)


class GrokCoordinator:
    """Cadence, evidence terminal, xAI call, risk firewall, and audit boundary."""

    def __init__(self, *, broker: Any, symbol: str, cfg: GrokConfig) -> None:
        self.broker = broker
        self.symbol = symbol.upper()
        self.cfg = cfg
        self.audit = GrokAuditStore(
            cfg.audit_db_path,
            input_price=cfg.input_price_per_million,
            output_price=cfg.output_price_per_million,
        )
        self.firewall = RiskFirewall(cfg)
        self.agent = GrokAgent(cfg) if cfg.enabled else None
        self._last_call_at: dt.datetime | None = None
        self._last_spot: float | None = None
        self._last_regime: str | None = None
        self._last_flip_side: int | None = None
        restored = self.audit.restore_open_positions(self.broker, dt.datetime.now(ET))
        if restored:
            log.warning("Restored %d Grok paper position(s) after restart", restored)

    @classmethod
    def from_env(cls, *, broker: Any, symbol: str, state_dir: str) -> "GrokCoordinator":
        default_db = os.path.join(state_dir, "grok_audit.sqlite")
        return cls(broker=broker, symbol=symbol,
                   cfg=GrokConfig.from_env(default_audit_db=default_db))

    def _has_position(self) -> bool:
        return any(self.broker._track_of(p) == "grok" for p in self.broker.open_positions)

    def _trigger(self, now: dt.datetime, result: Any) -> str | None:
        snap = getattr(result, "snapshot", None)
        market = getattr(snap, "market", None) if snap is not None else None
        if market is None:
            return None
        spot = float(getattr(market, "spot", 0.0) or 0.0)
        flip = float(getattr(market, "gamma_flip", spot) or spot)
        regime = str(getattr(getattr(result, "regime", None), "dominant_regime", "unknown"))
        flip_side = 1 if spot >= flip else -1
        interval = self.cfg.position_interval_seconds if self._has_position() else self.cfg.base_interval_seconds
        elapsed = float("inf") if self._last_call_at is None else (now - self._last_call_at).total_seconds()
        trigger: str | None = None
        if self._last_call_at is None:
            trigger = "startup_review"
        elif self._last_regime is not None and regime != self._last_regime and elapsed >= self.cfg.min_event_gap_seconds:
            trigger = "regime_change"
        elif self._last_flip_side is not None and flip_side != self._last_flip_side and elapsed >= self.cfg.min_event_gap_seconds:
            trigger = "gamma_flip_cross"
        elif self._last_spot and abs(spot / self._last_spot - 1.0) >= self.cfg.event_spot_move_pct and elapsed >= self.cfg.min_event_gap_seconds:
            trigger = "material_spot_move"
        elif elapsed >= interval:
            trigger = "position_review" if self._has_position() else "scheduled_review"
        self._last_spot = spot
        self._last_regime = regime
        self._last_flip_side = flip_side
        return trigger

    def _mandatory_exit_due(self, now: dt.datetime) -> bool:
        current = (now.astimezone(ET).hour, now.astimezone(ET).minute)
        for pos in self.broker.open_positions:
            if self.broker._track_of(pos) != "grok":
                continue
            plan = (getattr(pos, "entry_ctx", {}) or {}).get("grok_plan") or {}
            raw = str(plan.get("mandatory_exit_time") or "").strip()
            try:
                hour, minute = (int(x) for x in raw.split(":"))
            except (TypeError, ValueError):
                continue
            if current >= (hour, minute):
                return True
        return False

    def on_tick(self, now: dt.datetime, result: Any) -> GrokOutcome:
        if not self.cfg.enabled or self.agent is None:
            return GrokOutcome()
        if self._mandatory_exit_due(now):
            approved, events = self.firewall.close_grok_positions(
                now=now, result=result, broker=self.broker,
                reason="mandatory_exit_time",
            )
            self.audit.record_action(
                now=now, cycle_id=None, action="mandatory_exit",
                approved=approved, payload={"reason": "mandatory_exit_time"},
                reasons=[] if approved else events,
            )
            return GrokOutcome(events=events if approved else [
                "GROK mandatory exit deferred: " + ",".join(events)
            ])
        trigger = self._trigger(now, result)
        if trigger is None:
            return GrokOutcome()
        daily_cost = self.audit.daily_cost(now)
        monthly_cost = self.audit.monthly_cost(now)
        daily_cycles = self.audit.daily_cycles(now)
        if (daily_cost >= self.cfg.daily_hard_cap_usd
                or monthly_cost >= self.cfg.monthly_hard_cap_usd
                or daily_cycles >= self.cfg.max_cycles_per_day):
            return GrokOutcome(events=[
                f"GROK LOCKOUT daily=${daily_cost:.2f} monthly=${monthly_cost:.2f} "
                f"cycles={daily_cycles}; deterministic paper exits remain active"
            ])
        allow_new_entry = daily_cost < self.cfg.daily_soft_cap_usd
        terminal = EvidenceTerminal(
            now=now,
            result=result,
            broker=self.broker,
            symbol=self.symbol,
            max_rows=self.cfg.max_raw_rows_per_tool,
            memory=self.audit.get_memory(now),
        )
        pending_intent: dict | None = None
        action_events: list[str] = []
        action_record: tuple[str, bool, dict, list[str]] | None = None

        def dispatch(name: str, args: dict) -> dict:
            nonlocal pending_intent, action_record
            action_tools = {"submit_paper_trade", "close_grok_position", "stand_down"}
            if name in action_tools and action_record is not None:
                return {"ok": False, "error": "one_action_per_cycle"}
            if name == "get_terminal_summary":
                return {"ok": True, "data": terminal.summary()}
            if name == "get_raw_snapshot":
                return {"ok": True, "data": terminal.raw_section(
                    str(args.get("section") or ""),
                    offset=int(args.get("offset") or 0),
                    limit=int(args.get("limit") or self.cfg.max_raw_rows_per_tool),
                )}
            if name == "get_chain_slice":
                return {"ok": True, "data": terminal.chain_slice(
                    center=float(args["center"]),
                    width=float(args["width"]),
                    max_rows=int(args.get("max_rows") or 80),
                )}
            if name == "get_engine_analysis":
                return {"ok": True, "data": terminal.engine_analysis(str(args.get("engine") or ""))}
            if name == "get_account_state":
                return {"ok": True, "data": terminal.account_state()}
            if name == "submit_paper_trade":
                fw = self.firewall.validate_entry(
                    now=now,
                    plan=args,
                    result=result,
                    broker=self.broker,
                    allow_new_entry=allow_new_entry,
                )
                pending_intent = fw.paper_intent if fw.approved else None
                action_record = (name, fw.approved, args, list(fw.reasons))
                return {
                    "ok": fw.approved,
                    "approved": fw.approved,
                    "reasons": list(fw.reasons),
                    "diagnostics": fw.diagnostics or {},
                }
            if name == "close_grok_position":
                approved, events_or_reasons = self.firewall.close_grok_positions(
                    now=now,
                    result=result,
                    broker=self.broker,
                    reason=str(args.get("reason") or "Grok discretionary exit"),
                )
                if approved:
                    action_events.extend(events_or_reasons)
                action_record = (name, approved, args, [] if approved else events_or_reasons)
                return {"ok": approved, "approved": approved, "events_or_reasons": events_or_reasons}
            if name == "stand_down":
                memory = {
                    "last_review_at": now.isoformat(),
                    "last_action": "stand_down",
                    "reason": str(args.get("reason") or "")[:1500],
                    "confidence": float(args.get("confidence") or 0.0),
                    "watch_conditions": list(args.get("watch_conditions") or [])[:12],
                }
                self.audit.put_memory(now, memory)
                action_record = (name, True, args, [])
                return {"ok": True, "recorded": True}
            return {"ok": False, "error": "unknown_tool"}

        result_ai = self.agent.run(
            trigger=trigger,
            terminal_summary=terminal.summary(),
            dispatch=dispatch,
        )
        self._last_call_at = now
        status = "error" if result_ai.error else "ok"
        cycle_id = self.audit.record_cycle(
            now=now,
            trigger=trigger,
            model=self.cfg.model,
            status=status,
            response_id=result_ai.response_id,
            latency_ms=result_ai.latency_ms,
            usage=result_ai.usage,
            note=result_ai.error or result_ai.text,
        )
        if action_record is not None:
            action, approved, payload, reasons = action_record
            self.audit.record_action(
                now=now,
                cycle_id=cycle_id,
                action=action,
                approved=approved,
                payload=payload,
                reasons=reasons,
            )
            if action != "stand_down":
                self.audit.put_memory(now, {
                    "last_review_at": now.isoformat(),
                    "last_action": action,
                    "approved": approved,
                    "reasons": reasons,
                    "model_summary": result_ai.text[:2000],
                })
        if result_ai.error:
            action_events.append(f"GROK ERROR {result_ai.error}")
        else:
            action_events.append(
                f"GROK {trigger} {result_ai.action_name or 'no_action'} "
                f"latency={result_ai.latency_ms/1000:.1f}s "
                f"cost=${self.audit.estimate_cost(result_ai.usage):.3f}"
            )
        return GrokOutcome(paper_intent=pending_intent, events=action_events)
