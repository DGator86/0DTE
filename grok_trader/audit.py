from __future__ import annotations

import datetime as dt
import json
import sqlite3
import uuid
from dataclasses import dataclass
from typing import Any
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0


class GrokAuditStore:
    """Small persistent audit and cost ledger for all model activity."""

    def __init__(self, path: str, *, input_price: float, output_price: float) -> None:
        self.path = path
        self.input_price = input_price
        self.output_price = output_price
        self._db = sqlite3.connect(path)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS grok_cycles (
                id TEXT PRIMARY KEY,
                ts TEXT NOT NULL,
                session_date TEXT NOT NULL,
                trigger TEXT NOT NULL,
                model TEXT NOT NULL,
                status TEXT NOT NULL,
                response_id TEXT,
                latency_ms INTEGER NOT NULL,
                input_tokens INTEGER NOT NULL,
                cached_tokens INTEGER NOT NULL,
                output_tokens INTEGER NOT NULL,
                reasoning_tokens INTEGER NOT NULL,
                cost_usd REAL NOT NULL,
                note TEXT
            )
            """
        )
        self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS grok_actions (
                id TEXT PRIMARY KEY,
                cycle_id TEXT,
                ts TEXT NOT NULL,
                action TEXT NOT NULL,
                approved INTEGER NOT NULL,
                payload_json TEXT,
                reasons_json TEXT
            )
            """
        )
        self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS grok_session_memory (
                session_date TEXT PRIMARY KEY,
                updated_at TEXT NOT NULL,
                summary_json TEXT NOT NULL
            )
            """
        )
        self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS grok_open_positions (
                id TEXT PRIMARY KEY,
                payload_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        self._db.commit()

    def estimate_cost(self, usage: Usage) -> float:
        # Conservative: cached input is billed as normal input unless a future
        # xAI price schedule is explicitly configured in code.
        input_cost = usage.input_tokens * self.input_price / 1_000_000.0
        output_cost = usage.output_tokens * self.output_price / 1_000_000.0
        return input_cost + output_cost

    def record_cycle(
        self,
        *,
        now: dt.datetime,
        trigger: str,
        model: str,
        status: str,
        response_id: str | None,
        latency_ms: int,
        usage: Usage,
        note: str = "",
    ) -> str:
        cycle_id = uuid.uuid4().hex
        session_date = now.astimezone(ET).date().isoformat()
        self._db.execute(
            "INSERT INTO grok_cycles VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                cycle_id,
                now.isoformat(),
                session_date,
                trigger,
                model,
                status,
                response_id,
                int(latency_ms),
                int(usage.input_tokens),
                int(usage.cached_tokens),
                int(usage.output_tokens),
                int(usage.reasoning_tokens),
                round(self.estimate_cost(usage), 8),
                note[:4000],
            ),
        )
        self._db.commit()
        return cycle_id

    def record_action(
        self,
        *,
        now: dt.datetime,
        cycle_id: str | None,
        action: str,
        approved: bool,
        payload: Any,
        reasons: list[str] | tuple[str, ...] = (),
    ) -> None:
        self._db.execute(
            "INSERT INTO grok_actions VALUES (?,?,?,?,?,?,?)",
            (
                uuid.uuid4().hex,
                cycle_id,
                now.isoformat(),
                action,
                1 if approved else 0,
                json.dumps(payload, default=str, sort_keys=True),
                json.dumps(list(reasons)),
            ),
        )
        self._db.commit()

    def daily_cost(self, now: dt.datetime) -> float:
        day = now.astimezone(ET).date().isoformat()
        row = self._db.execute(
            "SELECT COALESCE(SUM(cost_usd), 0) FROM grok_cycles WHERE session_date=?",
            (day,),
        ).fetchone()
        return float(row[0] or 0.0)

    def monthly_cost(self, now: dt.datetime) -> float:
        month = now.astimezone(ET).strftime("%Y-%m")
        row = self._db.execute(
            "SELECT COALESCE(SUM(cost_usd), 0) FROM grok_cycles WHERE substr(session_date,1,7)=?",
            (month,),
        ).fetchone()
        return float(row[0] or 0.0)

    def daily_cycles(self, now: dt.datetime) -> int:
        day = now.astimezone(ET).date().isoformat()
        row = self._db.execute(
            "SELECT COUNT(*) FROM grok_cycles WHERE session_date=?",
            (day,),
        ).fetchone()
        return int(row[0] or 0)

    def get_memory(self, now: dt.datetime) -> dict:
        day = now.astimezone(ET).date().isoformat()
        row = self._db.execute(
            "SELECT summary_json FROM grok_session_memory WHERE session_date=?",
            (day,),
        ).fetchone()
        if not row:
            return {}
        try:
            value = json.loads(row[0])
            return value if isinstance(value, dict) else {}
        except (json.JSONDecodeError, TypeError):
            return {}

    def restore_open_positions(self, broker: Any, now: dt.datetime | None = None) -> int:
        """Restore Grok positions lost to a process restart.

        The legacy broker persists closed trades only.  This companion table is
        intentionally scoped to the Grok track and uses the broker's native
        PaperPosition/Leg types so subsequent marking and exits are unchanged.
        """
        from paper_broker import PaperPosition
        from spread_selector import Leg

        if any(broker._track_of(p) == "grok" for p in broker.open_positions):
            return 0
        rows = self._db.execute(
            "SELECT payload_json FROM grok_open_positions ORDER BY updated_at"
        ).fetchall()
        current_date = (now or dt.datetime.now(ET)).astimezone(ET).date()
        restored = 0
        for (payload_json,) in rows:
            try:
                data = json.loads(payload_json)
                legs = tuple(Leg(float(x["strike"]), str(x["kind"]), int(x["qty"]))
                             for x in data["legs"])
                opened_at = dt.datetime.fromisoformat(data["opened_at"])
                if opened_at.astimezone(ET).date() != current_date:
                    continue
                pos = PaperPosition(
                    id=str(data["id"]),
                    opened_at=opened_at,
                    family=str(data["family"]),
                    legs=legs,
                    contracts=int(data["contracts"]),
                    entry_credit=float(data["entry_credit"]),
                    max_profit_ps=float(data["max_profit_ps"]),
                    max_loss_ps=float(data["max_loss_ps"]),
                    short_strikes=tuple(data.get("short_strikes") or ()),
                    long_strikes=tuple(data.get("long_strikes") or ()),
                    peak_pnl_ps=float(data.get("peak_pnl_ps") or 0.0),
                    trailing_armed=bool(data.get("trailing_armed")),
                    last_pnl_ps=float(data.get("last_pnl_ps") or 0.0),
                    slip_entry_ps=float(data.get("slip_entry_ps") or 0.0),
                    entry_ctx=dict(data.get("entry_ctx") or {}),
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
            pos.entry_ctx["fill_track"] = "grok"
            broker.open_positions.append(pos)
            broker.position_monitor.register(pos.id, pos.entry_ctx)
            restored += 1
        return restored

    def persist_open_positions(self, broker: Any, now: dt.datetime) -> None:
        self._db.execute("DELETE FROM grok_open_positions")
        for pos in broker.open_positions:
            if broker._track_of(pos) != "grok":
                continue
            payload = {
                "id": pos.id,
                "opened_at": pos.opened_at.isoformat(),
                "family": pos.family,
                "legs": [
                    {"strike": leg.strike, "kind": leg.kind, "qty": leg.qty}
                    for leg in pos.legs
                ],
                "contracts": pos.contracts,
                "entry_credit": pos.entry_credit,
                "max_profit_ps": pos.max_profit_ps,
                "max_loss_ps": pos.max_loss_ps,
                "short_strikes": list(pos.short_strikes),
                "long_strikes": list(pos.long_strikes),
                "peak_pnl_ps": pos.peak_pnl_ps,
                "trailing_armed": pos.trailing_armed,
                "last_pnl_ps": pos.last_pnl_ps,
                "slip_entry_ps": pos.slip_entry_ps,
                "entry_ctx": pos.entry_ctx,
            }
            self._db.execute(
                "INSERT INTO grok_open_positions VALUES (?, ?, ?)",
                (pos.id, json.dumps(payload, default=str, sort_keys=True), now.isoformat()),
            )
        self._db.commit()

    def put_memory(self, now: dt.datetime, value: dict) -> None:
        day = now.astimezone(ET).date().isoformat()
        self._db.execute(
            """
            INSERT INTO grok_session_memory(session_date, updated_at, summary_json)
            VALUES (?, ?, ?)
            ON CONFLICT(session_date) DO UPDATE SET
                updated_at=excluded.updated_at,
                summary_json=excluded.summary_json
            """,
            (day, now.isoformat(), json.dumps(value, default=str, sort_keys=True)),
        )
        self._db.commit()
