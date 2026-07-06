"""
paper_broker.py  —  in-house paper trading: simulated auto-execution on LIVE data.

WHAT THIS IS
    Shadow mode journals every *evaluation* but never holds a position. This adds
    a virtual broker on top of it: when the pipeline emits a TRADE ticket, the
    broker "fills" it against the live option chain, then marks the position to
    market every tick and exits it on a stop-loss, profit-target, trailing-stop,
    or end-of-day rule. Every closed trade is journaled with realized P&L and the
    exit reason. NO REAL ORDERS ARE PLACED — it tracks a virtual cash account
    (default $1000) so a strategy can prove itself before any real money.

POSITION MATH (per share, options are ×100)
    A position is the candidate's leg list. At entry we collect `entry_credit`
    (negative for debit structures). At any later tick the cost to re-open the
    same legs is `credit_now = _credit(legs, chain)`. The position's P&L per share
    is therefore `entry_credit - credit_now`. Max profit / max loss come from the
    expiry payoff curve. Slippage (a fraction of each leg's bid-ask half-spread)
    is charged on both entry and exit so fills aren't free mid.

SECURITY: no credentials, no network. Pure simulation over the data the feed
already provides.

NOT financial advice.
"""
from __future__ import annotations

import datetime as dt
import json
import sqlite3
import uuid
from dataclasses import dataclass, field
from typing import Optional
from zoneinfo import ZoneInfo

import numpy as np

from spread_selector import Leg, _chain_maps, _credit, _payoff_curve

ET = ZoneInfo("America/New_York")


# --------------------------------------------------------------------------- #
# Config                                                                        #
# --------------------------------------------------------------------------- #
@dataclass
class PaperConfig:
    starting_cash: float = 1000.0
    multiplier: int = 100                 # option contract multiplier
    max_open_positions: int = 1           # 0DTE: one structure at a time by default
    # Risk fractions apply to CURRENT equity, not starting cash: the budget
    # compounds as the account grows and shrinks in a drawdown. Anchoring to
    # starting_cash both caps the upside path and keeps betting $500 when the
    # account is down to $600.
    risk_per_trade_frac: float = 0.50     # max fraction of equity risked (max-loss) per trade
    daily_loss_limit_frac: float = 0.50   # stop opening once down this fraction of day-start equity

    # --- signal-churn guards: one position per regime, not fifty ---
    # A persistent regime re-emits the same TRADE ticket every tick; without a
    # cooldown the broker exits on target/stop and re-enters the same structure
    # sixty seconds later, all day.
    reentry_cooldown_min: float = 15.0    # no new entry within this of ANY exit
    stop_cooldown_min: float = 30.0       # ... doubled cool-off after a stop-loss exit
    max_trades_per_day: int = 10          # hard cap on entries per session

    # Conviction scales size: the gate already maps its score to a Kelly
    # fraction (score_floor -> kelly_frac_min ... 100 -> kelly_frac_max), but
    # the broker used to ignore it and deploy the FULL risk budget on any
    # passing signal — a 14/100 score at 9:33 sized identically to an 85 at
    # 11:00. With this on, that 9:33 signal is a 1-lot probe, not half the
    # account.
    use_gate_kelly: bool = True

    # --- exits (fractions of the trade's own max profit / max loss) ---
    stop_loss_frac: float = 0.60          # exit when loss >= this * defined max loss
    profit_target_frac: float = 0.60      # exit when profit >= this * max profit
    trailing_arm_frac: float = 0.35       # arm the trailing stop after capturing this * max profit
    trailing_giveback_frac: float = 0.40  # once armed, exit if peak profit gives back this * max profit
    eod_close_et: tuple = (15, 55)        # force-close all positions at/after this ET time (0DTE)

    slippage_frac: float = 0.50           # fraction of each leg's bid-ask half-spread paid per side


# --------------------------------------------------------------------------- #
# Position                                                                      #
# --------------------------------------------------------------------------- #
@dataclass
class PaperPosition:
    id: str
    opened_at: dt.datetime
    family: str
    legs: tuple
    contracts: int
    entry_credit: float          # per share, AFTER entry slippage (negative = debit paid)
    max_profit_ps: float         # per share (> 0)
    max_loss_ps: float           # per share (> 0)
    short_strikes: tuple = ()
    long_strikes: tuple = ()
    peak_pnl_ps: float = 0.0     # best (highest) per-share P&L seen
    trailing_armed: bool = False
    last_pnl_ps: float = 0.0
    entry_ctx: dict = field(default_factory=dict)   # why we entered (regime/gate/EV)

    def strikes_str(self) -> str:
        s = "/".join(f"{lg.strike:g}{lg.kind}{'+' if lg.qty > 0 else '-'}" for lg in self.legs)
        return s


# --------------------------------------------------------------------------- #
# Broker                                                                        #
# --------------------------------------------------------------------------- #
class PaperBroker:
    """Virtual account that auto-executes TRADE tickets on simulated fills."""

    def __init__(self, db_path: str = "paper.sqlite", cfg: Optional[PaperConfig] = None,
                 notifier=None, symbol: str = "SPY") -> None:
        self.cfg = cfg or PaperConfig()
        self.symbol = symbol
        self._notifier = notifier
        self.open_positions: list[PaperPosition] = []
        self._db = sqlite3.connect(db_path)
        self._init_db()
        # Realized equity (starting + closed P&L). Open positions are in-memory
        # only and cannot survive a process restart -- but closed-trade history
        # is already persisted, so resume from the last recorded equity instead
        # of silently resetting the account to starting_cash on every restart.
        self.cash = self._restore_equity()
        self._day_realized: dict[str, float] = {}   # ET date -> realized $ that day
        self._day_start_cash: dict[str, float] = {} # ET date -> equity at first tick
        self._day_entries: dict[str, int] = {}      # ET date -> entries opened
        self._last_exit_at: Optional[dt.datetime] = None
        self._last_exit_reason: str = ""

    def _restore_equity(self) -> float:
        row = self._db.execute(
            "SELECT equity_after FROM paper_trades ORDER BY closed_at DESC LIMIT 1"
        ).fetchone()
        return row[0] if row and row[0] is not None else self.cfg.starting_cash

    # -- persistence --------------------------------------------------------
    def _init_db(self) -> None:
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS paper_trades (
                id TEXT PRIMARY KEY,
                symbol TEXT, family TEXT, strikes TEXT, contracts INTEGER,
                opened_at TEXT, closed_at TEXT, hold_min REAL,
                entry_credit REAL, exit_value REAL,
                max_profit_ps REAL, max_loss_ps REAL,
                pnl_ps REAL, pnl_dollars REAL, exit_reason TEXT,
                equity_after REAL
            )""")
        # migration: entry_ctx (why the trade was taken) for the journal view
        cols = {r[1] for r in self._db.execute("PRAGMA table_info(paper_trades)")}
        if "entry_ctx" not in cols:
            self._db.execute("ALTER TABLE paper_trades ADD COLUMN entry_ctx TEXT")
        self._db.commit()

    def _record(self, pos: PaperPosition, now: dt.datetime, credit_now: float,
                pnl_ps: float, pnl_dollars: float, reason: str) -> None:
        hold_min = (now - pos.opened_at).total_seconds() / 60.0
        self._db.execute(
            "INSERT OR REPLACE INTO paper_trades "
            "(id, symbol, family, strikes, contracts, opened_at, closed_at, "
            " hold_min, entry_credit, exit_value, max_profit_ps, max_loss_ps, "
            " pnl_ps, pnl_dollars, exit_reason, equity_after, entry_ctx) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (pos.id, self.symbol, pos.family, pos.strikes_str(), pos.contracts,
             pos.opened_at.isoformat(), now.isoformat(), round(hold_min, 1),
             round(pos.entry_credit, 4), round(credit_now, 4),
             round(pos.max_profit_ps, 4), round(pos.max_loss_ps, 4),
             round(pnl_ps, 4), round(pnl_dollars, 2), reason, round(self.cash, 2),
             json.dumps(pos.entry_ctx) if pos.entry_ctx else None),
        )
        self._db.commit()

    # -- public API ---------------------------------------------------------
    @property
    def equity(self) -> float:
        """Realized cash + unrealized mark of open positions (set by last mark)."""
        unreal = sum(p.last_pnl_ps * self.cfg.multiplier * p.contracts
                     for p in self.open_positions)
        return self.cash + unreal

    def on_tick(self, now: dt.datetime, result) -> list[str]:
        """Drive the broker for one tick from a unified_loop TickResult. Returns a
        list of human-readable event strings (entries/exits) for logging."""
        events: list[str] = []
        snap = getattr(result, "snapshot", None)
        chain = getattr(snap, "chain", None) if snap is not None else None
        if chain is None:
            return events                                   # no marks possible this tick

        cmid, pmid, spr = _chain_maps(chain)

        # 1) manage / exit open positions (iterate over a copy; we mutate the list)
        for pos in list(self.open_positions):
            ev = self._manage(pos, now, cmid, pmid, spr)
            if ev:
                events.append(ev)

        # 2) maybe open a new position from a TRADE decision
        ev = self._maybe_open(now, result, chain, cmid, pmid, spr)
        if ev:
            events.append(ev)

        return events

    # -- entry --------------------------------------------------------------
    def _maybe_open(self, now, result, chain, cmid, pmid, spr) -> Optional[str]:
        dec = getattr(result, "decision", None)
        if dec is None or getattr(dec, "decision", None) != "TRADE" or not getattr(dec, "gate_pass", False):
            return None
        cand = getattr(dec, "candidate", None)
        if cand is None:
            return None
        if len(self.open_positions) >= self.cfg.max_open_positions:
            return None

        date = now.astimezone(ET).date().isoformat()
        day_start = self._day_start_cash.setdefault(date, self.cash)
        if self._day_realized.get(date, 0.0) <= -self.cfg.daily_loss_limit_frac * day_start:
            return None                                     # daily loss limit hit; stand down
        if self._day_entries.get(date, 0) >= self.cfg.max_trades_per_day:
            return None                                     # enough for one session

        # Re-entry cooldown: a regime that persists re-emits the same ticket
        # every tick; one position per regime, not one per minute.
        if self._last_exit_at is not None:
            cool = (self.cfg.stop_cooldown_min if self._last_exit_reason == "stop"
                    else self.cfg.reentry_cooldown_min)
            since = (now - self._last_exit_at).total_seconds() / 60.0
            if since < cool:
                return None

        legs = tuple(cand.legs)
        slip_entry = self._slippage_ps(legs, spr)
        entry_credit = float(cand.credit) - slip_entry      # worse than mid by slippage

        mp, ml = self._payoff_extents(legs, chain.spot, entry_credit)
        if ml <= 0:
            return None                                     # degenerate / no defined risk

        # size against CURRENT equity, the regime size multiplier, and the
        # gate's conviction (score -> Kelly fraction)
        size_mult = float(getattr(result, "final_size_mult", 1.0)) or 1.0
        kelly = 1.0
        if self.cfg.use_gate_kelly:
            k = getattr(dec, "gate_kelly", None)
            if isinstance(k, (int, float)) and 0.0 < k <= 1.0:
                kelly = float(k)
        risk_budget = self.cash * self.cfg.risk_per_trade_frac
        per_contract_risk = ml * self.cfg.multiplier
        contracts = int(np.floor((risk_budget * size_mult * kelly) / per_contract_risk))
        # never risk more than the cash on hand
        contracts = min(contracts, int(np.floor(self.cash / per_contract_risk)))
        if contracts < 1:
            return None                                     # can't afford even one lot

        intent = getattr(result, "intent", None)
        regime = getattr(result, "regime", None)
        entry_ctx = {
            "regime": getattr(regime, "dominant_regime", None),
            "engine": getattr(regime, "permitted_engine", None),
            "cell": ([intent.exec_regime, intent.context_regime, intent.direction_bias]
                     if intent is not None else None),
            "direction": (intent.decision.direction if intent is not None else None),
            "conviction": (intent.decision.conviction if intent is not None else None),
            "capture": (intent.decision.capture if intent is not None else None),
            "gate_score": getattr(dec, "gate_score", None),
            "ev": getattr(cand, "ev", None),
            "ev_per_risk": getattr(cand, "ev_per_risk", None),
            "prob_profit": getattr(cand, "prob_profit", None),
            "credit_mid": getattr(cand, "credit", None),
            "size_mult": size_mult,
            "gate_kelly": kelly,
            "spot": getattr(chain, "spot", None),
            "risk_budget": round(risk_budget, 2),
            "equity_at_entry": round(self.cash, 2),
        }

        pos = PaperPosition(
            id=uuid.uuid4().hex[:12], opened_at=now, family=cand.family, legs=legs,
            contracts=contracts, entry_credit=entry_credit, max_profit_ps=mp, max_loss_ps=ml,
            short_strikes=tuple(cand.short_strikes), long_strikes=tuple(cand.long_strikes),
            entry_ctx=entry_ctx,
        )
        self.open_positions.append(pos)
        self._day_entries[date] = self._day_entries.get(date, 0) + 1
        self._notify("PAPER ENTRY",
                     f"{pos.family} {pos.strikes_str()} x{contracts} "
                     f"entry={entry_credit:+.2f} maxP={mp:.2f} maxL={ml:.2f}")
        return (f"PAPER ENTRY {pos.family} {pos.strikes_str()} x{contracts} "
                f"entry={entry_credit:+.2f}")

    # -- management / exit --------------------------------------------------
    def _manage(self, pos, now, cmid, pmid, spr) -> Optional[str]:
        credit_now = _credit(pos.legs, cmid, pmid)
        if credit_now is None:
            return None                                     # a leg has no quote; hold
        pnl_ps = pos.entry_credit - credit_now              # per share, gross of exit slippage
        pos.last_pnl_ps = pnl_ps
        pos.peak_pnl_ps = max(pos.peak_pnl_ps, pnl_ps)
        if not pos.trailing_armed and pos.peak_pnl_ps >= self.cfg.trailing_arm_frac * pos.max_profit_ps:
            pos.trailing_armed = True

        reason = self._exit_reason(pos, now, pnl_ps)
        if reason is None:
            return None

        slip_exit = self._slippage_ps(pos.legs, spr)
        net_ps = pnl_ps - slip_exit
        pnl_dollars = net_ps * self.cfg.multiplier * pos.contracts
        self.cash += pnl_dollars
        date = now.astimezone(ET).date().isoformat()
        self._day_realized[date] = self._day_realized.get(date, 0.0) + pnl_dollars

        self.open_positions.remove(pos)
        self._last_exit_at = now
        self._last_exit_reason = reason
        self._record(pos, now, credit_now, net_ps, pnl_dollars, reason)
        self._notify("PAPER EXIT",
                     f"{pos.family} {pos.strikes_str()} {reason} "
                     f"pnl=${pnl_dollars:+.2f} equity=${self.cash:.2f}")
        return (f"PAPER EXIT {pos.family} {reason} pnl=${pnl_dollars:+.2f} "
                f"equity=${self.cash:.2f}")

    def _exit_reason(self, pos, now, pnl_ps) -> Optional[str]:
        cfg = self.cfg
        if pnl_ps <= -cfg.stop_loss_frac * pos.max_loss_ps:
            return "stop"
        if pnl_ps >= cfg.profit_target_frac * pos.max_profit_ps:
            return "target"
        if pos.trailing_armed and (pos.peak_pnl_ps - pnl_ps) >= cfg.trailing_giveback_frac * pos.max_profit_ps:
            return "trail"
        et = now.astimezone(ET)
        if (et.hour, et.minute) >= cfg.eod_close_et:
            return "eod"
        return None

    # -- helpers ------------------------------------------------------------
    def _slippage_ps(self, legs, spr) -> float:
        """Per-share slippage = slippage_frac × sum of each leg's half bid-ask spread."""
        total = 0.0
        for lg in legs:
            s = spr.get(lg.strike)
            if not s:
                continue
            half = (s[0] if lg.kind == "C" else s[1]) * 0.5
            total += half
        return self.cfg.slippage_frac * total

    @staticmethod
    def _payoff_extents(legs, spot, entry_credit) -> tuple[float, float]:
        """Max profit and max loss per share from the expiry payoff curve."""
        lo, hi = max(spot * 0.80, 1.0), spot * 1.20
        grid = np.linspace(lo, hi, 4001)
        curve = _payoff_curve(legs, grid, entry_credit)
        return float(curve.max()), float(-curve.min())

    def _notify(self, title: str, body: str) -> None:
        if self._notifier is None:
            return
        try:
            self._notifier.send_text(title, body)           # optional; ignored if unsupported
        except Exception:
            pass

    # -- reporting ----------------------------------------------------------
    def _open_position_view(self, pos: PaperPosition, now: dt.datetime) -> dict:
        unrealized_dollars = pos.last_pnl_ps * self.cfg.multiplier * pos.contracts
        return {
            "id": pos.id,
            "family": pos.family,
            "strikes": pos.strikes_str(),
            "contracts": pos.contracts,
            "opened_at": pos.opened_at.isoformat(),
            "hold_min": round((now - pos.opened_at).total_seconds() / 60.0, 1),
            "entry_credit": round(pos.entry_credit, 4),
            "max_profit_ps": round(pos.max_profit_ps, 4),
            "max_loss_ps": round(pos.max_loss_ps, 4),
            "unrealized_pnl_ps": round(pos.last_pnl_ps, 4),
            "unrealized_pnl_dollars": round(unrealized_dollars, 2),
            "pct_of_max_profit": (round(pos.last_pnl_ps / pos.max_profit_ps, 4)
                                   if pos.max_profit_ps else None),
            "entry_ctx": pos.entry_ctx or None,
        }

    def report(self, now: Optional[dt.datetime] = None) -> dict:
        now = now or dt.datetime.now(ET)
        rows = list(self._db.execute(
            "SELECT pnl_dollars, exit_reason, equity_after FROM paper_trades ORDER BY closed_at"))
        n = len(rows)
        pnls = [r[0] for r in rows]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        gross_win = sum(wins)
        gross_loss = -sum(losses)
        equity_curve = [self.cfg.starting_cash] + [r[2] for r in rows]
        peak = equity_curve[0]
        max_dd = 0.0
        for e in equity_curve:
            peak = max(peak, e)
            max_dd = max(max_dd, peak - e)
        by_reason: dict[str, int] = {}
        for r in rows:
            by_reason[r[1]] = by_reason.get(r[1], 0) + 1
        return {
            "trades": n,
            "win_rate": (len(wins) / n) if n else 0.0,
            "total_pnl": round(sum(pnls), 2),
            "avg_win": round(gross_win / len(wins), 2) if wins else 0.0,
            "avg_loss": round(-gross_loss / len(losses), 2) if losses else 0.0,
            "profit_factor": round(gross_win / gross_loss, 2) if gross_loss > 0 else None,
            "max_drawdown": round(max_dd, 2),
            "equity": round(equity_curve[-1], 2),
            "open_positions": len(self.open_positions),
            "open": [self._open_position_view(p, now) for p in self.open_positions],
            "by_exit_reason": by_reason,
        }

    def print_report(self) -> None:
        r = self.report()
        print("\n" + "=" * 56)
        print("  Paper Trading Report  (simulated — no real orders)")
        print("=" * 56)
        print(f"  Starting cash:   ${self.cfg.starting_cash:,.2f}")
        print(f"  Equity now:      ${r['equity']:,.2f}   (open: {r['open_positions']})")
        print(f"  Closed trades:   {r['trades']}   win rate: {r['win_rate']*100:.1f}%")
        print(f"  Total P&L:       ${r['total_pnl']:+,.2f}")
        print(f"  Avg win/loss:    ${r['avg_win']:+.2f} / ${r['avg_loss']:+.2f}"
              f"   profit factor: {r['profit_factor']}")
        print(f"  Max drawdown:    ${r['max_drawdown']:,.2f}")
        print(f"  Exits:           {r['by_exit_reason']}")
        print("=" * 56)
