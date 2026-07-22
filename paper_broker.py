"""
paper_broker.py  —  in-house paper trading: simulated auto-execution on LIVE data.

WHAT THIS IS
    Shadow mode journals every *evaluation* but never holds a position. This adds
    a virtual broker on top of it: when the pipeline emits a TRADE ticket, the
    broker "fills" it against the live option chain, then marks the position to
    market every tick and exits it on a stop-loss, profit-target, trailing-stop,
    or end-of-day rule. Every closed trade is journaled with realized P&L and the
    exit reason. NO REAL ORDERS ARE PLACED — it tracks virtual cash accounts
    (default $1000 per track) so strategies can prove themselves before any real
    money.

    Parallel tracks (legacy / v2 / v3): each prediction engine can open its own
    paper position with an independent cash ledger. Journal `entry_ctx.fill_track`
    tags which engine filled the trade for side-by-side comparison.

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
import logging
import sqlite3
import uuid
from dataclasses import dataclass, field
from typing import Optional
from zoneinfo import ZoneInfo

import numpy as np

from spread_selector import Leg, _chain_maps, _credit, _payoff_curve
from regime_alignment import (
    build_entry_snapshot, derive_position_bias, entry_snapshot_to_dict,
    structure_class_from_family,
)
from risk_manager import PositionMonitor

ET = ZoneInfo("America/New_York")
log = logging.getLogger("paper_broker")

# Parallel paper tracks — each gets its own virtual cash ledger so legacy /
# V2 prediction-policy / V3 Part-3 / SPY-DER AI decisions can be compared
# side-by-side on the dashboard.
PAPER_TRACKS = ("legacy", "v2", "v3", "spy_der")
PAPER_TRACK_LABELS = {
    "legacy": "Legacy",
    "v2": "V2",
    "v3": "V3",
    "spy_der": "SPY-DER",
}


# --------------------------------------------------------------------------- #
# Config                                                                        #
# --------------------------------------------------------------------------- #
@dataclass
class PaperConfig:
    starting_cash: float = 1000.0
    multiplier: int = 100                 # option contract multiplier
    max_open_positions: int = 1           # per-track cap (0DTE: one structure each)
    # When True (default), open independent positions for each paper_intent
    # track on the tick (legacy / v2 / v3 / spy_der). When False, only the
    # legacy authoritative TradeDecision is filled.
    parallel_tracks: bool = True
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

    # --- exits ---
    stop_loss_frac: float = 0.60          # exit when loss >= this * defined max loss
    profit_target_frac: float = 0.60      # exit when profit >= this * max profit
    eod_close_et: tuple = (15, 55)        # force-close all positions at/after this ET time (0DTE)

    # --- trailing stop (peak-relative) ---
    # The old trail measured giveback as a fraction of MAX PROFIT: a trade that
    # armed at 35% and peaked at 38% could ride back to -2% before "trailing"
    # out — a losing exit from a winning trade. And far-OTM debit spreads
    # (tiny max loss, huge max profit) never reached the arm threshold at all,
    # so they had no protection until the hard stop. New rules:
    #   arm    when peak >= min(arm_frac * max_profit, arm_R * max_loss)
    #   floor  = peak * (1 - giveback_frac)      <- giveback is OF THE PEAK
    #   floor tightens once peak >= tighten_at * max_profit
    #   armed trades never exit below ~breakeven (entry slippage as the proxy
    #   for exit slippage), so an armed winner cannot become a loser.
    trailing_arm_frac: float = 0.35       # arm at this fraction of max profit ...
    trailing_arm_R: float = 0.75          # ... OR this multiple of max loss, whichever is FIRST
    trailing_giveback_frac: float = 0.40  # give back at most this fraction of the peak
    trailing_tighten_at: float = 0.60     # peak >= this * max profit ->
    trailing_tight_giveback: float = 0.25 # ... only this fraction of peak may be given back
    trailing_lock_breakeven: bool = True  # armed trades floor at ~breakeven

    slippage_frac: float = 0.50           # fraction of each leg's bid-ask half-spread paid per side

    # --- regime alignment (RAS) ---
    # True: an "exit" action from the position monitor closes the paper
    # position (exit reason "ras_invalidate"). Mirrors RASConfig.exit_enabled;
    # BOTH must be on for an automated exit. shadow_runner --no-ras-exit
    # turns both off for observation-only sessions.
    ras_exit_enabled: bool = True


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
    slip_entry_ps: float = 0.0   # entry slippage; proxy for exit cost in the breakeven floor
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
                 notifier=None, symbol: str = "SPY",
                 position_monitor: Optional[PositionMonitor] = None) -> None:
        self.cfg = cfg or PaperConfig()
        self.symbol = symbol
        self._notifier = notifier
        self.position_monitor = position_monitor or PositionMonitor()
        self.open_positions: list[PaperPosition] = []
        self._db = sqlite3.connect(db_path)
        self._init_db()
        # Per-track realized cash (starting + closed P&L). Open positions are
        # in-memory only and cannot survive a process restart — closed-trade
        # history is persisted, so each track resumes from its last equity.
        self.ledgers: dict[str, float] = self._restore_ledgers()
        self._day_realized: dict[str, float] = {}   # f"{date}|{track}" -> $
        self._day_start_cash: dict[str, float] = {} # f"{date}|{track}" -> equity
        self._day_entries: dict[str, int] = {}      # f"{date}|{track}" -> n
        self._last_exit_at: dict[str, Optional[dt.datetime]] = {
            t: None for t in PAPER_TRACKS}
        self._last_exit_reason: dict[str, str] = {t: "" for t in PAPER_TRACKS}
        self._ras_trail_mult: float = 1.0   # per-tick trailing giveback multiplier

    @property
    def cash(self) -> float:
        """Backward-compat: legacy-track cash (pre-triple-paper single ledger)."""
        return float(self.ledgers.get("legacy", self.cfg.starting_cash))

    @cash.setter
    def cash(self, value: float) -> None:
        self.ledgers["legacy"] = float(value)

    def _restore_ledgers(self) -> dict[str, float]:
        ledgers = {t: float(self.cfg.starting_cash) for t in PAPER_TRACKS}
        try:
            rows = self._db.execute(
                "SELECT equity_after, entry_ctx FROM paper_trades "
                "ORDER BY closed_at"
            ).fetchall()
        except sqlite3.Error:
            return ledgers
        for equity_after, ctx_json in rows:
            track = "legacy"
            if ctx_json:
                try:
                    ctx = json.loads(ctx_json)
                    t = str(ctx.get("fill_track") or "legacy").lower()
                    if t in ledgers:
                        track = t
                except (json.JSONDecodeError, TypeError, AttributeError):
                    track = "legacy"
            if equity_after is not None:
                ledgers[track] = float(equity_after)
        return ledgers

    def _track_of(self, pos: PaperPosition) -> str:
        t = str((pos.entry_ctx or {}).get("fill_track") or "legacy").lower()
        return t if t in self.ledgers else "legacy"

    def _open_count(self, track: str) -> int:
        return sum(1 for p in self.open_positions if self._track_of(p) == track)

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
        # migrations: entry_ctx (why entered), peak_pnl_ps (trail-discipline audit)
        cols = {r[1] for r in self._db.execute("PRAGMA table_info(paper_trades)")}
        if "entry_ctx" not in cols:
            self._db.execute("ALTER TABLE paper_trades ADD COLUMN entry_ctx TEXT")
        if "peak_pnl_ps" not in cols:
            self._db.execute("ALTER TABLE paper_trades ADD COLUMN peak_pnl_ps REAL")
        self._db.commit()

    def _record(self, pos: PaperPosition, now: dt.datetime, credit_now: float,
                pnl_ps: float, pnl_dollars: float, reason: str) -> None:
        hold_min = (now - pos.opened_at).total_seconds() / 60.0
        self._db.execute(
            "INSERT OR REPLACE INTO paper_trades "
            "(id, symbol, family, strikes, contracts, opened_at, closed_at, "
            " hold_min, entry_credit, exit_value, max_profit_ps, max_loss_ps, "
            " pnl_ps, pnl_dollars, exit_reason, equity_after, entry_ctx, peak_pnl_ps) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (pos.id, self.symbol, pos.family, pos.strikes_str(), pos.contracts,
             pos.opened_at.isoformat(), now.isoformat(), round(hold_min, 1),
             round(pos.entry_credit, 4), round(credit_now, 4),
             round(pos.max_profit_ps, 4), round(pos.max_loss_ps, 4),
             round(pnl_ps, 4), round(pnl_dollars, 2), reason,
             round(self.ledgers[self._track_of(pos)], 2),
             json.dumps(pos.entry_ctx) if pos.entry_ctx else None,
             round(pos.peak_pnl_ps, 4)),
        )
        self._db.commit()

    # -- public API ---------------------------------------------------------
    @property
    def equity(self) -> float:
        """Sum of per-track realized cash + unrealized marks."""
        unreal = sum(p.last_pnl_ps * self.cfg.multiplier * p.contracts
                     for p in self.open_positions)
        return sum(self.ledgers.values()) + unreal

    def track_equity(self, track: str, *, include_unrealized: bool = True) -> float:
        cash = float(self.ledgers.get(track, self.cfg.starting_cash))
        if not include_unrealized:
            return cash
        unreal = sum(
            p.last_pnl_ps * self.cfg.multiplier * p.contracts
            for p in self.open_positions if self._track_of(p) == track)
        return cash + unreal

    def on_tick(self, now: dt.datetime, result) -> list[str]:
        """Drive the broker for one tick from a unified_loop TickResult. Returns a
        list of human-readable event strings (entries/exits) for logging."""
        events: list[str] = []
        snap = getattr(result, "snapshot", None)
        chain = getattr(snap, "chain", None) if snap is not None else None

        # Even without a chain we must still honor RAS exits using the last
        # marked P&L — otherwise feed gaps leave positions unprotected.
        if chain is None:
            for pos in list(self.open_positions):
                ev = self._manage_ras_only(pos, now, result=result)
                if ev:
                    events.append(ev)
            return events

        cmid, pmid, spr = _chain_maps(chain)

        # 1) manage / exit open positions (iterate over a copy; we mutate the list)
        for pos in list(self.open_positions):
            ev = self._manage(pos, now, cmid, pmid, spr, result=result)
            if ev:
                events.append(ev)

        # 2) maybe open new positions — parallel tracks when paper_intents set
        intents = list(getattr(result, "paper_intents", None) or [])
        if intents and self.cfg.parallel_tracks:
            for intent in intents:
                ev = self._maybe_open_intent(now, result, intent, chain, cmid, pmid, spr)
                if ev:
                    events.append(ev)
        else:
            ev = self._maybe_open(now, result, chain, cmid, pmid, spr)
            if ev:
                events.append(ev)

        return events

    def _manage_ras_only(self, pos, now, result=None) -> Optional[str]:
        """Apply RAS exit/tighten without fresh chain quotes."""
        if result is None:
            return None
        ras_list = getattr(result, "ras_results", None) or []
        ras = next((r for r in ras_list if r.position_id == pos.id), None)
        if ras is None:
            return None
        action = self.position_monitor.evaluate(ras)
        pos.entry_ctx["ras_score"] = ras.score
        pos.entry_ctx["ras_action"] = ras.action
        pos.entry_ctx["ras_ema_score"] = ras.ema_score
        if action.action == "exit" and self.cfg.ras_exit_enabled:
            # Reconstruct a mark from the last known P&L.
            pnl_ps = float(pos.last_pnl_ps)
            credit_now = pos.entry_credit - pnl_ps
            net_ps = pnl_ps
            pnl_dollars = net_ps * self.cfg.multiplier * pos.contracts
            track = self._track_of(pos)
            self.ledgers[track] = self.ledgers.get(track, self.cfg.starting_cash) + pnl_dollars
            date = now.astimezone(ET).date().isoformat()
            key = f"{date}|{track}"
            self._day_realized[key] = self._day_realized.get(key, 0.0) + pnl_dollars
            self.open_positions.remove(pos)
            self.position_monitor.release(pos.id)
            pos.entry_ctx["ras_at_exit"] = ras.score
            pos.entry_ctx["ras_exit_note"] = "chain_unavailable"
            self._last_exit_at[track] = now
            self._last_exit_reason[track] = "ras_invalidate"
            self._record(pos, now, credit_now, net_ps, pnl_dollars, "ras_invalidate")
            self._notify("PAPER EXIT",
                         f"[{track}] {pos.family} {pos.strikes_str()} ras_invalidate "
                         f"(no chain) pnl=${pnl_dollars:+.2f}")
            return (f"PAPER EXIT [{track}] {pos.family} ras_invalidate (no chain) "
                    f"pnl=${pnl_dollars:+.2f}")
        return None

    # -- entry --------------------------------------------------------------
    def _maybe_open(self, now, result, chain, cmid, pmid, spr) -> Optional[str]:
        """Legacy single-path open (used when paper_intents absent / parallel off)."""
        dec = getattr(result, "decision", None)
        if dec is None or getattr(dec, "decision", None) != "TRADE" or not getattr(dec, "gate_pass", False):
            return None
        cand = getattr(dec, "candidate", None)
        if cand is None:
            return None
        sig = getattr(result, "signals", None) or {}
        mode_l = str(sig.get("policy_mode") or "").lower()
        # Pre-triple-paper: champion tagged the single fill as v2.
        track = "v2" if mode_l == "champion" else "legacy"
        return self._open_candidate(
            now, result, cand, chain, cmid, pmid, spr,
            track=track,
            size_mult=float(getattr(result, "final_size_mult", 1.0) or 1.0),
            gate_kelly=getattr(dec, "gate_kelly", None),
            gate_score=getattr(dec, "gate_score", None),
        )

    def _maybe_open_intent(self, now, result, intent, chain, cmid, pmid, spr) -> Optional[str]:
        """Open from a parallel paper_intent dict ({track, candidate, ...})."""
        if not isinstance(intent, dict):
            return None
        track = str(intent.get("track") or "").lower()
        if track not in PAPER_TRACKS:
            return None
        cand = intent.get("candidate")
        if cand is None:
            return None
        return self._open_candidate(
            now, result, cand, chain, cmid, pmid, spr,
            track=track,
            size_mult=float(intent.get("size_mult") or 1.0),
            gate_kelly=intent.get("gate_kelly"),
            gate_score=intent.get("gate_score"),
            intent_meta=intent,
        )

    def _open_candidate(
            self, now, result, cand, chain, cmid, pmid, spr, *,
            track: str, size_mult: float = 1.0,
            gate_kelly=None, gate_score=None,
            intent_meta: Optional[dict] = None) -> Optional[str]:
        # Belt-and-suspenders: never open during session warmup even if a
        # misconfigured gate somehow passed. GateConfig.morning_entry_time is
        # the single source of truth (default 10:00 ET = 30m after the open).
        try:
            from gate_scorer import GateConfig
            entry_open = GateConfig().morning_entry_time
            sig = getattr(result, "signals", None) or {}
            if float(sig.get("session_warmup") or 0.0) >= 1.0:
                log.info("paper entry suppressed [%s]: session_warmup", track)
                return None
            if now.astimezone(ET).time() < entry_open:
                log.info("paper entry suppressed [%s]: before morning_entry_time %s",
                         track, entry_open.strftime("%H:%M"))
                return None
        except Exception:
            pass
        if self._open_count(track) >= self.cfg.max_open_positions:
            return None

        date = now.astimezone(ET).date().isoformat()
        key = f"{date}|{track}"
        track_cash = float(self.ledgers.get(track, self.cfg.starting_cash))
        day_start = self._day_start_cash.setdefault(key, track_cash)
        if self._day_realized.get(key, 0.0) <= -self.cfg.daily_loss_limit_frac * day_start:
            return None
        if self._day_entries.get(key, 0) >= self.cfg.max_trades_per_day:
            return None

        last_exit = self._last_exit_at.get(track)
        if last_exit is not None:
            cool = (self.cfg.stop_cooldown_min
                    if self._last_exit_reason.get(track) == "stop"
                    else self.cfg.reentry_cooldown_min)
            since = (now - last_exit).total_seconds() / 60.0
            if since < cool:
                return None

        legs = tuple(cand.legs)
        slip_entry = self._slippage_ps(legs, spr)
        entry_credit = float(cand.credit) - slip_entry

        mp, ml = self._payoff_extents(legs, chain.spot, entry_credit)
        if ml <= 0:
            return None

        kelly = 1.0
        if self.cfg.use_gate_kelly:
            k = gate_kelly
            if isinstance(k, (int, float)) and 0.0 < k <= 1.0:
                kelly = float(k)
        risk_budget = track_cash * self.cfg.risk_per_trade_frac
        per_contract_risk = ml * self.cfg.multiplier
        contracts = int(np.floor((risk_budget * size_mult * kelly) / per_contract_risk))
        contracts = min(contracts, int(np.floor(track_cash / per_contract_risk)))
        if contracts < 1:
            return None

        intent = getattr(result, "intent", None)
        regime = getattr(result, "regime", None)
        intent_dec = getattr(intent, "decision", None) if intent is not None else None
        structure = getattr(intent_dec, "structure", None) if intent_dec else None
        direction = getattr(intent_dec, "direction", None) if intent_dec else None
        meta = intent_meta or {}
        if meta.get("structure"):
            structure = meta.get("structure")
        if meta.get("direction"):
            direction = meta.get("direction")
        structure_class = structure_class_from_family(cand.family)
        position_bias = derive_position_bias(
            direction or "none", structure or "", structure_class)
        entry_snapshot = None
        if regime is not None and intent is not None:
            market = getattr(getattr(result, "snapshot", None), "market", None)
            if (market is not None
                    and hasattr(regime, "confidences")
                    and hasattr(intent, "exec_regime")):
                try:
                    entry_snapshot = build_entry_snapshot(
                        regime, intent, market, structure_class, structure=structure)
                except Exception as exc:
                    log.warning("entry_snapshot build failed: %s", exc)
                    entry_snapshot = None
            else:
                log.warning("entry_snapshot skipped: market/regime/intent incomplete")
        if entry_snapshot is None:
            log.warning("opening paper position WITHOUT entry_snapshot — "
                        "RAS will not monitor this trade")
        sig = getattr(result, "signals", None) or {}
        policy_mode = sig.get("policy_mode")
        policy_source = sig.get("policy_source")
        try:
            disagree = float(sig.get("policy_disagreement") or 0.0) >= 1.0
        except (TypeError, ValueError):
            disagree = False

        entry_ctx = {
            "regime": getattr(regime, "dominant_regime", None),
            "engine": getattr(regime, "permitted_engine", None),
            "cell": ([intent.exec_regime, intent.context_regime, intent.direction_bias]
                     if intent is not None else None),
            "direction": direction,
            "structure": structure,
            "structure_class": structure_class,
            "position_bias": position_bias,
            "conviction": (intent.decision.conviction if intent is not None else None),
            "capture": (intent.decision.capture if intent is not None else None),
            "gate_score": gate_score,
            "ev": getattr(cand, "ev", None),
            "ev_per_risk": getattr(cand, "ev_per_risk", None),
            "prob_profit": getattr(cand, "prob_profit", None),
            "credit_mid": getattr(cand, "credit", None),
            "size_mult": size_mult,
            "gate_kelly": kelly,
            "spot": getattr(chain, "spot", None),
            "risk_budget": round(risk_budget, 2),
            "equity_at_entry": round(track_cash, 2),
            "entry_snapshot": (entry_snapshot_to_dict(entry_snapshot)
                               if entry_snapshot is not None else None),
            "ras_at_entry": None,
            "fill_track": track,
            "policy_mode": policy_mode,
            "policy_source": policy_source,
            "policy_disagreement": 1.0 if disagree else 0.0,
            "legacy_structure": (sig.get("legacy_policy_structure")
                                 or sig.get("policy_structure")
                                 or structure),
            "v2_policy_structure": sig.get("v2_policy_structure"),
            "v2_policy_action": sig.get("v2_policy_action"),
            "v2_policy_direction": sig.get("v2_policy_direction"),
            "v2_policy_confidence": sig.get("v2_policy_confidence"),
            "v3_action": meta.get("v3_action") or (
                (getattr(result, "part3", None) or {})
                .get("decision_summary", {}) or {}).get("action"),
            "v3_selected_candidate_id": meta.get("candidate_id"),
            "intent_reason": meta.get("reason"),
            # SPY-DER AI provenance — present only on the spy_der track so the
            # dashboard can show the agent's action and confidence per trade.
            "spy_der_action": meta.get("spy_der_action"),
            "spy_der_confidence": meta.get("confidence"),
            "spy_der_uncertainty": meta.get("uncertainty"),
        }

        pos = PaperPosition(
            id=uuid.uuid4().hex[:12], opened_at=now, family=cand.family, legs=legs,
            contracts=contracts, entry_credit=entry_credit, max_profit_ps=mp, max_loss_ps=ml,
            short_strikes=tuple(cand.short_strikes), long_strikes=tuple(cand.long_strikes),
            slip_entry_ps=slip_entry, entry_ctx=entry_ctx,
        )
        self.open_positions.append(pos)
        self.position_monitor.register(pos.id, entry_ctx)
        self._day_entries[key] = self._day_entries.get(key, 0) + 1
        self._notify("PAPER ENTRY",
                     f"[{track}] {pos.family} {pos.strikes_str()} x{contracts} "
                     f"entry={entry_credit:+.2f} maxP={mp:.2f} maxL={ml:.2f}")
        return (f"PAPER ENTRY [{track}] {pos.family} {pos.strikes_str()} x{contracts} "
                f"entry={entry_credit:+.2f}")

    # -- management / exit --------------------------------------------------
    def _manage(self, pos, now, cmid, pmid, spr, result=None) -> Optional[str]:
        credit_now = _credit(pos.legs, cmid, pmid)
        if credit_now is None:
            return None                                     # a leg has no quote; hold
        pnl_ps = pos.entry_credit - credit_now              # per share, gross of exit slippage
        pos.last_pnl_ps = pnl_ps
        pos.peak_pnl_ps = max(pos.peak_pnl_ps, pnl_ps)
        if not pos.trailing_armed:
            # arm on max-profit fraction OR R-multiple, whichever comes first —
            # the R path is what gives far-OTM debit spreads (tiny max loss,
            # huge max profit) any trailing protection at all
            arm_at = min(self.cfg.trailing_arm_frac * pos.max_profit_ps,
                         self.cfg.trailing_arm_R * pos.max_loss_ps)
            if pos.peak_pnl_ps >= arm_at > 0:
                pos.trailing_armed = True

        self._ras_trail_mult = 1.0
        ras_exit = False
        ras_event = None
        if result is not None:
            ras_list = getattr(result, "ras_results", None) or []
            ras = next((r for r in ras_list if r.position_id == pos.id), None)
            if ras is not None:
                action = self.position_monitor.evaluate(ras)
                if pos.entry_ctx.get("ras_at_entry") is None:
                    pos.entry_ctx["ras_at_entry"] = ras.score
                prev_worst = pos.entry_ctx.get("ras_worst")
                pos.entry_ctx["ras_worst"] = (ras.ema_score if prev_worst is None
                                              else min(prev_worst, ras.ema_score))
                pos.entry_ctx["ras_score"] = ras.score
                pos.entry_ctx["ras_action"] = ras.action
                pos.entry_ctx["ras_ema_score"] = ras.ema_score
                pos.entry_ctx["ras_components"] = [
                    {"name": c.name, "raw": c.raw, "note": c.note}
                    for c in ras.components
                ]
                if action.action == "warning":
                    pos.entry_ctx["ras_warning"] = action.reasons
                elif action.action == "tighten":
                    self._ras_trail_mult = 0.75
                elif action.action == "exit" and self.cfg.ras_exit_enabled:
                    ras_exit = True
                # Surface action transitions in the session log so escalation
                # is visible as it happens, not only at exit.
                if action.action != pos.entry_ctx.get("ras_last_action"):
                    pos.entry_ctx["ras_last_action"] = action.action
                    if action.action != "ok":
                        ras_event = (f"RAS {action.action.upper()} {pos.family} "
                                     f"{pos.strikes_str()} score={ras.ema_score:+.1f}")

        reason = self._exit_reason(pos, now, pnl_ps, ras_exit=ras_exit)
        if reason is None:
            return ras_event

        slip_exit = self._slippage_ps(pos.legs, spr)
        net_ps = pnl_ps - slip_exit
        pnl_dollars = net_ps * self.cfg.multiplier * pos.contracts
        track = self._track_of(pos)
        self.ledgers[track] = self.ledgers.get(track, self.cfg.starting_cash) + pnl_dollars
        date = now.astimezone(ET).date().isoformat()
        key = f"{date}|{track}"
        self._day_realized[key] = self._day_realized.get(key, 0.0) + pnl_dollars

        self.open_positions.remove(pos)
        self.position_monitor.release(pos.id)
        pos.entry_ctx["ras_at_exit"] = pos.entry_ctx.get("ras_score")
        self._last_exit_at[track] = now
        self._last_exit_reason[track] = reason
        self._record(pos, now, credit_now, net_ps, pnl_dollars, reason)
        eq = self.ledgers[track]
        self._notify("PAPER EXIT",
                     f"[{track}] {pos.family} {pos.strikes_str()} {reason} "
                     f"pnl=${pnl_dollars:+.2f} equity=${eq:.2f}")
        return (f"PAPER EXIT [{track}] {pos.family} {reason} pnl=${pnl_dollars:+.2f} "
                f"equity=${eq:.2f}")

    def _exit_reason(self, pos, now, pnl_ps, ras_exit: bool = False) -> Optional[str]:
        cfg = self.cfg
        if ras_exit:
            return "ras_invalidate"
        if pnl_ps <= -cfg.stop_loss_frac * pos.max_loss_ps:
            return "stop"
        if pnl_ps >= cfg.profit_target_frac * pos.max_profit_ps:
            return "target"
        if pos.trailing_armed and pnl_ps <= self._trail_floor(pos):
            return "trail"
        et = now.astimezone(ET)
        if (et.hour, et.minute) >= cfg.eod_close_et:
            return "eod"
        return None

    def _trail_floor(self, pos) -> float:
        """Per-share P&L level an ARMED position may not fall below.

        Peak-relative: keep at least (1 - giveback) of the best P&L seen,
        tightening once the peak is a real fraction of max profit. With the
        breakeven lock, the floor never sits below ~breakeven (entry slippage
        stands in for the exit slippage the close will pay), so an armed
        winner cannot round-trip into a loser — the exact failure mode of the
        old max-profit-denominated giveback.
        """
        cfg = self.cfg
        gb = (cfg.trailing_tight_giveback
              if pos.peak_pnl_ps >= cfg.trailing_tighten_at * pos.max_profit_ps
              else cfg.trailing_giveback_frac)
        gb *= getattr(self, "_ras_trail_mult", 1.0)
        floor = pos.peak_pnl_ps * (1.0 - gb)
        if cfg.trailing_lock_breakeven:
            floor = max(floor, pos.slip_entry_ps)
        return floor

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
            "SELECT pnl_dollars, exit_reason, equity_after, entry_ctx "
            "FROM paper_trades ORDER BY closed_at"))
        n = len(rows)
        pnls = [r[0] for r in rows]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        gross_win = sum(wins)
        gross_loss = -sum(losses)
        by_reason: dict[str, int] = {}
        by_track: dict[str, dict] = {
            t: {"trades": 0, "total_pnl": 0.0, "wins": 0,
                "equity": round(self.track_equity(t), 2),
                "open_positions": self._open_count(t)}
            for t in PAPER_TRACKS
        }
        for r in rows:
            by_reason[r[1]] = by_reason.get(r[1], 0) + 1
            track = "legacy"
            if r[3]:
                try:
                    ctx = json.loads(r[3])
                    t = str(ctx.get("fill_track") or "legacy").lower()
                    if t in by_track:
                        track = t
                except (json.JSONDecodeError, TypeError, AttributeError):
                    track = "legacy"
            by_track[track]["trades"] += 1
            by_track[track]["total_pnl"] = round(
                by_track[track]["total_pnl"] + float(r[0] or 0.0), 2)
            if float(r[0] or 0.0) > 0:
                by_track[track]["wins"] += 1
        for t, bt in by_track.items():
            nt = bt["trades"]
            bt["win_rate"] = (bt["wins"] / nt) if nt else 0.0
            bt["total_pnl"] = round(bt["total_pnl"], 2)
            bt.pop("wins", None)

        # Backward-compat equity curve uses legacy ledger (single-account era).
        legacy_equity_curve = [self.cfg.starting_cash]
        for r in rows:
            track = "legacy"
            if r[3]:
                try:
                    ctx = json.loads(r[3])
                    track = str(ctx.get("fill_track") or "legacy").lower()
                except (json.JSONDecodeError, TypeError, AttributeError):
                    track = "legacy"
            if track == "legacy" and r[2] is not None:
                legacy_equity_curve.append(float(r[2]))
        peak = legacy_equity_curve[0]
        max_dd = 0.0
        for e in legacy_equity_curve:
            peak = max(peak, e)
            max_dd = max(max_dd, peak - e)
        return {
            "trades": n,
            "win_rate": (len(wins) / n) if n else 0.0,
            "total_pnl": round(sum(pnls), 2),
            "avg_win": round(gross_win / len(wins), 2) if wins else 0.0,
            "avg_loss": round(-gross_loss / len(losses), 2) if losses else 0.0,
            "profit_factor": round(gross_win / gross_loss, 2) if gross_loss > 0 else None,
            "max_drawdown": round(max_dd, 2),
            "equity": round(self.track_equity("legacy"), 2),
            "open_positions": len(self.open_positions),
            "open": [self._open_position_view(p, now) for p in self.open_positions],
            "by_exit_reason": by_reason,
            "by_track": by_track,
            "ledgers": {t: round(float(self.ledgers.get(t, self.cfg.starting_cash)), 2)
                        for t in PAPER_TRACKS},
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
