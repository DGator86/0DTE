"""
Trade journal + churn guards + equity-based sizing:
  - the broker records WHY it entered (entry_ctx) and the journal API serves it
  - a persistent signal opens ONE position per regime, not one per minute
  - risk budget compounds with equity instead of staying pinned to $1000
  - the notifier dedups identical tickets instead of pushing every tick
"""
from __future__ import annotations

import datetime as dt
import json
import os
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from rnd_extractor import ChainQuote, ChainSnapshot
from spread_selector import Leg
from paper_broker import PaperBroker, PaperConfig

ET = ZoneInfo("America/New_York")
LEGS = (Leg(740.0, "P", -1), Leg(735.0, "P", 1))


def _chain(spot, p740, p735):
    def q(strike, pmid):
        return ChainQuote(strike=strike, call_bid=0.01, call_ask=0.03,
                          put_bid=pmid - 0.05, put_ask=pmid + 0.05)
    return ChainSnapshot([q(740.0, p740), q(735.0, p735)], spot=spot, t_years=2e-4)


def _candidate():
    return SimpleNamespace(legs=LEGS, credit=1.00, family="put_credit",
                           short_strikes=(740.0,), long_strikes=(735.0,),
                           max_loss=4.0, ev=0.21, ev_per_risk=0.05, prob_profit=0.71)


def _result(chain, trade=True):
    dec = SimpleNamespace(decision="TRADE" if trade else "NO_TRADE",
                          gate_pass=bool(trade), gate_score=64.2,
                          candidate=_candidate() if trade else None)
    intent = SimpleNamespace(
        exec_regime="compression", context_regime="compression",
        direction_bias="neutral",
        decision=SimpleNamespace(direction="both", conviction="HIGH",
                                 capture="pure theta"),
    )
    regime = SimpleNamespace(dominant_regime="compression",
                             permitted_engine="premium_selling")
    return SimpleNamespace(decision=dec, final_size_mult=1.0, intent=intent,
                           regime=regime, snapshot=SimpleNamespace(chain=chain))


def _broker(tmp_path, **cfg):
    return PaperBroker(db_path=str(tmp_path / "paper.sqlite"), cfg=PaperConfig(**cfg))


T = [dt.datetime(2026, 6, 30, 10, 0, tzinfo=ET) + dt.timedelta(minutes=i)
     for i in range(0, 300)]
ENTRY = _chain(742, 1.50, 0.50)
TARGET = _chain(744, 0.50, 0.15)


# --------------------------------------------------------------------------- #
# churn guards                                                                 #
# --------------------------------------------------------------------------- #
def test_persistent_signal_does_not_reopen_every_tick(tmp_path):
    b = _broker(tmp_path)
    b.on_tick(T[0], _result(ENTRY))
    assert len(b.open_positions) == 1
    b.on_tick(T[1], _result(TARGET))          # target exit fires...
    assert not b.open_positions
    # ...and the SAME signal keeps arriving each minute: cooldown must hold
    for i in range(2, 14):
        b.on_tick(T[i], _result(ENTRY))
        assert not b.open_positions, f"re-entered after only {i - 1} min"
    # after the 15-min cooldown a re-entry is allowed again
    b.on_tick(T[17], _result(ENTRY))
    assert len(b.open_positions) == 1


def test_stop_exit_gets_the_longer_cooldown(tmp_path):
    # $2k account so the post-stop equity still affords a lot — this test is
    # about the cooldown, not the (also correct) can't-afford refusal
    b = _broker(tmp_path, starting_cash=2000.0)
    b.on_tick(T[0], _result(ENTRY))
    b.on_tick(T[1], _result(_chain(736, 3.90, 0.50)))       # stop-loss exit
    assert b.report()["by_exit_reason"] == {"stop": 1}
    b.on_tick(T[20], _result(ENTRY))                        # 19 min < 30 min
    assert not b.open_positions
    b.on_tick(T[35], _result(ENTRY))                        # 34 min > 30 min
    assert len(b.open_positions) == 1


def test_max_trades_per_day(tmp_path):
    b = _broker(tmp_path, reentry_cooldown_min=0.0, max_trades_per_day=2)
    b.on_tick(T[0], _result(ENTRY))
    b.on_tick(T[1], _result(TARGET))
    b.on_tick(T[2], _result(ENTRY))
    b.on_tick(T[3], _result(TARGET))
    b.on_tick(T[4], _result(ENTRY))                         # third entry: refused
    assert not b.open_positions
    assert b.report()["trades"] == 2


# --------------------------------------------------------------------------- #
# equity-based sizing                                                          #
# --------------------------------------------------------------------------- #
def test_risk_budget_compounds_with_equity(tmp_path):
    b = _broker(tmp_path, reentry_cooldown_min=0.0)
    assert b.cash == 1000.0
    b.on_tick(T[0], _result(ENTRY))
    assert b.open_positions[0].contracts == 1               # 50% of $1000 -> 1 lot @ ~$405 risk
    b.cash = 4000.0                                          # simulate a grown account
    b.on_tick(T[1], _result(TARGET))                         # close the first
    b.on_tick(T[2], _result(ENTRY))
    assert b.open_positions[0].contracts >= 4                # 50% of $4k+ -> 4+ lots
    ctx = b.open_positions[0].entry_ctx
    assert ctx["equity_at_entry"] >= 4000.0


def test_daily_loss_limit_tracks_day_start_equity(tmp_path):
    b = _broker(tmp_path, reentry_cooldown_min=0.0)
    b.on_tick(T[0], _result(ENTRY))
    b.on_tick(T[1], _result(_chain(736, 3.90, 0.50)))       # big stop-loss
    day = T[0].astimezone(ET).date().isoformat()
    # force the day's realized loss past 50% of day-start equity
    b._day_realized[day] = -0.51 * b._day_start_cash[day]
    b.on_tick(T[40], _result(ENTRY))                        # past cooldown, but loss-limited
    assert not b.open_positions


# --------------------------------------------------------------------------- #
# entry context persisted + served                                              #
# --------------------------------------------------------------------------- #
def test_entry_ctx_recorded_and_served(tmp_path):
    b = _broker(tmp_path)
    b.on_tick(T[0], _result(ENTRY))
    open_view = b.report(T[1])["open"][0]
    assert open_view["entry_ctx"]["cell"] == ["compression", "compression", "neutral"]
    assert open_view["entry_ctx"]["gate_score"] == 64.2

    b.on_tick(T[1], _result(TARGET))                        # close it
    from dashboard.queries import paper_trades_journal
    data = paper_trades_journal(str(tmp_path / "paper.sqlite"))
    assert len(data["closed"]) == 1
    t = data["closed"][0]
    assert t["exit_reason"] == "target"
    assert t["entry_ctx"]["conviction"] == "HIGH"
    assert t["entry_ctx"]["prob_profit"] == 0.71


def test_entry_ctx_migration_on_legacy_db(tmp_path):
    # simulate a pre-existing VPS DB without the entry_ctx column
    import sqlite3
    path = str(tmp_path / "paper.sqlite")
    conn = sqlite3.connect(path)
    conn.execute("""CREATE TABLE paper_trades (
        id TEXT PRIMARY KEY, symbol TEXT, family TEXT, strikes TEXT,
        contracts INTEGER, opened_at TEXT, closed_at TEXT, hold_min REAL,
        entry_credit REAL, exit_value REAL, max_profit_ps REAL, max_loss_ps REAL,
        pnl_ps REAL, pnl_dollars REAL, exit_reason TEXT, equity_after REAL)""")
    conn.commit()
    conn.close()
    b = PaperBroker(db_path=path)                            # must migrate, not crash
    b.on_tick(T[0], _result(ENTRY))
    b.on_tick(T[1], _result(TARGET))
    from dashboard.queries import paper_trades_journal
    assert paper_trades_journal(path)["closed"][0]["entry_ctx"]["regime"] == "compression"


# --------------------------------------------------------------------------- #
# /api/trades endpoint                                                          #
# --------------------------------------------------------------------------- #
def test_api_trades_endpoint(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from dashboard.server import app, _configure
    from dashboard.state import write_live_state

    monkeypatch.setenv("DASHBOARD_TOKEN", "tok")
    b = _broker(tmp_path)
    b.on_tick(T[0], _result(ENTRY))
    b.on_tick(T[1], _result(TARGET))
    live = os.path.join(tmp_path, "live_state.json")
    write_live_state(live, {"ts": "x", "paper": b.report(T[2])})
    _configure(os.path.join(tmp_path, "shadow.db"), str(tmp_path / "paper.sqlite"), live)

    client = TestClient(app)
    r = client.get("/api/trades", headers={"Authorization": "Bearer tok"})
    assert r.status_code == 200
    data = r.json()
    assert len(data["closed"]) == 1
    assert data["closed"][0]["entry_ctx"]["cell"] == ["compression", "compression", "neutral"]
    assert isinstance(data["open"], list)


# --------------------------------------------------------------------------- #
# notifier dedup                                                                #
# --------------------------------------------------------------------------- #
def _ticket(family="iron_condor", short_puts=(597.0,)):
    from notifier import Ticket
    return Ticket(
        ts="2026-07-02T10:30:00-04:00", session_date="2026-07-02", symbol="SPY",
        dominant_regime="compression", exec_regime="compression",
        context_regime="compression", direction_bias="neutral", size_mult=1.0,
        family=family, direction="both",
        short_calls=[602.0], long_calls=[604.0],
        short_puts=list(short_puts), long_puts=[595.0],
        credit=1.42, max_loss=0.58, ev=0.31, ev_per_risk=0.53, prob_profit=0.72,
        gate_score=0.81, theta_per_day=0.18, contracts_per_1k=1,
    )


def test_notifier_dedups_repeated_ticket(monkeypatch):
    from notifier import Notifier
    sent = []
    n = Notifier()
    monkeypatch.setattr(Notifier, "_stdout", staticmethod(lambda text: sent.append(text)))
    monkeypatch.setattr(Notifier, "_file", staticmethod(lambda t: None))
    monkeypatch.setattr(Notifier, "_email", staticmethod(lambda t, x: None))
    monkeypatch.setattr(Notifier, "_ntfy", staticmethod(lambda t, x: None))
    monkeypatch.setenv("NOTIFY_COOLDOWN_MIN", "15")

    for _ in range(50):
        n.send(_ticket())                        # the "50 identical signals" day
    assert len(sent) == 1

    n.send(_ticket(short_puts=(596.0,)))         # strikes changed -> new signal
    assert len(sent) == 2

    monkeypatch.setenv("NOTIFY_COOLDOWN_MIN", "0")   # opt out -> every tick again
    n.send(_ticket(short_puts=(596.0,)))
    assert len(sent) == 3


def test_ticket_classifies_C_P_legs_correctly():
    from notifier import Ticket
    dec = SimpleNamespace(
        decision="TRADE", gate_pass=True, session_date="2026-07-02",
        gate_score=50.0,
        candidate=SimpleNamespace(
            legs=(Leg(740.0, "C", -1), Leg(745.0, "C", 1)),
            credit=0.5, max_loss=4.5, ev=0.1, ev_per_risk=0.02,
            prob_profit=0.6, theta=0.1, family="call_credit",
            short_strikes=(740.0,), long_strikes=(745.0,)),
    )
    res = SimpleNamespace(
        ts=T[0], decision=dec, final_size_mult=1.0,
        intent=SimpleNamespace(exec_regime="e", context_regime="c",
                               direction_bias="neutral",
                               decision=SimpleNamespace(direction="call")),
        regime=SimpleNamespace(dominant_regime="trend"),
    )
    t = Ticket.from_tick_result(res, "SPY")
    assert t.short_calls == [740.0] and t.long_calls == [745.0]
    assert t.short_puts == [] and t.long_puts == []
