"""Trade-journal learning + validation (journal_insights)."""
import json
import sqlite3

import pytest

from journal_insights import (
    journal_review,
    track_feedback,
    trade_lessons,
    validate_trades,
)


def _trade(pnl_dollars, pnl_ps, *, ev=0.10, pop=0.6, gate=55.0, family="iron_fly",
           exit_reason=None, peak=None, max_profit=0.4, max_loss=-0.6,
           track="legacy", regime="compression", conviction="HIGH",
           direction="put", cell=("compression", "compression", "bear"),
           ctx_json=True):
    ctx = {"ev": ev, "prob_profit": pop, "gate_score": gate, "fill_track": track,
           "regime": regime, "conviction": conviction, "direction": direction,
           "cell": list(cell)}
    return {
        "pnl_dollars": pnl_dollars,
        "pnl_ps": pnl_ps,
        "peak_pnl_ps": peak,
        "max_profit_ps": max_profit,
        "max_loss_ps": max_loss,
        "exit_reason": exit_reason or ("target" if pnl_dollars > 0 else "stop"),
        "family": family,
        "entry_ctx": json.dumps(ctx) if ctx_json else ctx,
    }


# --------------------------------------------------------------------------- #
# validation                                                                   #
# --------------------------------------------------------------------------- #
def test_validate_trades_empty():
    out = validate_trades([])
    assert out["n_trades"] == 0
    assert out["ev"]["n"] == 0
    assert out["prob_profit"]["n"] == 0
    assert out["gate_score"]["n"] == 0


def test_ev_bias_flags_overstated_ev():
    # promised $0.50/share, realized -$0.30/share on every trade
    trades = [_trade(-30.0, -0.30, ev=0.50) for _ in range(8)]
    ev = validate_trades(trades)["ev"]
    assert ev["n"] == 8
    assert ev["ev_bias"] == pytest.approx(-0.80)
    assert "OVERSTATED" in ev["verdict"]
    assert ev["frac_realized_at_least_ev"] == 0.0


def test_ev_honest_when_realized_matches():
    trades = [_trade(10.0, 0.10, ev=0.10) for _ in range(6)]
    ev = validate_trades(trades)["ev"]
    assert ev["ev_bias"] == pytest.approx(0.0)
    assert "honest" in ev["verdict"]


def test_pop_uninformative_when_constant():
    # constant 60% PoP with a 50% realized win rate carries no information
    trades = [_trade(10.0 if i % 2 else -10.0, 0.1 if i % 2 else -0.1, pop=0.6)
              for i in range(10)]
    pop = validate_trades(trades)["prob_profit"]
    assert pop["brier_skill"] is not None and pop["brier_skill"] <= 0
    assert "NO information" in pop["verdict"]
    assert pop["bins"]  # reliability table present


def test_pop_informative_when_discriminating():
    # high PoP wins, low PoP loses -> positive Brier skill
    trades = ([_trade(10.0, 0.1, pop=0.9) for _ in range(6)]
              + [_trade(-10.0, -0.1, pop=0.1) for _ in range(6)])
    pop = validate_trades(trades)["prob_profit"]
    assert pop["brier_skill"] > 0
    assert "informative" in pop["verdict"]


def test_gate_score_correlation_sign():
    # higher gate score loses more -> non-positive corr flagged
    trades = [_trade(-g, -g / 100.0, gate=g) for g in (30.0, 50.0, 70.0, 90.0, 60.0)]
    gate = validate_trades(trades)["gate_score"]
    assert gate["corr_gate_vs_pnl"] < 0
    assert "NOT earning" in gate["verdict"]


def test_validation_survives_malformed_ctx():
    trades = [
        {"pnl_dollars": 5.0, "pnl_ps": 0.05, "entry_ctx": "{not json"},
        {"pnl_dollars": 5.0, "pnl_ps": 0.05, "entry_ctx": None},
        {"pnl_dollars": 5.0, "pnl_ps": 0.05},
    ]
    out = validate_trades(trades)
    assert out["n_trades"] == 3
    assert out["ev"]["n"] == 0


# --------------------------------------------------------------------------- #
# lessons                                                                      #
# --------------------------------------------------------------------------- #
def test_lessons_rank_bleeders_first_and_respect_min_n():
    trades = ([_trade(-40.0, -0.40, family="long_put_spread") for _ in range(6)]
              + [_trade(25.0, 0.25, family="iron_fly") for _ in range(6)]
              # 2-trade segment must NOT become a lesson
              + [_trade(-99.0, -0.99, family="backspread") for _ in range(2)])
    out = trade_lessons(trades, min_n=5)
    fam = {s["key"]: s for s in out["segments"]["family"]}
    assert fam["long_put_spread"]["total_pnl"] == pytest.approx(-240.0)
    assert fam["long_put_spread"]["win_rate"] == 0.0
    texts = [l["text"] for l in out["lessons"]]
    assert any("family=long_put_spread is bleeding" in t for t in texts)
    assert any("family=iron_fly is earning" in t for t in texts)
    assert not any("backspread" in t for t in texts)
    # dollar-ranked: the worst bleed leads the segment lessons
    seg_lessons = [l for l in out["lessons"] if l["kind"] in ("bleed", "edge")]
    assert seg_lessons[0]["kind"] == "bleed"


def test_exit_discipline_round_trips_and_late_stops():
    # peaked at 75% of max profit, closed red -> round trip; stop at max loss
    trades = [_trade(-30.0, -0.60, peak=0.30, max_profit=0.40, max_loss=-0.6,
                     exit_reason="stop") for _ in range(6)]
    out = trade_lessons(trades, min_n=5)
    disc = out["exit_discipline"]
    assert disc["round_trips"] == 6
    assert disc["stops_near_max_loss"] == 6
    texts = [l["text"] for l in out["lessons"]]
    assert any("round-tripping" in t for t in texts)
    assert any("firing too late" in t for t in texts)


def test_winner_giveback_measured():
    # winners kept half their peak
    trades = [_trade(10.0, 0.10, peak=0.20) for _ in range(4)]
    disc = trade_lessons(trades)["exit_discipline"]
    assert disc["avg_winner_giveback"] == pytest.approx(0.5)


def test_segments_read_ctx_dimensions():
    trades = [_trade(10.0, 0.1, track="v2", regime="trend", conviction="MED",
                     direction="call", cell=("trend", "trend", "bull"))
              for _ in range(3)]
    seg = trade_lessons(trades)["segments"]
    assert seg["track"][0]["key"] == "v2"
    assert seg["regime"][0]["key"] == "trend"
    assert seg["conviction"][0]["key"] == "MED"
    assert seg["direction"][0]["key"] == "call"
    assert seg["cell"][0]["key"] == "trend × trend × bull"


# --------------------------------------------------------------------------- #
# journal_review over sqlite                                                   #
# --------------------------------------------------------------------------- #
def _make_paper_db(path, trades):
    conn = sqlite3.connect(path)
    conn.execute("""
        CREATE TABLE paper_trades (
            id TEXT PRIMARY KEY,
            symbol TEXT, family TEXT, strikes TEXT, contracts INTEGER,
            opened_at TEXT, closed_at TEXT, hold_min REAL,
            entry_credit REAL, exit_value REAL,
            max_profit_ps REAL, max_loss_ps REAL,
            pnl_ps REAL, pnl_dollars REAL, exit_reason TEXT,
            equity_after REAL, entry_ctx TEXT, peak_pnl_ps REAL
        )""")
    for i, t in enumerate(trades):
        conn.execute(
            "INSERT INTO paper_trades (id, family, closed_at, max_profit_ps, "
            "max_loss_ps, pnl_ps, pnl_dollars, exit_reason, entry_ctx, peak_pnl_ps) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (f"t{i}", t["family"], f"2026-07-23T15:{i:02d}:00",
             t["max_profit_ps"], t["max_loss_ps"], t["pnl_ps"],
             t["pnl_dollars"], t["exit_reason"], t["entry_ctx"],
             t["peak_pnl_ps"]))
    conn.commit()
    conn.close()


def test_journal_review_end_to_end(tmp_path):
    db = str(tmp_path / "paper.sqlite")
    trades = ([_trade(-40.0, -0.40, ev=0.5, family="long_put_spread")
               for _ in range(6)]
              + [_trade(25.0, 0.25, ev=0.1, family="iron_fly") for _ in range(6)])
    _make_paper_db(db, trades)
    out = journal_review(db)
    assert out["n_trades"] == 12
    assert out["validation"]["ev"]["n"] == 12
    assert any("bleeding" in l["text"] for l in out["lessons"]["lessons"])
    assert out["note"] is None


def test_track_feedback_filters_to_track(tmp_path):
    db = str(tmp_path / "paper.sqlite")
    trades = ([_trade(-40.0, -0.40, ev=0.5, family="long_put_spread",
                      track="spy_der") for _ in range(6)]
              + [_trade(25.0, 0.25, family="iron_fly", track="spy_der")
                 for _ in range(6)]
              # legacy trades must NOT leak into spy_der's record
              + [_trade(999.0, 9.99, family="iron_fly", track="legacy")
                 for _ in range(3)])
    _make_paper_db(db, trades)
    fb = track_feedback(db, track="spy_der")
    assert fb is not None
    assert fb["n_trades"] == 12
    assert fb["total_pnl"] == pytest.approx(-90.0)
    fams = {f["family"]: f for f in fb["by_family"]}
    assert fams["long_put_spread"]["total_pnl"] == pytest.approx(-240.0)
    assert "iron_fly" in fams
    assert fb["ev_bias_per_share"] is not None
    assert any("long_put_spread is bleeding" in t for t in fb["lessons"])
    # regime/cell lessons are filtered out — only agent-actionable dimensions
    assert not any(t.startswith(("regime=", "cell=", "track=", "conviction="))
                   for t in fb["lessons"])


def test_track_feedback_none_when_track_empty(tmp_path):
    db = str(tmp_path / "paper.sqlite")
    _make_paper_db(db, [_trade(10.0, 0.1, track="legacy")])
    assert track_feedback(db, track="spy_der") is None
    assert track_feedback("", track="spy_der") is None
    assert track_feedback(str(tmp_path / "missing.sqlite")) is None


def test_bridge_accepts_track_record_without_package():
    """decide_spy_der_tick must tolerate the new kwarg when the spy_der
    package is absent (returns UNAVAILABLE, never raises)."""
    import spy_der_bridge as b
    if b.spy_der_available():
        pytest.skip("spy_der package installed; unavailable path not testable")
    import datetime as dt
    out = b.decide_spy_der_tick(
        snapshot_id="s1", symbol="SPY",
        session_date=dt.date(2026, 7, 23), underlying_price=600.0,
        shadow_candidates=[], now=dt.datetime(2026, 7, 23, 15, 0),
        track_record={"n_trades": 3, "win_rate": 0.33, "total_pnl": -20.0},
    )
    assert out.action == "UNAVAILABLE"


def test_journal_review_degrades_gracefully(tmp_path):
    assert journal_review("")["note"] == "no paper database configured"
    missing = journal_review(str(tmp_path / "nope.sqlite"))
    assert missing["n_trades"] == 0 and missing["note"]
    # DB exists but no paper_trades table
    empty = str(tmp_path / "empty.sqlite")
    sqlite3.connect(empty).execute("CREATE TABLE x (id INTEGER)")
    out = journal_review(empty)
    assert out["n_trades"] == 0
    assert out["note"] == "paper_trades table not found"
