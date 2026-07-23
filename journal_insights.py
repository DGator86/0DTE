"""
journal_insights.py
===================
Learning + validation over the TRADE JOURNAL (paper_trades) — the trades the
system actually took, as opposed to the per-tick `evaluations` table that the
adaptive-learning engine mines.

Every paper trade is journaled with its entry thesis (`entry_ctx`: EV,
prob_profit, gate score, regime cell, conviction, RAS history) and its outcome
(pnl, exit reason, peak P&L, hold time). Until now nothing ever held those two
halves against each other. This module closes that loop:

  validate_trades()  — did the predictions that justified each entry come true?
                       EV bias, PoP calibration (Brier + reliability bins),
                       gate-score correlation. Per-trade verdicts included.
  trade_lessons()    — where is the money actually made and lost? Segment
                       breakdown (track / structure / exit / regime /
                       conviction / direction), exit-discipline audit
                       (peak-vs-close giveback, stops near max loss), and
                       plain-language lessons ranked by dollars.
  journal_review()   — both of the above straight off a paper_trades SQLite,
                       degrading gracefully on missing DB / table / context.

Pure stdlib, read-only. NOT financial advice.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Optional

# Trades below this count per segment are reported but not turned into
# "lessons" — a 2-trade segment proves nothing either way.
MIN_LESSON_N = 5

# A trade that reached this fraction of its max profit and still closed red is
# a "round trip": the edge was there and the exit logic gave it all back.
ROUND_TRIP_PEAK_FRAC = 0.50

# A loss within this fraction of planned max loss means the stop effectively
# never engaged — the position rode to (or past) its worst case.
STOP_NEAR_MAX_FRAC = 0.90


# --------------------------------------------------------------------------- #
# small numeric helpers (no numpy in the persistence/readout layer)           #
# --------------------------------------------------------------------------- #
def _num(v) -> Optional[float]:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _corr(xs: list[float], ys: list[float]) -> Optional[float]:
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    vx = sum((a - mx) ** 2 for a in xs)
    vy = sum((b - my) ** 2 for b in ys)
    return round(cov / (vx * vy) ** 0.5, 3) if vx > 0 and vy > 0 else None


def _ctx(trade: dict) -> dict:
    c = trade.get("entry_ctx")
    if isinstance(c, str):
        try:
            c = json.loads(c)
        except (json.JSONDecodeError, TypeError):
            c = None
    return c if isinstance(c, dict) else {}


def _pnl_ps(trade: dict) -> Optional[float]:
    return _num(trade.get("pnl_ps"))


def _won(trade: dict) -> Optional[bool]:
    p = _num(trade.get("pnl_dollars"))
    return None if p is None else p > 0


# --------------------------------------------------------------------------- #
# validation — were the numbers that justified each entry honest?             #
# --------------------------------------------------------------------------- #
def validate_trades(trades: list[dict], n_bins: int = 5) -> dict:
    """
    Hold every closed trade's entry predictions against its realized outcome.

      ev          predicted per-share EV (entry_ctx.ev) vs realized pnl_ps:
                  mean bias (realized − predicted; negative = EV oversold the
                  trade), MAE, correlation, and the fraction of trades that
                  realized at least their predicted EV.
      prob_profit predicted PoP vs realized win/loss: Brier score, Brier SKILL
                  vs always-quoting-the-base-rate (<= 0 means the quoted
                  probabilities carried no information), reliability bins.
      gate_score  correlation of the gate's 0-100 confidence vs realized P&L.

    Each panel carries a plain-language verdict so the dashboard can say
    what the numbers mean instead of printing them and walking away.
    """
    ev_pairs: list[tuple[float, float]] = []      # (predicted_ev, realized_pnl_ps)
    pop_pairs: list[tuple[float, float]] = []     # (prob_profit, won)
    gate_pairs: list[tuple[float, float]] = []    # (gate_score, pnl_dollars)

    for t in trades:
        ctx = _ctx(t)
        pnl_ps = _pnl_ps(t)
        won = _won(t)
        ev = _num(ctx.get("ev"))
        if ev is not None and pnl_ps is not None:
            ev_pairs.append((ev, pnl_ps))
        pop = _num(ctx.get("prob_profit"))
        if pop is not None and won is not None:
            pop_pairs.append((min(max(pop, 0.0), 1.0), 1.0 if won else 0.0))
        gate = _num(ctx.get("gate_score"))
        pnl_d = _num(t.get("pnl_dollars"))
        if gate is not None and pnl_d is not None:
            gate_pairs.append((gate, pnl_d))

    # -- EV panel -----------------------------------------------------------
    ev_panel: dict = {"n": len(ev_pairs)}
    if ev_pairs:
        pred = [p for p, _ in ev_pairs]
        real = [r for _, r in ev_pairs]
        errs = [r - p for p, r in ev_pairs]
        bias = sum(errs) / len(errs)
        hit = sum(1 for e in errs if e >= 0) / len(errs)
        ev_panel.update({
            "mean_predicted_ev": round(sum(pred) / len(pred), 4),
            "mean_realized_pnl_ps": round(sum(real) / len(real), 4),
            "ev_bias": round(bias, 4),
            "mae": round(sum(abs(e) for e in errs) / len(errs), 4),
            "corr_ev_vs_realized": _corr(pred, real),
            "frac_realized_at_least_ev": round(hit, 3),
        })
        if len(ev_pairs) < MIN_LESSON_N:
            ev_panel["verdict"] = "insufficient sample — keep journaling"
        elif bias < -0.05:
            ev_panel["verdict"] = (
                f"EV is OVERSTATED: trades realize ${abs(bias):.2f}/share less "
                "than promised on average — the entry math is selling optimism")
        elif bias > 0.05:
            ev_panel["verdict"] = (
                f"EV is understated: trades realize ${bias:.2f}/share more "
                "than promised on average")
        else:
            ev_panel["verdict"] = "EV is roughly honest (|bias| <= $0.05/share)"

    # -- PoP panel ----------------------------------------------------------
    pop_panel: dict = {"n": len(pop_pairs)}
    if pop_pairs:
        n = len(pop_pairs)
        base = sum(w for _, w in pop_pairs) / n
        brier = sum((p - w) ** 2 for p, w in pop_pairs) / n
        brier_base = sum((base - w) ** 2 for _, w in pop_pairs) / n
        skill = (1.0 - brier / brier_base) if brier_base > 0 else None
        bins = []
        for i in range(n_bins):
            lo, hi = i / n_bins, (i + 1) / n_bins
            inb = [(p, w) for p, w in pop_pairs
                   if (lo <= p < hi) or (i == n_bins - 1 and p == hi)]
            if inb:
                bins.append({
                    "bin": f"{lo:.1f}-{hi:.1f}",
                    "n": len(inb),
                    "mean_predicted": round(sum(p for p, _ in inb) / len(inb), 4),
                    "realized_rate": round(sum(w for _, w in inb) / len(inb), 4),
                })
        pop_panel.update({
            "base_rate": round(base, 4),
            "brier": round(brier, 4),
            "brier_skill": round(skill, 4) if skill is not None else None,
            "bins": bins,
        })
        if n < MIN_LESSON_N:
            pop_panel["verdict"] = "insufficient sample — keep journaling"
        elif skill is None:
            pop_panel["verdict"] = (
                "every trade had the same outcome — calibration unmeasurable yet")
        elif skill <= 0:
            pop_panel["verdict"] = (
                "PoP carries NO information: quoting the base win rate on every "
                "trade would score as well — do not size off these probabilities")
        else:
            pop_panel["verdict"] = (
                f"PoP is informative (Brier skill {skill:+.2f} vs base rate)")

    # -- gate panel ---------------------------------------------------------
    gate_panel: dict = {"n": len(gate_pairs)}
    if gate_pairs:
        c = _corr([g for g, _ in gate_pairs], [p for _, p in gate_pairs])
        gate_panel["corr_gate_vs_pnl"] = c
        if len(gate_pairs) < MIN_LESSON_N:
            gate_panel["verdict"] = "insufficient sample — keep journaling"
        elif c is None:
            gate_panel["verdict"] = "gate score has no variance across trades"
        elif c <= 0:
            gate_panel["verdict"] = (
                "higher gate confidence is NOT earning more — the gate score is "
                "not ranking trades by outcome")
        else:
            gate_panel["verdict"] = f"gate confidence ranks outcomes (corr {c:+.2f})"

    return {
        "n_trades": len(trades),
        "ev": ev_panel,
        "prob_profit": pop_panel,
        "gate_score": gate_panel,
    }


# --------------------------------------------------------------------------- #
# lessons — where the money is made and lost, in plain language               #
# --------------------------------------------------------------------------- #
def _segment_key(trade: dict, dim: str) -> Optional[str]:
    ctx = _ctx(trade)
    if dim == "track":
        return str(ctx.get("fill_track") or "legacy").lower()
    if dim == "family":
        v = trade.get("family")
        return str(v) if v else None
    if dim == "exit_reason":
        v = trade.get("exit_reason")
        return str(v) if v else None
    if dim == "regime":
        v = ctx.get("regime")
        return str(v) if v else None
    if dim == "conviction":
        v = ctx.get("conviction")
        return str(v) if v else None
    if dim == "direction":
        v = ctx.get("direction")
        return str(v) if v else None
    if dim == "cell":
        cell = ctx.get("cell")
        if isinstance(cell, (list, tuple)) and cell:
            return " × ".join(str(c) for c in cell)
        return None
    return None


SEGMENT_DIMS = ("track", "family", "exit_reason", "regime",
                "conviction", "direction", "cell")


def _seg_stats(rows: list[dict]) -> dict:
    pnls = [p for p in (_num(t.get("pnl_dollars")) for t in rows) if p is not None]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    gross_win, gross_loss = sum(wins), -sum(losses)
    return {
        "n": len(pnls),
        "total_pnl": round(sum(pnls), 2),
        "avg_pnl": round(sum(pnls) / len(pnls), 2) if pnls else None,
        "win_rate": round(len(wins) / len(pnls), 3) if pnls else None,
        "profit_factor": (round(gross_win / gross_loss, 2)
                          if gross_loss > 0 else None),
    }


def trade_lessons(trades: list[dict], min_n: int = MIN_LESSON_N) -> dict:
    """
    Aggregate closed trades into the answers a human journal review would
    produce: which segments earn, which bleed, and whether the exits are
    disciplined. `lessons` is a ranked list of plain-language findings, worst
    bleed first — only segments with >= min_n trades qualify, so one bad fill
    can't masquerade as a pattern.
    """
    segments: dict[str, list[dict]] = {}
    for dim in SEGMENT_DIMS:
        buckets: dict[str, list[dict]] = {}
        for t in trades:
            key = _segment_key(t, dim)
            if key is not None:
                buckets.setdefault(key, []).append(t)
        segments[dim] = sorted(
            [{"key": k, **_seg_stats(rows)} for k, rows in buckets.items()],
            key=lambda s: s["total_pnl"])

    # -- exit-discipline audit ---------------------------------------------
    round_trips = 0            # peaked green, closed red
    giveback_fracs: list[float] = []   # winners: how much of the peak was kept?
    stops_near_max = 0
    stops_total = 0
    for t in trades:
        pnl_ps = _pnl_ps(t)
        peak = _num(t.get("peak_pnl_ps"))
        max_profit = _num(t.get("max_profit_ps"))
        max_loss = _num(t.get("max_loss_ps"))
        if pnl_ps is None:
            continue
        if peak is not None and max_profit and peak >= ROUND_TRIP_PEAK_FRAC * max_profit \
                and pnl_ps <= 0:
            round_trips += 1
        if peak is not None and peak > 0 and pnl_ps > 0:
            giveback_fracs.append(max(0.0, min(1.0, 1.0 - pnl_ps / peak)))
        if t.get("exit_reason") == "stop" and max_loss:
            stops_total += 1
            if abs(pnl_ps) >= STOP_NEAR_MAX_FRAC * abs(max_loss):
                stops_near_max += 1

    exit_discipline = {
        "round_trips": round_trips,
        "round_trip_peak_frac": ROUND_TRIP_PEAK_FRAC,
        "avg_winner_giveback": (round(sum(giveback_fracs) / len(giveback_fracs), 3)
                                if giveback_fracs else None),
        "stops_near_max_loss": stops_near_max,
        "stops_total": stops_total,
    }

    # -- plain-language lessons, ranked by dollars --------------------------
    lessons: list[dict] = []
    for dim in SEGMENT_DIMS:
        for seg in segments[dim]:
            if seg["n"] < min_n or seg["total_pnl"] is None:
                continue
            if seg["total_pnl"] < 0:
                lessons.append({
                    "kind": "bleed",
                    "dollars": seg["total_pnl"],
                    "text": (f"{dim}={seg['key']} is bleeding: "
                             f"${seg['total_pnl']:+.2f} over {seg['n']} trades "
                             f"(win rate {seg['win_rate']:.0%})"),
                })
            elif seg["total_pnl"] > 0:
                lessons.append({
                    "kind": "edge",
                    "dollars": seg["total_pnl"],
                    "text": (f"{dim}={seg['key']} is earning: "
                             f"${seg['total_pnl']:+.2f} over {seg['n']} trades "
                             f"(win rate {seg['win_rate']:.0%})"),
                })
    lessons.sort(key=lambda x: x["dollars"])

    if round_trips >= max(2, min_n // 2):
        lessons.insert(0, {
            "kind": "discipline",
            "dollars": None,
            "text": (f"{round_trips} trades reached >= "
                     f"{ROUND_TRIP_PEAK_FRAC:.0%} of max profit and still "
                     "closed red — exits are round-tripping winners"),
        })
    if stops_total >= min_n and stops_near_max / stops_total > 0.5:
        lessons.insert(0, {
            "kind": "discipline",
            "dollars": None,
            "text": (f"{stops_near_max}/{stops_total} stop exits landed within "
                     f"{STOP_NEAR_MAX_FRAC:.0%} of planned max loss — stops are "
                     "firing too late to protect anything"),
        })

    return {
        "n_trades": len(trades),
        "min_lesson_n": min_n,
        "segments": segments,
        "exit_discipline": exit_discipline,
        "lessons": lessons,
    }


# --------------------------------------------------------------------------- #
# entry point over the paper DB                                               #
# --------------------------------------------------------------------------- #
def journal_review(paper_db_path: str, limit: int = 500) -> dict:
    """
    Full journal review off a paper_trades SQLite: validation + lessons.
    Read-only; degrades to an annotated empty payload when the DB or table is
    missing so the dashboard never breaks on a fresh install.
    """
    empty = {"n_trades": 0, "validation": validate_trades([]),
             "lessons": trade_lessons([])}
    if not paper_db_path:
        return {**empty, "note": "no paper database configured"}
    try:
        conn = sqlite3.connect(f"file:{paper_db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
    except sqlite3.Error:
        return {**empty, "note": "paper database unavailable"}
    try:
        rows = conn.execute(
            "SELECT * FROM paper_trades ORDER BY closed_at DESC LIMIT ?",
            (limit,)).fetchall()
    except sqlite3.Error:
        return {**empty, "note": "paper_trades table not found"}
    finally:
        conn.close()

    trades = [dict(r) for r in rows]
    return {
        "n_trades": len(trades),
        "validation": validate_trades(trades),
        "lessons": trade_lessons(trades),
        "note": None,
    }


if __name__ == "__main__":
    import argparse
    import pprint

    ap = argparse.ArgumentParser(description="Trade-journal learning + validation review")
    ap.add_argument("--paper-db", default="paper.sqlite")
    ap.add_argument("--limit", type=int, default=500)
    args = ap.parse_args()
    pprint.pprint(journal_review(args.paper_db, limit=args.limit))
