"""
journal.py  —  the decision log that turns 'I feel ready to scale' into a number.

Every decision is recorded with the MC's predicted probability at the time.
Later you record what price actually did (15/30/60 min) and, once live, the
realized P&L. Two payoffs:

  performance()  -> win_rate, avg_win, avg_loss, n   (feeds scale_risk)
  calibration()  -> did the MC's predicted P(target) match reality?

The calibration check is the honesty governor on the Monte Carlo. If MC predicts
60% and you realize 45% across a sample, the model is optimistic — you trust the
journal's realized numbers for sizing and you recalibrate the MC knobs.

SQLite, stdlib only, runs anywhere, ~zero cost.
"""
from __future__ import annotations
import sqlite3
import time

SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
  id INTEGER PRIMARY KEY, ts INTEGER, mode TEXT, action TEXT, regime TEXT,
  net_ratio REAL, flip REAL, spot REAL, target REAL, stop REAL,
  instrument TEXT, contracts INTEGER, risk_frac REAL,
  mc_p REAL, mc_ev REAL, win_R REAL
);
CREATE TABLE IF NOT EXISTS outcomes (
  decision_id INTEGER PRIMARY KEY, spot_15 REAL, spot_30 REAL, spot_60 REAL,
  hit_target INTEGER, realized_pnl REAL, note TEXT,
  FOREIGN KEY(decision_id) REFERENCES decisions(id)
);
"""


class Journal:
    def __init__(self, path: str = "spy0dte.sqlite") -> None:
        self.con = sqlite3.connect(path)
        self.con.executescript(SCHEMA)
        self.con.commit()

    def record_decision(self, *, mode, action, regime, net_ratio, flip, spot,
                        target, stop, instrument, contracts, risk_frac,
                        mc_p, mc_ev, win_R) -> int:
        cur = self.con.execute(
            """INSERT INTO decisions(ts,mode,action,regime,net_ratio,flip,spot,target,stop,
               instrument,contracts,risk_frac,mc_p,mc_ev,win_R)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (int(time.time()*1000), mode, action, regime, net_ratio, flip, spot, target, stop,
             instrument, contracts, risk_frac, mc_p, mc_ev, win_R))
        self.con.commit()
        return cur.lastrowid

    def record_outcome(self, decision_id, *, spot_15=None, spot_30=None, spot_60=None,
                       hit_target=None, realized_pnl=None, note="") -> None:
        self.con.execute(
            """INSERT OR REPLACE INTO outcomes(decision_id,spot_15,spot_30,spot_60,
               hit_target,realized_pnl,note) VALUES(?,?,?,?,?,?,?)""",
            (decision_id, spot_15, spot_30, spot_60,
             None if hit_target is None else int(hit_target), realized_pnl, note))
        self.con.commit()

    def performance(self, action: str | None = None) -> dict:
        """Realized win_rate / avg_win / avg_loss for scale_risk. Uses realized_pnl
        when present, else falls back to hit_target with win_R/-1 as an R proxy."""
        q = """SELECT d.win_R, o.hit_target, o.realized_pnl
               FROM decisions d JOIN outcomes o ON o.decision_id=d.id
               WHERE (o.hit_target IS NOT NULL OR o.realized_pnl IS NOT NULL)"""
        params: tuple = ()
        if action:
            q += " AND d.action=?"
            params = (action,)
        rows = self.con.execute(q, params).fetchall()
        wins, losses = [], []
        for win_R, hit, pnl in rows:
            val = pnl if pnl is not None else ((win_R or 1.0) if hit else -1.0)
            (wins if val > 0 else losses).append(val)
        n = len(wins) + len(losses)
        if n == 0:
            return {"n": 0, "win_rate": 0.0, "avg_win": 0.0, "avg_loss": 0.0}
        return {
            "n": n,
            "win_rate": round(len(wins) / n, 3),
            "avg_win": round(sum(wins) / len(wins), 3) if wins else 0.0,
            "avg_loss": round(abs(sum(losses) / len(losses)), 3) if losses else 1.0,
        }

    def calibration(self) -> dict:
        """Compare MC predicted P(target) to realized hit rate — the MC honesty check."""
        rows = self.con.execute(
            """SELECT d.mc_p, o.hit_target FROM decisions d JOIN outcomes o
               ON o.decision_id=d.id WHERE o.hit_target IS NOT NULL AND d.mc_p IS NOT NULL"""
        ).fetchall()
        if not rows:
            return {"n": 0, "note": "no resolved decisions with MC predictions yet"}
        pred = sum(r[0] for r in rows) / len(rows)
        real = sum(r[1] for r in rows) / len(rows)
        gap = real - pred
        verdict = ("MC well-calibrated" if abs(gap) < 0.05 else
                   "MC OPTIMISTIC — trust journal, lower MC drift knobs" if gap < 0 else
                   "MC pessimistic — it's underclaiming")
        return {"n": len(rows), "mc_predicted": round(pred, 3),
                "realized": round(real, 3), "gap": round(gap, 3), "verdict": verdict}


if __name__ == "__main__":
    # --- full loop smoke test: engine -> MC -> size -> journal -> performance ---
    import os
    if os.path.exists("demo.sqlite"):
        os.remove("demo.sqlite")
    import spy0dte as eng
    import mc

    j = Journal("demo.sqlite")
    # simulate 35 short-gamma trend decisions with a realistic ~45% realized hit rate
    import numpy as np
    rng = np.random.default_rng(7)
    for i in range(35):
        spot, target, stop, flip = 600.0, 602.0, 599.5, 599.5
        proj = mc.project(spot, target, stop, flip, minutes_left=120,
                          iv_annual=0.13, regime="trend", win_R=2.0, seed=i)
        did = bool(rng.random() < 0.45)   # ground-truth ~45% (MC will read higher)
        did_ = j.record_decision(mode="shadow", action="PUT", regime="trend",
                                 net_ratio=-0.35, flip=flip, spot=spot, target=target,
                                 stop=stop, instrument="SPY", contracts=1, risk_frac=0.02,
                                 mc_p=proj.p_target, mc_ev=proj.ev_R, win_R=2.0)
        j.record_outcome(did_, hit_target=did,
                         realized_pnl=(2.0 if did else -1.0))

    perf = j.performance()
    print("performance:", perf)
    r, why = eng.scale_risk(perf["n"], perf["win_rate"], perf["avg_win"], perf["avg_loss"])
    print(f"scaled risk -> {r:.0%} | {why}")
    print("calibration:", j.calibration())
