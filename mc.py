"""
mc.py  —  regime-conditioned Monte Carlo for SPY/XSP 0DTE.

WHY NOT PLAIN MONTE CARLO:
A GBM random-walk MC that simulates price to the close is just Black-Scholes
with noise — it reprices the option you can already price in closed form, and it
IGNORES the only thing that matters here: dealer hedging makes price NOT a random
walk. So this MC conditions the path dynamics on the gamma regime:

  short gamma (trend) -> momentum drift AWAY from the flip + elevated vol
  long  gamma (pin)   -> mean-reversion TOWARD the flip (OU) + suppressed vol

Output is one decision-grade number: P(target hit before stop), plus the EV that
follows from it. That probability feeds Kelly as a PRIOR — used to seed risk
sizing before the live journal has a real sample, then corrected by the journal.

This is a structured guess, not truth. It is only trustworthy once its predicted
probabilities are checked against realized hit rates (see journal.calibration()).
If MC says 60% and you realize 45%, believe the journal and recalibrate the knobs.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

MINUTES_PER_YEAR = 252 * 390  # trading minutes


@dataclass
class Projection:
    p_target: float      # P(reach target before stop) under regime dynamics
    p_stop: float        # P(hit stop first)
    ev_R: float          # expected value in R units (target=+win_R, stop=-1)
    n_paths: int
    note: str


# regime dynamics knobs — these are what you CALIBRATE against the journal
TREND_DRIFT_K = 0.55     # momentum strength (fraction of sigma per step, directional)
TREND_VOL_MULT = 1.25    # short-gamma vol is higher
PIN_REVERT_K = 0.04      # OU pull per minute toward flip
PIN_VOL_MULT = 0.70      # long-gamma vol is suppressed


def project(spot: float, target: float, stop: float, flip: float,
            minutes_left: int, iv_annual: float, regime: str,
            win_R: float, n_paths: int = 20000, seed: int | None = None) -> Projection:
    """
    First-passage simulation of the remaining session.
    target/stop are PRICE levels. win_R is the reward multiple if target is hit
    before stop (e.g. premium move to wall ≈ +2R) vs -1R if stopped.
    """
    rng = np.random.default_rng(seed)
    steps = max(1, int(minutes_left))
    sigma_min = iv_annual / np.sqrt(MINUTES_PER_YEAR)   # per-minute vol (fraction)

    up = target > spot  # direction of the target
    if regime == "trend":
        drift = TREND_DRIFT_K * sigma_min * (1.0 if up else -1.0)
        vol = sigma_min * TREND_VOL_MULT
    else:  # pin
        drift = 0.0                  # handled by mean-reversion term below
        vol = sigma_min * PIN_VOL_MULT

    # simulate paths in price space (lognormal-ish via multiplicative steps)
    prices = np.full(n_paths, spot, dtype=float)
    hit_target = np.zeros(n_paths, dtype=bool)
    hit_stop = np.zeros(n_paths, dtype=bool)
    live = np.ones(n_paths, dtype=bool)

    for _ in range(steps):
        z = rng.standard_normal(n_paths)
        if regime == "pin":
            # Ornstein-Uhlenbeck pull toward flip
            step = PIN_REVERT_K * (flip - prices) / prices + vol * z
        else:
            step = drift + vol * z
        prices = prices * (1.0 + step)

        if up:
            now_target = prices >= target
            now_stop = prices <= stop
        else:
            now_target = prices <= target
            now_stop = prices >= stop

        newly_t = live & now_target
        newly_s = live & now_stop & ~now_target
        hit_target |= newly_t
        hit_stop |= newly_s
        live &= ~(newly_t | newly_s)
        if not live.any():
            break

    p_t = float(hit_target.mean())
    p_s = float(hit_stop.mean())
    # unresolved paths (neither touched) count as scratch ~ small loss (theta); treat as half-stop
    p_scratch = 1.0 - p_t - p_s
    ev = p_t * win_R - p_s * 1.0 - p_scratch * 0.5
    note = (f"{regime}: P(target)={p_t:.2f} P(stop)={p_s:.2f} "
            f"scratch={p_scratch:.2f} -> EV={ev:.2f}R")
    return Projection(round(p_t, 3), round(p_s, 3), round(ev, 3), n_paths, note)


def p_to_kelly_inputs(proj: Projection, win_R: float) -> tuple[float, float, float]:
    """Translate an MC projection into (win_rate, avg_win, avg_loss) for scale_risk."""
    return proj.p_target, win_R, 1.0


def project_range(spot: float, lower_short: float, upper_short: float, flip: float,
                  minutes_left: int, iv_annual: float, regime: str,
                  win_R: float, n_paths: int = 20000, seed: int | None = None) -> Projection:
    """
    Condor survival: P(price NEVER touches either short strike before close).
    This is the correct question for a premium seller — not 'target before stop'
    but 'do I stay in the range'. In a pin regime this should be high; if the MC
    says it's low, the condor is mispriced for the structure and you skip it.
    win_R here is the credit/max-loss ratio (small), so EV stays honest about the
    seller's ugly payoff geometry.
    """
    rng = np.random.default_rng(seed)
    steps = max(1, int(minutes_left))
    sigma_min = iv_annual / np.sqrt(MINUTES_PER_YEAR)
    vol = sigma_min * (PIN_VOL_MULT if regime == "pin" else TREND_VOL_MULT)

    prices = np.full(n_paths, spot, dtype=float)
    breached = np.zeros(n_paths, dtype=bool)
    for _ in range(steps):
        z = rng.standard_normal(n_paths)
        if regime == "pin":
            step = PIN_REVERT_K * (flip - prices) / prices + vol * z
        else:
            step = vol * z
        prices = prices * (1.0 + step)
        breached |= (prices <= lower_short) | (prices >= upper_short)

    p_survive = float((~breached).mean())
    p_breach = 1.0 - p_survive
    ev = p_survive * win_R - p_breach * 1.0
    note = f"{regime}: P(stay in range)={p_survive:.2f} -> EV={ev:.2f}R (credit/maxloss={win_R:.2f})"
    return Projection(round(p_survive, 3), round(p_breach, 3), round(ev, 3), n_paths, note)


if __name__ == "__main__":
    # trend up: target above, stop at flip below -> should favor target
    pr = project(spot=600, target=602, stop=599.5, flip=599.5,
                 minutes_left=120, iv_annual=0.13, regime="trend", win_R=2.0, seed=1)
    print("TREND directional ", pr.note, "| ev", pr.ev_R)
    # pin condor: short strikes at 599/602, win_R = credit/maxloss (e.g. 0.21/0.79)
    pr2 = project_range(spot=600, lower_short=599, upper_short=602, flip=600.0,
                        minutes_left=120, iv_annual=0.10, regime="pin", win_R=0.27, seed=1)
    print("PIN condor        ", pr2.note, "| ev", pr2.ev_R)
