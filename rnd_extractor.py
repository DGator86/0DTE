"""
rnd_extractor.py
================
Breeden-Litzenberger risk-neutral density (RND) extraction from a 0DTE option
chain, plus the physical-vs-risk-neutral divergence that upgrades the crude
`straddle_rich` signal in gate_scorer.py into a full-distribution edge measure.

Why not differentiate raw prices
--------------------------------
B-L says q(K) = e^{rT} d2C/dK2. Applied naively to quoted mids, the second
difference amplifies bid-ask noise into a spiky, often-negative "density".
Robust pipeline instead:

  1. Recover forward F and discount factor DF jointly from put-call parity
     (linear regression of C-P on K) -- no need to input r or dividend yield.
  2. Invert each liquid OTM option to TOTAL VARIANCE  w = (sigma^2 T).
     Working in total variance makes the annualization convention irrelevant
     and the smile far smoother than the price curve.
  3. Fit a liquidity-weighted smoothing spline to w(k), k = log-moneyness.
  4. Reconstruct a dense, analytically smooth call curve from the fitted smile.
  5. Second-difference THAT (stable) -> RND. Clip butterfly-arb negatives,
     renormalize, and report the violation magnitude as a quality flag.

Then:
  6. Moments, CDF, P(S_T > K), P(touch K before close) via reflection.
  7. EV-of-selling each strike under your PHYSICAL density, and the
     RN/physical variance ratio -> the richness signal the gate consumes.

Deps: numpy, scipy.  NOT financial advice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np
from scipy.interpolate import make_smoothing_spline
from scipy.stats import norm

SQRT2PI = np.sqrt(2 * np.pi)


# --------------------------------------------------------------------------- #
# Inputs                                                                       #
# --------------------------------------------------------------------------- #
@dataclass
class ChainQuote:
    strike: float
    call_bid: float
    call_ask: float
    put_bid: float
    put_ask: float

    @property
    def call_mid(self) -> float:
        return 0.5 * (self.call_bid + self.call_ask)

    @property
    def put_mid(self) -> float:
        return 0.5 * (self.put_bid + self.put_ask)

    @property
    def call_spread(self) -> float:
        return max(self.call_ask - self.call_bid, 1e-9)

    @property
    def put_spread(self) -> float:
        return max(self.put_ask - self.put_bid, 1e-9)


@dataclass
class ChainSnapshot:
    quotes: list[ChainQuote]
    spot: float
    t_years: float                 # time to expiry in years (any consistent convention)
    r: float = 0.05                # short rate; barely matters at 0DTE but used if parity is thin

    def sorted_quotes(self) -> list[ChainQuote]:
        return sorted(self.quotes, key=lambda q: q.strike)


@dataclass
class RNDConfig:
    grid_step: float = 0.10        # $ spacing of the fine reconstruction grid
    grid_pad_sigma: float = 6.0    # extend grid +/- this many RN std-devs past F
    min_price: float = 0.02        # ignore OTM quotes cheaper than this (junk inversion)
    max_rel_spread: float = 0.60   # ignore quotes whose spread > this fraction of mid
    parity_atm_window: int = 8     # # of near-ATM strikes used in the F/DF regression
    spline_lam: Optional[float] = None  # smoothing spline penalty; None = auto (GCV-like)
    vol_risk_premium: float = 0.18 # default physical haircut: var_phys = (1-vrp)*var_RN
    touch_cap: float = 1.0


# --------------------------------------------------------------------------- #
# Black-76 in total-variance form (s = sigma*sqrt(T))                          #
# --------------------------------------------------------------------------- #
def _bs_call_fwd(F: float, K: float, s: float) -> float:
    """Undiscounted (forward) Black call value with total vol s."""
    if s <= 0:
        return max(F - K, 0.0)
    d1 = np.log(F / K) / s + 0.5 * s
    d2 = d1 - s
    return F * norm.cdf(d1) - K * norm.cdf(d2)


def _bs_vega_s(F: float, K: float, s: float) -> float:
    if s <= 0:
        return 0.0
    d1 = np.log(F / K) / s + 0.5 * s
    return F * norm.pdf(d1)


def _implied_total_vol(target_fwd: float, F: float, K: float) -> Optional[float]:
    """Invert undiscounted call price -> total vol s. Newton + bisection guard."""
    intrinsic = max(F - K, 0.0)
    if target_fwd <= intrinsic + 1e-9 or target_fwd >= F:
        return None
    lo, hi = 1e-4, 5.0
    s = 0.20
    for _ in range(60):
        price = _bs_call_fwd(F, K, s)
        diff = price - target_fwd
        if abs(diff) < 1e-8:
            return s
        v = _bs_vega_s(F, K, s)
        if v > 1e-10:
            step = diff / v
            s_new = s - step
        else:
            s_new = 0.5 * (lo + hi)
        if not (lo < s_new < hi):
            # bisection fallback
            if diff > 0:
                hi = s
            else:
                lo = s
            s_new = 0.5 * (lo + hi)
        else:
            if diff > 0:
                hi = s
            else:
                lo = s
        s = s_new
    return s


# --------------------------------------------------------------------------- #
# Forward / discount factor from put-call parity                              #
# --------------------------------------------------------------------------- #
def _forward_and_df(snap: ChainSnapshot, cfg: RNDConfig) -> tuple[float, float]:
    """
    Parity:  C - P = DF * (F - K)  =>  (C-P) = DF*F  -  DF*K.
    Regress (C-P) on K over near-ATM strikes -> slope=-DF, intercept=DF*F.
    Recovers BOTH forward and discount factor from the chain itself.
    """
    qs = snap.sorted_quotes()
    strikes = np.array([q.strike for q in qs])
    atm_idx = int(np.argmin(np.abs(strikes - snap.spot)))
    w = cfg.parity_atm_window
    lo = max(0, atm_idx - w)
    hi = min(len(qs), atm_idx + w + 1)
    sub = qs[lo:hi]
    K = np.array([q.strike for q in sub])
    cmp_ = np.array([q.call_mid - q.put_mid for q in sub])
    # Weight by quote tightness: ATM (tightest, cleanest parity) dominates,
    # noisy wings are down-weighted. Center K to keep it well-conditioned.
    wt = np.array([1.0 / (q.call_spread + q.put_spread) for q in sub])
    K0 = float(np.mean(K))
    x = K - K0
    sw = np.sqrt(wt)
    A = np.vstack([sw * np.ones_like(x), sw * x]).T   # weighted [intercept', slope']
    coef, *_ = np.linalg.lstsq(A, sw * cmp_, rcond=None)
    intercept_p, slope_p = coef
    DF = -slope_p
    if DF <= 0 or not np.isfinite(DF):
        DF = np.exp(-snap.r * snap.t_years)        # fallback to input rate
        F = snap.spot / DF
    else:
        F = K0 + intercept_p / DF                  # since intercept' = DF*(F - K0)
    return float(F), float(DF)


# --------------------------------------------------------------------------- #
# Result container                                                             #
# --------------------------------------------------------------------------- #
@dataclass
class RiskNeutralDensity:
    grid: np.ndarray               # strike grid
    pdf: np.ndarray                # risk-neutral density q(K), integrates to ~1
    forward: float
    discount_factor: float
    t_years: float
    arb_violation: float           # mass clipped for negativity (0 = clean butterfly)
    n_quotes_used: int

    # ---- distribution helpers ----
    def cdf(self) -> np.ndarray:
        c = np.cumsum(self.pdf) * (self.grid[1] - self.grid[0])
        return np.clip(c, 0.0, 1.0)

    def prob_above(self, K: float) -> float:
        c = self.cdf()
        return float(1.0 - np.interp(K, self.grid, c))

    def prob_below(self, K: float) -> float:
        return 1.0 - self.prob_above(K)

    def mean(self) -> float:
        dx = self.grid[1] - self.grid[0]
        return float(np.sum(self.grid * self.pdf) * dx)

    def std(self) -> float:
        dx = self.grid[1] - self.grid[0]
        m = self.mean()
        var = np.sum((self.grid - m) ** 2 * self.pdf) * dx
        return float(np.sqrt(max(var, 0.0)))

    def skew(self) -> float:
        dx = self.grid[1] - self.grid[0]
        m, sd = self.mean(), self.std()
        if sd <= 0:
            return 0.0
        return float(np.sum(((self.grid - m) / sd) ** 3 * self.pdf) * dx)

    def excess_kurtosis(self) -> float:
        dx = self.grid[1] - self.grid[0]
        m, sd = self.mean(), self.std()
        if sd <= 0:
            return 0.0
        return float(np.sum(((self.grid - m) / sd) ** 4 * self.pdf) * dx) - 3.0

    def quantile(self, p: float) -> float:
        return float(np.interp(p, self.cdf(), self.grid))

    def prob_touch(self, K: float, cap: float = 1.0) -> float:
        """
        Reflection-principle approx for P(touch K before close) on a ~driftless
        path: ~2x the probability of finishing beyond K. Sanity bound, not a
        substitute for a barrier model.
        """
        beyond = self.prob_above(K) if K >= self.forward else self.prob_below(K)
        return float(min(cap, 2.0 * beyond))


# --------------------------------------------------------------------------- #
# Core extraction                                                              #
# --------------------------------------------------------------------------- #
def extract_rnd(snap: ChainSnapshot, cfg: Optional[RNDConfig] = None) -> RiskNeutralDensity:
    cfg = cfg or RNDConfig()
    F, DF = _forward_and_df(snap, cfg)
    inv_disc = 1.0 / DF                                  # undiscount factor e^{rT}

    # Build OTM total-variance points: puts below F, calls above F.
    ks, ws, weights = [], [], []
    for q in snap.sorted_quotes():
        K = q.strike
        if K <= 0:
            continue
        if K >= F:
            mid, spread = q.call_mid, q.call_spread
            fwd_price = mid * inv_disc
        else:
            # put -> equivalent forward call value via parity: C = P + DF*(F-K)
            mid, spread = q.put_mid, q.put_spread
            fwd_price = (q.put_mid + DF * (F - K)) * inv_disc
        if mid < cfg.min_price:
            continue
        if spread / max(mid, 1e-6) > cfg.max_rel_spread:
            continue
        s = _implied_total_vol(fwd_price, F, K)
        if s is None or not np.isfinite(s):
            continue
        ks.append(np.log(K / F))
        ws.append(s * s)                                 # total variance w = s^2
        weights.append(1.0 / spread)                     # tighter markets count more

    if len(ks) < 5:
        raise ValueError(f"Only {len(ks)} usable strikes after filtering; chain too thin.")

    ks = np.asarray(ks)
    ws = np.asarray(ws)
    weights = np.asarray(weights)
    order = np.argsort(ks)
    ks, ws, weights = ks[order], ws[order], weights[order]

    # Smoothing spline on the (smooth) total-variance smile.
    spline = make_smoothing_spline(ks, ws, w=weights, lam=cfg.spline_lam)

    # Fine strike grid sized off a quick RN std estimate (ATM total vol).
    atm_w = float(spline(0.0))
    approx_sd = F * np.sqrt(max(atm_w, 1e-8))
    lo = max(cfg.grid_step, F - cfg.grid_pad_sigma * approx_sd)
    hi = F + cfg.grid_pad_sigma * approx_sd
    grid = np.arange(lo, hi, cfg.grid_step)

    # Reconstruct smooth forward call curve, then 2nd-difference -> density.
    kk = np.log(grid / F)
    w_grid = np.clip(spline(kk), 1e-8, None)
    s_grid = np.sqrt(w_grid)
    call_fwd = np.array([_bs_call_fwd(F, K, s) for K, s in zip(grid, s_grid)])
    pdf = np.gradient(np.gradient(call_fwd, grid), grid)   # d2C/dK2 = q(K)

    # Butterfly-arb cleanup: clip negatives, track how much mass that cost.
    neg_mass = float(-np.sum(pdf[pdf < 0]) * cfg.grid_step)
    pdf = np.clip(pdf, 0.0, None)
    area = np.sum(pdf) * cfg.grid_step
    if area > 0:
        pdf = pdf / area
    arb_violation = neg_mass / (neg_mass + 1.0)            # 0 clean .. ->1 ugly

    return RiskNeutralDensity(
        grid=grid, pdf=pdf, forward=F, discount_factor=DF,
        t_years=snap.t_years, arb_violation=arb_violation, n_quotes_used=len(ks),
    )


# --------------------------------------------------------------------------- #
# Physical-vs-risk-neutral edge  (the straddle_rich upgrade)                   #
# --------------------------------------------------------------------------- #
@dataclass
class EdgeReport:
    variance_ratio: float                  # var_RN / var_physical   ( >1 => premium rich )
    richness_signal: float                 # 0..1, drop-in for gate_scorer straddle_rich
    rn_std: float
    physical_std: float
    rn_skew: float
    per_strike_ev: dict[float, float]      # strike -> EV of SELLING that option (your measure)
    best_call_strike: Optional[float]
    best_put_strike: Optional[float]


def _physical_pdf_from_rnd(rnd: RiskNeutralDensity, vrp: float) -> np.ndarray:
    """
    Default physical density when the caller supplies none: squeeze the RND
    toward its mean so that var_phys = (1-vrp)*var_RN. Encodes the empirical
    vol-risk-premium (implied usually > realized) without inventing a shape.
    """
    F = rnd.forward
    scale = np.sqrt(max(1.0 - vrp, 1e-6))
    # Narrow the RND toward F: physical Y = F + scale*(X-F), var_Y=(1-vrp)var_X.
    # Evaluating p_Y on the grid requires the inverse map (multiply by scale here);
    # renormalization below absorbs the Jacobian.
    src_x = F + (rnd.grid - F) * scale
    phys = np.interp(rnd.grid, src_x, rnd.pdf, left=0.0, right=0.0)
    area = np.sum(phys) * (rnd.grid[1] - rnd.grid[0])
    return phys / area if area > 0 else phys


def compute_edge(
    rnd: RiskNeutralDensity,
    snap: ChainSnapshot,
    cfg: Optional[RNDConfig] = None,
    physical_pdf: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> EdgeReport:
    """
    EV of SELLING strike K under YOUR physical measure:
        EV_sell = market_premium  -  E_phys[payoff at K]
    Positive => the market is paying you more than the move you actually expect.
    `physical_pdf` is an optional callable(grid)->density; default uses the VRP
    haircut so the module is runnable standalone.
    """
    cfg = cfg or RNDConfig()
    dx = rnd.grid[1] - rnd.grid[0]
    F = rnd.forward

    if physical_pdf is not None:
        phys = np.asarray(physical_pdf(rnd.grid), dtype=float)
        a = np.sum(phys) * dx
        phys = phys / a if a > 0 else phys
    else:
        phys = _physical_pdf_from_rnd(rnd, cfg.vol_risk_premium)

    # moments
    rn_std = rnd.std()
    m_phys = float(np.sum(rnd.grid * phys) * dx)
    phys_std = float(np.sqrt(max(np.sum((rnd.grid - m_phys) ** 2 * phys) * dx, 0.0)))
    var_ratio = (rn_std ** 2) / max(phys_std ** 2, 1e-9)

    # squash variance ratio -> 0..1 signal (1.0 ~ richly priced, <0.5 ~ cheap)
    richness = float(1.0 / (1.0 + np.exp(-3.0 * (var_ratio - 1.0))))

    # per-strike EV of selling, evaluated at quoted strikes
    per_strike: dict[float, float] = {}
    best_call = (None, -np.inf)
    best_put = (None, -np.inf)
    for q in snap.sorted_quotes():
        K = q.strike
        if K >= F:                                   # selling a call
            premium = q.call_mid
            payoff = np.clip(rnd.grid - K, 0.0, None)
        else:                                        # selling a put
            premium = q.put_mid
            payoff = np.clip(K - rnd.grid, 0.0, None)
        exp_payoff = float(np.sum(payoff * phys) * dx) * rnd.discount_factor
        ev = premium - exp_payoff
        per_strike[K] = round(ev, 4)
        if K >= F and ev > best_call[1]:
            best_call = (K, ev)
        if K < F and ev > best_put[1]:
            best_put = (K, ev)

    return EdgeReport(
        variance_ratio=round(var_ratio, 4),
        richness_signal=round(richness, 4),
        rn_std=round(rn_std, 4),
        physical_std=round(phys_std, 4),
        rn_skew=round(rnd.skew(), 4),
        per_strike_ev=per_strike,
        best_call_strike=best_call[0],
        best_put_strike=best_put[0],
    )


# --------------------------------------------------------------------------- #
# Demo                                                                         #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # Synthetic 0DTE chain from a known put-skewed smile (s = sigma*sqrt(T)).
    F0, r0, T0 = 600.0, 0.05, 5.0 / (24 * 365)
    DF0 = np.exp(-r0 * T0)
    qs = []
    for K in np.arange(582, 619, 1.0):
        k = np.log(K / F0)
        s = max(0.0050 - 0.030 * k, 0.0008)
        cm = _bs_call_fwd(F0, K, s) * DF0
        pm = max(cm - DF0 * (F0 - K), 0.0)
        cm = max(cm, 0.0)
        h = 0.01 + 0.002 * max(cm, pm)
        qs.append(ChainQuote(float(K), max(cm - h, 0), cm + h, max(pm - h, 0), pm + h))

    snap = ChainSnapshot(qs, spot=600.1, t_years=T0, r=r0)
    rnd = extract_rnd(snap)
    print(f"forward {rnd.forward:.3f}  rn_std {rnd.std():.3f}  "
          f"skew {rnd.skew():+.3f}  arb {rnd.arb_violation:.3f}")
    print(f"P(S_T>603) {rnd.prob_above(603):.3f}  "
          f"5/50/95 {rnd.quantile(.05):.2f}/{rnd.quantile(.5):.2f}/{rnd.quantile(.95):.2f}")

    edge = compute_edge(rnd, snap)
    print(f"variance_ratio {edge.variance_ratio}  richness_signal {edge.richness_signal}")
    print(f"best call to sell {edge.best_call_strike}  best put {edge.best_put_strike}")
