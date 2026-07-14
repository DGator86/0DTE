"""
spread_selector.py
==================
The bridge between "this option is overpriced" (rnd_extractor) and "this is the
exact 0DTE trade." Generates every reasonable defined-risk structure, prices and
risk-scores each against your PHYSICAL density, and returns the best risk-adjusted
candidate -- or no trade.

Core rule
---------
Do NOT sell the highest-EV strike. Sell the structure with the highest EV per
unit of *tail risk*, after multiplicative safety gates:

    score = (EV / max_loss)
            * liquidity_score      # can we actually fill it
            * wall_safety          # is a gamma wall defending the short
            * gamma_safety         # is a short sitting in/under the flip
            * touch_safety         # how likely is the short to be tagged

Design
------
Uniform leg-based representation: a structure is a list of Legs. One numerical
payoff curve yields credit (from chain mids), max_loss, EV (vs physical pdf),
breakevens, and prob_profit. Greeks per leg come from inverting that leg's mid
to vol through the same forward model used by the extractor. This keeps every
family -- verticals, condors, flies, broken wings -- on identical, tested math.

Consumes: RiskNeutralDensity + EdgeReport (rnd_extractor), the option chain, and
a small GammaContext (spot/walls/flip/net_gex, e.g. sliced from MarketSnapshot).

NOT financial advice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np
from scipy.stats import norm

from rnd_extractor import (
    ChainSnapshot, RiskNeutralDensity, EdgeReport,
    _bs_call_fwd, _implied_total_vol, _physical_pdf_from_rnd, RNDConfig,
)


# --------------------------------------------------------------------------- #
# Structure representation                                                     #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Leg:
    strike: float
    kind: str       # 'C' or 'P'
    qty: int        # +1 long (buy), -1 short (sell)


@dataclass
class GammaContext:
    """Just the dealer-positioning fields the selector needs."""
    spot: float
    call_wall: float
    put_wall: float
    gamma_flip: float
    net_gex: float
    gex_pct_rank: float = 0.5            # |GEX| percentile; gates naked structures
    # When True, soft-exempt short_gamma_regime / short_put<=gamma_flip vetoes
    # so credit candidates can fill into a flip/wall pin (even if GEX < 0).
    pin_active: bool = False

    @classmethod
    def from_market_snapshot(cls, ms, pin_active: bool = False) -> "GammaContext":
        return cls(ms.spot, ms.call_wall, ms.put_wall, ms.gamma_flip, ms.net_gex,
                   getattr(ms, "gex_pct_rank", 0.5), pin_active=pin_active)


@dataclass
class SpreadCandidate:
    family: str
    short_strikes: tuple
    long_strikes: tuple
    legs: tuple
    credit: float
    max_loss: float
    ev: float
    ev_per_risk: float
    theta: float                 # position theta per day (>0 = collecting decay)
    gamma: float                 # position gamma (<0 = short gamma)
    prob_profit: float
    prob_touch_short: float
    distance_to_wall: float      # signed $ cushion of nearest short to its wall (+=safe)
    liquidity_score: float
    wall_safety: float
    gamma_safety: float
    touch_safety: float
    score: float
    passes_vetoes: bool
    veto_reasons: tuple
    risk_mode: str = "defined"       # defined | cash_secured | naked_stop_defined
    capital: float = 0.0             # collateral / margin at risk per share
    ev_per_capital: float = 0.0      # the right axis for capital-heavy structures
    size_cap: float = 1.0            # max fraction of normal size (naked << 1)
    stop_level: float = 0.0          # synthetic-wing stop for naked structures
    # Prediction Engine V2 / PR 6: executable-price panel (mid / natural /
    # expected / conservative + fees). None when quotes were incomplete.
    # Mid `credit` / `ev` remain the diagnostic defaults; when
    # SelectorConfig.use_executable_economics is True, `ev` is recomputed
    # from net expected-fill credit instead.
    execution: Optional[dict] = None
    # Prediction Engine V2 / PR 7: touch source + optional legacy reflection
    # value for shadow comparison when V2 touch_probability_fn is active.
    touch_source: str = "reflection"                 # "v2" | "reflection"
    legacy_prob_touch_short: Optional[float] = None
    # Prediction Engine V2 / PR 8: observation-only utility from the shadow
    # ranker. Never replaces `score` for live ranking until promotion.
    v2_utility_score: Optional[float] = None
    v2_candidate_id: Optional[str] = None


@dataclass
class SelectorConfig:
    # candidate generation
    spread_widths: tuple = (1.0, 2.0, 3.0, 5.0)     # $ wing widths to enumerate
    short_min_otm: float = 0.0015                    # shorts at least this % OTM
    short_max_otm: float = 0.030                     # ... and at most this % OTM
    condor_max_skew: float = 0.010                   # max |put-dist - call-dist| as % spot

    # vetoes (hard)
    min_ev: float = 0.0                              # require positive expected value
    max_loss_cap: float = 6.0                        # $ per share; gate ruinous width
    min_liquidity: float = 0.25
    max_touch_short: float = 0.55                    # reject if short too likely to be tagged
    veto_short_below_flip: bool = True               # the "491P under the 490 flip" rule
    # Degenerate-structure floor: a $0.01 credit condor or a fraction-of-a-cent
    # debit spread is unfillable noise, but epsilon-EV / epsilon-risk ratios
    # rank absurdly high and pollute both ranking and the journaled would-be
    # candidate. Require real premium on either side.
    min_credit: float = 0.05                         # $ per share collected (credit families)
    min_debit: float = 0.05                          # $ per share paid (debit families)

    # safety shaping (sigmoid scales, in $)
    wall_scale: float = 1.5
    flip_scale: float = 1.5

    # family priors (premium-selling bias; iron_fly low unless pinned)
    family_weight: dict = field(default_factory=lambda: {
        "put_credit": 1.00,
        "call_credit": 0.97,
        "iron_condor": 0.95,
        "broken_wing": 0.92,
        "iron_fly": 0.80,
        "naked_defended_call": 0.88,
        "cash_secured_put": 0.55,
        # debit families
        "long_call_spread": 0.90,
        "long_put_spread": 0.90,
        "long_call": 0.70,           # undiversified premium buying; lower weight
        "long_put": 0.70,
        "long_strangle": 0.85,
        "backspread_call": 0.80,
        "backspread_put": 0.80,
    })
    iron_fly_pin_bonus: float = 0.20                 # restored toward 1.0 when pinned+GEX ok
    pin_band: float = 0.0025                         # within this % of a high-gamma strike = pinned

    # ---- undefined-risk families (OFF by default; opt in deliberately) ----
    enable_naked: bool = False
    naked_min_wall_safety: float = 0.55              # short at/just above the wall qualifies
    naked_min_gex_rank: float = 0.70                 # only in strongly long-gamma regimes
    naked_call_stop_buffer: float = 0.0030           # stop = call_wall * (1 + buffer)
    naked_slippage: float = 0.05                     # $ cushion added to stop-defined max_loss
    naked_size_cap: float = 0.35                     # naked sizes much smaller than spreads
    csp_size_cap: float = 0.60
    naked_gap_multiplier: float = 2.0                # ranking haircut: stop can gap; risk > stop-defined

    # ---- directional / debit families ----
    enable_directional: bool = True      # include debit families when no target_families given
    long_min_otm: float = -0.005         # buying leg: slightly ITM ...
    long_max_otm: float = 0.015          # ... to 1.5% OTM (0DTE ~0.35-0.50Δ)

    top_n: int = 8

    # ---- execution cost (Prediction Engine V2, PR 6) ----
    # Always attach an ExecutionEstimate to each candidate. When True, candidate
    # EV / ranking / min_ev veto use net expected-fill credit; mid `credit` on
    # the candidate remains the diagnostic midpoint entry either way. Default
    # False keeps the legacy mid-EV ranker stable; V2 economic metrics
    # (journal settlement, TearSheet, candidate labels) use expected-fill P&L
    # whenever the estimate is present, independent of this flag.
    use_executable_economics: bool = False
    execution_cost: Optional[object] = None          # ExecutionCostConfig | None
    quote_age_seconds: Optional[float] = None        # chain age at decision time
    minutes_to_close: Optional[float] = None
    realized_vol: Optional[float] = None

    # ---- barrier touch (Prediction Engine V2, PR 7) ----
    # Optional override for short-strike touch probability. When set, must be a
    # callable(strike: float) -> float in [0, 1] (e.g. a BarrierTouchModel or
    # path-model frequency wrapped per strike). When None, the legacy RND
    # reflection approximation (rnd.prob_touch) is used — retained as fallback.
    touch_probability_fn: Optional[Callable[[float], float]] = None
    # When True and touch_probability_fn is set, also journal the legacy
    # reflection touch alongside the V2 value for shadow comparison.
    journal_legacy_touch: bool = True


# --------------------------------------------------------------------------- #
# Pricing / payoff helpers                                                     #
# --------------------------------------------------------------------------- #
def _chain_maps(chain: ChainSnapshot) -> tuple[dict, dict, dict]:
    cmid, pmid, spr = {}, {}, {}
    for q in chain.quotes:
        cmid[q.strike] = q.call_mid
        pmid[q.strike] = q.put_mid
        spr[q.strike] = (q.call_spread, q.put_spread)
    return cmid, pmid, spr


def _leg_mid(leg: Leg, cmid: dict, pmid: dict) -> Optional[float]:
    m = (cmid if leg.kind == "C" else pmid).get(leg.strike)
    return m


def _intrinsic(leg: Leg, S: np.ndarray) -> np.ndarray:
    if leg.kind == "C":
        return np.clip(S - leg.strike, 0.0, None)
    return np.clip(leg.strike - S, 0.0, None)


def _credit(legs, cmid, pmid) -> Optional[float]:
    # cash collected at entry: short legs add, long legs subtract
    total = 0.0
    for lg in legs:
        m = _leg_mid(lg, cmid, pmid)
        if m is None:
            return None
        total += -lg.qty * m
    return total


def _payoff_curve(legs, S: np.ndarray, credit: float) -> np.ndarray:
    pnl = np.full_like(S, credit)
    for lg in legs:
        pnl = pnl + lg.qty * _intrinsic(lg, S)
    return pnl


# --------------------------------------------------------------------------- #
# Per-strike implied vol (for Greeks), cached                                 #
# --------------------------------------------------------------------------- #
def _strike_vol(strike: float, kind: str, cmid: dict, pmid: dict,
                rnd: RiskNeutralDensity) -> Optional[float]:
    F, DF = rnd.forward, rnd.discount_factor
    inv = 1.0 / DF
    if kind == "C":
        m = cmid.get(strike)
        if m is None:
            return None
        fwd_price = m * inv
    else:
        m = pmid.get(strike)
        if m is None:
            return None
        fwd_price = (m + DF * (F - strike)) * inv      # put -> equivalent fwd call
    s = _implied_total_vol(fwd_price, F, strike)        # total vol s = sigma*sqrt(T)
    if s is None or not np.isfinite(s):
        return None
    return s / np.sqrt(rnd.t_years)                      # annualized sigma


def _leg_greeks(leg: Leg, sigma: float, rnd: RiskNeutralDensity, spot: float):
    """Return (gamma, theta_per_day) for a LONG unit; caller multiplies by qty."""
    T = rnd.t_years
    F = rnd.forward
    s = sigma * np.sqrt(T)
    if s <= 0:
        return 0.0, 0.0
    d1 = np.log(F / leg.strike) / s + 0.5 * s
    phi = norm.pdf(d1)
    gamma = phi / (spot * s)                              # d2V/dS2
    theta_year = -(spot * phi * sigma) / (2.0 * np.sqrt(T))  # dominant 0DTE decay term
    return gamma, theta_year / 365.0


# --------------------------------------------------------------------------- #
# Safety multipliers                                                           #
# --------------------------------------------------------------------------- #
def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def _wall_safety(short_calls, short_puts, ctx: GammaContext, cfg: SelectorConfig) -> tuple[float, float]:
    """
    A wall defends a short when it sits between spot and that short.
    Call short safe when short_call >= call_wall (wall rejects price first).
    Put  short safe when short_put  <= put_wall.
    Returns (safety_multiplier, signed_cushion_of_nearest_short).
    """
    safeties, cushions = [], []
    for kc in short_calls:
        cushion = kc - ctx.call_wall                       # +ve: short above wall = safe
        safeties.append(_sigmoid(cushion / cfg.wall_scale))
        cushions.append(cushion)
    for kp in short_puts:
        cushion = ctx.put_wall - kp                        # +ve: short below wall = safe
        safeties.append(_sigmoid(cushion / cfg.wall_scale))
        cushions.append(cushion)
    if not safeties:
        return 1.0, 0.0
    # nearest short to its wall drives both the multiplier and the reported distance
    idx = int(np.argmin(np.abs(cushions)))
    return float(min(safeties)), float(cushions[idx])


def _gamma_safety(short_puts, ctx: GammaContext, cfg: SelectorConfig) -> tuple[float, bool, bool]:
    """
    The flip mainly threatens the downside: a short put at/under the zero-gamma
    level is selling into the regime where hedging amplifies declines.
    Returns (safety_multiplier, short_put_below_flip, regime_short_gamma).
    The two booleans are distinct: a call spread has no short put (first False)
    but is still unsafe if the whole regime is short gamma (second True).
    """
    regime_short = ctx.net_gex <= 0
    if not short_puts:
        return (0.10 if regime_short else 1.0), False, regime_short
    lowest = min(short_puts)
    margin = lowest - ctx.gamma_flip                       # +ve: short above flip = safe
    below = lowest <= ctx.gamma_flip
    safety = 0.10 if regime_short else _sigmoid(margin / cfg.flip_scale)
    return safety, below, regime_short


# --------------------------------------------------------------------------- #
# Candidate evaluation                                                         #
# --------------------------------------------------------------------------- #
def _evaluate(family: str, legs: tuple, chain: ChainSnapshot, rnd: RiskNeutralDensity,
              phys: np.ndarray, ctx: GammaContext, cfg: SelectorConfig,
              vol_cache: dict) -> Optional[SpreadCandidate]:
    cmid, pmid, spr = _chain_maps(chain)
    credit = _credit(legs, cmid, pmid)
    if credit is None:
        return None
    is_debit = family in DEBIT_FAMILIES
    if not is_debit and credit <= 0:
        return None                                        # credit structure must collect premium

    # ---- executable economics (PR 6): always attach; optionally drive EV ----
    execution = None
    try:
        from execution_cost import (ExecutionCostConfig, estimate_execution,
                                    quotes_from_chain)
        cost_cfg = cfg.execution_cost or ExecutionCostConfig()
        # relative spread across legs (half-spread / mid), for the fill prior
        rel_spread = 0.0
        n_ok = 0
        for lg in legs:
            cs, ps = spr.get(lg.strike, (0.05, 0.05))
            leg_spr = cs if lg.kind == "C" else ps
            m = _leg_mid(lg, cmid, pmid) or 0.01
            rel_spread += (0.5 * leg_spr) / max(m, 0.02)
            n_ok += 1
        rel_spread = rel_spread / max(n_ok, 1)
        opt_px = abs(credit) / max(len(legs), 1)
        execution = estimate_execution(
            legs, quotes_from_chain(chain), family,
            cfg=cost_cfg,
            quote_age_seconds=cfg.quote_age_seconds,
            minutes_to_close=cfg.minutes_to_close,
            relative_spread=rel_spread,
            option_price=opt_px,
            realized_vol=cfg.realized_vol,
        )
    except Exception:
        execution = None

    # Entry credit used for the payoff / EV integral. Mid by default; net
    # expected-fill (after entry fees) when executable economics are on.
    entry_for_ev = credit
    exit_drag = 0.0
    if cfg.use_executable_economics and execution is not None:
        entry_for_ev = execution.net_expected_credit
        exit_drag = (execution.exit_slippage_expected
                     + execution.exit_fees_expected)

    grid = rnd.grid
    dx = grid[1] - grid[0]
    pnl = _payoff_curve(legs, grid, entry_for_ev) - exit_drag

    # ---- risk accounting: defined vs undefined-but-managed families ----
    # max_loss / capital stay on the MID payoff (the contractual defined risk
    # does not shrink just because we expect a worse fill).
    risk_mode, capital, stop_level = "defined", 0.0, 0.0
    naked = family in NAKED_FAMILIES
    pnl_mid = _payoff_curve(legs, grid, credit)
    if family == "cash_secured_put":
        # single short put, collateralized to zero. Defined: max_loss = K - credit.
        K = legs[0].strike
        max_loss = float(K - credit)
        capital = float(K)                              # full strike held as cash
        risk_mode = "cash_secured"
    elif family == "naked_defended_call":
        # single short call; the wall-break STOP is the synthetic long wing.
        K = legs[0].strike
        # stop sits above BOTH the wall and the short, so loss is always defined
        stop_level = max(ctx.call_wall, K) * (1.0 + cfg.naked_call_stop_buffer)
        max_loss = float((stop_level - K) - credit + cfg.naked_slippage)
        if max_loss <= 1e-6:
            return None
        capital = max_loss                              # economic at-risk to the stop
        risk_mode = "naked_stop_defined"
        # cap the loss at the stop so EV reflects the managed exit, not infinity
        pnl = np.maximum(pnl, -max_loss)
        pnl_mid = np.maximum(pnl_mid, -max_loss)
    else:
        max_loss = float(-min(pnl_mid.min(), 0.0))
        capital = max_loss
        if max_loss <= 1e-6:
            return None                                 # no defined risk / degenerate

    ev = float(np.sum(pnl * phys) * dx)
    ev_per_risk = ev / max_loss
    ev_per_capital = ev / capital if capital > 0 else 0.0

    prob_profit = float(np.sum(phys[pnl > 0]) * dx)

    short_calls = [lg.strike for lg in legs if lg.kind == "C" and lg.qty < 0]
    short_puts = [lg.strike for lg in legs if lg.kind == "P" and lg.qty < 0]
    long_strikes = tuple(sorted(lg.strike for lg in legs if lg.qty > 0))
    short_strikes = tuple(sorted(short_calls + short_puts))

    # touch risk: most exposed short
    # Prefer calibrated V2 path/barrier model when provided; else RND reflection.
    legacy_touches = [rnd.prob_touch(k) for k in short_strikes] or [0.0]
    legacy_prob_touch = float(max(legacy_touches))
    if cfg.touch_probability_fn is not None and short_strikes:
        touches = [float(cfg.touch_probability_fn(k)) for k in short_strikes]
        prob_touch_short = float(max(touches)) if touches else 0.0
        touch_source = "v2"
    else:
        prob_touch_short = legacy_prob_touch
        touch_source = "reflection"
    touch_safety = float(np.clip(1.0 - prob_touch_short, 0.0, 1.0))
    legacy_for_journal = (
        round(legacy_prob_touch, 4)
        if (cfg.journal_legacy_touch and touch_source == "v2") else None
    )

    # greeks
    gamma_tot = theta_tot = 0.0
    for lg in legs:
        key = (lg.strike, lg.kind)
        if key not in vol_cache:
            vol_cache[key] = _strike_vol(lg.strike, lg.kind, cmid, pmid, rnd)
        sig = vol_cache[key]
        if sig is None:
            continue
        g, th = _leg_greeks(lg, sig, rnd, ctx.spot)
        gamma_tot += lg.qty * g
        theta_tot += lg.qty * th

    # liquidity: penalize wide legs
    rel = 0.0
    for lg in legs:
        cs, ps = spr.get(lg.strike, (0.05, 0.05))
        leg_spr = cs if lg.kind == "C" else ps
        m = _leg_mid(lg, cmid, pmid) or 0.01
        rel += leg_spr / max(m, 0.02)
    liquidity_score = float(1.0 / (1.0 + rel))

    if is_debit:
        # Debit (long) structures: wall/flip don't constrain the long side
        wall_safety, dist_to_wall = 1.0, 0.0
        gamma_safety, short_below_flip, regime_short = 1.0, False, False
    else:
        wall_safety, dist_to_wall = _wall_safety(short_calls, short_puts, ctx, cfg)
        gamma_safety, short_below_flip, regime_short = _gamma_safety(short_puts, ctx, cfg)

    # ---- vetoes ----
    reasons = []
    if ev <= cfg.min_ev:
        reasons.append(f"EV<={cfg.min_ev:g}")
    if is_debit:
        if -credit < cfg.min_debit:
            reasons.append(f"debit {-credit:.2f}<{cfg.min_debit:.2f}")
    elif credit < cfg.min_credit:
        reasons.append(f"credit {credit:.2f}<{cfg.min_credit:.2f}")
    if not naked and max_loss > cfg.max_loss_cap:
        reasons.append(f"max_loss {max_loss:.2f}>cap")
    if liquidity_score < cfg.min_liquidity:
        reasons.append("illiquid")
    if not is_debit:
        if prob_touch_short > cfg.max_touch_short:
            reasons.append(f"touch {prob_touch_short:.2f}>max({touch_source})")
        # Pin soft-exempt: allow credit into short-gamma / at-flip pins.
        if regime_short and not getattr(ctx, "pin_active", False):
            reasons.append("short_gamma_regime")
        if (cfg.veto_short_below_flip and short_below_flip
                and not getattr(ctx, "pin_active", False)):
            reasons.append("short_put<=gamma_flip")

    # ---- extra hard gating for undefined-risk families ----
    size_cap = 1.0
    if naked:
        size_cap = cfg.naked_size_cap if family == "naked_defended_call" else cfg.csp_size_cap
        if ctx.net_gex <= 0:
            reasons.append("naked_requires_long_gamma")
        if ctx.gex_pct_rank < cfg.naked_min_gex_rank:
            reasons.append(f"naked_gex_rank {ctx.gex_pct_rank:.2f}<{cfg.naked_min_gex_rank:.2f}")
        if wall_safety < cfg.naked_min_wall_safety:
            reasons.append(f"naked_wall_undefended {wall_safety:.2f}")
        if ctx.spot < ctx.gamma_flip:
            reasons.append("naked_below_flip")
    passes = len(reasons) == 0

    # ---- score ----
    # naked families carry gap risk the stop-defined max_loss understates, so we
    # haircut their effective risk for RANKING only (reported max_loss stays true)
    fam_w = cfg.family_weight.get(family, 0.9)
    risk_for_score = max_loss * (cfg.naked_gap_multiplier if naked else 1.0)
    ev_per_risk_scored = ev / risk_for_score if risk_for_score > 0 else 0.0
    if is_debit:
        score = max(ev_per_risk_scored, 0.0) * liquidity_score * fam_w
    else:
        score = (max(ev_per_risk_scored, 0.0) * liquidity_score * wall_safety
                 * gamma_safety * touch_safety * fam_w)

    return SpreadCandidate(
        family=family, short_strikes=short_strikes, long_strikes=long_strikes,
        legs=legs, credit=round(credit, 4), max_loss=round(max_loss, 4),
        ev=round(ev, 4), ev_per_risk=round(ev_per_risk, 4),
        theta=round(theta_tot, 4), gamma=round(gamma_tot, 6),
        prob_profit=round(prob_profit, 4), prob_touch_short=round(prob_touch_short, 4),
        distance_to_wall=round(dist_to_wall, 3),
        liquidity_score=round(liquidity_score, 4), wall_safety=round(wall_safety, 4),
        gamma_safety=round(gamma_safety, 4), touch_safety=round(touch_safety, 4),
        score=round(score, 6), passes_vetoes=passes, veto_reasons=tuple(reasons),
        risk_mode=risk_mode, capital=round(capital, 4),
        ev_per_capital=round(ev_per_capital, 6), size_cap=round(size_cap, 3),
        stop_level=round(stop_level, 3),
        execution=(execution.to_dict() if execution is not None else None),
        touch_source=touch_source,
        legacy_prob_touch_short=legacy_for_journal,
    )


def _normalize_phys(
    rnd: RiskNeutralDensity,
    physical_pdf: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> np.ndarray:
    """Physical density on the RND grid (same measure as select_spreads)."""
    if physical_pdf is not None:
        phys = np.asarray(physical_pdf(rnd.grid), dtype=float)
        a = np.sum(phys) * (rnd.grid[1] - rnd.grid[0])
        return phys / a if a > 0 else phys
    return _physical_pdf_from_rnd(rnd, RNDConfig().vol_risk_premium)


def reprice_candidates(
    candidates: Sequence[SpreadCandidate],
    chain: ChainSnapshot,
    rnd: RiskNeutralDensity,
    ctx: GammaContext,
    cfg: Optional[SelectorConfig] = None,
    physical_pdf: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> list[SpreadCandidate]:
    """
    Re-evaluate candidate *geometry* under a specific physical density.

    Shared-universe ticks freeze legs/family/identity once, then each decide()
    branch (live tilt, V2 shadow EV, pin counterfactual) must reprice EV,
    score, vetoes, and execution economics under its own density. Identity
    attributes (``candidate_id`` / aliases) are copied onto the fresh objects.
    """
    cfg = cfg or SelectorConfig()
    phys = _normalize_phys(rnd, physical_pdf)
    vol_cache: dict = {}
    out: list[SpreadCandidate] = []
    for src in candidates:
        family = getattr(src, "family", None)
        legs = getattr(src, "legs", None)
        if not family or not legs:
            continue
        fresh = _evaluate(
            str(family), tuple(legs), chain, rnd, phys, ctx, cfg, vol_cache)
        if fresh is None:
            continue
        for attr in ("candidate_id", "v2_candidate_id", "_v2_candidate_id",
                     "v2_utility_score"):
            val = getattr(src, attr, None)
            if val is None:
                continue
            try:
                setattr(fresh, attr, val)
            except Exception:
                try:
                    object.__setattr__(fresh, attr, val)
                except Exception:
                    pass
        out.append(fresh)
    return out


# --------------------------------------------------------------------------- #
# Candidate generation                                                         #
# --------------------------------------------------------------------------- #
def _otm_strikes(chain: ChainSnapshot, F: float, side: str, cfg: SelectorConfig) -> list[float]:
    ks = sorted(q.strike for q in chain.quotes)
    out = []
    for k in ks:
        rel = (k - F) / F if side == "call" else (F - k) / F
        if cfg.short_min_otm <= rel <= cfg.short_max_otm:
            out.append(k)
    return out


def _long_strikes(chain: ChainSnapshot, F: float, side: str, cfg: SelectorConfig) -> list[float]:
    """Strikes in the directional buying range: slightly ITM to slightly OTM."""
    ks = sorted(q.strike for q in chain.quotes)
    out = []
    for k in ks:
        rel = (k - F) / F if side == "call" else (F - k) / F
        if cfg.long_min_otm <= rel <= cfg.long_max_otm:
            out.append(k)
    return out


NAKED_FAMILIES = {"naked_defended_call", "cash_secured_put"}

DEBIT_FAMILIES: frozenset = frozenset({
    "long_call_spread", "long_put_spread", "long_call", "long_put",
    "long_strangle", "backspread_call", "backspread_put",
})

# Maps decision_matrix structure codes → selector family name sets.
# Used by live_feed_adapter.build_ticket() and decision_engine.decide()
# to limit spread generation to the families implied by the regime decision.
STRUCTURE_TO_FAMILIES: dict[str, frozenset] = {
    "IC":  frozenset({"iron_condor"}),
    "PCS": frozenset({"put_credit"}),
    "CCS": frozenset({"call_credit"}),
    "IF":  frozenset({"iron_fly"}),
    "LCS": frozenset({"long_call_spread"}),
    "LPS": frozenset({"long_put_spread"}),
    "LC":  frozenset({"long_call"}),
    "LP":  frozenset({"long_put"}),
    "STG": frozenset({"long_strangle"}),
    "BKS": frozenset({"backspread_call", "backspread_put"}),
}


def _gen_cash_secured_put(chain, F, ctx, cfg):
    """Single short put, collateralized to zero. Defined risk = strike - credit."""
    out = []
    for ks_short in _otm_strikes(chain, F, "put", cfg):
        out.append(("cash_secured_put", (Leg(ks_short, "P", -1),)))
    return out


def _gen_naked_defended_call(chain, F, ctx, cfg):
    """Single short call, risk capped by a wall-break stop (the synthetic wing).
    Only generate strikes at/above the call wall, where the wall does the work."""
    out = []
    for ks_short in _otm_strikes(chain, F, "call", cfg):
        if ks_short >= ctx.call_wall - 1e-9:          # short must sit at/above the wall
            out.append(("naked_defended_call", (Leg(ks_short, "C", -1),)))
    return out


def _gen_verticals(chain, F, ctx, cfg, kind):
    ks = set(q.strike for q in chain.quotes)
    side = "call" if kind == "C" else "put"
    fam = "call_credit" if kind == "C" else "put_credit"
    out = []
    for ks_short in _otm_strikes(chain, F, side, cfg):
        for w in cfg.spread_widths:
            ks_long = ks_short + w if kind == "C" else ks_short - w
            if ks_long in ks:
                legs = (Leg(ks_short, kind, -1), Leg(ks_long, kind, +1))
                out.append((fam, legs))
    return out


def _gen_condors(chain, F, ctx, cfg):
    ks = set(q.strike for q in chain.quotes)
    puts = _otm_strikes(chain, F, "put", cfg)
    calls = _otm_strikes(chain, F, "call", cfg)
    out = []
    for w in cfg.spread_widths:
        for kp in puts:
            kpl = kp - w
            if kpl not in ks:
                continue
            for kc in calls:
                kcl = kc + w
                if kcl not in ks:
                    continue
                skew = abs((F - kp) - (kc - F)) / F
                if skew > cfg.condor_max_skew:
                    continue
                legs = (Leg(kp, "P", -1), Leg(kpl, "P", +1),
                        Leg(kc, "C", -1), Leg(kcl, "C", +1))
                out.append(("iron_condor", legs))
    return out


def _gen_iron_fly(chain, F, cfg):
    ks = sorted(q.strike for q in chain.quotes)
    atm = min(ks, key=lambda k: abs(k - F))
    out = []
    for w in cfg.spread_widths:
        kpl, kcl = atm - w, atm + w
        if kpl in ks and kcl in ks:
            legs = (Leg(atm, "P", -1), Leg(kpl, "P", +1),
                    Leg(atm, "C", -1), Leg(kcl, "C", +1))
            out.append(("iron_fly", legs))
    return out


def _gen_broken_wing(chain, F, cfg):
    """Put broken-wing fly: wider lower wing than upper -> credit, no upside risk."""
    ks = set(q.strike for q in chain.quotes)
    ks_sorted = sorted(ks)
    body = min(ks_sorted, key=lambda k: abs(k - F))     # short body near ATM
    out = []
    for near in cfg.spread_widths:
        for far in cfg.spread_widths:
            if far <= near:
                continue
            upper = body + near                          # narrow upper wing
            lower = body - far                           # wide lower wing
            if upper in ks and lower in ks:
                legs = (Leg(upper, "P", +1), Leg(body, "P", -2), Leg(lower, "P", +1))
                out.append(("broken_wing", legs))
    return out


def _gen_long_call_spread(chain: ChainSnapshot, F: float, ctx: GammaContext,
                          cfg: SelectorConfig) -> list:
    """Buy call near ATM, sell call at K+w (bull debit spread)."""
    ks = set(q.strike for q in chain.quotes)
    out = []
    for k_buy in _long_strikes(chain, F, "call", cfg):
        for w in cfg.spread_widths:
            k_sell = k_buy + w
            if k_sell in ks:
                out.append(("long_call_spread", (Leg(k_buy, "C", +1), Leg(k_sell, "C", -1))))
    return out


def _gen_long_put_spread(chain: ChainSnapshot, F: float, ctx: GammaContext,
                         cfg: SelectorConfig) -> list:
    """Buy put near ATM, sell put at K-w (bear debit spread)."""
    ks = set(q.strike for q in chain.quotes)
    out = []
    for k_buy in _long_strikes(chain, F, "put", cfg):
        for w in cfg.spread_widths:
            k_sell = k_buy - w
            if k_sell in ks:
                out.append(("long_put_spread", (Leg(k_buy, "P", +1), Leg(k_sell, "P", -1))))
    return out


def _gen_long_call(chain: ChainSnapshot, F: float, cfg: SelectorConfig) -> list:
    """Single long call near ATM (convex bull, unlimited upside)."""
    return [("long_call", (Leg(k, "C", +1),)) for k in _long_strikes(chain, F, "call", cfg)]


def _gen_long_put(chain: ChainSnapshot, F: float, cfg: SelectorConfig) -> list:
    """Single long put near ATM (convex bear, unlimited downside capture)."""
    return [("long_put", (Leg(k, "P", +1),)) for k in _long_strikes(chain, F, "put", cfg)]


def _gen_long_strangle(chain: ChainSnapshot, F: float, cfg: SelectorConfig) -> list:
    """Buy OTM call + OTM put (long vol, expects realized > implied)."""
    call_strikes = _otm_strikes(chain, F, "call", cfg)
    put_strikes = _otm_strikes(chain, F, "put", cfg)
    out = []
    for kc in call_strikes:
        call_dist = (kc - F) / F
        for kp in put_strikes:
            put_dist = (F - kp) / F
            # only roughly symmetric strangles (within 50% relative distance)
            if call_dist > 0 and abs(call_dist - put_dist) / call_dist < 0.5:
                out.append(("long_strangle", (Leg(kc, "C", +1), Leg(kp, "P", +1))))
    return out


def _gen_backspread_call(chain: ChainSnapshot, F: float, cfg: SelectorConfig) -> list:
    """Sell 1 call near ATM, buy 2 calls OTM (net long gamma, benefits from up-breakout)."""
    ks = set(q.strike for q in chain.quotes)
    out = []
    for k_sell in _long_strikes(chain, F, "call", cfg):
        for k_buy in _otm_strikes(chain, F, "call", cfg):
            if k_buy > k_sell and k_buy in ks:
                out.append(("backspread_call", (Leg(k_sell, "C", -1), Leg(k_buy, "C", +2))))
    return out


def _gen_backspread_put(chain: ChainSnapshot, F: float, cfg: SelectorConfig) -> list:
    """Sell 1 put near ATM, buy 2 puts OTM (net long gamma, benefits from down-breakout)."""
    ks = set(q.strike for q in chain.quotes)
    out = []
    for k_sell in _long_strikes(chain, F, "put", cfg):
        for k_buy in _otm_strikes(chain, F, "put", cfg):
            if k_buy < k_sell and k_buy in ks:
                out.append(("backspread_put", (Leg(k_sell, "P", -1), Leg(k_buy, "P", +2))))
    return out


def _is_pinned(F: float, ctx: GammaContext, cfg: SelectorConfig) -> bool:
    """True when forward is near a wall OR the gamma flip.

    Does not require net_gex > 0 — negative-GEX pins still want the iron-fly
    family-weight bonus (pin_regime / policy decide whether to allow credit).
    """
    nearest_wall = min(abs(F - ctx.call_wall), abs(F - ctx.put_wall))
    wall_pin = nearest_wall / F <= cfg.pin_band if F else False
    flip = float(ctx.gamma_flip or 0.0)
    flip_pin = (abs(F - flip) / F <= cfg.pin_band) if (F and flip) else False
    return bool(wall_pin or flip_pin)


# --------------------------------------------------------------------------- #
# Top-level                                                                    #
# --------------------------------------------------------------------------- #
@dataclass
class SelectionResult:
    best: Optional[SpreadCandidate]
    ranked: list
    no_trade_reason: str = ""
    # Full evaluated set (passing + vetoed) for V2 shadow ranking / audit.
    # Legacy `ranked` remains the top_n slice used by diagnostics.
    all_candidates: list = field(default_factory=list)


def select_spreads(
    chain: ChainSnapshot,
    rnd: RiskNeutralDensity,
    edge: EdgeReport,
    ctx: GammaContext,
    cfg: Optional[SelectorConfig] = None,
    physical_pdf: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    target_families: Optional[frozenset] = None,
) -> SelectionResult:
    """
    target_families — if given, only generate candidates from that family set.
    Pass None (default) to generate every enabled family.
    """
    cfg = cfg or SelectorConfig()

    # physical density on the rnd grid (same measure used by compute_edge)
    phys = _normalize_phys(rnd, physical_pdf)

    F = rnd.forward
    pinned = _is_pinned(F, ctx, cfg)
    if pinned:
        cfg.family_weight = dict(cfg.family_weight)
        cfg.family_weight["iron_fly"] = min(1.0, cfg.family_weight["iron_fly"] + cfg.iron_fly_pin_bonus)

    gen = target_families  # None = generate all enabled families; frozenset = only those

    specs = []
    if gen is None or "put_credit" in gen:
        specs += _gen_verticals(chain, F, ctx, cfg, "P")
    if gen is None or "call_credit" in gen:
        specs += _gen_verticals(chain, F, ctx, cfg, "C")
    if gen is None or "iron_condor" in gen:
        specs += _gen_condors(chain, F, ctx, cfg)
    if gen is None or "iron_fly" in gen:
        specs += _gen_iron_fly(chain, F, cfg)
    if gen is None or "broken_wing" in gen:
        specs += _gen_broken_wing(chain, F, cfg)
    if cfg.enable_naked and (gen is None or "cash_secured_put" in gen):
        specs += _gen_cash_secured_put(chain, F, ctx, cfg)
    if cfg.enable_naked and (gen is None or "naked_defended_call" in gen):
        specs += _gen_naked_defended_call(chain, F, ctx, cfg)
    # directional / debit families: always when targeted; else only if enabled
    if gen is not None or cfg.enable_directional:
        if gen is None or "long_call_spread" in gen:
            specs += _gen_long_call_spread(chain, F, ctx, cfg)
        if gen is None or "long_put_spread" in gen:
            specs += _gen_long_put_spread(chain, F, ctx, cfg)
        if gen is None or "long_call" in gen:
            specs += _gen_long_call(chain, F, cfg)
        if gen is None or "long_put" in gen:
            specs += _gen_long_put(chain, F, cfg)
        if gen is None or "long_strangle" in gen:
            specs += _gen_long_strangle(chain, F, cfg)
        if gen is None or "backspread_call" in gen:
            specs += _gen_backspread_call(chain, F, cfg)
        if gen is None or "backspread_put" in gen:
            specs += _gen_backspread_put(chain, F, cfg)

    vol_cache: dict = {}
    cands = []
    for fam, legs in specs:
        c = _evaluate(fam, legs, chain, rnd, phys, ctx, cfg, vol_cache)
        if c is not None:
            cands.append(c)

    passing = [c for c in cands if c.passes_vetoes]
    passing.sort(key=lambda c: c.score, reverse=True)

    if not passing:
        n_pos = sum(1 for c in cands if c.ev > 0)
        reason = ("no positive-EV structure" if n_pos == 0
                  else f"{n_pos} positive-EV structures all failed vetoes")
        return SelectionResult(best=None, ranked=cands[:cfg.top_n],
                               no_trade_reason=reason, all_candidates=cands)

    return SelectionResult(best=passing[0], ranked=passing[:cfg.top_n],
                           all_candidates=cands)


# --------------------------------------------------------------------------- #
# Demo                                                                         #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    from rnd_extractor import ChainQuote, extract_rnd, compute_edge

    F0, r0, T0 = 600.0, 0.05, 5.0 / (24 * 365)
    DF0 = np.exp(-r0 * T0)
    qs = []
    for K in np.arange(580, 621, 1.0):
        k = np.log(K / F0)
        s = max(0.0050 - 0.030 * k, 0.0008)
        cm = _bs_call_fwd(F0, K, s) * DF0
        pm = max(cm - DF0 * (F0 - K), 0.0)
        cm = max(cm, 0.0)
        h = 0.01 + 0.002 * max(cm, pm)
        qs.append(ChainQuote(float(K), max(cm - h, 0), cm + h, max(pm - h, 0), pm + h))

    chain = ChainSnapshot(qs, spot=600.1, t_years=T0, r=r0)
    rnd = extract_rnd(chain)
    edge = compute_edge(rnd, chain)

    # gamma context: call wall 606 (resistance), put wall 595 (support), flip 593
    ctx = GammaContext(spot=600.1, call_wall=606.0, put_wall=595.0,
                       gamma_flip=593.0, net_gex=3.5e9)

    res = select_spreads(chain, rnd, edge, ctx)
    print(f"forward {rnd.forward:.2f}  richness {edge.richness_signal}  pinned={_is_pinned(rnd.forward, ctx, SelectorConfig())}\n")
    if res.best is None:
        print("NO TRADE:", res.no_trade_reason)
    else:
        print("Top candidates (risk-adjusted):")
        print(f"{'family':<12}{'shorts':<14}{'cr':>6}{'maxL':>7}{'EV':>8}"
              f"{'EV/risk':>9}{'pp':>6}{'tch':>6}{'wall':>6}{'gam':>6}{'score':>9}")
        for c in res.ranked:
            print(f"{c.family:<12}{str(c.short_strikes):<14}{c.credit:>6.2f}{c.max_loss:>7.2f}"
                  f"{c.ev:>8.3f}{c.ev_per_risk:>9.3f}{c.prob_profit:>6.2f}{c.prob_touch_short:>6.2f}"
                  f"{c.wall_safety:>6.2f}{c.gamma_safety:>6.2f}{c.score:>9.4f}")
        b = res.best
        print(f"\nBEST: {b.family} shorts={b.short_strikes} longs={b.long_strikes}")
        print(f"  credit {b.credit}  max_loss {b.max_loss}  EV {b.ev}  EV/risk {b.ev_per_risk}")
        print(f"  theta/day {b.theta}  gamma {b.gamma}  P(profit) {b.prob_profit}  "
              f"P(touch short) {b.prob_touch_short}")
