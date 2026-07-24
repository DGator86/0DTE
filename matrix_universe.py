"""
matrix_universe.py
==================
A combinatoric, evolving universe generator built from stacked Markov chains —
the "full universe simulator" behind dojo.py's sparring phase.

The hierarchy (bottom-up, each layer conditioning the one below):

  Layer 1 — VARIABLES.   Each driver variable (net GEX, realized vol, VRP,
            skew, drift) runs its own 3-state Markov chain {low, mid, high}
            with its own RNG stream. The chain picks a target; an OU process
            moves the numeric value toward it, so variables evolve smoothly
            but switch character stochastically.
  Layer 2 — VARIABLE SETS → REGIMES.  The per-variable transition matrices
            are conditioned on the current intraday REGIME (pin, drift_up,
            drift_down, compression, breakout): a pin regime pulls the GEX
            chain toward its high-positive state, a breakout regime pulls it
            negative, and so on. The regime itself is a minute-scale Markov
            chain.
  Layer 3 — REGIMES → MARKETS.  The regime transition matrix is conditioned
            on the day's MARKET ARCHETYPE (calm_pin, grind_up, grind_down,
            range_chop, vol_expansion, squeeze_melt_up, crash, gap_shock),
            which is itself a day-scale Markov chain.

Combinatorics: UniverseCatalog enumerates the situation lattice
(start archetype × regime persistence tilt × vol multiplier) into seeded,
fully deterministic UniverseSpec entries, and tracks COVERAGE — which
(archetype × regime) cells the generated universes actually visited — so
"knows what to do in every situation" becomes a measurable claim instead of
a hope.

Evolution: UniverseCatalog.evolve() takes per-archetype performance from the
previous generation and re-weights the next one toward the archetypes the
pipeline handled worst (spar hardest where weakest), with seeded Dirichlet
perturbation of the transition matrices so no two generations are identical.

MarkovWorldFeed implements the unified_loop DataFeed protocol (timestamps /
snapshot / settlement_price), so backtest.run_backtest and
walk_forward.run_walk_forward run on a generated universe exactly as they do
on recorded ticks. Every tick is labeled with its (archetype, regime) so the
dojo can attribute P&L by situation.

Honest caveat: this is a model built from the system's own thesis about how
dealer positioning couples to price. Surviving every archetype here does NOT
prove live edge — but failing one is a real, attributable weakness.

NOT financial advice.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import itertools
import math
from dataclasses import dataclass, field, replace
from typing import Optional
from zoneinfo import ZoneInfo

import numpy as np

from gate_scorer import MarketSnapshot
from gex_window import GexRankWindow
from massive_feed import _bar_technicals, _session_vwap_and_reversions
from resample import RawBars
from rnd_extractor import ChainQuote, ChainSnapshot, _bs_call_fwd, MINUTES_PER_YEAR
from unified_loop import TickSnapshot

ET = ZoneInfo("America/New_York")
MIN_PER_DAY = 390

# --------------------------------------------------------------------------- #
# Layer 3 — market archetypes (day-scale chain)                               #
# --------------------------------------------------------------------------- #
ARCHETYPES = (
    "calm_pin",         # long-gamma, vol suppressed, price pinned
    "grind_up",         # persistent low-vol upward drift
    "grind_down",       # persistent low-vol downward drift
    "range_chop",       # two-sided, regime flips intraday
    "vol_expansion",    # short-gamma, elevated vol, direction unstable
    "squeeze_melt_up",  # short-gamma chase higher, vol up AND price up
    "crash",            # short-gamma cascade lower, heavy vol, gap risk
    "gap_shock",        # large overnight gap then intraday normalization
)

# Row-stochastic day-to-day transitions. Ordinary tapes persist; stress tapes
# resolve back toward calm — matching how vol regimes actually decay.
_ARCH_TRANSITION: dict[str, dict[str, float]] = {
    "calm_pin":        {"calm_pin": .55, "grind_up": .12, "grind_down": .08,
                        "range_chop": .12, "vol_expansion": .08,
                        "squeeze_melt_up": .02, "crash": .01, "gap_shock": .02},
    "grind_up":        {"calm_pin": .20, "grind_up": .45, "grind_down": .05,
                        "range_chop": .12, "vol_expansion": .08,
                        "squeeze_melt_up": .06, "crash": .02, "gap_shock": .02},
    "grind_down":      {"calm_pin": .15, "grind_up": .05, "grind_down": .40,
                        "range_chop": .12, "vol_expansion": .15,
                        "squeeze_melt_up": .02, "crash": .08, "gap_shock": .03},
    "range_chop":      {"calm_pin": .22, "grind_up": .10, "grind_down": .10,
                        "range_chop": .40, "vol_expansion": .10,
                        "squeeze_melt_up": .03, "crash": .02, "gap_shock": .03},
    "vol_expansion":   {"calm_pin": .10, "grind_up": .05, "grind_down": .12,
                        "range_chop": .15, "vol_expansion": .35,
                        "squeeze_melt_up": .08, "crash": .10, "gap_shock": .05},
    "squeeze_melt_up": {"calm_pin": .15, "grind_up": .25, "grind_down": .03,
                        "range_chop": .12, "vol_expansion": .20,
                        "squeeze_melt_up": .20, "crash": .02, "gap_shock": .03},
    "crash":           {"calm_pin": .05, "grind_up": .03, "grind_down": .20,
                        "range_chop": .07, "vol_expansion": .30,
                        "squeeze_melt_up": .05, "crash": .22, "gap_shock": .08},
    "gap_shock":       {"calm_pin": .25, "grind_up": .10, "grind_down": .10,
                        "range_chop": .20, "vol_expansion": .20,
                        "squeeze_melt_up": .05, "crash": .05, "gap_shock": .05},
}

# --------------------------------------------------------------------------- #
# Layer 2 — intraday regimes (minute-scale chain, conditioned on archetype)   #
# --------------------------------------------------------------------------- #
REGIMES = ("pin", "drift_up", "drift_down", "compression", "breakout")

# Per-archetype regime transition rows. Persistence lives on the diagonal;
# UniverseSpec.persistence_tilt scales it (see _tilt_row). Rows are
# per-minute, so a .985 diagonal means ~65-minute mean regime duration.
_REGIME_TRANSITION: dict[str, dict[str, dict[str, float]]] = {
    "calm_pin": {
        "pin":         {"pin": .990, "drift_up": .002, "drift_down": .002, "compression": .005, "breakout": .001},
        "drift_up":    {"pin": .030, "drift_up": .960, "drift_down": .002, "compression": .005, "breakout": .003},
        "drift_down":  {"pin": .030, "drift_up": .002, "drift_down": .960, "compression": .005, "breakout": .003},
        "compression": {"pin": .015, "drift_up": .002, "drift_down": .002, "compression": .978, "breakout": .003},
        "breakout":    {"pin": .040, "drift_up": .010, "drift_down": .010, "compression": .010, "breakout": .930},
    },
    "grind_up": {
        "pin":         {"pin": .975, "drift_up": .015, "drift_down": .002, "compression": .005, "breakout": .003},
        "drift_up":    {"pin": .010, "drift_up": .980, "drift_down": .002, "compression": .005, "breakout": .003},
        "drift_down":  {"pin": .020, "drift_up": .020, "drift_down": .950, "compression": .005, "breakout": .005},
        "compression": {"pin": .008, "drift_up": .015, "drift_down": .002, "compression": .970, "breakout": .005},
        "breakout":    {"pin": .015, "drift_up": .030, "drift_down": .005, "compression": .010, "breakout": .940},
    },
    "grind_down": {
        "pin":         {"pin": .975, "drift_up": .002, "drift_down": .015, "compression": .005, "breakout": .003},
        "drift_up":    {"pin": .020, "drift_up": .950, "drift_down": .020, "compression": .005, "breakout": .005},
        "drift_down":  {"pin": .010, "drift_up": .002, "drift_down": .980, "compression": .005, "breakout": .003},
        "compression": {"pin": .008, "drift_up": .002, "drift_down": .015, "compression": .970, "breakout": .005},
        "breakout":    {"pin": .015, "drift_up": .005, "drift_down": .030, "compression": .010, "breakout": .940},
    },
    "range_chop": {
        "pin":         {"pin": .970, "drift_up": .010, "drift_down": .010, "compression": .008, "breakout": .002},
        "drift_up":    {"pin": .025, "drift_up": .940, "drift_down": .020, "compression": .010, "breakout": .005},
        "drift_down":  {"pin": .025, "drift_up": .020, "drift_down": .940, "compression": .010, "breakout": .005},
        "compression": {"pin": .015, "drift_up": .008, "drift_down": .008, "compression": .962, "breakout": .007},
        "breakout":    {"pin": .050, "drift_up": .015, "drift_down": .015, "compression": .010, "breakout": .910},
    },
    "vol_expansion": {
        "pin":         {"pin": .940, "drift_up": .015, "drift_down": .020, "compression": .005, "breakout": .020},
        "drift_up":    {"pin": .010, "drift_up": .950, "drift_down": .015, "compression": .005, "breakout": .020},
        "drift_down":  {"pin": .010, "drift_up": .010, "drift_down": .955, "compression": .005, "breakout": .020},
        "compression": {"pin": .010, "drift_up": .010, "drift_down": .010, "compression": .940, "breakout": .030},
        "breakout":    {"pin": .010, "drift_up": .015, "drift_down": .020, "compression": .005, "breakout": .950},
    },
    "squeeze_melt_up": {
        "pin":         {"pin": .940, "drift_up": .035, "drift_down": .005, "compression": .005, "breakout": .015},
        "drift_up":    {"pin": .008, "drift_up": .972, "drift_down": .003, "compression": .002, "breakout": .015},
        "drift_down":  {"pin": .015, "drift_up": .040, "drift_down": .930, "compression": .005, "breakout": .010},
        "compression": {"pin": .008, "drift_up": .025, "drift_down": .002, "compression": .950, "breakout": .015},
        "breakout":    {"pin": .008, "drift_up": .040, "drift_down": .005, "compression": .002, "breakout": .945},
    },
    "crash": {
        "pin":         {"pin": .920, "drift_up": .005, "drift_down": .045, "compression": .005, "breakout": .025},
        "drift_up":    {"pin": .010, "drift_up": .920, "drift_down": .045, "compression": .005, "breakout": .020},
        "drift_down":  {"pin": .005, "drift_up": .005, "drift_down": .970, "compression": .002, "breakout": .018},
        "compression": {"pin": .005, "drift_up": .005, "drift_down": .040, "compression": .930, "breakout": .020},
        "breakout":    {"pin": .005, "drift_up": .008, "drift_down": .042, "compression": .002, "breakout": .943},
    },
    "gap_shock": {
        "pin":         {"pin": .960, "drift_up": .010, "drift_down": .010, "compression": .010, "breakout": .010},
        "drift_up":    {"pin": .030, "drift_up": .940, "drift_down": .010, "compression": .010, "breakout": .010},
        "drift_down":  {"pin": .030, "drift_up": .010, "drift_down": .940, "compression": .010, "breakout": .010},
        "compression": {"pin": .020, "drift_up": .008, "drift_down": .008, "compression": .954, "breakout": .010},
        "breakout":    {"pin": .040, "drift_up": .010, "drift_down": .010, "compression": .010, "breakout": .930},
    },
}

# --------------------------------------------------------------------------- #
# Layer 1 — per-variable Markov chains (conditioned on regime)                #
# --------------------------------------------------------------------------- #
VAR_STATES = ("low", "mid", "high")

# Per-regime state-preference rows shared by every variable chain; each
# variable then maps {low, mid, high} onto its own numeric targets. `pull`
# gives the per-minute probability mass moved toward the regime's preferred
# state so variables track regimes without being deterministic functions of
# them (that residual randomness is what makes the matrix combinatoric).
_VAR_PREFERENCE: dict[str, dict[str, str]] = {
    # Skew convention: chain pricing uses s(K) = s_atm - skew*ln(K/F), so
    # POSITIVE skew raises put-strike vol (put-heavy) and negative skew is a
    # call bid. Hence drift_down/breakout steepen the put skew ("high") and
    # drift_up flattens it toward the calls ("low").
    #             gex        rv       vrp      skew     drift
    "pin":         {"gex": "high", "rv": "low",  "vrp": "high", "skew": "mid",  "drift": "mid"},
    "drift_up":    {"gex": "mid",  "rv": "mid",  "vrp": "mid",  "skew": "low",  "drift": "high"},
    "drift_down":  {"gex": "low",  "rv": "mid",  "vrp": "mid",  "skew": "high", "drift": "low"},
    "compression": {"gex": "high", "rv": "low",  "vrp": "high", "skew": "mid",  "drift": "mid"},
    "breakout":    {"gex": "low",  "rv": "high", "vrp": "low",  "skew": "high", "drift": "mid"},
}

# Numeric targets per variable state. GEX in $bn gamma notional, vols
# annualized, drift as fraction of minute-vol, skew in smile-slope units.
_VAR_TARGETS: dict[str, dict[str, float]] = {
    "gex":   {"low": -1.6e9, "mid": 0.4e9, "high": 3.2e9},
    "rv":    {"low": 0.08,   "mid": 0.14,  "high": 0.26},
    "vrp":   {"low": 0.92,   "mid": 1.10,  "high": 1.28},   # implied / realized
    "skew":  {"low": -0.055, "mid": 0.028, "high": 0.075},  # call-heavy .. put-heavy
    "drift": {"low": -0.10,  "mid": 0.0,   "high": 0.10},   # frac of minute-vol
}

_VAR_OU_THETA = {"gex": 0.06, "rv": 0.04, "vrp": 0.03, "skew": 0.03, "drift": 0.05}
_VAR_OU_NOISE = {"gex": 0.12e9, "rv": 0.004, "vrp": 0.01, "skew": 0.003, "drift": 0.01}
_VAR_STAY = 0.985           # per-minute probability a variable keeps its state
_VAR_PULL = 0.010           # extra mass toward the regime's preferred state


class VariableChain:
    """One driver variable: 3-state Markov chain + OU relaxation toward the
    active state's numeric target. Gets its own numpy Generator stream so
    every variable has an independent Markov RNG, as specced."""

    def __init__(self, name: str, rng: np.random.Generator,
                 scale: float = 1.0) -> None:
        self.name = name
        self.rng = rng
        self.scale = scale
        self.state = "mid"
        self.value = _VAR_TARGETS[name]["mid"] * scale

    def step(self, regime: str) -> float:
        prefer = _VAR_PREFERENCE[regime][self.name]
        others = [s for s in VAR_STATES if s != self.state]
        p_move = 1.0 - _VAR_STAY
        probs = {s: p_move / len(others) for s in others}
        probs[self.state] = _VAR_STAY
        probs[prefer] = probs.get(prefer, 0.0) + _VAR_PULL
        total = sum(probs.values())
        states = list(probs)
        self.state = str(self.rng.choice(
            states, p=[probs[s] / total for s in states]))
        target = _VAR_TARGETS[self.name][self.state] * self.scale
        theta = _VAR_OU_THETA[self.name]
        noise = _VAR_OU_NOISE[self.name] * self.scale
        self.value += theta * (target - self.value) + noise * self.rng.standard_normal()
        return self.value


# --------------------------------------------------------------------------- #
# Universe specification + combinatoric catalog                               #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class UniverseSpec:
    """One fully deterministic universe: the seed plus the lattice coordinates
    that condition the chains."""
    universe_id: str
    seed: int
    days: int
    start_archetype: str
    persistence_tilt: float = 1.0   # >1 = regimes stickier, <1 = choppier
    vol_mult: float = 1.0
    gap_mult: float = 1.0
    tick_stride: int = 1
    base_spot: float = 600.0
    generation: int = 0
    # Seeded Dirichlet perturbation of BOTH transition layers (archetype and
    # regime rows). 0 = the canonical matrices verbatim; >0 draws each row
    # from Dirichlet(row / jitter), so later generations explore nearby
    # dynamics deterministically per seed.
    transition_jitter: float = 0.0

    def to_dict(self) -> dict:
        return {
            "universe_id": self.universe_id, "seed": self.seed,
            "days": self.days, "start_archetype": self.start_archetype,
            "persistence_tilt": self.persistence_tilt,
            "vol_mult": self.vol_mult, "gap_mult": self.gap_mult,
            "tick_stride": self.tick_stride, "generation": self.generation,
            "transition_jitter": self.transition_jitter,
        }


def _spec_id(seed: int, arch: str, tilt: float, vol: float, gen: int) -> str:
    raw = f"{seed}|{arch}|{tilt}|{vol}|{gen}"
    return hashlib.sha256(raw.encode()).hexdigest()[:10]


def _dirichlet_rows(rows: dict[str, dict[str, float]],
                    rng: np.random.Generator,
                    jitter: float) -> dict[str, dict[str, float]]:
    """Redraw each row from Dirichlet(row / jitter): the row stays a proper
    distribution centered on the canonical one, with spread growing in
    `jitter`. Deterministic given the caller's seeded generator."""
    out: dict[str, dict[str, float]] = {}
    kappa = 1.0 / max(jitter, 1e-6)
    for state, row in rows.items():
        keys = list(row)
        alpha = np.array([max(row[k], 1e-4) for k in keys]) * kappa
        sample = rng.dirichlet(alpha)
        out[state] = {k: float(v) for k, v in zip(keys, sample)}
    return out


@dataclass
class UniverseCatalog:
    """Enumerates the situation lattice into UniverseSpec entries and evolves
    the sampling weights generation over generation."""
    seed: int = 20260723
    days: int = 8
    tick_stride: int = 5
    tilts: tuple[float, ...] = (0.85, 1.0, 1.15)
    vol_mults: tuple[float, ...] = (0.8, 1.0, 1.3)
    # sampling weight per archetype; evolve() re-balances these
    weights: dict = field(
        default_factory=lambda: {a: 1.0 for a in ARCHETYPES})
    generation: int = 0

    def lattice(self) -> list[UniverseSpec]:
        """The full combinatoric grid — every archetype × tilt × vol cell."""
        specs = []
        # Generation 0 spars on the canonical matrices; later generations add
        # growing (capped) Dirichlet jitter so no two generations replay the
        # exact same dynamics.
        jitter = min(0.02 * self.generation, 0.10)
        for i, (arch, tilt, vol) in enumerate(
                itertools.product(ARCHETYPES, self.tilts, self.vol_mults)):
            seed = self.seed + 1000 * self.generation + i
            specs.append(UniverseSpec(
                universe_id=_spec_id(seed, arch, tilt, vol, self.generation),
                seed=seed, days=self.days, start_archetype=arch,
                persistence_tilt=tilt, vol_mult=vol,
                gap_mult=1.5 if arch in ("gap_shock", "crash") else 1.0,
                tick_stride=self.tick_stride, generation=self.generation,
                transition_jitter=jitter))
        return specs

    def sample(self, n: int) -> list[UniverseSpec]:
        """Weighted sample of the lattice without replacement: archetypes the
        pipeline handles worst (higher weight) claim more of the n slots,
        but every archetype keeps at least lattice presence while n allows."""
        rng = np.random.default_rng(self.seed + 7919 * self.generation)
        full = self.lattice()
        if n >= len(full):
            return full
        w = np.array([self.weights.get(s.start_archetype, 1.0) for s in full])
        idx = rng.choice(len(full), size=n, replace=False, p=w / w.sum())
        return [full[i] for i in sorted(idx)]

    def evolve(self, archetype_scores: dict[str, float]) -> "UniverseCatalog":
        """Next generation: re-weight toward the worst-scoring archetypes
        (score = mean session P&L or any higher-is-better metric). Weight is
        1.0 for the best archetype scaling up to 3.0 for the worst, so the
        curriculum concentrates on weakness without abandoning coverage."""
        if archetype_scores:
            vals = list(archetype_scores.values())
            lo, hi = min(vals), max(vals)
            span = (hi - lo) or 1.0
            new_w = {a: 1.0 + 2.0 * (hi - archetype_scores.get(a, lo)) / span
                     for a in ARCHETYPES}
        else:
            new_w = dict(self.weights)
        return replace(self, weights=new_w, generation=self.generation + 1)


def _tilt_row(row: dict[str, float], state: str, tilt: float) -> dict[str, float]:
    """Scale a transition row's diagonal persistence by `tilt`, renormalized."""
    stay = min(row[state] * tilt, 0.999)
    others = {k: v for k, v in row.items() if k != state}
    rest = sum(others.values()) or 1e-12
    scale = (1.0 - stay) / rest
    out = {k: v * scale for k, v in others.items()}
    out[state] = stay
    return out


# --------------------------------------------------------------------------- #
# The feed                                                                    #
# --------------------------------------------------------------------------- #
@dataclass
class SituationLabel:
    """Per-tick provenance: which cell of the situation lattice this minute
    belongs to. The dojo aggregates P&L against these."""
    session_date: str
    archetype: str
    regime: str


class MarkovWorldFeed:
    """unified_loop.DataFeed over a hierarchical-Markov generated universe.

    Deterministic per UniverseSpec (same spec → identical world), independent
    RNG stream per variable chain, per-tick (archetype, regime) labels in
    `situation_log`, day-level archetypes in `day_archetype`."""

    def __init__(self, spec: UniverseSpec) -> None:
        self.spec = spec
        self._gex_rank = GexRankWindow()
        self._idx = 0
        self.situation_log: list[SituationLabel] = []
        self.day_archetype: dict[str, str] = {}
        self.day_close: dict[str, float] = {}
        self.regime_minutes: dict[str, dict[str, int]] = {}   # arch -> regime -> n
        self._generate()

    # -- world generation ----------------------------------------------------
    def _generate(self) -> None:
        sp = self.spec
        master = np.random.default_rng(sp.seed)
        # independent stream per layer + per variable — "a Markov chain RNG
        # for each variable", literally. (spawn is index-stable: the first 8
        # children are identical whether 8 or 9 are drawn, so jitter=0 worlds
        # match earlier generations exactly.)
        streams = master.spawn(9)
        rng_arch, rng_regime, rng_path, rng_micro = streams[:4]

        # generation evolution: seeded Dirichlet perturbation of both
        # transition layers (no-op at jitter=0 -> canonical matrices)
        if sp.transition_jitter > 0.0:
            rng_jit = streams[8]
            self._arch_T = _dirichlet_rows(_ARCH_TRANSITION, rng_jit,
                                           sp.transition_jitter)
            self._regime_T = {
                arch: _dirichlet_rows(rows, rng_jit, sp.transition_jitter)
                for arch, rows in _REGIME_TRANSITION.items()}
        else:
            self._arch_T = _ARCH_TRANSITION
            self._regime_T = _REGIME_TRANSITION
        chains = {
            "gex":   VariableChain("gex", streams[4]),
            "rv":    VariableChain("rv", streams[5], scale=sp.vol_mult),
            "vrp":   VariableChain("vrp", streams[6]),
            "skew":  VariableChain("skew", streams[7]),
            "drift": VariableChain("drift", np.random.default_rng(sp.seed + 99)),
        }

        ts, close, gex, pins, ivs, flips, skews = [], [], [], [], [], [], []
        spot = sp.base_spot
        pin = round(spot)
        start = dt.datetime(2026, 6, 1, 9, 30, tzinfo=ET)
        day0 = start.date()
        archetype = sp.start_archetype

        d = made = 0
        while made < sp.days:
            date = day0 + dt.timedelta(days=d)
            d += 1
            if date.weekday() >= 5:
                continue
            made += 1
            iso = date.isoformat()

            if made > 1:
                row = self._arch_T[archetype]
                archetype = str(rng_arch.choice(list(row), p=_norm(row)))
            self.day_archetype[iso] = archetype
            occupancy = self.regime_minutes.setdefault(
                archetype, {r: 0 for r in REGIMES})

            # overnight gap: archetype-conditioned. gap_shock gaps BOTH ways
            # (down-biased 60/40) — a shock archetype, not a crash rehearsal.
            gap_vol = 0.003 * sp.gap_mult
            gap_mu = {"crash": -0.008, "squeeze_melt_up": 0.005}.get(archetype, 0.0)
            if archetype == "gap_shock":
                gap_mu = 0.012 * (-1.0 if rng_path.random() < 0.6 else 1.0)
            spot *= math.exp(rng_path.normal(gap_mu, gap_vol))
            pin = round(pin + rng_path.integers(-2, 3))

            regime = _initial_regime(archetype, rng_regime)
            open_dt = dt.datetime(date.year, date.month, date.day, 9, 30, tzinfo=ET)
            breakout_dir = 1.0 if rng_path.random() < 0.5 else -1.0

            for m in range(MIN_PER_DAY):
                base_row = self._regime_T[archetype][regime]
                row = _tilt_row(base_row, regime, sp.persistence_tilt)
                new_regime = str(rng_regime.choice(list(row), p=_norm(row)))
                if new_regime == "breakout" and regime != "breakout":
                    breakout_dir = 1.0 if rng_path.random() < 0.5 else -1.0
                regime = new_regime
                occupancy[regime] += 1

                g = chains["gex"].step(regime)
                rv = max(chains["rv"].step(regime), 0.04)
                vrp = max(chains["vrp"].step(regime), 0.75)
                skew = chains["skew"].step(regime)
                drift = chains["drift"].step(regime)

                sig_min = rv / math.sqrt(MINUTES_PER_YEAR)
                if regime == "pin":
                    step = 0.012 * (pin - spot) / spot + sig_min * rng_micro.standard_normal()
                elif regime == "compression":
                    step = 0.006 * (pin - spot) / spot + 0.6 * sig_min * rng_micro.standard_normal()
                elif regime == "drift_up":
                    step = (0.06 + max(drift, 0.0)) * sig_min + sig_min * rng_micro.standard_normal()
                elif regime == "drift_down":
                    step = -(0.06 + max(-drift, 0.0)) * sig_min + sig_min * rng_micro.standard_normal()
                else:  # breakout
                    step = breakout_dir * 0.12 * sig_min + 1.4 * sig_min * rng_micro.standard_normal()
                spot *= (1.0 + step)

                ts.append(open_dt + dt.timedelta(minutes=m))
                close.append(spot)
                gex.append(g)
                pins.append(pin)
                ivs.append(rv * vrp)
                skews.append(skew)
                # flip sits below spot when dealers are long, chases from
                # above when they are short — same convention as the live map
                flips.append(pin - 4.0 if g > 0 else spot + 2.0)
                self.situation_log.append(SituationLabel(iso, archetype, regime))

            self.day_close[iso] = spot

        n = len(close)
        close_a = np.asarray(close)
        self._ts = np.array([np.datetime64(t.replace(tzinfo=None)) for t in ts],
                            dtype="datetime64[ns]")
        self._dt = ts
        self._close = close_a
        self._open = np.concatenate([[close_a[0]], close_a[:-1]])
        spread = np.abs(rng_micro.normal(0.0, 0.0004, n)) * close_a
        self._high = np.maximum(self._open, close_a) + spread
        self._low = np.minimum(self._open, close_a) - spread
        self._vol = rng_micro.integers(2_000, 30_000, n).astype(float)
        self._gex = np.asarray(gex)
        self._pin = np.asarray(pins)
        self._iv = np.asarray(ivs)
        self._skew = np.asarray(skews)
        self._flip = np.asarray(flips)

    # -- chain pricing -------------------------------------------------------
    def _chain(self, i: int) -> ChainSnapshot:
        spot = float(self._close[i])
        minute = i % MIN_PER_DAY
        minutes_left = max(MIN_PER_DAY - minute, 5)
        t_years = minutes_left / (365.25 * 24 * 60)
        r = 0.05
        DF = math.exp(-r * t_years)
        F = spot / DF
        s_atm = self._iv[i] * math.sqrt(minutes_left / MINUTES_PER_YEAR)
        smile_skew = float(self._skew[i])

        qs = []
        lo = math.floor(spot - 25.0)
        for K in np.arange(lo, spot + 26.0, 1.0):
            if K <= 0:
                continue
            k = math.log(K / F)
            s = max(s_atm - smile_skew * k, 0.0006)
            cm = _bs_call_fwd(F, K, s) * DF
            pm = max(cm - DF * (F - K), 0.0)
            cm = max(cm, 0.0)
            h = 0.012 + 0.002 * max(cm, pm)
            qs.append(ChainQuote(float(K), max(cm - h, 0.0), cm + h,
                                 max(pm - h, 0.0), pm + h))
        return ChainSnapshot(qs, spot=spot, t_years=t_years, r=r)

    # -- DataFeed protocol ---------------------------------------------------
    def timestamps(self) -> list[dt.datetime]:
        return list(self._dt[:: self.spec.tick_stride])

    def snapshot(self, now: dt.datetime) -> Optional[TickSnapshot]:
        i = self._idx * self.spec.tick_stride
        if i >= len(self._close):
            return None
        self._idx += 1

        lo = max(0, i + 1 - 2340)
        bars = RawBars(ts=self._ts[lo:i + 1], open=self._open[lo:i + 1],
                       high=self._high[lo:i + 1], low=self._low[lo:i + 1],
                       close=self._close[lo:i + 1], volume=self._vol[lo:i + 1])

        spot = float(self._close[i])
        pin = float(self._pin[i])
        g = float(self._gex[i])
        chain = self._chain(i)
        tech = _bar_technicals(bars)
        vwap, vwap_rev = _session_vwap_and_reversions(bars, self._dt[i])

        atm = min(chain.quotes, key=lambda q: abs(q.strike - spot))
        straddle = atm.call_mid + atm.put_mid
        minute = i % MIN_PER_DAY
        minutes_left = max(MIN_PER_DAY - minute, 5)
        iv = float(self._iv[i])
        expected_range = spot * iv * math.sqrt(minutes_left / MINUTES_PER_YEAR)

        iv_pts = iv * 100.0
        trending = g <= 0
        label = self.situation_log[i]
        stressed = label.archetype in ("crash", "vol_expansion", "gap_shock")
        market = MarketSnapshot(
            spot=spot, net_gex=g, gamma_flip=float(self._flip[i]),
            call_wall=pin + 5.0, put_wall=pin - 5.0,
            gex_pct_rank=self._gex_rank.rank(g),
            gex_rank_warm=self._gex_rank.is_warm,
            vix9d=iv_pts * (1.06 if trending else 0.94),
            vix=iv_pts,
            vix3m=iv_pts * (0.95 if trending else 1.12),
            vvix=112.0 if stressed else (103.0 if trending else 90.0),
            vvix_baseline=95.0,
            straddle_breakeven=straddle, expected_range=expected_range,
            adx=tech["adx"], rsi=tech["rsi"],
            bb_width=tech["bb_width"], bb_width_baseline=tech["bb_width_baseline"],
            vwap=vwap, vwap_reversion_count=vwap_rev,
            tick_abs_mean=820.0 if stressed else (700.0 if trending else 450.0),
            cvd_slope=tech["cvd_slope"],
            now=self._dt[i], has_catalyst=False,
        )
        return TickSnapshot(market=market, bars=bars, chain=chain)

    def settlement_price(self, session_date: str) -> Optional[float]:
        return self.day_close.get(session_date)

    # -- situation accounting ------------------------------------------------
    def coverage(self) -> dict[str, dict[str, int]]:
        """GENERATED minutes spent in each (archetype × regime) cell — the
        environment's occupancy, regardless of tick_stride."""
        return {a: dict(r) for a, r in self.regime_minutes.items()}

    def evaluated_coverage(self) -> dict[str, dict[str, int]]:
        """Ticks the pipeline actually evaluates per (archetype × regime)
        cell — the strided subset of the generated minutes. This is the
        honest 'situations sparred' count; coverage() is the environment."""
        out: dict[str, dict[str, int]] = {}
        for s in self.situation_log[:: self.spec.tick_stride]:
            out.setdefault(s.archetype, {r: 0 for r in REGIMES})[s.regime] += 1
        return out


# --------------------------------------------------------------------------- #
# helpers                                                                     #
# --------------------------------------------------------------------------- #
def _norm(row: dict[str, float]) -> list[float]:
    total = sum(row.values())
    return [v / total for v in row.values()]


def _initial_regime(archetype: str, rng: np.random.Generator) -> str:
    prefer = {
        "calm_pin": "pin", "grind_up": "drift_up", "grind_down": "drift_down",
        "range_chop": "pin", "vol_expansion": "breakout",
        "squeeze_melt_up": "drift_up", "crash": "drift_down",
        "gap_shock": "compression",
    }[archetype]
    return prefer if rng.random() < 0.7 else str(rng.choice(list(REGIMES)))


def merge_coverage(feeds: list[MarkovWorldFeed],
                   evaluated: bool = False) -> dict[str, dict[str, int]]:
    """Aggregate (archetype × regime) occupancy across many universes — the
    dojo's 'have we sparred everywhere' matrix. Cells at 0 are situations the
    catalog has not yet generated. evaluated=True counts only the ticks the
    pipeline actually evaluated (tick_stride subset) instead of every
    generated minute."""
    out: dict[str, dict[str, int]] = {a: {r: 0 for r in REGIMES} for a in ARCHETYPES}
    for f in feeds:
        cov = f.evaluated_coverage() if evaluated else f.coverage()
        for a, regs in cov.items():
            for r, n in regs.items():
                out[a][r] += n
    return out


# --------------------------------------------------------------------------- #
# demo                                                                        #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    catalog = UniverseCatalog(days=3, tick_stride=15)
    specs = catalog.sample(4)
    print(f"lattice={len(catalog.lattice())} universes; sampled {len(specs)}:")
    feeds = []
    for s in specs:
        f = MarkovWorldFeed(s)
        feeds.append(f)
        archs = sorted(set(f.day_archetype.values()))
        print(f"  {s.universe_id}  start={s.start_archetype:<15} "
              f"tilt={s.persistence_tilt} vol={s.vol_mult} "
              f"days={list(f.day_archetype.values())}")
        snap = f.snapshot(f.timestamps()[0])
        print(f"    first tick: spot={snap.market.spot:.2f} "
              f"gex={snap.market.net_gex/1e9:+.2f}bn "
              f"strikes={len(snap.chain.quotes)}")
    cov = merge_coverage(feeds)
    visited = sum(1 for a in cov for r in cov[a] if cov[a][r] > 0)
    print(f"coverage: {visited}/{len(ARCHETYPES) * len(REGIMES)} "
          f"(archetype × regime) cells visited")
