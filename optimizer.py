"""
optimizer.py
============
Grid or random search over EngineConfig parameters, evaluated out-of-sample
via walk-forward Sharpe (or another user-chosen metric).

Param space
-----------
A flat dict mapping dot-notation paths to lists of candidate values:

    param_space = {
        "gate.min_gex_pct_rank": [0.50, 0.65, 0.80],
        "gate.max_adx":          [15.0, 20.0, 25.0],
        "selector.min_ev":       [0.00, 0.01, 0.02],
    }

Supported prefixes: "gate.*" → GateConfig, "selector.*" → SelectorConfig,
"rnd.*" → RNDConfig.  Scalar float/int/bool fields only.

Search modes
------------
  grid    — exhaustive Cartesian product of all param values.
  random  — n_trials random draws, reproducible via seed.

Metric choices
--------------
  sharpe        — mean out-of-sample Sharpe across folds (default)
  total_pnl     — sum of out-of-sample P&L across folds
  win_rate      — mean win rate across folds
  sharpe_over_dd — mean Sharpe / (1 + mean max-drawdown), penalises tail risk

Output
------
  OptimResult.print() — ranked trial table + parameter importance
  OptimResult.best_engine_cfg — drop-in EngineConfig for the winning combo

NOT financial advice.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import itertools
import random as _random
from dataclasses import dataclass, field
from typing import Callable, Optional

from decision_engine import EngineConfig
from gate_scorer import GateConfig
from spread_selector import SelectorConfig
from rnd_extractor import RNDConfig
from risk_manager import RiskConfig
from walk_forward import WalkForwardConfig, WalkForwardResult, run_walk_forward


# --------------------------------------------------------------------------- #
# Score extractor                                                               #
# --------------------------------------------------------------------------- #
_NEG_INF = float("-inf")


def _score(wf: WalkForwardResult, metric: str) -> float:
    folds = wf.folds
    if not folds:
        return _NEG_INF

    if metric == "sharpe":
        vals = [f.tearsheet.sharpe for f in folds if f.tearsheet.sharpe is not None]
        return sum(vals) / len(vals) if vals else _NEG_INF

    if metric == "total_pnl":
        return sum(f.tearsheet.total_pnl for f in folds)

    if metric == "win_rate":
        vals = [f.tearsheet.win_rate for f in folds if f.tearsheet.win_rate is not None]
        return sum(vals) / len(vals) if vals else _NEG_INF

    if metric == "sharpe_over_dd":
        sh   = [f.tearsheet.sharpe       for f in folds if f.tearsheet.sharpe is not None]
        dd   = [f.tearsheet.max_drawdown for f in folds]
        if not sh:
            return _NEG_INF
        mu_sh = sum(sh) / len(sh)
        mu_dd = sum(dd) / len(dd) if dd else 0.0
        return mu_sh / (1.0 + mu_dd)

    raise ValueError(f"Unknown metric: {metric!r}")


# --------------------------------------------------------------------------- #
# Config builder                                                                #
# --------------------------------------------------------------------------- #
def _build_engine_cfg(base: EngineConfig, params: dict) -> EngineConfig:
    """Apply a flat param dict (dot-notation paths) on top of a base EngineConfig."""
    gate_kw: dict = {}
    sel_kw:  dict = {}
    rnd_kw:  dict = {}

    for path, val in params.items():
        prefix, _, key = path.partition(".")
        if prefix == "gate":
            gate_kw[key] = val
        elif prefix == "selector":
            sel_kw[key] = val
        elif prefix == "rnd":
            rnd_kw[key] = val
        else:
            raise ValueError(f"Unknown param prefix: {prefix!r} in {path!r}")

    gate = dataclasses.replace(base.gate, **gate_kw) if gate_kw else base.gate
    sel  = dataclasses.replace(base.selector, **sel_kw) if sel_kw else base.selector
    rnd  = dataclasses.replace(base.rnd, **rnd_kw) if rnd_kw else base.rnd
    return EngineConfig(rnd=rnd, selector=sel, gate=gate)


# --------------------------------------------------------------------------- #
# Trial generation                                                              #
# --------------------------------------------------------------------------- #
def _grid_params(param_space: dict) -> list[dict]:
    keys = list(param_space.keys())
    combos = list(itertools.product(*[param_space[k] for k in keys]))
    return [dict(zip(keys, c)) for c in combos]


def _random_params(param_space: dict, n: int, seed: int) -> list[dict]:
    rng = _random.Random(seed)
    return [{k: rng.choice(v) for k, v in param_space.items()} for _ in range(n)]


# --------------------------------------------------------------------------- #
# Result types                                                                  #
# --------------------------------------------------------------------------- #
@dataclass
class Trial:
    trial_id: int
    params: dict
    engine_cfg: EngineConfig
    wf_result: WalkForwardResult
    score: float


@dataclass
class OptimizerConfig:
    search: str = "grid"          # "grid" | "random"
    n_trials: int = 20            # used only for random search
    metric: str = "sharpe"        # scoring metric (see module docstring)
    seed: int = 42                # reproducibility for random search
    # Selection-bias guard: picking the max of N trials on the SAME folds makes
    # the winner in-sample with respect to the search itself. Reserve the final
    # fraction of the timeline; the search never sees it, and only the single
    # winning config is evaluated there once. Judge by the holdout number.
    holdout_frac: float = 0.0     # 0 disables; 0.2 = final 20% untouched


@dataclass
class OptimResult:
    opt_cfg: OptimizerConfig
    wf_cfg: WalkForwardConfig
    param_space: dict
    trials: list[Trial]           # sorted best → worst
    holdout_score: Optional[float] = None      # winner's score on the untouched window
    holdout_result: Optional[WalkForwardResult] = None

    @property
    def best_trial(self) -> Trial:
        return self.trials[0]

    @property
    def best_engine_cfg(self) -> EngineConfig:
        return self.best_trial.engine_cfg

    # -- output --------------------------------------------------------------

    def print(self, top_n: int = 10) -> None:
        w = 76
        print("=" * w)
        print(f"  Optimizer Result  search={self.opt_cfg.search}  "
              f"metric={self.opt_cfg.metric}  trials={len(self.trials)}")
        print("=" * w)

        # ranked trial table
        keys = list(self.param_space.keys())
        short_keys = [k.split(".", 1)[1] for k in keys]
        hdr = f"  {'#':>3}  {'Score':>8}  " + "  ".join(f"{k:<16}" for k in short_keys)
        print(hdr)
        print("-" * w)
        for t in self.trials[:top_n]:
            vals = "  ".join(f"{t.params.get(k, '?')!s:<16}" for k in keys)
            mark = " ← best" if t.trial_id == self.best_trial.trial_id else ""
            print(f"  {t.trial_id:>3}  {t.score:>+8.4f}  {vals}{mark}")

        # parameter importance: for each param, show mean score per value
        print("\n  Parameter importance (mean score per value):")
        print("-" * w)
        for key in keys:
            short = key.split(".", 1)[1]
            vals = sorted(set(t.params[key] for t in self.trials))
            parts = []
            for v in vals:
                scores = [t.score for t in self.trials
                          if t.params[key] == v and t.score > _NEG_INF]
                mu = sum(scores) / len(scores) if scores else float("nan")
                parts.append(f"{v}→{mu:+.3f}")
            span = max(
                (t.score for t in self.trials if t.score > _NEG_INF),
                default=0.0,
            ) - min(
                (t.score for t in self.trials if t.score > _NEG_INF),
                default=0.0,
            )
            # range across this param's means
            means = [sum(t.score for t in self.trials if t.params[key] == v
                         and t.score > _NEG_INF) /
                     max(1, sum(1 for t in self.trials if t.params[key] == v
                                and t.score > _NEG_INF))
                     for v in vals]
            param_range = max(means) - min(means) if len(means) > 1 else 0.0
            bar = "█" * min(20, int(param_range / max(span, 1e-9) * 20)) if span > 0 else ""
            print(f"    {short:<28}  {' | '.join(parts)}  [{bar}]")

        print("=" * w)
        print(f"  Best: trial #{self.best_trial.trial_id}  "
              f"score={self.best_trial.score:+.4f}")
        for k, v in self.best_trial.params.items():
            print(f"    {k} = {v}")
        if self.holdout_score is not None:
            drop = self.best_trial.score - self.holdout_score
            print(f"  HOLDOUT (untouched final {self.opt_cfg.holdout_frac:.0%}): "
                  f"score={self.holdout_score:+.4f}  "
                  f"(search-window score was {self.best_trial.score:+.4f}; "
                  f"a large drop means the search overfit)")
        print("=" * w)

    def to_dict(self) -> dict:
        return {
            "search": self.opt_cfg.search,
            "metric": self.opt_cfg.metric,
            "n_trials": len(self.trials),
            "best_score": self.best_trial.score,
            "best_params": self.best_trial.params,
            "holdout_score": self.holdout_score,
            "trials": [
                {"id": t.trial_id, "params": t.params, "score": t.score}
                for t in self.trials
            ],
        }


# --------------------------------------------------------------------------- #
# Main entry point                                                              #
# --------------------------------------------------------------------------- #
def run_optimizer(
    feed_factory: Callable,
    timestamps: list[dt.datetime],
    param_space: dict,
    opt_cfg: Optional[OptimizerConfig] = None,
    wf_cfg: Optional[WalkForwardConfig] = None,
    base_engine_cfg: Optional[EngineConfig] = None,
    risk_cfg: Optional[RiskConfig] = None,
) -> OptimResult:
    """
    Search `param_space` over walk-forward folds and return ranked trials.

    feed_factory  — callable() returning a fresh DataFeed for each trial+fold.
    timestamps    — the full tick sequence to walk-forward over.
    param_space   — dict of {"prefix.field": [val1, val2, ...], ...}.
    opt_cfg       — optimizer settings (search mode, metric, n_trials).
    wf_cfg        — walk-forward settings passed to each trial evaluation.
    base_engine_cfg — starting point for all configs; defaults to EngineConfig().
    risk_cfg      — optional risk guard applied identically across all trials.
    """
    opt  = opt_cfg or OptimizerConfig()
    wf   = wf_cfg  or WalkForwardConfig()
    base = base_engine_cfg or EngineConfig()

    # Carve off the untouched holdout BEFORE the search sees anything.
    holdout_ts: list[dt.datetime] = []
    search_ts = timestamps
    if opt.holdout_frac > 0.0:
        cut = int(len(timestamps) * (1.0 - opt.holdout_frac))
        search_ts, holdout_ts = timestamps[:cut], timestamps[cut:]

    if opt.search == "grid":
        param_list = _grid_params(param_space)
    elif opt.search == "random":
        param_list = _random_params(param_space, opt.n_trials, opt.seed)
    else:
        raise ValueError(f"Unknown search mode: {opt.search!r}")

    n_total = len(param_list)
    print(f"  Optimizer: {opt.search} search, {n_total} trials, "
          f"metric={opt.metric}, wf={wf.mode}/{wf.n_folds}-fold"
          + (f", holdout={opt.holdout_frac:.0%}" if holdout_ts else ""))

    trials: list[Trial] = []
    for i, params in enumerate(param_list, start=1):
        engine_cfg = _build_engine_cfg(base, params)
        param_str = "  ".join(f"{k.split('.',1)[1]}={v}" for k, v in params.items())
        print(f"  Trial {i:>3}/{n_total}  {param_str}", end="", flush=True)

        wf_result = run_walk_forward(
            feed_factory=feed_factory,
            timestamps=search_ts,
            wf_cfg=wf,
            engine_cfg=engine_cfg,
            risk_cfg=risk_cfg,
        )
        sc = _score(wf_result, opt.metric)
        print(f"  → score={sc:+.4f}")
        trials.append(Trial(
            trial_id=i, params=params,
            engine_cfg=engine_cfg, wf_result=wf_result, score=sc,
        ))

    trials.sort(key=lambda t: t.score, reverse=True)

    # Evaluate ONLY the winner, ONCE, on the untouched window: warm-up on the
    # full search timeline, one test fold on the holdout.
    holdout_score = holdout_result = None
    if holdout_ts:
        print(f"  Holdout: evaluating winner on final {len(holdout_ts):,} ticks "
              f"(never seen by the search)")
        holdout_result = run_walk_forward(
            feed_factory=feed_factory,
            timestamps=search_ts + holdout_ts,
            wf_cfg=WalkForwardConfig(
                mode="expanding", n_folds=1,
                train_frac=len(search_ts) / max(1, len(search_ts) + len(holdout_ts)),
            ),
            engine_cfg=trials[0].engine_cfg,
            risk_cfg=risk_cfg,
        )
        holdout_score = _score(holdout_result, opt.metric)
        print(f"  Holdout score: {holdout_score:+.4f} "
              f"(search score {trials[0].score:+.4f})")

    return OptimResult(
        opt_cfg=opt, wf_cfg=wf,
        param_space=param_space, trials=trials,
        holdout_score=holdout_score, holdout_result=holdout_result,
    )


# --------------------------------------------------------------------------- #
# Demo                                                                          #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import math
    import numpy as np
    import datetime as dt
    from zoneinfo import ZoneInfo
    from rnd_extractor import ChainSnapshot, ChainQuote, _bs_call_fwd
    from unified_loop import SyntheticUnifiedFeed

    ET = ZoneInfo("America/New_York")
    DAYS = 10
    spot0 = 600.0
    T0, r0 = 4.0 / (24 * 365), 0.05
    DF0 = math.exp(-r0 * T0)
    F0  = spot0 * math.exp(r0 * T0)

    qs = []
    for K in np.arange(spot0 - 15, spot0 + 16, 1.0):
        k = math.log(K / F0)
        s = max(0.0050 - 0.030 * k, 0.0008)
        cm = _bs_call_fwd(F0, K, s) * DF0
        pm = max(cm - DF0 * (F0 - K), 0.0)
        cm = max(cm, 0.0)
        h  = 0.01 + 0.002 * max(cm, pm)
        qs.append(ChainQuote(float(K), max(cm - h, 0), cm + h,
                             max(pm - h, 0), pm + h))
    chain = ChainSnapshot(qs, spot=spot0, t_years=T0, r=r0)

    start = dt.datetime(2026, 6, 1, 9, 30, tzinfo=ET)
    ticks = [start + dt.timedelta(minutes=i) for i in range(DAYS * 390)]

    def make_feed():
        return SyntheticUnifiedFeed(days=DAYS, chain=chain, settle=spot0)

    param_space = {
        "gate.min_gex_pct_rank": [0.50, 0.70],
        "gate.max_adx":          [15.0, 25.0],
    }

    print("=" * 76)
    print("  Parameter Optimizer Demo — 10 days, 2-fold expanding, grid search")
    print("=" * 76)

    result = run_optimizer(
        feed_factory=make_feed,
        timestamps=ticks,
        param_space=param_space,
        opt_cfg=OptimizerConfig(search="grid", metric="total_pnl"),
        wf_cfg=WalkForwardConfig(mode="expanding", n_folds=2, train_frac=0.6),
    )
    print()
    result.print()
