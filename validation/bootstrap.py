"""
validation/bootstrap.py
=======================
Session-level bootstrap confidence intervals.

Why (docs/PREDICTION_ENGINE_V2_HANDOFF.md §3.1, §18.8): intraday ticks are
heavily correlated — a few dozen trading sessions can masquerade as thousands
of observations. Confidence intervals must therefore resample COMPLETE
sessions with replacement, so the effective sample size is the number of
independent sessions, not the number of journal rows.

Defaults follow the spec: 1,000 replications, 95% interval. Deterministic
given the seed (stdlib random.Random, no numpy dependency).
"""
from __future__ import annotations

import random
from typing import Callable, Optional, Sequence

DEFAULT_REPLICATIONS = 1000
DEFAULT_CI = 0.95
DEFAULT_SEED = 20260710


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs)


def bootstrap_ci(
    values: Sequence[float],
    statistic: Callable[[Sequence[float]], float] = _mean,
    n_boot: int = DEFAULT_REPLICATIONS,
    ci: float = DEFAULT_CI,
    seed: int = DEFAULT_SEED,
) -> dict:
    """
    Percentile bootstrap CI for `statistic` over `values`, where each value
    is one INDEPENDENT unit (one session). Returns a dict:

        {"n": ..., "stat": ..., "ci_low": ..., "ci_high": ...,
         "ci_level": ..., "n_boot": ..., "seed": ...}

    n == 0 → stat/ci_* are None. n == 1 → the interval collapses to the
    point estimate (one session is one observation; there is no spread to
    estimate, and pretending otherwise would be exactly the false-confidence
    failure this module exists to prevent).
    """
    vals = list(values)
    n = len(vals)
    out = {"n": n, "stat": None, "ci_low": None, "ci_high": None,
           "ci_level": ci, "n_boot": n_boot, "seed": seed}
    if n == 0:
        return out
    point = statistic(vals)
    out["stat"] = round(point, 6)
    if n == 1:
        out["ci_low"] = out["ci_high"] = out["stat"]
        return out

    rng = random.Random(seed)
    stats = sorted(
        statistic([vals[rng.randrange(n)] for _ in range(n)])
        for _ in range(n_boot)
    )
    alpha = (1.0 - ci) / 2.0
    lo_idx = min(n_boot - 1, max(0, int(alpha * n_boot)))
    hi_idx = min(n_boot - 1, max(0, int((1.0 - alpha) * n_boot) - 1))
    out["ci_low"] = round(stats[lo_idx], 6)
    out["ci_high"] = round(stats[hi_idx], 6)
    return out


def session_bootstrap(
    session_values: dict[str, float],
    statistic: Callable[[Sequence[float]], float] = _mean,
    n_boot: int = DEFAULT_REPLICATIONS,
    ci: float = DEFAULT_CI,
    seed: int = DEFAULT_SEED,
) -> dict:
    """
    Bootstrap over a {session_date: value} mapping — the canonical entry
    point for per-session P&L or per-session hit rates. Values are ordered
    by session date so results are independent of dict insertion order.
    """
    ordered = [session_values[d] for d in sorted(session_values)]
    out = bootstrap_ci(ordered, statistic=statistic, n_boot=n_boot,
                       ci=ci, seed=seed)
    out["n_sessions"] = out.pop("n")
    return out
