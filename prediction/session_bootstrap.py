"""
prediction/session_bootstrap.py
================================
Session-level bootstrap confidence intervals (V3 Part 1 §11).

Tick bootstrap is prohibited for headline confidence intervals — resample
complete sessions with replacement so the effective N is the number of
independent sessions.

NOT financial advice.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np


def bootstrap_metric_by_session(
    session_ids: Sequence[str],
    values: Sequence[float],
    metric_fn: Callable[[Sequence[float]], float],
    *,
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """
    Bootstrap a metric by resampling complete sessions.

    `values` are per-row observations aligned with `session_ids`. For each
    bootstrap replicate, all rows belonging to sampled sessions are pooled
    and `metric_fn` is applied. Returns point estimate + percentile CI.
    """
    session_ids = list(session_ids)
    values = list(values)
    if len(session_ids) != len(values):
        raise ValueError("session_ids and values length mismatch")

    # Group rows by session
    by_sess: dict[str, list[float]] = {}
    for s, v in zip(session_ids, values):
        by_sess.setdefault(s, []).append(float(v))
    uniq = sorted(by_sess.keys())
    n_sess = len(uniq)
    n_rows = len(values)
    out = {
        "point_estimate": None,
        "lower": None,
        "upper": None,
        "n_sessions": n_sess,
        "n_rows": n_rows,
        "n_bootstrap": n_bootstrap,
        "alpha": alpha,
        "seed": seed,
    }
    if n_sess == 0:
        return out

    all_vals = [v for s in uniq for v in by_sess[s]]
    point = float(metric_fn(all_vals))
    out["point_estimate"] = point
    if n_sess == 1:
        out["lower"] = point
        out["upper"] = point
        return out

    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_bootstrap):
        sampled = rng.choice(uniq, size=n_sess, replace=True)
        pooled = [v for s in sampled for v in by_sess[s]]
        stats.append(float(metric_fn(pooled)))
    stats_a = np.sort(np.asarray(stats, dtype=float))
    lo_q = alpha / 2.0
    hi_q = 1.0 - alpha / 2.0
    out["lower"] = float(np.quantile(stats_a, lo_q))
    out["upper"] = float(np.quantile(stats_a, hi_q))
    return out
