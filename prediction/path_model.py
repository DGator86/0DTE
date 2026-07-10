"""
prediction/path_model.py
========================
Empirical residual block-bootstrap path simulator
(docs/PREDICTION_ENGINE_V2_HANDOFF.md §15).

Replaces the terminal reflection touch approximation (rnd.prob_touch ≈ 2×
beyond) and complements the structured Gaussian MC in mc.py (retained as
baseline) with paths that preserve:

  * serial correlation
  * volatility clustering
  * momentum bursts / mean-reversion runs
  * contiguous return-block structure

Sampling process (§15.2):
  1. Build a library of one-minute standardized residual blocks from history.
  2. Sample contiguous blocks of 5–15 minutes (configurable).
  3. Rescale blocks to the current predicted per-minute volatility.
  4. Add the independently forecast mean return.
  5. Stitch until the horizon.
  6. Score target/stop, wall, flip, and range events on each path.

Same-bar ambiguity is handled CONSERVATIVELY: when a single step's
[low, high] contains both a favorable and an adverse barrier, the adverse
event is assigned first (matches prediction.labels.first_passage).

NOT financial advice.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from prediction.models.base import RANDOM_STATE

PATH_MODEL_VERSION = "v2.0.0-pr7"
MINUTES_PER_YEAR = 252 * 390


# --------------------------------------------------------------------------- #
# Config / library                                                             #
# --------------------------------------------------------------------------- #
@dataclass
class PathModelConfig:
    block_min: int = 5
    block_max: int = 15
    n_paths: int = 4000
    seed: int = RANDOM_STATE
    # Minimum residuals required before the library is usable.
    min_library_residuals: int = 60
    # When True, filter library blocks by matching gex_sign / vol_quantile
    # buckets when those labels are present on the library (soft conditioning).
    condition_on_state: bool = True


@dataclass
class ResidualLibrary:
    """
    Contiguous standardized residual stream plus optional conditioning labels
    aligned 1:1 with each residual (the residual from t-1 → t carries the
    state observed at t-1).
    """
    residuals: np.ndarray                          # shape (N,), standardized
    session_id: np.ndarray                         # shape (N,) int/str labels
    minute_of_session: Optional[np.ndarray] = None
    vol_quantile: Optional[np.ndarray] = None      # 0..n_buckets-1
    gex_sign: Optional[np.ndarray] = None          # -1 / 0 / +1
    # Session boundaries: list of (start_idx, end_idx) into residuals so a
    # sampled block never crosses a session gap.
    session_spans: list = field(default_factory=list)

    def __post_init__(self):
        self.residuals = np.asarray(self.residuals, dtype=float)
        self.session_id = np.asarray(self.session_id)
        if not self.session_spans:
            self.session_spans = _infer_session_spans(self.session_id)

    def __len__(self) -> int:
        return int(self.residuals.size)


def _infer_session_spans(session_id: np.ndarray) -> list:
    spans = []
    if session_id.size == 0:
        return spans
    start = 0
    cur = session_id[0]
    for i in range(1, len(session_id)):
        if session_id[i] != cur:
            spans.append((start, i))
            start = i
            cur = session_id[i]
    spans.append((start, len(session_id)))
    return spans


def standardize_returns(returns: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Z-score a return series (library construction)."""
    r = np.asarray(returns, dtype=float)
    mu = float(np.nanmean(r))
    sd = float(np.nanstd(r))
    if not math.isfinite(sd) or sd < eps:
        sd = eps
    out = (r - mu) / sd
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def build_residual_library(
    returns_by_session: dict,
    *,
    gex_sign_by_session: Optional[dict] = None,
    vol_quantile_by_session: Optional[dict] = None,
) -> ResidualLibrary:
    """
    Build a ResidualLibrary from {session_id: 1-min simple-return array}.

    Residuals are standardized *within each session* so a high-vol day does
    not dominate the library; the simulator re-scales to the live forecast vol.
    """
    res_parts, sid_parts, gex_parts, vol_parts, minute_parts = [], [], [], [], []
    spans = []
    cursor = 0
    for sid, rets in sorted(returns_by_session.items(), key=lambda kv: str(kv[0])):
        r = np.asarray(rets, dtype=float)
        if r.size < 2:
            continue
        z = standardize_returns(r)
        n = z.size
        res_parts.append(z)
        sid_parts.append(np.full(n, sid, dtype=object))
        minute_parts.append(np.arange(n, dtype=float))
        if gex_sign_by_session is not None:
            gex_parts.append(np.full(n, gex_sign_by_session.get(sid, 0)))
        if vol_quantile_by_session is not None:
            vol_parts.append(np.full(n, vol_quantile_by_session.get(sid, 0)))
        spans.append((cursor, cursor + n))
        cursor += n
    if not res_parts:
        return ResidualLibrary(residuals=np.zeros(0), session_id=np.zeros(0, dtype=object),
                               session_spans=[])
    return ResidualLibrary(
        residuals=np.concatenate(res_parts),
        session_id=np.concatenate(sid_parts),
        minute_of_session=np.concatenate(minute_parts),
        vol_quantile=(np.concatenate(vol_parts) if vol_parts else None),
        gex_sign=(np.concatenate(gex_parts) if gex_parts else None),
        session_spans=spans,
    )


# --------------------------------------------------------------------------- #
# Path simulation                                                              #
# --------------------------------------------------------------------------- #
def _eligible_starts(library: ResidualLibrary, block_len: int,
                     gex_sign: Optional[int] = None,
                     vol_quantile: Optional[int] = None,
                     condition: bool = True) -> np.ndarray:
    """Indices where a contiguous block of `block_len` fits inside one session."""
    starts = []
    for lo, hi in library.session_spans:
        if hi - lo < block_len:
            continue
        for i in range(lo, hi - block_len + 1):
            if condition and gex_sign is not None and library.gex_sign is not None:
                if int(library.gex_sign[i]) != int(gex_sign):
                    continue
            if condition and vol_quantile is not None and library.vol_quantile is not None:
                if int(library.vol_quantile[i]) != int(vol_quantile):
                    continue
            starts.append(i)
    return np.asarray(starts, dtype=int)


def simulate_paths(
    spot: float,
    n_steps: int,
    sigma_per_min: float,
    *,
    library: ResidualLibrary,
    mean_per_min: float = 0.0,
    cfg: Optional[PathModelConfig] = None,
    gex_sign: Optional[int] = None,
    vol_quantile: Optional[int] = None,
) -> np.ndarray:
    """
    Simulate `cfg.n_paths` price paths of length `n_steps` (returns shape
    (n_paths, n_steps + 1) including the starting spot).

    Each path is a stitch of contiguous residual blocks, rescaled so the
    per-step shock has std ≈ sigma_per_min, then shifted by mean_per_min.
    Raises ValueError when the library is too thin.
    """
    cfg = cfg or PathModelConfig()
    if len(library) < cfg.min_library_residuals:
        raise ValueError(
            f"residual library too thin ({len(library)} < {cfg.min_library_residuals})")
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")
    if not (math.isfinite(sigma_per_min) and sigma_per_min > 0):
        raise ValueError(f"sigma_per_min must be > 0, got {sigma_per_min!r}")

    rng = np.random.default_rng(cfg.seed)
    n_paths = int(cfg.n_paths)
    paths = np.empty((n_paths, n_steps + 1), dtype=float)
    paths[:, 0] = float(spot)

    # Precompute eligible starts per block length; fall back to unconditioned
    # if the conditioned set is empty.
    starts_by_len: dict[int, np.ndarray] = {}
    for L in range(cfg.block_min, cfg.block_max + 1):
        s = _eligible_starts(library, L, gex_sign, vol_quantile,
                             condition=cfg.condition_on_state)
        if s.size == 0:
            s = _eligible_starts(library, L, condition=False)
        starts_by_len[L] = s
    usable_lens = [L for L, s in starts_by_len.items() if s.size > 0]
    if not usable_lens:
        raise ValueError("no eligible residual blocks in library")

    for p in range(n_paths):
        price = float(spot)
        t = 0
        while t < n_steps:
            L = int(rng.choice(usable_lens))
            L = min(L, n_steps - t)
            # re-pick a length that still has starts and fits
            candidates = [x for x in usable_lens if x <= (n_steps - t)
                          and starts_by_len[x].size > 0]
            if not candidates:
                # last resort: single residual steps from any session
                L = 1
                starts = _eligible_starts(library, 1, condition=False)
            else:
                L = int(rng.choice(candidates))
                starts = starts_by_len[L]
            start = int(rng.choice(starts))
            block = library.residuals[start:start + L]
            for z in block:
                shock = mean_per_min + sigma_per_min * float(z)
                price = price * (1.0 + shock)
                t += 1
                paths[p, t] = price
                if t >= n_steps:
                    break
    return paths


def annual_to_per_min(sigma_annual: float) -> float:
    return float(sigma_annual) / math.sqrt(MINUTES_PER_YEAR)


# --------------------------------------------------------------------------- #
# Event scoring on simulated paths                                             #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PathEventResult:
    """Barrier / first-passage frequencies across simulated paths (§15.3)."""
    p_target_first: float
    p_stop_first: float
    p_neither: float
    p_touch_call_wall: float
    p_touch_put_wall: float
    p_cross_gamma_flip: float
    p_call_wall_first: float
    p_put_wall_first: float
    p_neither_wall: float
    p_range_survive: float
    terminal_mean: float
    terminal_std: float
    mfe_mean: float
    mae_mean: float
    n_paths: int
    n_steps: int
    ambiguous_same_step_rate: float
    model_version: str = PATH_MODEL_VERSION
    diagnostics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "p_target_first": self.p_target_first,
            "p_stop_first": self.p_stop_first,
            "p_neither": self.p_neither,
            "p_touch_call_wall": self.p_touch_call_wall,
            "p_touch_put_wall": self.p_touch_put_wall,
            "p_cross_gamma_flip": self.p_cross_gamma_flip,
            "p_call_wall_first": self.p_call_wall_first,
            "p_put_wall_first": self.p_put_wall_first,
            "p_neither_wall": self.p_neither_wall,
            "p_range_survive": self.p_range_survive,
            "terminal_mean": self.terminal_mean,
            "terminal_std": self.terminal_std,
            "mfe_mean": self.mfe_mean,
            "mae_mean": self.mae_mean,
            "n_paths": self.n_paths,
            "n_steps": self.n_steps,
            "ambiguous_same_step_rate": self.ambiguous_same_step_rate,
            "model_version": self.model_version,
            "diagnostics": dict(self.diagnostics),
        }


def _step_hl(prev: float, curr: float) -> tuple[float, float]:
    """Close-to-close step implied high/low (conservative barrier envelope)."""
    return (min(prev, curr), max(prev, curr))


def score_path_events(
    paths: np.ndarray,
    *,
    spot: float,
    call_wall: Optional[float] = None,
    put_wall: Optional[float] = None,
    gamma_flip: Optional[float] = None,
    target: Optional[float] = None,
    stop: Optional[float] = None,
    lower: Optional[float] = None,
    upper: Optional[float] = None,
) -> PathEventResult:
    """
    Score barrier events on an array of paths (n_paths, n_steps+1).

    Target/stop direction is inferred from levels relative to spot.
    Same-step ambiguity (both barriers inside one step's [low, high]) is
    resolved CONSERVATIVELY to the adverse event (stop / nearer adverse wall).
    """
    paths = np.asarray(paths, dtype=float)
    n_paths, cols = paths.shape
    n_steps = cols - 1
    if n_steps < 1:
        raise ValueError("paths must include at least one step")

    # --- target / stop first-passage ---
    hit_target = np.zeros(n_paths, dtype=bool)
    hit_stop = np.zeros(n_paths, dtype=bool)
    amb_count = 0
    live = np.ones(n_paths, dtype=bool)
    if target is not None and stop is not None:
        up = target >= spot
        for t in range(1, cols):
            if not live.any():
                break
            prev, curr = paths[:, t - 1], paths[:, t]
            lo = np.minimum(prev, curr)
            hi = np.maximum(prev, curr)
            if up:
                t_hit = hi >= target
                s_hit = lo <= stop
            else:
                t_hit = lo <= target
                s_hit = hi >= stop
            both = live & t_hit & s_hit
            only_t = live & t_hit & ~s_hit
            only_s = live & s_hit & ~t_hit
            # conservative: same-step → stop first
            amb_count += int(both.sum())
            hit_stop |= both | only_s
            hit_target |= only_t
            live &= ~(both | only_t | only_s)

    p_t = float(hit_target.mean()) if target is not None else float("nan")
    p_s = float(hit_stop.mean()) if stop is not None else float("nan")
    p_n = float((~(hit_target | hit_stop)).mean()) if target is not None else float("nan")
    if math.isfinite(p_t):
        p_t = float(np.clip(p_t, 0.0, 1.0))
    if math.isfinite(p_s):
        p_s = float(np.clip(p_s, 0.0, 1.0))
    if math.isfinite(p_n):
        p_n = float(np.clip(p_n, 0.0, 1.0))

    # --- wall touches + first-passage ---
    touch_c = np.zeros(n_paths, dtype=bool)
    touch_p = np.zeros(n_paths, dtype=bool)
    first_c = np.zeros(n_paths, dtype=bool)
    first_p = np.zeros(n_paths, dtype=bool)
    wall_done = np.zeros(n_paths, dtype=bool)
    if call_wall is not None or put_wall is not None:
        for t in range(1, cols):
            prev, curr = paths[:, t - 1], paths[:, t]
            lo = np.minimum(prev, curr)
            hi = np.maximum(prev, curr)
            c_hit = (hi >= call_wall) if call_wall is not None else np.zeros(n_paths, dtype=bool)
            p_hit = (lo <= put_wall) if put_wall is not None else np.zeros(n_paths, dtype=bool)
            touch_c |= c_hit
            touch_p |= p_hit
            both = ~wall_done & c_hit & p_hit
            only_c = ~wall_done & c_hit & ~p_hit
            only_p = ~wall_done & p_hit & ~c_hit
            # conservative wall ordering: the wall AGAINST a move from spot
            # (put wall if spot was above mid-channel) — use put-first when
            # ambiguous (adverse for long-premium / credit short-put risk).
            first_p |= both | only_p
            first_c |= only_c
            wall_done |= both | only_c | only_p

    # --- gamma flip cross ---
    cross_flip = np.zeros(n_paths, dtype=bool)
    if gamma_flip is not None:
        side0 = np.sign(spot - gamma_flip)
        for t in range(1, cols):
            side = np.sign(paths[:, t] - gamma_flip)
            if side0 == 0:
                cross_flip |= paths[:, t] != gamma_flip
            else:
                cross_flip |= side == -side0

    # --- range survival ---
    survive = np.ones(n_paths, dtype=bool)
    if lower is not None and upper is not None:
        for t in range(1, cols):
            prev, curr = paths[:, t - 1], paths[:, t]
            lo = np.minimum(prev, curr)
            hi = np.maximum(prev, curr)
            survive &= (lo > lower) & (hi < upper)

    terminal = paths[:, -1]
    # MFE/MAE as max favorable / adverse excursion from spot (simple return)
    rets = paths / spot - 1.0
    mfe = rets.max(axis=1)
    mae = rets.min(axis=1)

    return PathEventResult(
        p_target_first=p_t,
        p_stop_first=p_s,
        p_neither=p_n,
        p_touch_call_wall=float(touch_c.mean()) if call_wall is not None else float("nan"),
        p_touch_put_wall=float(touch_p.mean()) if put_wall is not None else float("nan"),
        p_cross_gamma_flip=float(cross_flip.mean()) if gamma_flip is not None else float("nan"),
        p_call_wall_first=float(first_c.mean()) if call_wall is not None else float("nan"),
        p_put_wall_first=float(first_p.mean()) if put_wall is not None else float("nan"),
        p_neither_wall=float((~wall_done).mean()) if (call_wall is not None or put_wall is not None) else float("nan"),
        p_range_survive=float(survive.mean()) if (lower is not None and upper is not None) else float("nan"),
        terminal_mean=float(terminal.mean()),
        terminal_std=float(terminal.std()),
        mfe_mean=float(mfe.mean()),
        mae_mean=float(mae.mean()),
        n_paths=n_paths,
        n_steps=n_steps,
        ambiguous_same_step_rate=float(amb_count) / max(n_paths, 1),
        diagnostics={"spot": float(spot), "has_target_stop": target is not None,
                     "has_walls": call_wall is not None or put_wall is not None,
                     "has_range": lower is not None and upper is not None},
    )


def project_barriers(
    spot: float,
    n_steps: int,
    sigma_annual: float,
    library: ResidualLibrary,
    *,
    mean_return_horizon: float = 0.0,
    call_wall: Optional[float] = None,
    put_wall: Optional[float] = None,
    gamma_flip: Optional[float] = None,
    target: Optional[float] = None,
    stop: Optional[float] = None,
    lower: Optional[float] = None,
    upper: Optional[float] = None,
    gex_sign: Optional[int] = None,
    vol_quantile: Optional[int] = None,
    cfg: Optional[PathModelConfig] = None,
) -> PathEventResult:
    """
    End-to-end: simulate residual-bootstrap paths and score barrier events.

    mean_return_horizon is the total expected LOG/simple return over the
    horizon; it is spread evenly across minutes as a per-step drift.
    """
    cfg = cfg or PathModelConfig()
    sigma_min = annual_to_per_min(sigma_annual)
    mean_min = float(mean_return_horizon) / max(n_steps, 1)
    paths = simulate_paths(
        spot, n_steps, sigma_min, library=library, mean_per_min=mean_min,
        cfg=cfg, gex_sign=gex_sign, vol_quantile=vol_quantile)
    return score_path_events(
        paths, spot=spot, call_wall=call_wall, put_wall=put_wall,
        gamma_flip=gamma_flip, target=target, stop=stop,
        lower=lower, upper=upper)
