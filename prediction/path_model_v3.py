"""
prediction/path_model_v3.py
===========================
State-conditioned empirical path simulation (V3 Part 2 §24–§31, PR 14).

Extends the V2 block-bootstrap simulator with:
  * ResidualBlockMeta / richer conditioning features
  * Distance kernel sampling with source-session caps
  * Explicit conditioning backoff hierarchy (levels 0–6)
  * Deterministic seeds from snapshot_id + version + horizon + config hash
  * PathForecastV3 contract

Gaussian/OU remains a labeled Level-6 fallback only.

NOT financial advice.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass, field
from typing import Mapping, Optional, Sequence

import numpy as np

from prediction.models.base import RANDOM_STATE
from prediction.path_model import (
    PathModelConfig,
    ResidualLibrary,
    _eligible_starts,
    score_path_events,
)

PATH_MODEL_VERSION = "v3.0.0"

# Backoff feature sets (§25) — ordered from richest to sparsest
_BACKOFF_FEATURES = (
    # level 0
    ("minute_of_session", "volatility_quantile", "gex_sign",
     "gex_concentration", "gex_disagreement", "distance_to_flip",
     "trend_strength", "breadth_alignment", "overnight_gap", "data_quality"),
    # level 1 — remove breadth
    ("minute_of_session", "volatility_quantile", "gex_sign",
     "gex_concentration", "gex_disagreement", "distance_to_flip",
     "trend_strength", "overnight_gap", "data_quality"),
    # level 2 — remove concentration / disagreement
    ("minute_of_session", "volatility_quantile", "gex_sign",
     "distance_to_flip", "trend_strength", "overnight_gap", "data_quality"),
    # level 3
    ("minute_of_session", "volatility_quantile", "gex_sign", "distance_to_flip"),
    # level 4
    ("minute_of_session", "volatility_quantile"),
    # level 5 — unconditioned
    (),
)


@dataclass(frozen=True)
class ResidualBlockMeta:
    session_id: str
    start_index: int
    block_length: int
    minute_of_session: Optional[int] = None
    volatility_quantile: Optional[int] = None
    gex_sign: Optional[int] = None
    gex_concentration: Optional[float] = None
    gex_disagreement: Optional[float] = None
    distance_to_flip: Optional[float] = None
    trend_strength: Optional[float] = None
    breadth_alignment: Optional[float] = None
    overnight_gap: Optional[float] = None
    regime_probabilities: dict[str, float] = field(default_factory=dict)
    data_quality: Optional[float] = None

    def feature_dict(self) -> dict[str, float]:
        out = {}
        mapping = {
            "minute_of_session": self.minute_of_session,
            "volatility_quantile": self.volatility_quantile,
            "gex_sign": self.gex_sign,
            "gex_concentration": self.gex_concentration,
            "gex_disagreement": self.gex_disagreement,
            "distance_to_flip": self.distance_to_flip,
            "trend_strength": self.trend_strength,
            "breadth_alignment": self.breadth_alignment,
            "overnight_gap": self.overnight_gap,
            "data_quality": self.data_quality,
        }
        for k, v in mapping.items():
            if v is not None and math.isfinite(float(v)):
                out[k] = float(v)
        return out


@dataclass
class PathModelV3Config:
    block_min: int = 5
    block_max: int = 15
    n_paths_shadow: int = 5_000
    n_paths_offline: int = 20_000
    n_paths_test: int = 250
    condition_temperature: float = 1.0
    min_effective_support: float = 30.0
    max_source_session_weight: float = 0.10
    allow_gaussian_fallback: bool = True
    decision_facing_max_backoff: int = 5
    same_step_adverse_first: bool = True
    feature_weights: dict[str, float] = field(default_factory=dict)
    epsilon: float = 1e-12

    def n_paths_for(self, mode: str = "test") -> int:
        return {
            "shadow": self.n_paths_shadow,
            "offline": self.n_paths_offline,
            "test": self.n_paths_test,
        }.get(mode, self.n_paths_test)


@dataclass(frozen=True)
class PathForecastV3:
    p_target_first: float
    p_stop_first: float
    p_neither: float
    p_touch_call_wall: Optional[float]
    p_touch_put_wall: Optional[float]
    p_cross_gamma_flip: Optional[float]
    p_call_wall_first: Optional[float]
    p_put_wall_first: Optional[float]
    p_neither_wall: Optional[float]
    p_range_survive: Optional[float]
    terminal_quantiles: dict[float, float]
    mfe_quantiles: dict[float, float]
    mae_quantiles: dict[float, float]
    terminal_mean: float
    terminal_std: float
    effective_support: float
    source_session_concentration: float
    conditioning_backoff_level: int
    uncertainty: float
    n_paths: int
    n_steps: int
    model_version: str
    diagnostics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def derive_path_seed(
    snapshot_id: str,
    *,
    model_version: str = PATH_MODEL_VERSION,
    horizon: str = "30m",
    configuration_hash: str = "",
) -> int:
    material = f"{snapshot_id}|{model_version}|{horizon}|{configuration_hash}"
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % (2**31 - 1)


def config_hash(cfg: PathModelV3Config) -> str:
    payload = (
        f"{cfg.block_min}|{cfg.block_max}|{cfg.condition_temperature}|"
        f"{cfg.min_effective_support}|{cfg.max_source_session_weight}|"
        f"{cfg.decision_facing_max_backoff}|{sorted(cfg.feature_weights.items())}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def standardized_distance(
    current: Mapping[str, float],
    block: Mapping[str, float],
    feature_weights: Mapping[str, float],
    feature_names: Sequence[str],
    *,
    epsilon: float = 1e-12,
) -> Optional[float]:
    """Weighted Euclidean distance; missing features excluded + renorm (§24.7)."""
    names = [n for n in feature_names
             if n in current and n in block
             and math.isfinite(float(current[n]))
             and math.isfinite(float(block[n]))]
    if not names:
        return None
    weights = np.array([float(feature_weights.get(n, 1.0)) for n in names])
    wsum = float(weights.sum())
    if wsum <= epsilon:
        return None
    weights = weights / wsum
    dist2 = 0.0
    for w, n in zip(weights, names):
        dist2 += w * (float(current[n]) - float(block[n])) ** 2
    return float(math.sqrt(dist2))


def kernel_weights(
    distances: Sequence[float],
    *,
    temperature: float = 1.0,
    epsilon: float = 1e-12,
) -> np.ndarray:
    d = np.asarray(distances, dtype=float)
    temp = max(float(temperature), epsilon)
    w = np.exp(-(d ** 2) / temp)
    s = float(w.sum())
    if s <= epsilon:
        return np.ones(len(d)) / max(len(d), 1)
    return w / s


def apply_session_cap(
    weights: np.ndarray,
    session_ids: Sequence[str],
    *,
    max_weight: float = 0.10,
    epsilon: float = 1e-12,
) -> np.ndarray:
    """Cap aggregate weight per source session, then renormalize (§24.9)."""
    w = np.asarray(weights, dtype=float).copy()
    n = len(w)
    if n == 0:
        return w
    sessions = np.asarray(list(session_ids), dtype=object)
    s = float(w.sum())
    w = w / s if s > epsilon else np.ones(n) / n

    fixed = np.zeros(n, dtype=bool)
    for _ in range(len(set(sessions.tolist())) + 2):
        # Session totals among unfixed
        totals: dict[str, float] = {}
        for i in range(n):
            if fixed[i]:
                continue
            sid = str(sessions[i])
            totals[sid] = totals.get(sid, 0.0) + float(w[i])
        offenders = [sid for sid, tot in totals.items() if tot > max_weight + epsilon]
        if not offenders:
            break
        for sid in offenders:
            idx = [i for i in range(n)
                   if (not fixed[i]) and str(sessions[i]) == sid]
            tot = sum(float(w[i]) for i in idx)
            if tot <= epsilon:
                continue
            scale = max_weight / tot
            for i in idx:
                w[i] *= scale
                fixed[i] = True
        # Redistribute remaining mass to unfixed
        fixed_mass = float(w[fixed].sum()) if fixed.any() else 0.0
        remain = max(0.0, 1.0 - fixed_mass)
        free = ~fixed
        if free.any() and remain > epsilon:
            free_sum = float(w[free].sum())
            if free_sum > epsilon:
                w[free] = w[free] / free_sum * remain
            else:
                w[free] = remain / int(free.sum())
        elif not free.any():
            # All sessions capped but mass remains — impossible to honor
            # max_weight * n_sessions < 1. Fall back to equal session weights.
            uniq = list(dict.fromkeys(str(s) for s in sessions))
            per = 1.0 / max(len(uniq), 1)
            for sid in uniq:
                idx = [i for i in range(n) if str(sessions[i]) == sid]
                share = per / max(len(idx), 1)
                for i in idx:
                    w[i] = share
            break
    s = float(w.sum())
    return w / s if s > epsilon else np.ones(n) / n


def effective_sample_size(weights: np.ndarray, *, epsilon: float = 1e-12) -> float:
    w = np.asarray(weights, dtype=float)
    s2 = float(np.sum(w ** 2))
    if s2 <= epsilon:
        return 0.0
    return float(1.0 / s2)


def select_backoff_level(
    current: Mapping[str, float],
    block_features: Sequence[Mapping[str, float]],
    cfg: PathModelV3Config,
) -> tuple[int, np.ndarray, dict]:
    """
    Walk backoff hierarchy until effective support is adequate (§25).
    Returns (level, normalized_weights, diagnostics).
    """
    diag: dict = {"attempts": []}
    n = len(block_features)
    if n == 0:
        return 6, np.zeros(0), {"reason": "empty_library"}

    for level, feats in enumerate(_BACKOFF_FEATURES):
        if not feats:
            w = np.ones(n) / n
            ess = effective_sample_size(w)
            diag["attempts"].append({
                "level": level, "features": list(feats),
                "ess": ess, "reason": "unconditioned",
            })
            if ess >= cfg.min_effective_support or level >= 5:
                return level, w, diag
            continue
        distances = []
        valid = []
        for bf in block_features:
            d = standardized_distance(
                current, bf, cfg.feature_weights, feats, epsilon=cfg.epsilon)
            if d is None:
                distances.append(None)
                valid.append(False)
            else:
                distances.append(d)
                valid.append(True)
        if not any(valid):
            diag["attempts"].append({
                "level": level, "features": list(feats),
                "ess": 0.0, "reason": "no_overlap",
            })
            continue
        # Fill missing distances with large value then kernel
        fill = max((d for d in distances if d is not None), default=1.0) + 1.0
        d_arr = [fill if d is None else d for d in distances]
        w = kernel_weights(d_arr, temperature=cfg.condition_temperature,
                           epsilon=cfg.epsilon)
        ess = effective_sample_size(w)
        diag["attempts"].append({
            "level": level, "features": list(feats),
            "ess": ess, "support_before": ess,
        })
        if ess >= cfg.min_effective_support:
            return level, w, diag

    # Level 5 unconditioned already tried; level 6 = gaussian
    w = np.ones(n) / n
    return 5, w, diag


def _gaussian_paths(
    spot: float,
    n_steps: int,
    n_paths: int,
    sigma_per_min: float,
    mean_per_min: float,
    rng: np.random.Generator,
) -> np.ndarray:
    shocks = rng.normal(mean_per_min, sigma_per_min, size=(n_paths, n_steps))
    rets = shocks  # simple returns approximation
    paths = np.empty((n_paths, n_steps + 1), dtype=float)
    paths[:, 0] = spot
    for t in range(n_steps):
        paths[:, t + 1] = paths[:, t] * (1.0 + rets[:, t])
    return paths


def simulate_paths_v3(
    spot: float,
    n_steps: int,
    sigma_per_min: float,
    *,
    library: ResidualLibrary,
    block_metas: Sequence[ResidualBlockMeta],
    current_state: Mapping[str, float],
    mean_per_min: float = 0.0,
    cfg: Optional[PathModelV3Config] = None,
    snapshot_id: str = "test",
    horizon: str = "30m",
    mode: str = "test",
) -> tuple[np.ndarray, dict]:
    """
    State-conditioned block-bootstrap paths.

    Returns (paths array shape (n_paths, n_steps+1), diagnostics).
    """
    cfg = cfg or PathModelV3Config()
    seed = derive_path_seed(
        snapshot_id, model_version=PATH_MODEL_VERSION, horizon=horizon,
        configuration_hash=config_hash(cfg))
    rng = np.random.default_rng(seed)
    n_paths = cfg.n_paths_for(mode)

    if len(library) < 10:
        if not cfg.allow_gaussian_fallback:
            raise ValueError("library too thin and gaussian fallback disabled")
        paths = _gaussian_paths(
            spot, n_steps, n_paths, sigma_per_min, mean_per_min, rng)
        return paths, {
            "conditioning_backoff_level": 6,
            "reason": "thin_library_gaussian_fallback",
            "effective_support": 0.0,
            "gaussian_fallback": True,
            "seed": seed,
        }

    # Align metas to eligible block starts of typical length
    block_len = max(cfg.block_min, min(cfg.block_max, n_steps))
    starts = _eligible_starts(library, block_len, condition=False)
    if starts.size == 0:
        paths = _gaussian_paths(
            spot, n_steps, n_paths, sigma_per_min, mean_per_min, rng)
        return paths, {
            "conditioning_backoff_level": 6,
            "reason": "no_eligible_blocks",
            "gaussian_fallback": True,
            "seed": seed,
            "effective_support": 0.0,
        }

    # Build feature maps for each eligible start (from metas or library labels)
    meta_by_start = {m.start_index: m for m in block_metas}
    block_feats = []
    session_ids = []
    for s in starts:
        if s in meta_by_start:
            bf = meta_by_start[s].feature_dict()
            sid = str(meta_by_start[s].session_id)
        else:
            bf = {}
            if library.minute_of_session is not None:
                bf["minute_of_session"] = float(library.minute_of_session[s])
            if library.vol_quantile is not None:
                bf["volatility_quantile"] = float(library.vol_quantile[s])
            if library.gex_sign is not None:
                bf["gex_sign"] = float(library.gex_sign[s])
            sid = str(library.session_id[s])
        block_feats.append(bf)
        session_ids.append(sid)

    level, weights, backoff_diag = select_backoff_level(
        current_state, block_feats, cfg)
    if level > cfg.decision_facing_max_backoff and cfg.allow_gaussian_fallback:
        paths = _gaussian_paths(
            spot, n_steps, n_paths, sigma_per_min, mean_per_min, rng)
        return paths, {
            "conditioning_backoff_level": 6,
            "reason": "backoff_exceeded_decision_max",
            "prior_level": level,
            "gaussian_fallback": True,
            "seed": seed,
            "effective_support": 0.0,
            "backoff": backoff_diag,
        }

    weights = apply_session_cap(
        weights, session_ids, max_weight=cfg.max_source_session_weight,
        epsilon=cfg.epsilon)
    ess = effective_sample_size(weights)
    # Source-session concentration = max session weight
    sess_tot: dict[str, float] = {}
    for w, sid in zip(weights, session_ids):
        sess_tot[sid] = sess_tot.get(sid, 0.0) + float(w)
    concentration = max(sess_tot.values()) if sess_tot else 1.0

    # Simulate
    paths = np.empty((n_paths, n_steps + 1), dtype=float)
    paths[:, 0] = spot
    for p in range(n_paths):
        t = 0
        px = spot
        while t < n_steps:
            blen = int(rng.integers(cfg.block_min, cfg.block_max + 1))
            blen = min(blen, n_steps - t)
            # Recompute eligible for this blen if needed — reuse starts of
            # block_len for simplicity when blen <= block_len
            idx = int(rng.choice(len(starts), p=weights))
            start = int(starts[idx])
            # Ensure block fits in session
            span = next(
                (sp for sp in library.session_spans
                 if sp[0] <= start < sp[1]),
                None)
            if span is None or start + blen > span[1]:
                blen = min(blen, (span[1] - start) if span else 1)
                blen = max(1, blen)
            block = library.residuals[start:start + blen]
            for j in range(blen):
                if t >= n_steps:
                    break
                shock = mean_per_min + sigma_per_min * float(block[j])
                px = px * (1.0 + shock)
                t += 1
                paths[p, t] = px
            # If block empty, gaussian step
            if blen <= 0:
                shock = mean_per_min + sigma_per_min * float(rng.normal())
                px = px * (1.0 + shock)
                t += 1
                paths[p, t] = px

    return paths, {
        "conditioning_backoff_level": int(level),
        "effective_support": float(ess),
        "source_session_concentration": float(concentration),
        "gaussian_fallback": False,
        "seed": seed,
        "backoff": backoff_diag,
        "n_candidate_blocks": int(len(starts)),
    }


def forecast_from_paths(
    paths: np.ndarray,
    *,
    spot: float,
    target: Optional[float] = None,
    stop: Optional[float] = None,
    call_wall: Optional[float] = None,
    put_wall: Optional[float] = None,
    gamma_flip: Optional[float] = None,
    range_low: Optional[float] = None,
    range_high: Optional[float] = None,
    diagnostics: Optional[dict] = None,
    uncertainty: float = 0.0,
) -> PathForecastV3:
    diag = dict(diagnostics or {})
    n_paths, cols = paths.shape
    n_steps = cols - 1
    terminals = paths[:, -1]
    q_grid = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
    terminal_q = {q: float(np.quantile(terminals, q)) for q in q_grid}

    start = paths[:, 0:1]
    mfe = np.max(paths - start, axis=1)
    mae = np.max(start - paths, axis=1)
    mfe_q = {q: float(np.quantile(mfe, q)) for q in q_grid}
    mae_q = {q: float(np.quantile(mae, q)) for q in q_grid}

    events = score_path_events(
        paths,
        spot=spot,
        target=target,
        stop=stop,
        call_wall=call_wall,
        put_wall=put_wall,
        gamma_flip=gamma_flip,
        lower=range_low,
        upper=range_high,
    )

    return PathForecastV3(
        p_target_first=float(events.p_target_first),
        p_stop_first=float(events.p_stop_first),
        p_neither=float(events.p_neither),
        p_touch_call_wall=float(events.p_touch_call_wall),
        p_touch_put_wall=float(events.p_touch_put_wall),
        p_cross_gamma_flip=float(events.p_cross_gamma_flip),
        p_call_wall_first=float(events.p_call_wall_first),
        p_put_wall_first=float(events.p_put_wall_first),
        p_neither_wall=float(events.p_neither_wall),
        p_range_survive=float(events.p_range_survive),
        terminal_quantiles=terminal_q,
        mfe_quantiles=mfe_q,
        mae_quantiles=mae_q,
        terminal_mean=float(np.mean(terminals)),
        terminal_std=float(np.std(terminals)),
        effective_support=float(diag.get("effective_support", 0.0)),
        source_session_concentration=float(
            diag.get("source_session_concentration", 1.0)),
        conditioning_backoff_level=int(
            diag.get("conditioning_backoff_level", 5)),
        uncertainty=float(uncertainty),
        n_paths=int(n_paths),
        n_steps=int(n_steps),
        model_version=PATH_MODEL_VERSION,
        diagnostics={**diag, "ambiguous_same_step_rate":
                     float(events.ambiguous_same_step_rate)},
    )
