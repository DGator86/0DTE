"""
tests/test_path_model_v3.py / conditioning / backoff
====================================================
V3 Part 2 PR14 — state-conditioned path simulation (§46).
"""
from __future__ import annotations

import numpy as np
import pytest

from prediction.path_model import ResidualLibrary, build_residual_library
from prediction.path_model_v3 import (
    PATH_MODEL_VERSION,
    PathModelV3Config,
    ResidualBlockMeta,
    apply_session_cap,
    derive_path_seed,
    effective_sample_size,
    forecast_from_paths,
    kernel_weights,
    select_backoff_level,
    simulate_paths_v3,
    standardized_distance,
)


def _library(n_sessions=6, n_per=40, seed=0):
    rng = np.random.default_rng(seed)
    by_sess = {}
    for i in range(n_sessions):
        by_sess[f"S{i}"] = rng.normal(0, 0.001, size=n_per)
    return build_residual_library(
        by_sess,
        gex_sign_by_session={f"S{i}": (1 if i % 2 == 0 else -1)
                             for i in range(n_sessions)},
        vol_quantile_by_session={f"S{i}": i % 3 for i in range(n_sessions)},
    )


def test_version():
    assert PATH_MODEL_VERSION == "v3.0.0"


def test_blocks_never_cross_sessions():
    lib = _library()
    for lo, hi in lib.session_spans:
        sids = set(lib.session_id[lo:hi])
        assert len(sids) == 1


def test_similar_states_get_greater_weight():
    current = {"gex_sign": 1.0, "volatility_quantile": 1.0}
    near = {"gex_sign": 1.0, "volatility_quantile": 1.0}
    far = {"gex_sign": -1.0, "volatility_quantile": 3.0}
    feats = ("gex_sign", "volatility_quantile")
    d_near = standardized_distance(current, near, {}, feats)
    d_far = standardized_distance(current, far, {}, feats)
    assert d_near < d_far
    w = kernel_weights([d_near, d_far], temperature=1.0)
    assert w[0] > w[1]


def test_missing_features_excluded():
    current = {"gex_sign": 1.0, "breadth_alignment": 0.5}
    block = {"gex_sign": 1.0}  # missing breadth
    d = standardized_distance(
        current, block, {"gex_sign": 1.0, "breadth_alignment": 1.0},
        ("gex_sign", "breadth_alignment"))
    assert d == pytest.approx(0.0)


def test_session_cap_enforced():
    weights = np.array([0.5, 0.2, 0.15, 0.1, 0.05])
    sessions = ["A", "A", "B", "C", "D"]
    capped = apply_session_cap(weights, sessions, max_weight=0.25)
    # Aggregate per session
    tot = {}
    for w, s in zip(capped, sessions):
        tot[s] = tot.get(s, 0.0) + float(w)
    assert tot["A"] <= 0.25 + 1e-6
    assert capped.sum() == pytest.approx(1.0)
    assert all(v <= 0.25 + 1e-6 for v in tot.values())


def test_effective_support():
    w = np.ones(10) / 10
    assert effective_sample_size(w) == pytest.approx(10.0)
    peaked = np.array([0.91] + [0.01] * 9)
    assert effective_sample_size(peaked) < 3


def test_backoff_level_explicit():
    current = {"minute_of_session": 100.0, "volatility_quantile": 2.0}
    # Blocks with no overlapping rich features → backoff
    blocks = [{"overnight_gap": 0.01} for _ in range(50)]
    cfg = PathModelV3Config(min_effective_support=30.0)
    level, w, diag = select_backoff_level(current, blocks, cfg)
    assert level >= 4
    assert "attempts" in diag


def test_same_seed_identical_paths():
    lib = _library()
    cfg = PathModelV3Config(n_paths_test=20, block_min=5, block_max=8,
                            min_effective_support=1.0)
    current = {"minute_of_session": 30.0, "volatility_quantile": 1.0,
               "gex_sign": 1.0}
    a, da = simulate_paths_v3(
        600.0, 20, 0.0005, library=lib, block_metas=[],
        current_state=current, cfg=cfg, snapshot_id="snap-1", mode="test")
    b, db = simulate_paths_v3(
        600.0, 20, 0.0005, library=lib, block_metas=[],
        current_state=current, cfg=cfg, snapshot_id="snap-1", mode="test")
    assert np.allclose(a, b)
    assert da["seed"] == db["seed"]


def test_greater_vol_increases_dispersion():
    lib = _library()
    cfg = PathModelV3Config(n_paths_test=80, min_effective_support=1.0)
    current = {"gex_sign": 1.0, "volatility_quantile": 1.0}
    low, _ = simulate_paths_v3(
        600.0, 30, 0.0002, library=lib, block_metas=[],
        current_state=current, cfg=cfg, snapshot_id="v", mode="test")
    high, _ = simulate_paths_v3(
        600.0, 30, 0.002, library=lib, block_metas=[],
        current_state=current, cfg=cfg, snapshot_id="v", mode="test")
    assert np.std(high[:, -1]) > np.std(low[:, -1])


def test_positive_drift_increases_mean():
    lib = _library()
    cfg = PathModelV3Config(n_paths_test=80, min_effective_support=1.0)
    current = {"gex_sign": 1.0}
    zero, _ = simulate_paths_v3(
        600.0, 30, 0.0005, library=lib, block_metas=[],
        current_state=current, mean_per_min=0.0, cfg=cfg,
        snapshot_id="d", mode="test")
    pos, _ = simulate_paths_v3(
        600.0, 30, 0.0005, library=lib, block_metas=[],
        current_state=current, mean_per_min=0.001, cfg=cfg,
        snapshot_id="d", mode="test")
    assert np.mean(pos[:, -1]) > np.mean(zero[:, -1])


def test_gaussian_fallback_labeled():
    empty = ResidualLibrary(residuals=np.zeros(0),
                            session_id=np.zeros(0, dtype=object))
    cfg = PathModelV3Config(n_paths_test=10, allow_gaussian_fallback=True)
    paths, diag = simulate_paths_v3(
        600.0, 5, 0.001, library=empty, block_metas=[],
        current_state={}, cfg=cfg, snapshot_id="g", mode="test")
    assert diag["gaussian_fallback"] is True
    assert diag["conditioning_backoff_level"] == 6
    assert paths.shape == (10, 6)


def test_forecast_contract():
    lib = _library()
    cfg = PathModelV3Config(n_paths_test=30, min_effective_support=1.0)
    paths, diag = simulate_paths_v3(
        600.0, 15, 0.0008, library=lib, block_metas=[],
        current_state={"gex_sign": 1.0}, cfg=cfg, snapshot_id="f", mode="test")
    fc = forecast_from_paths(
        paths, spot=600.0, target=605.0, stop=595.0,
        call_wall=610.0, put_wall=590.0, gamma_flip=598.0,
        diagnostics=diag)
    assert abs(fc.p_target_first + fc.p_stop_first + fc.p_neither - 1.0) <= 1e-5
    assert fc.model_version == "v3.0.0"
    assert 0.5 in fc.terminal_quantiles


def test_derive_seed_stable():
    assert derive_path_seed("a") == derive_path_seed("a")
    assert derive_path_seed("a") != derive_path_seed("b")
