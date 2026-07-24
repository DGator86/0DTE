"""Tests for the Markov-universe regime-label bridge + transition calibration
(regime_calibration.py). Distinct from tests/test_regime_calibration.py, which
covers the unrelated V3 prediction regime model."""
from __future__ import annotations

import numpy as np
import pytest

from matrix_universe import (
    ARCHETYPES, REGIMES, MarkovWorldFeed, UniverseSpec,
)
from regime_calibration import (
    Calibration, LabelConfig, RegimeFeatures, SessionContext,
    archetype_labeler_accuracy, calibrate_from_feed, estimate_transitions,
    features_from_snapshot, label_archetype, label_regime, labeler_accuracy,
)


# --------------------------------------------------------------------------- #
# estimator: round-trip against a known matrix (validates the math)           #
# --------------------------------------------------------------------------- #
def _simulate_markov(matrix: dict, states: list[str], n: int,
                     seed: int) -> list[str]:
    rng = np.random.default_rng(seed)
    s = states[0]
    out = [s]
    for _ in range(n):
        probs = [matrix[s][t] for t in states]
        s = states[int(rng.choice(len(states), p=probs))]
        out.append(s)
    return out


def test_estimate_transitions_recovers_known_matrix():
    states = ["a", "b", "c"]
    true = {
        "a": {"a": 0.90, "b": 0.07, "c": 0.03},
        "b": {"a": 0.10, "b": 0.80, "c": 0.10},
        "c": {"a": 0.05, "b": 0.15, "c": 0.80},
    }
    seq = _simulate_markov(true, states, 200_000, seed=1)
    est = estimate_transitions([seq], states, smoothing=0.01)
    for s in states:
        for t in states:
            assert est[s][t] == pytest.approx(true[s][t], abs=0.01)
        assert sum(est[s].values()) == pytest.approx(1.0)


def test_estimate_transitions_rows_are_stochastic_and_smoothed():
    states = ["x", "y", "z"]
    est = estimate_transitions([["x", "x", "y"]], states, smoothing=0.5)
    for s in states:
        assert sum(est[s].values()) == pytest.approx(1.0)
        assert all(v > 0 for v in est[s].values())     # smoothing => no hard 0


def test_estimate_transitions_unseen_state_uses_prior():
    states = ["p", "q"]
    prior = {"p": {"p": 0.9, "q": 0.1}, "q": {"p": 0.3, "q": 0.7}}
    est = estimate_transitions([["p", "p", "p"]], states, prior=prior)
    assert est["q"]["p"] == pytest.approx(0.3)
    assert est["q"]["q"] == pytest.approx(0.7)


# --------------------------------------------------------------------------- #
# labeler: beats chance on the simulator's own ground truth                    #
# --------------------------------------------------------------------------- #
def test_regime_labeler_beats_chance():
    feed = MarkovWorldFeed(UniverseSpec("d", 5, 8, "range_chop", tick_stride=5))
    acc = labeler_accuracy(feed)
    # latent-regime labeling is approximate; the honest bar is a clear margin
    # over the 1/5 chance line, not near 1.0
    assert acc["accuracy"] > acc["chance"] * 1.3
    assert acc["n"] > 100


def test_archetype_labeler_beats_chance():
    feeds = [MarkovWorldFeed(UniverseSpec(f"a{i}", i * 3 + 1, 5, a, tick_stride=8))
             for i, a in enumerate(ARCHETYPES)]
    acc = archetype_labeler_accuracy(feeds)
    assert acc["accuracy"] > acc["chance"] * 2.0


def test_label_regime_session_relative_extremes():
    ctx = SessionContext(rv_median=0.12)
    cfg = LabelConfig()
    hi = RegimeFeatures(0.0, 0.30, 20, 1.5, 0.001, 1.0)   # rv >> session median
    lo = RegimeFeatures(0.0, 0.03, 20, 0.5, 0.001, 1.0)   # rv << session median
    assert label_regime(hi, ctx, cfg) == "breakout"
    assert label_regime(lo, ctx, cfg) == "compression"
    up = RegimeFeatures(0.01, 0.12, 20, 1.0, 0.001, 1.0)  # clear up-move
    assert label_regime(up, ctx, cfg) == "drift_up"


def test_label_archetype_clear_cases():
    down = np.full(390, -0.00005)      # steady drift down
    assert label_archetype(down, 0.12, 0.0, 0.3, rv_ref=0.12) == "grind_down"
    assert label_archetype(np.zeros(390), 0.12, -0.02, 0.3, rv_ref=0.12) == "gap_shock"
    crash = np.full(390, -0.0001)      # stressed + strongly negative
    assert label_archetype(crash, 0.30, 0.0, 0.1, rv_ref=0.12) == "crash"


def test_features_from_snapshot_extracts_fields():
    feed = MarkovWorldFeed(UniverseSpec("d", 1, 1, "calm_pin", tick_stride=10))
    # snapshot() is sequential (idx-based); the first tick has one bar, so
    # advance until a trailing window exists
    f = None
    for t in feed.timestamps()[:5]:
        f = features_from_snapshot(feed.snapshot(t))
        if f is not None:
            break
    assert f is not None
    assert f.gex_sign in (1.0, -1.0)
    assert f.rv_recent >= 0.0 and f.range_ratio > 0.0


# --------------------------------------------------------------------------- #
# calibration -> valid config -> flows into the generator                      #
# --------------------------------------------------------------------------- #
def test_calibrate_from_feed_produces_valid_matrices():
    spec = UniverseSpec("d", 5, 8, "range_chop", tick_stride=5)
    cal = calibrate_from_feed(lambda: MarkovWorldFeed(spec),
                              MarkovWorldFeed(spec).timestamps())
    assert isinstance(cal, Calibration)
    assert set(cal.arch_transition) == set(ARCHETYPES)
    for a in ARCHETYPES:
        assert set(cal.regime_transition[a]) == set(REGIMES)
        for s in REGIMES:
            assert sum(cal.regime_transition[a][s].values()) == pytest.approx(1.0)
        assert sum(cal.arch_transition[a].values()) == pytest.approx(1.0)
    assert cal.n_sessions == 8 and cal.n_minutes > 100
    assert "arch_transition" in cal.to_dict()


def test_calibrated_override_changes_generated_regime_occupancy():
    """A degenerate calibrated regime matrix (everything -> pin) must dominate
    the generated situation_log, proving the override drives the generator."""
    base = UniverseSpec("d", 3, 3, "range_chop", tick_stride=10)
    all_pin = {a: {s: {q: (1.0 if q == "pin" else 0.0) for q in REGIMES}
                   for s in REGIMES} for a in ARCHETYPES}
    forced = MarkovWorldFeed(base, regime_transition=all_pin)
    pin_frac = sum(1 for s in forced.situation_log if s.regime == "pin") / \
        len(forced.situation_log)
    canonical = MarkovWorldFeed(base)
    canon_pin = sum(1 for s in canonical.situation_log if s.regime == "pin") / \
        len(canonical.situation_log)
    assert pin_frac > 0.95
    assert pin_frac > canon_pin


def test_override_none_matches_canonical_world():
    """Passing no override reproduces the exact canonical world."""
    spec = UniverseSpec("d", 7, 2, "crash", tick_stride=10)
    a = MarkovWorldFeed(spec)
    b = MarkovWorldFeed(spec, arch_transition=None, regime_transition=None)
    assert (a._close == b._close).all()
    assert [s.regime for s in a.situation_log] == [s.regime for s in b.situation_log]
