"""Tests for the hierarchical Markov universe generator (matrix_universe.py)."""
from __future__ import annotations

import numpy as np
import pytest

np.random.default_rng  # touch to fail fast if numpy is broken

from matrix_universe import (
    ARCHETYPES, REGIMES, MarkovWorldFeed, UniverseCatalog, UniverseSpec,
    _ARCH_TRANSITION, _REGIME_TRANSITION, _VAR_PREFERENCE, merge_coverage,
)

MIN_PER_DAY = 390


def _spec(**kw) -> UniverseSpec:
    base = dict(universe_id="test", seed=42, days=2,
                start_archetype="calm_pin", tick_stride=30)
    base.update(kw)
    return UniverseSpec(**base)


# --------------------------------------------------------------------------- #
# transition-matrix integrity                                                 #
# --------------------------------------------------------------------------- #
def test_archetype_rows_are_stochastic_and_complete():
    assert set(_ARCH_TRANSITION) == set(ARCHETYPES)
    for arch, row in _ARCH_TRANSITION.items():
        assert set(row) == set(ARCHETYPES), arch
        assert abs(sum(row.values()) - 1.0) < 1e-9, arch


def test_regime_rows_are_stochastic_for_every_archetype():
    assert set(_REGIME_TRANSITION) == set(ARCHETYPES)
    for arch, rows in _REGIME_TRANSITION.items():
        assert set(rows) == set(REGIMES), arch
        for state, row in rows.items():
            assert set(row) == set(REGIMES), (arch, state)
            assert abs(sum(row.values()) - 1.0) < 1e-9, (arch, state)


def test_variable_preferences_cover_every_regime():
    assert set(_VAR_PREFERENCE) == set(REGIMES)
    for regime, prefs in _VAR_PREFERENCE.items():
        assert set(prefs) == {"gex", "rv", "vrp", "skew", "drift"}, regime


# --------------------------------------------------------------------------- #
# feed determinism + DataFeed protocol                                        #
# --------------------------------------------------------------------------- #
def test_same_spec_generates_identical_world():
    f1, f2 = MarkovWorldFeed(_spec()), MarkovWorldFeed(_spec())
    assert (f1._close == f2._close).all()
    assert f1.day_archetype == f2.day_archetype
    assert [s.regime for s in f1.situation_log] == \
           [s.regime for s in f2.situation_log]


def test_different_seed_generates_different_world():
    f1, f2 = MarkovWorldFeed(_spec()), MarkovWorldFeed(_spec(seed=43))
    assert not (f1._close == f2._close).all()


def test_datafeed_protocol_snapshot_and_settlement():
    feed = MarkovWorldFeed(_spec())
    ticks = feed.timestamps()
    assert len(ticks) == (2 * MIN_PER_DAY) // 30

    snap = feed.snapshot(ticks[0])
    assert snap is not None
    m = snap.market
    assert m.spot > 0 and m.vix > 0 and len(snap.chain.quotes) > 10
    assert snap.bars is not None and len(snap.bars.close) >= 1
    # settlement exists for every generated session and equals the day close
    for day, px in feed.day_close.items():
        assert feed.settlement_price(day) == pytest.approx(px)
    assert feed.settlement_price("1999-01-01") is None


def test_situation_log_labels_every_minute():
    feed = MarkovWorldFeed(_spec())
    assert len(feed.situation_log) == 2 * MIN_PER_DAY
    assert all(s.archetype in ARCHETYPES and s.regime in REGIMES
               for s in feed.situation_log)
    # day 1 label matches the spec's start archetype
    assert feed.situation_log[0].archetype == "calm_pin"


def test_stress_archetypes_raise_realized_vol():
    calm = MarkovWorldFeed(_spec(start_archetype="calm_pin", days=1, seed=7))
    crash = MarkovWorldFeed(_spec(start_archetype="crash", days=1, seed=7))
    calm_rets = np.diff(np.log(calm._close))
    crash_rets = np.diff(np.log(crash._close))
    assert crash_rets.std() > calm_rets.std()


# --------------------------------------------------------------------------- #
# catalog combinatorics + evolution                                           #
# --------------------------------------------------------------------------- #
def test_lattice_enumerates_full_grid():
    cat = UniverseCatalog(days=1)
    specs = cat.lattice()
    assert len(specs) == len(ARCHETYPES) * len(cat.tilts) * len(cat.vol_mults)
    assert len({s.universe_id for s in specs}) == len(specs)  # unique ids
    assert {s.start_archetype for s in specs} == set(ARCHETYPES)


def test_sample_is_deterministic_and_bounded():
    cat = UniverseCatalog(days=1)
    s1, s2 = cat.sample(5), cat.sample(5)
    assert [s.universe_id for s in s1] == [s.universe_id for s in s2]
    assert len(s1) == 5
    assert len(cat.sample(10_000)) == len(cat.lattice())


def test_evolve_overweights_weakest_archetype():
    cat = UniverseCatalog(days=1)
    scores = {a: 1.0 for a in ARCHETYPES}
    scores["crash"] = -5.0          # worst performance
    nxt = cat.evolve(scores)
    assert nxt.generation == cat.generation + 1
    assert nxt.weights["crash"] == max(nxt.weights.values())
    assert nxt.weights["crash"] > nxt.weights["calm_pin"]
    # new generation draws different seeds -> different universes
    assert {s.universe_id for s in nxt.lattice()}.isdisjoint(
        {s.universe_id for s in cat.lattice()})


def test_merge_coverage_counts_minutes():
    feeds = [MarkovWorldFeed(_spec()), MarkovWorldFeed(_spec(seed=99))]
    cov = merge_coverage(feeds)
    assert set(cov) == set(ARCHETYPES)
    total = sum(n for regs in cov.values() for n in regs.values())
    assert total == sum(len(f.situation_log) for f in feeds)
