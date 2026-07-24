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


# --------------------------------------------------------------------------- #
# review fixes (skew convention, Dirichlet evolution, gaps, coverage)         #
# --------------------------------------------------------------------------- #
def test_skew_preferences_match_pricing_convention():
    """Chain pricing is s(K) = s_atm - skew*ln(K/F): POSITIVE skew raises
    put-strike vol. Down regimes must steepen the put skew; up regimes bid
    the calls."""
    assert _VAR_PREFERENCE["drift_down"]["skew"] == "high"   # put-heavy
    assert _VAR_PREFERENCE["breakout"]["skew"] == "high"     # fear premium
    assert _VAR_PREFERENCE["drift_up"]["skew"] == "low"      # call bid


def test_positive_skew_prices_put_wing_richer():
    feed = MarkovWorldFeed(_spec())
    feed._skew = np.full_like(feed._skew, 0.075)             # force put-heavy
    chain = feed._chain(0)
    spot = chain.spot
    low = min(chain.quotes, key=lambda q: q.strike)
    high = max(chain.quotes, key=lambda q: q.strike)
    # equidistant OTM wings: the put wing must carry more premium
    dist = min(spot - low.strike, high.strike - spot) * 0.8
    put = min(chain.quotes, key=lambda q: abs(q.strike - (spot - dist)))
    call = min(chain.quotes, key=lambda q: abs(q.strike - (spot + dist)))
    assert put.put_mid > call.call_mid


def test_transition_jitter_is_deterministic_and_off_by_default():
    base = MarkovWorldFeed(_spec())
    assert base._arch_T is _ARCH_TRANSITION          # jitter=0 -> canonical
    j1 = MarkovWorldFeed(_spec(transition_jitter=0.05))
    j2 = MarkovWorldFeed(_spec(transition_jitter=0.05))
    assert j1._arch_T == j2._arch_T                  # seeded determinism
    assert j1._arch_T != _ARCH_TRANSITION            # actually perturbed
    # perturbed rows stay proper distributions over the full state sets
    for state, row in j1._arch_T.items():
        assert set(row) == set(ARCHETYPES)
        assert abs(sum(row.values()) - 1.0) < 1e-9
    for arch, rows in j1._regime_T.items():
        for state, row in rows.items():
            assert set(row) == set(REGIMES)
            assert abs(sum(row.values()) - 1.0) < 1e-9


def test_lattice_applies_jitter_from_generation_one():
    cat = UniverseCatalog(days=1)
    assert all(s.transition_jitter == 0.0 for s in cat.lattice())
    gen1 = cat.evolve({})
    assert all(s.transition_jitter == pytest.approx(0.02)
               for s in gen1.lattice())


def test_gap_shock_gaps_in_both_directions():
    signs = set()
    for seed in range(24):
        feed = MarkovWorldFeed(_spec(start_archetype="gap_shock", days=1,
                                     seed=seed))
        gap = np.log(feed._close[0] / feed.spec.base_spot)
        signs.add(gap > 0)
        if signs == {True, False}:
            break
    assert signs == {True, False}


def test_evaluated_coverage_counts_strided_ticks():
    feed = MarkovWorldFeed(_spec())
    ev = feed.evaluated_coverage()
    assert sum(n for regs in ev.values() for n in regs.values()) == \
        len(feed.timestamps())
    # generated-minute coverage is the superset
    cov = feed.coverage()
    for a, regs in ev.items():
        for r, n in regs.items():
            assert cov[a][r] >= n
    merged = merge_coverage([feed], evaluated=True)
    assert sum(n for regs in merged.values() for n in regs.values()) == \
        len(feed.timestamps())


# --------------------------------------------------------------------------- #
# direction-dependent skew + archetype-biased breakout direction              #
# --------------------------------------------------------------------------- #
def test_skew_state_tracks_move_direction():
    from matrix_universe import _skew_state
    # up moves flatten toward the calls, down moves steepen the puts
    assert _skew_state("drift_up", 0.0) == "low"
    assert _skew_state("drift_down", 0.0) == "high"
    assert _skew_state("breakout", 1.0) == "low"    # up breakout -> call bid
    assert _skew_state("breakout", -1.0) == "high"  # down breakout -> put skew
    assert _skew_state("pin", 1.0) == "mid"
    assert _skew_state("compression", -1.0) == "mid"


def test_breakout_direction_biased_by_archetype():
    from matrix_universe import _breakout_direction
    rng = np.random.default_rng(0)
    crash_up = sum(_breakout_direction("crash", rng) > 0 for _ in range(2000))
    squeeze_up = sum(_breakout_direction("squeeze_melt_up", rng) > 0
                     for _ in range(2000))
    calm_up = sum(_breakout_direction("calm_pin", rng) > 0 for _ in range(2000))
    assert crash_up / 2000 < 0.2          # crash breaks down
    assert squeeze_up / 2000 > 0.8        # squeeze breaks up
    assert 0.4 < calm_up / 2000 < 0.6     # symmetric default


def test_upside_archetype_is_call_skewed_downside_is_put_skewed():
    """Coherence: a squeeze_melt_up world should carry a lower (more
    call-heavy) mean smile slope than a crash world, since its moves resolve
    up and skew follows direction."""
    up = np.mean([MarkovWorldFeed(_spec(start_archetype="squeeze_melt_up",
                                        days=1, seed=s))._skew.mean()
                  for s in range(6)])
    down = np.mean([MarkovWorldFeed(_spec(start_archetype="crash",
                                          days=1, seed=s))._skew.mean()
                    for s in range(6)])
    assert up < down


def test_breakout_skew_override_is_deterministic():
    # the direction override must not break per-spec determinism
    f1 = MarkovWorldFeed(_spec(start_archetype="vol_expansion", seed=3))
    f2 = MarkovWorldFeed(_spec(start_archetype="vol_expansion", seed=3))
    assert (f1._skew == f2._skew).all()


# --------------------------------------------------------------------------- #
# skew responsiveness + reproducibility (PR #141 follow-up)                    #
# --------------------------------------------------------------------------- #
def test_skew_ou_target_follows_override_fast():
    """A sustained direction override must move the skew VALUE most of the way
    to its target within a typical breakout (~15 min), not wait ~58 min for
    the discrete Markov state to switch."""
    from matrix_universe import VariableChain, _VAR_TARGETS
    hi = _VAR_TARGETS["skew"]["high"]
    mid = _VAR_TARGETS["skew"]["mid"]
    ch = VariableChain("skew", np.random.default_rng(1))
    val = mid
    for _ in range(15):
        val = ch.step("breakout", "high")   # sustained down-breakout
    assert (val - mid) / (hi - mid) > 0.5   # >50% of the way to put-heavy

    lo = _VAR_TARGETS["skew"]["low"]
    ch2 = VariableChain("skew", np.random.default_rng(2))
    val2 = mid
    for _ in range(15):
        val2 = ch2.step("breakout", "low")  # sustained up-breakout
    # symmetric assertion: >50% of the way toward the call-heavy target
    assert (mid - val2) / (mid - lo) > 0.5


def test_override_does_not_change_non_overridden_vars():
    """gex/rv/vrp/drift never receive an override, so their OU target stays
    the discrete state — unchanged behavior."""
    from matrix_universe import VariableChain
    a = VariableChain("gex", np.random.default_rng(5))
    b = VariableChain("gex", np.random.default_rng(5))
    seq_a = [a.step("pin") for _ in range(30)]
    seq_b = [b.step("pin") for _ in range(30)]
    assert seq_a == seq_b


def test_simulator_config_is_self_describing():
    from matrix_universe import simulator_config, _BREAKOUT_P_UP, _SIMULATOR_VERSION
    cfg = simulator_config()
    assert cfg["version"] == _SIMULATOR_VERSION
    assert cfg["breakout_p_up"] == _BREAKOUT_P_UP
    assert cfg["breakout_p_up"]["crash"] < 0.5      # crash breaks down
    assert cfg["breakout_p_up"]["squeeze_melt_up"] > 0.5
    assert set(cfg["var_targets"]) == {"gex", "rv", "vrp", "skew", "drift"}
    assert cfg["var_ou_theta"]["skew"] >= 0.08      # responsiveness pinned


def test_simulator_config_is_complete():
    """The config must carry EVERY generative layer, not just the recently
    touched params — the transition matrices, var preference, and all the
    inline price/gap/chain/snapshot params via gen_params."""
    from matrix_universe import (simulator_config, _ARCH_TRANSITION,
                                 _REGIME_TRANSITION, _GEN_PARAMS)
    cfg = simulator_config()
    assert cfg["arch_transition"] == _ARCH_TRANSITION
    assert cfg["regime_transition"] == _REGIME_TRANSITION
    gp = cfg["gen_params"]
    for block in ("session", "gap", "price_process", "dealer_map", "bars",
                  "chain", "snapshot_map", "initial_regime", "jitter", "floors"):
        assert block in gp, block
    # key price-path coefficients are present
    assert gp["price_process"]["breakout_drift"] == 0.12
    assert gp["chain"]["strike_span"] == 25.0
    assert gp["gap"]["gap_shock_p_down"] == 0.6
    # returned config is a copy — mutating it must not corrupt the module state
    gp["price_process"]["breakout_drift"] = 999.0
    assert _GEN_PARAMS["price_process"]["breakout_drift"] == 0.12


def test_simulator_config_hash_is_stable_and_content_sensitive():
    """Same constants -> same hash across calls; a changed constant -> a
    different hash (so a forgotten version bump can't hide a model change)."""
    from matrix_universe import simulator_config_hash, _GEN_PARAMS
    h1 = simulator_config_hash()
    assert h1 == simulator_config_hash()
    assert len(h1) == 64                            # sha256 hex
    saved = _GEN_PARAMS["price_process"]["breakout_drift"]
    _GEN_PARAMS["price_process"]["breakout_drift"] = saved + 0.01
    try:
        assert simulator_config_hash() != h1
    finally:
        _GEN_PARAMS["price_process"]["breakout_drift"] = saved
    assert simulator_config_hash() == h1            # restored
