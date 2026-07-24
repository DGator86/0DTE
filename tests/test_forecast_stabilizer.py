"""Tests for the forecast stabilizer (whiplash control on the cone target)."""
from __future__ import annotations

import pytest

from forecast_stabilizer import (
    BreakSignals, ForecastStabilizer, StabilizerConfig,
)


def _s(**kw) -> ForecastStabilizer:
    return ForecastStabilizer(StabilizerConfig(**kw))


# --------------------------------------------------------------------------- #
# seeding + inertia                                                            #
# --------------------------------------------------------------------------- #
def test_first_tick_seeds_from_raw():
    st = _s()
    r = st.update(raw_target=743.2, spot=742.0, sigma_short=0.2)
    assert r.transition == "seed"
    assert r.target == 743.2 and r.changed


def test_confidence_scales_update_speed():
    # same raw jump; higher confidence => target moves further this tick
    slow = _s(deadband_k=0.0)   # disable deadband to isolate the gain
    fast = _s(deadband_k=0.0)
    for st in (slow, fast):
        st.update(742.0, spot=742.0, sigma_short=0.2)   # seed at spot
    r_slow = slow.update(744.0, spot=742.0, sigma_short=0.2, confidence=0.2)
    r_fast = fast.update(744.0, spot=742.0, sigma_short=0.2, confidence=1.0)
    assert r_fast.alpha > r_slow.alpha
    assert (r_fast.target - 742.0) > (r_slow.target - 742.0)
    # low-confidence target still creeps toward the raw (never frozen)
    assert r_slow.target > 742.0


def test_regime_intensity_scales_update_speed():
    stable = _s(deadband_k=0.0)
    changing = _s(deadband_k=0.0)
    for st in (stable, changing):
        st.update(742.0, 742.0, 0.2)
    r_stable = stable.update(744.0, 742.0, 0.2, regime_change_intensity=0.1)
    r_change = changing.update(744.0, 742.0, 0.2, regime_change_intensity=1.0)
    assert r_change.alpha > r_stable.alpha


def test_exponential_convergence_to_a_held_raw():
    st = _s(deadband_k=0.0, alpha_base=0.5)
    st.update(742.0, 742.0, 0.2)
    prev = 742.0
    for _ in range(30):
        r = st.update(745.0, 742.0, 0.2)
        assert r.target >= prev            # monotone toward the target
        prev = r.target
    assert st.target == pytest.approx(745.0, abs=0.05)


# --------------------------------------------------------------------------- #
# deadband                                                                      #
# --------------------------------------------------------------------------- #
def test_deadband_holds_trivial_moves():
    st = _s(deadband_k=0.5)
    st.update(742.0, 742.0, 0.2)                      # seed
    # a 5-cent nudge with sigma 0.2 -> deadband 0.10; 0.05 < 0.10 -> held
    r = st.update(742.05, 742.0, 0.2)
    assert not r.changed and r.alpha == 0.0
    assert st.target == 742.0


def test_deadband_scales_with_sigma():
    calm = _s(deadband_k=0.5)
    calm.update(742.0, 742.0, sigma_short=0.05)       # deadband 0.025
    r_calm = calm.update(742.10, 742.0, sigma_short=0.05)
    assert r_calm.changed                              # 0.10 > 0.025
    vol = _s(deadband_k=0.5)
    vol.update(742.0, 742.0, sigma_short=0.5)          # deadband 0.25
    r_vol = vol.update(742.10, 742.0, sigma_short=0.5)
    assert not r_vol.changed                            # 0.10 < 0.25


# --------------------------------------------------------------------------- #
# hysteresis                                                                    #
# --------------------------------------------------------------------------- #
def test_reversal_needs_more_evidence_than_continuation():
    # alpha_base=1.0 makes an accepted update land the target exactly on raw,
    # so the hysteresis thresholds are reasoned about directly. sigma=1.0 makes
    # the deadband (0.5) exceed the flat band so leans are unambiguous.
    cfg = dict(deadband_k=0.5, continue_mult=1.0, neutralize_mult=1.6,
               reverse_mult=3.0, alpha_base=1.0)

    # continuation: from a bullish target, extend it by a move just over the
    # continue threshold (0.5) -> accepted
    cont = _s(**cfg)
    cont.update(742.0, 742.0, 1.0)
    cont.update(743.0, 742.0, 1.0)                     # target 743, lean +1
    r_cont = cont.update(743.6, 742.0, 1.0)            # |743.6-743|=0.6 > 0.5
    assert r_cont.transition == "continue" and r_cont.changed

    # reversal: from a modest bullish target, a cross below spot whose
    # magnitude clears the continue threshold (0.5) but NOT the reverse
    # threshold (0.5*1.0*3 = 1.5) is held
    rev = _s(**cfg)
    rev.update(742.0, 742.0, 1.0)
    rev.update(742.7, 742.0, 1.0)                      # target 742.7, lean +1
    r_rev = rev.update(741.4, 742.0, 1.0)              # crosses below; |Δ|=1.3 < 1.5
    assert r_rev.transition == "reverse" and not r_rev.changed
    assert rev.target == 742.7                          # held
    # a reversal beyond the raised threshold IS accepted
    r_big = rev.update(740.5, 742.0, 1.0)              # |742.7-740.5|=2.2 > 1.5
    assert r_big.changed and r_big.lean == -1


# --------------------------------------------------------------------------- #
# structural-break override                                                     #
# --------------------------------------------------------------------------- #
def test_break_override_bypasses_deadband():
    st = _s(deadband_k=0.5, break_alpha=0.9)
    st.update(742.0, 742.0, 0.2)
    # a tiny move that the deadband would normally hold...
    brk = BreakSignals(gamma_flip_failed_reclaim=True)
    r = st.update(742.05, 742.0, 0.2, breaks=brk)
    assert r.changed                                   # override moved it
    assert r.override == ("gamma_flip_failed_reclaim",)
    assert r.target == pytest.approx(742.0 + 0.9 * 0.05, abs=1e-6)


def test_break_override_snaps_hard_toward_raw():
    st = _s(deadband_k=0.5, reverse_mult=3.0, break_alpha=0.9)
    st.update(743.0, 742.0, 0.2)                       # bullish target
    brk = BreakSignals(vwap_loss_confirmed=True, vol_expansion=True)
    r = st.update(740.0, 742.0, 0.2, breaks=brk)       # sharp reversal on a break
    # snaps most of the way despite the reversal hysteresis that would block it
    assert r.target < 741.0 and r.lean == -1
    assert set(r.override) == {"vwap_loss_confirmed", "vol_expansion"}


def test_break_signals_helpers():
    assert not BreakSignals().any()
    b = BreakSignals(macro_release=True, call_wall_acceptance=True)
    assert b.any()
    assert set(b.active()) == {"macro_release", "call_wall_acceptance"}


# --------------------------------------------------------------------------- #
# state management + determinism                                               #
# --------------------------------------------------------------------------- #
def test_reset_clears_state():
    st = _s()
    st.update(743.0, 742.0, 0.2)
    st.reset()
    assert st.target is None and st.lean == 0
    r = st.update(700.0, 700.0, 0.2)
    assert r.transition == "seed" and r.target == 700.0


def test_deterministic_across_instances():
    seq = [(743.0, 742.0), (743.1, 742.2), (742.6, 742.4), (744.0, 742.1)]
    outs = []
    for _ in range(2):
        st = _s()
        run = [st.update(rt, sp, 0.2, confidence=0.7).target for rt, sp in seq]
        outs.append(run)
    assert outs[0] == outs[1]


def test_result_is_json_serializable():
    st = _s()
    st.update(742.0, 742.0, 0.2)
    d = st.update(743.0, 742.0, 0.2).to_dict()
    import json
    assert json.loads(json.dumps(d))["transition"] in (
        "seed", "continue", "neutralize", "reverse", "hold")
