"""The stabilizer's live wiring in UnifiedOrchestrator._stabilize_forecast:
raw q50 median -> stabilized target + asymmetric band journaled as
v2_fc_stab_* (which flow into live.forecast via forecast_summary)."""
from __future__ import annotations

import datetime as dt
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from forecast_stabilizer import ForecastStabilizer
from unified_loop import UnifiedOrchestrator

ET = ZoneInfo("America/New_York")


def _orch() -> UnifiedOrchestrator:
    # bypass the heavy __init__; the method only needs the stabilizer state
    o = object.__new__(UnifiedOrchestrator)
    o._forecast_stab = ForecastStabilizer()
    o._forecast_stab_session = None
    return o


def _snap(spot=742.0, when=None, er=1.2, catalyst=False):
    when = when or dt.datetime(2026, 7, 24, 10, 0, tzinfo=ET)
    m = SimpleNamespace(spot=spot, now=when, expected_range=er,
                        has_catalyst=catalyst)
    return SimpleNamespace(market=m)


def _bundle(q50=0.0015, q10=-0.002, q90=0.003, erm=0.001, unc=0.4):
    return SimpleNamespace(return_q50_30m=q50, return_q10_30m=q10,
                           return_q90_30m=q90, expected_realized_move_30m=erm,
                           uncertainty=unc)


def _regime(ig=8.0, vetoes=()):
    return SimpleNamespace(global_information_gain=ig, vetoes=list(vetoes))


def test_journals_stabilized_target_and_asymmetric_band():
    o = _orch()
    sig = {}
    o._stabilize_forecast(_snap(), sig, _bundle(), _regime())
    # all stabilizer keys are v2_fc_-prefixed so they flow via forecast_summary
    assert all(k.startswith("v2_fc_") for k in sig)
    # the RETURN is stabilized (seed = raw q50) and the price reconstructed
    assert sig["v2_fc_stab_ret"] == 0.0015
    assert sig["v2_fc_stab_target"] == 742.0 * (1 + 0.0015)
    assert sig["v2_fc_stab_horizon_min"] == 30.0
    # band is asymmetric: q10=-0.2% / q90=+0.3% -> wider downside from target
    lo, hi, tgt = sig["v2_fc_stab_lo"], sig["v2_fc_stab_hi"], sig["v2_fc_stab_target"]
    assert lo < tgt < hi
    assert (tgt - lo) > (hi - tgt)                            # skew preserved


def test_stabilized_target_tracks_spot_without_forecast_revision():
    """Belief (return) stabilization: at a fixed q50, a rising spot moves the
    price target 1:1 WITHOUT the stabilizer reading it as a revision (no
    spurious deadband/hysteresis) — the target is reconstructed from the
    current spot each tick, the return is what's held."""
    o = _orch()
    s1 = {}
    o._stabilize_forecast(_snap(spot=742.0), s1, _bundle(q50=0.0015), _regime())
    s2 = {}
    o._stabilize_forecast(_snap(spot=744.0), s2, _bundle(q50=0.0015), _regime())
    # unchanged belief -> stabilized return unchanged; price tracks spot exactly
    assert s2["v2_fc_stab_ret"] == s1["v2_fc_stab_ret"]
    assert s2["v2_fc_stab_target"] == 744.0 * (1 + 0.0015)
    assert s2["v2_fc_stab_changed"] == 0.0        # no forecast revision


def test_misordered_quantiles_suppress_the_band():
    o = _orch()
    sig = {}
    # q10 > q50 (mis-ordered / garbage) -> no band edges emitted
    o._stabilize_forecast(_snap(), sig, _bundle(q10=0.005, q50=0.0015, q90=0.003),
                          _regime())
    assert "v2_fc_stab_target" in sig
    assert "v2_fc_stab_lo" not in sig and "v2_fc_stab_hi" not in sig


def test_deadband_holds_target_across_ticks():
    o = _orch()
    s1 = {}
    o._stabilize_forecast(_snap(), s1, _bundle(q50=0.0015), _regime())
    seed = s1["v2_fc_stab_target"]
    s2 = {}
    # a tiny q50 nudge -> deadband holds the target, changed=0
    o._stabilize_forecast(_snap(), s2, _bundle(q50=0.00152), _regime())
    assert s2["v2_fc_stab_changed"] == 0.0
    assert s2["v2_fc_stab_target"] == seed


def test_new_session_resets_the_stabilizer():
    o = _orch()
    day1 = dt.datetime(2026, 7, 24, 15, 0, tzinfo=ET)
    o._stabilize_forecast(_snap(when=day1), {}, _bundle(q50=0.01), _regime())
    # next session, a very different median seeds fresh (no carryover blend)
    day2 = dt.datetime(2026, 7, 27, 10, 0, tzinfo=ET)
    s = {}
    o._stabilize_forecast(_snap(when=day2), s, _bundle(q50=-0.01), _regime())
    assert s["v2_fc_stab_target"] == 742.0 * (1 - 0.01)       # reseeded, not blended


def test_catalyst_sets_break_override():
    o = _orch()
    o._stabilize_forecast(_snap(), {}, _bundle(q50=0.0), _regime())   # seed at spot
    s = {}
    # a scheduled catalyst is a structural break -> override bypasses deadband
    o._stabilize_forecast(_snap(catalyst=True), s, _bundle(q50=0.0004), _regime())
    assert s["v2_fc_stab_override"] == 1.0


def test_no_bundle_or_no_median_is_noop():
    o = _orch()
    s = {}
    o._stabilize_forecast(_snap(), s, None, _regime())
    assert s == {}
    s2 = {}
    o._stabilize_forecast(_snap(), s2, SimpleNamespace(return_q50_30m=None), _regime())
    assert s2 == {}
