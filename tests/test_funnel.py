"""
Decision-funnel readout + GEX rank warm-up handling.

Covers:
  - GexRankWindow.is_warm and the premium gate's no-opinion-must-not-veto rule
  - journal.decision_funnel() aggregation (routing, gates, vetoes, flips)
  - unified_loop journaling of routing provenance into signals_json
  - journal's local structure vocabulary staying in sync with the source of
    truth (spread_selector / decision_matrix)
"""
from __future__ import annotations

import datetime as dt
import json
from zoneinfo import ZoneInfo

import pytest

from gate_scorer import Decision, GateConfig, MarketSnapshot, evaluate, score_setup
from gex_window import GexRankWindow
from journal import (
    CREDIT_FAMILIES, DEBIT_FAMILIES, STRUCTURE_CODE_TO_FAMILY,
    UNDEFINED_RISK_FAMILIES, Journal,
)

ET = ZoneInfo("America/New_York")


# --------------------------------------------------------------------------- #
# fixtures                                                                     #
# --------------------------------------------------------------------------- #
def _snap(**kw) -> MarketSnapshot:
    """A textbook premium-selling tape; override fields per test."""
    base = dict(
        spot=602.50, net_gex=4.2e9, gamma_flip=596.0,
        call_wall=603.0, put_wall=598.0, gex_pct_rank=0.88,
        vix9d=12.1, vix=13.0, vix3m=15.2, vvix=92.0, vvix_baseline=95.0,
        straddle_breakeven=4.10, expected_range=3.20,
        adx=12.5, rsi=52.0, bb_width=1.4, bb_width_baseline=2.0,
        vwap=601.9, vwap_reversion_count=5,
        tick_abs_mean=480.0, cvd_slope=0.05,
        now=dt.datetime(2026, 7, 8, 11, 20, tzinfo=ET),
        has_catalyst=False,
    )
    base.update(kw)
    return MarketSnapshot(**base)


def _row(session="2026-07-01", ts="2026-07-01T10:00:00-04:00", spot=600.0,
         family=None, was_traded=0, gate_pass=None, gate_failed=None,
         veto_reasons=None, no_trade_reason="", candidate_present=None,
         gex_pct_rank=0.85, signals=None):
    if gate_pass is None:
        gate_pass = 1 if was_traded else 0
    if candidate_present is None:
        candidate_present = 1 if family in (
            CREDIT_FAMILIES | DEBIT_FAMILIES | UNDEFINED_RISK_FAMILIES
        ) else 0
    return {
        "session_date": session, "ts": ts, "spot": spot,
        "net_gex": 2e9, "gex_regime": "long", "gex_pct_rank": gex_pct_rank,
        "zero_gamma_dist": 1.0, "zero_gamma_dist_pct": 0.001, "adx": 15.0,
        "call_wall": spot + 5, "put_wall": spot - 5,
        "selected_family": family,
        "short_strikes": None, "long_strikes": None, "legs_json": None,
        "credit": None, "candidate_score": None, "ev": None,
        "max_loss": None, "ev_per_risk": None, "theta": None, "gamma": None,
        "prob_profit": None, "prob_touch_short": None,
        "liquidity_score": None, "wall_safety": None,
        "gamma_safety": None, "touch_safety": None,
        "gate_pass": gate_pass, "gate_score": 50.0,
        "gate_failed": json.dumps(gate_failed or []),
        "veto_reasons": json.dumps(veto_reasons or []),
        "decision": "TRADE" if was_traded else "NO_TRADE",
        "no_trade_reason": no_trade_reason,
        "was_traded": was_traded,
        "candidate_present": candidate_present,
        "regime_direction": "none",
        "signals_json": json.dumps(signals) if signals else None,
    }


# --------------------------------------------------------------------------- #
# GexRankWindow warm-up + gate behavior                                        #
# --------------------------------------------------------------------------- #
def test_gex_window_is_warm_tracks_min_samples():
    w = GexRankWindow(min_samples=3)
    assert not w.is_warm
    w.rank(1e9)
    w.rank(2e9)
    assert not w.is_warm
    w.rank(3e9)
    assert w.is_warm


def test_unwarmed_neutral_rank_does_not_fail_premium_gate():
    """The warm-up sentinel 0.5 means "no opinion" — it must not read as
    GEX_WEAK and veto premium (the post-deploy suppression bug)."""
    s = _snap(gex_pct_rank=0.5, gex_rank_warm=False)
    r = evaluate(s, GateConfig())
    assert r.decision is Decision.GO
    assert not any(f.startswith("GEX_WEAK") for f in r.failed_gates)


def test_warm_low_rank_still_fails_premium_gate():
    s = _snap(gex_pct_rank=0.5, gex_rank_warm=True)
    r = evaluate(s, GateConfig())
    assert r.decision is Decision.NO_GO
    assert any(f.startswith("GEX_WEAK") for f in r.failed_gates)


def test_gex_short_still_vetoes_even_unwarmed():
    """The sign check protects the dangerous case regardless of warm-up."""
    s = _snap(net_gex=-1e9, gex_pct_rank=0.5, gex_rank_warm=False)
    r = evaluate(s, GateConfig())
    assert r.decision is Decision.NO_GO
    assert any(f.startswith("GEX_SHORT") for f in r.failed_gates)


def test_unwarmed_rank_scores_neutral_not_zero():
    cfg = GateConfig()
    cold = score_setup(_snap(gex_pct_rank=0.5, gex_rank_warm=False), cfg)
    assert cold["gex_magnitude"] == pytest.approx(cfg.w_gex_magnitude * 0.5)
    warm = score_setup(_snap(gex_pct_rank=0.5, gex_rank_warm=True), cfg)
    assert warm["gex_magnitude"] == pytest.approx(0.0)


def test_synthetic_world_reports_warmup_honestly():
    from synthetic_world import CoupledSyntheticFeed, WorldConfig
    feed = CoupledSyntheticFeed(WorldConfig(days=2, tick_stride=30))
    first = feed.snapshot(feed.timestamps()[0])
    assert first.market.gex_rank_warm is False
    assert first.market.gex_pct_rank == 0.5


# --------------------------------------------------------------------------- #
# journal.decision_funnel                                                      #
# --------------------------------------------------------------------------- #
def test_decision_funnel_aggregates_the_whole_pipeline():
    j = Journal(":memory:")

    # 1. a credit trade that fired
    j.log(_row(family="put_credit", was_traded=1,
               signals={"routed_structure": "PCS", "premium_flip": 0.0}))
    # 2. premium routed but killed by the gate
    j.log(_row(family="iron_condor", gate_pass=0,
               gate_failed=["GEX_WEAK: |GEX| rank 0.50 < 0.60",
                            "TRENDING: ADX 22.0 >= 20"],
               no_trade_reason="gate:GEX_WEAK,TRENDING | selector:no candidate",
               gex_pct_rank=0.5,
               signals={"routed_structure": "IC", "premium_flip": 0.0}))
    # 3. a credit cell flipped to its debit cousin by a dealer veto, traded
    j.log(_row(family="long_call_spread", was_traded=1,
               signals={"routed_structure": "LCS", "premium_flip": 1.0,
                        "regime_vetoes": "term_backwardation"}))
    # 4. a regime stand-down stub (no candidate; veto_reasons = regime vetoes)
    j.log(_row(family=None, candidate_present=0,
               veto_reasons=["short_gamma_regime", "below_gamma_flip"],
               no_trade_reason="stand_down:compression",
               gate_failed=["stand_down:compression"],
               signals={"routed_structure": "NT", "premium_flip": 0.0}))

    f = j.decision_funnel()

    assert f["n"] == 4 and f["sessions"] == 1
    assert f["class_mix"]["credit"] == {"n": 2, "traded": 1}
    assert f["class_mix"]["debit"] == {"n": 1, "traded": 1}
    assert f["class_mix"]["stand_down"]["n"] == 1

    assert f["structure_mix"]["put_credit"] == {"n": 1, "traded": 1, "blocked": 0}
    assert f["structure_mix"]["iron_condor"]["blocked"] == 1

    assert f["routed_structures"] == {"PCS": 1, "IC": 1, "LCS": 1, "NT": 1}
    assert f["premium_flips"]["n"] == 1

    assert f["gate_failures"]["GEX_WEAK"] == 1
    assert f["gate_failures"]["TRENDING"] == 1
    # stub-row stand-down markers stay out of the hard-gate histogram
    assert "stand_down" not in f["gate_failures"]
    assert f["no_trade_reasons"]["gate"] == 1
    assert f["no_trade_reasons"]["selector"] == 1
    assert f["no_trade_reasons"]["stand_down"] == 1

    # regime vetoes from BOTH sources: stub-row veto_reasons + signals_json
    assert f["regime_vetoes"]["short_gamma_regime"] == 1
    assert f["regime_vetoes"]["below_gamma_flip"] == 1
    assert f["regime_vetoes"]["term_backwardation"] == 1

    g = f["gex_rank"]
    assert g["n"] == 4
    assert g["frac_at_warmup_neutral"] == pytest.approx(0.25)
    assert g["frac_below_gate_floor"] == pytest.approx(0.25)


def test_decision_funnel_normalizes_structure_codes():
    """No-trade stub rows store decision_matrix CODES in selected_family;
    the funnel must fold them into the same buckets as family names."""
    j = Journal(":memory:")
    j.log(_row(family="LCS", candidate_present=0, no_trade_reason="no_chain"))
    j.log(_row(family="long_call_spread", was_traded=1))
    f = j.decision_funnel()
    assert f["structure_mix"]["long_call_spread"]["n"] == 2
    assert f["class_mix"]["debit"]["n"] == 2
    assert f["no_trade_reasons"]["no_chain"] == 1


def test_decision_funnel_empty_journal():
    f = Journal(":memory:").decision_funnel()
    assert f["n"] == 0
    assert f["gex_rank"]["n"] == 0


def test_decision_funnel_last_sessions_filter():
    j = Journal(":memory:")
    j.log(_row(session="2026-07-01", family="put_credit", was_traded=1))
    j.log(_row(session="2026-07-02", family="iron_condor", was_traded=1))
    f = j.decision_funnel(last_sessions=1)
    assert f["n"] == 1 and f["sessions"] == 1
    assert "iron_condor" in f["structure_mix"]


# --------------------------------------------------------------------------- #
# unified_loop journals routing provenance                                     #
# --------------------------------------------------------------------------- #
def test_unified_loop_journals_routing_provenance():
    from unified_loop import SyntheticUnifiedFeed, UnifiedOrchestrator

    feed = SyntheticUnifiedFeed(days=3)
    jrn = Journal(":memory:")
    orch = UnifiedOrchestrator(feed=feed, journal=jrn)
    start = dt.datetime(2026, 7, 6, 9, 30, tzinfo=ET)
    orch.run_replay([start + dt.timedelta(minutes=i) for i in range(60)])

    rows = jrn.fetch()
    assert rows
    tagged = 0
    for r in rows:
        sig = json.loads(r["signals_json"]) if r["signals_json"] else {}
        if "routed_structure" in sig:
            tagged += 1
            assert isinstance(sig["routed_structure"], str)
            assert sig["premium_flip"] in (0.0, 1.0)
    assert tagged == len(rows)          # every tick carries provenance

    f = jrn.decision_funnel()
    assert sum(f["routed_structures"].values()) == len(rows)


# --------------------------------------------------------------------------- #
# premium-veto thresholds: dead-zone alignment + calibratable knobs            #
# --------------------------------------------------------------------------- #
def test_classifier_trending_threshold_matches_the_gate():
    """One fact, one threshold: the classifier's trending veto must default to
    the premium gate's max_adx, or the 20-25 dead zone (credit routed into a
    guaranteed TRENDING gate fail) comes back."""
    from regime_classifier import ClassifierConfig
    assert ClassifierConfig().adx_no_premium == GateConfig().max_adx


def test_dead_zone_adx_now_emits_trending_veto():
    from regime_classifier import ClassifierConfig, ClassifierContext, _vetoes
    ctx = ClassifierContext(market=_snap(adx=22.0))     # the former dead zone
    names = [r for r, _ in _vetoes(ctx, ClassifierConfig())]
    assert "trending" in names
    # and it is calibratable back up without code edits
    names_hi = [r for r, _ in _vetoes(ctx, ClassifierConfig(adx_no_premium=25.0))]
    assert "trending" not in names_hi


def test_term_backwardation_ratio_is_calibratable():
    from regime_classifier import ClassifierConfig, ClassifierContext, _vetoes
    flat = ClassifierContext(market=_snap(vix=16.0, vix3m=16.2))
    names = [r for r, _ in _vetoes(flat, ClassifierConfig())]
    assert "term_backwardation" not in names            # default: plain inversion
    inverted = ClassifierContext(market=_snap(vix=16.3, vix3m=16.2))
    names = [r for r, _ in _vetoes(inverted, ClassifierConfig())]
    assert "term_backwardation" in names
    # tightened: even near-inversion forbids premium
    names = [r for r, _ in _vetoes(flat, ClassifierConfig(term_backwardation_ratio=0.98))]
    assert "term_backwardation" in names


def test_trending_veto_flips_credit_to_debit_not_dead():
    """A trending tape must convert a routed credit cell into its debit cousin
    (take what the market gives) instead of a guaranteed gate NO_TRADE."""
    from decision_matrix import NO_PREMIUM_VETOES, PREMIUM_STRUCTURES, decide_from_matrix
    from mtf_matrix import build_matrix, demo_input, regime_rows

    assert "trending" in NO_PREMIUM_VETOES
    rows = build_matrix(demo_input())
    regimes = regime_rows(rows)
    intent = decide_from_matrix(rows, regimes, vetoes=["trending"])
    assert intent.decision.structure not in PREMIUM_STRUCTURES
    if intent.decision.structure != "NT":
        assert "premium veto" in intent.note
        assert "trending" in intent.note                # attribution for the funnel


# --------------------------------------------------------------------------- #
# vocabulary sync with the source of truth                                     #
# --------------------------------------------------------------------------- #
def test_funnel_vocabulary_matches_selector_and_matrix():
    import decision_matrix
    import spread_selector

    # every routing code the matrix can emit (minus NT) is normalizable
    codes = {d.structure for d in decision_matrix.DECISION_TABLE.values()}
    codes |= {"PCS", "CCS", "IC", "IF", "LCS", "LPS"}   # flip targets
    codes.discard("NT")
    assert codes <= set(STRUCTURE_CODE_TO_FAMILY)

    # code -> family agrees with spread_selector's mapping
    for code, fams in spread_selector.STRUCTURE_TO_FAMILIES.items():
        assert code in STRUCTURE_CODE_TO_FAMILY
        mapped = STRUCTURE_CODE_TO_FAMILY[code]
        if len(fams) == 1:
            assert mapped in fams
        else:                                            # BKS -> backspread_*
            assert all(f.startswith(mapped) for f in fams)
            assert fams <= DEBIT_FAMILIES

    # class buckets agree with the engine's own family sets
    assert spread_selector.DEBIT_FAMILIES <= (DEBIT_FAMILIES | {"backspread"})
    assert spread_selector.NAKED_FAMILIES == set(UNDEFINED_RISK_FAMILIES)
    premium_fams = {STRUCTURE_CODE_TO_FAMILY[c]
                    for c in decision_matrix.PREMIUM_STRUCTURES}
    assert premium_fams <= CREDIT_FAMILIES

    # every ranked family in the selector's priors lands in exactly one bucket
    for fam in spread_selector.SelectorConfig().family_weight:
        buckets = [fam in CREDIT_FAMILIES, fam in DEBIT_FAMILIES,
                   fam in UNDEFINED_RISK_FAMILIES]
        assert sum(buckets) == 1, fam
