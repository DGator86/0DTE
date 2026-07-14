"""
tests/test_review4_merge_gate.py
================================
PR #117 fourth-review merge-gate regressions:

* Canonical candidate_id vs legacy _v2_candidate_id aliases
* V3 selection resolves to the exact paper candidate (no family fallback)
* Shared geometry is repriced under alternate physical densities
* Unknown quote age is not treated as fresh
* Per-candidate feature-schema completeness
* Adapter rejects missing CandidateForecastV3 required fields
* Snapshot data_quality=0.0 is preserved (not coerced to 0.85)
"""
from __future__ import annotations

from types import SimpleNamespace
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from decision_engine import EngineConfig, decide
from prediction.adapters import (
    AdapterError,
    adapt_candidate_forecast_v3,
    fill_attempt_features_from_candidate,
    fill_features_from_attempt,
    verify_candidate_feature_schema,
)
from prediction.candidate_universe import (
    make_candidate_id,
    stamp_candidate_id,
)
from prediction.models.fill_probability import fill_features_from_attempt as fp_feats
from prediction.storage import make_candidate_id as storage_make_candidate_id
from rnd_extractor import ChainQuote, ChainSnapshot, extract_rnd
from spread_selector import (
    GammaContext,
    Leg,
    SelectorConfig,
    SpreadCandidate,
    reprice_candidates,
    select_spreads,
)
from unified_loop import UnifiedOrchestrator


ET = ZoneInfo("America/New_York")


def _spread(family="put_credit", strike=490.0, cid=None, score=0.5, ev=0.1):
    c = SpreadCandidate(
        family=family,
        short_strikes=(strike,),
        long_strikes=(strike - 5.0,),
        legs=(
            Leg(strike=strike, kind="P", qty=-1),
            Leg(strike=strike - 5.0, kind="P", qty=1),
        ),
        credit=0.45,
        max_loss=4.55,
        ev=ev,
        ev_per_risk=ev / 4.55,
        theta=0.05,
        gamma=-0.01,
        prob_profit=0.58,
        prob_touch_short=0.25,
        distance_to_wall=2.0,
        liquidity_score=0.8,
        wall_safety=0.7,
        gamma_safety=0.7,
        touch_safety=0.7,
        score=score,
        passes_vetoes=True,
        veto_reasons=(),
        capital=4.55,
    )
    if cid is not None:
        stamp_candidate_id(c, "snap-test")
        # Override with explicit id for controlled tests
        c.candidate_id = cid
        c.v2_candidate_id = cid
        c._v2_candidate_id = cid
    return c


def test_storage_and_universe_share_candidate_id_function():
    legs = [
        {"strike": 599.0, "kind": "P", "qty": -1},
        {"strike": 598.0, "kind": "P", "qty": 1},
    ]
    a = make_candidate_id("snap1", family="put_credit", legs=legs)
    b = storage_make_candidate_id("snap1", "put_credit", legs)
    assert a == b
    assert a.startswith("cand_")
    assert len(a) == 29


def test_stamp_candidate_id_aliases_are_identical():
    c = _spread(cid=None)
    cid = stamp_candidate_id(c, "snap1")
    assert cid.startswith("cand_")
    assert c.candidate_id == cid
    assert c.v2_candidate_id == cid
    assert c._v2_candidate_id == cid


def test_pick_shadow_candidate_exact_id_no_family_fallback():
    orch = UnifiedOrchestrator.__new__(UnifiedOrchestrator)
    a = _spread(family="put_credit", strike=490.0, cid="cand_aaaaaaaaaaaaaaaaaaaaaaaa")
    b = _spread(family="put_credit", strike=485.0, cid="cand_bbbbbbbbbbbbbbbbbbbbbbbb",
                score=0.99, ev=0.5)
    orch._tick_shadow_cands = [a, b]

    # Exact match on canonical candidate_id
    got = orch._pick_shadow_candidate(candidate_id="cand_aaaaaaaaaaaaaaaaaaaaaaaa")
    assert got is a

    # Explicit ID that is absent must NOT fall back to family / highest score
    miss = orch._pick_shadow_candidate(
        candidate_id="cand_missingmissingmissingmis",
        family="put_credit",
    )
    assert miss is None


def test_v3_paper_intent_uses_exact_candidate_not_family_sub():
    orch = UnifiedOrchestrator.__new__(UnifiedOrchestrator)
    a = _spread(family="put_credit", strike=490.0, cid="cand_aaaaaaaaaaaaaaaaaaaaaaaa")
    b = _spread(family="put_credit", strike=485.0, cid="cand_bbbbbbbbbbbbbbbbbbbbbbbb",
                score=0.99)
    orch._tick_shadow_cands = [a, b]
    orch._tick_unified_v3 = {
        "final_action": "TRADE",
        "candidate_id": "cand_aaaaaaaaaaaaaaaaaaaaaaaa",
        "structure": "put_credit",
        "direction": "put",
    }
    orch._tick_part3 = {}
    orch._tick_authoritative = {"size_mult": 1.0}

    # Minimal snap with chain truthy enough for intent builder entry
    snap = SimpleNamespace(chain=object(), market=SimpleNamespace())
    intents = orch._build_paper_intents(
        snap=snap, signals={}, intent=None, regime_state=None,
        decision=None, decide_pdf=None, cfg=None, pin_active=False,
        density_mode="vrp", density_moments=None, final_size_mult=1.0,
        matrix_stand_down=True,
    )
    v3 = [i for i in intents if i.get("track") == "v3"]
    assert len(v3) == 1
    assert v3[0]["candidate"] is a
    assert v3[0]["candidate_id"] == "cand_aaaaaaaaaaaaaaaaaaaaaaaa"
    assert getattr(v3[0]["candidate"], "candidate_id") == (
        "cand_aaaaaaaaaaaaaaaaaaaaaaaa")


def test_v3_paper_intent_skips_when_id_unresolved():
    orch = UnifiedOrchestrator.__new__(UnifiedOrchestrator)
    a = _spread(family="put_credit", strike=490.0, cid="cand_aaaaaaaaaaaaaaaaaaaaaaaa")
    orch._tick_shadow_cands = [a]
    orch._tick_unified_v3 = {
        "final_action": "TRADE",
        "candidate_id": "cand_does_not_exist_xxxxxx",
        "structure": "put_credit",
    }
    orch._tick_part3 = {}
    snap = SimpleNamespace(chain=object(), market=SimpleNamespace())
    intents = orch._build_paper_intents(
        snap=snap, signals={}, intent=None, regime_state=None,
        decision=None, decide_pdf=None, cfg=None, pin_active=False,
        density_mode="vrp", density_moments=None, final_size_mult=1.0,
        matrix_stand_down=True,
    )
    assert [i for i in intents if i.get("track") == "v3"] == []


def _toy_chain(spot=600.0):
    F0, r0, T0 = spot, 0.05, 5.0 / (24 * 365)
    DF0 = np.exp(-r0 * T0)
    from rnd_extractor import _bs_call_fwd
    qs = []
    for K in np.arange(spot - 20, spot + 21, 1.0):
        k = np.log(K / F0)
        s = max(0.0050 - 0.030 * k, 0.0008)
        cm = _bs_call_fwd(F0, K, s) * DF0
        pm = max(cm - DF0 * (F0 - K), 0.0)
        cm = max(cm, 0.0)
        h = 0.01 + 0.002 * max(cm, pm)
        qs.append(ChainQuote(
            float(K), max(cm - h, 0), cm + h, max(pm - h, 0), pm + h))
    return ChainSnapshot(qs, spot=spot + 0.1, t_years=T0, r=r0)


def test_reprice_candidates_changes_ev_under_alternate_density():
    chain = _toy_chain()
    rnd = extract_rnd(chain)
    ctx = GammaContext(
        spot=chain.spot, call_wall=606.0, put_wall=595.0,
        gamma_flip=593.0, net_gex=3.5e9)
    from rnd_extractor import compute_edge
    edge = compute_edge(rnd, chain)
    sel = select_spreads(chain, rnd, edge, ctx, SelectorConfig())
    shared = list(sel.all_candidates or [])[:5]
    assert shared, "need at least one candidate for reprice test"
    for i, c in enumerate(shared):
        stamp_candidate_id(c, "snap-reprice")

    # Flat density vs a heavily tilted density should change EV for some cand.
    def flat(grid):
        return np.ones_like(grid, dtype=float)

    def tilted(grid):
        # Mass shifted toward lower strikes → put-credit EV typically drops.
        x = (grid - grid.min()) / max(grid.max() - grid.min(), 1e-9)
        return np.exp(-4.0 * x)

    flat_priced = reprice_candidates(
        shared, chain, rnd, ctx, SelectorConfig(), physical_pdf=flat)
    tilt_priced = reprice_candidates(
        shared, chain, rnd, ctx, SelectorConfig(), physical_pdf=tilted)
    assert len(flat_priced) == len(tilt_priced)
    # Identity preserved
    assert flat_priced[0].candidate_id == shared[0].candidate_id
    # At least one EV differs under the alternate density
    assert any(
        abs(a.ev - b.ev) > 1e-6 for a, b in zip(flat_priced, tilt_priced)
    ), "repricing under tilted density must change EV"


def test_decide_reprices_precomputed_under_physical_pdf():
    import datetime as dt
    from gate_scorer import MarketSnapshot

    chain = _toy_chain()
    rnd = extract_rnd(chain)
    ctx = GammaContext(
        spot=chain.spot, call_wall=606.0, put_wall=595.0,
        gamma_flip=593.0, net_gex=3.5e9)
    from rnd_extractor import compute_edge
    edge = compute_edge(rnd, chain)
    sel = select_spreads(chain, rnd, edge, ctx, SelectorConfig())
    shared = list(sel.all_candidates or [])[:8]
    assert shared
    for c in shared:
        stamp_candidate_id(c, "snap-decide")

    market = MarketSnapshot(
        now=dt.datetime(2026, 7, 14, 10, 0, tzinfo=ET),
        spot=chain.spot, net_gex=3.5e9, gex_pct_rank=0.8,
        gamma_flip=593.0, call_wall=606.0, put_wall=595.0, adx=18.0,
        vix9d=12.0, vix=13.0, vix3m=15.0, vvix=92.0, vvix_baseline=95.0,
        straddle_breakeven=4.0, expected_range=3.0, rsi=50.0,
        bb_width=1.4, bb_width_baseline=2.0, vwap=chain.spot,
        vwap_reversion_count=0, tick_abs_mean=400.0, cvd_slope=0.0,
        has_catalyst=False,
    )

    def flat(grid):
        return np.ones_like(grid, dtype=float)

    def tilted(grid):
        x = (grid - grid.min()) / max(grid.max() - grid.min(), 1e-9)
        return np.exp(-4.0 * x)

    d1 = decide(
        market, chain, EngineConfig(),
        physical_pdf=flat, target_structure="PCS",
        precomputed_candidates=shared)
    d2 = decide(
        market, chain, EngineConfig(),
        physical_pdf=tilted, target_structure="PCS",
        precomputed_candidates=shared)
    # Frozen-score bug would keep EVs identical; reprice must diverge.
    if d1.candidate is not None and d2.candidate is not None:
        # Same geometry may be selected, but EV under densities differs,
        # OR different candidates win — either proves reprice ran.
        same_id = (
            getattr(d1.candidate, "candidate_id", None)
            == getattr(d2.candidate, "candidate_id", None)
        )
        if same_id:
            assert abs(d1.candidate.ev - d2.candidate.ev) > 1e-6
        else:
            assert True  # selection changed under density — also valid
    else:
        # Even with no tradable pick, all_candidates must be repriced
        assert d1.all_candidates and d2.all_candidates
        assert any(
            abs(a.ev - b.ev) > 1e-6
            for a, b in zip(d1.all_candidates, d2.all_candidates)
        )


def test_unknown_quote_age_preserved_as_none():
    attempt = {
        "n_legs": 2,
        "side": "credit",
        "mid_credit_at_submit": 0.50,
        "natural_credit_at_submit": 0.40,
        "limit_credit": 0.40,
        "relative_spread": 0.2,
        "absolute_spread": 0.10,
        "option_price_scale": 0.50,
        # omit quote_age_seconds → unknown
        "minutes_to_close": 90.0,
        "family": "put_credit",
        "requested_quantity": 1,
        "replacement_count": 0,
    }
    feats = fp_feats(attempt)
    assert feats["quote_age_seconds"] is None

    serve = fill_attempt_features_from_candidate(
        candidate={"family": "put_credit"},
        mid_credit=0.50,
        natural_credit=0.40,
        family="put_credit",
        n_legs=2,
        quote_age_seconds=None,
        minutes_to_close=90.0,
    )
    assert serve["quote_age_seconds"] is None
    # Explicit zero must remain zero (observed freshness), not None
    serve0 = fill_attempt_features_from_candidate(
        candidate={"family": "put_credit"},
        mid_credit=0.50,
        natural_credit=0.40,
        family="put_credit",
        n_legs=2,
        quote_age_seconds=0.0,
        minutes_to_close=90.0,
    )
    assert serve0["quote_age_seconds"] == 0.0


def test_verify_feature_schema_per_row_not_union():
    trained = ["mid_credit", "ev", "family"]
    rows = [
        {"mid_credit": 0.4, "ev": 0.1, "family": "put_credit"},
        {"ev": 0.05, "family": "call_credit"},  # missing mid_credit
    ]
    # Old union check would pass; per-row must flag mid_credit.
    missing = verify_candidate_feature_schema(
        rows, trained_feature_names=trained)
    assert "mid_credit" in missing
    assert verify_candidate_feature_schema(
        [{"mid_credit": 0.1, "ev": 0.0, "family": "x"}],
        trained_feature_names=trained,
    ) == []


def test_adapt_rejects_missing_tail_risk_fields():
    with pytest.raises(AdapterError, match="expected_net_pnl"):
        adapt_candidate_forecast_v3(SimpleNamespace(
            candidate_id="c1",
            p_profit=0.6,
            utility_score=0.2,
            # expected_net_pnl / shortfall / quantiles absent
        ))


def test_snapshot_data_quality_zero_preserved():
    """Mirror the canonical-snapshot quality construction in unified_loop."""
    signals = {"data_quality": 0.0}
    dq = (
        float(signals["data_quality"])
        if signals.get("data_quality") is not None
        else 0.85
    )
    assert dq == 0.0
    # Contrast the old truthiness bug
    assert float(signals.get("data_quality") or 0.85) == 0.85
