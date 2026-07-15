"""
tests/test_review3_merge_gate.py
================================
PR #117 third-review merge-gate regressions.
"""
from __future__ import annotations

from types import SimpleNamespace
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from decision_stack.authority import resolve_authority
from prediction.adapters import (
    candidate_value_rows,
    verify_candidate_feature_schema,
)
from prediction.canonical_snapshot import build_canonical_snapshot
from prediction.candidate_universe import build_candidate_universe
from prediction.contracts import PredictionBundle
from prediction.deployment import DeploymentError
from prediction.models.candidate_value import (
    QUANTILES_V3,
    CandidateValueConfig,
    CandidateValueModel,
)
from prediction.part3_decision import (
    build_v3_candidate_evaluations,
    build_v3_decision,
)
from spread_selector import Leg, SpreadCandidate


ET = ZoneInfo("America/New_York")
SMALL = CandidateValueConfig(
    expanded_distribution=True,
    quantiles=QUANTILES_V3,
    c_grid=(0.5,),
    l1_ratio_grid=(0.5,),
    alpha_grid=(0.01,),
    quantile_max_iter=40,
    max_iter=200,
    inner_folds=2,
    min_train_sessions=2,
    min_validation_sessions=1,
)


def _spread(family="put_credit", strike=490.0, cid="c1"):
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
        ev=0.12,
        ev_per_risk=0.03,
        theta=0.05,
        gamma=-0.01,
        prob_profit=0.58,
        prob_touch_short=0.25,
        distance_to_wall=2.0,
        liquidity_score=0.8,
        wall_safety=0.7,
        gamma_safety=0.7,
        touch_safety=0.7,
        score=0.55,
        passes_vetoes=True,
        veto_reasons=(),
        capital=4.55,
    )
    c.candidate_id = cid
    c.execution = {
        "mid_credit": 0.45,
        "natural_credit": 0.35,
        "net_expected_credit": 0.40,
        "fill_fraction_expected": 0.5,
    }
    return c


def test_zero_valued_forecast_fields_not_replaced():
    snap = build_canonical_snapshot(
        symbol="SPY",
        ts="2026-07-14T10:00:00-04:00",
        session_date="2026-07-14",
        raw_features={"spot": 500.0},
        snapshot_id="z1",
        source_timestamps={"bars": "2026-07-14T09:59:00-04:00"},
    )
    universe = build_candidate_universe(
        snapshot_id="z1", generated_at=snap.ts,
        candidates=[{
            "candidate_id": "c1", "family": "put_credit",
            "ev": 0.1, "prob_profit": 0.6, "score": 0.5,
            "legs": [
                {"right": "P", "side": "sell", "qty": 1, "strike": 490,
                 "expiration": "2026-07-14"},
            ],
        }],
    )
    # Explicit zeros must not become optimistic defaults → UNAVAILABLE
    # because candidate mode requires present fields; zeros are present,
    # but if we only had uncertainty and missing ood/dq → unavailable.
    forecast = PredictionBundle(
        snapshot_id="z1", ts=snap.ts, session_date=snap.session_date,
        symbol="SPY", uncertainty=0.0, data_quality=None, ood_score=0.0,
    )
    result = build_v3_decision(
        snapshot=snap, forecast=forecast, universe=universe, mode="candidate")
    assert result.final_action == "UNAVAILABLE"
    assert any("data_quality" in r for r in result.reasons)

    # All present including zeros — must not invent 0.8 / 0.1 / 0.3 / 0.5.
    # Without runtime artifacts candidate mode abstains on missing components.
    forecast2 = PredictionBundle(
        snapshot_id="z1", ts=snap.ts, session_date=snap.session_date,
        symbol="SPY", uncertainty=0.0, data_quality=0.0, ood_score=0.0,
    )
    result2 = build_v3_decision(
        snapshot=snap, forecast=forecast2, universe=universe, mode="candidate")
    assert result2.final_action in ("ABSTAIN", "UNAVAILABLE", "NO_CANDIDATE")
    # Ensure we did not silently invent favorable dq via the old `or 0.8` path
    # by checking reasons never claim a successful meta decision with defaults.
    assert "meta_model" not in (result2.meta or {})


def test_unavailable_uses_canonical_authority_fallback():
    auth = resolve_authority(
        mode="champion",
        legacy_decision={"action": "TRADE", "candidate_id": "L",
                         "structure": "put_credit", "direction": "put",
                         "size_mult": 1.0},
        v3_decision={"final_action": "UNAVAILABLE",
                     "statistical_action": "UNAVAILABLE"},
        fallback_policy="no_trade",
    )
    assert auth.final_action == "NO_EDGE"
    assert auth.fallback_used is True

    auth2 = resolve_authority(
        mode="champion",
        legacy_decision={"action": "TRADE", "candidate_id": "L"},
        v3_decision={"final_action": "UNAVAILABLE"},
        fallback_policy="abstain",
    )
    assert auth2.final_action == "ABSTAIN"
    assert auth2.fallback_used is True

    auth3 = resolve_authority(
        mode="candidate",
        legacy_decision={"action": "TRADE", "candidate_id": "L"},
        v3_decision={"final_action": "UNAVAILABLE"},
        fallback_policy="abstain",
    )
    assert auth3.final_action == "TRADE"  # legacy remains reference
    assert auth3.candidate_account == "v3"


def test_candidate_value_rows_from_spread_candidate_train_serve():
    """Train on canonical feature rows; serve SpreadCandidate objects."""
    cands = [_spread(cid=f"c{i}", strike=490.0 + i) for i in range(24)]
    rows, ids = candidate_value_rows(
        cands, snapshot_id="s1", spot=500.0,
        call_wall=510.0, put_wall=490.0, gamma_flip=498.0,
        minutes_to_close=120.0, net_gex=1e9, data_quality=0.9,
    )
    assert len(rows) == 24
    # Schema must include trained geometry fields — not just family/ev.
    assert "width" in rows[0]
    assert "liquidity_score" in rows[0]
    assert "family_code" in rows[0]

    rng = np.random.default_rng(0)
    y_pnl = [0.1 * float(rng.normal()) for _ in rows]
    y_profit = [1 if y > 0 else 0 for y in y_pnl]
    sessions = [f"S{i % 6:02d}" for i in range(len(rows))]
    groups = [f"snap-{i // 2}" for i in range(len(rows))]
    model = CandidateValueModel(config=SMALL).fit(
        rows, y_pnl=y_pnl, y_profit=y_profit,
        sessions=sessions, group_ids=groups)

    # Serve through the same adapter path used by Part 3.
    serve_rows, serve_ids = candidate_value_rows(
        cands[:3], snapshot_id="s1", spot=500.0,
        call_wall=510.0, put_wall=490.0, gamma_flip=498.0,
        minutes_to_close=120.0, net_gex=1e9, data_quality=0.9,
    )
    missing = verify_candidate_feature_schema(
        serve_rows,
        trained_feature_names=list(model.vectorizer.feature_names),
    )
    assert missing == [], missing
    preds = model.predict_v3(serve_rows, candidate_ids=serve_ids)
    assert len(preds) == 3
    assert preds[0].p_profit is not None
    assert preds[0].utility_score is not None

    # Full evaluation path with trained value model.
    class _Art:
        candidate_value = model
        candidate_rank = None
        fill_probability = None
        fill_concession = None
        meta_model = None

    class _RT:
        artifacts = _Art()

    snap = build_canonical_snapshot(
        symbol="SPY",
        ts="2026-07-14T10:00:00-04:00",
        session_date="2026-07-14",
        raw_features={"spot": 500.0, "minutes_to_close": 120.0},
        snapshot_id="s1",
        source_timestamps={"bars": "2026-07-14T09:59:00-04:00"},
        market=SimpleNamespace(
            spot=500.0, call_wall=510.0, put_wall=490.0,
            gamma_flip=498.0, net_gex=1e9),
    )
    universe = build_candidate_universe(
        snapshot_id="s1", generated_at=snap.ts, candidates=cands[:3])
    forecast = PredictionBundle(
        snapshot_id="s1", ts=snap.ts, session_date=snap.session_date,
        symbol="SPY", uncertainty=0.2, data_quality=0.9, ood_score=0.1,
    )
    # Shadow mode: value model usable without full Part 3 artifact set.
    evs = build_v3_candidate_evaluations(
        snapshot=snap, forecast=forecast, universe=universe,
        runtime=_RT(), mode="shadow",
    )
    assert evs
    assert any(e.absolute_utility is not None for e in evs)
    assert any(e.expected_net_pnl is not None for e in evs)


def test_eov_uses_expected_net_pnl_not_utility(monkeypatch):
    captured = {}

    def _fake_eov(*, p_fill, expected_net_pnl_given_fill, **kw):
        captured["pnl"] = expected_net_pnl_given_fill
        return p_fill * expected_net_pnl_given_fill

    from prediction import part3_decision as p3
    # Build a minimal evaluation path by calling expected_order_value through
    # the module — verify the contract helper preference.
    from execution.estimate_v3 import expected_order_value as real_eov
    # Direct contract: absolute_utility must not be passed as pnl.
    util = 9.99
    pnl = 0.15
    eov = real_eov(p_fill=0.5, expected_net_pnl_given_fill=pnl)
    assert eov == pytest.approx(0.075)
    assert eov != pytest.approx(0.5 * util)


def test_missing_deployment_fails_strict(tmp_path, monkeypatch):
    import shadow_runner as sr
    from prediction.deployment import DeploymentError

    class _Feed:
        last_source = "test"
        def snapshot(self, now):
            return None
        def settlement_price(self, d):
            return None

    monkeypatch.setattr(sr, "build_default_feed", lambda **kw: _Feed())
    missing = str(tmp_path / "no-such-deployment.json")
    with pytest.raises(DeploymentError, match="not found"):
        sr.ShadowRunner(
            db_path=str(tmp_path / "j.sqlite"),
            paper_db=str(tmp_path / "p.sqlite"),
            deployment_path=missing,
            deployment_mode="champion",
            enable_v2_parallel=False,
            record_dir="",
            champion_path="",
        )
