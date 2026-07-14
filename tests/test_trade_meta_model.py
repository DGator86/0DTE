"""
tests/test_trade_meta_model.py
==============================
V3 Part 3 PR25–26 — trade meta decisions (§18–§19 / §49).
"""
from __future__ import annotations

import pytest

from prediction.models.trade_meta import (
    MetaThresholdConfig, TradeMetaModel, apply_hard_vetoes,
    decide_meta_action, meta_features_from_inputs, select_thresholds_nested,
)


def test_abstain_high_uncertainty():
    action, reasons = decide_meta_action(
        p_positive_utility=0.9, expected_order_value=1.0,
        selected_candidate_id="c1", selected_candidate_utility=1.0,
        composite_uncertainty=0.9, ood_score=0.1, data_quality=1.0,
    )
    assert action == "ABSTAIN"
    assert "high_model_uncertainty" in reasons


def test_abstain_ood_and_data_quality():
    a1, _ = decide_meta_action(
        p_positive_utility=0.9, expected_order_value=1.0,
        selected_candidate_id="c1", selected_candidate_utility=1.0,
        composite_uncertainty=0.1, ood_score=0.99, data_quality=1.0,
    )
    assert a1 == "ABSTAIN"
    a2, _ = decide_meta_action(
        p_positive_utility=0.9, expected_order_value=1.0,
        selected_candidate_id="c1", selected_candidate_utility=1.0,
        composite_uncertainty=0.1, ood_score=0.1, data_quality=0.2,
    )
    assert a2 == "ABSTAIN"


def test_no_edge_cases():
    assert decide_meta_action(
        p_positive_utility=0.9, expected_order_value=1.0,
        selected_candidate_id=None, selected_candidate_utility=None,
        composite_uncertainty=0.1, ood_score=0.1, data_quality=1.0,
    )[0] == "NO_EDGE"
    assert decide_meta_action(
        p_positive_utility=0.9, expected_order_value=1.0,
        selected_candidate_id="c1", selected_candidate_utility=-0.5,
        composite_uncertainty=0.1, ood_score=0.1, data_quality=1.0,
    )[0] == "NO_EDGE"
    assert decide_meta_action(
        p_positive_utility=0.9, expected_order_value=-0.1,
        selected_candidate_id="c1", selected_candidate_utility=1.0,
        composite_uncertainty=0.1, ood_score=0.1, data_quality=1.0,
    )[0] == "NO_EDGE"
    assert decide_meta_action(
        p_positive_utility=0.4, expected_order_value=1.0,
        selected_candidate_id="c1", selected_candidate_utility=1.0,
        composite_uncertainty=0.1, ood_score=0.1, data_quality=1.0,
    )[0] == "NO_EDGE"


def test_trade_when_valid():
    action, reasons = decide_meta_action(
        p_positive_utility=0.7, expected_order_value=0.5,
        selected_candidate_id="c1", selected_candidate_utility=0.5,
        composite_uncertainty=0.2, ood_score=0.1, data_quality=0.9,
    )
    assert action == "TRADE"
    assert "meta_probability_above_threshold" in reasons


def test_hard_veto_overrides_trade():
    final, vetoes = apply_hard_vetoes("TRADE", ["daily_loss_limit"])
    assert final == "HARD_VETO"
    assert "daily_loss_limit" in vetoes


def test_prohibited_features_rejected():
    with pytest.raises(ValueError, match="prohibited"):
        meta_features_from_inputs(candidate={"realized_pnl": 1.0})


def test_model_fit_and_decide():
    rows = [{"c_utility": float(i), "e_p_fill": 0.6, "x_dq": 0.9}
            for i in range(10)]
    labels = [1 if i > 4 else 0 for i in range(10)]
    model = TradeMetaModel().fit(rows, labels)
    d = model.decide(
        rows[-1],
        expected_order_value=1.0,
        selected_candidate_id="c1",
        selected_candidate_utility=1.0,
        composite_uncertainty=0.1,
        ood_score=0.05,
        data_quality=0.9,
    )
    assert d.action in ("TRADE", "NO_EDGE", "ABSTAIN")
    d2 = model.decide(
        rows[-1],
        expected_order_value=1.0,
        selected_candidate_id="c1",
        selected_candidate_utility=1.0,
        composite_uncertainty=0.1,
        ood_score=0.05,
        data_quality=0.9,
        hard_vetoes=("market_closed",),
    )
    assert d2.action == "HARD_VETO"
    assert d2.diagnostics["statistical_action"] != "HARD_VETO"
