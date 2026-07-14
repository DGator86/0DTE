"""
tests/test_part2_replay.py
==========================
V3 Part 2 PR16 — deterministic replay of structural + bundle attach (§48).
"""
from __future__ import annotations

from prediction.contracts import PredictionBundle
from prediction.part2_shadow import run_part2_shadow_tick
from prediction.storage import PredictionStore
from prediction.reports.part2_evaluation import (
    build_part2_evaluation_report,
    summarize_structural_quality,
)


def test_replay_deterministic(tmp_path):
    sources = {
        "oi": {"net_gex": 1e9, "gamma_flip": 598.0,
               "call_wall": 610.0, "put_wall": 590.0,
               "abs_gamma_by_strike": {600.0: 5.0, 605.0: 3.0}},
        "hybrid": {"net_gex": 1.2e9, "gamma_flip": 597.5,
                   "call_wall": 609.5, "put_wall": 590.5},
    }
    hist = [{"ts": "2026-07-14T14:55:00Z", "gamma_flip": 597.0,
             "call_wall": 609.0, "put_wall": 591.0}]
    kwargs = dict(
        spot=600.0, symbol="SPY", ts="2026-07-14T15:00:00Z",
        current_sources=sources, historical_states=hist,
        expected_remaining_move=2.0,
    )

    def run(db_name):
        store = PredictionStore(db_path=str(tmp_path / db_name))
        base = PredictionBundle(
            snapshot_id="replay-1", ts=kwargs["ts"],
            session_date="2026-07-14", symbol="SPY", uncertainty=0.2,
        )
        return run_part2_shadow_tick(
            base_bundle=base, store=store, **kwargs)

    a = run("a.sqlite")
    b = run("b.sqlite")
    assert a.structural_state.to_dict() == b.structural_state.to_dict()
    assert a.bundle.to_dict() == b.bundle.to_dict()


def test_storage_part2_tables(tmp_path):
    store = PredictionStore(db_path=str(tmp_path / "all.sqlite"))
    names = {r[0] for r in store.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    for t in ("structural_states", "regime_outputs", "competing_risk_outputs",
              "path_forecasts", "ensemble_outputs"):
        assert t in names
    store.log_competing_risk_output(
        "s", "directional", "30m", "cr-v1",
        {"p_target_first": 0.4, "p_stop_first": 0.3, "p_neither": 0.3})
    store.log_path_forecast(
        "s", "path-v3", "30m",
        {"p_target_first": 0.4, "p_stop_first": 0.3, "p_neither": 0.3},
        distribution={"q50": 600.0}, diagnostics={"backoff": 0})
    store.log_ensemble_output(
        "s", "p_up", "30m", "ens-v1",
        {"prediction": 0.55, "weights": {"global": 1.0}})
    assert store.fetch_competing_risk_outputs("s")
    assert store.fetch_path_forecasts("s")
    assert store.fetch_ensemble_outputs("s")


def test_evaluation_report_scaffold():
    report = build_part2_evaluation_report(
        structural_summary=summarize_structural_quality([
            {"net_gex_oi": 1.0, "net_gex_volume": None, "net_gex_hybrid": 1.0,
             "quality_score": 0.8, "gex_disagreement": None},
        ]),
        regime_metrics={"log_loss": 0.9, "unclassified_rate": 0.1},
    )
    d = report.to_dict()
    assert d["structural_data_quality"]["n"] == 1
    assert d["regime_model"]["log_loss"] == 0.9
    assert d["diagnostics"]["report_version"] == "v3.part2"
