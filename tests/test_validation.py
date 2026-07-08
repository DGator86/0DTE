"""Tests for the validation infrastructure: validation_reports journal table,
the daily/weekly validation pipeline, degradation flags, mtf feature toggles,
the run-config loader, and the feature-impact report."""
from __future__ import annotations

import json
import os

import pytest

from journal import COLUMNS, Journal


# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #
def _seed_journal(db_path: str, n: int = 40, session: str = "2026-07-07") -> None:
    """Settled candidates: even rows traded (winners), odd rows gate-blocked."""
    jrn = Journal(db_path)
    for i in range(n):
        row = {c: None for c in COLUMNS}
        row.update({
            "session_date": session,
            "ts": f"{session}T10:{i:02d}:00-04:00",
            "spot": 600.0,
            "gex_regime": "long" if i % 3 else "short",
            "was_traded": 1 if i % 2 == 0 else 0,
            "candidate_present": 1,
            "gate_pass": 1 if i % 2 == 0 else 0,
            "decision": "TRADE" if i % 2 == 0 else "NO_TRADE",
            "credit": 1.0, "ev": 0.1, "prob_profit": 0.6,
            "legs_json": json.dumps([
                {"qty": -1, "strike": 601.0, "kind": "C"},
                {"qty": 1, "strike": 603.0, "kind": "C"},
            ]),
            "regime_direction": "call",
        })
        jrn.log(row)
    jrn.settle_session(session, 600.5)
    jrn.close()


@pytest.fixture
def seeded_db(tmp_path):
    db = str(tmp_path / "shadow.db")
    _seed_journal(db)
    return db


# --------------------------------------------------------------------------- #
# validation_reports table round-trip                                         #
# --------------------------------------------------------------------------- #
def test_validation_report_roundtrip(tmp_path):
    db = str(tmp_path / "j.db")
    jrn = Journal(db)
    rid = jrn.log_validation_report(
        "2026-07-08", "daily",
        {"journal": {"win_rate": 0.61}}, "summary text",
        flags=[{"flag": "x", "severity": "warn", "detail": "d"}],
        notes="a note")
    assert rid == 1

    rows = jrn.fetch_validation_reports()
    assert len(rows) == 1
    r = rows[0]
    assert r["report_type"] == "daily"
    assert r["report_date"] == "2026-07-08"
    assert r["metrics"]["journal"]["win_rate"] == 0.61
    assert r["flags"][0]["flag"] == "x"
    assert r["notes"] == "a note"
    assert r["generated_at"]
    jrn.close()


def test_validation_report_filters(tmp_path):
    db = str(tmp_path / "j.db")
    jrn = Journal(db)
    jrn.log_validation_report("2026-07-01", "daily", {}, "d1")
    jrn.log_validation_report("2026-07-05", "weekly", {}, "w1")
    jrn.log_validation_report("2026-07-08", "feature_impact", {}, "f1")

    assert len(jrn.fetch_validation_reports()) == 3
    assert [r["report_type"] for r in jrn.fetch_validation_reports()] == \
        ["feature_impact", "weekly", "daily"]          # newest first
    assert len(jrn.fetch_validation_reports(report_type="daily")) == 1
    assert len(jrn.fetch_validation_reports(since="2026-07-05")) == 2
    assert len(jrn.fetch_validation_reports(limit=1)) == 1
    jrn.close()


def test_validation_table_created_on_legacy_db(tmp_path):
    """Reopening an existing DB (schema pre-dating the table) must add it."""
    db = str(tmp_path / "j.db")
    Journal(db).close()                       # create
    jrn = Journal(db)                         # reopen — CREATE IF NOT EXISTS
    assert jrn.fetch_validation_reports() == []
    jrn.close()


# --------------------------------------------------------------------------- #
# Daily / weekly pipeline                                                     #
# --------------------------------------------------------------------------- #
def test_daily_validation_report(seeded_db):
    from validation_pipeline import run_daily_validation

    rep = run_daily_validation(seeded_db, record_dir="")
    assert rep["report_type"] == "daily"
    jm = rep["metrics"]["journal"]
    assert jm["n_settled_trades"] == 20
    assert jm["win_rate"] == 1.0               # all winners by construction
    assert jm["brier"] is not None
    # no recordings -> insufficient_data info flag, walk_forward None
    assert rep["metrics"]["walk_forward"] is None
    assert any(f["flag"] == "insufficient_data" for f in rep["flags"])
    # persisted
    jrn = Journal(seeded_db)
    stored = jrn.fetch_validation_reports(report_type="daily")
    jrn.close()
    assert len(stored) == 1
    assert stored[0]["summary"] == rep["summary"]


def test_weekly_validation_report(seeded_db):
    from validation_pipeline import run_daily_validation, run_weekly_validation

    run_daily_validation(seeded_db, record_dir="")     # feeds daily_aggregate
    rep = run_weekly_validation(seeded_db, record_dir="")
    m = rep["metrics"]
    assert rep["report_type"] == "weekly"
    assert set(m["per_regime"]) == {"long", "short"}
    assert m["per_regime"]["long"]["taken"]["n"] > 0
    assert m["daily_aggregate"]["n_daily_reports"] == 1
    assert isinstance(m["recommendations"], list) and m["recommendations"]
    assert isinstance(m["gate_trend"], list)


def test_daily_deltas_vs_previous(seeded_db):
    from validation_pipeline import run_daily_validation

    first = run_daily_validation(seeded_db, record_dir="", report_date="2026-07-07")
    second = run_daily_validation(seeded_db, record_dir="", report_date="2026-07-08")
    assert first["metrics"]["deltas"] == {}
    d = second["metrics"]["deltas"]
    assert d["vs_report_date"] == "2026-07-07"
    assert d["win_rate"] == 0.0                # same underlying data


# --------------------------------------------------------------------------- #
# Flags                                                                       #
# --------------------------------------------------------------------------- #
def test_compute_flags_degradation():
    from validation_pipeline import compute_flags

    prior = [{"report_date": "2026-07-07",
              "metrics": {"journal": {"win_rate": 0.60},
                          "walk_forward": {"mean_sharpe": 2.0}}}]
    metrics = {
        "journal": {
            "win_rate": 0.40,                  # >15% relative drop
            "brier_skill": -0.10,              # below floor
            "gate_effectiveness": {
                "trades_taken": {"mean": 0.10},
                "blocked_by_gate": {"mean": 0.50},   # reversal
            },
        },
        "walk_forward": {"mean_sharpe": 1.0},  # >20% below trailing 2.0
    }
    names = {f["flag"] for f in compute_flags(metrics, prior)}
    assert names == {"gate_effectiveness_reversed", "brier_skill_negative",
                     "sharpe_degraded", "win_rate_degraded"}


def test_compute_flags_healthy():
    from validation_pipeline import compute_flags

    prior = [{"report_date": "2026-07-07",
              "metrics": {"journal": {"win_rate": 0.55},
                          "walk_forward": {"mean_sharpe": 1.5}}}]
    metrics = {
        "journal": {
            "win_rate": 0.60, "brier_skill": 0.05,
            "gate_effectiveness": {
                "trades_taken": {"mean": 0.50},
                "blocked_by_gate": {"mean": -0.20},
            },
        },
        "walk_forward": {"mean_sharpe": 1.6},
    }
    assert compute_flags(metrics, prior) == []


def test_alert_only_on_alert_severity(capsys):
    from validation_pipeline import send_degradation_alert

    assert send_degradation_alert(
        {"report_type": "daily", "summary": "s",
         "flags": [{"flag": "w", "severity": "warn", "detail": "d"}]}) is False
    assert send_degradation_alert(
        {"report_type": "daily", "summary": "s",
         "flags": [{"flag": "a", "severity": "alert", "detail": "bad"}]}) is True
    out = capsys.readouterr().out
    assert "bad" in out


# --------------------------------------------------------------------------- #
# mtf feature toggles                                                         #
# --------------------------------------------------------------------------- #
def test_mtf_disabled_vars_param():
    from mtf_matrix import VARS, build_matrix, demo_input, regime_rows

    channel = {v.name for v in VARS if v.domain == "channel"}
    rows_all = build_matrix(demo_input())
    rows_off = build_matrix(demo_input(), disabled_vars=channel)
    assert any(r.domain == "channel" for r in rows_all)
    assert not any(r.domain == "channel" for r in rows_off)
    regs = regime_rows(rows_off, disabled_vars=channel)
    assert set(regs) == {"compression", "trend", "breakout"}


def test_mtf_disabled_vars_global():
    from mtf_matrix import (build_matrix, demo_input, get_disabled_vars,
                            set_disabled_vars)

    try:
        set_disabled_vars({"adx_strength"})
        assert "adx_strength" in get_disabled_vars()
        rows = build_matrix(demo_input())
        assert not any(r.variable == "adx_strength" for r in rows)
    finally:
        set_disabled_vars(None)
    assert get_disabled_vars() == frozenset()
    rows = build_matrix(demo_input())
    assert any(r.variable == "adx_strength" for r in rows)


def test_regime_rows_skip_disabled_in_blend():
    """Disabling a blend member changes the regime confidence (weight drops
    out of the weighted mean) — proving the blend actually skips it."""
    from mtf_matrix import build_matrix, demo_input, regime_rows

    rows = build_matrix(demo_input())
    with_all = regime_rows(rows)["compression"]["1h"]
    without = regime_rows(rows, disabled_vars={"adx_strength"})["compression"]["1h"]
    assert with_all is not None and without is not None
    assert with_all != without


# --------------------------------------------------------------------------- #
# Config loader                                                               #
# --------------------------------------------------------------------------- #
def test_load_shipped_configs():
    from config_loader import load_run_config

    base = load_run_config("configs/baseline.yaml")
    var = load_run_config("configs/with_channels.yaml")
    assert base.name == "baseline"
    assert "bb_squeeze" in base.disabled_vars
    assert len(base.disabled_vars) == 11
    assert var.disabled_vars == frozenset()


def test_load_config_overrides(tmp_path):
    from config_loader import load_run_config

    p = tmp_path / "c.yaml"
    p.write_text(
        "name: t\n"
        "overrides:\n"
        "  gate.max_adx: 22.5\n"
        "  selector.min_ev: 0.02\n"
        "  classifier.min_dominant_confidence: 55\n")
    rc = load_run_config(str(p))
    assert rc.engine_cfg.gate.max_adx == 22.5
    assert rc.engine_cfg.selector.min_ev == 0.02
    assert rc.classifier_cfg.min_dominant_confidence == 55


def test_load_config_rejects_unknown(tmp_path):
    from config_loader import load_run_config

    p = tmp_path / "bad.yaml"
    p.write_text("mtf:\n  disabled_vars: [not_a_real_var]\n")
    with pytest.raises(ValueError, match="not_a_real_var"):
        load_run_config(str(p))

    p.write_text("overrides:\n  bogus.field: 1\n")
    with pytest.raises(ValueError, match="bogus"):
        load_run_config(str(p))


# --------------------------------------------------------------------------- #
# Feature-impact report (unit level; the full pipeline run is exercised by    #
# the CLI which is too slow for CI)                                           #
# --------------------------------------------------------------------------- #
def _fake_side(name, sharpe, win, pnl, dd, wf_sharpe, nprof):
    return {
        "config_name": name, "disabled_vars": [],
        "backtest": {"sharpe": sharpe, "win_rate": win,
                     "mean_pnl_per_trade": pnl, "total_pnl": pnl * 10,
                     "max_drawdown": dd, "gate_pass_rate": 0.5,
                     "gate_edge": 0.1, "ev_accuracy": 0.2, "trade_ticks": 50},
        "walk_forward": {"mean_sharpe": wf_sharpe, "mean_win_rate": win,
                         "n_profitable": nprof, "n_folds": 3},
        "per_regime": {"long": {"taken": {"n": 5, "mean_pnl": pnl,
                                          "win_rate": win},
                                "blocked": {"n": 2, "mean_pnl": -0.1,
                                            "win_rate": 0.0}}},
    }


def test_feature_impact_recommend_and_markdown(tmp_path):
    import sys
    sys.path.insert(0, "scripts")
    from feature_impact import (_deltas, _recommend, log_feature_impact,
                                render_markdown)

    base = _fake_side("baseline", 1.0, 0.50, 0.10, 0.30, 1.0, 1)
    var = _fake_side("variant", 1.5, 0.60, 0.20, 0.20, 1.4, 3)
    deltas = _deltas(base, var)
    assert deltas["sharpe"] == 0.5
    assert deltas["max_drawdown"] == pytest.approx(-0.1)

    tier, reasons = _recommend(deltas)
    assert tier == "Strong Positive"           # every scored metric improved
    assert reasons

    # degraded variant -> Negative
    worse = _fake_side("variant", 0.2, 0.30, -0.10, 0.60, 0.1, 0)
    tier2, _ = _recommend(_deltas(base, worse))
    assert tier2 == "Negative"

    metrics = {
        "feature": "test_feature", "report_date": "2026-07-08",
        "generated_at": "2026-07-08T00:00:00+00:00",
        "data_source": "synthetic (test)",
        "baseline": base, "variant": var, "deltas": deltas,
        "recommendation": tier, "recommendation_reasons": reasons,
    }
    md = render_markdown("test_feature", metrics)
    assert "# Feature Impact Report" in md
    assert "Strong Positive" in md
    assert "| sharpe |" in md
    assert "Per-regime impact" in md

    # journal logging lands in validation_reports as feature_impact
    db = str(tmp_path / "j.db")
    rid = log_feature_impact(db, "test_feature", metrics, notes="keep")
    jrn = Journal(db)
    rows = jrn.fetch_validation_reports(report_type="feature_impact")
    jrn.close()
    assert rows[0]["id"] == rid
    assert rows[0]["metrics"]["recommendation"] == "Strong Positive"
    assert rows[0]["notes"] == "keep"
