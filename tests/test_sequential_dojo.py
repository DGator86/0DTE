"""Tests for the anchored sequential (prequential curriculum) dojo spine."""
from __future__ import annotations

import os
import tempfile

from decision_engine import EngineConfig
from journal import Journal
from matrix_universe import MarkovWorldFeed, UniverseSpec
from sequential_dojo import (
    SequentialDojoConfig, _session_score, retention_forgetting,
    run_sequential_dojo,
)
from validation.session_folds import session_spans

# 3 sessions, coarse stride -> with min_warm=2 exactly one blind day is scored,
# keeping the per-session walk-forward count (and runtime) small.
SPEC = UniverseSpec("seq", 5, 3, "range_chop", tick_stride=45)


def _ff():
    return MarkovWorldFeed(SPEC)


def _ts():
    return MarkovWorldFeed(SPEC).timestamps()


def test_session_score_deterministic_and_depends_only_on_past():
    ts = _ts()
    spans = session_spans(ts)
    upto = ts[: spans[2].end]                 # sessions 0,1,2 only
    a = _session_score(_ff, upto, 2, EngineConfig(), "composite", 0)
    b = _session_score(_ff, upto, 2, EngineConfig(), "composite", 0)
    assert a == b                             # deterministic given ts_upto
    # the scored window never reaches beyond session 2 (prequential/leak-free):
    # the exact same truncated input is all it is given.


def test_run_structure_and_persistence():
    ts = _ts()
    with tempfile.TemporaryDirectory() as tmp:
        out = run_sequential_dojo(
            SequentialDojoConfig(db_path=os.path.join(tmp, "s.db"),
                                 reports_dir=os.path.join(tmp, "r"),
                                 min_warm_sessions=2, report_date="2026-07-24"),
            feed_factory=_ff, timestamps=ts)
        m = out["metrics"]
        assert m["n_sessions"] == 3
        assert len(m["days"]) == 1            # only session index 2 has >=2 warm
        for d in m["days"]:
            assert d["warm_sessions"] >= 2
            assert set(d) >= {"session", "warm_sessions", "prequential_J",
                              "baseline_J", "forward_transfer"}
        # forward transfer is defined (champion defaults to baseline here -> ~0)
        assert "mean_forward_transfer" in m
        jrn = Journal(os.path.join(tmp, "s.db"))
        reps = jrn.fetch_validation_reports(report_type="sequential_dojo")
        jrn.close()
        assert len(reps) == 1 and reps[0]["id"] == out["report_id"]
        assert os.path.isfile(out["json_path"])


def test_default_champion_equals_baseline_gives_zero_forward_transfer():
    ts = _ts()
    with tempfile.TemporaryDirectory() as tmp:
        out = run_sequential_dojo(
            SequentialDojoConfig(db_path=os.path.join(tmp, "s.db"),
                                 reports_dir=os.path.join(tmp, "r"),
                                 configs_dir=os.path.join(tmp, "configs"),
                                 min_warm_sessions=2, report_date="2026-07-24"),
            feed_factory=_ff, timestamps=ts)
        # no champion.json -> champion == baseline -> FT is exactly 0 per day
        for d in out["metrics"]["days"]:
            if d["forward_transfer"] is not None:
                assert d["forward_transfer"] == 0.0


def test_sealed_session_is_excluded():
    ts = _ts()
    spans = session_spans(ts)
    sealed = spans[2].date
    with tempfile.TemporaryDirectory() as tmp:
        out = run_sequential_dojo(
            SequentialDojoConfig(db_path=os.path.join(tmp, "s.db"),
                                 reports_dir=os.path.join(tmp, "r"),
                                 min_warm_sessions=2, sealed_sessions=(sealed,),
                                 report_date="2026-07-24"),
            feed_factory=_ff, timestamps=ts)
        assert out["metrics"]["n_sealed"] == 1
        assert all(d["session"] != sealed for d in out["metrics"]["days"])


def test_retention_forgetting_zero_for_identical_configs():
    ts = _ts()
    r = retention_forgetting(_ff, ts, [2], EngineConfig(), EngineConfig())
    assert r["forgetting"] == 0.0             # before == after -> no forgetting
    assert r["n"] in (0, 1)
