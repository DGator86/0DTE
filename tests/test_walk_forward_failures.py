"""
Walk-forward failure accounting (walk_forward.py, PR 1 of Prediction Engine
V2): tick exceptions are recorded as structured TickFailure records instead
of being silently swallowed, and a fold whose test failure fraction exceeds
the configured threshold is marked invalid and excluded from aggregates.
"""
import datetime as dt

import pytest

from synthetic_world import CoupledSyntheticFeed, WorldConfig
from validation.session_folds import make_session_folds
from walk_forward import WalkForwardConfig, run_walk_forward


class FailingFeed:
    """Wraps a feed and raises on snapshot() for a chosen set of ticks."""

    def __init__(self, inner, fail_at: set[dt.datetime]):
        self._inner = inner
        self._fail_at = fail_at

    def timestamps(self):
        return self._inner.timestamps()

    def snapshot(self, t):
        if t in self._fail_at:
            raise RuntimeError(f"injected feed failure at {t}")
        return self._inner.snapshot(t)

    def settlement_price(self, session_date):
        return self._inner.settlement_price(session_date)


def _world():
    return CoupledSyntheticFeed(WorldConfig(days=4, seed=9, tick_stride=30))


def _fold_windows(ticks, cfg):
    return make_session_folds(
        ticks, mode=cfg.mode, n_folds=cfg.n_folds, train_frac=cfg.train_frac,
        embargo_sessions=cfg.embargo_sessions)


def test_injected_failure_appears_in_report():
    ticks = _world().timestamps()
    cfg = WalkForwardConfig(mode="expanding", n_folds=1, train_frac=0.5,
                            max_failed_tick_frac=0.5)  # tolerant: stays valid
    fold = _fold_windows(ticks, cfg)[0]
    warm_victim = ticks[fold.warm_start]
    test_victim = ticks[fold.test_start]

    result = run_walk_forward(
        feed_factory=lambda: FailingFeed(_world(), {warm_victim, test_victim}),
        timestamps=ticks, wf_cfg=cfg,
    )

    fr = result.folds[0]
    assert fr.n_failed_warm == 1
    assert fr.n_failed_test == 1
    stages = {f.stage for f in fr.failures}
    assert stages == {"warm", "test"}
    for f in fr.failures:
        assert f.exception_type == "RuntimeError"
        assert "injected feed failure" in f.message
        assert f.session_date and f.traceback_hash
    # a tolerated failure never disappears: it is visible in every report
    assert fr.valid
    summary = result.failure_summary()
    assert summary["n_failed_warm"] == 1
    assert summary["n_failed_test"] == 1
    assert summary["by_exception"] == {"RuntimeError": 2}
    d = result.to_dict()
    assert d["failures"]["n_failed_test"] == 1
    assert d["folds"][0]["n_failed_test"] == 1
    assert d["folds"][0]["valid"] is True


def test_excessive_failures_invalidate_fold():
    ticks = _world().timestamps()
    cfg = WalkForwardConfig(mode="expanding", n_folds=1, train_frac=0.5)
    fold = _fold_windows(ticks, cfg)[0]
    # kill an entire test session's ticks — way past the 1% default
    victims = set(ticks[fold.test_start:fold.test_end])

    result = run_walk_forward(
        feed_factory=lambda: FailingFeed(_world(), victims),
        timestamps=ticks, wf_cfg=cfg,
    )

    fr = result.folds[0]
    assert not fr.valid
    assert fr.invalid_reason and "test ticks failed" in fr.invalid_reason
    # invalid folds are excluded from aggregates but still listed
    assert result.valid_folds == []
    assert result.n_profitable() == 0
    assert result.to_dict()["mean_pnl"] is None
    assert result.to_dict()["n_valid_folds"] == 0
    assert len(result.to_dict()["folds"]) == 1
    assert result.failure_summary()["n_invalid_folds"] == 1


def test_clean_run_has_no_failures_and_session_stats():
    ticks = _world().timestamps()
    result = run_walk_forward(
        feed_factory=_world,
        timestamps=ticks,
        wf_cfg=WalkForwardConfig(mode="expanding", n_folds=2, train_frac=0.5),
    )
    assert all(f.valid for f in result.folds)
    assert result.failure_summary()["n_failed_test"] == 0

    # session invariants: no session on both sides; sessions counted honestly
    for fr in result.folds:
        assert not (set(fr.warm_sessions) & set(fr.test_sessions))
        assert fr.n_test_sessions == len(fr.test_sessions) > 0
    d = result.to_dict()
    assert d["fold_unit"] == "session"
    assert d["n_test_sessions"] == result.n_test_sessions() > 0
    boot = d["session_pnl_bootstrap"]
    assert boot["n_sessions"] == result.n_test_sessions()
