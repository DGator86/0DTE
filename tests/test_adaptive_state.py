"""Adaptive state must survive restarts: GEX rank window + ScaleBooks."""
from __future__ import annotations

import os

import pytest

from gex_window import GexRankWindow
from regime_classifier import ScaleBook


# --------------------------------------------------------------------------- #
# GexRankWindow                                                                #
# --------------------------------------------------------------------------- #
def test_neutral_until_min_samples():
    w = GexRankWindow(min_samples=10)
    for i in range(9):
        assert w.rank(1e9 + i, now_epoch=1000.0 + i) == 0.5


def test_ranks_by_magnitude_not_sign():
    w = GexRankWindow(min_samples=5)
    t = 0.0
    for g in [1e9, 2e9, 3e9, 4e9, 5e9]:
        w.rank(g, now_epoch=(t := t + 60))
    # -6e9 is the largest MAGNITUDE seen: rank must be high, not 0
    # (population includes the current print: 5 of 6 below it)
    assert w.rank(-6e9, now_epoch=t + 60) == pytest.approx(5 / 6)


def test_declining_gex_no_longer_pins_at_zero_with_history(tmp_path):
    # a big |GEX| print ranks high against multi-day history even if it is
    # the lowest of the last hour (the old signed 100-tick window pinned at 0)
    w = GexRankWindow(min_samples=5)
    t = 0.0
    for g in [1e9, 2e9, 3e9, 4e9, 9e9, 7e9]:
        w.rank(g, now_epoch=(t := t + 60))
    assert w.rank(6e9, now_epoch=t + 60) > 0.5


def test_persists_across_instances(tmp_path):
    path = os.path.join(tmp_path, "gex.json")
    w1 = GexRankWindow(path=path, min_samples=3)
    t = 0.0
    for g in [1e9, 2e9, 3e9, 4e9]:
        w1.rank(g, now_epoch=(t := t + 60))
    # "restart": a new instance sees the old history immediately
    w2 = GexRankWindow(path=path, min_samples=3)
    assert len(w2) == 4
    assert w2.rank(5e9, now_epoch=t + 60) == pytest.approx(4 / 5)


def test_prunes_old_entries():
    w = GexRankWindow(min_samples=1, max_age_days=1.0)
    w.rank(1e9, now_epoch=0.0)
    w.rank(2e9, now_epoch=60.0)
    w.rank(3e9, now_epoch=3 * 86400.0)     # 3 days later: first two age out
    assert len(w) == 1


def test_corrupt_state_rewarns_instead_of_crashing(tmp_path):
    path = os.path.join(tmp_path, "gex.json")
    with open(path, "w") as f:
        f.write("{not json")
    w = GexRankWindow(path=path)
    assert len(w) == 0
    assert w.rank(1e9, now_epoch=1.0) == 0.5


# --------------------------------------------------------------------------- #
# ScaleBook round-trip                                                         #
# --------------------------------------------------------------------------- #
def test_scalebook_roundtrip():
    sb = ScaleBook()
    for x in [1.0, 2.0, 3.0, 4.0, 5.0]:
        sb.update("ema_slope", x)
    data = sb.to_dict()

    sb2 = ScaleBook()
    sb2.load_dict(data)
    assert sb2.std("ema_slope", 99.0) == pytest.approx(sb.std("ema_slope", 99.0))
    assert sb2.reliability("ema_slope") == pytest.approx(sb.reliability("ema_slope"))


def test_scalebook_load_corrupt_resets():
    sb = ScaleBook()
    sb.load_dict({"bad": "not-a-triple"})
    assert sb.std("bad", 7.0) == 7.0      # falls back to prior


# --------------------------------------------------------------------------- #
# UnifiedOrchestrator state persistence                                        #
# --------------------------------------------------------------------------- #
def test_orchestrator_saves_and_reloads_state(tmp_path):
    import datetime as dt
    from zoneinfo import ZoneInfo
    from unified_loop import UnifiedOrchestrator, SyntheticUnifiedFeed

    ET = ZoneInfo("America/New_York")
    path = os.path.join(tmp_path, "adaptive_state.json")

    feed = SyntheticUnifiedFeed(days=3)
    orch = UnifiedOrchestrator(feed=feed, state_path=path)
    start = dt.datetime(2026, 6, 26, 9, 30, tzinfo=ET)
    orch.run_replay([start + dt.timedelta(minutes=i) for i in range(30)])
    orch._save_state()
    assert os.path.isfile(path)

    feed2 = SyntheticUnifiedFeed(days=3)
    orch2 = UnifiedOrchestrator(feed=feed2, state_path=path)
    # the reloaded matrix scale book must carry the learned samples
    stats = orch2._matrix_scale_book.to_dict()
    assert stats, "reloaded scale book is empty"
    assert any(v[0] > 0 for v in stats.values())
