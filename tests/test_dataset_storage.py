"""
tests/test_dataset_storage.py
=============================
PR 3 acceptance — canonical dataset storage and audit linkage:
  * rebuilding from identical recordings produces IDENTICAL dataset hashes;
  * chain_store.replay_ticks() serves as-of-filtered bars (no future bar can
    reach an earlier observation, even from a malformed recording);
  * journal rows carry snapshot_id (with legacy-DB migration);
  * UnifiedOrchestrator writes feature_snapshots keyed by the SAME
    snapshot_id that lands on the journal row;
  * Parquet export materializes the partitioned columnar view.
"""
from __future__ import annotations

import datetime as dt
import json
import sqlite3
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from chain_store import ChainRecorder, RecordedFeed
from gate_scorer import MarketSnapshot
from journal import COLUMNS, Journal, _coltype
from prediction.dataset import build_dataset_from_recording
from prediction.storage import PredictionStore
from resample import RawBars
from unified_loop import SyntheticUnifiedFeed, TickSnapshot, UnifiedOrchestrator

ET = ZoneInfo("America/New_York")


def _market(spot: float, now: dt.datetime) -> MarketSnapshot:
    return MarketSnapshot(
        spot=spot, net_gex=4.0e9, gamma_flip=spot - 6.0,
        call_wall=spot + 5.0, put_wall=spot - 5.0, gex_pct_rank=0.8,
        vix9d=12.0, vix=13.0, vix3m=15.0, vvix=90.0, vvix_baseline=95.0,
        straddle_breakeven=4.0, expected_range=3.2,
        adx=13.0, rsi=51.0, bb_width=1.4, bb_width_baseline=2.0,
        vwap=spot, vwap_reversion_count=3,
        tick_abs_mean=450.0, cvd_slope=0.01,
        now=now, has_catalyst=False,
    )


def _bars(n: int, start: dt.datetime) -> RawBars:
    ts = (np.datetime64(start.astimezone(dt.timezone.utc).replace(tzinfo=None))
          + np.arange(n) * np.timedelta64(1, "m"))
    close = 600.0 + 0.02 * np.arange(n)
    return RawBars(ts=ts.astype("datetime64[ns]"), open=close,
                   high=close + 0.05, low=close - 0.05, close=close,
                   volume=np.ones(n))


def _record_session(directory: str, n_ticks: int = 30) -> None:
    rec = ChainRecorder(directory)
    start = dt.datetime(2026, 7, 6, 9, 30, tzinfo=ET)
    for i in range(n_ticks):
        now = start + dt.timedelta(minutes=i)
        snap = TickSnapshot(market=_market(600.0 + 0.02 * i, now),
                            bars=_bars(i + 1, start), chain=None)
        rec.record(now, snap)
    rec.record_settlement("2026-07-06", 600.0 + 0.02 * (n_ticks - 1))


# --------------------------------------------------------------------------- #
# Deterministic rebuild                                                        #
# --------------------------------------------------------------------------- #
class TestDeterministicRebuild:
    def test_identical_recordings_identical_hashes(self, tmp_path):
        _record_session(str(tmp_path / "ticks"))
        h = []
        for name in ("a.sqlite", "b.sqlite"):
            store = PredictionStore(str(tmp_path / name))
            stats = build_dataset_from_recording(str(tmp_path / "ticks"), store)
            assert stats["observations"] == 30
            assert stats["labeled"] == 30
            assert stats["sessions"] == ["2026-07-06"]
            h.append(store.dataset_hash())
            store.close()
        assert h[0] == h[1]

    def test_rebuild_is_idempotent(self, tmp_path):
        _record_session(str(tmp_path / "ticks"))
        store = PredictionStore(str(tmp_path / "s.sqlite"))
        build_dataset_from_recording(str(tmp_path / "ticks"), store)
        h1 = store.dataset_hash()
        build_dataset_from_recording(str(tmp_path / "ticks"), store)
        assert store.dataset_hash() == h1                # REPLACE, not duplicate
        assert len(store.fetch_feature_snapshots()) == 30
        store.close()

    def test_labels_present_and_sane(self, tmp_path):
        _record_session(str(tmp_path / "ticks"))
        store = PredictionStore(str(tmp_path / "s.sqlite"))
        build_dataset_from_recording(str(tmp_path / "ticks"), store)
        snaps = store.fetch_feature_snapshots()
        labels = {r["snapshot_id"]: r["labels"] for r in store.fetch_labels()}
        first = labels[snaps[0]["snapshot_id"]]
        # steady uptrend: the 15m forward return from the first tick is > 0
        assert first["fwd_return_15m"] is not None and first["fwd_return_15m"] > 0
        assert first["up_15m"] == 1
        # 60m horizon extends past this 30-minute recording -> None
        assert first["fwd_return_60m"] is None
        # last tick has no future -> everything None
        last = labels[snaps[-1]["snapshot_id"]]
        assert last["fwd_return_5m"] is None
        store.close()


# --------------------------------------------------------------------------- #
# replay_ticks as-of filtering                                                 #
# --------------------------------------------------------------------------- #
class TestReplayTicks:
    def test_future_bars_filtered_from_malformed_recording(self, tmp_path):
        # hand-write a recording whose first tick carries bars from the FUTURE
        d = tmp_path / "ticks"
        d.mkdir()
        rec = ChainRecorder(str(d))
        start = dt.datetime(2026, 7, 6, 9, 30, tzinfo=ET)
        snap = TickSnapshot(market=_market(600.0, start),
                            bars=_bars(120, start), chain=None)  # 2h of bars!
        rec.record(start, snap)

        feed = RecordedFeed(str(d))
        ticks = list(feed.replay_ticks())
        assert len(ticks) == 1
        seq, ts, tick = ticks[0]
        assert seq == 0
        # only the single bar ending at/before 9:30 ET survives
        assert len(tick.bars.ts) == 1

    def test_seq_and_snapshots_roundtrip(self, tmp_path):
        _record_session(str(tmp_path / "ticks"), n_ticks=5)
        feed = RecordedFeed(str(tmp_path / "ticks"))
        seqs = [s for s, _, _ in feed.replay_ticks()]
        assert seqs == [0, 1, 2, 3, 4]
        # replay_ticks does not consume the serving iterator
        assert feed.snapshot(dt.datetime(2026, 7, 6, 9, 30, tzinfo=ET)) is not None


# --------------------------------------------------------------------------- #
# Journal linkage                                                              #
# --------------------------------------------------------------------------- #
class TestJournalSnapshotId:
    def _row(self, **kw):
        row = {c: None for c in COLUMNS}
        row.update(session_date="2026-07-06", ts="2026-07-06T10:00:00-04:00",
                   spot=600.0, decision="NO_TRADE", was_traded=0,
                   candidate_present=0, gate_pass=0, gate_score=0.0)
        row.update(kw)
        return row

    def test_snapshot_id_persisted(self, tmp_path):
        j = Journal(str(tmp_path / "j.sqlite"))
        j.log(self._row(snapshot_id="abc123"))
        j.log(self._row())                       # legacy caller: no snapshot_id
        rows = j.fetch()
        assert rows[0]["snapshot_id"] == "abc123"
        assert rows[1]["snapshot_id"] is None
        j.close()

    def test_legacy_db_migrates(self, tmp_path):
        db = str(tmp_path / "legacy.sqlite")
        legacy_cols = [c for c in COLUMNS]       # pre-PR3: no snapshot_id
        conn = sqlite3.connect(db)
        conn.execute(
            "CREATE TABLE evaluations (id INTEGER PRIMARY KEY AUTOINCREMENT, "
            + ", ".join(f"{c} {_coltype(c)}" for c in legacy_cols)
            + ", settle_price REAL, realized_pnl REAL, ev_error REAL, "
              "settled INTEGER NOT NULL DEFAULT 0)")
        conn.commit()
        conn.close()

        j = Journal(db)                          # must migrate, not crash
        j.log(self._row(snapshot_id="xyz"))
        assert j.fetch()[0]["snapshot_id"] == "xyz"
        j.close()


# --------------------------------------------------------------------------- #
# Live-loop capture                                                            #
# --------------------------------------------------------------------------- #
class TestOrchestratorCapture:
    def test_feature_snapshots_linked_to_journal(self, tmp_path):
        store = PredictionStore(str(tmp_path / "pred.sqlite"))
        jrn = Journal(str(tmp_path / "j.sqlite"))
        orch = UnifiedOrchestrator(feed=SyntheticUnifiedFeed(days=2),
                                   journal=jrn, prediction_store=store)
        start = dt.datetime(2026, 6, 26, 9, 30, tzinfo=ET)
        ticks = [start + dt.timedelta(minutes=i) for i in range(15)]
        results = orch.run_replay(ticks)
        assert len(results) == 15

        snaps = store.fetch_feature_snapshots()
        assert len(snaps) == 15
        journal_ids = [r["snapshot_id"] for r in jrn.fetch()]
        assert all(jid is not None for jid in journal_ids)
        assert set(journal_ids) == {s["snapshot_id"] for s in snaps}

        # raw + standardized + quality captured
        s0 = snaps[0]
        assert s0["symbol"] == "SPY"
        assert "gamma_sign" in s0["features"]
        assert s0["quality"]["has_chain"] is False
        assert 0.0 <= s0["quality"]["feature_coverage"] <= 1.0
        store.close()
        jrn.close()

    def test_no_store_no_snapshot_rows_but_ids_still_journaled(self, tmp_path):
        jrn = Journal(str(tmp_path / "j.sqlite"))
        orch = UnifiedOrchestrator(feed=SyntheticUnifiedFeed(days=2),
                                   journal=jrn)
        start = dt.datetime(2026, 6, 26, 9, 30, tzinfo=ET)
        orch.run_replay([start + dt.timedelta(minutes=i) for i in range(3)])
        ids = [r["snapshot_id"] for r in jrn.fetch()]
        assert len(ids) == 3 and all(i is not None for i in ids)
        assert len(set(ids)) == 3
        jrn.close()

    def test_replay_reproducible_ids(self, tmp_path):
        def run():
            jrn = Journal(":memory:")
            orch = UnifiedOrchestrator(feed=SyntheticUnifiedFeed(days=2),
                                       journal=jrn)
            start = dt.datetime(2026, 6, 26, 9, 30, tzinfo=ET)
            orch.run_replay([start + dt.timedelta(minutes=i) for i in range(5)])
            out = [r["snapshot_id"] for r in jrn.fetch()]
            jrn.close()
            return out

        assert run() == run()


# --------------------------------------------------------------------------- #
# Parquet export                                                               #
# --------------------------------------------------------------------------- #
class TestParquetExport:
    def test_partitioned_export(self, tmp_path):
        pytest.importorskip("pyarrow")
        import pandas as pd

        _record_session(str(tmp_path / "ticks"), n_ticks=10)
        store = PredictionStore(str(tmp_path / "s.sqlite"))
        build_dataset_from_recording(str(tmp_path / "ticks"), store)
        paths = store.export_features_parquet(str(tmp_path / "derived"))
        assert len(paths) == 1
        assert "session_date=2026-07-06" in paths[0]
        df = pd.read_parquet(paths[0])
        assert len(df) == 10
        assert "snapshot_id" in df.columns
        assert any(c.startswith("feat_") for c in df.columns)
        store.close()
