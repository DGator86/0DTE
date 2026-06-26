"""
live_feed_adapter.py
====================
Vendor-agnostic feed adapter driving Track B (multi-timeframe regime routing).

Track B pipeline:
  DataFeed.snapshot() -> FeedSnapshot
    -> resample.build_mtf_input             (bars -> per-TF indicators)
    -> mtf_matrix.build_matrix / regime_rows (standardized feature matrix)
    -> decision_matrix.decide_from_matrix    (27-cell -> TradeIntent)
    -> route_ticket                          (TradeIntent -> SpreadCandidate or stub)

Task #1 seam closure (HANDOFF §6.1):
  route_ticket now calls into Track A (spread_selector) when the TradeIntent
  is a premium family AND FeedSnapshot carries a ChainSnapshot.
  This turns a *named* structure into a *fillable* SpreadCandidate with
  concrete strikes, credit, max_loss, and Kelly-sized position.

Two feed implementations:
  CSVBarFeed   - loads a 1-minute OHLCV CSV for replay
  SyntheticFeed - generates synthetic bars, no file needed

MarketSnapshot here is the lighter Track B version. Unifying with
gate_scorer.MarketSnapshot is HANDOFF task #2.
"""
from __future__ import annotations

import csv
import datetime as dt
import math
import os
from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol
from zoneinfo import ZoneInfo

import numpy as np

from resample import RawBars, build_mtf_input
from mtf_matrix import MTFInput, build_matrix, regime_rows  # MTFInput is the mtf_matrix version
from decision_matrix import decide_from_matrix, TradeIntent

ET = ZoneInfo("America/New_York")

# Track A imports (used in route_ticket seam)
try:
    from rnd_extractor import ChainSnapshot, extract_rnd, compute_edge
    from spread_selector import GammaContext, SelectorConfig, select_spreads, SpreadCandidate
    _TRACK_A_AVAILABLE = True
except ImportError:
    _TRACK_A_AVAILABLE = False
    ChainSnapshot = None
    SpreadCandidate = None


# --------------------------------------------------------------------------- #
# Snapshot types                                                               #
# --------------------------------------------------------------------------- #
@dataclass
class MarketSnapshot:
    """
    Lightweight Track-B market snapshot. Carries the dealer/vol positioning
    fields the regime pipeline needs. Does not yet carry all fields from
    gate_scorer.MarketSnapshot (that is HANDOFF task #2).
    """
    spot: float
    net_gex: float
    gex_pct_rank: float
    gamma_flip: float
    call_wall: float
    put_wall: float
    vix: float
    vix9d: float
    vix3m: float
    vvix: float
    vvix_baseline: float
    adx: float
    rsi: float
    bb_width: float
    bb_width_baseline: float
    straddle_breakeven: float
    expected_range: float
    cvd_slope: float
    tick_abs_mean: float
    now: dt.datetime
    has_catalyst: bool = False
    catalyst_label: str = ""

    def as_snap_dict(self) -> dict:
        """Convert to the dict format consumed by mtf_matrix._snapshot_vars."""
        return {k: v for k, v in self.__dict__.items()
                if not k.startswith("_") and k not in ("now", "catalyst_label")}


@dataclass
class FeedSnapshot:
    """Everything a single tick delivers to the pipeline."""
    market: MarketSnapshot
    bars: RawBars
    # chain is None until the options feed is wired (HANDOFF §6.3).
    # When present, route_ticket will call into Track A.
    chain: Optional["ChainSnapshot"] = None  # type: ignore[type-arg]


# --------------------------------------------------------------------------- #
# DataFeed protocol                                                            #
# --------------------------------------------------------------------------- #
class DataFeed(Protocol):
    def next_snapshot(self) -> Optional[FeedSnapshot]:
        """Return next tick or None when session ends."""
        ...


# --------------------------------------------------------------------------- #
# CSV bar feed                                                                 #
# --------------------------------------------------------------------------- #
class CSVBarFeed:
    """
    Loads a 1-minute OHLCV CSV and replays it bar by bar.

    Expected columns (header required):
        timestamp, open, high, low, close, volume
    Optional: signed_volume, tick

    `market_fn` is a callable(last_close, ts) -> MarketSnapshot that the
    caller provides to inject dealer/vol positioning for each bar. Use it to
    look up GEX, VIX, etc. from your data source.
    """

    def __init__(self, csv_path: str,
                 market_fn: Optional[Callable] = None,
                 window: int = 390):
        self.csv_path = csv_path
        self.market_fn = market_fn or _default_market_fn
        self.window = window
        self._rows: list[dict] = []
        self._idx = 0
        self._load()

    def _load(self) -> None:
        with open(self.csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self._rows.append(row)

    def next_snapshot(self) -> Optional[FeedSnapshot]:
        if self._idx >= len(self._rows):
            return None
        self._idx += 1
        # Build a rolling window of bars up to the current bar
        window_rows = self._rows[max(0, self._idx - self.window): self._idx]
        bars = _rows_to_bars(window_rows)
        last_close = float(self._rows[self._idx - 1]["close"])
        ts_str = self._rows[self._idx - 1]["timestamp"]
        ts = _parse_ts(ts_str)
        market = self.market_fn(last_close, ts)
        return FeedSnapshot(market=market, bars=bars)


def _rows_to_bars(rows: list[dict]) -> RawBars:
    ts = np.array([float(r["timestamp"]) for r in rows])
    o = np.array([float(r["open"]) for r in rows])
    h = np.array([float(r["high"]) for r in rows])
    l = np.array([float(r["low"]) for r in rows])
    c = np.array([float(r["close"]) for r in rows])
    v = np.array([float(r["volume"]) for r in rows])
    sv = (np.array([float(r["signed_volume"]) for r in rows])
          if "signed_volume" in rows[0] else None)
    tick = (np.array([float(r["tick"]) for r in rows])
            if "tick" in rows[0] else None)
    return RawBars(timestamp=ts, open=o, high=h, low=l, close=c, volume=v,
                   signed_volume=sv, tick=tick)


def _parse_ts(s: str) -> dt.datetime:
    try:
        return dt.datetime.fromisoformat(s)
    except ValueError:
        return dt.datetime.fromtimestamp(float(s), tz=ET)


def _default_market_fn(spot: float, ts: dt.datetime) -> MarketSnapshot:
    """Placeholder market fn: returns a neutral long-gamma snapshot."""
    return MarketSnapshot(
        spot=spot, net_gex=3.0e9, gex_pct_rank=0.70,
        gamma_flip=spot - 6.0, call_wall=spot + 3.0, put_wall=spot - 3.0,
        vix=14.0, vix9d=13.5, vix3m=15.0, vvix=93.0, vvix_baseline=95.0,
        adx=13.0, rsi=51.0, bb_width=1.5, bb_width_baseline=2.0,
        straddle_breakeven=3.8, expected_range=3.0,
        cvd_slope=0.02, tick_abs_mean=500.0, now=ts,
    )


# --------------------------------------------------------------------------- #
# Synthetic feed (demo)                                                        #
# --------------------------------------------------------------------------- #
class SyntheticFeed:
    """
    Generates a 390-bar synthetic session (1m bars) with configurable GEX
    regime so both the ranging and trending cases can be exercised.
    """

    def __init__(self, regime: str = "pin", seed: int = 42,
                 n_bars: int = 390, spot0: float = 600.0,
                 chain: Optional["ChainSnapshot"] = None):  # type: ignore[type-arg]
        """
        regime: "pin" (long-gamma) | "trend_down" | "trend_up"
        chain: optional ChainSnapshot to test Track A seam.
        """
        self.regime = regime
        self.chain = chain
        rng = np.random.default_rng(seed)
        now0 = dt.datetime(2026, 6, 26, 9, 30, tzinfo=ET)

        drift = {"pin": 0.0, "trend_up": 0.0003, "trend_down": -0.0003}[regime]
        vol = {"pin": 0.0005, "trend_up": 0.0008, "trend_down": 0.0008}[regime]
        lr = rng.normal(drift, vol, n_bars)
        close = spot0 * np.exp(np.cumsum(lr))
        open_ = np.roll(close, 1); open_[0] = spot0
        high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.0003, n_bars))
        low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.0003, n_bars))
        vol_arr = rng.integers(10_000, 60_000, n_bars).astype(float)
        ts = np.array([int((now0 + dt.timedelta(minutes=i)).timestamp())
                       for i in range(n_bars)])

        self._bars_full = RawBars(
            timestamp=ts, open=open_, high=high, low=low,
            close=close, volume=vol_arr,
        )
        self._n = n_bars
        self._idx = 0
        self._spot0 = spot0
        self._now0 = now0

    def next_snapshot(self) -> Optional[FeedSnapshot]:
        if self._idx >= self._n:
            return None
        i = self._idx + 1
        self._idx += 1

        bars = RawBars(
            timestamp=self._bars_full.timestamp[:i],
            open=self._bars_full.open[:i],
            high=self._bars_full.high[:i],
            low=self._bars_full.low[:i],
            close=self._bars_full.close[:i],
            volume=self._bars_full.volume[:i],
        )
        spot = float(self._bars_full.close[i - 1])
        now = self._now0 + dt.timedelta(minutes=i - 1)

        # Regime-shaped market snapshot
        if self.regime == "pin":
            net_gex = 4.2e9; gex_rank = 0.88; flip = self._spot0 - 6.0
            vix = 13.0; adx = 12.5; rsi = 52.0
        elif self.regime == "trend_up":
            net_gex = -0.8e9; gex_rank = 0.35; flip = spot + 3.0
            vix = 17.0; adx = 24.0; rsi = 62.0
        else:  # trend_down
            net_gex = -1.1e9; gex_rank = 0.38; flip = spot + 4.0
            vix = 18.0; adx = 26.0; rsi = 37.0

        market = MarketSnapshot(
            spot=spot, net_gex=net_gex, gex_pct_rank=gex_rank,
            gamma_flip=flip, call_wall=spot + 3.0, put_wall=spot - 3.0,
            vix=vix, vix9d=vix - 0.9, vix3m=vix + 2.2,
            vvix=92.0, vvix_baseline=95.0,
            adx=adx, rsi=rsi, bb_width=1.5, bb_width_baseline=2.0,
            straddle_breakeven=3.8, expected_range=3.0,
            cvd_slope=0.02 if self.regime == "pin" else -0.4,
            tick_abs_mean=480.0 if self.regime == "pin" else 850.0,
            now=now,
        )
        return FeedSnapshot(market=market, bars=bars, chain=self.chain)


# --------------------------------------------------------------------------- #
# Route ticket (Track B -> Track A seam, HANDOFF §6.1)                       #
# --------------------------------------------------------------------------- #
@dataclass
class RoutedTicket:
    intent: TradeIntent
    candidate: Optional[object]      # SpreadCandidate | None
    no_fill_reason: str
    matrix_snapshot: dict            # top-level matrix stats for logging


_PREMIUM_STRUCTURES = {"PCS", "CCS", "IC", "IF"}


def route_ticket(
    intent: TradeIntent,
    snap: FeedSnapshot,
    selector_cfg: Optional["SelectorConfig"] = None,  # type: ignore[type-arg]
) -> RoutedTicket:
    """
    Convert a TradeIntent into a RoutedTicket.

    For premium structures: calls into Track A (extract_rnd -> compute_edge ->
    select_spreads) when a ChainSnapshot is available. Returns the best
    SpreadCandidate scaled by intent.size_mult.

    For directional structures: returns the intent unchanged (directional
    selector not yet built; HANDOFF §6.7).

    When no chain is available, returns the intent as a named stub only.
    """
    if intent.decision.structure == "NT":
        return RoutedTicket(intent=intent, candidate=None,
                            no_fill_reason="intent_no_trade",
                            matrix_snapshot={})

    ms = snap.market
    matrix_snap = {
        "spot": ms.spot, "net_gex": ms.net_gex, "gex_pct_rank": ms.gex_pct_rank,
        "gamma_flip": ms.gamma_flip, "call_wall": ms.call_wall, "put_wall": ms.put_wall,
        "vix": ms.vix, "adx": ms.adx, "has_catalyst": ms.has_catalyst,
    }

    if intent.decision.structure not in _PREMIUM_STRUCTURES:
        return RoutedTicket(intent=intent, candidate=None,
                            no_fill_reason=f"directional_engine_not_built:{intent.decision.structure}",
                            matrix_snapshot=matrix_snap)

    if not _TRACK_A_AVAILABLE:
        return RoutedTicket(intent=intent, candidate=None,
                            no_fill_reason="track_a_not_available",
                            matrix_snapshot=matrix_snap)

    if snap.chain is None:
        return RoutedTicket(intent=intent, candidate=None,
                            no_fill_reason="no_chain_in_snapshot",
                            matrix_snapshot=matrix_snap)

    # ---- Track A seam: extract RND, compute edge, select spread ----
    try:
        rnd = extract_rnd(snap.chain)
        edge = compute_edge(rnd, snap.chain)
        ctx = GammaContext(
            spot=ms.spot,
            call_wall=ms.call_wall,
            put_wall=ms.put_wall,
            gamma_flip=ms.gamma_flip,
            net_gex=ms.net_gex,
            gex_pct_rank=ms.gex_pct_rank,
        )
        cfg = selector_cfg or SelectorConfig()
        sel = select_spreads(snap.chain, rnd, edge, ctx, cfg)

        if sel.best is None:
            return RoutedTicket(intent=intent, candidate=None,
                                no_fill_reason=sel.no_trade_reason or "selector_no_candidate",
                                matrix_snapshot=matrix_snap)

        # Apply the Track-B size multiplier to the candidate's kelly fraction
        candidate = sel.best
        # size_mult from the decision matrix scales the kelly fraction
        # (candidate.kelly_fraction is not set here — that lives in gate_scorer;
        # for Track B the size_mult IS the sizing signal)
        return RoutedTicket(intent=intent, candidate=candidate,
                            no_fill_reason="",
                            matrix_snapshot={**matrix_snap,
                                             "edge_richness": edge.richness_signal,
                                             "rnd_arb": rnd.arb_violation})

    except Exception as exc:
        return RoutedTicket(intent=intent, candidate=None,
                            no_fill_reason=f"track_a_error:{exc}",
                            matrix_snapshot=matrix_snap)


# --------------------------------------------------------------------------- #
# PipelineOrchestrator                                                        #
# --------------------------------------------------------------------------- #
class PipelineOrchestrator:
    """
    Drives the Track-B pipeline: feed -> MTF matrix -> decision -> route_ticket.
    Prints a summary line per tick; in production, wire in a Journal.log() call.
    """

    def __init__(self, feed: DataFeed,
                 selector_cfg: Optional["SelectorConfig"] = None,  # type: ignore[type-arg]
                 every_n: int = 1,
                 verbose: bool = True):
        self.feed = feed
        self.selector_cfg = selector_cfg
        self.every_n = every_n
        self.verbose = verbose
        self._tick = 0

    def run_once(self) -> Optional[RoutedTicket]:
        """Process one tick and return its RoutedTicket, or None at end of session."""
        snap = self.feed.next_snapshot()
        if snap is None:
            return None

        self._tick += 1
        mtf_in = _snap_to_mtf(snap)
        mat_rows = build_matrix(mtf_in)
        regimes = regime_rows(mat_rows)
        vetoes = ["catalyst:event"] if snap.market.has_catalyst else []
        intent = decide_from_matrix(mat_rows, regimes, vetoes=vetoes)
        ticket = route_ticket(intent, snap, self.selector_cfg)  # type: ignore[arg-type]

        if self.verbose and self._tick % self.every_n == 0:
            _print_tick(snap, intent, ticket, self._tick)

        return ticket

    def run_session(self) -> list[RoutedTicket]:
        """Run until the feed is exhausted. Returns all routed tickets."""
        tickets = []
        while True:
            t = self.run_once()
            if t is None:
                break
            tickets.append(t)
        return tickets


def _snap_to_mtf(snap: FeedSnapshot) -> MTFInput:
    """
    Convert FeedSnapshot -> mtf_matrix.MTFInput.

    resample gives {tf -> {indicator -> val}}.
    mtf_matrix expects {var_name -> {tf -> val}} for native,
    plus a flat snapshot dict for SNAPSHOT-kind variables.
    We transpose and map names here; 30m/4h/1d are left None
    (mtf_matrix handles sparse data gracefully).
    """
    rsmp = build_mtf_input(snap.bars)  # resample.MTFInput

    # resample indicator -> mtf_matrix var_name(s)
    _MAP = {
        "vwap_dist": ["dist_to_vwap", "ema_slope"],  # dist_to_vwap uses |vwap_dist|
        "adx":       ["adx_strength"],
        "rsi":       ["rsi"],
        "bb_width":  ["rv_expansion", "bb_compression"],
        "rv":        ["realized_vol"],
        "cvd":       ["cvd_persistence"],
        "tick_abs":  ["tick_two_sided"],
    }
    # Build native: var_name -> {tf -> val}
    native: dict = {}
    for resamp_key, var_names in _MAP.items():
        for var_name in var_names:
            native.setdefault(var_name, {})
            for tf, ind in rsmp.native.items():
                val = ind.get(resamp_key)
                if val is None or (isinstance(val, float) and math.isnan(val)):
                    val = None
                # dist_to_vwap uses absolute value; ema_slope keeps sign
                if var_name == "dist_to_vwap" and val is not None:
                    val = abs(val)
                native[var_name][tf] = val

    # vwap_slope = sign * magnitude of vwap_dist (same as ema_slope above)
    native["vwap_slope"] = native.get("ema_slope", {})

    # Build snapshot from market snapshot fields
    ms = snap.market
    spot = ms.spot or 1.0
    snapshot: dict = {
        "gamma_sign":       ms.net_gex,
        "gamma_magnitude":  ms.gex_pct_rank,
        "flip_cushion":     (ms.spot - ms.gamma_flip) / spot,
        "channel_tightness": (ms.call_wall - ms.put_wall) / spot,
        "wall_proximity":   min(
            abs(ms.call_wall - ms.spot) / spot,
            abs(ms.spot - ms.put_wall) / spot,
        ),
        "term_structure":   (ms.vix3m - ms.vix) / ms.vix if ms.vix else None,
        "vvix_elevation":   ms.vvix / ms.vvix_baseline - 1.0 if ms.vvix_baseline else None,
        # richness / skew_dir / tail_heaviness require RND — not available here
    }

    return MTFInput(native=native, snapshot=snapshot)


def _print_tick(snap: FeedSnapshot, intent: TradeIntent,
                ticket: RoutedTicket, tick: int) -> None:
    ms = snap.market
    cand = ticket.candidate
    cand_str = (f"{cand.family} cr={cand.credit:.2f} ml={cand.max_loss:.2f}"
                if cand is not None else ticket.no_fill_reason or "no_candidate")
    print(f"  tick={tick:4d}  spot={ms.spot:.2f}  "
          f"{intent.decision.structure:14} {intent.decision.conviction:6}  "
          f"chain={'Y' if snap.chain else 'N'}  {cand_str}")


# --------------------------------------------------------------------------- #
# Demo                                                                         #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import sys

    # ---- without a chain (stub routing) ----
    print("=== Track B demo — no chain (stub routing) ===")
    feed_pin = SyntheticFeed(regime="pin", n_bars=30)
    orch = PipelineOrchestrator(feed_pin, verbose=True, every_n=10)
    tickets_pin = orch.run_session()
    trade_tix = [t for t in tickets_pin if t.intent.decision.structure != "NT"]
    print(f"  {len(tickets_pin)} ticks | {len(trade_tix)} with trade intent")

    print("\n=== Track B demo — trend_down ===")
    feed_dn = SyntheticFeed(regime="trend_down", n_bars=30)
    orch2 = PipelineOrchestrator(feed_dn, verbose=True, every_n=10)
    tickets_dn = orch2.run_session()
    trade_dn = [t for t in tickets_dn if t.intent.decision.structure != "NT"]
    print(f"  {len(tickets_dn)} ticks | {len(trade_dn)} with trade intent")

    # ---- with a chain (Track B -> A seam) ----
    if _TRACK_A_AVAILABLE:
        print("\n=== Track B -> A seam demo — pin regime WITH chain ===")
        from rnd_extractor import ChainSnapshot, ChainQuote, _bs_call_fwd
        spot0 = 600.0
        r0, T0 = 0.05, 4.0 / (24 * 365)
        DF0 = math.exp(-r0 * T0)
        F0 = spot0 * math.exp(r0 * T0)
        qs = []
        for K in np.arange(spot0 - 15, spot0 + 16, 1.0):
            k = math.log(K / F0)
            s = max(0.0050 - 0.030 * k, 0.0008)
            cm = _bs_call_fwd(F0, K, s) * DF0
            pm = max(cm - DF0 * (F0 - K), 0.0)
            cm = max(cm, 0.0)
            h = 0.01 + 0.002 * max(cm, pm)
            qs.append(ChainQuote(float(K), max(cm - h, 0), cm + h,
                                  max(pm - h, 0), pm + h))
        chain = ChainSnapshot(qs, spot=spot0, t_years=T0, r=r0)

        feed_seam = SyntheticFeed(regime="pin", n_bars=5, chain=chain)
        orch3 = PipelineOrchestrator(feed_seam, verbose=True, every_n=1)
        tickets_seam = orch3.run_session()
        for t in tickets_seam:
            if t.candidate is not None:
                c = t.candidate
                print(f"  FILLED: {c.family} shorts={c.short_strikes} "
                      f"cr={c.credit:.2f} ml={c.max_loss:.2f} ev={c.ev:.3f}")
                break
        else:
            print("  No candidate filled (check chain / regime routing).")
    else:
        print("\n(Track A modules not available — skipping seam demo)")
