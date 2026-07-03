"""
synthetic_world.py
==================
A COUPLED synthetic market for the walk-forward / backtest harness.

The old SyntheticUnifiedFeed has random-walk bars but a frozen dealer surface
(fixed GEX/flip/walls), one static chain regardless of spot, and a constant
settlement — on that world "prediction" is unmeasurable by construction, so
backtests could only ever validate plumbing.

This world couples the pieces the way the system's thesis says they couple:

  net GEX (OU, regime-switching)  ──►  price dynamics
        gex > 0  →  mean-reversion toward the pin (suppressed vol)
        gex <= 0 →  persistent directional drift (elevated vol)
  price + regime                  ──►  chain repriced EVERY tick off current
                                       spot, T = time to 16:00, with a
                                       regime-dependent vol-risk premium
                                       (implied usually > realized, not always)
  the actual path                 ──►  settlement = the day's real close

So on this feed the pipeline's predictions are testable in principle:
premium selling should earn the VRP on pin days and get hurt on trend days;
the direction bias should beat a coin on trend days; EV calibration and
prob_profit calibration have real outcome variance to score against.

It is still a model — passing here does not prove live edge; failing here
means the machinery cannot even exploit a world built from its own thesis.

NOT financial advice.
"""
from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field
from typing import Optional
from zoneinfo import ZoneInfo

import numpy as np

from gate_scorer import MarketSnapshot
from gex_window import GexRankWindow
from massive_feed import _bar_technicals, _session_vwap_and_reversions
from resample import RawBars
from rnd_extractor import ChainQuote, ChainSnapshot, _bs_call_fwd, MINUTES_PER_YEAR
from unified_loop import TickSnapshot

ET = ZoneInfo("America/New_York")
MIN_PER_DAY = 390


@dataclass
class WorldConfig:
    days: int = 20
    seed: int = 7
    base_spot: float = 600.0
    tick_stride: int = 1            # serve every Nth minute (5 = fast CI runs)
    lookback_minutes: int = 2340    # trailing bar window served per tick (~6 days)

    # regime process
    p_trend_day: float = 0.30       # chance a day runs short-gamma / trending
    gex_pin_level: float = 3.0e9
    gex_trend_level: float = -1.2e9
    gex_noise: float = 0.4e9

    # price dynamics (annualized vols)
    pin_pull: float = 0.012         # per-minute pull toward the pin, gex>0
    pin_vol: float = 0.09
    trend_vol: float = 0.19
    trend_drift_frac: float = 0.06  # drift/min as fraction of minute-vol (~1.5%/day)
    overnight_gap_vol: float = 0.003

    # implied-vol premium: lognormal around slightly rich, occasionally cheap
    vrp_mu: float = 0.13            # exp(0.13) ≈ 1.14x realized
    vrp_sigma: float = 0.18

    # chain construction
    strike_span: float = 25.0
    smile_skew: float = 0.030
    half_spread: float = 0.012


class CoupledSyntheticFeed:
    """unified_loop.DataFeed over a pre-generated coupled world."""

    def __init__(self, cfg: Optional[WorldConfig] = None, **overrides) -> None:
        base = cfg or WorldConfig()
        if overrides:
            base = dataclass_replace(base, **overrides)
        self.cfg = base
        self._rng = np.random.default_rng(base.seed)
        self._gex_rank = GexRankWindow()          # memory-only, honest warm-up
        self._idx = 0
        self._generate()

    # -- world generation --------------------------------------------------------
    def _generate(self) -> None:
        c = self.cfg
        rng = self._rng
        sig_min_pin = c.pin_vol / math.sqrt(MINUTES_PER_YEAR)
        sig_min_trend = c.trend_vol / math.sqrt(MINUTES_PER_YEAR)

        ts, close, gex, pins, ivs, flips = [], [], [], [], [], []
        day_close: dict[str, float] = {}
        trend_days: dict[str, bool] = {}

        spot = c.base_spot
        pin = round(spot)
        start = dt.datetime(2026, 6, 1, 9, 30, tzinfo=ET)
        day0 = start.date()

        d = 0
        made = 0
        while made < c.days:
            date = day0 + dt.timedelta(days=d)
            d += 1
            if date.weekday() >= 5:
                continue
            made += 1

            is_trend = rng.random() < c.p_trend_day
            trend_dir = 1.0 if rng.random() < 0.5 else -1.0
            pin = round(pin + rng.integers(-2, 3))
            vrp = math.exp(rng.normal(c.vrp_mu, c.vrp_sigma))
            realized = c.trend_vol if is_trend else c.pin_vol
            iv_day = realized * vrp
            spot *= math.exp(rng.normal(0.0, c.overnight_gap_vol))

            gex_level = c.gex_trend_level if is_trend else c.gex_pin_level
            g = gex_level
            open_dt = dt.datetime(date.year, date.month, date.day, 9, 30, tzinfo=ET)
            flip_day = (pin - 4.0) if not is_trend else (spot + 3.0)

            for m in range(MIN_PER_DAY):
                g += 0.05 * (gex_level - g) + rng.normal(0.0, c.gex_noise) * 0.1
                if g > 0:
                    step = c.pin_pull * (pin - spot) / spot + sig_min_pin * rng.standard_normal()
                else:
                    step = (trend_dir * c.trend_drift_frac * sig_min_trend
                            + sig_min_trend * rng.standard_normal())
                spot *= (1.0 + step)

                ts.append(open_dt + dt.timedelta(minutes=m))
                close.append(spot)
                gex.append(g)
                pins.append(pin)
                ivs.append(iv_day)
                flips.append(flip_day)

            day_close[date.isoformat()] = spot
            trend_days[date.isoformat()] = is_trend

        n = len(close)
        close_a = np.asarray(close)
        self._ts = np.array([np.datetime64(t.replace(tzinfo=None)) for t in ts],
                            dtype="datetime64[ns]")
        self._dt = ts
        self._close = close_a
        self._open = np.concatenate([[close_a[0]], close_a[:-1]])
        spread = np.abs(rng.normal(0.0, 0.0004, n)) * close_a
        self._high = np.maximum(self._open, close_a) + spread
        self._low = np.minimum(self._open, close_a) - spread
        self._vol = rng.integers(2_000, 30_000, n).astype(float)
        self._gex = np.asarray(gex)
        self._pin = np.asarray(pins)
        self._iv = np.asarray(ivs)
        self._flip = np.asarray(flips)
        self.day_close = day_close
        self.trend_days = trend_days

    # -- chain pricing -------------------------------------------------------------
    def _chain(self, i: int) -> Optional[ChainSnapshot]:
        c = self.cfg
        spot = float(self._close[i])
        minute = i % MIN_PER_DAY
        minutes_left = max(MIN_PER_DAY - minute, 5)
        t_years = minutes_left / (365.25 * 24 * 60)     # calendar, like live feeds
        r = 0.05
        DF = math.exp(-r * t_years)
        F = spot / DF
        # implied total vol in TRADING-time convention, consistent with how the
        # realized vols above are annualized (MINUTES_PER_YEAR trading minutes)
        s_atm = self._iv[i] * math.sqrt(minutes_left / MINUTES_PER_YEAR)

        qs = []
        lo = math.floor(spot - c.strike_span)
        for K in np.arange(lo, spot + c.strike_span + 1, 1.0):
            if K <= 0:
                continue
            k = math.log(K / F)
            s = max(s_atm - c.smile_skew * k, 0.0006)
            cm = _bs_call_fwd(F, K, s) * DF
            pm = max(cm - DF * (F - K), 0.0)
            cm = max(cm, 0.0)
            h = c.half_spread + 0.002 * max(cm, pm)
            qs.append(ChainQuote(float(K), max(cm - h, 0.0), cm + h,
                                 max(pm - h, 0.0), pm + h))
        return ChainSnapshot(qs, spot=spot, t_years=t_years, r=r)

    # -- DataFeed protocol ------------------------------------------------------------
    def timestamps(self) -> list[dt.datetime]:
        return list(self._dt[:: self.cfg.tick_stride])

    def snapshot(self, now: dt.datetime) -> Optional[TickSnapshot]:
        i = self._idx * self.cfg.tick_stride
        if i >= len(self._close):
            return None
        self._idx += 1

        lo = max(0, i + 1 - self.cfg.lookback_minutes)
        bars = RawBars(ts=self._ts[lo:i + 1], open=self._open[lo:i + 1],
                       high=self._high[lo:i + 1], low=self._low[lo:i + 1],
                       close=self._close[lo:i + 1], volume=self._vol[lo:i + 1])

        spot = float(self._close[i])
        pin = float(self._pin[i])
        g = float(self._gex[i])
        chain = self._chain(i)
        tech = _bar_technicals(bars)
        vwap, vwap_rev = _session_vwap_and_reversions(bars, self._dt[i])

        # ATM straddle from the freshly priced chain
        atm = min(chain.quotes, key=lambda q: abs(q.strike - spot))
        straddle = atm.call_mid + atm.put_mid
        minute = i % MIN_PER_DAY
        minutes_left = max(MIN_PER_DAY - minute, 5)
        realized = (self.cfg.trend_vol if g <= 0 else self.cfg.pin_vol)
        expected_range = spot * realized * math.sqrt(minutes_left / MINUTES_PER_YEAR)

        iv_pts = self._iv[i] * 100.0
        trending = g <= 0
        market = MarketSnapshot(
            spot=spot, net_gex=g, gamma_flip=float(self._flip[i]),
            call_wall=pin + 5.0, put_wall=pin - 5.0,
            gex_pct_rank=self._gex_rank.rank(g),
            vix9d=iv_pts * (1.06 if trending else 0.94),
            vix=iv_pts,
            vix3m=iv_pts * (0.95 if trending else 1.12),
            vvix=105.0 if trending else 90.0, vvix_baseline=95.0,
            straddle_breakeven=straddle, expected_range=expected_range,
            adx=tech["adx"], rsi=tech["rsi"],
            bb_width=tech["bb_width"], bb_width_baseline=tech["bb_width_baseline"],
            vwap=vwap, vwap_reversion_count=vwap_rev,
            tick_abs_mean=780.0 if trending else 450.0,
            cvd_slope=tech["cvd_slope"],
            now=self._dt[i], has_catalyst=False,
        )
        return TickSnapshot(market=market, bars=bars, chain=chain)

    def settlement_price(self, session_date: str) -> Optional[float]:
        return self.day_close.get(session_date)


def dataclass_replace(cfg: WorldConfig, **kw) -> WorldConfig:
    import dataclasses
    return dataclasses.replace(cfg, **kw)


if __name__ == "__main__":
    feed = CoupledSyntheticFeed(WorldConfig(days=5, tick_stride=5))
    ticks = feed.timestamps()
    print(f"{len(ticks)} ticks over 5 days; "
          f"trend days: {sum(feed.trend_days.values())}/5")
    print("settles:", {k: round(v, 2) for k, v in feed.day_close.items()})
    snap = feed.snapshot(ticks[0])
    print(f"first tick: spot={snap.market.spot:.2f} gex={snap.market.net_gex/1e9:+.2f}bn "
          f"chain_strikes={len(snap.chain.quotes)} bars={len(snap.bars.close)}")
