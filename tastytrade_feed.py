"""
tastytrade_feed.py  —  live DataFeed adapter backed by the Tastytrade API.

WHY THIS EXISTS
    The chain fallback for Tradier. Tastytrade is a broker (so: real-time option
    NBBO + greeks, free with an account, and a second execution venue) whose
    market data is delivered ONLY over a DXLink (dxFeed) websocket — there is no
    one-shot REST "chain with quotes" call like Tradier's. So this adapter:
        1. REST  get_option_chain(symbol)         -> today's 0DTE streamer symbols
        2. DXLink subscribe Quote + Greeks + Summary for those symbols
        3. snapshot the stream once -> OptionRow list -> same TickSnapshot
    It produces the identical TickSnapshot as TradierDataFeed/MassiveDataFeed, so
    it is a drop-in for UnifiedOrchestrator and for CompositeFeed failover.

    Division of labour: this adapter owns the HARD role (option NBBO+greeks+OI and
    a real-time equity spot). The easy three (bars / VIX / settlement) come from
    the free YahooBackstop, so the adapter stays small and never reimplements the
    Candle-stream dance.

AUTH (OAuth2 — the current SDK path). Credentials from environment ONLY:
    export TASTYTRADE_CLIENT_SECRET=...      # from your personal OAuth grant
    export TASTYTRADE_REFRESH_TOKEN=...      # long-lived; no password on the box
    export TASTYTRADE_TEST=1                 # optional: use the cert/sandbox env

DEPENDENCY: the `tastytrade` SDK (pip install tastytrade), Python 3.10+. It is
imported lazily so the rest of the system has no hard dependency on it.

NOT financial advice.
"""
from __future__ import annotations

import asyncio
import datetime as dt
import os
from datetime import datetime
from typing import Any, Optional
from zoneinfo import ZoneInfo

from spy0dte import OptionRow, build_gamma_map
from gate_scorer import MarketSnapshot
from unified_loop import TickSnapshot
from yahoo_feed import YahooBackstop
from gex_window import GexRankWindow
# Reuse the chain/technical helpers already proven against the Massive feed.
from massive_feed import (
    _option_rows_to_chain_snapshot, _bar_technicals,
    _session_vwap_and_reversions, _atm_straddle_price,
)

ET = ZoneInfo("America/New_York")


# --------------------------------------------------------------------------- #
# Session                                                                       #
# --------------------------------------------------------------------------- #
def _make_session():
    """Build a Tastytrade OAuth2 Session from the environment. Lazy-imports the
    SDK so importing this module never requires the package to be installed."""
    secret = os.environ.get("TASTYTRADE_CLIENT_SECRET")
    refresh = os.environ.get("TASTYTRADE_REFRESH_TOKEN")
    if not secret or not refresh:
        raise RuntimeError(
            "TASTYTRADE_CLIENT_SECRET and TASTYTRADE_REFRESH_TOKEN must be set "
            "(create a personal OAuth grant in your Tastytrade account)."
        )
    try:
        from tastytrade import Session
    except ImportError as e:
        raise RuntimeError(
            "tastytrade SDK not installed. Run: pip install tastytrade"
        ) from e
    is_test = os.environ.get("TASTYTRADE_TEST", "").lower() in ("1", "true", "yes")
    return Session(secret, refresh, is_test=is_test)


# --------------------------------------------------------------------------- #
# Chain helpers                                                                  #
# --------------------------------------------------------------------------- #
def _todays_0dte_options(session, symbol: str) -> list:
    """Return the list of SDK Option objects expiring TODAY (the 0DTE set), or
    [] if the symbol does not expire today."""
    from tastytrade.instruments import get_option_chain
    chain = get_option_chain(session, symbol)          # {date: [Option, ...]}
    today = datetime.now(ET).date()
    return chain.get(today, [])


def _side_of(opt) -> Optional[str]:
    """Map the SDK OptionType enum to our 'call'/'put' regardless of whether it
    exposes .value ('C'/'P') or .name (CALL/PUT)."""
    ot = getattr(opt, "option_type", None)
    raw = getattr(ot, "value", None) or getattr(ot, "name", None) or str(ot)
    raw = str(raw).upper()
    if raw.startswith("C"):
        return "call"
    if raw.startswith("P"):
        return "put"
    return None


def _row_from(meta: dict, q: Any, g: Any, oi: Optional[int]) -> Optional[OptionRow]:
    """Assemble one OptionRow from the chain metadata + streamed events.
    Returns None when any required field is missing or the market is one-sided."""
    side = meta["side"]
    strike = meta["strike"]
    if q is None or g is None:
        return None
    bid = getattr(q, "bid_price", None)
    ask = getattr(q, "ask_price", None)
    gamma = getattr(g, "gamma", None)
    delta = getattr(g, "delta", None)
    if None in (bid, ask, gamma, delta):
        return None
    bid = float(bid); ask = float(ask)
    if bid <= 0 or ask <= 0:                # no two-sided market -> unusable
        return None
    return OptionRow(
        side=side, strike=float(strike),
        oi=int(oi) if oi is not None else 0,
        gamma=float(gamma), bid=bid, ask=ask, delta=abs(float(delta)),
        quote_source="tastytrade_live", quote_valid=True,
    )


async def _snapshot_chain(session, options: list, equity_symbol: str,
                          budget_s: float = 8.0) -> tuple[list[OptionRow], Optional[float]]:
    """Open a DXLink stream, subscribe Quote/Greeks/Summary for the option
    streamer symbols (+ Quote for the equity), collect ONE snapshot within a
    time budget, and return (rows, spot_from_equity)."""
    from tastytrade import DXLinkStreamer
    from tastytrade.dxfeed import Quote, Greeks, Summary

    # streamer_symbol -> {side, strike}
    meta: dict[str, dict] = {}
    for o in options:
        ss = getattr(o, "streamer_symbol", None)
        side = _side_of(o)
        strike = getattr(o, "strike_price", None)
        if ss and side and strike is not None:
            meta[ss] = {"side": side, "strike": strike}
    if not meta:
        return [], None

    opt_syms = list(meta.keys())
    quotes: dict[str, Any] = {}
    greeks: dict[str, Any] = {}
    ois: dict[str, int] = {}
    equity_quote: Any = None

    async with DXLinkStreamer(session) as streamer:
        await streamer.subscribe(Quote, [equity_symbol] + opt_syms)
        await streamer.subscribe(Greeks, opt_syms)
        await streamer.subscribe(Summary, opt_syms)

        async def _drain(event_type, store_quote: bool):
            nonlocal equity_quote
            async for ev in streamer.listen(event_type):
                sym = getattr(ev, "event_symbol", None)
                if sym is None:
                    continue
                if event_type is Quote:
                    if sym == equity_symbol:
                        equity_quote = ev
                    else:
                        quotes[sym] = ev
                elif event_type is Greeks:
                    greeks[sym] = ev
                elif event_type is Summary:
                    oi = getattr(ev, "open_interest", None)
                    if oi is not None:
                        ois[sym] = int(oi)
                # Stop early once both sides of the book are populated.
                if (len(quotes) >= len(opt_syms)
                        and len(greeks) >= len(opt_syms)):
                    return

        try:
            await asyncio.wait_for(
                asyncio.gather(_drain(Quote, True), _drain(Greeks, False),
                               _drain(Summary, False)),
                timeout=budget_s,
            )
        except asyncio.TimeoutError:
            pass  # partial snapshot is fine; rows missing data drop out below

    rows: list[OptionRow] = []
    for ss, m in meta.items():
        row = _row_from(m, quotes.get(ss), greeks.get(ss), ois.get(ss))
        if row is not None:
            rows.append(row)

    spot = None
    if equity_quote is not None:
        b = getattr(equity_quote, "bid_price", None)
        a = getattr(equity_quote, "ask_price", None)
        if b and a:
            spot = (float(b) + float(a)) / 2.0
    return rows, spot


# --------------------------------------------------------------------------- #
# Feed                                                                          #
# --------------------------------------------------------------------------- #
class TastytradeDataFeed:
    """
    Drop-in live DataFeed for UnifiedOrchestrator backed by Tastytrade (real-time
    option NBBO + greeks over DXLink). Mirrors TradierDataFeed's TickSnapshot and
    reuses the Massive chain/technical helpers; the easy-three fields come from
    the free Yahoo backstop.

    SECURITY: OAuth2 credentials from environment ONLY.
    """

    def __init__(
        self,
        underlying: str = "SPY",
        lookback_minutes: int = 7800,
        r: float = 0.05,
        vix9d: float = 14.0,
        vix: float = 15.0,
        vix3m: float = 17.0,
        vvix: float = 92.0,
        vvix_baseline: float = 95.0,
        use_live_vix: bool = True,
        vix_refresh_seconds: int = 600,
        gex_history_len: int = 100,        # retained for API compat; window is time-based now
        has_catalyst: bool = False,
        catalyst_label: Optional[str] = None,
        stream_budget_s: float = 8.0,
        gex_history_path: Optional[str] = None,   # persist |GEX| rank window across restarts
    ) -> None:
        self.underlying = underlying
        self.lookback_minutes = lookback_minutes
        self.r = r
        self._vix9d, self._vix, self._vix3m = vix9d, vix, vix3m
        self._vvix, self._vvix_baseline = vvix, vvix_baseline
        self._use_live_vix = use_live_vix
        self._vix_refresh_seconds = vix_refresh_seconds
        self._vix_ts: Optional[datetime] = None
        self._gex_window = GexRankWindow(path=gex_history_path)
        self.has_catalyst = has_catalyst
        self.catalyst_label = catalyst_label
        self.stream_budget_s = stream_budget_s
        self._backstop = YahooBackstop(underlying)
        self._session = None

    # -- internals -----------------------------------------------------------
    def _ensure_session(self):
        if self._session is None:
            self._session = _make_session()
        return self._session

    def _gex_pct_rank(self, net_gex: float) -> float:
        return self._gex_window.rank(net_gex)

    def _t_years(self, now: dt.datetime) -> float:
        today = now.astimezone(ET)
        expiry = dt.datetime(today.year, today.month, today.day, 16, 0, 0, tzinfo=ET)
        return max((expiry - today).total_seconds(), 60.0) / (365.25 * 24.0 * 3600.0)

    def _maybe_refresh_vix(self, now: dt.datetime) -> None:
        if not self._use_live_vix:
            return
        if (self._vix_ts is not None
                and (now - self._vix_ts).total_seconds() < self._vix_refresh_seconds):
            return
        ts = self._backstop.vix_term_structure()      # real CBOE indices, free
        if ts and ts["vix"] > 0:
            self._vix9d, self._vix, self._vix3m = ts["vix9d"], ts["vix"], ts["vix3m"]
            self._vvix = ts.get("vvix", self._vvix)
            self._vix_ts = now

    # -- DataFeed protocol --
    def snapshot(self, now: dt.datetime) -> Optional[TickSnapshot]:
        try:
            session = self._ensure_session()
            options = _todays_0dte_options(session, self.underlying)
            if not options:
                return None                  # not a 0DTE session for this symbol
            rows, stream_spot = asyncio.run(
                _snapshot_chain(session, options, self.underlying, self.stream_budget_s)
            )
        except Exception:
            return None
        if not rows:
            return None

        spot = stream_spot or self._backstop.spot()
        if not spot or spot <= 0:
            return None

        raw = self._backstop.bars(self.lookback_minutes)
        if raw is None:
            return None

        self._maybe_refresh_vix(now)

        gm = build_gamma_map(rows, spot)
        gex_rank = self._gex_pct_rank(gm.net_gex)
        chain = _option_rows_to_chain_snapshot(spot, rows, self._t_years(now), self.r)
        tech = _bar_technicals(raw)
        vwap, vwap_rev = _session_vwap_and_reversions(raw, now)
        straddle_be = _atm_straddle_price(rows, spot)

        market = MarketSnapshot(
            spot=spot, net_gex=gm.net_gex, gamma_flip=gm.gamma_flip,
            call_wall=gm.call_wall, put_wall=gm.put_wall, gex_pct_rank=gex_rank,
            vix9d=self._vix9d, vix=self._vix, vix3m=self._vix3m,
            vvix=self._vvix, vvix_baseline=self._vvix_baseline,
            straddle_breakeven=straddle_be, expected_range=straddle_be / 1.25,
            adx=tech["adx"], rsi=tech["rsi"],
            bb_width=tech["bb_width"], bb_width_baseline=tech["bb_width_baseline"],
            vwap=vwap, vwap_reversion_count=vwap_rev,
            tick_abs_mean=480.0,            # $TICK not sourced here; calm default
            cvd_slope=tech["cvd_slope"],
            now=now, has_catalyst=self.has_catalyst, catalyst_label=self.catalyst_label,
        )
        return TickSnapshot(market=market, bars=raw, chain=chain)

    def settlement_price(self, session_date: str) -> Optional[float]:
        return self._backstop.settlement(session_date)


# --------------------------------------------------------------------------- #
# Diagnostic                                                                    #
# --------------------------------------------------------------------------- #
def diagnose(symbol: str = "SPY") -> None:
    """Confirm OAuth + real-time NBBO without exposing credentials."""
    if not (os.environ.get("TASTYTRADE_CLIENT_SECRET")
            and os.environ.get("TASTYTRADE_REFRESH_TOKEN")):
        print("TASTYTRADE_CLIENT_SECRET / TASTYTRADE_REFRESH_TOKEN not set — "
              "create a personal OAuth grant and export them, then rerun.")
        return
    try:
        session = _make_session()
        print("AUTH OK — session established.")
    except RuntimeError as e:
        print("AUTH FAILED:", e)
        return
    options = _todays_0dte_options(session, symbol)
    print(f"0DTE today? {'YES — ' + str(len(options)) + ' contracts' if options else 'no expiration today'}")
    if options:
        rows, spot = asyncio.run(_snapshot_chain(session, options, symbol, budget_s=8.0))
        print(f"spot (equity mid) = {spot}")
        live = sum(1 for r in rows if r.quote_valid)
        print(f"chain: {len(rows)} contracts with real-time NBBO+greeks ({live} valid)")
        for r in rows[:5]:
            print(f"  {r.side:4s} {r.strike:7.1f} OI={r.oi:<6d} "
                  f"Γ={r.gamma:.4f} Δ={r.delta:.2f}  {r.bid:.2f}/{r.ask:.2f}")


if __name__ == "__main__":
    import sys
    diagnose(sys.argv[1] if len(sys.argv) > 1 else "SPY")
