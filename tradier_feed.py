"""
tradier_feed.py — live adapter: Tradier options chain -> OptionRow.

Uses Tradier market data + ORATS greeks. Set TRADIER_API_TOKEN in .env.
Run: python tradier_feed.py SPY
     python tradier_feed.py diagnose SPY
"""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import spy0dte as eng
from spy0dte import OptionRow
import tradier_client as tc

ET = ZoneInfo("America/New_York")


def _today_et() -> str:
    return datetime.now(ET).strftime("%Y-%m-%d")


def _spot_from_quote(symbol: str) -> float:
    data = tc.get_quotes(symbol)
    quote = data.get("quotes", {}).get("quote", {})
    if isinstance(quote, list):
        quote = quote[0] if quote else {}
    for key in ("last", "bid", "ask", "close"):
        val = quote.get(key)
        if val is not None and float(val) > 0:
            if key in ("bid", "ask") and quote.get("bid") and quote.get("ask"):
                return (float(quote["bid"]) + float(quote["ask"])) / 2
            return float(val)
    return 0.0


def _row_from_option(opt: dict) -> OptionRow | None:
    side = opt.get("option_type")
    strike = opt.get("strike")
    oi = opt.get("open_interest")
    greeks = opt.get("greeks") or {}
    gamma = greeks.get("gamma")
    delta = greeks.get("delta")
    bid = opt.get("bid")
    ask = opt.get("ask")
    if None in (side, strike, gamma, delta, oi):
        return None
    if bid is None or ask is None:
        close = opt.get("last") or opt.get("close")
        if close is None:
            return None
        bid = ask = close
    if float(bid) <= 0 or float(ask) <= 0:
        return None
    return OptionRow(
        side=side,
        strike=float(strike),
        oi=int(oi),
        gamma=float(gamma),
        bid=float(bid),
        ask=float(ask),
        delta=abs(float(delta)),
    )


def get_chain(underlying: str, zero_dte_only: bool = True) -> tuple[float, list[OptionRow]]:
    today = _today_et()
    expirations = tc.get_options_expirations(underlying)
    if zero_dte_only:
        if today not in expirations:
            raise RuntimeError(f"No 0DTE expiration for {underlying} on {today}")
        expirations = [today]

    spot = _spot_from_quote(underlying)
    rows: list[OptionRow] = []
    for exp in expirations:
        for opt in tc.get_options_chain(underlying, exp, greeks=True):
            row = _row_from_option(opt)
            if row:
                rows.append(row)
    return spot, rows


def diagnose(underlying: str = "SPY") -> None:
    print("Tradier feed diagnose")
    try:
        tc.diagnose()
    except RuntimeError as e:
        print("PROFILE FAILED:", e)
        return

    today = _today_et()
    print(f"\n--- chain mapping for {underlying} ({today}) ---")
    try:
        spot, rows = get_chain(underlying)
    except RuntimeError as e:
        print("CHAIN FAILED:", e)
        return

    print(f"spot={spot} | {len(rows)} rows with greeks/OI/quotes")
    for r in rows[:6]:
        print(f"  {r.side:4s} {r.strike:7.1f}  OI={r.oi:<6d} Γ={r.gamma:.4f} "
              f"Δ={r.delta:.2f}  {r.bid:.2f}/{r.ask:.2f}")
    if rows:
        gm = eng.build_gamma_map(rows, spot)
        print(f"  -> netGEX ratio {gm.net_ratio} | flip {gm.gamma_flip} | "
              f"walls {gm.put_wall}/{gm.call_wall} | regime {gm.regime.upper()}")
        d = eng.decide(gm, price_accepting=0)
        print(f"DECISION: {d.action} — {d.reason}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "diagnose":
        diagnose(sys.argv[2] if len(sys.argv) > 2 else "SPY")
    else:
        sym = sys.argv[1] if len(sys.argv) > 1 else "SPY"
        spot, rows = get_chain(sym)
        print(f"{sym}: spot={spot} | {len(rows)} 0DTE rows with complete greeks/quotes")
        for r in rows[:6]:
            print(f"  {r.side:4s} {r.strike:7.1f}  OI={r.oi:<6d} Γ={r.gamma:.4f} "
                  f"Δ={r.delta:.2f}  {r.bid:.2f}/{r.ask:.2f}")
        if rows:
            gm = eng.build_gamma_map(rows, spot)
            print(f"  -> netGEX ratio {gm.net_ratio} | flip {gm.gamma_flip} | "
                  f"walls {gm.put_wall}/{gm.call_wall} | regime {gm.regime.upper()}")
            d = eng.decide(gm, price_accepting=0)
            print(f"DECISION: {d.action} — {d.reason}")
            if d.action in ("CALL", "PUT"):
                risk, _ = eng.scale_risk(n_trades=0, win_rate=0.0, avg_win=0, avg_loss=0)
                order = eng.select_order(rows, d, equity=1000.0, risk_frac=risk)
                if order:
                    print("ORDER:", order.thesis, f"| risk ${order.dollar_risk:.0f}")
