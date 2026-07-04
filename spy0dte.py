"""
spy0dte.py  —  one mission: SPY 0DTE, directional, with the dealer flow.

THESIS (the whole system in one sentence):
    Buy SPY 0DTE in the direction of dealer hedging flow ONLY when dealers are
    short gamma (trend regime), targeting the next gamma wall and stopping on a
    flip reclaim. Stand aside when dealers are long gamma (pin regime).

Why this and nothing else:
  - 0DTE intraday price is dominated by dealer gamma hedging.
  - Short-gamma (negative GEX) => dealers chase => trends => long premium pays.
  - Long-gamma (positive GEX)  => dealers fade => pins  => long premium bleeds.
  So you only take directional shots when structure is on your side, and you
  refuse the rest. The refusal is the edge; the convexity is the payoff.

This file computes the gamma map from a chain snapshot and outputs ONE decision:
CALL / PUT / STAND ASIDE, with entry, target wall, stop, contract, and size.
It does not place orders. Swap SyntheticChain for your Massive adapter.

Not financial advice. 0DTE directional buying is the highest-variance path
there is; this engine is built to keep you in the game, not to promise wins.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from math import floor

# ----------------------------- knobs (few on purpose) ---------------------- #
MULT = 100                 # index/ETF options multiplier
RISK_CEILING = 0.10        # the MOST you may risk per trade — unlocked by evidence, not default
RISK_FLOOR = 0.02          # where you start, and stay, until logged expectancy earns more
KELLY_FRACTION = 0.5       # half-Kelly: never bet full-Kelly on an estimated edge
MIN_TRADES_TO_SCALE = 30   # below this sample, you do not know your edge — stay at the floor
PREMIUM_STOP = 0.35        # buyer: exit if option loses this fraction of entry premium
DELTA_LOW, DELTA_HIGH = 0.45, 0.62   # buyer: ATM/slightly-ITM band for directional convexity
MAX_SPREAD_PCT = 0.08      # reject contracts wider than this
MIN_NET_RATIO = 0.08       # |net/gross gamma| below this = no conviction, stand aside
WALL_MIN_DISTANCE = 0.0015 # target wall must be at least this far (frac of spot) to be worth it

# Seller side (pin regime). MUST be a cash-settled European index to avoid 0DTE
# assignment/pin risk — XSP, not SPY. Defined-risk only (iron condor).
SELLER_INSTRUMENT = "XSP"
CONDOR_WIDTH = 1.0        # points from short to long wing (defined max loss); 1-wide fits small accounts
CONDOR_PROFIT_TARGET = 0.50  # close at 50% of credit collected
CONDOR_STOP_MULT = 2.0    # bail if loss reaches 2x credit, or a short strike breaks


# --------------------------------- data ------------------------------------ #
@dataclass
class OptionRow:
    side: str        # "call" | "put"
    strike: float
    oi: int
    gamma: float     # per-share gamma from the chain
    bid: float
    ask: float
    delta: float     # absolute value
    # quote provenance — fallback quotes (bid=ask=close) make spread look like 0
    # and silently defeat the spread filter. Track it; reject it for live trades.
    quote_source: str = "live_quote"   # "live_quote" | "day_close_fallback"
    quote_valid: bool = True
    volume: int = 0                    # same-day contract volume (0 = not provided)

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2

    @property
    def spread_pct(self) -> float:
        return (self.ask - self.bid) / self.mid if self.mid > 0 else 9.99


@dataclass
class GammaMap:
    spot: float
    net_gex: float          # $bn of dealer gamma per 1% move (signed)
    net_ratio: float        # net / gross gamma — scale-free conviction (-1..+1)
    gamma_flip: float       # estimated price where net gamma crosses zero
    call_wall: float        # largest call-gamma strike above spot (resistance)
    put_wall: float         # largest put-gamma strike below spot (support)
    regime: str             # "trend" (short gamma) | "pin" (long gamma)


@dataclass
class Decision:
    action: str             # "CALL" | "PUT" | "STAND_ASIDE"
    reason: str
    entry_ref: float = 0.0
    target: float = 0.0
    stop_ref: float = 0.0


@dataclass
class Order:
    action: str
    strike: float
    delta: float
    entry_mid: float
    spread_pct: float
    contracts: int
    dollar_risk: float
    premium_stop: float
    thesis: str


# ----------------------- earned risk scaling (Kelly) ----------------------- #
def scale_risk(n_trades: int, win_rate: float, avg_win: float, avg_loss: float) -> tuple[float, str]:
    """
    The honest answer to 'how do I scale risk with certainty': you don't get
    certainty, you get a measured edge, and you bet a FRACTION of what that edge
    mathematically supports.

    Feed it your logged results. Returns (risk_fraction, explanation).
      - Below MIN_TRADES_TO_SCALE you don't yet know your edge -> RISK_FLOOR.
      - Else Kelly f* = W - (1-W)/R, take half, clamp to [FLOOR, CEILING].
      - Zero/negative edge -> floor. You never scale into a losing sample.
    """
    if n_trades < MIN_TRADES_TO_SCALE:
        return RISK_FLOOR, f"only {n_trades} logged trades — floor until {MIN_TRADES_TO_SCALE}"
    if avg_loss <= 0 or win_rate <= 0:
        return RISK_FLOOR, "no measurable edge yet — floor"
    R = avg_win / avg_loss
    kelly = win_rate - (1 - win_rate) / R
    if kelly <= 0:
        return RISK_FLOOR, f"Kelly {kelly:.2f} <= 0 — edge not positive, do NOT scale (floor)"
    frac = max(RISK_FLOOR, min(RISK_CEILING, kelly * KELLY_FRACTION))
    note = f"W={win_rate:.0%}, R={R:.2f} -> Kelly {kelly:.2f}, half {kelly*KELLY_FRACTION:.2f} -> risk {frac:.0%}"
    if kelly * KELLY_FRACTION > RISK_CEILING:
        note += f" (capped at {RISK_CEILING:.0%})"
    return frac, note


# ------------------------------ the gamma map ------------------------------ #
def build_gamma_map(chain: list[OptionRow], spot: float) -> GammaMap:
    """Per-strike signed GEX, net GEX, flip, and the two walls."""
    # signed dollar gamma per strike: calls +, puts -  (naive dealer model)
    by_strike: dict[float, float] = {}
    call_g: dict[float, float] = {}
    put_g: dict[float, float] = {}
    for r in chain:
        dollar_gamma = r.gamma * r.oi * MULT * spot * spot * 0.01
        signed = dollar_gamma if r.side == "call" else -dollar_gamma
        by_strike[r.strike] = by_strike.get(r.strike, 0.0) + signed
        if r.side == "call":
            call_g[r.strike] = call_g.get(r.strike, 0.0) + dollar_gamma
        else:
            put_g[r.strike] = put_g.get(r.strike, 0.0) + dollar_gamma

    net_gex = sum(by_strike.values()) / 1e9   # express in $bn/1%
    gross_gex = sum(abs(v) for v in by_strike.values()) / 1e9
    net_ratio = (net_gex / gross_gex) if gross_gex else 0.0

    # gamma flip: walk strikes low->high, find where cumulative signed GEX crosses 0
    strikes = sorted(by_strike)
    cum = 0.0
    flip = spot
    prev_k, prev_cum = strikes[0], 0.0
    for k in strikes:
        cum += by_strike[k]
        if prev_cum < 0 <= cum or prev_cum > 0 >= cum:
            # linear interp between prev_k and k
            span = (cum - prev_cum)
            flip = prev_k + (k - prev_k) * (0 - prev_cum) / span if span else k
            break
        prev_k, prev_cum = k, cum

    # walls: largest gamma concentration on each side of spot
    calls_above = {k: g for k, g in call_g.items() if k >= spot}
    puts_below = {k: g for k, g in put_g.items() if k <= spot}
    call_wall = (max(calls_above, key=calls_above.get) if calls_above
                 else max(call_g, key=call_g.get) if call_g else spot)
    put_wall = (max(puts_below, key=puts_below.get) if puts_below
                else min(put_g, key=put_g.get) if put_g else spot)

    regime = "pin" if spot > flip else "trend"
    return GammaMap(spot, round(net_gex, 3), round(net_ratio, 3), round(flip, 2),
                    call_wall, put_wall, regime)


# ------------------------------ the decision ------------------------------- #
def decide(gm: GammaMap, price_accepting: int) -> Decision:
    """
    price_accepting: +1 = breaking/holding above flip, -1 = below flip, 0 = at level.
    In live use, derive this from your 1m bar vs the flip (acceptance, not a wick).
    """
    # 1) conviction gate: flat gamma = no dealer pressure = no trade
    if abs(gm.net_ratio) < MIN_NET_RATIO:
        return Decision("STAND_ASIDE", f"net ratio {gm.net_ratio} too flat — no dealer flow to ride")

    # 2) pin regime: buying premium bleeds -> become the SELLER (defined risk, XSP)
    if gm.regime == "pin":
        return Decision("SELL_CONDOR",
                        f"long-gamma PIN (spot {gm.spot} > flip {gm.gamma_flip}) — harvest theta, "
                        f"short the walls {gm.put_wall}/{gm.call_wall}",
                        entry_ref=gm.spot, target=gm.gamma_flip, stop_ref=0.0)

    # 3) trend regime: trade WITH acceptance toward the matching wall
    if price_accepting > 0:
        target = gm.call_wall
        if (target - gm.spot) / gm.spot < WALL_MIN_DISTANCE:
            return Decision("STAND_ASIDE", f"call wall {target} too close — no room")
        return Decision("CALL", f"short-gamma TREND, accepted above flip {gm.gamma_flip} → chase to call wall",
                        entry_ref=gm.spot, target=target, stop_ref=gm.gamma_flip)
    if price_accepting < 0:
        target = gm.put_wall
        if (gm.spot - target) / gm.spot < WALL_MIN_DISTANCE:
            return Decision("STAND_ASIDE", f"put wall {target} too close — no room")
        return Decision("PUT", f"short-gamma TREND, accepted below flip {gm.gamma_flip} → chase to put wall",
                        entry_ref=gm.spot, target=target, stop_ref=gm.gamma_flip)

    return Decision("STAND_ASIDE", "at the flip, no acceptance — wait for the break")


# --------------------------- contract + sizing ----------------------------- #
def select_order(chain: list[OptionRow], d: Decision, equity: float, risk_frac: float,
                 require_live: bool = True) -> Order | None:
    if d.action not in ("CALL", "PUT"):
        return None
    want = "call" if d.action == "CALL" else "put"
    cands = [r for r in chain
             if r.side == want and DELTA_LOW <= r.delta <= DELTA_HIGH
             and r.spread_pct <= MAX_SPREAD_PCT and r.mid > 0
             and (r.quote_valid or not require_live)]   # never size live off a fallback quote
    if not cands:
        return None
    mid_band = (DELTA_LOW + DELTA_HIGH) / 2
    cands.sort(key=lambda r: (abs(r.delta - mid_band), r.spread_pct))
    pick = cands[0]

    budget = equity * risk_frac
    risk_per_contract = pick.mid * MULT * PREMIUM_STOP
    contracts = floor(budget / risk_per_contract)
    if contracts < 1:
        return None  # can't size to the risk rule -> abstain

    thesis = (f"{d.action} SPY {pick.strike:.0f} 0DTE @ ~{pick.mid:.2f} "
              f"(Δ{pick.delta:.2f}) x{contracts} | {d.reason} | "
              f"target {d.target:.2f}, stop on flip {d.stop_ref:.2f} or -{int(PREMIUM_STOP*100)}%")
    return Order(d.action, pick.strike, pick.delta, round(pick.mid, 2),
                 round(pick.spread_pct, 4), contracts,
                 round(contracts * risk_per_contract, 2),
                 round(pick.mid * (1 - PREMIUM_STOP), 2), thesis)


# --------------------------- seller: iron condor --------------------------- #
@dataclass
class CondorOrder:
    short_put: float
    long_put: float
    short_call: float
    long_call: float
    credit: float
    max_loss: float
    contracts: int
    dollar_risk: float
    profit_target: float
    thesis: str


def _nearest(chain: list[OptionRow], side: str, strike: float) -> OptionRow | None:
    rows = [r for r in chain if r.side == side]
    return min(rows, key=lambda r: abs(r.strike - strike)) if rows else None


def select_condor(chain: list[OptionRow], gm: GammaMap, equity: float,
                  risk_frac: float, require_live: bool = True) -> CondorOrder | None:
    """
    Sell the walls. Short put at the put wall, short call at the call wall, long
    wings CONDOR_WIDTH beyond. The walls are the natural short strikes: in a pin
    regime price is least likely to break them — that IS the thesis.
    Defined risk = (width - credit). Sized off your scaled risk fraction.
    NOTE: intended for XSP (cash-settled). Do not run this on SPY 0DTE.
    """
    sp = _nearest(chain, "put", gm.put_wall)
    lp = _nearest(chain, "put", gm.put_wall - CONDOR_WIDTH)
    sc = _nearest(chain, "call", gm.call_wall)
    lc = _nearest(chain, "call", gm.call_wall + CONDOR_WIDTH)
    if not all((sp, lp, sc, lc)):
        return None
    if require_live and not all((sp.quote_valid, lp.quote_valid,
                                 sc.quote_valid, lc.quote_valid)):
        return None  # any fallback-quoted leg -> credit is fiction -> abstain
    credit = (sp.mid - lp.mid) + (sc.mid - lc.mid)
    if credit <= 0:
        return None
    max_loss_per = (CONDOR_WIDTH - credit) * MULT
    if max_loss_per <= 0:
        return None  # credit >= width: free money artifact of bad quotes, skip
    budget = equity * risk_frac
    contracts = floor(budget / max_loss_per)
    if contracts < 1:
        return None
    thesis = (f"SELL {SELLER_INSTRUMENT} iron condor "
              f"{lp.strike:.0f}/{sp.strike:.0f}P - {sc.strike:.0f}/{lc.strike:.0f}C 0DTE | "
              f"credit ~{credit:.2f} x{contracts}, max loss ${max_loss_per*contracts:.0f} | "
              f"close at {int(CONDOR_PROFIT_TARGET*100)}% credit, bail at {CONDOR_STOP_MULT}x or wall break")
    return CondorOrder(sp.strike, lp.strike, sc.strike, lc.strike,
                       round(credit, 2), round(max_loss_per, 2), contracts,
                       round(max_loss_per * contracts, 2),
                       round(credit * CONDOR_PROFIT_TARGET, 2), thesis)


def synthetic_chain(spot: float, skew: str) -> list[OptionRow]:
    """skew: 'trend_down' | 'trend_up' | 'pin' — shapes OI to move the flip/walls."""
    rows: list[OptionRow] = []
    for i in range(-15, 16):
        k = round(spot + i)
        moneyness = abs(k - spot)
        gamma = max(0.001, 0.06 * 2.71828 ** (-(moneyness ** 2) / 8))  # peaked ATM
        # OI shaping
        if skew == "pin":
            call_oi = int(45000 * 2.71828 ** (-(moneyness ** 2) / 6)) + 2000
            put_oi = call_oi
        elif skew == "trend_down":
            put_oi = int(60000 * 2.71828 ** (-((k - (spot - 4)) ** 2) / 8)) + 1000
            call_oi = int(15000 * 2.71828 ** (-(moneyness ** 2) / 10)) + 500
        else:  # trend_up
            call_oi = int(60000 * 2.71828 ** (-((k - (spot + 4)) ** 2) / 8)) + 1000
            put_oi = int(15000 * 2.71828 ** (-(moneyness ** 2) / 10)) + 500
        for side, oi, dl in (("call", call_oi, max(0.02, min(0.95, 0.5 - 0.07 * i))),
                             ("put", put_oi, max(0.02, min(0.95, 0.5 + 0.07 * i)))):
            mid = max(0.05, 0.45 + gamma * 12 + max(0, (i if side == "call" else -i)) * 0.04)
            w = mid * 0.03
            rows.append(OptionRow(side, k, oi, round(gamma, 4),
                                  round(mid - w, 2), round(mid + w, 2), round(dl, 2)))
    return rows


def run_once(spot: float, skew: str, accepting: int, equity: float,
             risk_frac: float, iv_annual: float = 0.15, minutes_left: int = 120) -> None:
    """
    iv_annual / minutes_left: used by the MC to project the trade — informational
    only. Risk is always sized from risk_frac (journal-derived or floor). MC is a
    prior and a display; it does NOT override sizing.
    """
    import mc as _mc  # lazy import keeps mc/numpy optional for pure engine tests

    chain = synthetic_chain(spot, skew)
    gm = build_gamma_map(chain, spot)
    d = decide(gm, accepting)
    print(f"\nspot {spot} | netGEX {gm.net_gex} $bn (ratio {gm.net_ratio}) | flip {gm.gamma_flip} | "
          f"call_wall {gm.call_wall} | put_wall {gm.put_wall} | regime {gm.regime.upper()}")
    print(f"DECISION: {d.action} — {d.reason}")

    if d.action in ("CALL", "PUT"):
        dist_target = abs(d.target - spot)
        dist_stop = abs(spot - d.stop_ref) if d.stop_ref else dist_target / 2
        win_R = max(0.5, dist_target / dist_stop) if dist_stop > 0 else 1.0
        proj = _mc.project(spot=spot, target=d.target, stop=d.stop_ref,
                           flip=gm.gamma_flip, minutes_left=minutes_left,
                           iv_annual=iv_annual, regime=gm.regime, win_R=win_R, seed=42)
        print(f"MC PRIOR:  {proj.note}")

        order = select_order(chain, d, equity, risk_frac)
        if order:
            print("ORDER:", order.thesis, f"| risk ${order.dollar_risk:.0f}")
        else:
            print(f"NO ORDER: no contract fit, or stop-risk exceeds {risk_frac:.0%} of ${equity:.0f} — abstain")

    elif d.action == "SELL_CONDOR":
        win_R = CONDOR_PROFIT_TARGET / (1 - CONDOR_PROFIT_TARGET)
        proj = _mc.project_range(spot=spot, lower_short=gm.put_wall, upper_short=gm.call_wall,
                                 flip=gm.gamma_flip, minutes_left=minutes_left,
                                 iv_annual=iv_annual, regime=gm.regime, win_R=win_R, seed=42)
        print(f"MC PRIOR:  {proj.note}")

        condor = select_condor(chain, gm, equity, risk_frac)
        if condor:
            print("ORDER:", condor.thesis, f"| risk ${condor.dollar_risk:.0f}")
        else:
            print(f"NO ORDER: condor didn't fit {risk_frac:.0%} of ${equity:.0f} — abstain")


if __name__ == "__main__":
    EQ = 1000.0
    # earned risk: with no track record you are at the floor, full stop.
    risk0, why0 = scale_risk(n_trades=0, win_rate=0.0, avg_win=0, avg_loss=0)
    print(f"[risk] starting fraction {risk0:.0%} — {why0}")
    # illustrate what a logged edge would unlock:
    riskE, whyE = scale_risk(n_trades=40, win_rate=0.40, avg_win=2.6, avg_loss=1.0)
    print(f"[risk] example after 40 trades: {riskE:.0%} — {whyE}")

    print(f"\n--- running scenarios at the EARNED fraction ({riskE:.0%}); at the {risk0:.0%} "
          f"floor nothing fits a $1k account, which is the honest result ---")
    run_once(600.0, "trend_down", accepting=-1, equity=EQ, risk_frac=riskE)  # short gamma -> PUT
    run_once(600.0, "trend_up",   accepting=+1, equity=EQ, risk_frac=riskE)  # long gamma -> SELL condor
    run_once(600.0, "pin",        accepting=+1, equity=EQ, risk_frac=riskE)  # balanced OI -> flat ratio
    run_once(600.0, "trend_up",   accepting=0,  equity=EQ, risk_frac=riskE)  # no acceptance -> aside
