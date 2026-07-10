"""
execution_cost.py
=================
Executable economics for Prediction Engine V2
(docs/PREDICTION_ENGINE_V2_HANDOFF.md §13).

Midpoint P&L is a diagnostic. Primary V2 economic labels and metrics use
net executable P&L under an explicit fill model:

  mid_price              — strategy value at option mids
  natural_price          — buy at ask / sell at bid (full concession)
  expected_fill_price    — mid moved toward natural by fill_fraction
  conservative_fill_price — same, with a higher fill_fraction

Credit convention (matches spread_selector._credit):
  credit > 0  => net premium collected
  credit < 0  => net debit paid

  expected_credit = mid_credit - fill_fraction * (mid_credit - natural_credit)

So expected credit never exceeds mid credit, and expected debit is never
cheaper than mid debit. Conservative is never better than expected.

Also: per-leg fees, quote-age penalties, exit-cost estimates, and a
paper/manual fill capture record for the eventual empirical fill model.

NOT financial advice.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional, Sequence

from prediction.models.fill import (
    FillPriorConfig, fill_fraction_for, n_legs as _n_legs,
)


# --------------------------------------------------------------------------- #
# Config                                                                       #
# --------------------------------------------------------------------------- #
@dataclass
class ExecutionCostConfig:
    """Operational priors for executable pricing (§13.1–13.4)."""
    fill: FillPriorConfig = field(default_factory=FillPriorConfig)
    # Conservative fill uses a higher concession than expected.
    conservative_fill_boost: float = 0.25       # added to expected fill_fraction, clipped to 1
    # Fees in $/share (matches the rest of the system's per-share accounting).
    # Rough retail prior: ~$0.65/contract round-trip / 100 = $0.0065/share/leg
    # round-trip; we split entry vs exit.
    fee_per_leg_entry: float = 0.0035
    fee_per_leg_exit: float = 0.0035
    fee_per_contract_entry: float = 0.0         # flat per-structure add-on
    fee_per_contract_exit: float = 0.0
    # Exit fill: closing a multi-leg structure is typically no easier than entry.
    exit_fill_boost: float = 0.10               # added to entry fill_fraction for exits
    stop_exit_fill_boost: float = 0.25          # stops cross the spread harder
    # When True, selector EV / ranking use expected-fill credit (PR 6 default
    # for V2 economic metrics). Mid credit remains on the candidate for
    # diagnostics and journal backward compatibility.
    use_executable_economics: bool = True


# --------------------------------------------------------------------------- #
# Quote view                                                                   #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LegQuote:
    """Bid/ask for one option contract (call or put at a strike)."""
    bid: float
    ask: float

    @property
    def mid(self) -> float:
        return 0.5 * (self.bid + self.ask)

    @property
    def spread(self) -> float:
        return max(self.ask - self.bid, 0.0)

    @property
    def half_spread(self) -> float:
        return 0.5 * self.spread


def quotes_from_chain(chain) -> dict:
    """
    Build {(strike, kind): LegQuote} from a ChainSnapshot / duck-typed chain
    with .quotes of objects exposing strike, call_bid/ask, put_bid/ask.
    """
    out: dict = {}
    for q in chain.quotes:
        out[(float(q.strike), "C")] = LegQuote(float(q.call_bid), float(q.call_ask))
        out[(float(q.strike), "P")] = LegQuote(float(q.put_bid), float(q.put_ask))
    return out


def _leg_kind(leg) -> str:
    return leg.kind if hasattr(leg, "kind") else leg["kind"]


def _leg_strike(leg) -> float:
    return float(leg.strike if hasattr(leg, "strike") else leg["strike"])


def _leg_qty(leg) -> int:
    return int(leg.qty if hasattr(leg, "qty") else leg["qty"])


def _as_leg_dicts(legs: Sequence) -> list:
    return [{"strike": _leg_strike(lg), "kind": _leg_kind(lg),
             "qty": _leg_qty(lg)} for lg in legs]


# --------------------------------------------------------------------------- #
# Strategy prices                                                              #
# --------------------------------------------------------------------------- #
def mid_credit(legs: Sequence, quotes: dict) -> Optional[float]:
    """Net credit at mids. None if any leg quote is missing."""
    total = 0.0
    for lg in legs:
        q = quotes.get((_leg_strike(lg), _leg_kind(lg)))
        if q is None:
            return None
        total += -_leg_qty(lg) * q.mid
    return float(total)


def natural_credit(legs: Sequence, quotes: dict) -> Optional[float]:
    """
    Net credit at the natural (adverse) side of every quote:
      buy legs (qty > 0) fill at ask; sell legs (qty < 0) fill at bid.
    Always weakly worse than mid_credit for a consistent book.
    """
    total = 0.0
    for lg in legs:
        q = quotes.get((_leg_strike(lg), _leg_kind(lg)))
        if q is None:
            return None
        px = q.ask if _leg_qty(lg) > 0 else q.bid
        total += -_leg_qty(lg) * px
    return float(total)


def half_spread_cost(legs: Sequence, quotes: dict) -> Optional[float]:
    """mid_credit - natural_credit (>= 0). The full midpoint-to-natural concession."""
    mid = mid_credit(legs, quotes)
    nat = natural_credit(legs, quotes)
    if mid is None or nat is None:
        return None
    return float(max(mid - nat, 0.0))


def fill_credit(mid: float, natural: float, fill_frac: float) -> float:
    """
    Interpolate mid → natural by fill_fraction ∈ [0, 1].
    fill_frac=0 → mid; fill_frac=1 → natural.
    """
    f = float(np_clip(fill_frac, 0.0, 1.0))
    return float(mid - f * (mid - natural))


def np_clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))


# --------------------------------------------------------------------------- #
# Fees + exit costs                                                            #
# --------------------------------------------------------------------------- #
def entry_fees(n_legs: int, cfg: ExecutionCostConfig) -> float:
    """Total entry fees in $/share (positive cost)."""
    return float(n_legs * cfg.fee_per_leg_entry + cfg.fee_per_contract_entry)


def exit_fees(n_legs: int, cfg: ExecutionCostConfig) -> float:
    return float(n_legs * cfg.fee_per_leg_exit + cfg.fee_per_contract_exit)


def exit_half_spread_cost(entry_half_spread: float, fill_frac_exit: float) -> float:
    """Expected exit concession: fill_frac × the same half-spread cost scale."""
    return float(max(fill_frac_exit, 0.0) * max(entry_half_spread, 0.0))


# --------------------------------------------------------------------------- #
# Execution estimate                                                           #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ExecutionEstimate:
    """Full executable-price panel for one multi-leg candidate (§13.1)."""
    mid_credit: float
    natural_credit: float
    expected_credit: float
    conservative_credit: float
    half_spread_cost: float
    fill_fraction_expected: float
    fill_fraction_conservative: float
    entry_fees: float
    exit_fees_expected: float
    exit_slippage_expected: float
    exit_slippage_stop: float
    # Net executable entry credit after entry fees (still before exit costs).
    net_expected_credit: float
    net_conservative_credit: float
    # Round-trip cost drag vs mid (entry concession + fees + expected exit).
    round_trip_cost_expected: float
    n_legs: int
    family: str
    diagnostics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def estimate_execution(
    legs: Sequence,
    quotes: dict,
    family: str,
    *,
    cfg: Optional[ExecutionCostConfig] = None,
    quote_age_seconds: Optional[float] = None,
    minutes_to_close: Optional[float] = None,
    relative_spread: Optional[float] = None,
    option_price: Optional[float] = None,
    realized_vol: Optional[float] = None,
) -> Optional[ExecutionEstimate]:
    """
    Build the mid / natural / expected / conservative credit panel for one
    candidate. Returns None when any leg quote is missing.
    """
    cfg = cfg or ExecutionCostConfig()
    mid = mid_credit(legs, quotes)
    nat = natural_credit(legs, quotes)
    if mid is None or nat is None:
        return None
    # Numerical guard: natural must not be better than mid (quote crossed book).
    if nat > mid + 1e-9:
        nat = mid

    hs = float(max(mid - nat, 0.0))
    n = _n_legs(legs)
    frac_exp, fill_diag = fill_fraction_for(
        family, n_legs=n, quote_age_seconds=quote_age_seconds,
        minutes_to_close=minutes_to_close, relative_spread=relative_spread,
        option_price=option_price, realized_vol=realized_vol, cfg=cfg.fill)
    frac_con = np_clip(frac_exp + cfg.conservative_fill_boost, 0.0, 1.0)

    exp_c = fill_credit(mid, nat, frac_exp)
    con_c = fill_credit(mid, nat, frac_con)
    # Hard monotonicity (acceptance criteria) — defend against float noise.
    exp_c = min(exp_c, mid)
    con_c = min(con_c, exp_c)

    fees_in = entry_fees(n, cfg)
    fees_out = exit_fees(n, cfg)
    frac_exit = np_clip(frac_exp + cfg.exit_fill_boost, 0.0, 1.0)
    frac_stop = np_clip(frac_exp + cfg.stop_exit_fill_boost, 0.0, 1.0)
    exit_slip = exit_half_spread_cost(hs, frac_exit)
    stop_slip = exit_half_spread_cost(hs, frac_stop)

    net_exp = exp_c - fees_in
    net_con = con_c - fees_in
    # Round-trip drag vs mid: entry concession + entry fees + expected exit
    # concession + exit fees. Always >= 0.
    rt_cost = (mid - exp_c) + fees_in + exit_slip + fees_out

    return ExecutionEstimate(
        mid_credit=float(mid),
        natural_credit=float(nat),
        expected_credit=float(exp_c),
        conservative_credit=float(con_c),
        half_spread_cost=hs,
        fill_fraction_expected=float(frac_exp),
        fill_fraction_conservative=float(frac_con),
        entry_fees=fees_in,
        exit_fees_expected=fees_out,
        exit_slippage_expected=float(exit_slip),
        exit_slippage_stop=float(stop_slip),
        net_expected_credit=float(net_exp),
        net_conservative_credit=float(net_con),
        round_trip_cost_expected=float(max(rt_cost, 0.0)),
        n_legs=n,
        family=family,
        diagnostics=fill_diag,
    )


def net_pnl(legs: Sequence, entry_credit: float, settle_price: float,
            *, entry_fees: float = 0.0, exit_fees: float = 0.0,
            exit_slippage: float = 0.0) -> float:
    """
    Settlement P&L under an executable entry credit, minus fees and exit
    slippage. Same intrinsic math as journal.realized_pnl / labels._structure_value.
    """
    total = float(entry_credit)
    for lg in legs:
        k, kind, qty = _leg_strike(lg), _leg_kind(lg), _leg_qty(lg)
        intrinsic = (max(settle_price - k, 0.0) if kind == "C"
                     else max(k - settle_price, 0.0))
        total += qty * intrinsic
    return float(total - entry_fees - exit_fees - exit_slippage)


# --------------------------------------------------------------------------- #
# Paper / manual fill capture (§13.3)                                          #
# --------------------------------------------------------------------------- #
@dataclass
class FillRecord:
    """One observed fill for the eventual empirical fill_fraction model."""
    candidate_id: str
    snapshot_id: str
    family: str
    decision_ts: str
    submission_ts: Optional[str] = None
    fill_ts: Optional[str] = None
    legs_json: list = field(default_factory=list)
    quoted_bid_ask: dict = field(default_factory=dict)   # strike:kind -> [bid, ask]
    mid_credit: Optional[float] = None
    natural_credit: Optional[float] = None
    limit_price: Optional[float] = None
    fill_price: Optional[float] = None                   # signed credit convention
    partial: bool = False
    cancelled: bool = False
    broker_fees: Optional[float] = None
    source: str = "paper"                                # paper | manual | live

    def realized_fill_fraction(self) -> Optional[float]:
        """Where the fill sat between mid (0) and natural (1). None if unknown."""
        if (self.fill_price is None or self.mid_credit is None
                or self.natural_credit is None):
            return None
        span = self.mid_credit - self.natural_credit
        if abs(span) < 1e-12:
            return 0.0
        return float(np_clip((self.mid_credit - self.fill_price) / span, 0.0, 1.0))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["realized_fill_fraction"] = self.realized_fill_fraction()
        return d


def make_fill_record(*, candidate_id: str, snapshot_id: str, family: str,
                     decision_ts: str, legs: Sequence, quotes: dict,
                     fill_price: Optional[float] = None,
                     limit_price: Optional[float] = None,
                     submission_ts: Optional[str] = None,
                     fill_ts: Optional[str] = None,
                     broker_fees: Optional[float] = None,
                     partial: bool = False, cancelled: bool = False,
                     source: str = "paper") -> FillRecord:
    """Capture a paper/manual fill with the quoted book at decision time."""
    qba = {f"{_leg_strike(lg)}:{_leg_kind(lg)}":
           [quotes[(_leg_strike(lg), _leg_kind(lg))].bid,
            quotes[(_leg_strike(lg), _leg_kind(lg))].ask]
           for lg in legs
           if (_leg_strike(lg), _leg_kind(lg)) in quotes}
    return FillRecord(
        candidate_id=candidate_id, snapshot_id=snapshot_id, family=family,
        decision_ts=decision_ts,
        submission_ts=submission_ts or decision_ts,
        fill_ts=fill_ts,
        legs_json=_as_leg_dicts(legs),
        quoted_bid_ask=qba,
        mid_credit=mid_credit(legs, quotes),
        natural_credit=natural_credit(legs, quotes),
        limit_price=limit_price, fill_price=fill_price,
        partial=partial, cancelled=cancelled,
        broker_fees=broker_fees, source=source,
    )
