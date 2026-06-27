"""
risk_manager.py
===============
Hard position-level and portfolio-level risk guards for the 0DTE pipeline.

Guards (all disabled by default — set to enable):
  daily_loss_limit      Max cumulative max_loss committed in a session.
                        Measured in $-per-contract as stored in the journal.
                        When exhausted, no new trades open until the next session.
  max_open_positions    Max concurrent same-day positions (0 = unlimited).
  max_portfolio_gamma   Max net |gamma| across open positions (0 = unlimited).

Design
------
* check() is read-only: it returns a RiskCheck without mutating state.
* record_trade() commits a trade: call it only after check() returned approved.
* State is session-scoped: counters reset automatically when session_date changes.
* close_positions() clears intraday state explicitly (call at EOD / settlement).

Integration: pass a RiskManager instance to UnifiedOrchestrator.risk_manager.
The orchestrator calls check() → if approved, record_trade() → then journals
and notifies. Risk-vetoed trades are logged as NO_TRADE with no_trade_reason
prefixed "risk:" so gate_effectiveness() can distinguish them.

NOT financial advice.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# --------------------------------------------------------------------------- #
# Config & result types                                                         #
# --------------------------------------------------------------------------- #
@dataclass
class RiskConfig:
    daily_loss_limit: float = 0.0       # 0 = disabled; dollars per contract
    max_open_positions: int = 0         # 0 = unlimited
    max_portfolio_gamma: float = 0.0    # 0 = disabled; net |gamma| units


@dataclass
class RiskCheck:
    approved: bool
    vetoes: list[str]


@dataclass
class _Pos:
    max_loss: float
    gamma: float
    family: str


# --------------------------------------------------------------------------- #
# RiskManager                                                                   #
# --------------------------------------------------------------------------- #
class RiskManager:
    """
    Intraday risk guard.  Thread-unsafe by design — the pipeline is single-threaded.
    """

    def __init__(self, cfg: Optional[RiskConfig] = None) -> None:
        self._cfg = cfg or RiskConfig()
        self._session: str = ""
        self._positions: list[_Pos] = []
        self._daily_loss: float = 0.0

    # -- two-phase commit interface ------------------------------------------

    def check(self, candidate, session_date: str) -> RiskCheck:
        """Read-only gate. Returns RiskCheck without mutating any state."""
        self._maybe_reset(session_date)
        cfg = self._cfg
        vetoes: list[str] = []

        # daily loss budget
        if cfg.daily_loss_limit > 0:
            projected = self._daily_loss + (candidate.max_loss or 0.0)
            if projected > cfg.daily_loss_limit:
                vetoes.append(
                    f"daily_loss:{projected:.4f}>{cfg.daily_loss_limit:.4f}"
                )

        # position count
        if cfg.max_open_positions > 0 and len(self._positions) >= cfg.max_open_positions:
            vetoes.append(
                f"max_positions:{len(self._positions)}>={cfg.max_open_positions}"
            )

        # portfolio gamma
        if cfg.max_portfolio_gamma > 0:
            net_g = sum(p.gamma for p in self._positions) + abs(candidate.gamma or 0.0)
            if net_g > cfg.max_portfolio_gamma:
                vetoes.append(
                    f"max_gamma:{net_g:.6f}>{cfg.max_portfolio_gamma:.6f}"
                )

        return RiskCheck(approved=not vetoes, vetoes=vetoes)

    def record_trade(self, candidate, session_date: str) -> None:
        """Commit a trade. Call only after check() returned approved=True."""
        self._maybe_reset(session_date)
        ml = candidate.max_loss or 0.0
        self._positions.append(
            _Pos(max_loss=ml, gamma=abs(candidate.gamma or 0.0), family=candidate.family)
        )
        self._daily_loss += ml

    def close_positions(self) -> None:
        """Clear open position tracking. Call at EOD / after settlement."""
        self._positions.clear()

    # -- status --------------------------------------------------------------

    def status(self) -> dict:
        return {
            "session_date": self._session,
            "open_positions": len(self._positions),
            "daily_loss_committed": round(self._daily_loss, 6),
            "net_gamma": round(sum(p.gamma for p in self._positions), 6),
            "families": [p.family for p in self._positions],
        }

    # -- internal ------------------------------------------------------------

    def _maybe_reset(self, session_date: str) -> None:
        if session_date != self._session:
            self._session = session_date
            self._positions.clear()
            self._daily_loss = 0.0


# --------------------------------------------------------------------------- #
# Smoke test                                                                    #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    from dataclasses import dataclass as dc

    @dc
    class _FakeCand:
        max_loss: float
        gamma: float
        family: str = "iron_condor"

    cfg = RiskConfig(daily_loss_limit=1.0, max_open_positions=2, max_portfolio_gamma=0.05)
    rm = RiskManager(cfg)

    c1 = _FakeCand(max_loss=0.30, gamma=0.01)
    c2 = _FakeCand(max_loss=0.30, gamma=0.01)
    c3 = _FakeCand(max_loss=0.50, gamma=0.01)  # would bust max_positions

    for cand, label in [(c1, "trade-1"), (c2, "trade-2"), (c3, "trade-3")]:
        chk = rm.check(cand, "2026-06-26")
        if chk.approved:
            rm.record_trade(cand, "2026-06-26")
            print(f"{label}: APPROVED  status={rm.status()}")
        else:
            print(f"{label}: VETOED    reasons={chk.vetoes}")
