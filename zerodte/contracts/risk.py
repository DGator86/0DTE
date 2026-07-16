"""Deterministic operational, portfolio, and sizing contracts."""
from __future__ import annotations

from dataclasses import dataclass


RISK_SCHEMA = "risk.envelope.v1"


@dataclass(frozen=True)
class OperationalState:
    market_open: bool = False
    session_warmup: bool = False
    entry_locked: bool = False
    data_valid: bool = False
    broker_available: bool = False
    hard_vetoes: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def entries_allowed(self) -> bool:
        return (
            self.market_open
            and not self.session_warmup
            and not self.entry_locked
            and self.data_valid
            and self.broker_available
            and not self.hard_vetoes
        )


@dataclass(frozen=True)
class PortfolioState:
    account_id: str
    equity: float
    cash: float
    open_positions: int = 0
    daily_realized_pnl: float = 0.0
    daily_unrealized_pnl: float = 0.0
    portfolio_gamma: float = 0.0
    portfolio_delta: float = 0.0

    def __post_init__(self) -> None:
        if not self.account_id:
            raise ValueError("account_id is required")
        if self.equity < 0:
            raise ValueError("equity cannot be negative")
        if self.cash < 0:
            raise ValueError("cash cannot be negative")
        if self.open_positions < 0:
            raise ValueError("open_positions cannot be negative")


@dataclass(frozen=True)
class RiskEnvelope:
    approved: bool
    max_risk_dollars: float
    max_size_scalar: float
    remaining_daily_loss_budget: float
    remaining_position_slots: int
    hard_vetoes: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    schema_version: str = RISK_SCHEMA

    def __post_init__(self) -> None:
        if self.max_risk_dollars < 0:
            raise ValueError("max_risk_dollars cannot be negative")
        if not 0.0 <= self.max_size_scalar <= 1.0:
            raise ValueError("max_size_scalar must be within [0, 1]")
        if self.remaining_position_slots < 0:
            raise ValueError("remaining_position_slots cannot be negative")
        if self.hard_vetoes and self.approved:
            raise ValueError("risk envelope cannot be approved with hard vetoes")
        if not self.approved and self.max_size_scalar != 0.0:
            raise ValueError("rejected risk envelope must use max_size_scalar=0")

    @classmethod
    def rejected(cls, *reasons: str) -> "RiskEnvelope":
        return cls(
            approved=False,
            max_risk_dollars=0.0,
            max_size_scalar=0.0,
            remaining_daily_loss_budget=0.0,
            remaining_position_slots=0,
            hard_vetoes=tuple(reasons) or ("risk_rejected",),
        )
