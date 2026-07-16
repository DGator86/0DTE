from __future__ import annotations

import os
from dataclasses import dataclass


def _bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw not in (None, "") else default


def _float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw not in (None, "") else default


@dataclass(frozen=True)
class GrokConfig:
    """Runtime and risk policy for the Grok paper trader.

    All defaults fail closed. Merely installing the code never calls xAI and
    never adds a paper order; both ``enabled`` and ``submission_enabled`` must
    be explicitly set in the VPS secrets file.
    """

    enabled: bool = False
    submission_enabled: bool = False
    paper_only: bool = True
    api_key: str = ""
    base_url: str = "https://api.x.ai/v1"
    model: str = "grok-4.5"
    reasoning_effort: str = "high"
    timeout_seconds: float = 120.0
    max_output_tokens: int = 6000
    max_tool_rounds: int = 5

    base_interval_seconds: int = 300
    position_interval_seconds: int = 90
    min_event_gap_seconds: int = 45
    event_spot_move_pct: float = 0.0015

    daily_soft_cap_usd: float = 5.0
    daily_hard_cap_usd: float = 8.0
    monthly_hard_cap_usd: float = 170.0
    max_cycles_per_day: int = 100
    input_price_per_million: float = 2.0
    output_price_per_million: float = 6.0

    allowed_symbols: tuple[str, ...] = ("SPY", "XSP")
    allowed_families: tuple[str, ...] = (
        "put_credit",
        "call_credit",
        "iron_condor",
        "iron_fly",
        "broken_wing",
    )
    max_risk_fraction: float = 0.05
    max_requested_risk_fraction: float = 0.05
    max_leg_spread_abs: float = 0.50
    max_leg_relative_spread: float = 0.75
    max_quote_age_seconds: float = 15.0
    entry_start_hour: int = 10
    entry_start_minute: int = 0
    entry_cutoff_hour: int = 15
    entry_cutoff_minute: int = 15
    max_raw_rows_per_tool: int = 250

    audit_db_path: str = "grok_audit.sqlite"

    @classmethod
    def from_env(cls, *, default_audit_db: str | None = None) -> "GrokConfig":
        symbols = tuple(
            s.strip().upper()
            for s in os.getenv("GROK_ALLOWED_SYMBOLS", "SPY,XSP").split(",")
            if s.strip()
        )
        families = tuple(
            s.strip().lower()
            for s in os.getenv(
                "GROK_ALLOWED_FAMILIES",
                "put_credit,call_credit,iron_condor,iron_fly,broken_wing",
            ).split(",")
            if s.strip()
        )
        cfg = cls(
            enabled=_bool("GROK_ENABLED", False),
            submission_enabled=_bool("GROK_ORDER_SUBMISSION_ENABLED", False),
            paper_only=_bool("GROK_PAPER_ONLY", True),
            api_key=os.getenv("XAI_API_KEY", "").strip(),
            base_url=os.getenv("XAI_BASE_URL", "https://api.x.ai/v1").rstrip("/"),
            model=os.getenv("XAI_MODEL", "grok-4.5").strip(),
            reasoning_effort=os.getenv("GROK_REASONING_EFFORT", "high").strip().lower(),
            timeout_seconds=_float("GROK_TIMEOUT_SECONDS", 120.0),
            max_output_tokens=_int("GROK_MAX_OUTPUT_TOKENS", 6000),
            max_tool_rounds=_int("GROK_MAX_TOOL_ROUNDS", 5),
            base_interval_seconds=_int("GROK_BASE_INTERVAL_SECONDS", 300),
            position_interval_seconds=_int("GROK_POSITION_INTERVAL_SECONDS", 90),
            min_event_gap_seconds=_int("GROK_MIN_EVENT_GAP_SECONDS", 45),
            event_spot_move_pct=_float("GROK_EVENT_SPOT_MOVE_PCT", 0.0015),
            daily_soft_cap_usd=_float("GROK_DAILY_SOFT_CAP_USD", 5.0),
            daily_hard_cap_usd=_float("GROK_DAILY_HARD_CAP_USD", 8.0),
            monthly_hard_cap_usd=_float("GROK_MONTHLY_HARD_CAP_USD", 170.0),
            max_cycles_per_day=_int("GROK_MAX_DAILY_CYCLES", 100),
            input_price_per_million=_float("GROK_INPUT_PRICE_PER_MILLION", 2.0),
            output_price_per_million=_float("GROK_OUTPUT_PRICE_PER_MILLION", 6.0),
            allowed_symbols=symbols,
            allowed_families=families,
            max_risk_fraction=_float("GROK_MAX_RISK_FRACTION", 0.05),
            max_requested_risk_fraction=_float("GROK_MAX_REQUESTED_RISK_FRACTION", 0.05),
            max_leg_spread_abs=_float("GROK_MAX_LEG_SPREAD_ABS", 0.50),
            max_leg_relative_spread=_float("GROK_MAX_LEG_RELATIVE_SPREAD", 0.75),
            max_quote_age_seconds=_float("GROK_MAX_QUOTE_AGE_SECONDS", 15.0),
            entry_start_hour=_int("GROK_ENTRY_START_HOUR", 10),
            entry_start_minute=_int("GROK_ENTRY_START_MINUTE", 0),
            entry_cutoff_hour=_int("GROK_ENTRY_CUTOFF_HOUR", 15),
            entry_cutoff_minute=_int("GROK_ENTRY_CUTOFF_MINUTE", 15),
            max_raw_rows_per_tool=_int("GROK_MAX_RAW_ROWS_PER_TOOL", 250),
            audit_db_path=os.getenv(
                "GROK_AUDIT_DB",
                default_audit_db or "grok_audit.sqlite",
            ),
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if not self.paper_only:
            raise ValueError("Grok integration is paper-only; GROK_PAPER_ONLY must be 1")
        if self.enabled and not self.api_key:
            raise ValueError("GROK_ENABLED=1 requires XAI_API_KEY")
        if self.reasoning_effort not in {"low", "medium", "high"}:
            raise ValueError("GROK_REASONING_EFFORT must be low, medium, or high")
        if self.model != "grok-4.5":
            raise ValueError("This integration is pinned to XAI_MODEL=grok-4.5")
        if self.submission_enabled and not self.enabled:
            raise ValueError("GROK_ORDER_SUBMISSION_ENABLED requires GROK_ENABLED=1")
        if not 0 < self.max_risk_fraction <= 0.10:
            raise ValueError("GROK_MAX_RISK_FRACTION must be in (0, 0.10]")
        if self.daily_soft_cap_usd > self.daily_hard_cap_usd:
            raise ValueError("Grok soft cost cap cannot exceed hard cost cap")