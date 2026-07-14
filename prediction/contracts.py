"""
prediction/contracts.py
=======================
Core data contract of Prediction Engine V2: the PredictionBundle
(docs/PREDICTION_ENGINE_V2_HANDOFF.md §6).

The bundle is the single forecast object every downstream layer consumes.
Contract rules enforced here:

  * every probability is in [0, 1] or None (None = required inputs
    unavailable — never a silent neutral value);
  * returns are decimal LOG returns (ln(future/current)) — one convention
    repo-wide (§6.2) — and None when the horizon extends past the close;
  * the bundle must NOT receive policy outputs (selected structure/family/
    strikes, conviction, gate result, candidate score). There are simply no
    fields for them (§6.3) — the forecast is created before policy runs.

NOT financial advice.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Optional

# Fields validated as probabilities / bounded quality scores (0..1 or None).
_BOUNDED_PREFIXES = ("p_",)
_BOUNDED_FIELDS = (
    "uncertainty", "data_quality", "feature_coverage",
    "ood_score", "ood_percentile", "calibration_support",
)


@dataclass(frozen=True)
class PredictionBundle:
    snapshot_id: str
    ts: str
    session_date: str
    symbol: str

    # Direction probabilities
    p_up_5m: Optional[float] = None
    p_up_15m: Optional[float] = None
    p_up_30m: Optional[float] = None
    p_up_60m: Optional[float] = None
    p_up_close: Optional[float] = None

    # Continuous return forecasts (decimal LOG returns)
    expected_return_15m: Optional[float] = None
    expected_return_30m: Optional[float] = None
    expected_return_60m: Optional[float] = None
    expected_return_close: Optional[float] = None
    return_q10_30m: Optional[float] = None
    return_q50_30m: Optional[float] = None
    return_q90_30m: Optional[float] = None
    return_q10_close: Optional[float] = None
    return_q50_close: Optional[float] = None
    return_q90_close: Optional[float] = None

    # Volatility and range
    expected_realized_move_30m: Optional[float] = None
    expected_realized_move_close: Optional[float] = None
    p_range_survive_15m: Optional[float] = None
    p_range_survive_30m: Optional[float] = None
    p_range_survive_60m: Optional[float] = None
    p_range_survive_close: Optional[float] = None

    # Structural barrier events
    p_touch_call_wall_30m: Optional[float] = None
    p_touch_put_wall_30m: Optional[float] = None
    p_touch_gamma_flip_30m: Optional[float] = None
    p_touch_call_wall_close: Optional[float] = None
    p_touch_put_wall_close: Optional[float] = None
    p_cross_gamma_flip_close: Optional[float] = None

    # First-passage ordering
    p_call_wall_first: Optional[float] = None
    p_put_wall_first: Optional[float] = None
    p_neither_wall_close: Optional[float] = None

    # Model quality
    uncertainty: Optional[float] = None
    data_quality: Optional[float] = None
    feature_coverage: Optional[float] = None
    feature_version: str = ""
    model_versions: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)

    # V3 Part 1 observation-specific uncertainty (optional; safe defaults)
    uncertainty_components: dict = field(default_factory=dict)
    uncertainty_reasons: tuple = ()
    ood_score: Optional[float] = None
    ood_percentile: Optional[float] = None
    calibration_support: Optional[float] = None
    ensemble_size: Optional[int] = None

    # V3 Part 2 forecast extensions (optional; safe defaults — §34)
    regime_probabilities: dict = field(default_factory=dict)
    regime_uncertainty: Optional[float] = None
    dominant_regime: Optional[str] = None
    return_distributions: dict = field(default_factory=dict)
    competing_risk_forecasts: dict = field(default_factory=dict)
    path_forecasts: dict = field(default_factory=dict)
    ensemble_forecasts: dict = field(default_factory=dict)
    structural_state_version: Optional[str] = None
    forecast_model_group_version: Optional[str] = None

    def __post_init__(self):
        # Normalize uncertainty_reasons to a tuple for frozen dataclass callers
        # that may pass a list via from_dict.
        if isinstance(self.uncertainty_reasons, list):
            object.__setattr__(self, "uncertainty_reasons",
                               tuple(self.uncertainty_reasons))
        for f in dataclasses.fields(self):
            if not (f.name.startswith(_BOUNDED_PREFIXES)
                    or f.name in _BOUNDED_FIELDS):
                continue
            v = getattr(self, f.name)
            if v is None:
                continue
            if not isinstance(v, (int, float)) or not (0.0 <= float(v) <= 1.0):
                raise ValueError(
                    f"PredictionBundle.{f.name} must be in [0, 1] or None, "
                    f"got {v!r}")
        # Component values that are present must also be in [0, 1]
        for k, v in (self.uncertainty_components or {}).items():
            if v is None:
                continue
            if not isinstance(v, (int, float)) or not (0.0 <= float(v) <= 1.0):
                raise ValueError(
                    f"PredictionBundle.uncertainty_components[{k!r}] must be "
                    f"in [0, 1] or None, got {v!r}")
        # Regime uncertainty is also [0,1] when present
        if self.regime_uncertainty is not None:
            ru = float(self.regime_uncertainty)
            if not (0.0 <= ru <= 1.0):
                raise ValueError(
                    f"regime_uncertainty must be in [0,1] or None, got {ru!r}")
        # Regime probabilities when present must be bounded (not forced to sum
        # here — calibrator guarantees that at production time)
        for k, v in (self.regime_probabilities or {}).items():
            if v is None:
                continue
            if not isinstance(v, (int, float)) or not (0.0 <= float(v) <= 1.0):
                raise ValueError(
                    f"regime_probabilities[{k!r}] must be in [0,1], got {v!r}")

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PredictionBundle":
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})
