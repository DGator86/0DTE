"""
prediction/part2_shadow.py
==========================
Part 2 shadow-mode forecast sequence (V3 Part 2 §37, PR 16).

Builds StructuralState → regime probs → mixture/competing/path/ensemble
attachments onto a PredictionBundle, persisting each stage. Failures are
recorded structurally and do not crash the legacy loop.

Research / shadow only — no live order routing.

NOT financial advice.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from prediction.contracts import PredictionBundle
from prediction.structural_state import (
    STRUCTURAL_STATE_VERSION,
    StructuralState,
    StructuralStateBuilder,
)


@dataclass
class Part2ShadowResult:
    bundle: PredictionBundle
    structural_state: Optional[StructuralState]
    errors: list[dict] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)


def run_part2_shadow_tick(
    *,
    base_bundle: PredictionBundle,
    spot: float,
    symbol: str,
    ts: str,
    current_sources: dict,
    historical_states: list | None = None,
    expected_remaining_move: float | None = None,
    store: Any = None,
    regime_predict: Optional[Callable[[dict], Any]] = None,
    feature_row: Optional[dict] = None,
    mode: str = "shadow",
) -> Part2ShadowResult:
    """
    Execute the Part 2 shadow sequence with failure isolation (§37.2).

    Optional callables are injected so unit tests need not train full models.
    """
    errors: list[dict] = []
    diagnostics: dict = {"mode": mode}
    structural = None
    regime_probs: dict = {}
    regime_unc = None
    dominant = None

    # 1–3. Structural state
    try:
        structural = StructuralStateBuilder().build(
            ts=ts,
            symbol=symbol,
            spot=spot,
            expected_remaining_move=expected_remaining_move,
            current_sources=current_sources or {},
            historical_states=historical_states or [],
        )
        if store is not None:
            store.log_structural_state(
                base_bundle.snapshot_id, structural.to_dict())
    except Exception as exc:  # structured record — do not swallow silently
        errors.append({"stage": "structural_state", "error": repr(exc)})

    # 4. Regime probabilities
    if regime_predict is not None and feature_row is not None:
        try:
            rp = regime_predict(feature_row)
            if hasattr(rp, "as_dict"):
                regime_probs = rp.as_dict()
                regime_unc = float(rp.uncertainty)
                dominant = str(rp.dominant_regime)
                if store is not None:
                    store.log_regime_output(
                        base_bundle.snapshot_id,
                        getattr(rp, "model_version", "regime"),
                        rp.to_dict() if hasattr(rp, "to_dict") else regime_probs,
                        uncertainty=regime_unc, mode=mode,
                    )
            elif isinstance(rp, dict):
                regime_probs = dict(rp)
        except Exception as exc:
            errors.append({"stage": "regime_probabilities", "error": repr(exc)})

    # Attach Part 2 fields onto a new bundle (frozen dataclass)
    payload = base_bundle.to_dict()
    payload["regime_probabilities"] = regime_probs
    payload["regime_uncertainty"] = regime_unc
    payload["dominant_regime"] = dominant
    payload["structural_state_version"] = (
        structural.version if structural is not None
        else STRUCTURAL_STATE_VERSION)
    payload["forecast_model_group_version"] = payload.get(
        "forecast_model_group_version") or "v3.part2.shadow"
    if errors:
        diag = dict(payload.get("diagnostics") or {})
        diag["part2_errors"] = errors
        payload["diagnostics"] = diag
        # Increase uncertainty when components fail
        unc = payload.get("uncertainty")
        payload["uncertainty"] = min(
            1.0, (float(unc) if unc is not None else 0.5) + 0.1 * len(errors))

    try:
        bundle = PredictionBundle.from_dict(payload)
    except Exception as exc:
        errors.append({"stage": "bundle_attach", "error": repr(exc)})
        bundle = base_bundle

    diagnostics["n_errors"] = len(errors)
    return Part2ShadowResult(
        bundle=bundle,
        structural_state=structural,
        errors=errors,
        diagnostics=diagnostics,
    )
