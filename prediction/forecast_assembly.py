"""
prediction/forecast_assembly.py
===============================
Complete V3 forecast assembly from CanonicalSnapshot
(docs/UNIFIED_V1_V2_V3_HANDOFF.md §11.3 / PR3).

Uses PredictionBundle as the canonical ForecastBundle contract.
Component failures are explicit; never invents a neutral forecast silently.

NOT financial advice.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from prediction.contracts import PredictionBundle
from prediction.dataset import FEATURE_VERSION, LABEL_VERSION


@dataclass
class ForecastModelSet:
    """Optional loaded Part 2 / V2 model handles."""

    model_group: Any = None
    regime_model: Any = None
    competing_risk: Any = None
    path_model: Any = None
    ensemble: Any = None


def build_v3_forecast(
    *,
    snapshot: Any,
    runtime: Any = None,
    models: Optional[ForecastModelSet] = None,
    store: Any = None,
    mode: str = "shadow",
) -> PredictionBundle:
    """
    Assemble a complete forecast bundle from pre-decision information only.

    Stages (best-effort; failures recorded in diagnostics via model_versions
    / uncertainty inflation — never silent zero-fill of structural state):
      1 structural state (already on snapshot when available)
      2–14 regime / experts / conformal / risks / path / ensemble /
         uncertainty / OOD / persistence
    """
    models = models or ForecastModelSet()
    if runtime is not None and getattr(runtime, "artifacts", None) is not None:
        if runtime.artifacts.model_group is not None:
            models.model_group = runtime.artifacts.model_group

    snapshot_id = getattr(snapshot, "snapshot_id", "") or ""
    ts = getattr(snapshot, "ts", "") or ""
    session_date = getattr(snapshot, "session_date", "") or ""
    symbol = getattr(snapshot, "symbol", "SPY") or "SPY"
    feature_version = getattr(snapshot, "feature_version", FEATURE_VERSION)
    row = dict(getattr(snapshot, "standardized_features", {}) or {})
    if not row:
        row = dict(getattr(snapshot, "raw_features", {}) or {})

    diagnostics: dict = {"stages": [], "fallbacks": [], "errors": []}
    model_versions: dict = {}
    uncertainty = 0.25
    ood_score = 0.0
    data_quality = float(
        (getattr(snapshot, "quality", {}) or {}).get("data_quality") or 0.85)
    feature_coverage = float(
        (getattr(snapshot, "quality", {}) or {}).get("feature_coverage")
        or (len(row) / 40.0 if row else 0.0))
    feature_coverage = max(0.0, min(1.0, feature_coverage))

    # Stage 1: structural state present?
    structural = getattr(snapshot, "structural_state", None)
    diagnostics["stages"].append("structural_state")
    if structural is None:
        diagnostics["fallbacks"].append("structural_state_missing")
        uncertainty = min(1.0, uncertainty + 0.1)

    bundle: Optional[PredictionBundle] = None

    # Stages 2–6: trained model group when available
    group = models.model_group
    if group is not None:
        try:
            from prediction.training import build_prediction_bundle
            bundle = build_prediction_bundle(
                group, row,
                snapshot_id=snapshot_id,
                ts=ts,
                session_date=session_date,
                symbol=symbol,
            )
            model_versions.update(getattr(group, "model_versions", lambda: {})())
            diagnostics["stages"].append("trained_group")
        except Exception as exc:
            diagnostics["errors"].append({
                "stage": "trained_group",
                "exception_type": type(exc).__name__,
                "message": str(exc),
            })
            uncertainty = min(1.0, uncertainty + 0.2)
            if mode in ("candidate", "champion"):
                # Fail closed — return unavailable-labeled bundle.
                return _unavailable_bundle(
                    snapshot_id, ts, session_date, symbol,
                    feature_version, diagnostics, reason="trained_group_failed")

    # Heuristic baseline only when allowed
    if bundle is None:
        allow_heuristic = True
        if runtime is not None:
            allow_heuristic = bool(
                getattr(runtime.bundle, "allows_heuristic_fallback",
                        lambda: True)())
        if mode in ("candidate", "champion"):
            allow_heuristic = False
        if not allow_heuristic:
            return _unavailable_bundle(
                snapshot_id, ts, session_date, symbol,
                feature_version, diagnostics,
                reason="required_component_missing")
        try:
            from prediction.inference import heuristic_bundle_from_tick
            # Minimal snap-like adapter
            market = getattr(snapshot, "market", None)
            if market is not None:
                class _Snap:
                    pass
                snap = _Snap()
                snap.market = market
                bundle = heuristic_bundle_from_tick(
                    snap, {}, snapshot_id=snapshot_id, symbol=symbol)
                model_versions["bundle"] = "heuristic_baseline"
                diagnostics["fallbacks"].append("heuristic_baseline")
                diagnostics["stages"].append("heuristic")
                uncertainty = min(1.0, uncertainty + 0.15)
            else:
                return _unavailable_bundle(
                    snapshot_id, ts, session_date, symbol,
                    feature_version, diagnostics,
                    reason="no_market_for_heuristic")
        except Exception as exc:
            diagnostics["errors"].append({
                "stage": "heuristic",
                "exception_type": type(exc).__name__,
                "message": str(exc),
            })
            return _unavailable_bundle(
                snapshot_id, ts, session_date, symbol,
                feature_version, diagnostics,
                reason="heuristic_failed")

    # Attach Part 2 enrichment when possible
    try:
        from prediction.part2_shadow import run_part2_shadow_tick
        market = getattr(snapshot, "market", None)
        spot = float(getattr(market, "spot", 0.0) or 0.0) if market else 0.0
        sources = dict(getattr(snapshot, "structural_sources", {}) or {})
        if spot > 0:
            part2 = run_part2_shadow_tick(
                base_bundle=bundle,
                spot=spot,
                symbol=symbol,
                ts=ts,
                current_sources=sources,
                feature_row=row,
                mode=mode,
                store=store,
            )
            bundle = part2.bundle
            if part2.errors:
                diagnostics["errors"].extend(part2.errors)
                uncertainty = min(1.0, uncertainty + 0.05 * len(part2.errors))
            diagnostics["stages"].append("part2")
            model_versions["part2"] = "v3.part2.shadow"
    except Exception as exc:
        diagnostics["errors"].append({
            "stage": "part2",
            "exception_type": type(exc).__name__,
            "message": str(exc),
        })
        uncertainty = min(1.0, uncertainty + 0.1)

    # Inflate uncertainty / OOD onto bundle via replace
    from dataclasses import replace
    merged_versions = {
        **dict(getattr(bundle, "model_versions", {}) or {}),
        **model_versions,
        "assembly": "v3.forecast_assembly",
        "feature_version": feature_version,
        "label_version": LABEL_VERSION,
    }
    merged_diag = {
        **dict(getattr(bundle, "diagnostics", {}) or {}),
        "forecast_assembly": diagnostics,
    }
    bundle = replace(
        bundle,
        uncertainty=max(float(getattr(bundle, "uncertainty", 0) or 0),
                        uncertainty),
        ood_score=max(float(getattr(bundle, "ood_score", 0) or 0), ood_score),
        data_quality=min(float(getattr(bundle, "data_quality", 1) or 1),
                         data_quality),
        feature_coverage=feature_coverage,
        model_versions=merged_versions,
        diagnostics=merged_diag,
    )

    if store is not None and hasattr(store, "log_prediction"):
        try:
            store.log_prediction(bundle.to_dict())
        except Exception:
            pass

    return bundle


def _unavailable_bundle(
    snapshot_id, ts, session_date, symbol, feature_version, diagnostics,
    *, reason: str,
) -> PredictionBundle:
    return PredictionBundle(
        snapshot_id=str(snapshot_id),
        ts=str(ts),
        session_date=str(session_date),
        symbol=str(symbol),
        uncertainty=1.0,
        data_quality=0.0,
        feature_coverage=0.0,
        ood_score=1.0,
        model_versions={
            "assembly": "unavailable",
            "reason": reason,
            "feature_version": feature_version,
            "diagnostics": diagnostics,
        },
    )
