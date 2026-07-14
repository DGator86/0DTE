"""
decision_stack/persistence.py
=============================
Persist unified decision graph rows (handoff §16).

NOT financial advice.
"""
from __future__ import annotations

from typing import Any, Optional


def persist_unified_decision(
    store: Any,
    record: Any,
    *,
    snapshot: Any = None,
    universe: Any = None,
    forecast: Any = None,
) -> None:
    """Best-effort persistence into PredictionStore-like backends."""
    if store is None:
        return
    if hasattr(store, "log_unified_decision"):
        store.log_unified_decision(record.to_dict() if hasattr(record, "to_dict")
                                   else dict(record))
    if snapshot is not None and hasattr(store, "log_canonical_snapshot"):
        store.log_canonical_snapshot(
            snapshot.to_dict() if hasattr(snapshot, "to_dict") else snapshot)
    if universe is not None and hasattr(store, "log_candidate_universe"):
        store.log_candidate_universe(
            universe.to_dict() if hasattr(universe, "to_dict") else universe)
    if forecast is not None and hasattr(store, "log_forecast_bundle"):
        payload = forecast.to_dict() if hasattr(forecast, "to_dict") else forecast
        store.log_forecast_bundle(payload)
