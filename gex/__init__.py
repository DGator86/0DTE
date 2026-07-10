"""
gex
===
GEX measurement research program (docs/PREDICTION_ENGINE_V2_HANDOFF.md §16).

Parallel providers (PR 9):
  oi           — OI-only 0DTE baseline (matches spy0dte.build_gamma_map)
  weekly       — OI + nearest weeklies with DTE decay
  volume_proxy — intraday volume-weighted gamma (never falls back to OI)
  hybrid       — quality / time-of-day / vol-OI blend

All variants share gex.contracts.GEXSnapshot. They are observation-only:
MarketSnapshot / gates / selector continue to use the feed's OI baseline
until a variant passes promotion criteria.
"""
from gex.contracts import (
    GEXSnapshot, GexAssumption, GexDisagreement, GexVariantBundle, GexVariantId,
)
from gex.base import compute_all_variants, compute_disagreement

__all__ = [
    "GEXSnapshot", "GexAssumption", "GexDisagreement", "GexVariantBundle",
    "GexVariantId", "compute_all_variants", "compute_disagreement",
]
