"""
prediction/v3_part2_config.py
=============================
Load / validate configs/prediction_v3_part2.json (Part 2 §38).

Invalid configuration fails closed for the V3 component.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from prediction.structural_state import StructuralStateConfig

_DEFAULT_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "prediction_v3_part2.json"
)

_REQUIRED_TOP = (
    "structural_state",
    "regime_model",
    "mixture_experts",
    "competing_risk",
    "conformal",
    "path_model",
    "ensemble",
)


class Part2ConfigError(ValueError):
    """Invalid Part 2 configuration — fail closed."""


def load_part2_config(path: Optional[str | Path] = None) -> dict[str, Any]:
    p = Path(path) if path is not None else _DEFAULT_PATH
    try:
        raw = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise Part2ConfigError(f"cannot load Part 2 config from {p}: {exc}") from exc
    validate_part2_config(raw)
    return raw


def validate_part2_config(cfg: dict) -> None:
    if not isinstance(cfg, dict):
        raise Part2ConfigError("config must be a JSON object")
    for key in _REQUIRED_TOP:
        if key not in cfg:
            raise Part2ConfigError(f"missing top-level key: {key}")
    ss = cfg["structural_state"]
    if not isinstance(ss.get("velocity_windows"), list) or not ss["velocity_windows"]:
        raise Part2ConfigError("structural_state.velocity_windows must be a non-empty list")
    fo = ss.get("fallback_order")
    if not isinstance(fo, list) or not fo:
        raise Part2ConfigError("structural_state.fallback_order must be a non-empty list")
    for name in fo:
        if name not in ("hybrid", "oi", "volume", "weekly"):
            raise Part2ConfigError(f"unknown fallback source: {name!r}")
    rm = cfg["regime_model"]
    for k in ("minimum_sessions", "minimum_effective_sessions", "minimum_rows"):
        if not isinstance(rm.get(k), int) or rm[k] < 0:
            raise Part2ConfigError(f"regime_model.{k} must be a non-negative int")
    ens = cfg["ensemble"]
    mw = ens.get("maximum_component_weight")
    if not isinstance(mw, (int, float)) or not (0.0 < float(mw) <= 1.0):
        raise Part2ConfigError(
            "ensemble.maximum_component_weight must be in (0, 1]")


def structural_config_from_part2(cfg: Optional[dict] = None) -> StructuralStateConfig:
    raw = cfg if cfg is not None else load_part2_config()
    ss = raw["structural_state"]
    return StructuralStateConfig(
        velocity_windows=tuple(int(x) for x in ss["velocity_windows"]),
        stability_window_minutes=int(ss.get("stability_window_minutes", 15)),
        concentration_top_n=int(ss.get("concentration_top_n", 5)),
        epsilon=float(ss.get("epsilon", 1e-9)),
        fallback_order=tuple(str(x) for x in ss["fallback_order"]),
    )
