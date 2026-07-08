"""
config_loader.py
================
YAML run-config overlays for controlled experiments (the feature-impact
workflow's config system). The system's canonical configuration remains the
Python dataclasses (EngineConfig, ClassifierConfig, ...); a YAML overlay only
DESCRIBES a delta from those defaults, so two overlay files define a clean
baseline-vs-variant comparison without forking code.

Schema
------
    name: with_channels                # experiment label
    description: free text
    mtf:
      disabled_vars: [bb_width, ...]   # mtf_matrix variables to switch OFF
    overrides:                         # dot-notation dataclass overrides
      gate.min_gex_pct_rank: 0.65      # gate.* / selector.* / rnd.*  -> EngineConfig
      classifier.min_dominant_confidence: 55  # classifier.* -> ClassifierConfig

All sections are optional; an empty file is the pure default configuration.

NOT financial advice.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Optional

from decision_engine import EngineConfig
from regime_classifier import ClassifierConfig


@dataclass
class RunConfig:
    """One resolved experiment configuration."""
    name: str = "default"
    description: str = ""
    engine_cfg: EngineConfig = field(default_factory=EngineConfig)
    classifier_cfg: Optional[ClassifierConfig] = None
    disabled_vars: frozenset = frozenset()
    source_path: str = ""


def _apply_overrides(overrides: dict) -> tuple[EngineConfig, Optional[ClassifierConfig]]:
    """Split dot-notation overrides into an EngineConfig (gate./selector./rnd.)
    and an optional ClassifierConfig (classifier.)."""
    from optimizer import _build_engine_cfg   # canonical dot-path applier

    engine_params = {}
    classifier_kw = {}
    for path, val in (overrides or {}).items():
        prefix, _, key = str(path).partition(".")
        if prefix == "classifier":
            if key not in {f.name for f in dataclasses.fields(ClassifierConfig)}:
                raise ValueError(f"Unknown ClassifierConfig field: {key!r}")
            classifier_kw[key] = val
        elif prefix in ("gate", "selector", "rnd"):
            engine_params[path] = val
        else:
            raise ValueError(f"Unknown override prefix: {prefix!r} in {path!r} "
                             "(expected gate./selector./rnd./classifier.)")

    engine_cfg = _build_engine_cfg(EngineConfig(), engine_params) \
        if engine_params else EngineConfig()
    classifier_cfg = ClassifierConfig(**classifier_kw) if classifier_kw else None
    return engine_cfg, classifier_cfg


def load_run_config(path: str) -> RunConfig:
    """Load a YAML overlay into a resolved RunConfig. Validates that every
    disabled variable actually exists in the mtf_matrix registry so a typo
    fails loudly instead of silently testing nothing."""
    import yaml
    from mtf_matrix import VARS

    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: top level must be a mapping")

    mtf = raw.get("mtf") or {}
    disabled = frozenset(mtf.get("disabled_vars") or ())
    known = {v.name for v in VARS}
    unknown = disabled - known
    if unknown:
        raise ValueError(f"{path}: unknown mtf variables in disabled_vars: "
                         f"{sorted(unknown)}")

    engine_cfg, classifier_cfg = _apply_overrides(raw.get("overrides") or {})

    return RunConfig(
        name=str(raw.get("name") or "default"),
        description=str(raw.get("description") or ""),
        engine_cfg=engine_cfg,
        classifier_cfg=classifier_cfg,
        disabled_vars=disabled,
        source_path=path,
    )
