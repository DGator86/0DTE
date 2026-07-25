"""Mechanical champion config loader + engine override applier.

Ownership: SPY-DER produces champion/challenger JSON under
``/var/lib/spy-der/configs/``. 0DTE only applies overrides to deterministic
engine dataclasses. No diagnose / hypothesize / promote / learn logic.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

from decision_engine import EngineConfig
from gate_scorer import GateConfig
from regime_classifier import ClassifierConfig
from rnd_extractor import RNDConfig
from spread_selector import SelectorConfig

log = logging.getLogger("integrations.spy_der.champion_reader")

_PREFIX_TO_CLS = {
    "gate": GateConfig,
    "selector": SelectorConfig,
    "rnd": RNDConfig,
    "classifier": ClassifierConfig,
}

SIZE_MULT_KEY = "size_mult"

DEFAULT_SPYDER_CHAMPION = "/var/lib/spy-der/configs/champion.json"
DEFAULT_LEGACY_CHAMPION = "/var/lib/zerodte/configs/champion.json"


def validate_overrides(overrides: dict, allow_classifier: bool = True) -> None:
    """Fail loudly on unknown prefixes or dataclass fields."""
    for path in (overrides or {}):
        prefix, _, key = str(path).partition(".")
        cls = _PREFIX_TO_CLS.get(prefix)
        if cls is None or not key:
            raise ValueError(
                f"Unknown override path: {path!r} "
                f"(expected gate./selector./rnd./classifier. + field)"
            )
        if prefix == "classifier" and not allow_classifier:
            raise ValueError(
                f"classifier.* overrides are not allowed here: {path!r}"
            )
        if key not in {f.name for f in dataclasses.fields(cls)}:
            raise ValueError(f"Unknown {cls.__name__} field: {key!r} in {path!r}")


def build_engine_cfg(base: EngineConfig, params: dict) -> EngineConfig:
    """Apply a flat dot-notation param dict on top of a base EngineConfig."""
    gate_kw: dict = {}
    sel_kw: dict = {}
    rnd_kw: dict = {}
    for path, val in (params or {}).items():
        prefix, _, key = str(path).partition(".")
        if prefix == "gate":
            gate_kw[key] = val
        elif prefix == "selector":
            sel_kw[key] = val
        elif prefix == "rnd":
            rnd_kw[key] = val
        else:
            raise ValueError(f"Unknown param prefix: {prefix!r} in {path!r}")

    gate = dataclasses.replace(base.gate, **gate_kw) if gate_kw else base.gate
    sel = dataclasses.replace(base.selector, **sel_kw) if sel_kw else base.selector
    rnd = dataclasses.replace(base.rnd, **rnd_kw) if rnd_kw else base.rnd
    return EngineConfig(rnd=rnd, selector=sel, gate=gate)


def apply_overrides(
    overrides: dict,
    base_engine: Optional[EngineConfig] = None,
    base_classifier: Optional[ClassifierConfig] = None,
) -> tuple[EngineConfig, Optional[ClassifierConfig]]:
    validate_overrides(overrides)
    engine_params: dict = {}
    classifier_kw: dict = {}
    for path, val in (overrides or {}).items():
        prefix, _, key = str(path).partition(".")
        if prefix == "classifier":
            classifier_kw[key] = val
        else:
            engine_params[path] = val

    engine_cfg = build_engine_cfg(base_engine or EngineConfig(), engine_params)
    classifier_cfg = base_classifier
    if classifier_kw:
        classifier_cfg = dataclasses.replace(
            base_classifier or ClassifierConfig(), **classifier_kw
        )
    return engine_cfg, classifier_cfg


def validate_regime_overrides(regime_overrides: dict) -> None:
    if not isinstance(regime_overrides or {}, dict):
        raise ValueError("regime_overrides must be a mapping")
    for regime, block in (regime_overrides or {}).items():
        if not isinstance(block, dict):
            raise ValueError(f"regime_overrides[{regime!r}] must be a mapping")
        engine_keys = {k: v for k, v in block.items() if k != SIZE_MULT_KEY}
        validate_overrides(engine_keys, allow_classifier=False)
        if SIZE_MULT_KEY in block:
            sm = block[SIZE_MULT_KEY]
            if not isinstance(sm, (int, float)) or sm < 0:
                raise ValueError(
                    f"regime_overrides[{regime!r}].{SIZE_MULT_KEY} must be a "
                    f"non-negative number, got {sm!r}"
                )


def engine_cfg_for_regime(
    base: EngineConfig,
    regime_overrides: dict,
    regime: Optional[str],
) -> tuple[EngineConfig, float]:
    block = (regime_overrides or {}).get(regime or "unknown")
    if not block:
        return base, 1.0
    engine_keys = {k: v for k, v in block.items() if k != SIZE_MULT_KEY}
    cfg = build_engine_cfg(base, engine_keys) if engine_keys else base
    return cfg, float(block.get(SIZE_MULT_KEY, 1.0))


@dataclass
class ConfigRecord:
    config_id: str = ""
    created_at: str = ""
    parent_id: Optional[str] = None
    label: str = ""
    overrides: dict = field(default_factory=dict)
    regime_overrides: dict = field(default_factory=dict)
    optimizer: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    promotion_reason: str = ""
    author: str = ""
    status: str = "candidate"

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ConfigRecord":
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"Unknown config record keys: {sorted(unknown)}")
        return cls(**{k: v for k, v in d.items() if k in known})

    def engine_cfg(
        self,
        base_engine: Optional[EngineConfig] = None,
        base_classifier: Optional[ClassifierConfig] = None,
    ) -> tuple[EngineConfig, Optional[ClassifierConfig]]:
        return apply_overrides(self.overrides, base_engine, base_classifier)


def load_config(path: str) -> ConfigRecord:
    with open(path, encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: top level must be a JSON object")
    record = ConfigRecord.from_dict(raw)
    validate_overrides(record.overrides)
    validate_regime_overrides(record.regime_overrides)
    return record


@dataclass
class ChampionConfig:
    record: ConfigRecord
    engine_cfg: EngineConfig
    classifier_cfg: Optional[ClassifierConfig]
    regime_overrides: dict
    source_path: str


def resolve_champion_path(explicit: Optional[str] = None) -> Optional[str]:
    """Prefer SPY-DER champion, then legacy 0DTE paths, then repo configs/."""
    if explicit == "":
        return None
    candidates: list[str] = []
    if explicit:
        candidates.append(explicit)
    env = os.environ.get("SPY_DER_CHAMPION_PATH", "").strip()
    if env:
        candidates.append(env)
    candidates.extend(
        [
            DEFAULT_SPYDER_CHAMPION,
            DEFAULT_LEGACY_CHAMPION,
            os.path.join("configs", "champion.json"),
        ]
    )
    for path in candidates:
        if path and os.path.isfile(path):
            return path
    return explicit  # may be a missing path the caller wants to check


def load_champion_file(path: str) -> ChampionConfig:
    record = load_config(path)
    engine_cfg, classifier_cfg = record.engine_cfg()
    return ChampionConfig(
        record=record,
        engine_cfg=engine_cfg,
        classifier_cfg=classifier_cfg,
        regime_overrides=record.regime_overrides,
        source_path=path,
    )


def load_champion(configs_dir: Optional[str] = None) -> Optional[ChampionConfig]:
    """Load champion from configs_dir/champion.json or the cutover search path."""
    if configs_dir is not None:
        path = os.path.join(configs_dir, "champion.json")
        if not os.path.isfile(path):
            return None
        return load_champion_file(path)
    path = resolve_champion_path(None)
    if not path or not os.path.isfile(path):
        return None
    return load_champion_file(path)
