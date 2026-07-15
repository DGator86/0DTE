"""
adaptive_learning/config_store.py
=================================
Champion / challenger configuration records and the ONE shared implementation
of "apply dot-notation overrides to the engine dataclasses".

A config record is a JSON file describing a delta from the dataclass defaults
(the canonical configuration remains the Python dataclasses, exactly like
config_loader.py's YAML overlays):

    {
      "config_id": "a3f1...",             # uuid4 hex
      "created_at": "2026-07-09T...Z",
      "parent_id": null,                  # config this one was derived from
      "label": "gate_fix",
      "overrides": {                      # flat dot-notation, optimizer keys
        "gate.max_adx": 24.0,
        "selector.min_ev": 0.01
      },
      "regime_overrides": {               # per-dominant-regime engine deltas
        "compression": {"gate.max_adx": 18.0},
        "trend":       {"selector.min_ev": 0.02},
        "unknown":     {"size_mult": 0.25}
      },
      "optimizer": {...},                 # how it was found (metadata)
      "metrics": {...},                   # scores at creation time
      "promotion_reason": "...",
      "author": "learner",
      "status": "candidate"
    }

Directory layout (relative to a configs_dir, default "configs"):

    configs/champion.json                  the ONE live config
    configs/candidates/<date>_<label>.json learner output, never live
    configs/promoted/pending_review.json   passed validation, awaiting a human
    configs/archive/                       previous champions

Only adaptive_learning.promoter (the human CLI) writes champion.json.

NOT financial advice.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import json
import os
import uuid
from dataclasses import dataclass, field
from typing import Optional

from decision_engine import EngineConfig
from gate_scorer import GateConfig
from spread_selector import SelectorConfig
from rnd_extractor import RNDConfig
from regime_classifier import ClassifierConfig

_PREFIX_TO_CLS = {
    "gate": GateConfig,
    "selector": SelectorConfig,
    "rnd": RNDConfig,
    "classifier": ClassifierConfig,
}

# Special regime-override key: scales the final position size for ticks whose
# dominant regime matches. Everything else must be a dot-notation engine key.
SIZE_MULT_KEY = "size_mult"


# --------------------------------------------------------------------------- #
# Override validation + application (shared by optimizer / config_loader /    #
# the live champion loader — one implementation, no drift)                    #
# --------------------------------------------------------------------------- #
def validate_overrides(overrides: dict, allow_classifier: bool = True) -> None:
    """Fail loudly on unknown prefixes or dataclass fields."""
    for path in (overrides or {}):
        prefix, _, key = str(path).partition(".")
        cls = _PREFIX_TO_CLS.get(prefix)
        if cls is None or not key:
            raise ValueError(
                f"Unknown override path: {path!r} "
                f"(expected gate./selector./rnd./classifier. + field)")
        if prefix == "classifier" and not allow_classifier:
            raise ValueError(
                f"classifier.* overrides are not allowed here: {path!r}")
        if key not in {f.name for f in dataclasses.fields(cls)}:
            raise ValueError(f"Unknown {cls.__name__} field: {key!r} in {path!r}")


def build_engine_cfg(base: EngineConfig, params: dict) -> EngineConfig:
    """Apply a flat dot-notation param dict (gate./selector./rnd.) on top of a
    base EngineConfig. The canonical dot-path applier (optimizer delegates
    here)."""
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
    """Split validated dot-notation overrides into an EngineConfig and an
    optional ClassifierConfig, applied on top of the given bases."""
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
            base_classifier or ClassifierConfig(), **classifier_kw)
    return engine_cfg, classifier_cfg


def validate_regime_overrides(regime_overrides: dict) -> None:
    """Per-regime blocks may hold engine dot-keys plus the special size_mult.
    classifier.* is per-run, not per-tick, so it is rejected here."""
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
                    f"non-negative number, got {sm!r}")


def engine_cfg_for_regime(
    base: EngineConfig, regime_overrides: dict, regime: Optional[str],
) -> tuple[EngineConfig, float]:
    """Resolve the (EngineConfig, size_mult) pair for one dominant regime.
    Unlisted regimes get the base config and size_mult 1.0."""
    block = (regime_overrides or {}).get(regime or "unknown")
    if not block:
        return base, 1.0
    engine_keys = {k: v for k, v in block.items() if k != SIZE_MULT_KEY}
    cfg = build_engine_cfg(base, engine_keys) if engine_keys else base
    return cfg, float(block.get(SIZE_MULT_KEY, 1.0))


# --------------------------------------------------------------------------- #
# Config records                                                               #
# --------------------------------------------------------------------------- #
CONFIG_STATUSES = ("candidate", "pending_review", "promoted", "rejected", "archived")


@dataclass
class ConfigRecord:
    config_id: str = ""
    created_at: str = ""
    parent_id: Optional[str] = None
    label: str = ""
    overrides: dict = field(default_factory=dict)
    regime_overrides: dict = field(default_factory=dict)
    optimizer: dict = field(default_factory=dict)     # provenance metadata
    metrics: dict = field(default_factory=dict)       # scores at creation
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
        return cls(**d)

    def engine_cfg(
        self,
        base_engine: Optional[EngineConfig] = None,
        base_classifier: Optional[ClassifierConfig] = None,
    ) -> tuple[EngineConfig, Optional[ClassifierConfig]]:
        return apply_overrides(self.overrides, base_engine, base_classifier)


def new_candidate(
    overrides: dict,
    label: str = "",
    parent_id: Optional[str] = None,
    regime_overrides: Optional[dict] = None,
    optimizer: Optional[dict] = None,
    metrics: Optional[dict] = None,
    promotion_reason: str = "",
    author: str = "learner",
) -> ConfigRecord:
    """Validated challenger record; raises on unknown keys before anything is
    written to disk."""
    validate_overrides(overrides)
    validate_regime_overrides(regime_overrides or {})
    return ConfigRecord(
        config_id=uuid.uuid4().hex,
        created_at=dt.datetime.now(dt.timezone.utc).isoformat(),
        parent_id=parent_id,
        label=label,
        overrides=dict(overrides or {}),
        regime_overrides=dict(regime_overrides or {}),
        optimizer=dict(optimizer or {}),
        metrics=dict(metrics or {}),
        promotion_reason=promotion_reason,
        author=author,
        status="candidate",
    )


# --------------------------------------------------------------------------- #
# Filesystem layout                                                            #
# --------------------------------------------------------------------------- #
def champion_path(configs_dir: str = "configs") -> str:
    return os.path.join(configs_dir, "champion.json")


def candidates_dir(configs_dir: str = "configs") -> str:
    return os.path.join(configs_dir, "candidates")


def pending_review_path(configs_dir: str = "configs") -> str:
    return os.path.join(configs_dir, "promoted", "pending_review.json")


def archive_dir(configs_dir: str = "configs") -> str:
    return os.path.join(configs_dir, "archive")


def save_config(record: ConfigRecord, path: str) -> str:
    """Validate + atomically write one config record as JSON."""
    validate_overrides(record.overrides)
    validate_regime_overrides(record.regime_overrides)
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(record.to_dict(), f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)
    return path


def load_config(path: str) -> ConfigRecord:
    """Load + validate one config record. Raises (never silently degrades) on
    unknown keys — a typo'd champion must not trade on defaults unnoticed."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: top level must be a JSON object")
    record = ConfigRecord.from_dict(raw)
    validate_overrides(record.overrides)
    validate_regime_overrides(record.regime_overrides)
    return record


def candidate_file_path(record: ConfigRecord, configs_dir: str = "configs") -> str:
    date = (record.created_at or "")[:10] or dt.date.today().isoformat()
    label = record.label or record.config_id[:8]
    safe = "".join(c if (c.isalnum() or c in "-_") else "_" for c in label)
    return os.path.join(candidates_dir(configs_dir), f"{date}_{safe}.json")


def save_candidate(record: ConfigRecord, configs_dir: str = "configs") -> str:
    return save_config(record, candidate_file_path(record, configs_dir))


# --------------------------------------------------------------------------- #
# Champion loading (the live path)                                             #
# --------------------------------------------------------------------------- #
@dataclass
class ChampionConfig:
    record: ConfigRecord
    engine_cfg: EngineConfig
    classifier_cfg: Optional[ClassifierConfig]
    regime_overrides: dict
    source_path: str


def load_champion(configs_dir: str = "configs") -> Optional[ChampionConfig]:
    """Resolve configs/champion.json into live-ready configs. Returns None when
    no champion exists (dataclass defaults apply); raises on an invalid file."""
    path = champion_path(configs_dir)
    if not os.path.isfile(path):
        return None
    record = load_config(path)
    engine_cfg, classifier_cfg = record.engine_cfg()
    return ChampionConfig(
        record=record,
        engine_cfg=engine_cfg,
        classifier_cfg=classifier_cfg,
        regime_overrides=record.regime_overrides,
        source_path=path,
    )


# --------------------------------------------------------------------------- #
# Registry-compatible rule-config artifacts (UNIFIED handoff §11.7)           #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class RuleConfigArtifact:
    """Versioned V1 rule configuration referenced by DeploymentBundle."""

    rule_config_id: str
    overrides: dict
    regime_overrides: dict
    configuration_hash: str
    parent_id: Optional[str] = None
    status: str = "candidate"
    metrics: dict = field(default_factory=dict)
    label: str = ""
    author: str = ""

    def to_dict(self) -> dict:
        return {
            "rule_config_id": self.rule_config_id,
            "overrides": dict(self.overrides),
            "regime_overrides": dict(self.regime_overrides),
            "configuration_hash": self.configuration_hash,
            "parent_id": self.parent_id,
            "status": self.status,
            "metrics": dict(self.metrics),
            "label": self.label,
            "author": self.author,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RuleConfigArtifact":
        return cls(
            rule_config_id=str(d["rule_config_id"]),
            overrides=dict(d.get("overrides") or {}),
            regime_overrides=dict(d.get("regime_overrides") or {}),
            configuration_hash=str(d.get("configuration_hash") or ""),
            parent_id=d.get("parent_id"),
            status=str(d.get("status") or "candidate"),
            metrics=dict(d.get("metrics") or {}),
            label=str(d.get("label") or ""),
            author=str(d.get("author") or ""),
        )

    def to_config_record(self) -> ConfigRecord:
        return ConfigRecord(
            config_id=self.rule_config_id,
            parent_id=self.parent_id,
            label=self.label,
            overrides=dict(self.overrides),
            regime_overrides=dict(self.regime_overrides),
            metrics=dict(self.metrics),
            author=self.author,
            status=self.status if self.status in CONFIG_STATUSES else "candidate",
        )


def _rule_config_hash(overrides: dict, regime_overrides: dict) -> str:
    import hashlib
    import json as _json
    payload = {"overrides": overrides, "regime_overrides": regime_overrides}
    return hashlib.sha256(
        _json.dumps(payload, sort_keys=True, separators=(",", ":"),
                    default=str).encode("utf-8")).hexdigest()


def new_rule_config_artifact(
    *,
    overrides: dict,
    regime_overrides: Optional[dict] = None,
    parent_id: Optional[str] = None,
    metrics: Optional[dict] = None,
    author: str = "learner",
    status: str = "candidate",
    label: str = "",
    rule_config_id: Optional[str] = None,
) -> RuleConfigArtifact:
    """Produce a versioned rule-config candidate (never auto-promotes)."""
    validate_overrides(overrides)
    validate_regime_overrides(regime_overrides or {})
    rid = rule_config_id or uuid.uuid4().hex
    ch = _rule_config_hash(overrides, regime_overrides or {})
    return RuleConfigArtifact(
        rule_config_id=rid,
        overrides=dict(overrides),
        regime_overrides=dict(regime_overrides or {}),
        configuration_hash=ch,
        parent_id=parent_id,
        status=status,
        metrics=dict(metrics or {}),
        label=label,
        author=author,
    )


def export_champion_compatibility(
    artifact: RuleConfigArtifact,
    configs_dir: str = "configs",
) -> str:
    """
    Write configs/champion.json as a compatibility export.
    Primary deployment truth remains the DeploymentBundle pointer.
    """
    rec = artifact.to_config_record()
    rec.status = "promoted"
    path = champion_path(configs_dir)
    return save_config(rec, path)


# --------------------------------------------------------------------------- #
# Demo                                                                          #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import tempfile

    print("=" * 64)
    print("  config_store demo — candidate round-trip + regime overrides")
    print("=" * 64)
    rec = new_candidate(
        {"gate.max_adx": 24.0, "selector.min_ev": 0.01},
        label="gate_fix",
        regime_overrides={
            "compression": {"gate.max_adx": 18.0},
            "unknown": {"size_mult": 0.25},
        },
        promotion_reason="gate_effectiveness_reversed",
    )
    with tempfile.TemporaryDirectory() as d:
        path = save_candidate(rec, configs_dir=d)
        back = load_config(path)
        eng, clf = back.engine_cfg()
        print(f"  saved -> {os.path.relpath(path, d)}")
        print(f"  config_id={back.config_id[:8]}  gate.max_adx={eng.gate.max_adx}  "
              f"selector.min_ev={eng.selector.min_ev}")
        comp_cfg, comp_sm = engine_cfg_for_regime(eng, back.regime_overrides,
                                                  "compression")
        unk_cfg, unk_sm = engine_cfg_for_regime(eng, back.regime_overrides,
                                                None)
        print(f"  compression: gate.max_adx={comp_cfg.gate.max_adx}  size_mult={comp_sm}")
        print(f"  unknown:     gate.max_adx={unk_cfg.gate.max_adx}  size_mult={unk_sm}")
        try:
            validate_overrides({"gate.not_a_field": 1})
        except ValueError as e:
            print(f"  rejected bad key as expected: {e}")
    print("=" * 64)
