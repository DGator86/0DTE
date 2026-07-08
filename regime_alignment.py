"""
regime_alignment.py
===================
Position-relative Regime Alignment Score (RAS).

Consumes absolute regime output (RegimeState + TradeIntent) and evaluates
whether the current market environment still cooperates with an open position's
thesis. Does not re-implement regime classification — regime_classifier.py and
mtf_matrix.py stay the single source of absolute regime truth; THIS module is
the single source of position-RELATIVE alignment.

Public API
----------
compute_regime_alignment(regime, intent, market, position_ctx, cfg) -> RASResult
    The main entry point: score how well the environment still supports an
    open position. (`compute_ras` is the same function; both names are kept.)
build_entry_snapshot(regime, intent, market, structure_class, structure)
    Freeze the regime/dealer state at trade entry into an EntrySnapshot.
entry_snapshot_to_dict / entry_snapshot_from_dict
    JSON-safe round-trip for persisting the snapshot in a position's entry_ctx.
position_context_from_entry_ctx(position_id, entry_ctx) -> PositionContext
    Rebuild the PositionContext (entry snapshot + bias + previous EMA score)
    from a stored entry_ctx dict.

Score semantics
---------------
Each component scorer returns a raw value in [-1, +1] plus a human-readable
note (every RASResult carries the full breakdown for journaling):

  direction_alignment   Is the matrix direction bias still with the position?
  fast_momentum         Raw fast (1m/5m/15m) composite vs the position — the
                        early-warning channel; the blended bias is 60% slow
                        and structurally late at intraday turns. Skipped when
                        the intent predates bias_fast.
  matrix_alignment      Has the exec/context matrix cell moved for/against it?
  gamma_alignment       Dealer gamma surface (flip, net GEX) vs the thesis.
  veto_escalation       New vetoes since entry that undermine this structure.
  confidence_erosion    Has the structure-relevant regime confidence decayed?
  regime_flip           Permitted engine / dominant regime turned hostile.

The weighted mean of the components maps to a score in [-100, +100], smoothed
by an EMA (RASConfig.ema_alpha) so one noisy tick cannot trigger an action.
Actions from the EMA score: ok > warning (<= warning_threshold, default -30)
> tighten (<= tighten_threshold, default -50) > exit (<= exit_threshold,
default -70).

Paper-trading safety
--------------------
RAS actively manages PAPER positions only — no real orders exist anywhere in
this system. RASConfig.exit_enabled defaults to True: an "exit" action closes
the paper position (exit reason "ras_invalidate") and "tighten" narrows the
trailing stop. To run observation-only, pass RASConfig(exit_enabled=False)
(and PaperConfig(ras_exit_enabled=False)); shadow_runner exposes this as the
--no-ras-exit flag. Either flag alone is sufficient to suppress exits: the
action is downgraded to "warning" in compute_ras / PositionMonitor.evaluate
when exit_enabled is off, and the broker additionally checks its own
ras_exit_enabled before closing. Every evaluation is journaled
(journal.Journal.log_ras) with the full component breakdown so score moves
are explainable and thresholds can be recalibrated from data.

NOT financial advice.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Optional

from decision_matrix import PREMIUM_STRUCTURES, TradeIntent
from gate_scorer import MarketSnapshot
from regime_classifier import RegimeState
from spread_selector import DEBIT_FAMILIES

VOL_STRUCTURES = frozenset({"STG"})

DEFAULT_WEIGHTS = {
    "direction_alignment": 1.5,
    "fast_momentum": 1.3,
    "matrix_alignment": 1.2,
    "gamma_alignment": 1.5,
    "veto_escalation": 1.0,
    "confidence_erosion": 0.8,
    "regime_flip": 1.0,
}

PREMIUM_VETOES = frozenset({
    "short_gamma", "short_gamma_regime",
    "below_flip", "below_gamma_flip",
    "term_backwardation", "trending",
})

STRUCTURE_CONFIDENCE = {
    "LCS": "directional_confidence",
    "LPS": "directional_confidence",
    "LC": "directional_confidence",
    "LP": "directional_confidence",
    "BKS": "breakout_confidence",
    "PCS": "compression",
    "CCS": "compression",
    "IC": "range_confidence",
    "IF": "range_confidence",
    "STG": "expansion",
}

BULL_FAVOR_REGIMES = frozenset({"trend", "breakout"})
BEAR_FAVOR_REGIMES = frozenset({"trend", "breakout"})
PREMIUM_FAVOR_REGIMES = frozenset({"compression"})
DIRECTIONAL_FAVOR_REGIMES = frozenset({"trend", "breakout"})


# --------------------------------------------------------------------------- #
# Types                                                                        #
# --------------------------------------------------------------------------- #
@dataclass
class EntrySnapshot:
    dominant_regime: str
    permitted_engine: str
    exec_regime: str
    context_regime: str
    direction_bias: str
    bias_value: float
    vetoes: list[str]
    net_gex: float
    gamma_flip: float
    flip_cushion: float
    spot: float
    structure: str
    structure_class: str
    dominant_confidence: float = 0.0


@dataclass
class PositionContext:
    position_id: str
    direction: str
    position_bias: str
    entry: EntrySnapshot
    prev_ema_score: Optional[float] = None


@dataclass
class RASComponent:
    name: str
    raw: float
    weight: float
    contribution: float
    note: str


@dataclass
class RASResult:
    score: float
    components: list[RASComponent]
    action: str
    position_id: str
    ema_score: float = 0.0


@dataclass
class RASConfig:
    enabled: bool = True
    # Paper-only automation: True lets an "exit" action close a paper position.
    # Set False (or shadow_runner --no-ras-exit) for observation-only mode.
    exit_enabled: bool = True
    warning_threshold: float = -30.0
    tighten_threshold: float = -50.0
    exit_threshold: float = -70.0
    component_weights: dict = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    ema_alpha: float = 0.4


# --------------------------------------------------------------------------- #
# Serialization helpers                                                        #
# --------------------------------------------------------------------------- #
def entry_snapshot_to_dict(snap: EntrySnapshot) -> dict:
    return asdict(snap)


def entry_snapshot_from_dict(d: dict) -> EntrySnapshot:
    if not d:
        return EntrySnapshot(
            dominant_regime="none", permitted_engine="none",
            exec_regime="none", context_regime="none",
            direction_bias="neutral", bias_value=50.0,
            vetoes=[], net_gex=0.0, gamma_flip=0.0,
            flip_cushion=0.0, spot=0.0, structure="NT",
            structure_class="directional", dominant_confidence=0.0,
        )
    return EntrySnapshot(
        dominant_regime=d.get("dominant_regime", "none"),
        permitted_engine=d.get("permitted_engine", "none"),
        exec_regime=d.get("exec_regime", "none"),
        context_regime=d.get("context_regime", "none"),
        direction_bias=d.get("direction_bias", "neutral"),
        bias_value=float(d.get("bias_value", 50.0)),
        vetoes=list(d.get("vetoes") or []),
        net_gex=float(d.get("net_gex", 0.0)),
        gamma_flip=float(d.get("gamma_flip", 0.0)),
        flip_cushion=float(d.get("flip_cushion", 0.0)),
        spot=float(d.get("spot", 0.0)),
        structure=d.get("structure", "NT"),
        structure_class=d.get("structure_class", "directional"),
        dominant_confidence=float(d.get("dominant_confidence", 0.0)),
    )


def position_context_from_entry_ctx(position_id: str, entry_ctx: dict) -> Optional[PositionContext]:
    if not entry_ctx:
        return None
    snap_raw = entry_ctx.get("entry_snapshot")
    if not snap_raw:
        return None
    entry = entry_snapshot_from_dict(snap_raw)
    return PositionContext(
        position_id=position_id,
        direction=entry_ctx.get("direction", "none"),
        position_bias=entry_ctx.get("position_bias", "neutral"),
        entry=entry,
        prev_ema_score=entry_ctx.get("ras_ema_score"),
    )


# --------------------------------------------------------------------------- #
# Entry helpers                                                                #
# --------------------------------------------------------------------------- #
def structure_class_from_family(family: str) -> str:
    return "directional" if family in DEBIT_FAMILIES else "premium"


def structure_class_from_structure(structure: str) -> str:
    return "premium" if structure in PREMIUM_STRUCTURES else "directional"


def derive_position_bias(direction: str, structure: str,
                         structure_class: str) -> str:
    if structure_class == "premium":
        return "neutral"
    if direction == "call":
        return "bull"
    if direction == "put":
        return "bear"
    if structure in VOL_STRUCTURES or direction == "both":
        return "vol"
    return "neutral"


def build_entry_snapshot(regime: RegimeState, intent: TradeIntent,
                         market: MarketSnapshot, structure_class: str,
                         structure: Optional[str] = None) -> EntrySnapshot:
    structure = structure or intent.decision.structure
    spot = market.spot
    flip = market.gamma_flip
    flip_cushion = (spot - flip) / spot if spot else 0.0
    conf_key = _relevant_confidence_key(structure)
    dom_conf = regime.confidences.get(conf_key, 0.0)
    return EntrySnapshot(
        dominant_regime=regime.dominant_regime,
        permitted_engine=regime.permitted_engine,
        exec_regime=intent.exec_regime,
        context_regime=intent.context_regime,
        direction_bias=intent.direction_bias,
        bias_value=intent.bias_value,
        vetoes=list(regime.vetoes),
        net_gex=market.net_gex,
        gamma_flip=flip,
        flip_cushion=flip_cushion,
        spot=spot,
        structure=structure,
        structure_class=structure_class,
        dominant_confidence=dom_conf,
    )


# --------------------------------------------------------------------------- #
# Component scorers                                                            #
# --------------------------------------------------------------------------- #
def _clip_unit(x: float) -> float:
    return max(-1.0, min(1.0, x))


def _score_direction_alignment(intent: TradeIntent, bias: str) -> tuple[float, str]:
    if intent is None:
        return 0.0, "no intent"
    cur = intent.direction_bias
    val = intent.bias_value
    if bias == "neutral":
        if cur == "neutral" or 42 <= val <= 58:
            return 1.0, "neutral position, market still range-bound"
        return -0.5, f"direction moved to {cur} ({val})"
    if bias == "vol":
        if cur == "neutral":
            return 0.5, "vol position, direction still neutral"
        return -0.3, f"direction resolved to {cur}"
    if bias == "bull":
        if cur == "bull":
            return 1.0, "bull bias aligned"
        if cur == "bear":
            return -1.0, "bias flipped bear"
        return -0.2 if val < 50 else 0.0, f"bias neutral ({val})"
    if bias == "bear":
        if cur == "bear":
            return 1.0, "bear bias aligned"
        if cur == "bull":
            return -1.0, "bias flipped bull"
        return -0.2 if val > 50 else 0.0, f"bias neutral ({val})"
    return 0.0, "unknown bias"


def _score_fast_momentum(intent: Optional[TradeIntent],
                         bias: str) -> Optional[tuple[float, str]]:
    """Raw fast-timeframe (1m/5m/15m) direction composite vs the position.

    This is the early-warning channel: the blended bias_value is 60% weighted
    to session-anchored slow timeframes and mathematically cannot flip during
    the first leg of an intraday reversal, so exits key off the fast composite
    directly. Asymmetric by design (quick to cut, slow to add): full -1.0
    downside when fast momentum turns against the position, but upside capped
    at +0.5 so a hot fast read cannot mask deterioration elsewhere.

    Returns None (component skipped, score unchanged) when the intent predates
    bias_fast — e.g. replaying journal rows written before this existed.
    """
    fast = getattr(intent, "bias_fast", None) if intent is not None else None
    if fast is None or not (isinstance(fast, (int, float)) and math.isfinite(fast)):
        return None

    dev = (fast - 50.0) / 25.0                 # ±1.0 at composite 25/75
    if bias == "bull":
        raw = _clip_unit(dev)
    elif bias == "bear":
        raw = _clip_unit(-dev)
    elif bias == "vol":
        raw = _clip_unit(0.5 * abs(dev))       # vol positions want movement
    else:
        # premium/neutral: strong fast momentum either way threatens the range
        raw = _clip_unit(-max(0.0, abs(dev) - 0.3))
    raw = min(raw, 0.5)

    blend = getattr(intent, "bias_value", None)
    blend_txt = f", blend {blend:.0f}" if isinstance(blend, (int, float)) else ""
    side = "with" if raw >= 0 else "against"
    return raw, f"fast composite {fast:.0f} ({side} {bias}{blend_txt})"


def _regime_favor(exec_r: str, ctx_r: str, bias: str, structure_class: str) -> float:
    if structure_class == "premium":
        favor = PREMIUM_FAVOR_REGIMES
        hostile = {"trend", "breakout"}
    elif bias == "bull":
        favor = BULL_FAVOR_REGIMES
        hostile = {"compression"}
    elif bias == "bear":
        favor = BEAR_FAVOR_REGIMES
        hostile = {"compression"}
    else:
        favor = {"compression", "trend"}
        hostile = set()

    def cell_score(r: str) -> float:
        if r in favor:
            return 1.0
        if r in hostile:
            return -1.0
        if r == "none":
            return -0.5
        return 0.0

    exec_s = cell_score(exec_r)
    ctx_s = cell_score(ctx_r)
    return _clip_unit(0.4 * exec_s + 0.6 * ctx_s)


def _score_matrix_alignment(intent: TradeIntent, entry: EntrySnapshot,
                            bias: str) -> tuple[float, str]:
    if intent is None:
        return 0.0, "no intent"
    cur_exec, cur_ctx = intent.exec_regime, intent.context_regime
    ent_exec, ent_ctx = entry.exec_regime, entry.context_regime

    if cur_exec == ent_exec and cur_ctx == ent_ctx:
        return 1.0, "matrix cell unchanged"

    cur_favor = _regime_favor(cur_exec, cur_ctx, bias, entry.structure_class)
    ent_favor = _regime_favor(ent_exec, ent_ctx, bias, entry.structure_class)
    delta = cur_favor - ent_favor
    note = (f"exec {ent_exec}->{cur_exec}, context {ent_ctx}->{cur_ctx}, "
            f"favor delta {delta:+.2f}")
    return _clip_unit(delta), note


def _std_feature(regime: RegimeState, name: str) -> Optional[float]:
    pair = regime.standardized.get(name)
    if not pair:
        return None
    v, rel = pair
    if v is None or rel <= 0:
        return None
    return float(v)


def _score_gamma_alignment(regime: RegimeState, market: MarketSnapshot,
                           entry: EntrySnapshot, bias: str,
                           structure_class: str) -> tuple[float, str]:
    if market is None or market.spot <= 0:
        return 0.0, "missing market data"

    spot = market.spot
    flip = market.gamma_flip
    cushion = (spot - flip) / spot if spot else 0.0
    short_gamma = market.net_gex <= 0
    below_flip = spot < flip

    prox = _std_feature(regime, "flip_proximity")
    gex_sign = _std_feature(regime, "gamma_sign")

    if structure_class == "premium":
        score = 0.0
        if short_gamma:
            score -= 0.7
        if below_flip:
            score -= 0.5
        if prox is not None and prox > 70:
            score -= 0.3
        if not short_gamma and not below_flip:
            score += 0.5
        return _clip_unit(score), (
            f"premium: short_gamma={short_gamma}, below_flip={below_flip}, "
            f"cushion={cushion:+.4f}"
        )

    # Directional debit — hostile when flip/bias move against thesis
    score = 0.0
    if bias == "bull":
        if below_flip:
            score -= 0.8
        elif cushion < 0.002:
            score -= 0.4
        else:
            score += 0.3
        if short_gamma and cushion < 0.005:
            score -= 0.5
    elif bias == "bear":
        if not below_flip and cushion > 0.005:
            score -= 0.7
        elif cushion > 0:
            score -= 0.3
        else:
            score += 0.3
        if short_gamma and not below_flip:
            score -= 0.4
    else:
        if short_gamma:
            score -= 0.3

    if gex_sign is not None:
        if bias == "bull" and gex_sign < 40:
            score -= 0.2
        if bias == "bear" and gex_sign > 60:
            score -= 0.2

    return _clip_unit(score), (
        f"directional {bias}: short_gamma={short_gamma}, below_flip={below_flip}, "
        f"cushion={cushion:+.4f}"
    )


def _veto_undermines(veto: str, structure_class: str) -> bool:
    if veto.startswith("catalyst"):
        return True
    if structure_class == "premium":
        return veto in PREMIUM_VETOES
    return veto.startswith("catalyst")


def _score_veto_escalation(regime: RegimeState,
                           entry: EntrySnapshot) -> tuple[float, str]:
    cur = set(regime.vetoes or [])
    ent = set(entry.vetoes or [])
    new_vetoes = cur - ent
    if not new_vetoes:
        return 0.0, "no new vetoes"
    hits = [v for v in new_vetoes if _veto_undermines(v, entry.structure_class)]
    if not hits:
        return 0.0, f"new vetoes benign: {sorted(new_vetoes)}"
    return _clip_unit(-len(hits) / max(len(hits), 1)), f"new hostile vetoes: {hits}"


def _relevant_confidence_key(structure: str) -> str:
    return STRUCTURE_CONFIDENCE.get(structure, "directional_confidence")


def _score_confidence_erosion(regime: RegimeState,
                              entry: EntrySnapshot) -> tuple[float, str]:
    key = _relevant_confidence_key(entry.structure)
    cur = regime.confidences.get(key, 0.0)
    ent = entry.dominant_confidence
    delta = cur - ent
    if delta >= -5:
        return 0.0, f"{key} confidence stable ({cur:.1f})"
    if delta >= -15:
        return -0.5, f"{key} confidence eroded {delta:+.1f}"
    return -1.0, f"{key} confidence collapsed {delta:+.1f}"


def _engine_compatible(engine: str, structure_class: str) -> bool:
    if engine == "none":
        return False
    if structure_class == "premium":
        return engine == "premium_selling"
    return engine in ("directional", "vol_expansion")


def _score_regime_flip(regime: RegimeState, intent: TradeIntent,
                       entry: EntrySnapshot, bias: str) -> tuple[float, str]:
    notes = []
    score = 0.0

    if not _engine_compatible(regime.permitted_engine, entry.structure_class):
        score -= 0.7
        notes.append(
            f"engine {entry.permitted_engine}->{regime.permitted_engine}"
        )

    if intent is not None:
        ent_favor = _regime_favor(
            entry.exec_regime, entry.context_regime, bias, entry.structure_class)
        cur_favor = _regime_favor(
            intent.exec_regime, intent.context_regime, bias, entry.structure_class)
        if cur_favor < ent_favor - 0.5:
            score -= 0.5
            notes.append(f"matrix favor dropped {ent_favor:.2f}->{cur_favor:.2f}")

    if regime.dominant_regime != entry.dominant_regime:
        notes.append(f"dominant {entry.dominant_regime}->{regime.dominant_regime}")

    if not notes:
        return 0.0, "regime stable"
    return _clip_unit(score), "; ".join(notes)


def _action_from_score(score: float, cfg: RASConfig) -> str:
    if score <= cfg.exit_threshold:
        return "exit"
    if score <= cfg.tighten_threshold:
        return "tighten"
    if score <= cfg.warning_threshold:
        return "warning"
    return "ok"


# --------------------------------------------------------------------------- #
# Main API                                                                     #
# --------------------------------------------------------------------------- #
def compute_ras(regime: RegimeState, intent: Optional[TradeIntent],
                market: Optional[MarketSnapshot], ctx: PositionContext,
                cfg: Optional[RASConfig] = None) -> RASResult:
    cfg = cfg or RASConfig()
    if not cfg.enabled:
        return RASResult(
            score=0.0, components=[], action="ok",
            position_id=ctx.position_id, ema_score=ctx.prev_ema_score or 0.0,
        )

    bias = ctx.position_bias
    entry = ctx.entry
    weights = cfg.component_weights or DEFAULT_WEIGHTS

    scorers = [
        ("direction_alignment", *_score_direction_alignment(intent, bias)),
    ]
    fast_mom = _score_fast_momentum(intent, bias)
    if fast_mom is not None:                   # None: intent predates bias_fast
        scorers.append(("fast_momentum", *fast_mom))
    scorers += [
        ("matrix_alignment", *_score_matrix_alignment(intent, entry, bias)),
        ("gamma_alignment", *_score_gamma_alignment(
            regime, market, entry, bias, entry.structure_class)),
        ("veto_escalation", *_score_veto_escalation(regime, entry)),
        ("confidence_erosion", *_score_confidence_erosion(regime, entry)),
        ("regime_flip", *_score_regime_flip(regime, intent, entry, bias)),
    ]

    components: list[RASComponent] = []
    total_w = 0.0
    weighted_sum = 0.0
    for name, raw, note in scorers:
        w = float(weights.get(name, 1.0))
        contrib = raw * w
        components.append(RASComponent(
            name=name, raw=round(raw, 3), weight=w,
            contribution=round(contrib, 3), note=note,
        ))
        weighted_sum += contrib
        total_w += w

    raw_score = (weighted_sum / total_w * 100.0) if total_w > 0 else 0.0
    raw_score = max(-100.0, min(100.0, raw_score))

    prev_ema = ctx.prev_ema_score
    if prev_ema is not None and math.isfinite(prev_ema):
        ema = cfg.ema_alpha * raw_score + (1.0 - cfg.ema_alpha) * prev_ema
    else:
        ema = raw_score
    ema = max(-100.0, min(100.0, ema))

    action = _action_from_score(ema, cfg)
    if not cfg.exit_enabled and action == "exit":
        action = "warning"

    return RASResult(
        score=round(ema, 1),
        components=components,
        action=action,
        position_id=ctx.position_id,
        ema_score=round(ema, 1),
    )


def compute_regime_alignment(regime: RegimeState, intent: Optional[TradeIntent],
                             market: Optional[MarketSnapshot],
                             position_ctx: PositionContext,
                             cfg: Optional[RASConfig] = None) -> RASResult:
    """Public entry point for position-relative regime alignment.

    Identical to compute_ras; this is the canonical documented name. See the
    module docstring for component semantics and paper-safety behavior.
    """
    return compute_ras(regime, intent, market, position_ctx, cfg=cfg)


def ras_to_signals(ras: RASResult) -> dict:
    """Flatten RAS for signals_json / component_correlations."""
    out = {
        "ras_score": ras.score,
        "ras_action": {"ok": 0, "warning": 1, "tighten": 2, "exit": 3}.get(ras.action, 0),
    }
    for c in ras.components:
        out[f"ras_{c.name}"] = c.raw
    return out
