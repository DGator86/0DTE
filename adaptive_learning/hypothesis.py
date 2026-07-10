"""
adaptive_learning/hypothesis.py
===============================
Diagnosis -> targeted parameter search space. The one rule: only modify
parameters associated with a diagnosed failure. Never generate a blind grid
over everything — that is how selection bias sneaks back in.

Each mapping proposes values on BOTH sides of the current default. When the
gate is reversed the right move is to loosen and reshape it, not to blindly
tighten; the space must let the optimizer discover which direction the data
actually supports.

NOT financial advice.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

from adaptive_learning.config_store import validate_overrides
from adaptive_learning.diagnostics import Diagnosis

# The first self-improvement target (see the spec): when the gate is reversed,
# search the gate's shape and the selector's EV floor — values straddle the
# defaults (min_gex_pct_rank 0.60, max_adx 20, flip_buffer_frac 0.0015,
# min_ev 0.0) so both loosening and reshaping are reachable.
GATE_INVERSION_SPACE = {
    "gate.min_gex_pct_rank": [0.45, 0.50, 0.55, 0.60, 0.65],
    "gate.max_adx": [16.0, 18.0, 20.0, 22.0, 24.0, 26.0],
    "gate.flip_buffer_frac": [0.0005, 0.001, 0.0015, 0.002],
    "selector.min_ev": [-0.02, 0.00, 0.01, 0.02],
}

EV_BIAS_SPACE = {
    "rnd.vol_risk_premium": [0.00, 0.05, 0.10, 0.15, 0.20],
    "selector.min_ev": [-0.02, 0.00, 0.01, 0.02],
}

DIRECTIONAL_SPACE = {
    "rnd.dir_drift_frac": [0.10, 0.20, 0.30, 0.40],
    "gate.dir_adx_floor": [15.0, 20.0, 25.0],
    "gate.dir_adx_full": [30.0, 35.0, 40.0],
}

# Over-tight funnel: the same gate space plus the selector's liquidity/touch
# vetoes, which are the other common reasons trades stop happening.
TRADE_COLLAPSE_SPACE = {
    **GATE_INVERSION_SPACE,
    "selector.max_touch_short": [0.45, 0.55, 0.65],
    "selector.min_liquidity": [0.15, 0.25, 0.35],
}

ISSUE_TO_SPACE = {
    "gate_effectiveness_reversed": GATE_INVERSION_SPACE,
    "ev_bias": EV_BIAS_SPACE,
    "directional_weak": DIRECTIONAL_SPACE,
    "trade_frequency_collapse": TRADE_COLLAPSE_SPACE,
    # sharpe_collapse alone gives no parameter target: re-search the gate space
    # (the highest-leverage knobs) rather than everything.
    "sharpe_collapse": GATE_INVERSION_SPACE,
}

# Issues that never map to a parameter search: they demand inspection or more
# data, and turning them into knob-twiddling would be the "random nonsense"
# the spec forbids.
OBSERVATION_ONLY_ISSUES = frozenset({
    "brier_skill_negative", "regime_concentration",
    "MODEL_DRIFT", "REGIME_DRIFT", "FEATURE_DRIFT",
})


@dataclass
class Hypothesis:
    issue: str
    param_space: dict = field(default_factory=dict)
    rationale: str = ""
    confidence: float = 0.0

    def to_dict(self) -> dict:
        return {"issue": self.issue, "param_space": self.param_space,
                "rationale": self.rationale, "confidence": self.confidence}


def generate(diagnoses: Iterable[Diagnosis]) -> list[Hypothesis]:
    """One hypothesis per diagnosable issue, ordered by diagnosis order (which
    diagnostics.diagnose already sorted by severity + confidence)."""
    out: list[Hypothesis] = []
    seen: set[str] = set()
    for d in diagnoses:
        if d.issue in seen or d.issue in OBSERVATION_ONLY_ISSUES:
            continue
        space = ISSUE_TO_SPACE.get(d.issue)
        if not space:
            continue
        validate_overrides(space and {k: v[0] for k, v in space.items()})
        seen.add(d.issue)
        out.append(Hypothesis(
            issue=d.issue,
            param_space={k: list(v) for k, v in space.items()},
            rationale=d.recommendation,
            confidence=d.confidence,
        ))
    return out


def combined_param_space(hypotheses: Iterable[Hypothesis],
                         max_params: Optional[int] = None) -> dict:
    """Union of the hypotheses' spaces (first hypothesis wins on key clash —
    it came from the highest-severity diagnosis). max_params caps the search
    dimensionality, keeping the strongest-diagnosis keys."""
    space: dict = {}
    for h in hypotheses:
        for k, v in h.param_space.items():
            space.setdefault(k, list(v))
    if max_params is not None and len(space) > max_params:
        space = dict(list(space.items())[:max_params])
    return space


# --------------------------------------------------------------------------- #
# Demo                                                                          #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    from adaptive_learning.diagnostics import Diagnosis

    ds = [
        Diagnosis(issue="gate_effectiveness_reversed", severity="alert",
                  confidence=0.9, affected_module="gate_scorer",
                  likely_cause="", recommendation="loosen and reshape"),
        Diagnosis(issue="brier_skill_negative", severity="alert",
                  confidence=0.6, affected_module="mc",
                  likely_cause="", recommendation="inspect bins"),
        Diagnosis(issue="ev_bias", severity="warn", confidence=0.5,
                  affected_module="rnd_extractor",
                  likely_cause="", recommendation="search VRP"),
    ]
    print("=" * 64)
    print("  hypothesis demo")
    print("=" * 64)
    hyps = generate(ds)
    for h in hyps:
        print(f"  {h.issue}: {sorted(h.param_space)}")
    print(f"  combined: {sorted(combined_param_space(hyps))}")
    print("  (brier_skill_negative correctly produced NO hypothesis)")
    print("=" * 64)
