"""
policy/prediction_policy.py
===========================
PredictionPolicy — V2 policy that consumes PredictionBundle
(docs/PREDICTION_ENGINE_V2_HANDOFF.md §17.4, PR 10).

Maps calibrated forecasts + structural / operational risk into a
PolicyDecision. Does not train models and does not mutate the bundle.
When required forecast fields are missing, raises PredictionUnavailable
so the router can emit an explicit fallback_legacy decision (§17.5).

NOT financial advice.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from prediction.contracts import PredictionBundle
from policy.contracts import SOURCE_V2, PolicyDecision, PolicyInput
from spread_selector import STRUCTURE_TO_FAMILIES

PREDICTION_POLICY_VERSION = "v2-prediction-policy-v1"


class PredictionUnavailable(Exception):
    """V2 forecast inputs insufficient for a policy decision."""


@dataclass
class PredictionPolicyConfig:
    # Premium selling (§17.4)
    min_range_survive: float = 0.55
    max_realized_vs_implied: float = 0.90   # realized / implied
    max_uncertainty_premium: float = 0.45
    min_data_quality: float = 0.40

    # Directional debit
    min_direction_prob: float = 0.58
    min_expected_return: float = 0.0005     # log-return hurdle (gross)
    max_uncertainty_directional: float = 0.50
    # Quantile profile: q50 should agree with direction sign
    require_quantile_agreement: bool = True

    # Long volatility
    min_realized_vs_implied_vol: float = 1.15
    max_range_survive_for_vol: float = 0.45
    max_uncertainty_vol: float = 0.55

    # Global no-trade
    max_uncertainty_any: float = 0.75
    conflict_margin: float = 0.08          # premium vs directional conflict


def _families_for(structure_code: str) -> tuple[str, ...]:
    fams = STRUCTURE_TO_FAMILIES.get(structure_code)
    if not fams:
        return ()
    return tuple(sorted(fams))


def _first_not_none(*vals):
    for v in vals:
        if v is not None:
            return v
    return None


def bundle_is_usable(bundle: Optional[PredictionBundle]) -> bool:
    """True when enough forecast fields exist to attempt a V2 decision."""
    if bundle is None:
        return False
    has_dir = any(getattr(bundle, k) is not None for k in (
        "p_up_30m", "p_up_15m", "p_up_60m", "p_up_close"))
    has_range = any(getattr(bundle, k) is not None for k in (
        "p_range_survive_30m", "p_range_survive_15m",
        "p_range_survive_60m", "p_range_survive_close"))
    has_move = any(getattr(bundle, k) is not None for k in (
        "expected_realized_move_30m", "expected_realized_move_close"))
    return bool(has_dir or has_range or has_move)


class PredictionPolicy:
    """§17.4 example policy logic over PredictionBundle."""

    version: str = PREDICTION_POLICY_VERSION

    def __init__(self, cfg: Optional[PredictionPolicyConfig] = None):
        self.cfg = cfg or PredictionPolicyConfig()

    def decide(self, inp: PolicyInput) -> PolicyDecision:
        bundle = inp.predictions
        if not bundle_is_usable(bundle):
            raise PredictionUnavailable(
                "PredictionBundle missing or lacks usable forecast fields")

        assert bundle is not None
        cfg = self.cfg
        op = inp.operational_risk_state or {}
        hard_vetoes = tuple(op.get("hard_vetoes") or ())
        # Operational hard stops always win.
        if op.get("stand_down") or any(
                str(v).startswith("catalyst") for v in hard_vetoes):
            return self._no_trade(
                bundle, hard_vetoes,
                rationale=("operational hard veto / stand_down",),
                confidence=0.0)

        uncertainty = float(bundle.uncertainty if bundle.uncertainty is not None
                            else 1.0)
        data_quality = float(bundle.data_quality if bundle.data_quality is not None
                             else 0.0)

        if uncertainty >= cfg.max_uncertainty_any:
            return self._no_trade(
                bundle, hard_vetoes,
                rationale=(f"uncertainty {uncertainty:.2f} >= "
                           f"{cfg.max_uncertainty_any}",),
                confidence=0.0)
        if data_quality < cfg.min_data_quality:
            return self._no_trade(
                bundle, hard_vetoes,
                rationale=(f"data_quality {data_quality:.2f} < "
                           f"{cfg.min_data_quality}",),
                confidence=max(0.0, data_quality))

        implied_move = op.get("implied_remaining_move")
        realized = _first_not_none(
            bundle.expected_realized_move_30m,
            bundle.expected_realized_move_close)
        range_p = _first_not_none(
            bundle.p_range_survive_30m,
            bundle.p_range_survive_15m,
            bundle.p_range_survive_close,
            bundle.p_range_survive_60m)
        p_up = _first_not_none(
            bundle.p_up_30m, bundle.p_up_15m,
            bundle.p_up_60m, bundle.p_up_close)
        exp_ret = _first_not_none(
            bundle.expected_return_30m, bundle.expected_return_15m,
            bundle.expected_return_60m, bundle.expected_return_close)
        q50 = _first_not_none(bundle.return_q50_30m, bundle.return_q50_close)

        premium_ok = self._premium_eligible(
            range_p, realized, implied_move, uncertainty, hard_vetoes)
        directional = self._directional_eligible(
            p_up, exp_ret, q50, uncertainty, hard_vetoes)
        vol_ok = self._vol_eligible(
            range_p, realized, implied_move, uncertainty, hard_vetoes)

        # Conflict: premium wants range, directional wants trend.
        if premium_ok and directional is not None:
            # Mild conflict — prefer the higher-confidence path unless
            # both are strong; then stand down.
            prem_conf = float(range_p or 0.0)
            dir_conf = abs(float(p_up or 0.5) - 0.5) * 2.0
            if (prem_conf >= cfg.min_range_survive + cfg.conflict_margin
                    and dir_conf >= (cfg.min_direction_prob - 0.5) * 2.0
                    + cfg.conflict_margin):
                return self._no_trade(
                    bundle, hard_vetoes,
                    rationale=("forecast conflict: range survival and "
                               "directional edge both strong",),
                    confidence=min(prem_conf, dir_conf))

        if premium_ok and directional is None:
            # Lean with mild directional bias when available.
            structure, direction = self._premium_structure(p_up, hard_vetoes)
            if structure == "NT":
                return self._no_trade(
                    bundle, hard_vetoes,
                    rationale=("premium forbidden by operational vetoes",),
                    confidence=float(range_p or 0.0))
            return PolicyDecision(
                action="TRADE",
                direction=direction,
                eligible_families=_families_for(structure),
                confidence=float(range_p or 0.6),
                uncertainty=uncertainty,
                size_cap=self._size_cap(uncertainty, float(range_p or 0.6)),
                hard_vetoes=hard_vetoes,
                rationale=(
                    f"premium: range_survive={range_p}",
                    f"realized_move={realized}",
                ),
                policy_version=self.version,
                source=SOURCE_V2,
                structure_code=structure,
            )

        if directional is not None:
            structure, direction = directional
            return PolicyDecision(
                action="TRADE",
                direction=direction,
                eligible_families=_families_for(structure),
                confidence=abs(float(p_up or 0.5) - 0.5) * 2.0,
                uncertainty=uncertainty,
                size_cap=self._size_cap(
                    uncertainty, abs(float(p_up or 0.5) - 0.5) * 2.0),
                hard_vetoes=hard_vetoes,
                rationale=(
                    f"directional: p_up={p_up}",
                    f"expected_return={exp_ret}",
                ),
                policy_version=self.version,
                source=SOURCE_V2,
                structure_code=structure,
            )

        if vol_ok:
            return PolicyDecision(
                action="TRADE",
                direction="both",
                eligible_families=_families_for("STG"),
                confidence=min(1.0, float(realized or 0.0) /
                               max(float(implied_move or 1e-9), 1e-9) / 2.0),
                uncertainty=uncertainty,
                size_cap=self._size_cap(uncertainty, 0.5),
                hard_vetoes=hard_vetoes,
                rationale=(
                    f"long_vol: realized={realized} implied={implied_move}",
                    f"range_survive={range_p}",
                ),
                policy_version=self.version,
                source=SOURCE_V2,
                structure_code="STG",
            )

        return self._no_trade(
            bundle, hard_vetoes,
            rationale=("no V2 policy path cleared thresholds",),
            confidence=0.0)

    # ------------------------------------------------------------------ #
    def _no_trade(self, bundle: PredictionBundle, hard_vetoes: tuple,
                  *, rationale: tuple, confidence: float) -> PolicyDecision:
        unc = float(bundle.uncertainty if bundle.uncertainty is not None
                    else 1.0)
        return PolicyDecision(
            action="NO_TRADE",
            direction="none",
            eligible_families=(),
            confidence=float(confidence),
            uncertainty=unc,
            size_cap=0.0,
            hard_vetoes=hard_vetoes,
            rationale=rationale,
            policy_version=self.version,
            source=SOURCE_V2,
            structure_code="",
        )

    def _premium_eligible(self, range_p, realized, implied, uncertainty,
                          hard_vetoes) -> bool:
        cfg = self.cfg
        if range_p is None or range_p < cfg.min_range_survive:
            return False
        if uncertainty > cfg.max_uncertainty_premium:
            return False
        no_premium = any(
            v in {
                "short_gamma", "short_gamma_regime",
                "below_flip", "below_gamma_flip",
                "trending", "term_backwardation", "adx_trend",
            } or str(v).startswith("adx")
            for v in hard_vetoes)
        if no_premium:
            return False
        if implied is not None and realized is not None and implied > 0:
            if float(realized) / float(implied) > cfg.max_realized_vs_implied:
                return False
        return True

    def _directional_eligible(self, p_up, exp_ret, q50, uncertainty,
                              hard_vetoes):
        cfg = self.cfg
        if p_up is None:
            return None
        if uncertainty > cfg.max_uncertainty_directional:
            return None
        bull = p_up >= cfg.min_direction_prob
        bear = p_up <= (1.0 - cfg.min_direction_prob)
        if not bull and not bear:
            return None
        if exp_ret is not None:
            if bull and exp_ret < cfg.min_expected_return:
                return None
            if bear and exp_ret > -cfg.min_expected_return:
                return None
        if cfg.require_quantile_agreement and q50 is not None:
            if bull and q50 < 0:
                return None
            if bear and q50 > 0:
                return None
        if bull:
            return ("LCS", "call")
        return ("LPS", "put")

    def _vol_eligible(self, range_p, realized, implied, uncertainty,
                      hard_vetoes) -> bool:
        cfg = self.cfg
        if uncertainty > cfg.max_uncertainty_vol:
            return False
        if realized is None or implied is None or float(implied) <= 0:
            return False
        if float(realized) / float(implied) < cfg.min_realized_vs_implied_vol:
            return False
        if range_p is not None and range_p > cfg.max_range_survive_for_vol:
            return False
        return True

    def _premium_structure(self, p_up, hard_vetoes) -> tuple[str, str]:
        no_premium = any(
            v in {
                "short_gamma", "short_gamma_regime",
                "below_flip", "below_gamma_flip",
                "trending", "term_backwardation",
            } or str(v).startswith("adx")
            for v in hard_vetoes)
        if no_premium:
            return ("NT", "none")
        if p_up is None:
            return ("IC", "both")
        if p_up >= 0.55:
            return ("PCS", "put")
        if p_up <= 0.45:
            return ("CCS", "call")
        return ("IC", "both")

    @staticmethod
    def _size_cap(uncertainty: float, confidence: float) -> float:
        # Shrink with uncertainty; floor at 0, cap at 1.
        raw = max(0.0, min(1.0, confidence * (1.0 - uncertainty)))
        if raw >= 0.75:
            return 1.0
        if raw >= 0.45:
            return 0.6
        if raw >= 0.20:
            return 0.3
        return 0.0
