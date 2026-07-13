"""Tests for the Adaptive Learning Engine (adaptive_learning/*): diagnostics,
hypothesis generation, the config store + champion loading, the promoter rule
engine and human CLI, the TPE sampler and composite metric, the journal's
learning tables, an end-to-end synthetic learning cycle, and the dashboard's
Learning-tab routes."""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field

import pytest

from journal import COLUMNS, Journal


# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #
def _seed_inverted_journal(db_path: str, n: int = 40,
                           session: str = "2026-07-08") -> None:
    """The spec's first self-improvement target: taken trades lose while
    gate-blocked candidates would have won (gate inversion)."""
    jrn = Journal(db_path)
    for i in range(n):
        traded = i % 2 == 0
        row = {c: None for c in COLUMNS}
        row.update({
            "session_date": session, "ts": f"{session}T10:{i:02d}:00-04:00",
            "spot": 600.0, "gex_regime": "long" if i % 3 else "short",
            "was_traded": 1 if traded else 0, "candidate_present": 1,
            "gate_pass": 1 if traded else 0,
            "decision": "TRADE" if traded else "NO_TRADE",
            # taken: short 599/601 call spread, settle 601 -> full loss;
            # blocked: short 604/606 call spread -> full win
            "credit": 0.4 if traded else 1.2,
            "ev": 0.1, "prob_profit": 0.6,
            "legs_json": json.dumps([
                {"qty": -1, "strike": 599.0 if traded else 604.0, "kind": "C"},
                {"qty": 1, "strike": 601.0 if traded else 606.0, "kind": "C"},
            ]),
            "regime_direction": "call",
        })
        jrn.log(row)
    jrn.settle_session(session, 601.0)
    jrn.close()


def _seed_healthy_journal(db_path: str, n: int = 40,
                          session: str = "2026-07-07") -> None:
    """Taken trades win, blocked candidates would have lost — gate working."""
    jrn = Journal(db_path)
    for i in range(n):
        traded = i % 2 == 0
        row = {c: None for c in COLUMNS}
        row.update({
            "session_date": session, "ts": f"{session}T10:{i:02d}:00-04:00",
            "spot": 600.0, "gex_regime": "long" if i % 3 else "short",
            "was_traded": 1 if traded else 0, "candidate_present": 1,
            "gate_pass": 1 if traded else 0,
            "decision": "TRADE" if traded else "NO_TRADE",
            "credit": 1.2 if traded else 0.4,
            "ev": 0.1, "prob_profit": 0.6,
            "legs_json": json.dumps([
                {"qty": -1, "strike": 604.0 if traded else 599.0, "kind": "C"},
                {"qty": 1, "strike": 606.0 if traded else 601.0, "kind": "C"},
            ]),
            "regime_direction": "call",
        })
        jrn.log(row)
    jrn.settle_session(session, 601.0)
    jrn.close()


@pytest.fixture
def inverted_db(tmp_path):
    db = str(tmp_path / "inverted.db")
    _seed_inverted_journal(db)
    return db


@pytest.fixture
def healthy_db(tmp_path):
    db = str(tmp_path / "healthy.db")
    _seed_healthy_journal(db)
    return db


# Baseline evaluation pair that passes EVERY promotion rule (mutated per-rule
# in the rejection tests below).
def _passing_evals() -> tuple[dict, dict]:
    champion = {"score": -0.32, "holdout_score": -0.20, "gate_edge": -0.40,
                "brier_skill": 0.02, "trade_count": 40, "max_drawdown": 0.9,
                "n_folds": 4, "n_profitable": 1}
    challenger = {"score": 0.18, "holdout_score": 0.11, "gate_edge": 0.15,
                  "brier_skill": 0.04, "trade_count": 38, "max_drawdown": 0.6,
                  "n_folds": 4, "n_profitable": 3}
    return champion, challenger


# --------------------------------------------------------------------------- #
# Diagnostics                                                                 #
# --------------------------------------------------------------------------- #
def test_gate_inversion_diagnosis_fires(inverted_db):
    from adaptive_learning.diagnostics import diagnose

    jrn = Journal(inverted_db)
    diagnoses = diagnose(jrn, prior_reports=[])
    jrn.close()

    issues = {d.issue: d for d in diagnoses}
    assert "gate_effectiveness_reversed" in issues
    d = issues["gate_effectiveness_reversed"]
    assert d.severity == "alert"
    assert 0.0 < d.confidence <= 0.95
    assert d.affected_module == "gate_scorer"
    # evidence carries the raw numbers: blocked mean beats taken mean
    eff = d.evidence["gate_effectiveness"]
    assert eff["blocked_by_gate"]["mean"] > eff["trades_taken"]["mean"]
    assert d.evidence["gap"] > 0


def test_no_gate_inversion_on_healthy_journal(healthy_db):
    from adaptive_learning.diagnostics import diagnose

    jrn = Journal(healthy_db)
    issues = {d.issue for d in diagnose(jrn, prior_reports=[])}
    jrn.close()
    assert "gate_effectiveness_reversed" not in issues


def test_diagnoses_sorted_by_severity(inverted_db):
    from adaptive_learning.diagnostics import diagnose

    jrn = Journal(inverted_db)
    diagnoses = diagnose(jrn, prior_reports=[])
    jrn.close()
    rank = {"alert": 0, "warn": 1, "info": 2}
    sevs = [rank[d.severity] for d in diagnoses]
    assert sevs == sorted(sevs)


def test_drift_report_persisted(inverted_db):
    from adaptive_learning.diagnostics import compute_drift, log_drift_report

    jrn = Journal(inverted_db)
    drift = compute_drift(jrn)
    rid = log_drift_report(jrn, drift, report_date="2026-07-09")
    rows = jrn.fetch_validation_reports(report_type="drift")
    jrn.close()
    assert rows[0]["id"] == rid
    assert rows[0]["report_date"] == "2026-07-09"
    # single-session journal: not enough sessions to judge, but the snapshot
    # persists (this is what the daily cycle records)
    assert rows[0]["metrics"]["drifts"] == []


def test_sharpe_collapse_from_report_history(tmp_path):
    from adaptive_learning.diagnostics import diagnose

    db = str(tmp_path / "j.db")
    jrn = Journal(db)
    # newest-first history: latest Sharpe collapsed vs trailing average
    prior = [
        {"metrics": {"walk_forward": {"mean_sharpe": 0.2}}},
        {"metrics": {"walk_forward": {"mean_sharpe": 2.0}}},
        {"metrics": {"walk_forward": {"mean_sharpe": 1.8}}},
    ]
    issues = {d.issue for d in diagnose(jrn, prior_reports=prior,
                                        include_drift=False)}
    jrn.close()
    assert "sharpe_collapse" in issues


# --------------------------------------------------------------------------- #
# Hypothesis generation                                                       #
# --------------------------------------------------------------------------- #
def _diag(issue, severity="alert", confidence=0.8):
    from adaptive_learning.diagnostics import Diagnosis
    return Diagnosis(issue=issue, severity=severity, confidence=confidence,
                     affected_module="x", likely_cause="", recommendation="")


def test_gate_inversion_space_targeted_and_straddling():
    from adaptive_learning.hypothesis import generate
    from gate_scorer import GateConfig
    from spread_selector import SelectorConfig

    hyps = generate([_diag("gate_effectiveness_reversed")])
    assert len(hyps) == 1
    space = hyps[0].param_space
    # exactly the spec's space: gate shape + selector EV floor, nothing else
    assert set(space) == {"gate.min_gex_pct_rank", "gate.max_adx",
                          "gate.flip_buffer_frac", "selector.min_ev"}
    # values straddle the current defaults (loosen AND reshape reachable)
    g, s = GateConfig(), SelectorConfig()
    assert min(space["gate.min_gex_pct_rank"]) < g.min_gex_pct_rank
    assert max(space["gate.min_gex_pct_rank"]) > g.min_gex_pct_rank
    assert min(space["gate.max_adx"]) < g.max_adx < max(space["gate.max_adx"])
    assert (min(space["gate.flip_buffer_frac"]) < g.flip_buffer_frac
            < max(space["gate.flip_buffer_frac"]))
    assert min(space["selector.min_ev"]) < s.min_ev < max(space["selector.min_ev"])


def test_observation_only_issues_produce_no_hypothesis():
    from adaptive_learning.hypothesis import generate

    hyps = generate([_diag("brier_skill_negative"),
                     _diag("regime_concentration", severity="info"),
                     _diag("MODEL_DRIFT", severity="warn")])
    assert hyps == []


def test_combined_space_dedupes_and_caps():
    from adaptive_learning.hypothesis import combined_param_space, generate

    hyps = generate([_diag("gate_effectiveness_reversed"),
                     _diag("ev_bias", severity="warn")])
    combined = combined_param_space(hyps)
    # selector.min_ev appears in both spaces; the first (higher-severity)
    # hypothesis wins the key
    assert combined["selector.min_ev"] == hyps[0].param_space["selector.min_ev"]
    assert "rnd.vol_risk_premium" in combined

    capped = combined_param_space(hyps, max_params=2)
    assert len(capped) == 2
    assert list(capped) == list(combined)[:2]


def test_duplicate_diagnoses_yield_one_hypothesis():
    from adaptive_learning.hypothesis import generate

    hyps = generate([_diag("gate_effectiveness_reversed"),
                     _diag("gate_effectiveness_reversed")])
    assert len(hyps) == 1


# --------------------------------------------------------------------------- #
# Config store                                                                #
# --------------------------------------------------------------------------- #
def test_config_record_roundtrip(tmp_path):
    from adaptive_learning import config_store as cs

    rec = cs.new_candidate(
        {"gate.max_adx": 24.0, "selector.min_ev": 0.01},
        label="gate_fix", promotion_reason="gate_effectiveness_reversed",
        regime_overrides={"compression": {"gate.max_adx": 18.0},
                          "unknown": {"size_mult": 0.25}})
    path = cs.save_candidate(rec, configs_dir=str(tmp_path))
    assert os.path.isfile(path)
    back = cs.load_config(path)
    assert back.config_id == rec.config_id
    assert back.overrides == rec.overrides
    assert back.regime_overrides == rec.regime_overrides
    assert back.status == "candidate"

    eng, clf = back.engine_cfg()
    assert eng.gate.max_adx == 24.0
    assert eng.selector.min_ev == 0.01
    assert clf is None                       # no classifier.* overrides


def test_unknown_override_keys_rejected(tmp_path):
    from adaptive_learning import config_store as cs

    with pytest.raises(ValueError, match="not_a_field"):
        cs.validate_overrides({"gate.not_a_field": 1})
    with pytest.raises(ValueError, match="bogus"):
        cs.validate_overrides({"bogus.max_adx": 1})
    with pytest.raises(ValueError, match="not_a_field"):
        cs.new_candidate({"gate.not_a_field": 1})

    # unknown top-level record keys fail loudly at load time
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"config_id": "x", "surprise": 1}))
    with pytest.raises(ValueError, match="surprise"):
        cs.load_config(str(p))


def test_classifier_overrides_supported_but_not_per_regime():
    from adaptive_learning import config_store as cs

    eng, clf = cs.apply_overrides({"classifier.min_dominant_confidence": 55})
    assert clf is not None and clf.min_dominant_confidence == 55

    with pytest.raises(ValueError, match="classifier"):
        cs.validate_regime_overrides(
            {"trend": {"classifier.min_dominant_confidence": 55}})
    with pytest.raises(ValueError, match="size_mult"):
        cs.validate_regime_overrides({"unknown": {"size_mult": -1}})


def test_regime_override_application():
    from decision_engine import EngineConfig
    from adaptive_learning.config_store import engine_cfg_for_regime

    base = EngineConfig()
    ro = {"compression": {"gate.max_adx": 18.0, "size_mult": 0.5},
          "unknown": {"size_mult": 0.25}}

    cfg, sm = engine_cfg_for_regime(base, ro, "compression")
    assert cfg.gate.max_adx == 18.0 and sm == 0.5
    # unlisted regime -> base config untouched, full size
    cfg, sm = engine_cfg_for_regime(base, ro, "trend")
    assert cfg is base and sm == 1.0
    # None regime maps to the "unknown" block
    cfg, sm = engine_cfg_for_regime(base, ro, None)
    assert cfg is base and sm == 0.25


def test_champion_loading(tmp_path):
    from adaptive_learning import config_store as cs

    # no champion file -> None (dataclass defaults apply)
    assert cs.load_champion(str(tmp_path)) is None

    rec = cs.new_candidate({"gate.max_adx": 24.0},
                           regime_overrides={"unknown": {"size_mult": 0.25}})
    cs.save_config(rec, cs.champion_path(str(tmp_path)))
    champ = cs.load_champion(str(tmp_path))
    assert champ is not None
    assert champ.engine_cfg.gate.max_adx == 24.0
    assert champ.regime_overrides == {"unknown": {"size_mult": 0.25}}

    # a corrupted champion must raise, never silently trade on defaults
    with open(cs.champion_path(str(tmp_path)), "w") as f:
        json.dump({"config_id": "x", "overrides": {"gate.zzz": 1}}, f)
    with pytest.raises(ValueError, match="zzz"):
        cs.load_champion(str(tmp_path))


def test_orchestrator_resolves_regime_overrides_at_startup():
    """shadow_runner-style construction: per-regime configs are pre-resolved
    once at __post_init__ (deterministic live path) and bad keys fail fast."""
    from unified_loop import UnifiedOrchestrator

    orch = UnifiedOrchestrator(
        feed=None,
        regime_overrides={"compression": {"gate.max_adx": 18.0},
                          "unknown": {"size_mult": 0.25}})
    cfg, sm = orch._regime_cfg["compression"]
    assert cfg.gate.max_adx == 18.0 and sm == 1.0
    cfg, sm = orch._regime_cfg["unknown"]
    assert sm == 0.25

    with pytest.raises(ValueError, match="not_a_field"):
        UnifiedOrchestrator(feed=None,
                            regime_overrides={"trend": {"gate.not_a_field": 1}})


# --------------------------------------------------------------------------- #
# Promoter: rule engine                                                       #
# --------------------------------------------------------------------------- #
def test_promotion_passes_when_all_rules_pass():
    from adaptive_learning.promoter import check_promotion

    champ, chall = _passing_evals()
    decision = check_promotion(champ, chall)
    assert decision.promote
    assert decision.failing == []
    assert {r.name for r in decision.rules} == {
        "holdout_improves", "walk_forward_consistency", "gate_edge_improves",
        "brier_skill_positive", "trade_count_maintained",
        "drawdown_not_worse", "parameters_stable", "no_severe_drift"}


@pytest.mark.parametrize("rule_name,mutation", [
    ("holdout_improves", {"holdout_score": -0.25}),
    ("walk_forward_consistency", {"n_profitable": 2}),      # 2/4 < 75%
    ("gate_edge_improves", {"gate_edge": -0.50}),
    ("brier_skill_positive", {"brier_skill": -0.01}),
    ("trade_count_maintained", {"trade_count": 30}),        # < 90% of 40
    ("drawdown_not_worse", {"max_drawdown": 1.5}),
])
def test_each_rule_individually_rejects(rule_name, mutation):
    from adaptive_learning.promoter import check_promotion

    champ, chall = _passing_evals()
    chall.update(mutation)
    decision = check_promotion(champ, chall)
    assert not decision.promote
    assert decision.failing == [rule_name]


def test_unstable_changed_parameter_rejects():
    from adaptive_learning.promoter import check_promotion

    champ, chall = _passing_evals()
    stability = {"gate.max_adx": {"verdict": "unstable"},
                 "selector.min_ev": {"verdict": "stable"}}
    decision = check_promotion(champ, chall, stability=stability,
                               changed_params=["gate.max_adx"])
    assert decision.failing == ["parameters_stable"]
    # an unstable parameter that was NOT changed does not block
    decision = check_promotion(champ, chall, stability=stability,
                               changed_params=["selector.min_ev"])
    assert decision.promote


def test_severe_drift_diagnosis_rejects():
    from adaptive_learning.promoter import check_promotion

    champ, chall = _passing_evals()
    decision = check_promotion(champ, chall,
                               diagnoses=[_diag("MODEL_DRIFT", severity="alert")])
    assert decision.failing == ["no_severe_drift"]
    # warn-level drift does not block
    decision = check_promotion(champ, chall,
                               diagnoses=[_diag("MODEL_DRIFT", severity="warn")])
    assert decision.promote


def test_metric_lost_by_challenger_rejects():
    """Champion measures gate edge but the challenger lost the readout —
    absence of evidence must not pass as 'no regression'."""
    from adaptive_learning.promoter import check_promotion

    champ, chall = _passing_evals()
    chall["gate_edge"] = None
    assert check_promotion(champ, chall).failing == ["gate_edge_improves"]

    champ, chall = _passing_evals()
    champ["gate_edge"] = None
    champ["brier_skill"] = None
    chall["gate_edge"] = None
    chall["brier_skill"] = None
    # unmeasurable on BOTH sides passes (nothing to regress from)
    assert check_promotion(champ, chall).promote


# --------------------------------------------------------------------------- #
# Promoter: pending review + human CLI                                        #
# --------------------------------------------------------------------------- #
def test_pending_review_staged_not_promoted(tmp_path):
    from adaptive_learning import config_store as cs
    from adaptive_learning.promoter import check_promotion, write_pending_review

    db = str(tmp_path / "j.db")
    jrn = Journal(db)
    configs = str(tmp_path / "configs")
    rec = cs.new_candidate({"gate.max_adx": 24.0}, label="fix")
    champ, chall = _passing_evals()
    decision = check_promotion(champ, chall)

    path = write_pending_review(rec, decision, configs_dir=configs, jrn=jrn)
    assert path == cs.pending_review_path(configs)
    staged = cs.load_config(path)
    assert staged.status == "pending_review"
    # champion.json was NOT written — that is the human CLI's job
    assert not os.path.isfile(cs.champion_path(configs))
    promos = jrn.fetch_promotions(status="pending_review")
    assert len(promos) == 1
    assert promos[0]["config_id"] == rec.config_id
    assert promos[0]["decision"]["promote"] is True
    jrn.close()


def test_approve_installs_champion_and_archives_old(tmp_path):
    from adaptive_learning import config_store as cs
    from adaptive_learning.promoter import (approve, check_promotion,
                                            write_pending_review)

    db = str(tmp_path / "j.db")
    configs = str(tmp_path / "configs")

    # existing champion
    old = cs.new_candidate({"gate.max_adx": 20.0}, label="old_champ")
    cs.save_config(old, cs.champion_path(configs))

    # stage a passing challenger
    jrn = Journal(db)
    new = cs.new_candidate({"gate.max_adx": 24.0}, label="new_champ")
    jrn.log_candidate_config(new.config_id, new.created_at, new.overrides)
    champ, chall = _passing_evals()
    write_pending_review(new, check_promotion(champ, chall),
                         configs_dir=configs, jrn=jrn)
    jrn.close()

    # wrong id refused
    with pytest.raises(ValueError, match="not"):
        approve("deadbeef", configs_dir=configs)

    champ_file = approve(new.config_id[:8], configs_dir=configs,
                         db_path=db, author="human")
    installed = cs.load_config(champ_file)
    assert installed.config_id == new.config_id
    assert installed.status == "promoted"
    # pending file consumed, old champion archived
    assert not os.path.isfile(cs.pending_review_path(configs))
    archived = os.listdir(cs.archive_dir(configs))
    assert any(old.config_id[:8] in f for f in archived)
    # journal audit trail updated
    jrn = Journal(db)
    assert jrn.fetch_promotions(status="approved")[0]["approved_by"] == "human"
    assert jrn.fetch_candidate_configs(status="promoted")[0]["config_id"] \
        == new.config_id
    jrn.close()


def test_reject_archives_pending(tmp_path):
    from adaptive_learning import config_store as cs
    from adaptive_learning.promoter import (check_promotion, reject,
                                            write_pending_review)

    configs = str(tmp_path / "configs")
    rec = cs.new_candidate({"gate.max_adx": 24.0}, label="nope")
    champ, chall = _passing_evals()
    write_pending_review(rec, check_promotion(champ, chall),
                         configs_dir=configs)
    path = reject(rec.config_id[:8], configs_dir=configs)
    assert cs.load_config(path).status == "rejected"
    assert not os.path.isfile(cs.pending_review_path(configs))
    assert not os.path.isfile(cs.champion_path(configs))


# --------------------------------------------------------------------------- #
# Optimizer: TPE sampler + composite metric                                   #
# --------------------------------------------------------------------------- #
def test_tpe_deterministic_given_seed():
    import random
    from optimizer import _tpe_next

    space = {"gate.max_adx": [16.0, 20.0, 24.0],
             "selector.min_ev": [-0.02, 0.0, 0.02]}
    history = [({"gate.max_adx": a, "selector.min_ev": e}, a * 0.1 + e)
               for a in space["gate.max_adx"] for e in space["selector.min_ev"]]

    draws_a = [_tpe_next(space, history, random.Random(7)) for _ in range(5)]
    draws_b = [_tpe_next(space, history, random.Random(7)) for _ in range(5)]
    assert draws_a == draws_b
    for d in draws_a:
        assert d["gate.max_adx"] in space["gate.max_adx"]
        assert d["selector.min_ev"] in space["selector.min_ev"]


def test_tpe_biases_toward_good_values():
    import random
    from optimizer import _tpe_next

    space = {"a": [1, 2]}
    # a=1 always scores high, a=2 always low
    history = ([({"a": 1}, 1.0)] * 10) + ([({"a": 2}, 0.0)] * 10)
    rng = random.Random(3)
    draws = [_tpe_next(space, history, rng)["a"] for _ in range(60)]
    assert draws.count(1) > 45                # ~94% expected vs 50% uniform


def test_tpe_uniform_during_startup():
    import random
    from optimizer import _tpe_next

    space = {"a": [1, 2]}
    # fewer than n_startup scored trials -> plain uniform draw, both reachable
    draws = {(_tpe_next(space, [({"a": 1}, 1.0)], random.Random(i)))["a"]
             for i in range(20)}
    assert draws == {1, 2}


@dataclass
class _TS:
    sharpe: float = 2.0
    total_pnl: float = 1.0
    win_rate: float = 0.6
    max_drawdown: float = 0.1
    trade_ticks: int = 5
    gate_effectiveness: dict = field(default_factory=lambda: {
        "trades_taken": {"n": 5, "mean": 0.5},
        "blocked_by_gate": {"n": 5, "mean": -0.5}})
    brier_skill: float = 0.5
    regime_counts: dict = field(default_factory=lambda: {
        "long": 3, "short": 1, "unknown": 1})


@dataclass
class _Fold:
    tearsheet: object


@dataclass
class _WF:
    folds: list = field(default_factory=list)

    def n_profitable(self):
        return sum(1 for f in self.folds if f.tearsheet.total_pnl > 0)


def test_composite_score_weights():
    from optimizer import composite_score

    wf = _WF(folds=[_Fold(_TS()), _Fold(_TS())])
    expected = (0.30 * math.tanh(2.0 / 2.0)      # Sharpe
                + 0.25 * 1.0                     # both folds profitable
                + 0.20 * math.tanh(1.0)          # gate edge 0.5 - (-0.5)
                + 0.10 * 0.5                     # Brier skill
                + 0.10 * (1.0 - math.exp(-1.0))  # 5 trades/fold
                + 0.05 * 1.0)                    # 3 regimes
    assert composite_score(wf) == pytest.approx(expected)

    # empty walk-forward is unrankable
    assert composite_score(_WF()) == float("-inf")


def test_composite_missing_readouts_contribute_zero():
    from optimizer import composite_score

    ts = _TS(sharpe=None, gate_effectiveness={}, brier_skill=None,
             regime_counts={}, trade_ticks=0, total_pnl=-1.0)
    wf = _WF(folds=[_Fold(ts)])
    assert composite_score(wf) == pytest.approx(0.0)


def test_composite_metric_wired_into_score():
    from optimizer import _score, composite_score

    wf = _WF(folds=[_Fold(_TS())])
    assert _score(wf, "composite") == pytest.approx(composite_score(wf))
    with pytest.raises(ValueError, match="metric"):
        _score(wf, "not_a_metric")


def test_parameter_stability_verdicts():
    from adaptive_learning.stability import (parameter_stability,
                                             stability_acceptable)

    @dataclass
    class _Trial:
        params: dict
        score: float
        wf_result: object = None

    # adx: consistent positive effect in both folds; min_ev: sign flips
    trials = []
    for adx, mev in [(16, -0.02), (16, 0.02), (24, -0.02),
                     (24, 0.02), (20, 0.0), (26, 0.01)]:
        f1 = _Fold(_TS(total_pnl=adx * 0.10 + mev * 20))
        f2 = _Fold(_TS(total_pnl=adx * 0.10 - mev * 20))
        trials.append(_Trial(params={"gate.max_adx": adx,
                                     "selector.min_ev": mev},
                             score=adx * 0.10 + mev * 0.5,
                             wf_result=_WF([f1, f2])))

    stab = parameter_stability(trials)
    assert stab["gate.max_adx"]["verdict"] == "stable"
    assert stab["selector.min_ev"]["verdict"] == "unstable"

    ok, why = stability_acceptable(stab, ["gate.max_adx"])
    assert ok
    ok, why = stability_acceptable(stab, ["selector.min_ev"])
    assert not ok and "selector.min_ev" in why


def test_learner_refuses_zero_holdout(tmp_path):
    from adaptive_learning.learner import LearnerConfig, run_learning_cycle

    cfg = LearnerConfig(db_path=str(tmp_path / "j.db"), holdout_frac=0.0)
    with pytest.raises(ValueError, match="holdout"):
        run_learning_cycle(cfg)


def test_learner_insufficient_data_is_soft_outcome(tmp_path):
    """Scheduled evening runs must not crash when ticks are thin."""
    from adaptive_learning.learner import LearnerConfig, run_learning_cycle

    db = str(tmp_path / "j.db")
    _seed_inverted_journal(db)
    empty_ticks = str(tmp_path / "ticks")
    os.makedirs(empty_ticks)
    cfg = LearnerConfig(
        db_path=db, record_dir=empty_ticks,
        configs_dir=str(tmp_path / "configs"),
        reports_dir=str(tmp_path / "reports"),
        min_ticks=100, min_sessions=3, n_trials=2, wf_folds=2,
    )
    out = run_learning_cycle(cfg, mode="evening")
    assert out["outcome"] == "insufficient_data"
    assert out["mode"] == "evening"
    runs = Journal(db).fetch_learning_runs()
    assert runs and runs[0]["outcome"] == "insufficient_data"


def test_evening_mode_cli_accepted():
    from adaptive_learning import learner as L
    assert callable(L.run_evening)
    src = open(L.__file__, encoding="utf-8").read()
    assert '"evening"' in src
    assert "--mode" in src


# --------------------------------------------------------------------------- #
# Feature lab                                                                 #
# --------------------------------------------------------------------------- #
def test_feature_lab_metrics_and_lifecycle(tmp_path):
    import random
    from adaptive_learning.feature_lab import run_feature_lab

    rng = random.Random(3)
    db = str(tmp_path / "j.db")
    jrn = Journal(db)
    session = "2026-07-08"
    for i in range(60):
        driver = rng.uniform(-1, 1)
        row = {c: None for c in COLUMNS}
        row.update({
            "session_date": session, "ts": f"{session}T10:{i:02d}:00-04:00",
            "spot": 600.0, "gex_regime": "long",
            "was_traded": 1, "candidate_present": 1, "gate_pass": 1,
            "decision": "TRADE",
            "credit": driver,                       # realized pnl == driver
            "ev": 0.1, "prob_profit": 0.5 + 0.3 * driver,
            "legs_json": json.dumps([{"qty": -1, "strike": 610.0, "kind": "C"},
                                     {"qty": 1, "strike": 612.0, "kind": "C"}]),
            "signals_json": json.dumps({
                "predictive": driver + rng.gauss(0, 0.3),
                "noise": rng.gauss(0, 1.0)}),
            "regime_direction": "call",
        })
        jrn.log(row)
    jrn.settle_session(session, 600.0)

    report = run_feature_lab(jrn, as_of="2026-07-09", seed=0)
    pred, noise = report["sig:predictive"], report["sig:noise"]
    assert abs(pred["pearson"]) > 0.5
    assert abs(pred["spearman"]) > 0.5
    assert pred["perm_importance"] > 0.3
    assert abs(noise["pearson"] or 0) < 0.3
    # the predictive signal earns experimental; noise stays observation
    assert pred["status"] == "experimental"
    assert noise["status"] == "observation"

    # rows persisted; latest_only dedupes across repeated runs
    run_feature_lab(jrn, as_of="2026-07-10", seed=0)
    latest = jrn.fetch_feature_scores(latest_only=True)
    assert len([r for r in latest if r["feature"] == "sig:predictive"]) == 1
    assert all(r["as_of"] == "2026-07-10" for r in latest
               if r["feature"] == "sig:predictive")
    jrn.close()


def test_recommend_status_never_skips_or_auto_promotes():
    from adaptive_learning.feature_lab import recommend_status

    strong = {"n": 100, "pearson": 0.4, "perm_importance": 0.3}
    # observation can only step to experimental, even with passing stability
    assert recommend_status(strong, {"passes": True},
                            current="observation") == "experimental"
    # experimental + passing stability earns candidate
    assert recommend_status(strong, {"passes": True},
                            current="experimental") == "candidate"
    # production is manual: never recommended, never demoted here
    assert recommend_status(strong, None, current="production") == "production"
    weak = {"n": 100, "pearson": 0.01, "perm_importance": -0.1}
    assert recommend_status(weak, None, current="observation") == "observation"


# --------------------------------------------------------------------------- #
# Reports                                                                     #
# --------------------------------------------------------------------------- #
def test_changed_params_map_defaults_and_overrides():
    from adaptive_learning.reports import changed_params_map

    changed = changed_params_map(
        base_overrides={"gate.max_adx": 22.0},
        best_params={"gate.max_adx": 24.0,          # old from champion override
                     "gate.min_gex_pct_rank": 0.50,  # old from dataclass default
                     "selector.min_ev": 0.0})        # unchanged vs default
    assert changed["gate.max_adx"] == [22.0, 24.0]
    assert changed["gate.min_gex_pct_rank"] == [0.60, 0.50]
    assert "selector.min_ev" not in changed


def test_promotion_report_persisted_both_ways(tmp_path):
    from adaptive_learning.promoter import check_promotion
    from adaptive_learning.reports import (build_promotion_report,
                                           persist_promotion_report)

    champ, chall = _passing_evals()
    decision = check_promotion(champ, chall)
    report = build_promotion_report(
        reason="gate_effectiveness_reversed",
        champion_eval=champ, challenger_eval=chall,
        decision=decision.to_dict(),
        changed_params={"gate.max_adx": [20.0, 24.0]})
    # spec headline block present
    assert report["promote"] is True
    assert report["current_config_score"] == champ["score"]
    assert report["candidate_config_score"] == chall["score"]
    assert report["holdout_score"] == chall["holdout_score"]

    db = str(tmp_path / "j.db")
    jrn = Journal(db)
    reports_dir = str(tmp_path / "reports")
    out = persist_promotion_report(jrn, report, reports_dir=reports_dir,
                                   report_date="2026-07-09")
    rows = jrn.fetch_validation_reports(report_type="promotion_candidate")
    assert rows[0]["id"] == out["report_id"]
    assert rows[0]["metrics"]["promote"] is True
    assert os.path.isfile(out["json_path"])
    assert os.path.isfile(out["md_path"])
    with open(out["md_path"], encoding="utf-8") as f:
        md = f.read()
    assert "# Promotion Candidate Report" in md
    assert "`gate.max_adx` | 20.0 | 24.0" in md
    assert "PROMOTE (pending human review)" in md

    # second cycle same date suffixes instead of overwriting the audit trail
    out2 = persist_promotion_report(jrn, report, reports_dir=reports_dir,
                                    report_date="2026-07-09")
    assert out2["json_path"] != out["json_path"]
    jrn.close()


# --------------------------------------------------------------------------- #
# Journal learning tables                                                     #
# --------------------------------------------------------------------------- #
def test_learning_tables_roundtrip(tmp_path):
    db = str(tmp_path / "j.db")
    jrn = Journal(db)

    jrn.log_learning_run(
        "run1", "weekly", "2026-07-09T00:00:00Z", "2026-07-09T00:05:00Z",
        diagnostics=[{"issue": "gate_effectiveness_reversed"}],
        param_space={"gate.max_adx": [16.0, 24.0]},
        n_trials=8, best_score=0.4, holdout_score=0.2,
        trials=[{"id": 1, "params": {"gate.max_adx": 24.0}, "score": 0.4}],
        outcome="promotion_recommended")
    runs = jrn.fetch_learning_runs()
    assert len(runs) == 1
    assert runs[0]["diagnostics"][0]["issue"] == "gate_effectiveness_reversed"
    assert runs[0]["param_space"]["gate.max_adx"] == [16.0, 24.0]
    assert runs[0]["trials"][0]["score"] == 0.4

    jrn.log_candidate_config("cfg1", "2026-07-09T00:05:00Z",
                             {"gate.max_adx": 24.0}, label="fix")
    assert jrn.fetch_candidate_configs()[0]["status"] == "candidate"
    jrn.update_candidate_status("cfg1", "pending_review")
    assert jrn.fetch_candidate_configs(status="pending_review")[0]["config_id"] \
        == "cfg1"

    jrn.log_promotion("cfg1", {"promote": True, "rules": []})
    assert jrn.fetch_promotions(status="pending_review")[0]["approved_at"] is None
    jrn.update_promotion("cfg1", "approved", approved_by="human")
    row = jrn.fetch_promotions(status="approved")[0]
    assert row["approved_by"] == "human" and row["approved_at"]
    # already-resolved promotions are not re-updatable
    assert jrn.update_promotion("cfg1", "rejected") == 0

    # reopening the DB keeps the tables (CREATE IF NOT EXISTS migration path)
    jrn.close()
    jrn = Journal(db)
    assert len(jrn.fetch_learning_runs()) == 1
    jrn.close()


# --------------------------------------------------------------------------- #
# End-to-end: synthetic gate-inversion learning cycle                         #
# --------------------------------------------------------------------------- #
def test_learning_cycle_end_to_end(tmp_path, monkeypatch):
    """Full offline cycle on the coupled synthetic world: diagnose the seeded
    gate inversion, search a targeted space with mandatory holdout, and leave
    every artifact behind — learning_runs row, candidate file, promotion
    report, and journal rows the dashboard routes then serve."""
    from fastapi.testclient import TestClient
    from synthetic_world import CoupledSyntheticFeed, WorldConfig
    from adaptive_learning.learner import LearnerConfig, run_learning_cycle
    from dashboard.server import app, _configure

    db = str(tmp_path / "shadow.db")
    _seed_inverted_journal(db)

    def make_feed():
        return CoupledSyntheticFeed(WorldConfig(days=4, seed=11,
                                                tick_stride=39))
    ticks = make_feed().timestamps()

    configs_dir = str(tmp_path / "configs")
    cfg = LearnerConfig(
        db_path=db, configs_dir=configs_dir,
        reports_dir=str(tmp_path / "reports"),
        search="random", n_trials=2, holdout_frac=0.25,
        wf_folds=2, max_params=3)
    out = run_learning_cycle(cfg, mode="manual", feed_factory=make_feed,
                             timestamps=ticks, report_date="2026-07-09")

    # the seeded failure was diagnosed and drove the search space
    assert out["reason"] == "gate_effectiveness_reversed"
    issues = {d["issue"] for d in out["diagnoses"]}
    assert "gate_effectiveness_reversed" in issues
    assert set(out["param_space"]) <= {
        "gate.min_gex_pct_rank", "gate.max_adx", "gate.flip_buffer_frac",
        "selector.min_ev"}
    assert out["outcome"] in ("promotion_recommended", "rejected")
    # holdout was actually scored (mandatory)
    assert out["challenger_eval"]["holdout_score"] is not None

    # artifacts on disk
    assert os.path.isfile(out["candidate"]["path"])
    assert os.path.isfile(out["report"]["md_path"])
    from adaptive_learning.config_store import champion_path
    assert not os.path.isfile(champion_path(configs_dir))   # never auto-promoted

    # journal rows
    jrn = Journal(db)
    runs = jrn.fetch_learning_runs()
    assert len(runs) == 1 and runs[0]["outcome"] == out["outcome"]
    assert runs[0]["holdout_score"] is not None
    cands = jrn.fetch_candidate_configs()
    assert cands[0]["config_id"] == out["candidate"]["config_id"]
    assert jrn.fetch_validation_reports(report_type="promotion_candidate")
    assert jrn.fetch_validation_reports(report_type="drift")
    jrn.close()

    # dashboard routes serve the new data
    monkeypatch.setenv("DASHBOARD_TOKEN", "test-secret-token")
    _configure(db, str(tmp_path / "paper.sqlite"),
               str(tmp_path / "live.json"), configs_dir=configs_dir)
    c = TestClient(app)
    hdrs = {"Authorization": "Bearer test-secret-token"}

    r = c.get("/api/learning", headers=hdrs)
    assert r.status_code == 200
    assert r.json()["runs"][0]["run_id"] == out["run_id"]

    r = c.get("/api/candidates", headers=hdrs)
    assert r.status_code == 200
    body = r.json()
    assert body["champion"] is None
    assert body["candidates"][0]["config_id"] == out["candidate"]["config_id"]

    assert c.get("/api/promotions", headers=hdrs).status_code == 200
    r = c.get("/api/feature-scores", headers=hdrs)
    assert r.status_code == 200 and r.json()["features"]
    r = c.get("/api/drift", headers=hdrs)
    assert r.status_code == 200 and r.json()["reports"]
    r = c.get("/api/validation?report_type=promotion_candidate", headers=hdrs)
    assert r.status_code == 200 and r.json()["reports"]


# --------------------------------------------------------------------------- #
# Dashboard Learning-tab routes (fast, direct seeding)                        #
# --------------------------------------------------------------------------- #
def test_learning_routes_direct(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from adaptive_learning import config_store as cs
    from dashboard.server import app, _configure

    monkeypatch.setenv("DASHBOARD_TOKEN", "test-secret-token")
    db = str(tmp_path / "shadow.db")
    configs_dir = str(tmp_path / "configs")

    champ = cs.new_candidate({"gate.max_adx": 24.0}, label="champ")
    cs.save_config(champ, cs.champion_path(configs_dir))

    jrn = Journal(db)
    jrn.log_learning_run("runX", "daily", "2026-07-09T00:00:00Z",
                         "2026-07-09T00:01:00Z", outcome="diagnostics_only")
    jrn.log_candidate_config("cfgA", "2026-07-09T00:02:00Z",
                             {"gate.max_adx": 22.0}, status="pending_review")
    jrn.log_promotion("cfgA", {"promote": True, "rules": [
        {"name": "holdout_improves", "passed": True, "detail": "ok"}]})
    jrn.log_feature_score("sig:predictive", "2026-07-09", 60,
                          pearson=0.6, status="experimental")
    jrn.log_validation_report("2026-07-09", "drift", {"drifts": []},
                              "Drift check — no drift beyond thresholds")
    jrn.close()

    _configure(db, str(tmp_path / "paper.sqlite"), str(tmp_path / "live.json"),
               configs_dir=configs_dir)
    c = TestClient(app)
    hdrs = {"Authorization": "Bearer test-secret-token"}

    # all five routes require auth
    for ep in ("/api/learning", "/api/candidates", "/api/promotions",
               "/api/feature-scores", "/api/drift"):
        assert c.get(ep).status_code == 401

    assert c.get("/api/learning", headers=hdrs).json()["runs"][0]["run_id"] \
        == "runX"

    body = c.get("/api/candidates", headers=hdrs).json()
    assert body["champion"]["config_id"] == champ.config_id
    assert body["candidates"][0]["status"] == "pending_review"
    r = c.get("/api/candidates?status=promoted", headers=hdrs)
    assert r.json()["candidates"] == []
    assert c.get("/api/candidates?status=bogus", headers=hdrs).status_code == 422

    promo = c.get("/api/promotions?status=pending_review",
                  headers=hdrs).json()["promotions"][0]
    assert promo["decision"]["rules"][0]["name"] == "holdout_improves"

    feats = c.get("/api/feature-scores", headers=hdrs).json()["features"]
    assert feats[0]["feature"] == "sig:predictive"
    assert feats[0]["status"] == "experimental"

    drift = c.get("/api/drift", headers=hdrs).json()["reports"]
    assert drift[0]["report_type"] == "drift"

    # missing DB degrades gracefully on every route
    _configure(str(tmp_path / "absent.db"), str(tmp_path / "p.sqlite"),
               str(tmp_path / "live.json"), configs_dir=str(tmp_path / "nc"))
    for ep, key in (("/api/learning", "runs"), ("/api/candidates", "candidates"),
                    ("/api/promotions", "promotions"),
                    ("/api/feature-scores", "features"),
                    ("/api/drift", "reports")):
        r = c.get(ep, headers=hdrs)
        assert r.status_code == 200
        assert r.json()[key] == []
