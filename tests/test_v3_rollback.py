"""
tests/test_v3_rollback.py
=========================
V3 Part 3 PR31 — atomic rollback (§39 / §53).
"""
from __future__ import annotations

from prediction.deployment import (
    load_deployment_pointer, rollback_deployment, write_deployment_pointer,
)


def _ptr(group, **kw):
    d = {
        "mode": "shadow",
        "prediction_model_group": group,
        "candidate_value_model": f"cv-{group}",
        "candidate_rank_model": f"cr-{group}",
        "fill_probability_model": f"fp-{group}",
        "fill_concession_model": f"fc-{group}",
        "meta_model": f"mm-{group}",
    }
    d.update(kw)
    return d


def test_rollback_restores_prior(tmp_path):
    path = str(tmp_path / "deployment.json")
    prior = _ptr("prior")
    current = _ptr("current")
    write_deployment_pointer(path, prior)
    prior_loaded = load_deployment_pointer(path)
    write_deployment_pointer(path, current)
    audit = rollback_deployment(
        path, prior_pointer=prior_loaded, reason="artifact_validation_failure",
        trigger_source="automatic",
    )
    restored = load_deployment_pointer(path)
    assert restored["prediction_model_group"] == "prior"
    assert restored["configuration_hash"] == prior_loaded["configuration_hash"]
    assert audit["reason"] == "artifact_validation_failure"
    assert audit["human_or_automatic"] == "automatic"
