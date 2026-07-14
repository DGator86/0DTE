"""tests/test_atomic_full_stack_rollback.py"""
from prediction.deployment import (
    DeploymentBundle, load_deployment_pointer, rollback_deployment,
    write_deployment_bundle,
)


def test_rollback_restores_complete_bundle(tmp_path):
    path = str(tmp_path / "deployment.json")
    prior = DeploymentBundle(
        deployment_id="prior",
        mode="shadow",
        prediction_model_group_id="g0",
        candidate_value_model_id="cv0",
        candidate_rank_model_id="cr0",
        fill_probability_model_id="fp0",
        fill_concession_model_id="fc0",
        meta_model_id="mm0",
        feature_version="v2.0.0",
        label_version="v2.0.0",
        structural_state_version="v3.0.0",
        policy_version="v1",
        execution_version="v3",
        risk_version="v1",
        authority_source="legacy",
        fallback_policy="abstain",
    )
    current = DeploymentBundle(
        deployment_id="current",
        mode="shadow",
        prediction_model_group_id="g1",
        candidate_value_model_id="cv1",
        candidate_rank_model_id="cr1",
        fill_probability_model_id="fp1",
        fill_concession_model_id="fc1",
        meta_model_id="mm1",
        feature_version="v2.0.0",
        label_version="v2.0.0",
        authority_source="legacy",
        fallback_policy="abstain",
        rollback_deployment_id="prior",
    )
    write_deployment_bundle(path, current)
    rollback_deployment(
        path, prior_pointer=prior.to_dict(), reason="bad_oos",
        trigger_source="human")
    restored = load_deployment_pointer(path)
    assert restored["deployment_id"] == "prior"
    assert restored["prediction_model_group"] == "g0"
    assert restored["feature_version"] == "v2.0.0"
    assert restored["fallback_policy"] == "abstain"
