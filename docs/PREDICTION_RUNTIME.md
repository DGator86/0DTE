# PredictionRuntime

`prediction.runtime.PredictionRuntime` is the production-serving boundary for the versioned V2/V3 machine-learning stack.

## Responsibilities

The runtime:

1. Loads one `DeploymentBundle`.
2. Verifies the bundle configuration hash.
3. Runs the existing deployment and registry fail-closed validation.
4. Loads every referenced artifact once at startup.
5. Reconstructs the canonical `PredictionModelGroup` from registry metadata.
6. Exposes loaded candidate-value, candidate-rank, fill, and meta-policy artifacts to downstream adapters.
7. Produces one canonical `PredictionBundle` for each market snapshot.
8. Composes observation-specific uncertainty and marks high-uncertainty results `ABSTAIN`.
9. Journals the model IDs, artifact hashes, deployment ID, mode, latency, and runtime health.
10. Fails closed when an artifact fails, a required input is absent, or an inference contract is violated.

The runtime does not create option legs, select broker orders, alter legacy tickets, or bypass deterministic risk controls.

## Fallback policy

Heuristic forecast substitution is permitted only when all of the following are true:

- deployment mode is `research` or `shadow`;
- `fallback_policy` is `legacy`;
- an explicit heuristic provider was supplied.

Heuristic results are labeled `source=heuristic_fallback` and runtime status `DEGRADED`.

`advisory`, `candidate`, and `champion` never invoke heuristic fallback. Any required-component failure produces a canonical abstention bundle with uncertainty `1.0` and the component error recorded in diagnostics.

## Startup

```python
from prediction.registry import ModelRegistry
from prediction.runtime import PredictionRuntime

registry = ModelRegistry(directory="models")
runtime = PredictionRuntime.from_path(
    "configs/prediction_deployment.json",
    registry,
    store=prediction_store,
)
```

Startup fails if:

- the deployment hash is wrong;
- a referenced artifact is missing or tampered with;
- feature or label versions conflict;
- artifact status is not permitted for the requested deployment mode;
- a strict deployment omits a required model slot;
- the prediction group contains an unsupported target.

## Inference

```python
result = runtime.infer(
    snapshot_id=snapshot_id,
    ts=timestamp,
    session_date=session_date,
    symbol="SPY",
    feature_row=feature_row,
    structural=structural_state,
    quality=data_quality,
)

if result.actionable:
    prediction_bundle = result.bundle
else:
    # Explicit abstention. Do not create an entry decision.
    prediction_bundle = result.bundle
```

`RuntimeHealth` reports:

- `OK` — trained artifacts served successfully;
- `DEGRADED` — an explicit research/shadow heuristic fallback was used;
- `ABSTAIN` — the output must not authorize a new trade.

## Orchestrator adapter

`make_runtime_bundle_provider(runtime)` returns the existing `UnifiedOrchestrator.prediction_bundle_provider` callable shape. This allows the runtime to replace `prediction.inference.make_bundle_provider` without changing downstream policy contracts.

```python
from prediction.runtime import make_runtime_bundle_provider

bundle_provider = make_runtime_bundle_provider(runtime, symbol="SPY")
```

## Candidate and execution models

The deployment bundle's Part 3 artifacts are loaded and available through:

```python
runtime.decision_models
```

Expected keys are:

- `candidate_value_model`
- `candidate_rank_model`
- `fill_probability_model`
- `fill_concession_model`
- `meta_model`

They remain separate from physical-market forecasting. Downstream candidate evaluation and decision-packet construction should consume these artifacts only after the canonical candidate universe has been generated.

## Remaining work

This runtime closes the artifact-serving gap. Separate work remains for:

- empirical fill-model training from real paper order attempts;
- conformal uncertainty artifacts;
- counterfactual candidate outcome settlement;
- drift-triggered rollback automation;
- durable order and position state;
- deterministic pre-execution revalidation.
