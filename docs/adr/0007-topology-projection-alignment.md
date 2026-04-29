# ADR-0007: Topology Projection Alignment

Date: 2026-04-27
Status: Proposed

## Context

`DynamicGraphBuilder` supports topology projection modes, including:

```yaml
mapping.graph_feature_mapping.topology_projection.gateway_mode: collapse_for_prediction
```

The debt worklog records that `collapse_for_prediction` can break indexing or
alignment between projected topology and structural tensors if alignment is not
verified after topology modification.

This ADR does not solve duplicate activity labels. If several BPMN prediction
nodes share the same log activity label, that is a separate identity ambiguity:
the projection layer must remain compatible with a future node-identity-aware
mapping, but this ADR only validates alignment for the current prediction
class-space.

The current implementation resolves projection behavior in
`DynamicGraphBuilder` and includes tests around collapsed masks, but the final
research-grade projection alignment contract is not yet documented as accepted.

## Decision

Target contract:

Any topology projection that changes graph nodes or edges must produce an
explicit post-projection alignment artifact before structural tensors are used
for research-grade runs.

Projection must not silently invalidate:

1. class-space activity indices,
2. allowed target masks,
3. structural edge indexes,
4. stats feature rows in `struct_x`,
5. edge weights derived from stats indexes.

## Consequences

Positive:

1. projected topology becomes auditable,
2. mask and structural tensor semantics stay aligned,
3. gateway collapse can be used without hiding index drift,
4. future projection modes have a clear validation requirement.

Negative:

1. projection implementation needs extra diagnostics,
2. collapsed topology may need explicit mapping tables,
3. some existing tests may need stricter assertions before this can be accepted.

## Runtime Rules

Current behavior:

1. `preserve` and `collapse_for_prediction` are supported by
   `DynamicGraphBuilder`.
2. `TopologyProjectionCompiler` produces projected paths and diagnostics.
3. `DynamicGraphBuilder` exposes scalar projection diagnostics on
   `GraphTensorContract` without adding non-tensor dictionaries to per-sample
   contracts.
4. `ModelTrainer` logs topology projection counters in forward stats for train,
   inference, and drift evaluation.
5. `EOPKGGATv2` fails fast on out-of-bounds structural edge indices instead of
   modulo-repairing them.
6. Domain/application/integration pytest coverage for the implementation passes
   when the project venv command is run with escalated permissions in Codex
   sandbox.

Target behavior:

1. projection must produce or expose source-to-projected mapping,
2. post-projection alignment must be checked before research-grade structural
   forward,
3. projection diagnostics must be logged or attached to experiment artifacts,
4. if projection changes indexing and alignment cannot be proven, fail fast in
   research-grade mode.

Out of scope:

1. resolving duplicate activity labels into unique BPMN node identities,
2. changing the prediction target from activity-label class-space to node-id
   class-space,
3. route-inferred disambiguation of events without stable node ids.

## Affected Files

- `src/domain/services/dynamic_graph_builder.py`
- `src/domain/services/topology_projection_alignment.py`
- `src/domain/entities/tensor_contract.py`
- `src/domain/models/eopkg_models.py`
- `src/application/use_cases/trainer.py`
- `tests/domain/test_dynamic_graph_builder_masks.py`
- `tests/domain/test_topology_projection_alignment.py`
- `configs/experiments/`
- `configs/ui/config_catalog.yaml`

## Related

- `docs/worklogs/MVP2_5_Canonical_Doc_Sync_and_Architecture_Debt_2026-04-24.MD`
- `docs/current/architecture-debt.md#duplicate_activity_identity_ambiguity`
- `docs/GNN_RUNTIME_MVP2_5.MD`
- `docs/DATA_MODEL_MVP2_5.MD`
