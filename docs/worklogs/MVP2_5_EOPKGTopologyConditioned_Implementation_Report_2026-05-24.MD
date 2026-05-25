# MVP2.5 EOPKGTopologyConditioned Implementation Report

- `date`: 2026-05-24
- `status`: implemented
- `scope`: topology-conditioned candidate scoring Stage 1
- `model_type`: `EOPKGTopologyConditioned`
- `plan`: `docs/worklogs/MVP2_5_EOPKGTopologyConditioned_Implementation_Plan_2026-05-24.MD`

---

## Summary

Implemented `EOPKGTopologyConditioned` as a separate `model.type`.

This is not another `EOPKGGATv2.fusion_mode`. It is a new architecture path
where topology defines the candidate space and the observed prefix is scored
against topology candidate embeddings.

Stage 1 keeps compatibility with the current trainer/evaluator contract:

```text
model forward output: [B, C_train]
internal node scores: [B, |V|]
```

Dynamic per-version candidate output `[B, C_v]` remains future work and is
tracked as architecture debt.

---

## Implemented Runtime Behavior

### Model

Added:

```text
src/domain/models/topology_conditioned.py
```

Registered as:

```yaml
model.type: EOPKGTopologyConditioned
```

Supported Stage 1 config:

```yaml
model.observed_encoder: GATv2
model.struct_encoder: GATv2
model.candidate_scoring: cosine       # cosine | bilinear
model.candidate_pooling: logmeanexp   # logmeanexp | mean | max
model.candidate_temperature_init: 0.1
model.candidate_temperature_min: 0.05
model.candidate_temperature_max: 10.0
model.candidate_temperature_trainable: false
```

### Candidate Identity Grounding

`struct_x` alone is not enough semantic grounding for candidate scoring.
Therefore Stage 1 adds minimal trainable candidate identity grounding:

- `struct_node_to_class_index >= 0`: class embedding is added to structural
  node features;
- `struct_node_to_class_index == -1`: shared non-target embedding is used;
- non-target nodes still participate in structural message passing;
- non-target nodes are excluded from class pooling.

`struct_x` is optional enrichment for this model. If graph feature mapping or
statistics are disabled and `struct_x` is absent, the model starts structural
candidate states from identity embeddings only. Required structural fields are
`structural_edge_index` and `struct_node_to_class_index`.

### Duplicate Activity Handling

Duplicate BPMN nodes mapped to one activity class are pooled through MIL:

- default: normalized `logmeanexp`;
- alternatives: `mean`, `max`.

Normalized `logmeanexp` avoids raw duplicate-count inflation that would happen
with plain `logsumexp`.

### Mask Semantics

The model returns clean candidate/class scores.

It does not apply `allowed_target_mask` inside `forward`.
Mask-guided inference, OOS diagnostics, and allowed-set losses remain
trainer/evaluator responsibilities.

---

## Audit / Metrics / Traces To Check

These are the important parameters for understanding whether the model is
actually using topology correctly.

### Candidate Score Scale

```text
candidate_node_score_mean_abs
candidate_class_score_mean_abs
candidate_temperature
```

Interpretation:

- node scores near zero: candidate scorer is weak or not learning;
- class scores much larger than node scores: pooling/temperature amplification;
- unstable temperature: candidate score calibration may be unstable.

### Candidate Ranking Confidence

```text
candidate_prediction_entropy
candidate_target_score
candidate_pred_score
candidate_score_gap
```

Interpretation:

- low entropy with poor F1/OOS: overconfident candidate ranking;
- low `candidate_score_gap`: target and prediction are close, likely ranking
  or calibration issue;
- high wrong `candidate_pred_score`: topology candidate path may be pulling to
  an incorrect candidate.

### Duplicate Activity Ambiguity

```text
duplicate_candidate_count_max
```

Interpretation:

- `> 1` means MIL pooling is active;
- compare errors by class where duplicate candidates exist;
- if duplicate-heavy classes dominate errors, Stage 2 needs node-level target
  disambiguation or better MIL loss.

### Trace Flat Attributes

Selective traces now expose these as searchable flat fields:

```text
candidate_temperature
candidate_prediction_entropy
candidate_score_gap
candidate_node_score_mean_abs
candidate_class_score_mean_abs
duplicate_candidate_count_max
```

Use traces for qualitative debugging only. Dissertation/article evidence should
still be based on aggregate metrics, drift windows, calibration/OOS/mask
diagnostics, and seed variance.

---

## Updated Files

Code:

- `src/domain/models/topology_conditioned.py`
- `src/domain/models/factory.py`
- `src/cli.py`
- `src/application/services/structural_trace_payload_builder.py`
- `src/application/use_cases/trainer.py`

Config/UI catalog:

- `configs/ui/config_catalog.yaml`

Tests:

- `tests/domain/test_eopkg_topology_conditioned.py`
- `tests/application/test_structural_trace_payload_builder.py`
- `tests/application/test_cli_flatten_config.py`
- `tests/application/test_experiment_ui_catalog_fields.py`
- `tests/application/test_trainer_forward_stats_logging.py`

Docs:

- `docs/GNN_RUNTIME_MVP2_5.MD`
- `docs/GNN_LEARNING_METODOLOGY.MD`
- `docs/current/project-state.md`
- `docs/current/architecture-debt.md`
- `docs/dissertation_changes.MD`

---

## Verification

Targeted verification passed:

```powershell
.\.venv-modern\Scripts\python.exe -m pytest `
  tests\domain\test_eopkg_topology_conditioned.py `
  tests\application\test_structural_trace_payload_builder.py `
  tests\application\test_cli_flatten_config.py::test_build_model_factory_kwargs_passes_topology_conditioned_candidate_config `
  tests\application\test_experiment_ui_catalog_fields.py::test_catalog_includes_topology_conditioned_model_fields `
  tests\application\test_trainer_forward_stats_logging.py::test_trainer_logs_topology_conditioned_candidate_diagnostics `
  -v
```

Result:

```text
24 passed, 3 warnings
```

Warnings are existing third-party deprecation warnings from PyTorch /
PyTorch-Geometric.

---

## How To Validate Experimentally

Use this as first controlled profile:

```yaml
model.type: EOPKGTopologyConditioned
model.observed_encoder: GATv2
model.struct_encoder: GATv2
model.candidate_scoring: cosine
model.candidate_pooling: logmeanexp
model.candidate_temperature_init: 0.1
model.candidate_temperature_trainable: false
experiment.structural_mode: true
training.learning_strategy: standard
```

Recommended first comparison:

1. train on `v1`;
2. drift test on `v1,v2,v3,v4`;
3. compare with `_Base-UN` and current best structural ablation;
4. inspect candidate diagnostics around version transitions.

Only after Stage 1 is stable should this be combined with
`training.learning_strategy=topology_conditioned`.

---

## Known Limits

1. Stage 1 does not solve unseen activity-class prediction.
2. Output remains fixed `[B, C_train]`.
3. Full business-valid zero-shot requires dynamic candidate contract `[B, C_v]`.
4. Semantic/topological prototypes are not implemented yet.
5. Duplicate activity identity ambiguity is handled by MIL pooling, not solved
   as node-level target disambiguation.
