# MVP2.5 EOPKGTopologyConditioned Impulse Activation Routing Report

- `date`: 2026-05-31
- `status`: implemented
- `scope`: topology-native Impulse Activation Routing for `EOPKGTopologyConditioned`
- `model_type`: `EOPKGTopologyConditioned`
- `plan`: `docs/worklogs/MVP2_5_EOPKGTopologyConditioned_Impulse_Activation_Plan_2026-05-31.MD`

---

## Summary

Implemented experimental Impulse Activation Routing inside
`EOPKGTopologyConditioned`.

This is not an `EOPKGGATv2.fusion_mode`. The implementation keeps the
topology-native candidate contract as the primary prediction surface:

```text
forward_candidate(contract) -> CandidatePredictionOutput
candidate_logits: [B, K_v]
```

`K_v` is the candidate count of the current topology version. Fixed-label
projection remains available only for compatibility metrics and reporting.

The new mode is enabled with:

```yaml
model.type: EOPKGTopologyConditioned
model.topology_conditioning_mode: impulse_activation_routing
training.candidate_contract_mode: candidate_id
training.candidate_identity_mode: topology_native
```

---

## Implemented Runtime Behavior

### Model

Modified:

```text
src/domain/models/topology_conditioned.py
```

Added config fields:

```yaml
model.topology_conditioning_mode: static_candidates | impulse_activation_routing
model.impulse_activation_enabled: false
model.impulse_state_channels:
  - is_last_event
  - prefix_executed_count_log1p
  - prefix_recency_norm
model.impulse_scale_init: 0.1
model.impulse_scale_max: 2.0
model.impulse_gnn_layers: 1
model.impulse_residual_h0: true
model.candidate_temperature_max: 2.0
```

`model.impulse_activation_enabled=true` is accepted as a convenience toggle and
selects `impulse_activation_routing`.

### Impulse Node Initialization

Impulse routing requires `struct_prefix_state_x` and uses the mandatory Pre-LN
merge:

```text
base = layer_norm(base_candidate_features)
impulse = layer_norm(impulse_proj(struct_prefix_state_x))
h0 = base + impulse_scale * impulse
```

This avoids the Post-LN residual pattern that previously made structural signal
scale hard to interpret.

### Structural Propagation

The impulse state is propagated over the current topology using the existing
structural GNN path:

```text
h_gnn = struct_gnn(h0, structural_edge_index)
h_nodes = h_gnn + h0
```

The first implementation supports one or two structural GNN layers and defaults
to one layer to limit over-smoothing.

Topology is not rebuilt inside the model. The model consumes prebuilt
`structural_edge_index`, candidate metadata, and prefix-state tensors from the
existing graph contract.

### Candidate Scoring

Candidate scoring remains topology-native:

```text
candidate_logits[:, j] = score(obs_context, h_nodes[:, j])
```

Supported scoring:

```yaml
model.candidate_scoring: cosine | bilinear
```

For cosine scoring, temperature is clamped. The new default upper clamp is:

```yaml
model.candidate_temperature_max: 2.0
```

Unseen topology candidates with `candidate_class_index=-1` remain scoreable.
They are not removed from the candidate axis.

---

## Trainer / Loss Behavior

Modified:

```text
src/application/use_cases/trainer.py
```

`candidate_id` training continues to use candidate-set cross entropy over
`candidate_logits [B, K_v]`.

When `training.candidate_identity_mode=topology_native` and candidate labels are
available, targets are mapped with raw `target_label` against:

```text
candidate_ids
candidate_labels
```

If candidate labels are absent, the trainer falls back to the existing global
class-index mapping to preserve legacy candidate-id tests and compatibility
stubs.

Fail-fast guard added for:

```yaml
training.candidate_contract_mode: candidate_id
training.candidate_identity_mode: topology_native
model.topology_conditioning_mode: impulse_activation_routing
```

Required fields:

```text
struct_prefix_state_x
candidate_ids
candidate_labels
target_label
```

If any are missing, training/evaluation fails with an explicit `ValueError`.

`group_by_topology` remains not implemented. The current valid policy is still
topology-homogeneous batching with fail-fast protection for mixed topology
batches.

---

## Diagnostics And Tracing

Modified:

```text
src/application/services/structural_trace_payload_builder.py
```

Model/trainer diagnostics now include:

```text
impulse_activation_mean_abs
impulse_activation_max_abs
impulse_to_base_node_ratio
impulse_gnn_oversmoothing_ratio
candidate_temperature
candidate_unseen_score_mean
candidate_seen_score_mean
candidate_seen_unseen_score_gap
```

Selective traces now include candidate-native fields where available:

```text
top_k_candidate_ids
top_k_candidate_labels
top_k_candidate_logits
target_candidate_ids
last_event_candidate_id
```

Flat trace attributes include:

```text
fusion_family
topology_conditioning_mode
candidate_identity_mode
```

For `EOPKGTopologyConditioned`, `fusion_family` is set to
`topology_conditioned`.

---

## Config / UI Surface

Modified:

```text
configs/ui/config_catalog.yaml
src/cli.py
tools/experiment_ui.py
```

Added all new `model.*` keys to the config catalog. The legacy Tkinter UI now
has labels/help text for the impulse routing controls. `src/cli.py` forwards
the new model config values into `create_model()`.

The PySide6 prototype uses the catalog-driven registry, so catalog entries are
the source of truth for exposing these fields there.

---

## Tests Added Or Updated

Modified:

```text
tests/domain/test_eopkg_topology_conditioned.py
tests/application/test_trainer_topology_conditioned_learning.py
tests/application/test_experiment_ui_catalog_fields.py
tests/application/test_cli_flatten_config.py
```

TDD RED/GREEN coverage added for:

1. impulse routing returns `candidate_logits [B, K_v]`, not fixed-vocab logits;
2. moving `is_last_event` in `struct_prefix_state_x` changes candidate logits;
3. unseen candidates remain scoreable;
4. missing `struct_prefix_state_x` fails only when impulse routing is enabled;
5. trainer fail-fast guard for missing impulse candidate contract fields;
6. config catalog coverage for new `model.*` keys;
7. CLI forwarding of new model config keys.

The initial RED run failed as expected because impulse routing was not yet
implemented:

```text
3 failed, 1 passed, 20 deselected
```

After implementation, the same impulse tests passed.

---

## Documentation Updated

Modified:

```text
docs/GNN_RUNTIME_MVP2_5.MD
docs/GNN_LEARNING_STRATEGY.MD
docs/GNN_LEARNING_METODOLOGY.MD
docs/current/project-state.md
docs/current/architecture-debt.md
docs/dissertation_changes.MD
```

The docs now explicitly state:

```text
Impulse Activation Routing belongs to EOPKGTopologyConditioned.
It is not an EOPKGGATv2 fusion mode.
It predicts in topology-native candidate space [B, K_v].
```

Debt status remains conservative: mixed-topology `group_by_topology` batching
is still deferred, and exact BPMN-node supervision is still future work.

---

## Verification

Targeted verification passed:

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\domain\test_eopkg_topology_conditioned.py -v
.\.venv-modern\Scripts\python.exe -m pytest tests\application\test_trainer_topology_conditioned_learning.py -v
.\.venv-modern\Scripts\python.exe -m pytest tests\application\test_experiment_ui_catalog_fields.py -q
.\.venv-modern\Scripts\python.exe -m pytest tests\application\test_structural_trace_payload_builder.py tests\application\test_cli_flatten_config.py -q
```

Results:

```text
tests/domain/test_eopkg_topology_conditioned.py: 24 passed
tests/application/test_trainer_topology_conditioned_learning.py: 10 passed
tests/application/test_experiment_ui_catalog_fields.py: 13 passed
tests/application/test_structural_trace_payload_builder.py + test_cli_flatten_config.py: 23 passed
```

Architecture guard passed:

```powershell
.\.venv-modern\Scripts\python.exe tools\architecture_guard.py
```

Result:

```text
[ARCH_GUARD] OK: architecture dependency rules satisfied.
```

Full suite passed:

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\ -v
```

Result:

```text
546 passed, 4 warnings
```

MVP1 regression gate passed:

```powershell
.\.venv-modern\Scripts\python.exe -m pytest -m mvp1_regression -v
```

Result:

```text
20 passed, 526 deselected, 3 warnings
```

Warnings are existing PyTorch / PyTorch-Geometric deprecation warnings plus one
expected mixed-snapshot warning test.

---

## Recommended First Experiment

Minimal first run:

```yaml
model:
  type: EOPKGTopologyConditioned
  topology_conditioning_mode: impulse_activation_routing
  candidate_scoring: cosine
  candidate_temperature_init: 0.1
  candidate_temperature_min: 0.05
  candidate_temperature_max: 2.0
  candidate_temperature_trainable: false
  impulse_scale_init: 0.1
  impulse_scale_max: 2.0
  impulse_gnn_layers: 1
  impulse_residual_h0: true

training:
  candidate_contract_mode: candidate_id
  candidate_identity_mode: topology_native
  candidate_batch_topology_policy: single_topology_required
  learning_strategy: standard

experiment:
  statistic_enabled: false
  gateway_mode: collapse_for_prediction
```

First comparison:

```text
Baseline observed-only
EOPKGTopologyConditioned static_candidates
EOPKGTopologyConditioned impulse_activation_routing
```

Primary diagnostics:

```text
candidate_missing_target_rate
candidate_oos_rate
candidate_target_in_candidate_set_rate
impulse_to_base_node_ratio
impulse_gnn_oversmoothing_ratio
candidate_seen_unseen_score_gap
```

The first success criterion is not perfect zero-shot F1. The useful first signal
is that moving the execution pointer changes topology-local candidate ranking
without collapsing back to fixed training vocabulary behavior.

---

## Known Limits

1. `group_by_topology` mixed-batch splitting remains deferred.
2. XES supervision is still label-level; exact BPMN-node targets require a
   source that provides stable node/activity-instance identity.
3. Unseen candidates are scoreable, but semantic/text prototypes are not
   implemented yet.
4. The current structural GNN path still uses topology edges but does not add a
   new edge-weight-aware impulse-specific encoder.
5. Fixed-label metrics remain compatibility diagnostics when
   `candidate_class_index=-1` appears.

