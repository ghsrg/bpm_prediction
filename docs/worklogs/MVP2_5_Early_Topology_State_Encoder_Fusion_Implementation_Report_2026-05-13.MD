# MVP2.5 Early TopologyStateEncoder Fusion Implementation Report

- `date`: 2026-05-13
- `status`: implemented
- `scope`: experimental early/input-level structural fusion for `EOPKGGATv2`
- `plan`: `docs/worklogs/MVP2_5_Early_Topology_State_Encoder_Fusion_Plan_2026-05-13.MD`

## Summary

Implemented `model.fusion_mode=TopologyStateEncoder` as an experimental
early-fusion ablation. The mode projects prefix execution state onto structural
nodes before the structural GNN and produces class logits via node-to-class
pooling.

The existing modes remain supported:

- `Attention` / `ClassMeanAttention`
- `Concat` / `ClassMeanConcat`
- `ClassAwareAdditive` / `ClassAwareStructuralScoring`
- observed-only path when `experiment.structural_mode=false`

## Runtime Behavior

New optional contract field:

```text
struct_prefix_state_x: FloatTensor [|V|, 6] per graph, [B, |V|, 6] in batch
```

Feature order:

1. `prefix_executed_count_log1p`
2. `prefix_was_executed`
3. `prefix_is_last_event`
4. `prefix_last_position_norm`
5. `prefix_recency_norm`
6. `prefix_is_active_after_complete`

`DynamicGraphBuilder` emits this tensor when `struct_node_to_class_index` is
available. Non-target structural nodes (`-1`) receive zero prefix-state
features.

`EOPKGGATv2` flow for `TopologyStateEncoder`:

```text
base_v = structural_identity_v + structural_stats_beta * stats_projection(struct_x_v)
prefix_v = W_prefix struct_prefix_state_x_v
gate_v = sigmoid(W_gate struct_prefix_state_x_v + topology_state_gate_init_bias)
h0_v = LayerNorm(base_v + beta * gate_v * prefix_v)
h_topology_v = StructuralGNN(h0_v, structural_edge_index)
node_logit_v = W_node h_topology_v
class_logit_j = pool(node_logit_v where struct_node_to_class_index[v] = j)
```

Default pooling is `logmeanexp`, not raw `logsumexp`, to reduce duplicate
activity inflation.

## Configuration

Added model config keys:

- `model.topology_state_beta`
- `model.topology_state_beta_max`
- `model.topology_state_gate_init_bias`
- `model.topology_state_class_pooling`
- `model.topology_state_dropout`

Added `TopologyStateEncoder` to `model.fusion_mode` UI/catalog options.

Graph dataset cache schema was bumped from `3` to `4`, because
`TopologyStateEncoder` requires `struct_prefix_state_x` in cached graph samples.

## Diagnostics

Trainer forward stats now log and track:

- `topology_state_prefix_mean_abs`
- `topology_state_prefix_max_abs`
- `topology_state_entropy`
- `topology_state_mean_class_cardinality`
- `topology_state_max_class_cardinality`
- `topology_state_gate_mean`
- `topology_state_gate_max`

Use these to detect:

- near-zero prefix-state participation;
- overconfident topology-state logits via low entropy;
- duplicate activity pressure via class cardinality;
- excessive or inactive prefix gate values.

## Compatibility Notes

- `experiment.structural_mode=false` keeps observed-only behavior even if
  `model.fusion_mode=TopologyStateEncoder`.
- Existing late-fusion modes are unchanged.
- `TopologyStateEncoder` requires `struct_prefix_state_x`; old graph caches
  must be rebuilt for this mode.
- New trainable topology-state projection layers are registered during model
  initialization, before optimizer creation.

## 2026-05-13 Hotfix

First runtime attempt with `preset_name=S_Str-UNcTop-loan` failed during dry
run:

```text
TopologyStateEncoder requires struct_prefix_state_x.
```

Root cause: `DynamicGraphBuilder` produced `struct_prefix_state_x`, and trainer
could consume it, but `src/cli.py` did not copy `struct_prefix_state_x` or
`struct_node_to_class_index` into PyG `Data` during graph dataset construction.
The bug affected both in-memory graph dataset construction and sharded disk
spill construction.

Fix:

- `_build_graph_dataset_sharded()` now stores `struct_node_to_class_index` and
  `struct_prefix_state_x` in per-prefix `Data`.
- `_build_graph_dataset()` now stores the same fields for non-sharded runs.
- `struct_prefix_state_x` remains per-prefix data and is not deduplicated into
  shared structural payloads.

Added regression tests:

- `test_build_graph_dataset_sharded_preserves_topology_state_prefix_tensor`
- `test_build_graph_dataset_preserves_topology_state_prefix_tensor`

## Verification

Focused regression suite:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/contracts/test_graph_tensor_contract_mvp2_optional_fields.py tests/domain/test_dynamic_graph_builder_topology_state.py tests/domain/test_eopkg_topology_state_encoder.py tests/domain/test_eopkg_models_forward.py tests/domain/test_eopkg_gatv2_dual_encoder.py tests/application/test_trainer_topology_state_contract.py tests/application/test_trainer_preloaded_data.py tests/application/test_trainer_forward_stats_logging.py tests/application/test_cli_flatten_config.py -v
```

Result:

```text
74 passed, 11 warnings
```

Warnings are existing third-party deprecations from `mlflow` / `packaging`.

## How To Run

Use a structural EOPKG preset and set:

```yaml
experiment.structural_mode: true
model.fusion_mode: TopologyStateEncoder
model.topology_state_class_pooling: logmeanexp
```

Recommended first ablation:

```yaml
model.topology_state_beta: 0.25
model.topology_state_beta_max: 1.0
model.topology_state_gate_init_bias: -2.0
```

Then compare against:

- `fusion_mode=ClassAwareStructuralScoring`
- `fusion_mode=ClassMeanAttention`
- `experiment.structural_mode=false`

For drift evaluation, rebuild graph cache first because schema `4` is required.

## Interpretation

This implementation should be treated as an ablation for fusion-level validity:
early topology-state fusion versus late class-aware structural scoring. It is
not expected to automatically solve structural drift generalization. If it
improves seen-version metrics but worsens unseen-version strict F1, that is
evidence of structural overfitting / negative transfer and supports the need
for a separate transfer methodology.
