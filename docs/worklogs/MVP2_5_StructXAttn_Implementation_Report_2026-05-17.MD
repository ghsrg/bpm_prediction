# MVP2.5 StructXAttn Implementation Report

Date: 2026-05-17
Status: implemented
Scope: `model.fusion_mode=StructXAttn`

## Summary

Implemented TokenXAttn-style structural fusion for `EOPKGGATv2`.

`StructXAttn` keeps the observed IG encoder active and lets observed node
representations query structural node representations before pooling and
classification.

## Implemented Runtime Behavior

### Fusion Mode

Canonical config:

```yaml
model.fusion_mode: StructXAttn
```

Normalized internal mode:

```text
struct_xattn
```

Supported aliases:

```text
StructXAttn
StructuralCrossAttention
LayerwiseStructuralCrossAttention
DualStreamTokenAttention
```

### Layer Placement

Implemented:

```yaml
model.struct_xattn_layers: post_conv2
model.struct_xattn_layers: after_each_conv
```

`post_conv2`:

```text
conv1 -> conv2 -> StructXAttn -> pooling -> classifier
```

`after_each_conv`:

```text
conv1 -> StructXAttn_1 -> conv2 -> StructXAttn_2 -> pooling -> classifier
```

### TokenXAttn Formula

```text
H_ig = observed node hidden states
H_struct = structural node hidden states

Q = W_q H_ig
K = W_k H_struct
V = W_v H_struct

C = softmax(QK^T / sqrt(d))V
H_ig' = LayerNorm(H_ig + gamma * gate * C)
```

`gamma` is learnable and clamped by:

```yaml
model.struct_xattn_scale_max
```

The gate is controlled by:

```yaml
model.struct_xattn_use_gate
model.struct_xattn_gate_init_bias
```

## Training Methodology

Implemented train-only correct-vs-corrupted topology objective.

Config:

```yaml
training.struct_xattn_contrastive_enabled: true
training.struct_xattn_contrastive_weight: 0.03
training.struct_xattn_contrastive_margin: 0.05
training.struct_xattn_contrastive_temperature: 0.1
training.struct_xattn_contrastive_max_loss: 0.5
training.struct_xattn_contrastive_warmup_epochs: 3
training.struct_xattn_corruption_policy: edge_drop
training.struct_xattn_corruption_edge_drop_prob: 0.15
training.struct_xattn_corruption_feature_noise_std: 0.0
```

Loss:

```text
raw_gap = margin + CE_correct - CE_corrupt
L_contrastive = clamp(tau * softplus(raw_gap / tau), max=max_loss)
L_total = CE_correct + weight * L_contrastive
```

Important:

- corrupted topology forward is used only during train;
- validation/test/eval_drift use only the correct structural payload;
- corruption works on already prepared structural tensors and does not rebuild
  topology or query raw sources.

## Diagnostics

Model/trainer diagnostics added:

```text
struct_xattn_context_mean_abs
struct_xattn_delta_mean_abs
struct_xattn_to_observed_ratio
struct_xattn_attention_entropy
struct_xattn_scale
struct_xattn_gate_mean
struct_xattn_l1_delta_mean_abs
struct_xattn_l2_delta_mean_abs
struct_xattn_l1_attention_entropy
struct_xattn_l2_attention_entropy
```

Contrastive diagnostics:

```text
struct_xattn_correct_ce
struct_xattn_corrupt_ce
struct_xattn_correct_vs_corrupt_ce_delta
struct_xattn_contrastive_loss
struct_xattn_prediction_flip_rate
```

Main interpretation:

```text
struct_xattn_correct_vs_corrupt_ce_delta > 0
```

means correct topology gives lower CE than corrupted topology.

## Files Changed

Code:

```text
src/domain/models/eopkg_models.py
src/application/use_cases/trainer.py
src/cli.py
```

Config/UI:

```text
configs/ui/config_catalog.yaml
```

Tests:

```text
tests/domain/test_eopkg_struct_xattn.py
tests/domain/test_eopkg_models_forward.py
tests/application/test_trainer_struct_xattn_training.py
tests/application/test_trainer_forward_stats_logging.py
tests/application/test_cli_flatten_config.py
```

Docs:

```text
docs/GNN_RUNTIME_MVP2_5.MD
docs/current/project-state.md
docs/dissertation_changes.MD
```

## How To Verify Manually

Recommended smoke config:

```yaml
experiment.structural_mode: true
model.type: EOPKGGATv2
model.fusion_mode: StructXAttn
model.struct_xattn_layers: post_conv2
training.struct_xattn_contrastive_enabled: false
```

Recommended research config:

```yaml
experiment.structural_mode: true
model.type: EOPKGGATv2
model.fusion_mode: StructXAttn
model.struct_xattn_layers: after_each_conv
training.struct_xattn_contrastive_enabled: true
```

Check MLflow/log metrics:

```text
*_struct_xattn_to_observed_ratio
*_struct_xattn_attention_entropy
train_struct_xattn_correct_vs_corrupt_ce_delta
train_struct_xattn_contrastive_loss
```

## Known Limitations

`StructXAttn` inherits existing structural payload batching behavior:

```text
one batch forward can use the first graph structural payload
```

This is already tracked as:

```text
docs/current/architecture-debt.md#snapshot_homogeneous_batching
```

For dissertation-grade StructXAttn experiments, prefer homogeneous batches or
interpret results with this limitation in mind until that debt is closed.

## Verification

Targeted verification:

```text
tests/domain/test_eopkg_struct_xattn.py
tests/domain/test_eopkg_models_forward.py
tests/application/test_trainer_struct_xattn_training.py
tests/application/test_trainer_forward_stats_logging.py
tests/application/test_cli_flatten_config.py
tests/application/test_experiment_ui_catalog_fields.py
```

Result:

```text
56 passed
```

Architecture guard:

```text
[ARCH_GUARD] OK: architecture dependency rules satisfied.
```

Full suite first run:

```text
385 passed
1 failed
```

The failure was unrelated to StructXAttn:

```text
tests/domain/test_dynamic_graph_builder_cache_policy.py::test_cache_policy_full_caches_dto_for_strict_asof
```

Root cause:

```text
DynamicGraphBuilder._resolve_dto disabled DTO cache for strict_asof even though
the cache key already includes as_of_ts.
```

Fix applied:

```text
src/domain/services/dynamic_graph_builder.py
```

The repeat verification command could not be run because the Codex environment
rejected further escalated test execution due to usage limit. Re-run required:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\domain\test_dynamic_graph_builder_cache_policy.py -v
.\.venv\Scripts\python.exe -m pytest tests\ -v
```
