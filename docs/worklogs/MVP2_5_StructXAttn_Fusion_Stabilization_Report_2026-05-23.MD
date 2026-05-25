# MVP2.5 StructXAttn Fusion Stabilization Report

## Metadata

- `date`: 2026-05-23
- `scope`: StructXAttn residual fusion stabilization
- `status`: implemented
- `plan`: `docs/worklogs/MVP2_5_StructXAttn_Fusion_Stabilization_Plan_2026-05-23.MD`

## Summary

Implemented configurable residual merge semantics for
`model.fusion_mode=StructXAttn`.

The change addresses the diagnostics observed in run
`d9d019d3827548cc920554edbf47a75f`, where the pre-merge structural delta was
moderate but post-merge `LayerNorm` amplified the hidden-state change:

```text
pre_norm_to_observed_ratio  ~= 0.25
post_norm_to_observed_ratio ~= 6.75
```

## Runtime Changes

New model config keys:

```yaml
model.struct_xattn_merge_mode: post_norm_residual
model.struct_xattn_delta_ratio_max: null
```

Supported `model.struct_xattn_merge_mode` values:

1. `post_norm_residual`
   - historical behavior;
   - computes `LayerNorm(node_hidden + delta)`;
   - kept for compatibility and reproduction.

2. `pre_norm_context`
   - recommended stabilization mode;
   - computes `LayerNorm(context)` before residual merge;
   - does not apply `LayerNorm(node_hidden + delta)` after merge.

3. `residual_only`
   - ablation mode;
   - applies raw context delta without context normalization or post-merge
     normalization.

`model.struct_xattn_delta_ratio_max` optionally caps:

```text
mean_abs(delta) / mean_abs(observed)
```

## Files Changed

- `src/domain/models/eopkg_models.py`
  - added merge mode parsing;
  - added optional delta ratio clamp;
  - added affine-free context normalization for StructXAttn;
  - extracted merge helper.

- `src/cli.py`
  - forwards `struct_xattn_merge_mode`;
  - forwards `struct_xattn_delta_ratio_max`.

- `configs/ui/config_catalog.yaml`
  - added catalog entries for the new model keys.

- `tests/domain/test_eopkg_struct_xattn.py`
  - added merge mode validation test;
  - added `pre_norm_context` no-amplification test;
  - added delta ratio clamp test.

- `tests/application/test_cli_flatten_config.py`
  - covers CLI/model factory forwarding.

- `tests/application/test_experiment_ui_catalog_fields.py`
  - covers catalog entries.

- `docs/GNN_RUNTIME_MVP2_5.MD`
  - documents merge modes and recommended stabilization profile.

- `docs/current/project-state.md`
  - records current StructXAttn stabilization behavior.

- `docs/dissertation_changes.MD`
  - records dissertation interpretation note.

## Verification

RED tests before implementation:

```text
tests/domain/test_eopkg_struct_xattn.py::test_struct_xattn_rejects_unknown_merge_mode FAILED
tests/domain/test_eopkg_struct_xattn.py::test_struct_xattn_pre_norm_context_avoids_post_merge_layernorm_amplification FAILED
tests/domain/test_eopkg_struct_xattn.py::test_struct_xattn_delta_ratio_max_clamps_output_delta FAILED
```

Targeted verification after implementation:

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\domain\test_eopkg_struct_xattn.py tests\application\test_cli_flatten_config.py tests\application\test_experiment_ui_catalog_fields.py tests\application\test_desktop_ui_field_registry.py tests\application\test_structural_trace_payload_builder.py -v
```

Result:

```text
41 passed, 3 warnings
```

Architecture guard:

```powershell
.\.venv-modern\Scripts\python.exe tools\architecture_guard.py
```

Result:

```text
[ARCH_GUARD] OK: architecture dependency rules satisfied.
```

Warnings are external dependency deprecation warnings from PyTorch/PyG and do
not affect this change.

## Recommended Next Experiment

Use the same train/eval drift setup as the failed StructXAttn run, but compare:

```yaml
# A: historical reproduction
model.struct_xattn_merge_mode: post_norm_residual
model.struct_xattn_delta_ratio_max: null

# B: stabilized merge
model.struct_xattn_merge_mode: pre_norm_context
model.struct_xattn_delta_ratio_max: null

# C: bounded stabilized merge
model.struct_xattn_merge_mode: pre_norm_context
model.struct_xattn_delta_ratio_max: 0.3
```

Primary diagnostics:

```text
eval_drift_one_pass_struct_xattn_pre_norm_to_observed_ratio
eval_drift_one_pass_struct_xattn_post_norm_to_observed_ratio
drift_window_oos_confidence_mean
drift_window_test_oos
drift_window_test_ece
drift_window_test_set_nll
drift_window_strict_macro_f1
```

Expected first-order effect:

```text
post_norm_to_observed_ratio should no longer explode above pre_norm ratio in
pre_norm_context mode.
```

If OOS confidence remains high after this stabilization, the remaining issue is
likely learning methodology or topology negative sampling, not residual merge
mechanics.

