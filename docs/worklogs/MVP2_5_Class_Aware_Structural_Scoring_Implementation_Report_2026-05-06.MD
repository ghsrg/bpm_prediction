# MVP2.5 Class-Aware Structural Scoring Implementation Report

Date: 2026-05-06
Status: implemented and verified
Plan source: `docs/worklogs/MVP2_5_Class_Aware_Structural_Scoring_Plan_2026-05-06.MD`

## Summary

Implemented `ClassAwareStructuralScoring` for `EOPKGGATv2`.

The new mode no longer injects a single global structural context into every
class. It computes candidate-level structural scores and adds them to observed
class logits:

```text
logit_j = observed_logit_j + gamma * structural_score_j
```

The initial structural scale is conservative:

```text
gamma = 0.1
```

## Implemented Changes

- Added fusion-mode normalization:
  - `Attention` -> `ClassMeanAttention`
  - `Concat` -> `ClassMeanConcat`
  - `struct_pool_concat` -> `ClassMeanConcat`
  - `ClassAwareAdditive` -> `ClassAwareStructuralScoring`
  - `ClassAwareAttention` -> `ClassAwareStructuralScoring`
  - `ClassAwareStructuralScoring` -> `ClassAwareStructuralScoring`
- Added `ClassAwareStructuralScoring` in `src/domain/models/eopkg_models.py`.
- Added explicit `struct_node_to_class_index` contract field.
- Added late node-to-class aggregation:

```text
structural_logit_v = score(obs_context, h_struct_v)
structural_score_j = LogSumExp({structural_logit_v | node_to_class[v] = j})
```

- Preserved legacy behavior for class-space structures without `struct_x`.
- Added structural logit contribution diagnostics:
  - `observed_logits_mean_abs`
  - `structural_logits_mean_abs`
  - `structural_logits_max_abs`
  - `structural_to_observed_logit_ratio`
- Added trainer tracking metrics with stage prefixes, for example:
  - `train_structural_logits_mean_abs`
  - `validation_structural_logits_mean_abs`
  - `drift_window_structural_logits_mean_abs`
- Updated UI config catalog:
  - `model.fusion_mode` dropdown includes `ClassMeanAttention`, `ClassMeanConcat`, `ClassAwareStructuralScoring`.
  - `model.pooling_strategy` is a dropdown.
  - `model.graph_strategy` is readonly with one value: `sequential_prefix`.
- Added debug configs:
  - `configs/experiments/debug_base_un_fraction_001.yaml`
  - `configs/experiments/debug_s_classaware_unc_fraction_001.yaml`
- Updated runtime/data-model documentation:
  - `docs/GNN_RUNTIME_MVP2_5.MD`
  - `docs/DATA_MODEL_MVP2_5.MD`
  - `docs/current/project-state.md`

## Where To Verify During Training Or Drift

Check trainer log lines:

```text
Forward stats [train]: ... observed_logits_mean_abs=... structural_logits_mean_abs=... structural_logits_max_abs=... structural_to_observed_logit_ratio=...
Forward stats [validation]: ...
Forward stats [inference]: ...
Forward stats [eval_drift]: ...
```

Interpretation:

- `structural_logits_mean_abs ~= 0`: structure contributes almost nothing.
- `structural_to_observed_logit_ratio << 1`: structure is weak relative to observed logits.
- `structural_to_observed_logit_ratio ~= 0.1..0.5`: reasonable early range.
- `structural_to_observed_logit_ratio > 1`: structure may dominate observed evidence and should be treated as suspicious.

For MLflow, inspect metrics with prefixes:

```text
train_structural_logits_mean_abs
validation_structural_logits_mean_abs
inference_structural_logits_mean_abs
drift_window_structural_logits_mean_abs
```

## Verification

Passed:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/domain/test_eopkg_gatv2_dual_encoder.py tests/contracts/test_graph_tensor_contract_mvp2_optional_fields.py tests/domain/test_dynamic_graph_builder_masks.py tests/application/test_trainer_forward_stats_logging.py -v
```

Result:

```text
32 passed
```

Passed:

```powershell
.\.venv\Scripts\python.exe -m pytest -m mvp1_regression -v
```

Result:

```text
20 passed, 282 deselected
```

Passed:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -v
```

Result:

```text
302 passed
```

Passed:

```powershell
.\.venv\Scripts\python.exe tools\architecture_guard.py
```

Result:

```text
[ARCH_GUARD] OK: architecture dependency rules satisfied.
```

## Debug Smoke Runs

Executed smoke configs with `fraction=0.01`:

```powershell
.\.venv\Scripts\python.exe main.py --config configs\experiments\debug_base_un_fraction_001.yaml
.\.venv\Scripts\python.exe main.py --config configs\experiments\debug_s_classaware_unc_fraction_001.yaml
```

Both completed without crashing.

Important caveat: these smoke runs are not quality evidence. With `fraction=0.01`
and `mode=train`, cascade selection used only `v1`:

```text
DATA_SPLIT_VERSION_COLLAPSE: source log has 4 versions but cascade-selected data has 1 version
```

The runs also triggered numeric guard warnings and produced zero F1. Treat these
configs as crash/debug smoke only. For research-grade comparison, use larger
fractions or an explicit split that preserves the intended version coverage.

## Open Follow-Up

The current debug smoke demonstrates that the new mode is wired and does not
break the pipeline, but it is not sufficient to prove structural benefit.

Recommended next step: create a small but valid loan debug protocol where train,
validation, and test preserve meaningful version/activity coverage while still
being small enough for fast iteration.

