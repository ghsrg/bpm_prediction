# MVP2.5 GATv2 Mask Post-Filter Control Report

## Summary

Implemented the fixed-vocabulary reviewer-control baseline `GATv2 + Mask`.
The control keeps `BaselineGATv2` as the model and applies a forced hard
topology post-filter during inference through `training.mask_guided_policy:
hard`.

## Implemented Changes

- Added `training.mask_guided_policy` to `ModelTrainer`.
  - `auto`: existing reliability-based behavior.
  - `off`: no mask effect.
  - `soft`: subtract mask penalty from invalid logits.
  - `hard`: suppress invalid logits before softmax/argmax.
- Added config catalog field `training.mask_guided_policy`.
- Added active UI presets:
  - `_Base-MN-f1`
  - `_Base-MN-f1-drift`
- Updated runtime documentation to define `GATv2 + Mask` as a fixed-vocabulary
  reviewer-control baseline, not an EOPKG topology-native model.
- Updated current project state with the control baseline status.

## Intended Experimental Use

Use `_Base-MN-f1-drift` with the same checkpoint as `_Base-UN-f1`.
This isolates inference-time topology mask post-filtering:

1. Train `_Base-UN-f1`.
2. Run `_Base-UN-f1-drift`.
3. Run `_Base-MN-f1-drift` with the same checkpoint.
4. Compare against EOPKG-WI and full EOPKG synthetic runs.

Expected pattern:

- `_Base-MN-f1-drift` should reduce OOS versus `_Base-UN-f1-drift`.
- `_Base-MN-f1-drift` should remain limited by fixed `C_train`.
- Full EOPKG should remain the model that can use topology-native candidate
  scoring and impulse routing.

## Verification

Passed:

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\application\test_evaluator_metrics.py::test_mask_guided_policy_can_force_hard_in_eval_without_reliability_state tests\application\test_evaluator_metrics.py::test_mask_guided_hard_post_filter_changes_fixed_vocab_prediction_to_allowed_class tests\application\test_evaluator_metrics.py::test_mask_guided_policy_hard_when_reliable_and_soft_when_not tests\application\test_evaluator_metrics.py::test_mask_guided_logits_hard_and_soft_behaviour -v
```

Result: `4 passed`.

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\application\test_experiment_ui_catalog_fields.py::test_catalog_includes_mask_guided_policy -v
```

Result: `1 passed`.

```powershell
.\.venv-modern\Scripts\python.exe tools\architecture_guard.py
```

Result: `[ARCH_GUARD] OK`.

Preset JSON validation:

```text
OK
```

## Notes

`off` in YAML must be quoted in enum lists, otherwise YAML parses it as boolean
`False`. The catalog uses `'off'` explicitly.

