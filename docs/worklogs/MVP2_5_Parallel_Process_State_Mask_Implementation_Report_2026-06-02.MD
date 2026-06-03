# MVP2.5 Parallel Process-State Mask Implementation Report

Status: implemented
Date: 2026-06-02

## Summary

Implemented optional process-state-aware mask expansion for parallel/interleaved
XES logs. The feature reconstructs active tasks from lifecycle events and merges
them into:

- `allowed_target_mask`
- `candidate_allowed_target_mask`
- existing `struct_prefix_state_x[:, 5]` active-candidate channel

Tensor shape is unchanged: `struct_prefix_state_x` remains `[|V|, 6]`.

## Runtime Config

```yaml
training.process_state_mask_enabled: true
training.process_state_mask_source: lifecycle_active_set
training.process_state_mask_include_direct_successors: true
training.process_state_mask_include_active_candidates: true
```

Default behavior is unchanged because `process_state_mask_enabled=false`.

## Implemented Scope

1. `DynamicGraphBuilder`
   - Added process-state mask configuration.
   - Added lifecycle active-set reconstruction from
     `lifecycle:transition=start|complete`.
   - Uses `sim:activity_instance_id` to track active task instances.
   - Merges reconstructed active candidates into class and candidate masks.
   - Reuses the existing active candidate state channel.

2. CLI
   - Passes `training.process_state_mask_*` values into `DynamicGraphBuilder`.

3. UI config catalog
   - Added `training.process_state_mask_*` keys for desktop UI/presets.

4. Tests
   - Added regression test for a parallel case where target `C` remains active
     after `B complete`, while direct successor from `B` points to `D`.
   - Added catalog test for new config keys.

## Verification

Passed:

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\domain\test_dynamic_graph_builder_parallel_process_state_mask.py tests\domain\test_dynamic_graph_builder_masks.py tests\domain\test_dynamic_graph_builder_topology_native_candidates.py tests\domain\test_dynamic_graph_builder_topology_state.py tests\application\test_experiment_ui_catalog_fields.py -v
```

Result: `27 passed`

Passed:

```powershell
.\.venv-modern\Scripts\python.exe tools\architecture_guard.py
```

Result: `[ARCH_GUARD] OK`

## Known Limits

- `relaxed_reachability` is registered as a future source option but currently
  does not expand masks.
- Completion-only XES logs cannot reconstruct active task state reliably.
- This feature improves mask correctness for parallel/interleaved lifecycle logs;
  it does not solve candidate ranking ambiguity by itself.

## Suggested Experiment

For the complex parallel dataset, rerun drift with:

```yaml
training.process_state_mask_enabled: true
training.process_state_mask_source: lifecycle_active_set
training.process_state_mask_include_direct_successors: true
training.process_state_mask_include_active_candidates: true
```

Primary checks:

- `drift_window_target_in_mask_rate`
- `drift_window_pred_in_mask_rate`
- `drift_window_test_oos`
- `drift_window_strict_error_but_allowed_rate`
- `drift_window_macro_f1`
- `drift_window_strict_macro_f1`

