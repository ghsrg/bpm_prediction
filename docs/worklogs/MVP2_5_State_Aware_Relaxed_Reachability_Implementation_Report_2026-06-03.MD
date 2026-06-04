# MVP2.5 State-Aware Relaxed Reachability Implementation Report

Status: implemented
Date: 2026-06-03

## Summary

Implemented state-aware refinement for
`training.process_state_mask_source=relaxed_reachability`.

The previous relaxed mask expanded from recent prefix anchors, which improved
`target_in_mask_rate` and reduced OOS, but could over-allow stale completed
parallel siblings. That converted OOS errors into
`strict_error_but_allowed_rate` and degraded strict next-activity F1.

The new behavior keeps relaxed reachability approximate, but adds completed
candidate suppression with safety exceptions.

## Runtime Config

```yaml
training.process_state_mask_enabled: true
training.process_state_mask_source: relaxed_reachability
training.process_state_mask_relaxed_lookback_events: 8
training.process_state_mask_relaxed_max_depth: 1
training.process_state_mask_relaxed_max_cardinality_ratio: 0.35
training.process_state_mask_relaxed_suppress_completed: true
training.process_state_mask_relaxed_anchor_policy: open_successors
training.process_state_mask_relaxed_loop_policy: keep_direct_successor_repeats
```

## Implemented Scope

1. `DynamicGraphBuilder`
   - Added `process_state_mask_relaxed_suppress_completed`.
   - Added `process_state_mask_relaxed_anchor_policy`.
   - Added `process_state_mask_relaxed_loop_policy`.
   - Added prefix process-state extraction:
     - completed tokens;
     - active tokens;
     - completed instance IDs;
     - active instance IDs.
   - Suppresses completed relaxed-only candidates only when they are not:
     - active;
     - explicit active candidates;
     - direct successors of the last completed task;
     - direct loop/rework repeats.
   - Keeps direct successors and active candidates as authoritative additions.

2. CLI
   - Passes the new relaxed reachability settings into `DynamicGraphBuilder`.

3. Config Catalog
   - Added new keys to `configs/ui/config_catalog.yaml`.
   - Desktop UI catalog tests cover the new fields and enum values.

4. Trainer Diagnostics
   - Added forward stats / MLflow metrics:
     - `*_process_state_mask_relaxed_raw_candidate_count`
     - `*_process_state_mask_relaxed_suppressed_completed_count`
     - `*_process_state_mask_relaxed_final_candidate_count`
     - `*_process_state_mask_relaxed_anchor_count`
     - `*_process_state_mask_relaxed_skipped_closed_anchor_count`
     - `*_process_state_mask_target_suppressed_by_completed_filter_rate`
     - `*_process_state_mask_completed_suppression_rate`

5. Tests
   - Added domain coverage for:
     - completed parallel sibling suppression;
     - loop/direct successor repeat retention;
     - active new instance retention when the same token was previously completed.

## How to Verify in Experiments

No retraining is required. Re-run `eval_drift` using the same checkpoint.

Compare:

### Current relaxed compatibility

```yaml
training.process_state_mask_source: relaxed_reachability
training.process_state_mask_relaxed_suppress_completed: false
training.process_state_mask_relaxed_anchor_policy: recent_prefix
```

### State-aware relaxed

```yaml
training.process_state_mask_source: relaxed_reachability
training.process_state_mask_relaxed_suppress_completed: true
training.process_state_mask_relaxed_anchor_policy: open_successors
training.process_state_mask_relaxed_loop_policy: keep_direct_successor_repeats
```

Expected useful signal:

- `target_in_mask_rate` remains higher than lifecycle/direct baseline.
- `test_oos` remains lower than lifecycle/direct baseline.
- `strict_error_but_allowed_rate` decreases versus naive relaxed reachability.
- `ambiguous_prefix_rate` decreases versus naive relaxed reachability.
- `process_state_mask_target_suppressed_by_completed_filter_rate == 0`.
- `process_state_mask_completed_suppression_rate > 0` on the complex parallel dataset.

If `target_suppressed_by_completed_filter_rate > 0`, the suppression is too
aggressive for the dataset and the next step should be true BPMN marking replay
or simulator-provided enabled candidate state.

## Verification

Passed:

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\domain\test_dynamic_graph_builder_parallel_process_state_mask.py -v
```

Result: `5 passed`

Passed:

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\domain\test_dynamic_graph_builder_parallel_process_state_mask.py tests\application\test_experiment_ui_catalog_fields.py -v
```

Result: `19 passed`

Additional verification is listed in the final assistant report for this task.

Passed:

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\application\test_drift_one_pass_evaluation.py -v
```

Result: `8 passed`

Passed:

```powershell
.\.venv-modern\Scripts\python.exe tools\architecture_guard.py
```

Result: `[ARCH_GUARD] OK`
