# MVP2.5 Relaxed Reachability Mask Implementation Report

Status: implemented
Date: 2026-06-02

## Summary

Implemented `training.process_state_mask_source=relaxed_reachability` as an
approximate mask expansion strategy for flattened/interleaved parallel XES logs.

The previous `lifecycle_active_set` path only adds tasks that are already
started and not completed. For the complex parallel dataset this did not change
metrics because `XESAdapter` already emitted `active_activities_after_complete`
and `DynamicGraphBuilder` already merged them into masks.

The new fallback adds bounded reachable successors from recent completed prefix
anchors over the collapsed prediction topology.

## Runtime Config

```yaml
training.process_state_mask_enabled: true
training.process_state_mask_source: relaxed_reachability
training.process_state_mask_include_direct_successors: true
training.process_state_mask_include_active_candidates: true
training.process_state_mask_relaxed_lookback_events: 8
training.process_state_mask_relaxed_max_depth: 1
training.process_state_mask_relaxed_max_cardinality_ratio: 0.35
```

## Implemented Scope

1. `DynamicGraphBuilder`
   - Added relaxed reachability parameters.
   - Added recent-prefix anchor extraction.
   - Added bounded BFS over `candidate_allowed_masks_by_src`.
   - Added deterministic cardinality cap.
   - Added contract counters:
     - `process_state_mask_active_candidate_count`
     - `process_state_mask_relaxed_candidate_count`

2. CLI
   - Passes relaxed reachability parameters from `training.*` into
     `DynamicGraphBuilder`.

3. UI config catalog
   - Added relaxed reachability keys:
     - `training.process_state_mask_relaxed_lookback_events`
     - `training.process_state_mask_relaxed_max_depth`
     - `training.process_state_mask_relaxed_max_cardinality_ratio`

4. Trainer diagnostics
   - Logs mean active and relaxed candidate counts through forward stats and
     MLflow metric prefixes.

5. Tests
   - Added completion-only parallel test:
     `A complete, B complete -> target C complete`, where `C` is not a direct
     successor of `B` but is reachable from recent anchor `A`.

## Verification

Passed:

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\domain\test_dynamic_graph_builder_parallel_process_state_mask.py tests\application\test_experiment_ui_catalog_fields.py -v
```

Result: `16 passed`

Passed:

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\domain\test_dynamic_graph_builder_masks.py tests\domain\test_dynamic_graph_builder_topology_native_candidates.py tests\application\test_drift_one_pass_evaluation.py -v
```

Result: `19 passed`

## Expected Experiment Signal

Run drift first, not training. Compare the same checkpoint with:

- `process_state_mask_source=lifecycle_active_set`
- `process_state_mask_source=relaxed_reachability`

Primary expected changes:

- `drift_window_target_in_mask_rate` should increase above the current
  lifecycle/direct-successor baseline.
- `drift_window_test_oos` may decrease if predictions respect the wider mask.
- `drift_window_strict_error_but_allowed_rate` may increase because the mask is
  intentionally less strict.
- `drift_window_macro_f1` can improve before `drift_window_strict_macro_f1`.

If `target_in_mask_rate` does not improve, the remaining issue requires true
BPMN marking replay rather than relaxed reachability.

