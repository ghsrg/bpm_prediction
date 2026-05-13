# One-Pass Drift Window Evaluation Report

- `date`: 2026-05-13
- `status`: implemented
- `related_plan`: `docs/worklogs/MVP2_5_One_Pass_Drift_Window_Evaluation_Plan_2026-05-13.MD`
- `related_debt`: `stats_backed_structural_payload_caching`

---

## Root Cause

`eval_drift` previously generated chronological trace windows and called
`_build_loader(window_traces)` for every window. With dense sliding windows this
rebuilt graph tensors many times. For stats-backed structure this also repeated
stats/structural payload materialization work.

For example, `5054` traces with `drift_window_size=100` and
`drift_window_sliding=10` creates about `496` windows, so the same trace prefixes
can be rebuilt and re-evaluated many times.

---

## Implemented Changes

### Graph Dataset Metadata

`src/cli.py` now writes trace metadata into graph samples produced by both graph
dataset builders:

- `trace_idx`
- `prefix_idx`
- `trace_start_ts`
- `trace_end_ts`

`GRAPH_DATASET_CACHE_SCHEMA` was bumped from `2` to `3` so old graph caches
without `trace_idx` are not silently treated as optimized drift-compatible
caches.

### One-Pass Drift Evaluation

`src/application/use_cases/trainer.py` now passes `prebuilt_test_dataset` into
`_run_eval_drift`.

When the prebuilt dataset has `trace_idx`, trainer uses:

```text
Drift evaluation path: one_pass_prebuilt_dataset
```

The optimized path:

1. creates one non-shuffled loader over the prebuilt test dataset;
2. runs one model inference pass;
3. stores compact per-sample records;
4. resolves drift windows through `trace_idx`;
5. computes window metrics by array aggregation.

The implementation does not store full `N x C` probability matrices. It stores
scalar values such as `confidence`, `correct`, `top3_hit`, mask flags, set NLL,
prefix length, and version label.

### Fallback

Old datasets or non-prebuilt runs without `trace_idx` still use:

```text
Drift evaluation path: legacy_raw_trace_windows
```

This keeps backward compatibility, but it is slower.

---

## Tests Added

### `tests/application/test_cli_graph_dataset_cache.py`

- `test_build_graph_dataset_sharded_attaches_trace_metadata`

### `tests/application/test_trainer_preloaded_data.py`

- `test_eval_drift_receives_prebuilt_test_dataset`

### `tests/application/test_drift_one_pass_evaluation.py`

- `test_collect_drift_inference_records_is_compact`
- `test_record_metrics_match_evaluate_test_for_full_dataset`
- `test_resolve_drift_window_record_indices_uses_trace_idx_ranges`
- `test_run_eval_drift_uses_one_pass_prebuilt_dataset_without_legacy_graph_rebuild`
- `test_run_eval_drift_falls_back_when_prebuilt_dataset_lacks_trace_idx`
- `test_one_pass_drift_rows_preserve_legacy_output_keys`
- `test_one_pass_drift_logs_legacy_tracker_metric_names`

---

## Verification Run

Focused pool:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/application/test_cli_graph_dataset_cache.py tests/application/test_trainer_preloaded_data.py tests/application/test_drift_one_pass_evaluation.py -v
```

Result:

```text
19 passed
```

Final implementation verification:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/application/test_cli_graph_dataset_cache.py tests/application/test_trainer_preloaded_data.py tests/application/test_drift_one_pass_evaluation.py tests/application/test_evaluator_metrics.py tests/application/test_drift_eval_no_label_leakage.py tests/application/test_window_generator.py -v
```

Result:

```text
30 passed
```

Architecture guard:

```powershell
.\.venv\Scripts\python.exe tools\architecture_guard.py
```

Result:

```text
[ARCH_GUARD] OK: architecture dependency rules satisfied.
```

---

## How To Re-run `eval_drift`

Because graph dataset cache schema changed to `3`, rebuild graph cache before
measuring runtime. Do not reuse old cache entries without `trace_idx`.

Recommended rerun sequence:

1. Run a small smoke run first:

```yaml
preset_name: S_Str-USc-drift-loan
experiment.fraction: 0.01
```

2. Check logs contain:

```text
Drift evaluation path: one_pass_prebuilt_dataset
Drift one-pass records: samples=<n> traces=<n> max_trace_idx=<n>
```

3. Confirm logs do not show repeated graph building inside each drift window.

4. Run full:

```yaml
preset_name: S_Str-USc-drift-loan
```

Expected runtime shape:

- one graph build/load phase for test dataset;
- one model inference pass over test graphs;
- cheap drift-window aggregation.

---

## Remaining Risk

- Full real `S_Str-USc-drift-loan` validation is still required.
- `snapshot_homogeneous_batching` remains separate active debt.
- Old graph caches fall back to legacy behavior; they should be rebuilt for
  runtime comparison.
