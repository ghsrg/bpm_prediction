# MVP2.5 Selective MLflow Tracing Implementation Report

Date: 2026-05-20
Status: implemented

## Scope

Implemented selective structural prediction tracing for qualitative debugging
and explanation of structural-path behavior during test and one-pass drift
evaluation.

This feature is intentionally bounded. It is not a replacement for aggregate
metrics, drift-window metrics, calibration/OOS diagnostics, mask diagnostics, or
seed variance.

## Implemented

1. Added application trace port:
   - `src/application/ports/trace_recorder_port.py`

2. Added trace sampling policy:
   - `src/application/services/trace_sampling_policy.py`
   - Supports `interesting`, `random`, and `all` policies.
   - Interesting reasons include:
     - `strict_error_but_allowed`
     - `oos_prediction`
     - `high_confidence_error`
     - `low_confidence_correct`
     - `version_first`

3. Added structural prediction trace payload builder:
   - `src/application/services/structural_trace_payload_builder.py`
   - Detaches tensors with CPU serialization before payload construction.
   - Produces nested explanation payload plus flat searchable attributes.

4. Added MLflow tracing adapter:
   - `src/infrastructure/tracking/mlflow_trace_recorder.py`
   - Uses `mlflow.start_span()` when available.
   - Falls back to rank/PID-scoped JSONL files when span tracing is unavailable
     or fails.
   - Nonzero distributed ranks skip fallback writes by default.

5. Integrated tracing into `ModelTrainer`:
   - test evaluation
   - one-pass `eval_drift`
   - logs tracing overhead metrics:
     - `tracing_events_recorded`
     - `tracing_record_time_ms_total`
     - `tracing_record_time_ms_per_event`

6. Integrated tracing into CLI composition root:
   - builds recorder from `tracking.tracing.*`
   - passes recorder to `ModelTrainer`
   - logs JSONL fallback as MLflow artifact before closing tracker

7. Added config catalog fields:
   - `tracking.tracing.enabled`
   - `tracking.tracing.stages`
   - `tracking.tracing.sample_policy`
   - `tracking.tracing.max_traces_per_run`
   - `tracking.tracing.max_traces_per_stage`
   - `tracking.tracing.max_traces_per_version`
   - `tracking.tracing.random_sample_rate`
   - `tracking.tracing.high_confidence_threshold`
   - `tracking.tracing.low_confidence_threshold`
   - `tracking.tracing.top_k`
   - `tracking.tracing.log_json_artifact_fallback`
   - `tracking.tracing.allow_nonzero_rank_fallback`

## How To Enable

```yaml
tracking:
  enabled: true
  tracing:
    enabled: true
    stages: test,eval_drift_one_pass
    sample_policy: interesting
    max_traces_per_run: 80
    max_traces_per_stage: 40
    max_traces_per_version: 10
    random_sample_rate: 0.0
    high_confidence_threshold: 0.8
    low_confidence_threshold: 0.4
    top_k: 5
    log_json_artifact_fallback: true
```

## How To Inspect

In MLflow 3, inspect trace spans and filter by flat attributes such as:

```text
reason
fusion_mode
process_version
strict_correct
strict_error_but_allowed
target_in_mask
pred_in_mask
prediction_entropy
structural_to_observed_logit_ratio
struct_xattn_gate_mean
```

If MLflow tracing spans are unavailable, inspect the logged JSONL artifact under
the run artifacts. File names include run id, process id, and rank to avoid
shared-file corruption.

## Validation

Targeted tests added:

```text
tests/application/test_trace_sampling_policy.py
tests/application/test_structural_trace_payload_builder.py
tests/infrastructure/test_mlflow_trace_recorder.py
tests/application/test_trainer_selective_tracing.py
```

Existing integration surfaces updated:

```text
tests/application/test_cli_flatten_config.py
tests/application/test_experiment_ui_catalog_fields.py
```

## Notes

Tracing is disabled by default. Keep limits small for production-scale runs.

## 2026-05-22 Run Binding Fix

Issue observed on preset `_XAt-TC-UNcnt-drift`: eval_drift runs logged
`tracing_events_recorded=40`, but MLflow Traces did not appear under the
corresponding run.

Root cause:

- `MLflowTraceRecorder` used `mlflow.start_span()`, but did not explicitly guard
  the expected active run context.
- Passing an explicit MLflow `trace_destination` was tested and rejected because
  it creates traces in the experiment but drops the `mlflow.sourceRun` binding.

Fix:

- `MLflowTraceRecorder` now receives `run_id` and `experiment_id`.
- If no active MLflow run exists during trace recording, recorder reopens the
  expected run with `mlflow.start_run(run_id=...)`.
- Recorder no longer passes `trace_destination` to `mlflow.start_span()`, so
  MLflow can attach `mlflow.sourceRun` to the active run.
- JSONL fallback is now also written as a mirror when `start_span()` succeeds.
  This guarantees inspectable trace payloads in run artifacts even if the MLflow
  Traces UI/backend behaves unexpectedly.

Verification:

- Existing `_XAt-TC-UNcnt-drift` runs with tracing enabled:
  - `2da0004b8d9144f1a05a8573f99c854e`: `tracing_events_recorded=40`, no
    run-linked traces.
  - `976a8b16658f4b00aaf94d13c9eafcc3`: `tracing_events_recorded=40`, no
    run-linked traces.
- Local repro after fix created:
  - one MLflow trace with `trace_metadata.mlflow.sourceRun=<run_id>`;
  - one JSONL fallback artifact file.

Relevant validation commands:

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\application\test_trace_sampling_policy.py tests\application\test_structural_trace_payload_builder.py tests\application\test_trainer_selective_tracing.py tests\application\test_cli_flatten_config.py tests\infrastructure\test_mlflow_trace_recorder.py -v
.\.venv-modern\Scripts\python.exe tools\architecture_guard.py
```

## 2026-05-22 Seeded Trace ID Collision Fix

Issue:

- With experiment seed `42`, MLflow 3.12 file-store tracing repeatedly emitted:
  `Yaml file ... trace_info.yaml exists`.
- The trace IDs matched deterministic values produced by Python
  `random.seed(42)`.
- Root cause: MLflow trace ID generation uses global Python `random`, while the
  training pipeline seeds global random for reproducibility. Re-running the same
  seeded experiment generated the same `tr-*` IDs and collided with existing
  trace directories.

Fix:

- `MLflowTraceRecorder` now isolates global random state only while creating the
  MLflow span.
- It temporarily seeds random from `os.urandom(16)` for trace ID generation, then
  restores the exact previous random state.
- This avoids trace ID collisions without changing model/training
  reproducibility.

Verification:

- Added test:
  `tests/infrastructure/test_mlflow_trace_recorder.py::test_mlflow_trace_recorder_isolates_mlflow_trace_id_random`.
- Local repro: two separate MLflow runs with `random.seed(42)` created three
  traces each, all correctly linked by `trace_metadata.mlflow.sourceRun`.

## 2026-05-22 MVP Debug Diagnostics Extension

Added compact diagnostics for StructXAttn/topology-conditioned experiments:

1. Scale diagnostics:
   - `struct_xattn_raw_context_mean_abs`
   - `struct_xattn_pre_norm_delta_mean_abs`
   - `struct_xattn_post_norm_delta_mean_abs`
   - `struct_xattn_raw_to_observed_ratio`
   - `struct_xattn_pre_norm_to_observed_ratio`
   - `struct_xattn_post_norm_to_observed_ratio`

2. Loss/gradient diagnostics:
   - `allowed_set_mass_leakage`
   - `grad_norm_observed`
   - `grad_norm_struct_xattn`
   - `grad_norm_structural`
   - `struct_xattn_contrastive_margin_violation_rate`
   - `topology_conditioned_drop_edges_margin_violation_rate`
   - `topology_conditioned_wrong_version_margin_violation_rate`

3. Drift diagnostics:
   - `drift_window_oos_confidence_mean`

4. Trace diagnostics:
   - StructXAttn scale ratios are included in selected trace payloads and flat
     MLflow span attributes.

Purpose:

- Separate raw structural magnitude from residual `LayerNorm` amplification.
- Detect whether topology-conditioned losses create gradient imbalance.
- Distinguish high-confidence toxic OOS predictions from uncertain OOS behavior.

Trace payloads are intended for qualitative explanation of selected samples,
not for full dataset export.
