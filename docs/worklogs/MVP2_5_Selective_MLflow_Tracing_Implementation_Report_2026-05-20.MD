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
Trace payloads are intended for qualitative explanation of selected samples,
not for full dataset export.
