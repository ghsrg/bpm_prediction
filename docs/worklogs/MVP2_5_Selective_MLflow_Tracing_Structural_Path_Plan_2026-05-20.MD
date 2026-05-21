# MVP2.5 Selective MLflow Tracing for Structural Path Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `test-driven-development`
> and `verification-before-completion`. Implement task-by-task. Do not trace
> every batch by default.

**Goal:** add selective trace-level observability for structural-path
debugging across baseline and structural fusion modes without replacing
aggregate metrics.

**Architecture:** keep tracing behind an application port. `ModelTrainer`
selects a small number of interesting prediction samples during
`inference`/`eval_drift`, builds a compact explanation payload, and sends it to
an optional trace recorder. The MLflow adapter uses `mlflow.start_span()` when
available in MLflow 3.x and falls back to JSON artifacts/no-op behavior when
the installed MLflow does not support tracing.

**Tech Stack:** Python, PyTorch, PyG, MLflow tracking/tracing, pytest.

---

## 1. Context

Current state:

1. Aggregate model diagnostics already exist in `ModelTrainer`:
   `observed_logits_mean_abs`, `structural_to_observed_logit_ratio`,
   `topology_state_entropy`, `struct_xattn_attention_entropy`, OOS/mask
   metrics, drift-window rows, etc.
2. These metrics are good for evidence, but they do not explain individual
   predictions.
3. MLflow 3 tracing provides manual spans through `mlflow.start_span()` with
   `span.set_inputs()`, `span.set_outputs()`, and span attributes. This should
   be used only selectively because per-batch/per-sample full tracing can
   bloat storage and slow evaluation.

Sources:

- `https://mlflow.org/docs/latest/genai/tracing/app-instrumentation/manual-tracing`
- `https://github.com/mlflow/mlflow/blob/master/CHANGELOG.md`

---

## 2. Non-Goals

1. Do not use traces as dissertation-grade quantitative evidence.
2. Do not log full tensors, full probability matrices, or every batch.
3. Do not make domain/model code import MLflow.
4. Do not require MLflow 3.x for normal training with the current `.venv`.
5. Do not alter model logits, training loss, or drift metrics.

---

## 3. Target Behavior

When enabled, each selected trace represents one prediction sample:

```text
structural_prediction_debug
  inputs:
    run/stage/sample identity
    process version and snapshot identity
    fusion mode and structural mode
    compact graph/tensor shape summary
    mask summary
  outputs:
    target class
    predicted class
    confidence
    strict correctness
    top-k classes
    mode-specific structural diagnostics
```

Expected usage:

```yaml
tracking:
  enabled: true
  tracing:
    enabled: true
    stages: [inference, eval_drift_one_pass]
    sample_policy: interesting
    max_traces_per_run: 80
    max_traces_per_stage: 40
    max_traces_per_version: 10
    random_sample_rate: 0.0
    top_k: 5
    log_json_artifact_fallback: true
```

Default behavior:

```yaml
tracking.tracing.enabled: false
```

---

## 4. Trace Payload Contract

Create a compact payload that is stable across models:

```python
{
    "schema_version": "1.0",
    "stage": "eval_drift_one_pass",
    "sample": {
        "global_index": 123,
        "trace_idx": 17,
        "prefix_idx": 4,
        "prefix_len": 5,
        "process_version": "v2",
    },
    "run": {
        "model_type": "EOPKGGATv2",
        "fusion_mode": "StructXAttn",
        "structural_mode": True,
        "mask_guided_policy": "off",
    },
    "snapshot": {
        "stats_snapshot_version": "k000500",
        "stats_snapshot_as_of_ts": "2026-03-10T09:32:06.153000+00:00",
        "stats_allowed": True,
        "stats_missing_asof_snapshot": False,
    },
    "contract": {
        "observed_node_count": 5,
        "observed_edge_count": 4,
        "struct_node_count": 42,
        "struct_edge_count": 57,
        "struct_feature_dim": 5,
        "has_allowed_target_mask": True,
    },
    "prediction": {
        "target_index": 12,
        "target_label": "Approve",
        "pred_index": 15,
        "pred_label": "Reject",
        "confidence": 0.81,
        "strict_correct": False,
        "top_k": [
            {"index": 15, "label": "Reject", "probability": 0.81},
            {"index": 12, "label": "Approve", "probability": 0.11}
        ],
    },
    "mask": {
        "target_in_mask": True,
        "prediction_in_mask": True,
        "strict_error_but_allowed": True,
        "mask_cardinality": 3,
        "allowed_top_k": [
            {"index": 12, "label": "Approve"},
            {"index": 15, "label": "Reject"}
        ],
    },
    "diagnostics": {
        "generic": {
            "final_logits_mean_abs": 7.2,
            "prediction_entropy": 0.41,
        },
        "class_aware_structural_scoring": {
            "observed_logits_mean_abs": 12.1,
            "structural_logits_mean_abs": 0.72,
            "structural_to_observed_logit_ratio": 0.06,
            "target_observed_logit": 2.1,
            "target_structural_logit": 0.4,
            "pred_observed_logit": 2.4,
            "pred_structural_logit": 0.8
        },
        "struct_xattn": {
            "struct_xattn_to_observed_ratio": 0.18,
            "struct_xattn_attention_entropy": 1.9,
            "struct_xattn_gate_mean": 0.21
        }
    },
    "reason": "strict_error_but_allowed"
}
```

Payload rules:

1. Store scalar diagnostics and top-k summaries only.
2. Limit labels/top-k by config.
3. Do not store full `struct_x`, full logits matrix, or full attention matrix.
4. Every PyTorch-derived value must be converted with `detach().cpu()` and then
   `.item()` / `.tolist()` before insertion into any payload dictionary. The
   payload must never retain tensors, autograd graph references, or GPU memory.
5. Span attributes must be flat and search-friendly. Use attributes such as
   `stage`, `fusion_mode`, `structural_mode`, `reason`, `strict_correct`,
   `process_version`, `target_in_mask`, `prediction_in_mask`,
   `structural_to_observed_logit_ratio`, and `struct_xattn_attention_entropy`.
   Keep nested JSON only in `span.set_inputs()` / `span.set_outputs()`.
6. For future deep StructXAttn inspection, add a separate opt-in
   `tracking.tracing.include_attention_top_nodes=true` that logs only top
   structural nodes per selected observed node.

---

## 5. Sampling Policy

Implement deterministic, bounded sampling. A sample is eligible when one of
these reasons matches:

1. `strict_error_but_allowed`: prediction is wrong but inside allowed mask.
2. `oos_prediction`: prediction is outside allowed mask.
3. `high_confidence_error`: wrong prediction with confidence above threshold.
4. `low_confidence_correct`: correct prediction with confidence below threshold.
5. `version_first`: first selected sample for a process version.
6. `random`: deterministic random sample under configured sample rate.

Recommended defaults:

```yaml
tracking.tracing.sample_policy: interesting
tracking.tracing.high_confidence_threshold: 0.8
tracking.tracing.low_confidence_threshold: 0.4
tracking.tracing.max_traces_per_run: 80
tracking.tracing.max_traces_per_stage: 40
tracking.tracing.max_traces_per_version: 10
tracking.tracing.random_sample_rate: 0.0
```

Why this policy:

1. It focuses traces on cases that explain structural usefulness or failure.
2. It prevents MLflow storage blow-up.
3. It gives comparable trace slices across `_Base-UN`, `_CAS-UN`,
   `_TSG-MN`, `StructXAttn`, etc.

---

## 6. File Structure

Create:

1. `src/application/ports/trace_recorder_port.py`
   - application-level protocol and dataclasses.
2. `src/application/services/structural_trace_payload_builder.py`
   - pure payload builder from contract/logits/model diagnostics.
3. `src/application/services/trace_sampling_policy.py`
   - deterministic bounded sample selector.
4. `src/infrastructure/tracking/mlflow_trace_recorder.py`
   - MLflow tracing adapter with MLflow 3 span support and MLflow 2 fallback.
5. `tests/application/test_structural_trace_payload_builder.py`
6. `tests/application/test_trace_sampling_policy.py`
7. `tests/application/test_trainer_selective_tracing.py`
8. `tests/infrastructure/test_mlflow_trace_recorder.py`

Modify:

1. `src/application/ports/tracker_port.py`
   - no required changes unless artifact fallback should reuse tracker.
2. `src/application/use_cases/trainer.py`
   - accept optional trace recorder and call it only when selector accepts a
     sample.
3. `src/cli.py`
   - build trace recorder from `tracking.tracing.*`.
4. `configs/ui/config_catalog.yaml`
   - add all `tracking.tracing.*` keys.
5. `docs/GNN_RUNTIME_MVP2_5.MD`
   - document selective tracing and model-specific diagnostic fields.
6. `docs/EVF_MVP2_5.MD`
   - document traces as qualitative/debug evidence only.
7. `docs/current/project-state.md`
   - add runtime behavior if implemented.

---

## 7. Task Plan

### Task 1: Add Trace Recorder Port

**Files:**

- Create: `src/application/ports/trace_recorder_port.py`
- Test: `tests/application/test_trace_sampling_policy.py`

Steps:

1. Write failing tests for a no-op-compatible protocol payload:

```python
def test_trace_event_payload_is_plain_json_serializable():
    event = StructuralTraceEvent(
        name="structural_prediction_debug",
        inputs={"sample": {"trace_idx": 1}},
        outputs={"prediction": {"pred_index": 2}},
        attributes={"stage": "eval_drift_one_pass"},
    )
    json.dumps(event.to_dict())
```

2. Implement:

```python
@dataclass(frozen=True)
class StructuralTraceEvent:
    name: str
    inputs: dict[str, Any]
    outputs: dict[str, Any]
    attributes: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "attributes": self.attributes,
        }

class ITraceRecorder(Protocol):
    def record(self, event: StructuralTraceEvent) -> None:
        ...
```

`StructuralTraceEvent.attributes` is the flat searchable metadata contract.
`inputs` and `outputs` may contain nested JSON, but `attributes` must remain
one-level deep and scalar/string/bool-only.

3. Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\application\test_trace_sampling_policy.py -v
```

Expected: new test passes.

### Task 2: Add Deterministic Trace Sampling Policy

**Files:**

- Create: `src/application/services/trace_sampling_policy.py`
- Test: `tests/application/test_trace_sampling_policy.py`

Steps:

1. Test hard limits:

```python
def test_trace_sampling_policy_respects_stage_version_and_run_limits():
    policy = TraceSamplingPolicy(
        enabled=True,
        stages={"eval_drift_one_pass"},
        max_traces_per_run=3,
        max_traces_per_stage=2,
        max_traces_per_version=1,
    )
    assert policy.should_record(stage="eval_drift_one_pass", version="v1", reason="version_first")
    assert not policy.should_record(stage="eval_drift_one_pass", version="v1", reason="strict_error_but_allowed")
    assert policy.should_record(stage="eval_drift_one_pass", version="v2", reason="version_first")
```

2. Test interesting reasons:

```python
def test_trace_sampling_policy_selects_interesting_errors_before_random():
    policy = TraceSamplingPolicy(enabled=True, random_sample_rate=0.0)
    reason = classify_trace_reason(
        strict_correct=False,
        pred_in_mask=True,
        target_in_mask=True,
        confidence=0.9,
        version_seen=False,
    )
    assert reason == "strict_error_but_allowed"
```

3. Implement `TraceSamplingPolicy` and `classify_trace_reason`.

4. Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\application\test_trace_sampling_policy.py -v
```

Expected: pass.

### Task 3: Build Structural Trace Payloads

**Files:**

- Create: `src/application/services/structural_trace_payload_builder.py`
- Test: `tests/application/test_structural_trace_payload_builder.py`

Steps:

1. Test generic baseline payload:

```python
def test_payload_builder_handles_observed_only_model():
    payload = build_structural_prediction_trace_payload(
        stage="inference",
        global_index=0,
        contract=contract,
        logits=logits,
        effective_logits=logits,
        target_tensor=targets,
        pred_tensor=preds,
        probs=probs,
        model=model_without_structural_attrs,
        reverse_activity_vocab={0: "<UNK>", 1: "Approve"},
        row_index=0,
    )
    assert payload["run"]["structural_mode"] is False
    assert payload["diagnostics"]["generic"]["prediction_entropy"] >= 0.0
```

2. Test `ClassAwareStructuralScoring` payload includes observed and structural
   target/pred logits when `model.last_observed_logits` and
   `model.last_structural_class_logits` are available.

3. Test `StructXAttn` payload includes entropy/ratio/gate diagnostics.

4. Test recursive tensor detachment:

```python
def test_payload_builder_never_returns_tensors_or_cuda_references():
    logits = torch.tensor([[1.0, 2.0]], requires_grad=True)
    payload = build_structural_prediction_trace_payload(...)
    assert_no_tensors(payload)
    assert logits.grad_fn is None or "grad_fn" not in repr(payload)
```

Use a recursive assertion:

```python
def assert_no_tensors(value):
    if isinstance(value, torch.Tensor):
        raise AssertionError("trace payload must not contain torch.Tensor")
    if isinstance(value, dict):
        for item in value.values():
            assert_no_tensors(item)
    if isinstance(value, list):
        for item in value:
            assert_no_tensors(item)
```

5. Test flat attributes are emitted separately from nested payload:

```python
event = build_structural_prediction_trace_event(...)
assert event.attributes["fusion_mode"] == "StructXAttn"
assert event.attributes["reason"] == "strict_error_but_allowed"
assert isinstance(event.attributes["strict_correct"], bool)
assert all("." not in key for key in event.attributes)
```

6. Implement helpers:

```python
def safe_float(value: Any) -> float | None
def label_for_index(index: int, reverse_vocab: dict[int, str]) -> str
def top_k_payload(probs: torch.Tensor, reverse_vocab: dict[int, str], k: int) -> list[dict[str, Any]]
def build_structural_prediction_trace_payload(...) -> dict[str, Any]
def build_structural_prediction_trace_attributes(...) -> dict[str, Any]
def build_structural_prediction_trace_event(...) -> StructuralTraceEvent
```

Implementation rule:

```python
def tensor_scalar(value: torch.Tensor) -> float:
    return float(torch.nan_to_num(value.detach().float().cpu(), nan=0.0, posinf=1e6, neginf=-1e6).mean().item())
```

Every tensor-derived value must use `detach().cpu()` before entering the final
payload/event.

7. Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\application\test_structural_trace_payload_builder.py -v
```

Expected: pass.

### Task 4: Add MLflow Trace Recorder Adapter

**Files:**

- Create: `src/infrastructure/tracking/mlflow_trace_recorder.py`
- Test: `tests/infrastructure/test_mlflow_trace_recorder.py`

Steps:

1. Test MLflow 3 path using monkeypatch:

```python
def test_mlflow_trace_recorder_uses_start_span_when_available(monkeypatch):
    calls = []
    monkeypatch.setattr(mlflow_trace_recorder.mlflow, "start_span", fake_start_span)
    recorder = MLflowTraceRecorder(enabled=True)
    recorder.record(event)
    assert calls[0]["name"] == "structural_prediction_debug"
    assert calls[0]["attributes"]["fusion_mode"] == "StructXAttn"
    assert calls[0]["attributes"]["reason"] == "strict_error_but_allowed"
```

2. Test MLflow 2 fallback:

```python
def test_mlflow_trace_recorder_noops_when_start_span_missing_and_no_fallback(monkeypatch):
    monkeypatch.delattr(mlflow_trace_recorder.mlflow, "start_span", raising=False)
    recorder = MLflowTraceRecorder(enabled=True, fallback_json_path=None)
    recorder.record(event)
```

3. Test JSON fallback writes bounded JSONL events when `fallback_json_path` is
   set.

4. Test fallback path is process/rank isolated:

```python
def test_json_fallback_path_includes_pid_and_rank(monkeypatch, tmp_path):
    monkeypatch.setenv("RANK", "0")
    recorder = MLflowTraceRecorder(enabled=True, fallback_dir=str(tmp_path), run_id="abc")
    assert f"pid_{os.getpid()}" in str(recorder.fallback_json_path)
    assert "rank_0" in str(recorder.fallback_json_path)
```

5. Test non-zero rank skips fallback unless explicitly allowed:

```python
def test_json_fallback_skips_nonzero_rank_by_default(monkeypatch, tmp_path):
    monkeypatch.setenv("RANK", "1")
    recorder = MLflowTraceRecorder(enabled=True, fallback_dir=str(tmp_path), run_id="abc")
    recorder.record(event)
    assert not list(tmp_path.glob("*.jsonl"))
```

6. Implement:

```python
class MLflowTraceRecorder:
    def __init__(
        self,
        *,
        enabled: bool,
        fallback_dir: str | None = None,
        run_id: str | None = None,
        allow_nonzero_rank_fallback: bool = False,
    ) -> None: ...

    def record(self, event: StructuralTraceEvent) -> None:
        if not self.enabled:
            return
        start_span = getattr(mlflow, "start_span", None)
        if callable(start_span):
            with start_span(name=event.name, attributes=event.attributes) as span:
                span.set_inputs(event.inputs)
                span.set_outputs(event.outputs)
            return
        self._append_json_fallback(event)
```

Fallback file naming:

```text
structural_traces_run_<run_id>_pid_<pid>_rank_<rank>.jsonl
```

Write each event as one newline-delimited JSON object. Do not maintain one
shared JSON array.

7. Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\infrastructure\test_mlflow_trace_recorder.py -v
```

Expected: pass on current MLflow version.

### Task 5: Wire Selective Tracing into `ModelTrainer`

**Files:**

- Modify: `src/application/use_cases/trainer.py`
- Test: `tests/application/test_trainer_selective_tracing.py`

Steps:

1. Extend `ModelTrainer.__init__`:

```python
trace_recorder: Optional[ITraceRecorder] = None
```

2. Add trainer fields:

```python
self.trace_recorder = trace_recorder
self.trace_policy = TraceSamplingPolicy.from_config(config.get("tracking_config", {}).get("tracing", {}))
self._reverse_activity_vocab = self._resolve_reverse_activity_vocab()
```

3. Add test that `_evaluate_test(..., stage_label="inference")` records one
   trace for `strict_error_but_allowed` when tracing is enabled.

4. Add test that disabled tracing records nothing.

5. Add test that `_collect_drift_inference_records()` records traces without
   breaking compact records or one-pass drift metrics.

6. Add test that selected trace payloads contain no tensors after trainer
   recording:

```python
def test_trainer_trace_event_payload_contains_no_tensors(tmp_path):
    recorder = _RecordingTraceRecorder()
    trainer = _trainer(tmp_path, trace_recorder=recorder, tracing_enabled=True)
    trainer._evaluate_test(loader, stage_label="inference")
    assert_no_tensors(recorder.events[0].to_dict())
```

7. Implement a helper:

```python
def _maybe_record_prediction_traces(
    *,
    stage_label: str,
    data: Data,
    contract: GraphTensorContract,
    logits: torch.Tensor,
    effective_logits: torch.Tensor,
    probs: torch.Tensor,
    targets: torch.Tensor,
    predictions: torch.Tensor,
    allowed_mask: torch.Tensor | None,
    global_offset: int,
) -> None:
    ...
```

8. Call the helper only in:

```text
_evaluate_test()
_collect_drift_inference_records()
```

Do not call it in `_run_epoch()` by default. Training traces can be added later
as a separate opt-in if needed.

9. Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\application\test_trainer_selective_tracing.py -v
```

Expected: pass.

### Task 6: Wire CLI and Config Catalog

**Files:**

- Modify: `src/cli.py`
- Modify: `configs/ui/config_catalog.yaml`
- Test: `tests/application/test_cli_flatten_config.py`
- Test: `tests/application/test_experiment_ui_catalog_fields.py`

Steps:

1. Add config keys to catalog:

```yaml
tracking.tracing.enabled
tracking.tracing.stages
tracking.tracing.sample_policy
tracking.tracing.max_traces_per_run
tracking.tracing.max_traces_per_stage
tracking.tracing.max_traces_per_version
tracking.tracing.random_sample_rate
tracking.tracing.high_confidence_threshold
tracking.tracing.low_confidence_threshold
tracking.tracing.top_k
tracking.tracing.log_json_artifact_fallback
```

2. In `src/cli.py`, after tracker creation, build:

```python
trace_recorder = _build_trace_recorder(
    tracking_cfg=tracking_cfg,
    run_id=tracker.get_run_id() if tracker is not None and hasattr(tracker, "get_run_id") else None,
    output_dir=...
)
```

3. Pass `trace_recorder=trace_recorder` to `ModelTrainer`.

4. If JSON fallback file exists at run end and tracker exists, log it as an
   artifact.

5. Record tracing overhead metrics:

```text
tracing_events_recorded
tracing_record_time_ms_total
tracing_record_time_ms_per_event
```

These metrics are diagnostic only and must not affect evaluation scores.

6. Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\application\test_cli_flatten_config.py tests\application\test_experiment_ui_catalog_fields.py -v
```

Expected: pass.

### Task 7: Documentation

**Files:**

- Modify: `docs/GNN_RUNTIME_MVP2_5.MD`
- Modify: `docs/EVF_MVP2_5.MD`
- Modify: `docs/current/project-state.md`

Required content:

1. In `GNN_RUNTIME_MVP2_5.MD`, add `Selective Structural Tracing` section:
   - traces are qualitative/debug artifacts;
   - explain model-mode diagnostic differences:
     - baseline: logits/mask/confidence only;
     - `ClassAwareStructuralScoring`: observed vs structural logits;
     - `TopologyStateEncoder`: prefix-state/gate/cardinality/entropy;
     - `TopologyStateGraphEncoder`: graph context/logits entropy;
     - `StructuralPriorEncoder`: observed context vs structural prior context;
     - `StructXAttn`: cross-attention ratio/entropy/gate.
2. In `EVF_MVP2_5.MD`, add rule:
   - dissertation claims must use aggregate metrics and seed variance;
   - traces support explanation/debugging only.
3. In `project-state.md`, add implemented status after implementation.

### Task 8: Verification

Run targeted tests:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\application\test_trace_sampling_policy.py tests\application\test_structural_trace_payload_builder.py tests\application\test_trainer_selective_tracing.py tests\infrastructure\test_mlflow_trace_recorder.py -v
```

Run integration-adjacent tests:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\application\test_drift_one_pass_evaluation.py tests\application\test_trainer_forward_stats_logging.py tests\application\test_cli_flatten_config.py -v
```

Run full validation:

```powershell
.\.venv\Scripts\python.exe tools\architecture_guard.py
.\.venv\Scripts\python.exe -m pytest tests\ -v
```

If `.venv\Scripts\python.exe` fails with `Unable to create process using
...Python311...`, rerun the same command with escalated permissions.

---

## 8. Expected Result

After implementation:

1. MLflow 3.12 users see selected `structural_prediction_debug` traces in the
   Traces UI.
2. Current MLflow 2.x environment keeps working and can store fallback JSON
   artifacts if enabled.
3. Traces are comparable across model modes because payload schema is shared.
4. Aggregate metrics remain the primary evidence path.
5. Runtime overhead is bounded by trace limits.

---

## 9. Risks and Controls

### Risk: PyTorch memory retention / OOM

Control:

- payload builder must detach and move every tensor-derived value to CPU before
  storing it in dictionaries;
- tests must assert payload contains no `torch.Tensor` values recursively;
- trainer must pass tensors to the builder only inside `torch.no_grad()` eval
  paths or after explicitly detached values are produced.

### Risk: trace volume explosion

Control:

- disabled by default;
- hard limits per run/stage/version;
- no full tensor logging.

### Risk: MLflow version incompatibility

Control:

- adapter checks `hasattr(mlflow, "start_span")`;
- no-op or JSON fallback in MLflow 2.x.

### Risk: MLflow UI search misses nested JSON fields

Control:

- MLflow spans receive a flat attribute dictionary for searchable metadata;
- nested input/output payload remains for explanation detail only;
- tests verify that key fields are passed to `start_span(..., attributes=...)`.

### Risk: fallback JSON corruption under multi-process execution

Control:

- fallback path must include process id and optional rank in the filename, for
  example `structural_traces_run_<run_id>_pid_<pid>_rank_<rank>.jsonl`;
- if rank can be detected and is not `0`, skip fallback writes unless explicitly
  configured otherwise;
- JSON fallback writes newline-delimited JSON events, not a mutable shared array.

### Risk: synchronous trace I/O slows inference

Control:

- tracing is disabled by default;
- hard sample limits keep calls bounded;
- log trace count and optional elapsed tracing time in run metrics;
- if overhead becomes visible, add a buffered recorder as a later change.

### Risk: misleading interpretation

Control:

- docs explicitly state traces are qualitative/debug evidence;
- metrics, drift windows, calibration/OOS, and seed variance remain primary.

### Risk: StructXAttn lacks per-node attention detail

Control:

- Stage 1 logs scalar StructXAttn diagnostics only;
- optional Stage 2 can add top-attended structural node summaries after
  confirming storage and performance cost.

---

## 10. Recommended First Experiment

Run the same small drift slice with tracing enabled:

```yaml
tracking:
  enabled: true
  tracing:
    enabled: true
    stages: [eval_drift_one_pass]
    sample_policy: interesting
    max_traces_per_run: 40
    max_traces_per_stage: 40
    max_traces_per_version: 8
    random_sample_rate: 0.0
    top_k: 5
```

Compare:

1. `_Base-UN`
2. `_CAS-UN`
3. `_TSG-UN`
4. `_Pri-UN`
5. `StructXAttn` preset

Focus on traces where:

1. baseline is wrong and structural model is correct;
2. structural model is wrong but prediction is inside mask;
3. OOS decreases while strict-F1 does not improve;
4. structure has high confidence but wrong target.

This should show whether structure changes decision evidence, not just final
aggregate F1.
