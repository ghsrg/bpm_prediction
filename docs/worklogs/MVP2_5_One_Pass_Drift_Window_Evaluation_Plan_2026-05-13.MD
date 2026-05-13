# One-Pass Drift Window Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` or `executing-plans` to implement this plan task-by-task. Use `superpowers:test-driven-development` or `test-driven-development` for every implementation task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `eval_drift` reuse prebuilt test graph datasets and compute drift-window metrics from one streaming inference pass instead of rebuilding and re-evaluating graph windows repeatedly.

**Architecture:** Add trace-level metadata to graph samples, route `eval_drift` through the prebuilt `test_dataset` when available, collect compact per-sample inference records batch-by-batch, and aggregate drift-window metrics from those records. Keep the current raw-trace window path as a fallback for old caches or non-prebuilt runs.

**Tech Stack:** Python, PyTorch, PyTorch Geometric `Data` / `DataLoader`, NumPy, scikit-learn metrics, local sharded graph dataset cache, pytest.

---

## Context

Current `eval_drift` behavior:

```python
def _evaluate_drift_windows(self, traces):
    windows = self._generate_drift_windows(traces)
    for window_idx, (start, window_traces) in enumerate(windows):
        window_loader = self._build_loader(window_traces, shuffle=False)
        metrics = self._evaluate_test(window_loader, stage_label="eval_drift")
```

This means every drift window rebuilds graph tensors from raw traces. For dense
windows, for example `drift_window_size=100` and `drift_window_sliding=10`, about
496 windows are produced for 5054 traces. With structure + stats, every window
can repeatedly materialize stats-backed structural payloads.

Observed practical result:

- `S_Str-UNc-drift-loan` without stats: about 1 hour.
- `S_Str-USc-drift-loan` with stats: 5-7 hours.
- Graph build became faster after structural payload deduplication, but drift
  window evaluation still repeats expensive work.

The fix must optimize the full `eval_drift` phase, not just graph construction.

---

## Design Decision

Implement **one-pass drift window evaluation**:

1. CLI graph dataset builders attach trace metadata to every prefix graph.
2. `ModelTrainer.run()` passes `prebuilt_test_dataset` into `_run_eval_drift`.
3. `_run_eval_drift()` tries the optimized path when the dataset has trace metadata.
4. The optimized path:
   - wraps the prebuilt test dataset in a non-shuffled loader;
   - runs model inference once batch-by-batch;
   - stores compact per-sample arrays;
   - computes drift windows from trace indexes;
   - aggregates metrics from arrays without model forward or graph rebuild.
5. If metadata is absent, the existing raw-trace path remains the fallback.

Do not store full `probs[N, C]` unless a metric needs it. Store compact scalar
columns where possible:

- `trace_idx`
- `y_true`
- `y_pred`
- `confidence`
- `correct`
- `hybrid_correct`
- `hybrid_set_nll`
- `oos`
- `target_in_mask`
- `pred_in_mask`
- `strict_error_but_allowed`
- `mask_cardinality`
- `ambiguous`
- `prefix_len`
- `process_version_idx`

For `top3_accuracy`, either:

- store `topk_hit` scalar per sample, preferred for memory; or
- skip full probability retention and compute top-k during batch forward.

For `ECE`, store:

- `confidence = max_prob`
- `correct = y_pred == y_true`

Then compute ECE from confidence/correct arrays instead of retaining full
probability matrices.

---

## Files To Modify

### Runtime

- Modify: `src/cli.py`
  - Add `trace_idx`, `trace_start_ts`, `trace_end_ts`, and `prefix_idx` to every
    graph sample in `_build_graph_dataset_sharded`.
  - Add the same metadata to `_build_graph_dataset`.
  - Bump `GRAPH_DATASET_CACHE_SCHEMA` from `2` to `3` so new runs do not silently
    reuse old graph caches without trace metadata.
  - Keep shard format backward-compatible.

- Modify: `src/application/use_cases/trainer.py`
  - Pass prebuilt `test_dataset` into `_run_eval_drift`.
  - Add optimized one-pass drift path.
  - Add compact sample-record collection helper.
  - Add metrics aggregation helper that mirrors `_evaluate_test` semantics.
  - Keep fallback to existing `_evaluate_drift_windows(traces)` when trace
    metadata is unavailable.

### Tests

- Modify: `tests/application/test_cli_graph_dataset_cache.py`
  - Verify graph dataset builders attach trace metadata.
  - Verify sharded cache roundtrip preserves trace metadata.

- Modify: `tests/application/test_trainer_preloaded_data.py`
  - Verify `eval_drift` receives prebuilt test dataset from `prepared_data`.
  - Verify optimized path does not call `_build_loader(window_traces)`.

- Modify or create: `tests/application/test_drift_one_pass_evaluation.py`
  - Verify one-pass metrics match legacy window metrics on a small deterministic dataset.
  - Verify old datasets without `trace_idx` fall back to legacy behavior.
  - Verify one-pass path stores compact records and does not retain full `N x C` probability arrays.

### Documentation

- Modify after implementation:
  - `docs/current/project-state.md`
  - `docs/GNN_RUNTIME_MVP2_5.MD`
  - `docs/current/architecture-debt.md`
  - `docs/dissertation_changes.MD`

- Create final implementation report:
  - `docs/worklogs/MVP2_5_One_Pass_Drift_Window_Evaluation_Report_YYYY-MM-DD.MD`

---

## Non-Goals

- Do not change model checkpoints.
- Do not change trained model forward semantics.
- Do not change drift-window definitions.
- Do not solve `snapshot_homogeneous_batching`.
- Do not introduce a user-facing config flag unless fallback proves necessary.
- Do not remove the old raw-trace drift path in this change.

---

## Task 1: Add Trace Metadata To Graph Dataset Samples

**Files:**

- Modify: `src/cli.py`
- Test: `tests/application/test_cli_graph_dataset_cache.py`

### Steps

- [ ] **Step 1: Write failing test for sharded graph metadata**

Add a test similar to:

```python
def test_build_graph_dataset_sharded_attaches_trace_metadata(mock_feature_configs, tmp_path: Path):
    traces = [
        _trace("c1", "v1", ["A", "B"]),
        _trace("c2", "v1", ["A", "C"]),
    ]
    encoder = FeatureEncoder(feature_configs=mock_feature_configs, traces=traces)
    graph_builder = BaselineGraphBuilder(feature_encoder=encoder)

    payload = _build_graph_dataset_sharded(
        traces=traces,
        prefix_policy=PrefixPolicy(),
        graph_builder=graph_builder,  # type: ignore[arg-type]
        version_to_idx={},
        stats_snapshot_version_to_idx={},
        show_progress=False,
        tqdm_disable=True,
        desc="Build test graphs",
        progress_stage="test.build_graph",
        entry_dir=tmp_path / "entry",
        split_key="test",
        shard_size=1,
        max_ram_bytes=None,
    )

    items = list(_iter_graphs_from_dataset_payload(payload))
    assert items
    assert {int(item.trace_idx.view(-1)[0].item()) for item in items} == {0, 1}
    assert all(hasattr(item, "prefix_idx") for item in items)
    assert all(hasattr(item, "trace_start_ts") for item in items)
    assert all(hasattr(item, "trace_end_ts") for item in items)
```

- [ ] **Step 2: Run failing test**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/application/test_cli_graph_dataset_cache.py::test_build_graph_dataset_sharded_attaches_trace_metadata -v
```

Expected: fail because `trace_idx` is not present.

- [ ] **Step 3: Add metadata in `_build_graph_dataset_sharded`**

Inside the trace loop, compute:

```python
trace_index = int(idx - 1)
trace_start_ts = float(trace.events[0].timestamp) if trace.events else 0.0
trace_end_ts = float(trace.events[-1].timestamp) if trace.events else trace_start_ts
```

Add to each graph payload:

```python
"trace_idx": torch.tensor([trace_index], dtype=torch.long),
"prefix_idx": torch.tensor([int(slice_idx - 1)], dtype=torch.long),
"trace_start_ts": torch.tensor([trace_start_ts], dtype=torch.float64),
"trace_end_ts": torch.tensor([trace_end_ts], dtype=torch.float64),
```

- [ ] **Step 4: Add the same metadata in `_build_graph_dataset`**

Use the same fields and tensor dtypes as the sharded builder.

- [ ] **Step 5: Bump graph dataset cache schema**

Change:

```python
GRAPH_DATASET_CACHE_SCHEMA = 2
```

to:

```python
GRAPH_DATASET_CACHE_SCHEMA = 3
```

Reason: old cache entries lack `trace_idx`, so optimized drift evaluation must
not silently use them as if they are new.

- [ ] **Step 6: Run focused cache tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/application/test_cli_graph_dataset_cache.py -v
```

Expected: pass.

---

## Task 2: Route Prebuilt Test Dataset Into `eval_drift`

**Files:**

- Modify: `src/application/use_cases/trainer.py`
- Test: `tests/application/test_trainer_preloaded_data.py`

### Steps

- [ ] **Step 1: Write failing test**

Add a test proving `run()` passes `prebuilt_test_dataset` into drift mode:

```python
def test_eval_drift_receives_prebuilt_test_dataset(mock_feature_configs, mock_raw_trace, tmp_path: Path):
    traces = [mock_raw_trace, mock_raw_trace]
    prefix_policy = PrefixPolicy()
    encoder = FeatureEncoder(feature_configs=mock_feature_configs, traces=traces)
    graph_builder = BaselineGraphBuilder(feature_encoder=encoder)
    graph_dataset = [
        Data(
            x_cat=torch.zeros((1, 1), dtype=torch.long),
            x_num=torch.zeros((1, 1), dtype=torch.float32),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            edge_type=torch.empty((0,), dtype=torch.long),
            y=torch.tensor([1], dtype=torch.long),
            trace_idx=torch.tensor([0], dtype=torch.long),
        )
    ]
    model = BaselineGCN(
        feature_layout=encoder.feature_layout,
        hidden_dim=8,
        output_dim=len(encoder.categorical_vocabs[encoder.activity_feature_name]),
        dropout=0.0,
    )
    trainer = ModelTrainer(
        xes_adapter=_FailOnReadAdapter(),
        prefix_policy=prefix_policy,
        graph_builder=graph_builder,
        model=model,
        log_path="in_memory.xes",
        config={
            "epochs": 1,
            "batch_size": 2,
            "learning_rate": 0.001,
            "device": "cpu",
            "show_progress": False,
            "tqdm_disable": True,
            "checkpoint_dir": str(tmp_path),
            "experiment_config": {
                "name": "pytest_eval_drift_preloaded",
                "mode": "eval_drift",
                "fraction": 1.0,
                "split_strategy": "temporal",
                "train_ratio": 0.0,
                "split_ratio": [0.0, 0.0, 1.0],
            },
        },
        prepared_data={
            "prepared_traces": traces,
            "train_traces": [],
            "val_traces": [],
            "test_traces": traces,
            "train_dataset": [],
            "val_dataset": [],
            "test_dataset": graph_dataset,
            "idx_to_version": {0: "v1"},
        },
    )

    captured = {}

    def _fake_run_eval_drift(**kwargs):
        captured["prebuilt_test_dataset"] = kwargs.get("prebuilt_test_dataset")
        return {"mode": "eval_drift", "drift_metrics": []}

    trainer._run_eval_drift = _fake_run_eval_drift  # type: ignore[method-assign]
    trainer._prepare_checkpoint_state = lambda is_eval_mode: (None, 0, float("inf"), 0)  # type: ignore[assignment]

    result = trainer.run()

    assert result["mode"] == "eval_drift"
    assert captured["prebuilt_test_dataset"] is graph_dataset
```

- [ ] **Step 2: Run failing test**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/application/test_trainer_preloaded_data.py::test_eval_drift_receives_prebuilt_test_dataset -v
```

Expected: fail because `_run_eval_drift` does not accept/pass `prebuilt_test_dataset`.

- [ ] **Step 3: Update signatures**

Change `_run_eval_drift` signature:

```python
def _run_eval_drift(
    self,
    split_data: SplitData,
    drift_traces: Sequence[RawTrace],
    best_epoch: int,
    best_val_loss: float,
    prebuilt_test_dataset: Optional[Any] = None,
) -> Dict[str, Any]:
```

Change `run()` drift branch:

```python
prebuilt_test_dataset = (prebuilt_datasets or {}).get("test") if prebuilt_datasets else None
result = self._run_eval_drift(
    split_data=split_data,
    drift_traces=prepared_traces,
    best_epoch=best_epoch,
    best_val_loss=best_val_loss,
    prebuilt_test_dataset=prebuilt_test_dataset,
)
```

- [ ] **Step 4: Run focused test**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/application/test_trainer_preloaded_data.py::test_eval_drift_receives_prebuilt_test_dataset -v
```

Expected: pass.

---

## Task 3: Extract Compact Inference Record Collection

**Files:**

- Modify: `src/application/use_cases/trainer.py`
- Test: `tests/application/test_drift_one_pass_evaluation.py`

### Steps

- [ ] **Step 1: Add internal record dataclass or dictionary contract**

Add near trainer helpers:

```python
@dataclass
class DriftInferenceRecords:
    trace_idx: np.ndarray
    y_true: np.ndarray
    y_pred: np.ndarray
    confidence: np.ndarray
    correct: np.ndarray
    top3_hit: np.ndarray
    oos_flags: np.ndarray
    target_in_mask_flags: np.ndarray
    pred_in_mask_flags: np.ndarray
    strict_error_but_allowed_flags: np.ndarray
    mask_cardinality: np.ndarray
    hybrid_correct_flags: np.ndarray
    hybrid_set_nll: np.ndarray
    ambiguous_flags: np.ndarray
    prefix_lengths: np.ndarray
    version_labels: list[str]
    inference_ms_per_graph: float
```

This is an application-use-case internal DTO. It must not be exported as a
domain object.

- [ ] **Step 2: Write failing test for compact collection**

Use a tiny `DataLoader` with two samples and a deterministic model. Assert:

```python
records = trainer._collect_drift_inference_records(loader)
assert records.trace_idx.tolist() == [0, 1]
assert records.y_true.shape == (2,)
assert records.y_pred.shape == (2,)
assert records.confidence.shape == (2,)
assert not hasattr(records, "y_prob")
```

- [ ] **Step 3: Implement `_collect_drift_inference_records`**

Move the batch logic from `_evaluate_test` into a reusable helper, but keep the
first implementation scoped to drift.

Important implementation constraints:

- Use `with torch.no_grad()`.
- Use `self.model.eval()`.
- Use `_data_to_contract`.
- Use `_sanitize_contract_numeric_tensors`.
- Use `_accumulate_forward_stats`.
- Use `_apply_mask_guided_logits` with `training=False`.
- Compute `probs = torch.softmax(logits, dim=1)`.
- Store only scalar arrays.
- Compute `top3_hit` in-batch:

```python
top_k = min(3, int(probs.shape[1]))
topk = torch.topk(probs, k=top_k, dim=1).indices
top3_hit = (topk == target_tensor.to(topk.device).unsqueeze(1)).any(dim=1).float()
```

- Compute ECE later from `confidence` and `correct`.
- Require `trace_idx`; if missing, raise a local sentinel error:

```python
class DriftTraceMetadataMissing(ValueError):
    pass
```

or return `None` from an availability check.

- [ ] **Step 4: Run focused test**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/application/test_drift_one_pass_evaluation.py::test_collect_drift_inference_records_is_compact -v
```

Expected: pass.

---

## Task 4: Aggregate Window Metrics From Compact Records

**Files:**

- Modify: `src/application/use_cases/trainer.py`
- Test: `tests/application/test_drift_one_pass_evaluation.py`

### Steps

- [ ] **Step 1: Extract metric aggregation helper**

Add:

```python
def _compute_test_metrics_from_records(
    self,
    records: DriftInferenceRecords,
    idxs: np.ndarray,
) -> Dict[str, Any]:
```

The helper must mirror `_evaluate_test` metric semantics for the subset:

- strict accuracy, macro F1, weighted F1, precision, recall;
- hybrid accuracy and hybrid macro/weighted F1;
- `test_set_nll`;
- `test_top3_accuracy`;
- `test_ece`;
- `test_oos`;
- `test_target_in_mask_rate`;
- `test_pred_in_mask_rate`;
- `test_strict_error_but_allowed_rate`;
- `test_ambiguous_prefix_rate`;
- `test_set_hit_rate_ambiguous`;
- `test_accuracy_deterministic`;
- sliced prefix/version metrics via `_compute_sliced_metrics`.

For ECE, add a helper:

```python
def _expected_calibration_error_from_confidence(
    self,
    confidence: np.ndarray,
    correct: np.ndarray,
) -> float:
```

It should use the same binning logic as `_expected_calibration_error`.

- [ ] **Step 2: Write test comparing aggregate helper to `_evaluate_test`**

Create a deterministic loader with small data and model. Compute:

```python
legacy_metrics = trainer._evaluate_test(loader, stage_label="eval_drift")
records = trainer._collect_drift_inference_records(loader)
new_metrics = trainer._compute_test_metrics_from_records(records, np.arange(records.y_true.shape[0]))
```

Assert approximate equality for:

```python
[
    "test_macro_f1",
    "strict_test_macro_f1",
    "test_accuracy",
    "strict_test_accuracy",
    "test_top3_accuracy",
    "test_oos",
    "test_target_in_mask_rate",
    "test_pred_in_mask_rate",
    "test_strict_error_but_allowed_rate",
    "test_ambiguous_prefix_rate",
]
```

- [ ] **Step 3: Run focused comparison test**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/application/test_drift_one_pass_evaluation.py::test_record_metrics_match_evaluate_test_for_full_dataset -v
```

Expected: pass.

---

## Task 5: Build Drift Windows From Trace Metadata

**Files:**

- Modify: `src/application/use_cases/trainer.py`
- Test: `tests/application/test_drift_one_pass_evaluation.py`

### Steps

- [ ] **Step 1: Add trace-window index resolver**

Add:

```python
def _resolve_drift_window_record_indices(
    self,
    records: DriftInferenceRecords,
    traces: Sequence[RawTrace],
) -> list[tuple[int, np.ndarray, float, float]]:
```

The helper should:

1. call existing `_generate_drift_windows(traces)`;
2. for each `(start, window_traces)`, compute trace index range:

```python
start_trace_idx = int(start)
end_trace_idx = int(start + len(window_traces) - 1)
idxs = np.where(
    (records.trace_idx >= start_trace_idx)
    & (records.trace_idx <= end_trace_idx)
)[0]
```

3. compute `start_ts` and `end_ts` from `window_traces`, same as legacy code.

- [ ] **Step 2: Write test for sliding windows**

Use records with trace indexes `[0, 0, 1, 2, 2, 3]`, window size `2`, sliding
`1`, and four traces. Assert window record indexes:

```python
window 0: trace 0-1 -> sample idx [0, 1, 2]
window 1: trace 1-2 -> sample idx [2, 3, 4]
window 2: trace 2-3 -> sample idx [3, 4, 5]
```

- [ ] **Step 3: Run focused test**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/application/test_drift_one_pass_evaluation.py::test_resolve_drift_window_record_indices_uses_trace_idx_ranges -v
```

Expected: pass.

---

## Task 6: Implement Optimized One-Pass Drift Evaluation

**Files:**

- Modify: `src/application/use_cases/trainer.py`
- Test: `tests/application/test_drift_one_pass_evaluation.py`
- Test: `tests/application/test_trainer_preloaded_data.py`

### Steps

- [ ] **Step 1: Add optimized method**

Add:

```python
def _evaluate_drift_windows_from_prebuilt_dataset(
    self,
    traces: Sequence[RawTrace],
    prebuilt_test_dataset: Any,
) -> Optional[List[Dict[str, float]]]:
```

Behavior:

- Return `None` if dataset is missing or does not expose `trace_idx`.
- Build loader with `_build_loader_from_dataset(prebuilt_test_dataset, shuffle=False)`.
- Collect records once.
- Resolve drift windows from records.
- Aggregate metrics for each window.
- Log metrics to tracker with the same metric names and `step=window_idx`.
- Emit progress events with the same `eval_drift.windows` stage.
- Return drift result rows with the same keys as legacy `_evaluate_drift_windows`.

- [ ] **Step 2: Keep fallback**

In `_run_eval_drift`, use:

```python
drift_metrics = self._evaluate_drift_windows_from_prebuilt_dataset(
    drift_traces,
    prebuilt_test_dataset,
)
if drift_metrics is None:
    drift_metrics = self._evaluate_drift_windows(drift_traces)
```

Log which path is used:

```python
logger.info("Drift evaluation path: one_pass_prebuilt_dataset")
```

or:

```python
logger.info("Drift evaluation path: legacy_raw_trace_windows")
```

- [ ] **Step 3: Write test that optimized path avoids raw graph rebuild**

Patch `_build_loader` to raise:

```python
trainer._build_loader = lambda traces, shuffle: (_ for _ in ()).throw(
    AssertionError("legacy graph rebuild path should not be called")
)
```

Call `_run_eval_drift(..., prebuilt_test_dataset=graph_dataset_with_trace_idx)`.

Assert no exception and returned `drift_metrics` is not empty.

- [ ] **Step 4: Write fallback test**

Use prebuilt dataset without `trace_idx`.

Patch `_evaluate_drift_windows` to return a sentinel:

```python
trainer._evaluate_drift_windows = lambda traces: [{"window_index": 0.0}]
```

Assert `_run_eval_drift(..., prebuilt_test_dataset=old_dataset)` returns the
sentinel drift metrics.

- [ ] **Step 5: Run focused drift tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/application/test_drift_one_pass_evaluation.py tests/application/test_trainer_preloaded_data.py -v
```

Expected: pass.

---

## Task 7: Preserve Legacy Metric Output Contract

**Files:**

- Modify: `src/application/use_cases/trainer.py`
- Test: `tests/application/test_drift_one_pass_evaluation.py`

### Steps

- [ ] **Step 1: Add contract test for window row keys**

Assert every one-pass drift row contains:

```python
expected_keys = {
    "window_index",
    "window_start_trace",
    "window_end_trace",
    "window_start_ts",
    "window_end_ts",
    "window_macro_f1",
    "window_strict_macro_f1",
    "window_test_ece",
    "window_test_set_nll",
    "window_test_oos",
    "window_target_in_mask_rate",
    "window_pred_in_mask_rate",
    "window_strict_error_but_allowed_rate",
    "window_ambiguous_prefix_rate",
}
```

- [ ] **Step 2: Add MLflow/tracker metric-name test**

Use a fake tracker collecting metric names. Assert one-pass path logs the same
names as legacy:

```python
[
    "drift_window_macro_f1",
    "drift_window_strict_macro_f1",
    "drift_window_test_ece",
    "drift_window_test_set_nll",
    "drift_window_test_oos",
    "drift_window_target_in_mask_rate",
    "drift_window_pred_in_mask_rate",
    "drift_window_strict_error_but_allowed_rate",
    "drift_window_ambiguous_prefix_rate",
    "drift_window_start_ts",
    "drift_window_end_ts",
]
```

- [ ] **Step 3: Run contract tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/application/test_drift_one_pass_evaluation.py -v
```

Expected: pass.

---

## Task 8: Documentation Update

**Files:**

- Modify: `docs/current/project-state.md`
- Modify: `docs/GNN_RUNTIME_MVP2_5.MD`
- Modify: `docs/current/architecture-debt.md`
- Modify: `docs/dissertation_changes.MD`
- Create: `docs/worklogs/MVP2_5_One_Pass_Drift_Window_Evaluation_Report_YYYY-MM-DD.MD`

### Steps

- [ ] **Step 1: Update project state**

Add to graph dataset / drift runtime current state:

```text
eval_drift can use one-pass prebuilt test dataset evaluation when graph samples
carry trace_idx metadata. Drift-window metrics are aggregated from compact
per-sample inference records instead of rebuilding graph windows.
```

- [ ] **Step 2: Update GNN runtime docs**

Document:

- required graph sample metadata for optimized drift:
  - `trace_idx`
  - `prefix_idx`
  - `trace_start_ts`
  - `trace_end_ts`
- fallback behavior for old caches;
- memory rule: do not retain full `N x C` probabilities for drift windows.

- [ ] **Step 3: Update architecture debt**

Change `stats_backed_structural_payload_caching`:

- current behavior: snapshot cache and structural payload dedup implemented;
  one-pass drift evaluation implemented if validation passes.
- target state: full real `S_Str-USc-drift-loan` validation confirms runtime
  near structure-only drift and no repeated graph rebuild.

- [ ] **Step 4: Update dissertation changes**

Add note:

```text
Drift evaluation was optimized to avoid repeated graph construction and repeated
forward passes over overlapping windows. This matters for dissertation evidence:
the runtime now evaluates temporal windows from fixed prebuilt graph samples,
reducing infrastructure-induced bias and making stats-backed structural runs
reproducible at full dataset scale.
```

- [ ] **Step 5: Write implementation report**

Report must include:

- root cause;
- implementation summary;
- tests run;
- expected speed impact;
- how to verify in logs;
- remaining risk.

---

## Task 9: Verification

**Files:**

- No new files unless a regression report is needed.

### Steps

- [ ] **Step 1: Run focused tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/application/test_cli_graph_dataset_cache.py tests/application/test_trainer_preloaded_data.py tests/application/test_drift_one_pass_evaluation.py -v
```

Expected: all pass.

- [ ] **Step 2: Run relevant trainer/evaluator tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/application/test_evaluator_metrics.py tests/application/test_drift_eval_no_label_leakage.py tests/application/test_window_generator.py -v
```

Expected: all pass.

- [ ] **Step 3: Run architecture guard**

Run:

```powershell
.\.venv\Scripts\python.exe tools\architecture_guard.py
```

Expected: no violations.

- [ ] **Step 4: Run smoke drift with small fraction**

Use the same preset family with a small fraction:

```yaml
preset_name: S_Str-USc-drift-loan
experiment.fraction: 0.01
```

Expected logs:

```text
Drift evaluation path: one_pass_prebuilt_dataset
```

Expected absent behavior:

```text
Build graph dataset per drift window
```

- [ ] **Step 5: Run full validation**

Run full:

```yaml
preset_name: S_Str-USc-drift-loan
```

Expected:

- graph build uses sharded/deduplicated payload cache;
- drift windows run through one-pass path;
- no Neo4j defunct connection during window evaluation;
- no system-wide virtual-memory exhaustion;
- drift window runtime is close to one full test inference plus aggregation,
  not hundreds of graph rebuilds.

---

## Risks And Mitigations

### Risk: window metrics differ from legacy

Cause: `_evaluate_test` had implicit behavior not copied into record aggregation.

Mitigation:

- test full-dataset metric equality between `_evaluate_test` and record helper;
- test window metric equality on deterministic small windows;
- keep legacy fallback.

### Risk: old graph cache silently lacks `trace_idx`

Cause: cache created before this change.

Mitigation:

- bump `GRAPH_DATASET_CACHE_SCHEMA` to `3`;
- optimized path checks `trace_idx`;
- fallback logs legacy path.

### Risk: memory growth from probability storage

Cause: collecting `probs[N, C]`.

Mitigation:

- store scalar `confidence`, `correct`, `top3_hit`, `hybrid_set_nll`;
- do not add `y_prob` to `DriftInferenceRecords`;
- test record object does not expose full probability matrix.

### Risk: sharded dataset random access becomes expensive

Cause: window subsets reload shards repeatedly.

Mitigation:

- do not use per-window dataset subsets;
- run one sequential non-shuffled loader pass;
- aggregate windows from arrays.

### Risk: duplicate trace indexes after split

Cause: `trace_idx` is local to the split builder, while `drift_traces` must be in
the same order as `test_dataset`.

Mitigation:

- in `eval_drift`, CLI must build `test_dataset` from the same ordered
  `test_traces` passed as `prepared_traces` / `drift_traces`;
- tests must verify index alignment;
- log trace count and record trace index max:

```text
Drift one-pass records: samples=<N> traces=<T> max_trace_idx=<M>
```

---

## Expected Outcome

For `S_Str-USc-drift-loan`:

- `build_graph.test` remains the only expensive graph materialization step;
- `eval_drift.windows` no longer calls graph builder per window;
- dense sliding windows become cheap aggregation over compact arrays;
- stats-backed drift runtime should move from 5-7 hours toward approximately:

```text
one full test inference + lightweight window aggregation
```

For `S_Str-UNc-drift-loan`:

- also faster, but the visible gain is smaller because graph building without
  stats is already cheaper.

For baseline:

- minor speedup if using dense windows, mainly from avoiding repeated forward
  on overlapping windows.

---

## Self-Review

- Spec coverage:
  - avoids repeated graph build: Task 6;
  - avoids repeated forward over overlapping windows: Task 3 + Task 6;
  - prevents OOM from full probabilities: Task 3;
  - keeps old behavior fallback: Task 6;
  - documents dissertation/runtime impact: Task 8.

- Placeholder scan:
  - No `TBD` or unspecified implementation steps remain.

- Type consistency:
  - `trace_idx`, `prefix_idx`: `torch.long` tensors with one value per graph.
  - `trace_start_ts`, `trace_end_ts`: `torch.float64` tensors with one value per graph.
  - `DriftInferenceRecords` arrays use NumPy and are trainer-internal.

