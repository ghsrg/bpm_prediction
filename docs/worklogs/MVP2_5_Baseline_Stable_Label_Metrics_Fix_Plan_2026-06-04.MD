# MVP2.5 Baseline Stable Label Metrics Fix Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `test-driven-development`
> for implementation and `verification-before-completion` before reporting
> completion. Execute task-by-task.

**Goal:** make primary train/test/drift metrics comparable between Baseline and
`EOPKGTopologyConditioned` by computing exact next-activity quality in stable
activity-label space, while preserving old fixed-index metrics under
`fixed_label_*`.

**Architecture:** this is a metric/evaluation contract fix only. It must not
change graph cache, topology cache, model checkpoints, training loss, candidate
generation, or mask construction. Primary metrics use stable labels:
`true = target_label` and `pred = predicted activity label`. Legacy
fixed-vocabulary metrics remain available as `fixed_label_*`.

**Tech Stack:** Python, PyTorch, PyG `DataLoader`, sklearn metrics, MLflow
tracking.

---

## Problem

Current candidate-native metrics were fixed to use stable candidate labels, but
fixed-head Baseline/EOPKGGATv2 still report primary `strict_*` metrics in the
frozen training vocabulary. This inflates zero-shot drift results when future
versions add activities:

```text
C_train = 45
C_v5 = 64
```

If v5 contains 19 new activities, a fixed-head baseline cannot predict them.
The primary zero-shot metric must count those targets as errors instead of
collapsing them through `<UNK>` or evaluating only inside `C_train`.

Correct comparison space:

```text
Baseline:
  true_key = target_label from current-version log
  pred_key = reverse_activity_vocab[pred_class]

EOPKGTopologyConditioned:
  true_key = target_label from current-version log
  pred_key = candidate_labels[pred_candidate]
```

Old fixed-index metrics are still useful diagnostics, but must be renamed:

```text
fixed_label_strict_test_macro_f1
fixed_label_drift_window_strict_macro_f1
fixed_label_test_accuracy
...
```

---

## Files

- Modify: `src/application/use_cases/trainer.py`
- Modify: `tests/application/test_evaluator_metrics.py`
- Modify: `tests/application/test_drift_one_pass_evaluation.py`
- Modify: `docs/GNN_RUNTIME_MVP2_5.MD`
- Modify: `docs/GNN_LEARNING_METODOLOGY.MD`
- Modify: `docs/worklogs/MVP2_5_Candidate_Native_Metrics_Report_2026-06-04.MD`

No model files should be changed.

---

## Task 1: Add Stable Fixed-Head Label Metric Regression

**Files:**

- Modify: `tests/application/test_evaluator_metrics.py`

- [ ] **Step 1: Write failing test for fixed-head unseen target**

Add a test showing that a fixed-head model predicting a known class must fail
when the raw target label is an unseen future activity:

```python
def test_evaluate_test_fixed_head_primary_metrics_use_target_label_for_unseen_future_activity():
    class _KnownOnlyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.output_dim = 3

        def forward(self, contract):
            batch = contract["batch"]
            num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
            logits = torch.full((num_graphs, 3), -5.0, dtype=torch.float32, device=batch.device)
            logits[:, 1] = 5.0
            return logits

    trainer = ModelTrainer(
        xes_adapter=_DummyAdapter(),
        prefix_policy=_DummyPrefixPolicy(),
        graph_builder=_DummyGraphBuilder(),
        model=_KnownOnlyModel(),
        log_path="in_memory.xes",
        config={
            "mode": "eval_drift",
            "device": "cpu",
            "show_progress": False,
            "tqdm_disable": True,
        },
        prepared_data={
            "reverse_activity_vocab": {
                0: "<UNK>",
                1: "known_task",
                2: "other_task",
            }
        },
    )
    sample = Data(
        x_cat=torch.zeros((1, 0), dtype=torch.long),
        x_num=torch.ones((1, 1), dtype=torch.float32),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
        edge_type=torch.zeros((0,), dtype=torch.long),
        y=torch.tensor([0], dtype=torch.long),
        num_nodes=1,
        target_label="new_future_task",
        process_version_idx=torch.tensor([4], dtype=torch.long),
    )

    metrics = trainer._evaluate_test(DataLoader([sample], batch_size=1, shuffle=False))

    assert metrics["strict_test_accuracy"] == pytest.approx(0.0)
    assert metrics["strict_test_macro_f1"] == pytest.approx(0.0)
    assert metrics["fixed_label_strict_test_accuracy"] == pytest.approx(0.0)
```

The key assertion is that primary `strict_*` compares
`"new_future_task"` with `"known_task"`, not `0` with `1` or `<UNK>` with
`known_task`.

- [ ] **Step 2: Run test and verify it fails**

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\application\test_evaluator_metrics.py::test_evaluate_test_fixed_head_primary_metrics_use_target_label_for_unseen_future_activity -v
```

Expected before implementation: fail because primary metrics still use numeric
fixed labels.

---

## Task 2: Add Stable Label Helpers For Fixed-Head Predictions

**Files:**

- Modify: `src/application/use_cases/trainer.py`

- [ ] **Step 1: Add helper to read target labels**

Use the existing `_target_labels_from_contract(...)` for raw labels. Do not add
a second target-label extraction path.

- [ ] **Step 2: Add helper for fixed-head metric keys**

Add a helper next to `_candidate_metric_label_keys`:

```python
def _fixed_head_metric_label_keys(
    self,
    *,
    predictions: torch.Tensor,
    contract: GraphTensorContract,
    targets: torch.Tensor,
) -> tuple[List[str], List[str]]:
    target_labels = self._target_labels_from_contract(contract, targets)
    target_values = targets.detach().cpu().long().view(-1).tolist()
    pred_values = predictions.detach().cpu().long().view(-1).tolist()

    true_keys: List[str] = []
    pred_keys: List[str] = []
    for row_idx, pred_idx in enumerate(pred_values):
        if row_idx < len(target_labels) and str(target_labels[row_idx]).strip():
            true_key = str(target_labels[row_idx]).strip()
        elif row_idx < len(target_values):
            true_key = self._reverse_activity_vocab.get(
                int(target_values[row_idx]),
                f"class:{int(target_values[row_idx])}",
            )
        else:
            true_key = "__missing_target__"

        pred_keys.append(
            self._reverse_activity_vocab.get(int(pred_idx), f"class:{int(pred_idx)}")
        )
        true_keys.append(true_key)
    return true_keys, pred_keys
```

Rules:

- if `target_label` exists, it is the primary true key;
- if not, fallback to `reverse_activity_vocab[y]`;
- if prediction id is absent in vocab, use `class:<idx>`;
- this helper is used only for metrics, not for loss.

- [ ] **Step 3: Run targeted failing test**

Same command as Task 1 Step 2.

Expected after helper only: still fail until `_evaluate_test` uses it.

---

## Task 3: Fix `_evaluate_test` Primary Metrics

**Files:**

- Modify: `src/application/use_cases/trainer.py`
- Test: `tests/application/test_evaluator_metrics.py`

- [ ] **Step 1: Preserve fixed-index arrays**

Inside `_evaluate_test`, introduce separate arrays/lists:

```python
all_true: List[Any] = []
all_pred: List[Any] = []
all_fixed_true: List[int] = []
all_fixed_pred: List[int] = []
all_fixed_probs: List[np.ndarray] = []
```

Candidate-mode already has this pattern. Extend it to fixed-head mode.

- [ ] **Step 2: In fixed-head branch, append stable labels to primary metrics**

After `pred_tensor = torch.argmax(probs, dim=1).long()`:

```python
true_keys, pred_keys = self._fixed_head_metric_label_keys(
    predictions=pred_tensor,
    contract=contract,
    targets=target_tensor,
)
all_true.extend(true_keys)
all_pred.extend(pred_keys)
all_fixed_true.extend(target_tensor.detach().cpu().numpy().tolist())
all_fixed_pred.extend(pred_tensor.detach().cpu().numpy().tolist())
all_fixed_probs.extend(probs.detach().cpu().numpy())
```

Do not append numeric `targets` / `pred_tensor` to `all_true/all_pred` for
primary metrics when stable labels are available.

- [ ] **Step 3: Compute primary arrays as object dtype**

Use:

```python
y_true = np.asarray(all_true, dtype=object)
y_pred = np.asarray(all_pred, dtype=object)
```

sklearn supports string labels for `accuracy_score`, `f1_score`,
`precision_score`, and `recall_score`.

- [ ] **Step 4: Compute fixed-label metrics from numeric arrays for all modes**

For fixed-head and candidate modes, `fixed_label_*` should be computed from:

```python
fixed_y_true = np.asarray(all_fixed_true, dtype=np.int64)
fixed_y_pred = np.asarray(all_fixed_pred, dtype=np.int64)
```

For fixed-head runs this means:

```text
fixed_label_* == old numeric fixed-vocab metrics
```

For candidate runs this means:

```text
fixed_label_* == projected compatibility diagnostics
```

- [ ] **Step 5: Run evaluator tests**

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\application\test_evaluator_metrics.py -v
```

Expected: all evaluator tests pass after updating expected assertions if they
were intentionally checking old primary fixed-index behavior.

---

## Task 4: Fix One-Pass Drift Records For Fixed-Head Models

**Files:**

- Modify: `src/application/use_cases/trainer.py`
- Modify: `tests/application/test_drift_one_pass_evaluation.py`

- [ ] **Step 1: Add failing drift record regression**

Add a fixed-head one-pass drift test:

```python
def test_one_pass_fixed_head_records_use_stable_target_labels_for_future_activity(tmp_path):
    trainer = _trainer(tmp_path, drift_window_size=1, drift_window_sliding=1)
    trainer._reverse_activity_vocab = {0: "<UNK>", 1: "known_task", 2: "other_task"}
    sample = _sample(trace_idx=0, target=0, pred=1, mask=[True, False, False])
    sample.target_label = "new_future_task"

    records = trainer._collect_drift_inference_records(DataLoader([sample], batch_size=1))
    metrics = trainer._compute_test_metrics_from_records(records, np.asarray([0], dtype=np.int64))

    assert records.y_true.tolist() == ["new_future_task"]
    assert records.y_pred.tolist() == ["known_task"]
    assert metrics["strict_test_accuracy"] == pytest.approx(0.0)
```

- [ ] **Step 2: Run and verify failure**

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\application\test_drift_one_pass_evaluation.py::test_one_pass_fixed_head_records_use_stable_target_labels_for_future_activity -v
```

Expected before implementation: records still contain numeric values.

- [ ] **Step 3: Update `_collect_drift_inference_records`**

For fixed-head mode, after `pred_tensor` is known:

```python
true_keys, pred_keys = self._fixed_head_metric_label_keys(
    predictions=pred_tensor,
    contract=contract,
    targets=target_tensor,
)
all_true.extend(true_keys)
all_pred.extend(pred_keys)
```

Keep numeric compatibility fields:

```python
all_fixed_true.extend(target_tensor.detach().cpu().tolist())
all_fixed_pred.extend(pred_tensor.detach().cpu().tolist())
```

- [ ] **Step 4: Return primary records as object dtype for all modes**

Use:

```python
y_true=np.asarray(all_true, dtype=object)
y_pred=np.asarray(all_pred, dtype=object)
```

Do not use `np.int64` for primary drift records. The primary drift space is now
stable label space for both fixed-head and candidate-id models.

- [ ] **Step 5: Ensure `_compute_test_metrics_from_records` accepts object labels**

`accuracy_score`, `f1_score`, `precision_score`, `recall_score` can consume
object/string arrays. Hybrid labels must also stay object dtype:

```python
y_true_hybrid = np.asarray(y_true, dtype=object).copy()
```

Only fixed-label compatibility metrics should use numeric arrays.

- [ ] **Step 6: Run drift tests**

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\application\test_drift_one_pass_evaluation.py -v
```

Expected: all one-pass drift tests pass. Existing tests that compare old
numeric primary arrays should be updated to expect stable labels or use
`fixed_y_true/fixed_y_pred`.

---

## Task 5: Preserve Legacy Fixed-Index Metric Names Under `fixed_label_*`

**Files:**

- Modify: `src/application/use_cases/trainer.py`
- Modify: `tests/application/test_evaluator_metrics.py`
- Modify: `tests/application/test_drift_one_pass_evaluation.py`

- [ ] **Step 1: Ensure `_evaluate_test` always emits `fixed_label_*`**

For fixed-head models, emit:

```text
fixed_label_strict_test_accuracy
fixed_label_strict_test_macro_f1
fixed_label_strict_test_weighted_f1
fixed_label_test_accuracy
fixed_label_test_macro_f1
fixed_label_test_weighted_f1
fixed_label_test_ece
fixed_label_test_set_nll
```

These values should match old numeric fixed-vocab metrics.

- [ ] **Step 2: Ensure drift-window logging emits `fixed_label_drift_window_*`**

The existing drift-window code already logs `fixed_label_*` when metrics are
present. Confirm it now receives these values for fixed-head baseline too.

- [ ] **Step 3: Add regression assertion**

In one-pass drift tests, assert both primary and fixed-label metrics exist:

```python
assert "drift_window_strict_macro_f1" in row
assert "fixed_label_drift_window_strict_macro_f1" in row
```

For a future unseen target, primary strict should be lower or equal to
fixed-label compatibility:

```python
assert row["drift_window_strict_macro_f1"] <= row["fixed_label_drift_window_strict_macro_f1"]
```

This guards against accidentally promoting fixed-index metrics back to primary.

---

## Task 6: Documentation And Report Update

**Files:**

- Modify: `docs/GNN_RUNTIME_MVP2_5.MD`
- Modify: `docs/GNN_LEARNING_METODOLOGY.MD`
- Modify: `docs/worklogs/MVP2_5_Candidate_Native_Metrics_Report_2026-06-04.MD`

- [ ] **Step 1: Update runtime metric contract**

Add the rule:

```text
Primary strict/test/drift metrics are stable-label metrics for all models.
Fixed-head models convert prediction ids through reverse_activity_vocab.
Candidate-id models use candidate_labels/candidate_ids.
Numeric fixed-vocab metrics are logged only under fixed_label_*.
```

- [ ] **Step 2: Update methodology**

Clarify article/dissertation interpretation:

```text
Baseline zero-shot penalty for unseen future activities is intentional:
if the target label is absent from C_train, a fixed-head baseline cannot predict
it and receives an exact-next-activity error.
```

- [ ] **Step 3: Append report section**

Add a follow-up section:

```text
Baseline Stable Label Metrics Fix
```

Include:

- root cause;
- changed metric semantics;
- expected effect: baseline strict-F1 may drop on versions with new activities;
- cache/checkpoint impact: no rebuild or retrain required; rerun eval only.

---

## Task 7: Verification

- [ ] **Step 1: Run targeted tests**

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\application\test_evaluator_metrics.py tests\application\test_drift_one_pass_evaluation.py -v
```

Expected: pass.

- [ ] **Step 2: Run candidate tests to ensure previous fix remains valid**

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\application\test_candidate_target_mapping.py tests\application\test_evaluator_metrics.py tests\application\test_drift_one_pass_evaluation.py -v
```

Expected: pass.

- [ ] **Step 3: Run architecture guard**

```powershell
.\.venv-modern\Scripts\python.exe tools\architecture_guard.py
```

Expected:

```text
[ARCH_GUARD] OK: architecture dependency rules satisfied.
```

---

## Expected Experiment Impact

No cache reset is required:

- graph tensors are unchanged;
- topology payloads are unchanged;
- checkpoints are unchanged;
- only metric interpretation changes.

No retraining is required. Rerun `eval_drift` for:

```text
_Base-UN-...
_EOPKG_TC-UN-...
```

Expected result:

- fixed-head baseline primary `drift_window_strict_macro_f1` drops on versions
  containing unseen future activities;
- `fixed_label_drift_window_strict_macro_f1` preserves the old fixed-vocab
  curve;
- topology-native candidate model remains evaluated in the same stable-label
  metric space;
- comparison becomes valid for zero-shot structural drift.

---

## Self-Review

Spec coverage:

- primary metrics renamed semantically, not by key: covered in Tasks 3-4;
- old current metrics preserved as `fixed_label_*`: covered in Task 5;
- baseline inflated metrics fixed: covered in Tasks 1, 3, 4;
- candidate previous fix preserved: covered in Task 7.

No known placeholders remain. The plan intentionally avoids model/loss/cache
changes.
