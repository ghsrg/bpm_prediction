# MVP2.5 Candidate-Native Metrics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `test-driven-development` for every implementation task and `verification-before-completion` before reporting completion.

**Goal:** make the existing primary metric names (`strict_test_*`, `test_*`, `drift_window_*`) represent the same business metric across Baseline and `EOPKGTopologyConditioned`: exact next-activity quality, while `candidate_id + topology_native` is evaluated in the current-version candidate space.

**Architecture:** keep fixed-vocab metrics as compatibility diagnostics under the `fixed_label_*` prefix. For `candidate_contract_mode=candidate_id`, compute primary metrics from `candidate_logits [B, C_v]` and `target_label -> candidate_ids/candidate_labels`; for fixed-label models, keep the existing fixed-vocab calculation as the primary path.

**Tech Stack:** Python, PyTorch, PyG `Data`, sklearn metrics, existing `ModelTrainer`.

---

## Current Problem

For `training.candidate_contract_mode=candidate_id` and
`training.candidate_identity_mode=topology_native`, training loss already uses
topology-local candidates:

```text
candidate_logits [B, C_v]
target_label -> target_candidate_mask [B, C_v]
loss = candidate_set_cross_entropy(candidate_logits, target_candidate_mask)
```

But train/validation/test/drift metrics still project predictions back to the
fixed training vocabulary via `candidate_predictions_to_global()`. Unseen
version-specific candidates have `candidate_class_index = -1`, so they are
masked out and can never receive credit in `strict_test_macro_f1`.

This makes current `strict_*` metrics invalid as primary topology-native
metrics for versions with new candidates.

---

## Target Metric Contract

### Primary names

These names stay unchanged for charts and Baseline comparison:

```text
train_macro_f1
validation_macro_f1
strict_test_accuracy
strict_test_macro_f1
strict_test_weighted_f1
test_accuracy
test_macro_f1
test_weighted_f1
test_ece
test_set_nll
drift_window_strict_macro_f1
drift_window_macro_f1
drift_window_test_ece
drift_window_test_set_nll
```

Semantics:

- fixed-label models: unchanged fixed-vocab exact next-activity metric.
- `candidate_id + topology_native`: candidate-native exact next-activity metric.

### Compatibility names

For `candidate_id + topology_native`, also log old fixed-vocab projection metrics:

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

For drift windows:

```text
fixed_label_drift_window_strict_macro_f1
fixed_label_drift_window_macro_f1
fixed_label_drift_window_test_ece
fixed_label_drift_window_test_set_nll
```

### Candidate diagnostics

Keep candidate-specific names for diagnostics only:

```text
candidate_target_in_candidate_set_rate
candidate_missing_target_rate
candidate_unseen_target_rate
candidate_oos_rate
candidate_invalid_probability_mass
candidate_valid_probability_mass
candidate_valid_invalid_logit_margin
```

---

## Files

- Modify: `src/application/services/candidate_target_mapping.py`
- Modify: `src/application/use_cases/trainer.py`
- Modify: `tests/application/test_candidate_target_mapping.py`
- Modify: `tests/application/test_evaluator_metrics.py`
- Modify: `tests/application/test_drift_one_pass_evaluation.py`
- Update docs after code:
  - `docs/GNN_RUNTIME_MVP2_5.MD`
  - `docs/GNN_LEARNING_METODOLOGY.MD`
  - `docs/current/architecture-debt.md`

---

## Task 1: Candidate-Native Metric Helpers

**Files:**

- Modify: `src/application/services/candidate_target_mapping.py`
- Modify: `tests/application/test_candidate_target_mapping.py`

### Steps

- [ ] Add failing test: unseen candidate can be strict-correct in candidate space.

Test case:

```python
def test_candidate_set_accuracy_counts_unseen_label_prediction_as_correct():
    logits = torch.tensor([[0.1, 2.0, 0.0]], dtype=torch.float32)
    target_mask = torch.tensor([[False, True, False]], dtype=torch.bool)

    assert candidate_set_accuracy(logits, target_mask) == pytest.approx(1.0)
```

- [ ] Add failing test: candidate confidence and set NLL use target candidate mask.

Test case:

```python
def test_candidate_set_nll_uses_probability_mass_of_target_set():
    logits = torch.tensor([[0.0, 2.0, 2.0]], dtype=torch.float32)
    target_mask = torch.tensor([[False, True, True]], dtype=torch.bool)

    value = candidate_set_nll(logits, target_mask)

    assert value >= 0.0
    assert value < 0.5
```

- [ ] Implement small pure helpers:

```python
def candidate_set_predictions(candidate_logits: torch.Tensor) -> torch.LongTensor:
    return torch.argmax(candidate_logits, dim=1).long()

def candidate_set_correct(candidate_logits: torch.Tensor, target_candidate_mask: torch.Tensor) -> torch.FloatTensor:
    pred = candidate_set_predictions(candidate_logits)
    row_ids = torch.arange(candidate_logits.size(0), device=candidate_logits.device)
    return target_candidate_mask.to(candidate_logits.device).bool()[row_ids, pred].float()

def candidate_set_accuracy(candidate_logits: torch.Tensor, target_candidate_mask: torch.Tensor) -> float:
    correct = candidate_set_correct(candidate_logits, target_candidate_mask)
    return float(correct.mean().detach().cpu().item()) if correct.numel() else 0.0

def candidate_set_nll(candidate_logits: torch.Tensor, target_candidate_mask: torch.Tensor) -> torch.Tensor:
    log_probs = torch.log_softmax(candidate_logits, dim=1)
    mask = target_candidate_mask.to(device=candidate_logits.device, dtype=torch.bool)
    has_target = mask.any(dim=1)
    if not bool(has_target.any()):
        return candidate_logits.sum() * 0.0
    masked = log_probs.masked_fill(~mask, float("-inf"))
    return -torch.logsumexp(masked[has_target], dim=1).mean()
```

- [ ] Run:

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\application\test_candidate_target_mapping.py -v
```

Expected: all tests pass.

---

## Task 2: Train/Validation Epoch F1 Uses Candidate Space

**Files:**

- Modify: `src/application/use_cases/trainer.py`
- Modify: `tests/application/test_evaluator_metrics.py`

### Steps

- [ ] Add failing test for train epoch metrics:

Scenario:

```text
candidate_logits argmax -> candidate label "new_task"
candidate_class_index for "new_task" = -1
data.y = 0
target_label = "new_task"
```

Expected:

```text
train_macro_f1 or validation_macro_f1 = 1.0
fixed_label_train_macro_f1 = 0.0
```

- [ ] In epoch loop, when `candidate_output is not None`, compute:

```text
target_candidate_mask via target_label for topology_native
candidate_pred_label_index = argmax(candidate_logits)
candidate_correct = target_candidate_mask[row, candidate_pred]
```

- [ ] Store candidate-native per-sample truth/pred for sklearn F1.

Use a deterministic candidate-local label id for metrics:

```text
candidate_true_metric_id = first true candidate index in target mask
candidate_pred_metric_id = pred_candidate index
```

This preserves exact candidate identity for current homogeneous topology batches.

- [ ] Also compute fixed-label metric arrays from the old projection and log them under:

```text
fixed_label_train_macro_f1
fixed_label_validation_macro_f1
```

- [ ] Run targeted tests:

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\application\test_evaluator_metrics.py::test_train_epoch_candidate_id_uses_candidate_set_loss_without_fixed_projection -v
```

Add the new test name to this run after creating it.

---

## Task 3: `_evaluate_test()` Primary Metrics Become Candidate-Native

**Files:**

- Modify: `src/application/use_cases/trainer.py`
- Modify: `tests/application/test_evaluator_metrics.py`

### Steps

- [ ] Add failing test replacing/renaming current expectation:

Current test `test_evaluate_test_candidate_id_reports_global_metrics_from_candidate_predictions` should be updated or complemented with:

```text
target_label = "new_task"
data.y = 0
candidate_labels includes "new_task"
candidate_class_index for that candidate = -1
model predicts "new_task"
```

Expected:

```text
strict_test_accuracy == 1.0
strict_test_macro_f1 == 1.0
fixed_label_strict_test_accuracy == 0.0
candidate_unseen_target_rate == 1.0
```

- [ ] In `_evaluate_test()`, build target candidate mask using labels in the same way as train:

```python
target_labels = self._target_labels_from_contract(contract, target_tensor)
if self.candidate_identity_mode == "topology_native" and target_labels:
    target_candidate_mask = candidate_target_mask_from_labels(...)
else:
    target_candidate_mask = candidate_output.map_targets_to_candidate_mask(target_tensor)
```

- [ ] For candidate mode, primary arrays become candidate-local:

```text
all_true -> first target candidate index
all_pred -> argmax candidate index
all_probs -> candidate probabilities
```

- [ ] For fixed-label compatibility, maintain separate arrays:

```text
fixed_y_true
fixed_y_pred
fixed_y_prob
```

- [ ] Compute primary `strict_test_*`, `test_*`, ECE/NLL from candidate arrays.

- [ ] Compute `fixed_label_*` from fixed arrays.

- [ ] If a row has no target candidate, exclude it from candidate-native strict metrics and reflect it in:

```text
candidate_missing_target_rate
```

Do not silently count missing-target rows as wrong unless a specific metric name says so.

- [ ] Run:

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\application\test_evaluator_metrics.py -v
```

---

## Task 4: Drift One-Pass Records Carry Both Metric Spaces

**Files:**

- Modify: `src/application/use_cases/trainer.py`
- Modify: `tests/application/test_drift_one_pass_evaluation.py`

### Steps

- [ ] Extend `DriftInferenceRecords` with candidate-native and fixed-label fields.

Required additions:

```python
fixed_y_true: np.ndarray
fixed_y_pred: np.ndarray
fixed_confidence: np.ndarray
fixed_correct: np.ndarray
candidate_missing_target_flags: np.ndarray
candidate_unseen_target_flags: np.ndarray
```

Keep existing `y_true`, `y_pred`, `confidence`, `correct` as primary metric space.

- [ ] In `_collect_drift_inference_records()`, for candidate mode:

```text
y_true/y_pred/confidence/correct = candidate-native
fixed_y_true/fixed_y_pred/fixed_confidence/fixed_correct = old fixed-label projection
```

For fixed-label mode:

```text
fixed_* mirrors primary arrays
```

- [ ] Add failing drift test:

One-pass record with unseen target candidate must produce:

```text
strict_test_accuracy == 1.0
fixed_label_strict_test_accuracy == 0.0
```

when `_compute_test_metrics_from_records()` is called.

- [ ] Update `_compute_test_metrics_from_records()`:

Primary metrics use `records.y_true/y_pred`.

Fixed compatibility metrics use `records.fixed_y_true/fixed_y_pred`.

- [ ] Update drift logging so:

```text
drift_window_strict_macro_f1
```

uses candidate-native primary metrics, and:

```text
fixed_label_drift_window_strict_macro_f1
```

uses compatibility metrics.

- [ ] Run:

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\application\test_drift_one_pass_evaluation.py -v
```

---

## Task 5: Trace Payload and Diagnostics

**Files:**

- Modify: `src/application/services/structural_trace_payload_builder.py`
- Modify: `tests/application/test_structural_trace_payload_builder.py`

### Steps

- [ ] Add trace fields:

```text
metric_space
strict_correct_candidate_native
strict_correct_fixed_label
target_candidate_labels
pred_candidate_label
pred_candidate_id
pred_candidate_is_unseen
fixed_label_pred
fixed_label_target
```

- [ ] Ensure all tensor values are detached before JSON serialization.

- [ ] Run:

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\application\test_structural_trace_payload_builder.py -v
```

---

## Task 6: Documentation Sync

**Files:**

- Modify: `docs/GNN_RUNTIME_MVP2_5.MD`
- Modify: `docs/GNN_LEARNING_METODOLOGY.MD`
- Modify: `docs/current/architecture-debt.md`

### Steps

- [ ] Update runtime docs with metric contract:

```text
strict_* = primary business metric in active prediction space
fixed_label_* = compatibility diagnostic
candidate_* = topology-native diagnostics
```

- [ ] Update methodology docs to state that Scopus/dissertation comparisons use unchanged metric names across Baseline and EOPKGTC, but interpretation is candidate-native for topology-conditioned models.

- [ ] Update architecture debt:

If this implementation completes true candidate-native metric reporting, reduce `dynamic_candidate_prediction_contract` debt from “metrics projected back to fixed label” to remaining items only.

---

## Verification

Run targeted tests first:

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\application\test_candidate_target_mapping.py -v
.\.venv-modern\Scripts\python.exe -m pytest tests\application\test_evaluator_metrics.py -v
.\.venv-modern\Scripts\python.exe -m pytest tests\application\test_drift_one_pass_evaluation.py -v
```

Then architecture guard:

```powershell
.\.venv-modern\Scripts\python.exe tools\architecture_guard.py
```

Full test suite can be run manually by the user if runtime budget is limited:

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\ -v
```

---

## Acceptance Criteria

- For fixed-label Baseline, existing metric names and values remain unchanged.
- For `EOPKGTopologyConditioned + candidate_id + topology_native`, a correct unseen candidate prediction receives credit in `strict_test_accuracy` and `strict_test_macro_f1`.
- The old fixed-vocab projection is still visible as `fixed_label_*`.
- Drift windows log candidate-native `drift_window_strict_macro_f1` under the existing chart name.
- Documentation explicitly explains the metric-space distinction.

