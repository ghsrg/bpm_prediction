# MVP2 Stage 5 Report

Date: 2026-03-12  
Scope: Evaluator Metrics (OOS + Slicing) with MVP1 backward compatibility

## 1. Implemented

### 1.1 OOS metric in evaluation

Updated [trainer.py](C:\Users\korsr\PycharmProjects\bpm_prediction\src\application\use_cases\trainer.py):

1. Added tensor-only OOS flag computation:
   - `ModelTrainer._compute_oos_flags(y_hat, allowed_target_mask)`
   - formula: `(~allowed_target_mask[torch.arange(B), y_hat]).float()`
2. In `_evaluate_test(...)`:
   - after `y_hat` is computed, evaluator reads `contract.allowed_target_mask`
   - computes batch OOS flags and accumulates them
   - exports final average as `test_oos`
3. Fallback behavior:
   - if mask is absent, OOS is safely set to `0.0`
   - no impact on MVP1 baseline evaluation flow

### 1.2 Slicing metrics (length and version)

Updated [trainer.py](C:\Users\korsr\PycharmProjects\bpm_prediction\src\application\use_cases\trainer.py):

1. Added metadata propagation through DataLoader build:
   - `prefix_len` per graph
   - `process_version_idx` per graph
   - optional `allowed_target_mask`
2. Added helper methods:
   - `_length_bin(...)` for bins:
     - `len_1_5`, `len_6_10`, `len_11_20`, `len_21_plus`
   - `_compute_sliced_metrics(...)` for grouped metrics by:
     - prefix length
     - process version
3. Added sliced outputs:
   - by length: `test_accuracy_len_...`, `test_f1_len_...`, `test_oos_len_...`
   - by version: `test_accuracy_vX`, `test_f1_vX`, `test_oos_vX`
4. Slicing fallback:
   - if masks are absent, sliced OOS keys are skipped
   - base metrics still computed (Accuracy/F1/ECE/etc.)

### 1.3 Contract conversion safety

Updated `_data_to_contract(...)` in [trainer.py](C:\Users\korsr\PycharmProjects\bpm_prediction\src\application\use_cases\trainer.py):

1. If batch has `allowed_target_mask`, it is injected into `GraphTensorContract`.
2. If absent, contract remains MVP1-compatible.

## 2. Tests Added

Created [test_evaluator_metrics.py](C:\Users\korsr\PycharmProjects\bpm_prediction\tests\application\test_evaluator_metrics.py):

1. `test_oos_calculation`
   - validates OOS formula on synthetic predictions + mask
2. `test_evaluator_slicing_logic`
   - validates slicing keys by length and version
   - validates no crash and correct fallback when masks are absent

## 3. Verification

### 3.1 Targeted tests

Command:

```bash
.\.venv\Scripts\python.exe -m pytest tests/application/test_evaluator_metrics.py -v
```

Result:
- `2 passed`

### 3.2 Full suite

Command:

```bash
.\.venv\Scripts\python.exe -m pytest tests/ -v
```

Result:
- `68 passed, 1 warning`

### 3.3 MVP1 Regression Shield

Command:

```bash
.\.venv\Scripts\python.exe -m pytest -m mvp1_regression -v
```

Result:
- `16 passed, 52 deselected, 1 warning`

## 4. Readiness

Stage 5 is complete:

1. OOS (hallucination rate) is computed from tensor masks in evaluator.
2. Slicing by prefix length and process version is integrated.
3. MVP1 compatibility is preserved with safe fallbacks.

