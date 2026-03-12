# MVP2 OOS Math Debug Report

Date: 2026-03-12  
Scope: Investigation of `test_oos == 1.0` anomaly and OOS tensor math validation

## 1. What Was Changed

### 1.1 OOS tensor math debug logs (one-time)

Updated [trainer.py](C:\Users\korsr\PycharmProjects\bpm_prediction\src\application\use_cases\trainer.py):

1. Added one-time debug block in `_evaluate_test(...)` before OOS flag computation.
2. Added logs:
   - `DEBUG MASK: shape=..., All True? ...`
   - `--- OOS MATH DEBUG ---`
   - `y_hat` shape/min/max
   - mask shape/dtype
   - extracted mask flags at `[B, y_hat]`
   - resulting `oos_flags`

### 1.2 Mask-token alignment fix

Updated [topology_extractor_service.py](C:\Users\korsr\PycharmProjects\bpm_prediction\src\application\services\topology_extractor_service.py):

1. Added `_event_label(...)` with semantic preference:
   - `concept:name` -> `activity` -> `activity_id`
2. Transition extraction now uses this label for edges and start/end stats.

Updated [dynamic_graph_builder.py](C:\Users\korsr\PycharmProjects\bpm_prediction\src\domain\services\dynamic_graph_builder.py):

1. Last prefix activity is resolved via `FeatureEncoder.resolve_event_feature(...)`.
2. Allowed target indices are mapped to vocab after consistent token resolution.
3. Mask size remains `C = len(activity_vocab)`.

## 2. Debug Output (Quick Run)

Captured log from quick evaluation run:

```text
INFO:DEBUG MASK: shape=(1, 24), All True? False
INFO:--- OOS MATH DEBUG ---
INFO:y_hat shape: (1,), min=21, max=21
INFO:mask shape: (1, 24), dtype=torch.bool
INFO:extracted_flags (allowed_target_mask[B, y_hat]): shape=(1,), trues=0/1
INFO:oos_flags (~extracted_flags): trues=1/1
```

## 3. Findings

1. OOS indexing math is correct:
   - `y_hat` has shape `(B,)`
   - mask has shape `(B, C)` with expected `C=24`
   - extraction `allowed_target_mask[torch.arange(B), y_hat]` is valid.
2. In this debug batch, predicted class was not allowed by mask, so OOS=1 for that sample.
3. The previous shape concern (`(1, 4)`) is not a universal bug; with realistic setup mask shape is `(1, 24)`.

## 4. Regression Status

Command:

```bash
.\.venv\Scripts\python.exe -m pytest tests/ -q
```

Result:
- `70 passed, 1 warning`

