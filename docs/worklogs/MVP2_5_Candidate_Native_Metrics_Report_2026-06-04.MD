# MVP2.5 Candidate-Native Metrics Report

## Metadata

- `date`: 2026-06-04
- `scope`: EOPKGTopologyConditioned candidate-native metric contract
- `status`: implemented
- `plan`: `docs/worklogs/MVP2_5_Candidate_Native_Metrics_Plan_2026-06-04.MD`

---

## Summary

Primary train/test/drift metric names now stay aligned with Baseline charts, but
use the correct prediction space:

- fixed-label models keep the previous fixed-vocabulary semantics;
- `training.candidate_contract_mode=candidate_id` uses topology-local
  candidate logits `[B, C_v]` and maps `target_label` to
  `candidate_ids` / `candidate_labels`;
- old fixed-vocabulary projection metrics are retained under `fixed_label_*`.

This fixes the audit issue where unseen candidates were correct in
topology-native space but could never receive credit after projection back to
the frozen training vocabulary.

---

## Implemented Changes

### Candidate helper metrics

Updated `src/application/services/candidate_target_mapping.py`:

- `candidate_set_predictions`
- `candidate_set_correct`
- `candidate_set_accuracy`
- `candidate_set_nll`

These helpers operate directly on `[B, C_v]` candidate logits and target masks.

### Train / validation metrics

Updated `ModelTrainer._run_epoch`:

- candidate mode now computes epoch F1 from candidate-local predictions;
- target mapping uses the same label-aware candidate mask as the loss;
- fixed-vocabulary projection is no longer used as the primary epoch metric for
  `candidate_id`.

### Test metrics

Updated `ModelTrainer._evaluate_test`:

- for `candidate_id`, primary `strict_test_*`, `test_*`, `test_ece`, and
  `test_set_nll` are computed in candidate-native space;
- old fixed-vocabulary projection metrics are logged as `fixed_label_*`;
- unseen candidates can receive credit when their candidate id/label matches
  the next log event label.

### Drift one-pass metrics

Updated `DriftInferenceRecords` and one-pass drift collection:

- `records.y_true/y_pred/confidence/correct` are now the primary metric space;
- added `fixed_y_true`, `fixed_y_pred`, `fixed_confidence`, `fixed_correct`;
- drift-window metrics keep existing names for primary charts;
- fixed-label drift curves are logged as `fixed_label_drift_window_*`.

---

## Metric Interpretation

Use for article/dissertation comparisons:

```text
strict_test_macro_f1
drift_window_strict_macro_f1
test_macro_f1
drift_window_macro_f1
test_ece
test_set_nll
```

Use for compatibility diagnostics only:

```text
fixed_label_strict_test_macro_f1
fixed_label_strict_test_accuracy
fixed_label_drift_window_strict_macro_f1
fixed_label_test_ece
```

Use for topology-native diagnostics:

```text
candidate_target_in_candidate_set_rate
candidate_missing_target_rate
candidate_oos_rate
candidate_invalid_probability_mass
candidate_valid_probability_mass
candidate_valid_invalid_logit_margin
```

If `strict_*` is much higher than `fixed_label_*` on future versions, it means
the model is selecting candidates that are valid in the new topology but cannot
be represented by the frozen training vocabulary.

---

## Tests Run

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\application\test_candidate_target_mapping.py -v
.\.venv-modern\Scripts\python.exe -m pytest tests\application\test_evaluator_metrics.py -v
.\.venv-modern\Scripts\python.exe -m pytest tests\application\test_drift_one_pass_evaluation.py -v
```

All targeted tests passed during implementation.

---

## Next Validation

Recommended manual experiment check:

1. rerun `_EOPKG_TC-UN` train/eval_drift on the same checkpoint/data setup;
2. compare `drift_window_strict_macro_f1` against Baseline on the same graph;
3. inspect `fixed_label_drift_window_strict_macro_f1` separately;
4. check that `candidate_target_in_candidate_set_rate` remains high;
5. for datasets with new activities, verify that `strict_* > fixed_label_*` on
   unseen-heavy windows.

---

## Follow-up Fix: Stable Candidate Metric Keys

After comparing drift runs, candidate-native `strict_*` metrics were found to
be lower than expected while `test_set_nll` improved. Root cause: primary
candidate metrics used local candidate offsets (`0..C_v-1`) as class labels.
Those offsets are topology-local and can change between process versions, so
macro-F1 over drift windows was mixing unrelated activities that happened to
share the same local index.

Implemented fix:

- candidate-id train/validation metrics now use stable metric keys:
  `target_label` from the log and predicted `candidate_label` / `candidate_id`;
- final test metrics use the same stable label/id keys for primary `strict_*`
  and `test_*` metrics;
- one-pass drift records store stable label/id keys in `records.y_true` and
  `records.y_pred` for `candidate_id`;
- fixed-label compatibility metrics remain numeric under `fixed_label_*`;
- non-candidate baseline records remain `int64`, so legacy metrics are not
  affected.

Regression added:

- `test_one_pass_candidate_id_metrics_use_unseen_candidate_space` now asserts
  that one-pass drift records store `new_task` as the primary true/predicted
  key for an unseen candidate.

Verification:

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\application\test_candidate_target_mapping.py tests\application\test_evaluator_metrics.py tests\application\test_drift_one_pass_evaluation.py -v
```

Result: `35 passed`.

Interpretation update:

- primary `strict_*` curves are now the candidate-native article/dissertation
  comparison metrics;
- `fixed_label_*` curves show only how much information would be lost by
  projecting topology-native predictions back to the frozen training
  vocabulary.

---

## Follow-up Fix: Stable Label Calibration/NLL

After the stable-key fix, primary F1/accuracy were correct, but
`test_ece` and `test_set_nll` still used fixed numeric target probabilities.
This incorrectly treated a confident `<UNK>` prediction as well calibrated when
the true stable label was an unseen future activity.

Implemented fix:

- primary `test_ece` now uses confidence from the active prediction space and
  stable-label correctness;
- primary `test_set_nll` now uses stable target-label probability for fixed
  heads and candidate target-set probability for `candidate_id`;
- one-pass drift records now store separate fixed-label compatibility arrays:
  `fixed_confidence`, `fixed_correct`, and `fixed_set_nll`;
- `fixed_label_test_ece` and `fixed_label_test_set_nll` preserve the old
  frozen-vocabulary compatibility semantics.

Regression added:

- fixed-head `<UNK>` prediction against `target_label="new_future_task"` now
  produces poor primary `test_ece` / `test_set_nll`, while `fixed_label_*`
  remains near-perfect.

Verification:

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\application\test_candidate_target_mapping.py tests\application\test_evaluator_metrics.py tests\application\test_drift_one_pass_evaluation.py -v
```

Result: `37 passed`.
