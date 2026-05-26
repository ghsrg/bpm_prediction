# MVP2.5 EOPKGTopologyConditioned Run Analysis

- `date`: 2026-05-26
- `train_run`: `0f36f16dcd484c03b454c201d2723d14`
- `drift_run`: `709bd0a6f26f4940b014c8d14b929584`
- `comparison_drift_run`: `8900633b257943b3b795be1f7d43dca6`
- `model_type`: `EOPKGTopologyConditioned`
- `candidate_contract_mode`: `candidate_id`
- `learning_strategy`: `standard`
- `dataset`: `loan_v1_v4_simulated`

## Summary

The run validates that the true candidate-level path is wired and stable, but it
does not yet demonstrate meaningful topology-based knowledge transfer across
versions.

The main reason is not a training crash or missing topology payload. The reason
is methodological: in the current XES/collapse configuration, the dynamic
candidate space is effectively the same fixed activity-label space in every
version.

```text
candidate_dynamic_count = 37
duplicate_candidate_count_max = 1
candidate_target_in_candidate_set_rate = 1.0
candidate_missing_target_rate = 0.0
candidate_class_score_mean_abs == candidate_node_score_mean_abs
```

This means `candidate_id` currently changes the loss path, but it does not
create a genuinely different topology-local prediction space.

## Train Run Findings

Run `0f36f16dcd484c03b454c201d2723d14`:

```text
experiment.mode = train
experiment.train_ratio = 0.5
experiment.fraction = 0.2
experiment.split_strategy = versioned
experiment.version_scope_policy = train_cut
training.learning_strategy = standard
training.candidate_contract_mode = candidate_id
experiment.structural_mode = true
experiment.statistic_enabled = false
experiment.mask_guided_enabled = false
```

Key metrics:

```text
strict_train_macro_f1: first=0.2231 last=0.6206 max=0.6217
strict_val_macro_f1:   first=0.6523 last=0.8285 max=0.8370
strict_test_macro_f1:  0.8297
test_macro_f1:         1.0
test_ece:              0.0199
test_set_nll:          0.0184
```

Candidate diagnostics:

```text
train_candidate_dynamic_count = 37
validation_candidate_dynamic_count = 37
train_candidate_missing_target_rate = 0.0
validation_candidate_missing_target_rate = 0.0
train_candidate_target_in_candidate_set_rate = 1.0
validation_candidate_target_in_candidate_set_rate = 1.0
train_candidate_target_duplicate_count_max = 1.0
validation_candidate_target_duplicate_count_max = 1.0
train_candidate_target_set_entropy_mean = 0.0
validation_candidate_target_set_entropy_mean = 0.0
train_candidate_target_set_logit_variance_mean = 0.0
validation_candidate_target_set_logit_variance_mean = 0.0
```

Interpretation:

- Log/topology target mapping is valid.
- There are no duplicate candidate targets in the current candidate view.
- Candidate-level MIL is not being exercised.
- The model is mostly doing activity-label ranking over the same 37 labels,
  not version-specific BPMN node candidate scoring.

## Drift Run Findings

Run `709bd0a6f26f4940b014c8d14b929584`:

```text
drift_window_strict_macro_f1: first=0.8299 last=0.5902 mean=0.7212 min=0.5755
drift_window_macro_f1:        first=1.0000 last=0.7512 mean=0.8920 min=0.7299
drift_window_test_oos:        first=0.0000 last=0.1331 mean=0.0656 max=0.1377
drift_window_test_ece:        first=0.0149 last=0.0762 mean=0.0490 max=0.0936
drift_window_test_set_nll:    first=0.0173 last=0.6442 mean=0.2884 max=0.6942
```

Comparison with `8900633b257943b3b795be1f7d43dca6`:

```text
metric                         709...      890...
strict_macro_f1 mean           0.7212      0.7378
strict_macro_f1 last           0.5902      0.6341
macro_f1 mean                  0.8920      0.9193
macro_f1 last                  0.7512      0.8221
test_oos mean                  0.0656      0.0443
test_oos last                  0.1331      0.0935
test_set_nll mean              0.2884      0.1784
test_set_nll last              0.6442      0.4534
test_ece mean                  0.0490      0.0439
```

The `890...` run is better mainly because it is less overconfident and has lower
OOS/NLL. It does not prove stronger topology transfer; it is closer to a softer,
less aggressive ranking behavior.

Candidate one-pass diagnostics:

```text
metric                              709...    890...
candidate_dynamic_count             37        37
duplicate_candidate_count_max       1         1
candidate_prediction_entropy        0.2983    0.3344
candidate_score_gap                 0.2868    0.1679
candidate_target_score              8.3027    8.3333
candidate_pred_score                8.5894    8.5013
```

Interpretation:

- `709...` has a larger predicted-vs-target score gap.
- `709...` is more confident in the wrong candidate ranking.
- `890...` keeps a softer distribution, which reduces OOS/NLL and slightly
  improves strict-F1.

## Trace Findings

Trace artifact:

```text
mlruns/854778689611649472/709bd0a6f26f4940b014c8d14b929584/artifacts/
structural_traces_run_709bd0a6f26f4940b014c8d14b929584_pid_7024_rank_0.jsonl
```

Trace count:

```text
40 traces
10 per version: v1, v2, v3, v4
stage = eval_drift_one_pass
```

Reason distribution:

```text
strict_error_but_allowed = 23
oos_prediction           = 11
version_first            = 4
low_confidence_correct   = 2
```

By version:

```text
v1: strict=2/10, oos=0/10, strict_error_but_allowed=8/10
v2: strict=1/10, oos=0/10, strict_error_but_allowed=9/10
v3: strict=2/10, oos=6/10, strict_error_but_allowed=2/10
v4: strict=1/10, oos=5/10, strict_error_but_allowed=4/10
```

Important pattern:

- In seen/near-seen versions `v1/v2`, sampled failures are mostly
  `strict_error_but_allowed`: the model predicts an allowed candidate, but not
  the exact target.
- In future versions `v3/v4`, OOS appears strongly: the model starts selecting
  candidates that are not valid under the runtime mask.

Frequent trace error pairs:

```text
v2: t_plan_setup    -> predicted t_send_notice  (strict_error_but_allowed)
v1: t_plan_setup    -> predicted t_send_notice  (strict_error_but_allowed)
v1: t_cancel_loan   -> predicted t_approve_loan (strict_error_but_allowed)
v3: t_check_funding -> predicted t_assess_eligibility (oos_prediction)
v3: t_monitor_six   -> predicted t_monitor_five       (oos_prediction)
v4: t_monitor_six   -> predicted t_monitor_five       (oos_prediction)
v4: t_check_funding -> predicted t_assess_eligibility / t_prepare_pack (oos)
```

This supports the earlier hypothesis:

```text
topology helps define the valid area, but class-label candidate scoring alone
does not guarantee exact next activity under ambiguous prefixes or changed
version routing.
```

## Why Knowledge Transfer Did Not Appear

### 1. Candidate Space Did Not Actually Change Per Version

The expected benefit requires:

```text
new topology -> new/changed candidate set -> model scores candidates from topology
```

But current diagnostics show:

```text
C_v = 37 for every version
duplicate_candidate_count_max = 1
```

So the candidate axis behaves like the old fixed activity vocabulary. Added or
removed BPMN nodes are not represented as stable topology-local candidate ids in
the loss/evaluation contract.

### 2. The Model Uses Activity Labels, Not BPMN Node Identity

`candidate_class_index` is still activity-label based. For XES/collapse mode,
this is expected, but it means the model cannot distinguish:

```text
same activity label in different structural positions
new BPMN node with an already-known label
removed BPMN node with same label still present elsewhere
```

The current loss answers:

```text
which activity label?
```

not:

```text
which topology candidate node in this version?
```

### 3. `learning_strategy=standard` Does Not Force Topology Dependence

The config contains topology-conditioning flags, but the active strategy is:

```text
training.learning_strategy = standard
```

Therefore wrong-version topology negatives, drop-edge negatives, allowed-set
loss, and retention logic are not the active training methodology. The model is
not explicitly trained that:

```text
correct topology should improve predict ion;
wrong/corrupted topology should worsen prediction.
```

### 4. No Runtime Statistics Are Used

```text
experiment.statistic_enabled = false
```

So structure carries only topology/identity signal, not per-version execution
evidence. This is fine for the clean zero-shot hypothesis, but it limits the
signal available to separate competing candidates that are topologically close.

### 5. Trace Metadata Has a Diagnostic Bug

Trace attributes show:

```text
model_type = EOPKGTopologyConditioned
fusion_mode = ""
structural_mode = false
```

while run params show:

```text
model.fusion_mode = StructXAttn
model.structural_mode = True
experiment.structural_mode = True
```

This does not explain the metrics, but it weakens trace filtering and should be
fixed. `EOPKGTopologyConditioned` should expose trace metadata explicitly.

## Conclusions

1. The implementation is valid enough to continue experiments: no missing
   target mapping, no mixed-topology crash, no candidate loss failure.
2. The current `candidate_id` experiment does not yet test the full business
   zero-shot claim, because the topology-local candidate axis collapses to the
   same activity-label set.
3. The main bottleneck is not the trainer now; it is the candidate identity
   contract and training methodology.
4. `890...` looks better because it is less overconfident and produces fewer OOS
   predictions, not because it demonstrates stronger topology transfer.
5. The next experiment should make topology identity observable and measurable:
   candidate ids / BPMN node ids, candidate-level OOS/calibration, and a
   topology-dependence training objective.

## Recommended Next Steps

### P0: Fix Trace Metadata

Expose correct `fusion_mode` and `structural_mode` for `EOPKGTopologyConditioned`
so traces can be filtered reliably.

Expected fields:

```text
model_type = EOPKGTopologyConditioned
fusion_mode = TopologyConditionedCandidateScoring or candidate_scoring
structural_mode = true
candidate_contract_mode = candidate_id
```

### P0: Add Candidate Identity Audit

Before changing model math, add an audit that logs per version:

```text
process_version
candidate_dynamic_count
unique_activity_class_count
bpmn_target_candidate_count
added_candidate_ids_vs_train
removed_candidate_ids_vs_train
candidate_class_index coverage
```

This will answer whether the runtime candidate space is really version-local.

### P1: Move From Activity-Class Candidates To Stable BPMN Candidate IDs

Add stable candidate ids:

```text
candidate_id = process_id + process_version + bpmn_node_id
candidate_class = activity label
candidate_semantic_label = normalized task label
```

Keep global activity-label metrics for comparison, but train/evaluate candidate
selection in candidate-id space where possible.

### P1: Candidate-Level OOS / Calibration

Current OOS/calibration is still reported after mapping back to global labels.
Add direct candidate-space diagnostics:

```text
candidate_oos_rate
candidate_target_in_candidate_mask_rate
candidate_ece
candidate_nll
candidate_topology_version_shift
```

### P1: Activate A Controlled Topology-Dependence Strategy

After candidate identity audit, test a minimal topology-conditioned objective:

```text
correct topology score > wrong-version topology score
```

Do this first as an auxiliary diagnostic objective, not as a heavy loss. The
current run shows the model can become overconfident; avoid aggressive weights.

### P2: Re-test With Statistics After Candidate Identity Is Observable

Statistics should not be used to hide the candidate-contract problem. Add stats
only after candidate ids and candidate-level diagnostics are visible.

