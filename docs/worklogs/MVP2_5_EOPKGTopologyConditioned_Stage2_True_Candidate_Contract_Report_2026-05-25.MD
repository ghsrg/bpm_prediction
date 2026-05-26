# MVP2.5 EOPKGTopologyConditioned Stage 2 True Candidate Contract Report

- `date`: 2026-05-25
- `scope`: minimal key implementation for `candidate_contract_mode=candidate_id`
- `status`: implemented and targeted-tested

## Context

Previous Stage 2 wiring let `EOPKGTopologyConditioned.forward_candidate()` return
topology-local candidate logits `[B, C_v]`, but runtime still primarily used a
projection back to `[B, C_train]`. That validated wiring, but did not test the
main hypothesis that topology should define the prediction candidate space.

## Implemented

1. Added `src/application/services/candidate_target_mapping.py`.
2. Added set-aware candidate CE:

```text
target_candidate_mask[b, j] = candidate_class_index[j] == y_b
loss_b = -logsumexp(logits_b[target_candidate_mask_b]) + logsumexp(logits_b[:])
```

3. Added candidate-level train path for:

```yaml
training.candidate_contract_mode: candidate_id
```

4. Added candidate-level eval/drift compatibility:
   - train loss uses `[B, C_v]` directly;
   - metrics still map candidate predictions/probabilities back to global
     activity labels for existing F1/OOS/calibration dashboards.

5. Added target mapping diagnostics:

```text
candidate_target_in_candidate_set_rate
candidate_missing_target_rate
candidate_target_duplicate_count_max
candidate_target_set_logit_variance_mean
candidate_target_set_entropy_mean
```

6. Added safety threshold:

```yaml
training.candidate_missing_target_fail_threshold: 0.10
```

Training fails when the missing-target rate exceeds this threshold, because that
means log labels cannot be mapped to the topology-local candidate set.

## Current Semantics

`candidate_id` is now a real candidate-level loss path, not only a reserved
mode. It still reports metrics in the fixed global activity-label space for
comparison with previous experiments.

Important limitation: the current target is still an activity label, not an
exact BPMN node id. If duplicate BPMN nodes map to the same activity label, the
loss treats all matching candidates as the target set. Exact duplicate-node
supervision remains future work and requires log sources that can provide BPMN
node identity.

## Recommended Test Run

Start with a small run:

```yaml
training.candidate_contract_mode: candidate_id
training.candidate_batch_topology_policy: single_topology_required
training.candidate_missing_target_fail_threshold: 0.10
experiment.gateway_mode: collapse_for_prediction
```

Then check:

- `candidate_target_in_candidate_set_rate` should be close to `1.0`;
- `candidate_missing_target_rate` should be `0.0` or very low;
- non-zero `candidate_target_set_logit_variance_mean` indicates the model starts
  differentiating duplicate candidate nodes;
- if `candidate_missing_target_rate > 0.10`, the run is invalid and should be
  fixed at log/topology alignment level.

## Verification

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\application\test_evaluator_metrics.py tests\application\test_candidate_target_mapping.py tests\application\test_trainer_forward_stats_logging.py tests\application\test_experiment_ui_catalog_fields.py tests\domain\test_eopkg_topology_conditioned.py -v
```

Result:

```text
75 passed
```

