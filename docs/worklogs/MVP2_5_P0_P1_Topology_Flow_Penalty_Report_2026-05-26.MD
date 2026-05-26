# MVP2.5 P0/P1 Topology Flow Penalty Report

- `date`: 2026-05-26
- `scope`: EOPKGTopologyConditioned candidate-id diagnostics and topology-flow loss
- `status`: implemented

## Context

Previous `EOPKGTopologyConditioned` runs showed that the true candidate path was
wired, but drift quality stayed close to baseline. For the current loan BPMN
versions, the active case is simpler than the original duplicate-activity
scenario:

```text
candidate_dynamic_count = 37
duplicate_candidate_count_max = 1
candidate_target_in_candidate_set_rate = 1.0
candidate_missing_target_rate = 0.0
```

So the current dominant hypothesis is not MIL ambiguity. The more likely
problem is that the model sees the topology-local candidate set but does not
learn to respect changed topology flow/edges strongly enough.

## Implemented

### P0 Trace Metadata

`EOPKGTopologyConditioned` now exposes stable run metadata:

```text
structural_mode = true
fusion_mode = TopologyConditionedCandidateScoring
candidate_scoring_mode
candidate_pooling
```

Selective trace attributes now include these fields as flat searchable
attributes.

### P1 Candidate Flow Diagnostics

Added projection from global allowed mask to candidate axis:

```text
candidate_allowed_mask[b, j] = allowed_target_mask[b, candidate_class_index[j]]
```

Added train/eval/drift metrics:

```text
candidate_oos_rate
candidate_invalid_probability_mass
candidate_valid_probability_mass
candidate_valid_invalid_logit_margin
```

These metrics are logged in normal evaluation and drift-window rows where
`candidate_contract_mode=candidate_id` and `allowed_target_mask` is available.

### P1 Topology Flow Penalty

Added optional training objective:

```yaml
training.topology_flow_penalty_enabled: true
training.topology_flow_penalty_weight: 0.03
training.topology_flow_penalty_type: invalid_probability_mass
training.topology_flow_penalty_margin: 0.1
training.topology_flow_penalty_fail_on_missing_mask: true
```

Default penalty:

```text
L_flow = mean invalid probability mass over candidate logits
```

Alternative:

```text
L_flow = relu(max_invalid_logit - max_valid_logit + margin)
```

The default is intentionally bounded and should be safer for first sweeps.

### Candidate-ID Compatibility Guard

For `candidate_contract_mode=candidate_id`, legacy `wrong_version` and
`drop_edges` CE-negative objectives are skipped and logged as skipped
candidate-id batches. They are not mathematically valid for the dynamic
candidate axis without a separate candidate-space implementation.

## Suggested Next Run

Use `_EOPKGTC-UN` with:

```yaml
training.learning_strategy: topology_conditioned
training.candidate_contract_mode: candidate_id
training.candidate_batch_topology_policy: single_topology_required
training.topology_conditioning_wrong_version_negative_enabled: false
training.topology_conditioning_drop_edges_negative_enabled: false
training.topology_flow_penalty_enabled: true
training.topology_flow_penalty_weight: 0.03
training.topology_flow_penalty_type: invalid_probability_mass
```

Run a small sweep:

```text
topology_flow_penalty_weight: 0.01, 0.03, 0.05, 0.10
```

Compare to the same checkpoint/preset with `topology_flow_penalty_enabled=false`.

## What To Inspect

Primary:

```text
drift_window_strict_macro_f1
drift_window_candidate_oos
drift_window_candidate_invalid_probability_mass
drift_window_candidate_valid_invalid_logit_margin
drift_window_test_oos
drift_window_test_ece
```

Expected useful signal:

- lower `candidate_invalid_probability_mass` on future versions;
- lower `candidate_oos_rate`;
- no major NLL/ECE degradation;
- strict-F1 improvement is desirable, but not the first validation criterion.

## Verification

Targeted tests passed:

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\application\test_topology_conditioned_learning.py tests\application\test_trainer_topology_conditioned_learning.py tests\application\test_evaluator_metrics.py tests\application\test_structural_trace_payload_builder.py tests\application\test_experiment_ui_catalog_fields.py tests\application\test_drift_one_pass_evaluation.py tests\domain\test_eopkg_topology_conditioned.py -v
```

Result:

```text
77 passed, 3 warnings
```

Architecture guard passed:

```powershell
.\.venv-modern\Scripts\python.exe tools\architecture_guard.py
```

Result:

```text
[ARCH_GUARD] OK: architecture dependency rules satisfied.
```

Full regression passed:

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\ -v
```

Result:

```text
522 passed, 4 warnings
```
