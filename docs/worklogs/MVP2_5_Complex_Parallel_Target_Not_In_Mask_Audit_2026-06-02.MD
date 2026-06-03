# MVP2.5 Complex Parallel Target-Not-In-Mask Audit

Date: 2026-06-02

Scope:

- Dataset: `outputs/simulation/loan_v1_v5_complex_simulated.xes`
- Stats: `outputs/simulation/loan_v1_v5_complex_simulated.dataset_stats.json`
- Drift runs:
  - EOPKGTopologyConditioned: `d3921cde97a14e3c90a0a27d0cc03662`
  - BaselineGATv2: `66ea776a9695435c909c5ced93d905c7`

## Finding

`target_in_mask_rate` is low on the drift tail:

- EOPKGTopologyConditioned tail mean: about `0.529`
- Baseline tail mean: about `0.529`

This is not a model-only issue. The current last-node mask often does not
contain the true next complete activity in the complex parallel dataset.

## Version Semantics Check

XES trace-level and event-level `concept:version` are consistent:

- `mismatch_traces = 0`
- trace counts: `v1=1148`, `v2=1023`, `v3=1218`, `v4=1047`, `v5=1123`

Therefore, the low target-in-mask rate is not explained by drift windows using
the wrong process version.

## Parallel Active-Set Audit

For each transition between consecutive `complete` events, the audit checked
whether the target complete activity was already active after the previous
complete event.

Summary:

| version | complete transitions | target active after previous complete | rate | same timestamp rate |
|---|---:|---:|---:|---:|
| v1 | 30353 | 10049 | 0.331 | 0.197 |
| v2 | 27207 | 6292 | 0.231 | 0.099 |
| v3 | 32846 | 6330 | 0.193 | 0.064 |
| v4 | 25103 | 4170 | 0.166 | 0.047 |
| v5 | 29303 | 4748 | 0.162 | 0.048 |

Frequent examples include monitoring and parallel checks:

- `t_monitor_one -> t_monitor_four`
- `t_monitor_four -> t_monitor_two`
- `t_identity_check -> t_property_check`
- `t_property_check -> t_credit_check`

In these cases, the next complete event is often not a direct successor of the
previous completed node. It is another already active task from a parallel
branch.

## Interpretation

For a parallel process, a mask based only on:

```text
last_completed_activity -> direct BPMN successors
```

is not a valid oracle. The correct candidate set should include currently active
tasks or be derived from reconstructed process marking.

## Methodology Impact

The current complex-dataset result still supports the structure hypothesis:

- EOPKGTopologyConditioned improves strict F1, macro F1, OOS, ECE, and NLL over
  BaselineGATv2.

However, the achievable strict-F1 is artificially capped by an incomplete mask
semantics for parallel/interleaved XES logs.

## Next Options

Recommended order:

1. Add `active_candidates` / active task set to the simulator/XES output and use
   it in `DynamicGraphBuilder`.
2. For logs without active state, reconstruct process marking from lifecycle
   events when possible.
3. For pure XES task-label logs without lifecycle/marking, use a relaxed
   reachability/window mask and report it as exploratory or approximate.

