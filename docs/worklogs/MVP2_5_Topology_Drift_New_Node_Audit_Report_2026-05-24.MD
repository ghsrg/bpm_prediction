# MVP2.5 Topology Drift New-Node Audit Report

## Summary

Implemented the first offline topology drift audit for checking whether drift
degradation is related to added/removed BPMN candidates, changed
gateway-collapsed transitions, OOS predictions, or fixed classifier limitations.

## Implemented Files

- `src/application/services/topology_drift_audit.py`
  - topology snapshot dataclasses;
  - gateway-collapsed BPMN parser;
  - topology diff;
  - prediction error attribution.
- `tools/audit_topology_drift.py`
  - CLI for local `mlruns` + BPMN source files;
  - writes JSON/CSV/Markdown audit artifacts.
- `tests/application/test_topology_drift_audit.py`
  - synthetic topology diff tests;
  - attribution tests;
  - BPMN gateway-collapse parser tests;
  - XES `(trace_idx, prefix_len) -> prefix_last_activity` backfill test.
- `src/application/services/structural_trace_payload_builder.py`
  - future selective traces now include `prefix_last_activity_index` and
    `prefix_last_activity`.

## First Smoke Run

Command:

```powershell
.\.venv-modern\Scripts\python.exe tools\audit_topology_drift.py `
  --mlruns-dir mlruns `
  --experiment-id 854778689611649472 `
  --baseline-run 7681dd2ef7f841e68591e0b1145296ae `
  --structural-run 5af1d05da0524106b4031dd7c39767c6 `
  --train-run 9de636ce471745d9b818df6edefd73c2 `
  --audit-id loan_xattn_5af1d05 `
  --output-dir outputs\audits\topology_drift
```

Output:

```text
outputs/audits/topology_drift/loan_xattn_5af1d05/
```

Key summary:

```json
{
  "new_activity_labels_present": true,
  "added_activity_labels": ["Activity_07k7tyn"],
  "fixed_head_limitation_detected": true,
  "error_buckets": {
    "correct": 5,
    "oos_prediction": 13,
    "strict_error_but_allowed": 22
  }
}
```

The added activity id corresponds to `Monitor Ten` in
`data/camunda_exports/bpmn_xml/loan_v4_re_pl_cb.bpmn`.

Gateway-collapsed new transitions include:

```text
t_confirm_payout -> Activity_07k7tyn
Activity_07k7tyn -> t_update_risk
```

## Prefix Last Activity Update

Implemented `prefix_last_activity` support after the first audit:

- future graph payloads carry `prefix_last_activity_idx`;
- future structural traces expose `sample.prefix_last_activity` and flat
  `prefix_last_activity` attributes;
- old trace artifacts can be re-audited without rerunning train/eval_drift by
  passing `--log-path`; the audit reconstructs the last activity from XES using
  `(trace_idx, prefix_len)`.

Backfilled audit command:

```powershell
.\.venv-modern\Scripts\python.exe tools\audit_topology_drift.py `
  --experiment-id 854778689611649472 `
  --baseline-run 7681dd2ef7f841e68591e0b1145296ae `
  --structural-run 5af1d05da0524106b4031dd7c39767c6 `
  --train-run b624f13b76874201a28bd55e55927c37 `
  --train-versions v1,v2 `
  --eval-versions v1,v2,v3,v4 `
  --audit-id loan_xattn_5af1d05_prefix_last_activity `
  --log-path outputs\simulation\loan_v1_v4_simulated.xes
```

Output:

```text
outputs/audits/topology_drift/loan_xattn_5af1d05_prefix_last_activity/
```

Updated attribution:

```json
{
  "changed_transition_error_count": 2,
  "error_buckets": {
    "changed_transition_zone": 2,
    "correct": 5,
    "oos_prediction": 13,
    "strict_error_but_allowed": 20
  }
}
```

## Verification

Targeted and full tests:

```text
tools/architecture_guard.py - OK
related pytest subset - 36 passed, 3 warnings
tests/ - 461 passed, 3 warnings
```
