# MVP2.5 Activity-to-Topology Alignment Gate Report

Date: 2026-04-28
Status: completed
Debt: `activity_to_topology_alignment_gate`
ADR: `docs/adr/0006-research-grade-activity-topology-alignment-gate.md`

## Summary

Closed the producer-side activity-to-topology alignment gate debt.

The alignment computation was extracted from `tools/sync_stats.py` into a pure
domain service:

- `src/domain/services/activity_topology_alignment_service.py`

`sync-stats` now remains the orchestration layer:

- config parsing,
- event loading,
- repository reads/writes,
- fail policy handling,
- summary output.

The domain service owns:

- alignment profiles,
- candidate field matching,
- normalization,
- loggable-node denominator semantics,
- ambiguity diagnostics,
- alignment metrics.

## Implemented Profiles

1. `legacy_exact`
   - Compatibility profile.
   - Exact `activity_def_id` to topology node id matching.
   - Preserves existing `sync-stats` behavior by default.

2. `safe_normalized`
   - Uses `id` and `name` node candidate fields.
   - Matches exact id/name first, then normalized id/name.
   - Normalizes case and separators.
   - Strips lifecycle classifier suffixes.
   - Excludes structural-only BPMN nodes from `node_coverage`.
   - Fails closed on ambiguity.

3. `research_strict`
   - Uses safe matching semantics.
   - Provides strict default thresholds for research-grade runs.
   - Intended to be combined with `on_fail: raise`.

## Runtime Contract

Alignment metadata is persisted under:

```text
metadata.stats_contract.alignment
```

New diagnostic fields include:

- `profile`
- `loggable_node_count`
- `ignored_structural_node_count`
- `matched_loggable_node_count`
- `ambiguous_event_activities_top`
- `match_counts_by_strategy`
- `candidate_node_fields`

## Tests Added or Updated

Added pure domain tests:

- `tests/domain/test_activity_topology_alignment_service.py`

Added real-object application tests:

- `tests/application/test_sync_stats_alignment_gate_real_objects.py`

Updated integration coverage:

- `tests/integration/test_bpmn_xes_stats_to_gnn_cycle.py`

The BPMN/XES integration test now uses non-zero alignment thresholds with
`profile: safe_normalized` and `on_fail: raise`.

## Documentation Updated

- `docs/LLD_MVP2_5.MD`
- `docs/DATA_MODEL_MVP2_5.MD`
- `docs/runbooks/mvp2_5-commands.md`
- `docs/current/project-state.md`
- `docs/current/architecture-debt.md`
- `docs/adr/0006-research-grade-activity-topology-alignment-gate.md`
- `configs/ui/config_catalog.yaml`

## Follow-up: Desktop Experiment UI Visibility

After the initial implementation, `profile` was present in
`configs/ui/config_catalog.yaml`, but desktop `tools/experiment_ui.py` did not
show catalog-only `sync_stats` fields unless they also existed in scanned
experiment YAML files.

Fix:

- added `_merge_catalog_section_fields(...)` to merge base config fields,
  scanned pool fields, and catalog-only fields for a UI section,
- used it for `sync_stats_form`,
- added regression coverage in
  `tests/application/test_experiment_ui_catalog_fields.py`.

Verified actual catalog defaults visible through the desktop helper:

```text
sync_stats.alignment_gate.profile -> legacy_exact
sync_stats.alignment_gate.on_fail -> write_with_flag
```

No partial non-runnable experiment config was added. The research-strict config
snippet was documented in the runbook instead.

## Verification

Commands were run through the project venv with escalation because the Codex
sandbox blocks the base interpreter referenced by `.venv\pyvenv.cfg`.

```powershell
.\.venv\Scripts\python.exe tools\architecture_guard.py
```

Result:

```text
[ARCH_GUARD] OK: architecture dependency rules satisfied.
```

```powershell
.\.venv\Scripts\python.exe -m pytest tests/domain/test_activity_topology_alignment_service.py tests/application/test_sync_stats_alignment_gate_real_objects.py tests/application/test_sync_stats_tool.py tests/integration/test_bpmn_xes_stats_to_gnn_cycle.py -v
```

Result:

```text
19 passed
```

```powershell
.\.venv\Scripts\python.exe -m pytest -m mvp1_regression -v
```

Result:

```text
20 passed, 260 deselected
```

```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -v
```

Result:

```text
280 passed
```

Additional desktop UI regression after the follow-up fix:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/application/test_experiment_ui_catalog_fields.py -v
```

Result:

```text
1 passed
```

## Residual Notes

`activity_to_topology_alignment_gate` is closed. The next related P0 debt is
`topology_projection_alignment`, which covers post-projection consistency rather
than producer-side stats alignment.
