# MVP2.5 Stage 4.2 - XES Stats Fix Execution Update

Date: 2026-03-20
Status: Done

## Scope completed

1. Applied statistics calculation fix for XES -> Neo4j path in `sync-stats`.
2. Applied `OPEN-Q6` decision:
   - for versions without ranked `vN` semantics, keep exact-match only.
3. Preserved agreed behavior:
   - `skip` + warning when no scope events exist,
   - auto `as_of=max(event_ts)` using train-cut (anti-leakage),
   - visible logging of selected temporal boundaries.

## Validation status

1. `.\.venv\Scripts\python.exe -m pytest tests\application\test_sync_stats_tool.py -v` -> `6 passed`
2. `.\.venv\Scripts\python.exe -m pytest -m mvp1_regression -v` -> `17 passed`

## Remaining follow-up

1. Runtime quality gate for zero-dominant stats tensors (`>=95% zeros`) in graph builder path.
2. Additional technical debt review/report after runtime quality gate is merged.

## Update 2 (2026-03-20)

Status: Done

1. Added `skipped_details` to sync-stats summary JSON with explicit reason codes:
   - `no_versions_for_process`
   - `invalid_process_namespace`
   - `no_events_after_train_cut`
   - `process_structure_not_found`
   - `no_scope_events_up_to_as_of`
2. Kept warning logs in runtime, but now skip diagnostics are persisted into output summary for post-run analysis.
3. Removed noisy Neo4j DBMS notification source for `parent_scope_id` by switching to safe projection:
   - `coalesce(n.parent_scope_id, '') AS parent_scope_id`

Validation:
1. `.\.venv\Scripts\python.exe -m pytest tests\application\test_sync_stats_tool.py -v` -> `6 passed`
2. `.\.venv\Scripts\python.exe -m pytest -m mvp1_regression -v` -> `17 passed`
