# MVP2.5 Universal Stats Contract - Phase 1 Report

Date: 2026-03-20  
Status: Completed (contract introduction)

## Scope

Implemented initial transition to universal stats contract in `sync-stats` without enabling runtime gating behavior changes.

## Changes

Updated:
1. `tools/sync_stats.py`
2. `tests/application/test_sync_stats_tool.py`

### Contract additions

1. Added top-level summary field:
   - `stats_contract_version = "1.0"`
2. Added metadata block:
   - `metadata.stats_contract`
   - fields:
     - `version`
     - `key_format`
     - `identity`
     - `quality`
     - `policy`
3. Preserved backward-compatible fields:
   - `metadata.stats_index`
   - `metadata.stats_policy`
4. Extended per-version summary details with quality telemetry:
   - `non_zero_ratio_node`
   - `non_zero_ratio_edge`
   - `non_zero_ratio_global`
   - `non_zero_ratio_overall`
   - `is_usable_for_training`
   - `quality_reason`

### Quality fields (producer side, informational in Phase 1)

Computed and persisted:
1. non-zero ratios by node/edge/global/overall
2. `zero_dominant` flag (`>=95%` zeros threshold)
3. `is_usable_for_training`
4. `quality_reason`

Note:
1. In this phase quality is persisted only; consumer gate behavior is planned next.

## Validation

1. `.\.venv\Scripts\python.exe -m pytest tests\application\test_sync_stats_tool.py -v` -> `6 passed`
2. `.\.venv\Scripts\python.exe -m pytest -m mvp1_regression -v` -> `17 passed`

## Next step

Implement runtime quality gate for:
1. Producer policy finalization (`sync-stats` thresholds as config knobs).
2. Consumer enforcement in `DynamicGraphBuilder` (ignore stats tensors when quality gate fails, warn but do not fail run).
