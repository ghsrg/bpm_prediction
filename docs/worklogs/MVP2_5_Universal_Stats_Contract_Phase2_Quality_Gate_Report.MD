# MVP2.5 Universal Stats Contract - Phase 2 Quality Gate Report

Date: 2026-03-20  
Status: Completed (producer + consumer quality gate)

## Scope

Implemented quality gate and acceptance thresholds for:
1. Producer (`sync-stats`)
2. Consumer (`DynamicGraphBuilder`)

This phase follows universal contract v1 introduced in Phase 1.

## Code changes

Updated:
1. `tools/sync_stats.py`
2. `src/domain/services/dynamic_graph_builder.py`
3. `tests/application/test_sync_stats_tool.py`
4. `tests/domain/test_dynamic_graph_builder_stats_quality_gate.py` (new)

## Producer gate (`sync-stats`)

Added config block:
```yaml
sync_stats:
  quality_gate:
    enabled: true
    zero_dominant_threshold: 0.95
    min_non_zero_ratio_overall: 0.0
    min_history_coverage_percent: 0.0
    on_fail: "write_with_flag"  # write_with_flag | skip_snapshot
    warn_on_fail: true
```

Behavior:
1. Computes quality failures against thresholds.
2. Writes `quality_reason` and `quality_failures` to contract summary/metadata.
3. If `on_fail=write_with_flag`: snapshot is persisted with unusable flag.
4. If `on_fail=skip_snapshot`: snapshot is skipped and added to `skipped_details` with reason `quality_gate_rejected`.

## Consumer gate (`DynamicGraphBuilder`)

Added gate config (inside `graph_feature_mapping`):
```yaml
mapping:
  graph_feature_mapping:
    stats_quality_gate:
      enabled: true
      require_quality_metadata: false
      zero_dominant_threshold: 0.95
      min_non_zero_ratio_overall: 0.0
      min_history_coverage_percent: 0.0
      on_fail: "ignore_with_warning"  # ignore_with_warning | allow_with_warning | raise
      warn_on_fail: true
```

Behavior:
1. Reads `metadata.stats_contract.quality`.
2. Evaluates thresholds and producer usability flag.
3. Policy actions:
   - `ignore_with_warning`: disable stats tensors (`struct_x`), fallback to structural defaults.
   - `allow_with_warning`: keep stats tensors, only warning.
   - `raise`: fail fast with exception.
4. Warnings are deduplicated per `(process, version, reason)` to avoid spam.

## Tests

Added/updated tests:
1. `test_sync_stats_producer_quality_gate_can_skip_snapshot`
2. `test_dynamic_graph_builder_quality_gate_ignores_stats_on_fail`
3. `test_dynamic_graph_builder_quality_gate_allows_stats_when_policy_allows`

Validation:
1. `.\.venv\Scripts\python.exe -m pytest tests\application\test_sync_stats_tool.py tests\domain\test_dynamic_graph_builder_stats_quality_gate.py -v` -> `9 passed`
2. `.\.venv\Scripts\python.exe -m pytest -m mvp1_regression -v` -> `17 passed`

## Result

Quality gate and fallback policy are now active in both producer and consumer paths, with configurable thresholds and explicit warning behavior.
