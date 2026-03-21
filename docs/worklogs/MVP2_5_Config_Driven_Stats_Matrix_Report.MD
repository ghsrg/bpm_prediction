# MVP2.5 Config-Driven Stats Matrix Report

Date: 2026-03-20  
Status: Completed

## Scope

Implemented step: `Config -> canonical stats feature matrix` with explicit `encoding` support in `DynamicGraphBuilder`.

No model wiring changes were introduced in this step.

## Code changes

Updated:
1. `src/domain/services/dynamic_graph_builder.py`

Added:
1. `tests/domain/test_dynamic_graph_builder_stats_matrix_mapping.py`

## Functional changes

### 1. Node stats matrix (`struct_x`) is now encoding-aware

For each `mapping.graph_feature_mapping.node_numeric[*]` spec:
1. Builder resolves canonical key `window.scope.metric`.
2. Builds raw per-node column using stats_index + `default`.
3. Applies sequential encodings from config.

Supported encodings:
1. `identity` / `none` / `embedding` -> no-op
2. `log1p` -> signed `log1p(abs(x))`
3. `z-score` -> column standardization (`mean/std`, safe zero fallback for degenerate columns)

Unknown encoding values are treated as no-op for backward compatibility.

### 2. Edge weights support encoding too

`mapping.graph_feature_mapping.edge_weight.encoding` is supported with the same numeric encoding pipeline.

### 3. Backward compatibility

If `encoding` is not provided in graph feature mapping:
1. default behavior remains `identity`
2. existing configs continue to work unchanged.

## Tests

New tests:
1. `test_struct_x_identity_mapping_uses_defaults`
2. `test_struct_x_applies_log1p_then_zscore`
3. `test_edge_weight_applies_encoding`

Also validated with previously added quality-gate tests and sync-stats tests.

Validation commands:
1. `.\.venv\Scripts\python.exe -m pytest tests\domain\test_dynamic_graph_builder_stats_matrix_mapping.py tests\domain\test_dynamic_graph_builder_stats_quality_gate.py tests\application\test_sync_stats_tool.py -v` -> `12 passed`
2. `.\.venv\Scripts\python.exe -m pytest -m mvp1_regression -v` -> `17 passed`

## Outcome

Canonical stats matrix formation is now controlled by config and `encoding` semantics, ready for the next step: direct model-path wiring/usage policy over this canonical matrix.
