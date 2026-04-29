# MVP2.5 Topology Projection Alignment Implementation Report

Date: 2026-04-29
Status: implemented and pytest-verified with project venv using escalated permissions

## Summary

Implemented the `topology_projection_alignment` plan from
`docs/worklogs/MVP2_5_Topology_Projection_Alignment_Root_Cause_Audit_2026-04-29.MD`.

The implementation keeps `duplicate_activity_identity_ambiguity` separate:
duplicate BPMN labels are detected and counted, but not resolved into node-level
targets.

## What Changed

### Projection Compiler

Added `src/domain/services/topology_projection_alignment.py`.

It provides:

- `TopologyProjectionCompiler`
- `TopologyProjectionResult`
- `TopologyProjectionDiagnostics`

Diagnostics include:

- `is_aligned`
- `failure_reasons`
- `missing_vocab_nodes`
- `skipped_projected_edges`
- `missing_node_metadata`
- `duplicate_activity_labels`
- `projected_edge_count`
- `source_path_count`
- structural payload alignment checks for `struct_x_rows` and edge index max.

### DynamicGraphBuilder

Updated `src/domain/services/dynamic_graph_builder.py`.

Runtime now:

- uses `TopologyProjectionCompiler` for `preserve` and `collapse_for_prediction`;
- records projection diagnostics in the compiled topology cache;
- exposes scalar diagnostics in `GraphTensorContract`;
- supports `topology_projection.diagnostics_enabled`;
- supports `topology_projection.on_fail: warn | raise | disable_struct`;
- includes projection-relevant `dto.nodes` metadata in topology cache keys;
- disables structural payload and structural mask when `on_fail: disable_struct`.

### Tensor Contract

Updated `src/domain/entities/tensor_contract.py`.

Added scalar/batched projection diagnostic fields. These are intentionally not a
large dict, so the existing tensor-purity rule remains intact.

### Trainer Visibility

Updated `src/application/use_cases/trainer.py`.

Training, inference, and drift evaluation forward logs now include:

```text
topology_projection_aligned[true/false]=...
topology_projection_projected_edges_avg=...
topology_projection_source_paths_avg=...
topology_projection_skipped_edges=...
topology_projection_missing_vocab=...
topology_projection_duplicate_labels=...
topology_projection_missing_node_metadata=...
```

This is where to check during normal training and `eval_drift` whether topology
projection mapping was clean.

### Model Fail-Fast

Updated `src/domain/models/eopkg_models.py`.

`EOPKGGATv2` no longer modulo-repairs structural edge indices. If
`structural_edge_index.max() >= struct_x rows / class count`, it raises a clear
`ValueError`.

### Config Catalog

Updated `configs/ui/config_catalog.yaml`.

Added:

```yaml
mapping.graph_feature_mapping.topology_projection.diagnostics_enabled
mapping.graph_feature_mapping.topology_projection.on_fail
```

## Tests Added

Added:

- `tests/domain/test_topology_projection_alignment.py`
  - simple gateway collapse;
  - missing vocab endpoint diagnostics;
  - missing node metadata diagnostics;
  - multiple transparent paths;
  - duplicate activity label detection.

Updated:

- `tests/domain/test_dynamic_graph_builder_masks.py`
  - strict `on_fail: raise`;
  - scalar projection diagnostics in contract;
  - cache key changes when node metadata changes.
- `tests/domain/test_eopkg_gatv2_dual_encoder.py`
  - out-of-bounds structural edge index fail-fast.
- `tests/application/test_trainer_forward_stats_logging.py`
  - forward stats logs projection diagnostics.

## How To Configure Research Runs

Recommended strict config:

```yaml
mapping:
  graph_feature_mapping:
    topology_projection:
      gateway_mode: collapse_for_prediction
      diagnostics_enabled: true
      on_fail: raise
```

Development-compatible config:

```yaml
mapping:
  graph_feature_mapping:
    topology_projection:
      gateway_mode: collapse_for_prediction
      diagnostics_enabled: true
      on_fail: warn
```

Fallback config when you want to continue without structural tensors:

```yaml
mapping:
  graph_feature_mapping:
    topology_projection:
      gateway_mode: collapse_for_prediction
      diagnostics_enabled: true
      on_fail: disable_struct
```

## How To Verify In Training Logs

In `train`, `inference`, and `eval_drift`, look for `Forward stats [...]`.

Clean mapping should look like:

```text
topology_projection_aligned[true/false]=N/0
topology_projection_skipped_edges=0
topology_projection_missing_vocab=0
topology_projection_missing_node_metadata=0
```

Signals of a problem:

- `topology_projection_aligned[true/false]=.../M`, where `M > 0`;
- `topology_projection_skipped_edges > 0`;
- `topology_projection_missing_vocab > 0`;
- `topology_projection_missing_node_metadata > 0`;
- `topology_projection_duplicate_labels > 0`.

For drift evaluation, the same counters appear under:

```text
Forward stats [eval_drift]: ...
```

## Verification Performed

Passed:

```powershell
C:\Users\korsr\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe -m py_compile src/domain/services/topology_projection_alignment.py src/domain/services/dynamic_graph_builder.py src/domain/entities/tensor_contract.py src/domain/models/eopkg_models.py src/application/use_cases/trainer.py tests/domain/test_topology_projection_alignment.py tests/domain/test_dynamic_graph_builder_masks.py tests/domain/test_eopkg_gatv2_dual_encoder.py tests/application/test_trainer_forward_stats_logging.py
```

Passed with escalated permissions, because Codex sandbox blocks the venv base
interpreter process creation otherwise:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/domain/test_topology_projection_alignment.py
```

Result:

```text
5 passed
```

Passed with escalated permissions:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/domain/test_dynamic_graph_builder_masks.py tests/domain/test_eopkg_gatv2_dual_encoder.py tests/application/test_trainer_forward_stats_logging.py tests/integration/test_bpmn_xes_stats_to_gnn_cycle.py
```

Result:

```text
21 passed, 11 warnings
```

Passed with escalated permissions:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/domain tests/application/test_trainer_forward_stats_logging.py tests/integration/test_bpmn_xes_stats_to_gnn_cycle.py
```

Result:

```text
83 passed, 11 warnings
```

## Residual Risks

- Run-level JSON artifact beyond forward logs is not expanded yet; diagnostics
  are visible in forward logs and scalar contract fields.
- Duplicate activity labels are only detected and counted, not resolved.
