# MVP2.5 Stage4.2 - GNN Stats Forward Interface + Logging Report

Date: 2026-03-20  
Status: Completed

## Scope

Task:
1. Complete transfer of structural/statistical snapshot parameters into GNN forward interface.
2. Add runtime logs for train/inference/eval_drift showing:
   - count/shape characteristics of forwarded stats tensors
   - snapshot characteristics (`knowledge_version`, `as_of_ts`)

## Implemented Changes

Updated:
1. `src/domain/entities/tensor_contract.py`
2. `src/domain/services/dynamic_graph_builder.py`
3. `src/application/use_cases/trainer.py`
4. `src/cli.py`
5. `tests/domain/test_dynamic_graph_builder_stats_matrix_mapping.py`

Added:
1. `tests/application/test_trainer_forward_stats_logging.py`

## Functional Details

### 1) Snapshot metadata is now attached to GraphTensorContract

`DynamicGraphBuilder` now writes:
1. `stats_snapshot_version`
2. `stats_snapshot_as_of_ts`
3. `stats_snapshot_process_version`
4. `stats_quality_reason`
5. `stats_allowed`

Source:
1. `dto.metadata.knowledge_version`
2. `dto.metadata.as_of_ts`
3. `dto.version`
4. Quality-gate decision reason

### 2) Snapshot metadata is propagated into train/eval datasets

Both graph dataset builders now store snapshot metadata in `Data` payload:
1. `stats_snapshot_version_idx` (stable integer mapping)
2. `stats_snapshot_as_of_epoch` (float epoch from ISO)

Implemented in:
1. `trainer._build_loader(...)`
2. `cli._build_graph_dataset(...)`

CLI `prepare_data(...)` now exports mapping:
1. `idx_to_stats_snapshot_version`
2. `stats_snapshot_version_to_idx`

This mapping is restored in `ModelTrainer.__init__` when `prepared_data` is used.

### 3) Forward-path logging added for train/inference/eval_drift

Added aggregation and `INFO` summary in trainer:
1. per train epoch (`stage=train`)
2. per validation epoch (`stage=validation`)
3. per test/inference pass (`stage=inference`)
4. per drift window evaluation (`stage=eval_drift`)

Log now includes:
1. `batches`, `graphs`
2. `struct_x_batches`, `struct_edge_batches`
3. `struct_feature_dims`, `struct_rows`
4. structural edge count stats (`min/avg/max`)
5. `snapshot_versions` (unique list compacted)
6. `snapshot_as_of_ts` (unique list compacted)

## Tests

Executed:
1. `.\.venv\Scripts\python.exe -m pytest tests\domain\test_dynamic_graph_builder_stats_matrix_mapping.py tests\application\test_trainer_forward_stats_logging.py -v`
2. `.\.venv\Scripts\python.exe -m pytest tests\domain\test_dynamic_graph_builder_stats_quality_gate.py tests\application\test_sync_stats_tool.py -v`
3. `.\.venv\Scripts\python.exe -m pytest -m mvp1_regression -v`

Result:
1. All passed.

## Current Outcome

1. Stats snapshot metadata is now propagated to forward path consistently.
2. Forward runtime logs explicitly show which snapshot versions/dates were used and how many stats tensors were passed.
3. MVP1 regression marker suite remains green.
