# MVP2.5 Stage4.2 - Option A Unbatch Fix (EOPKG struct_x mismatch)

Date: 2026-03-20  
Status: Completed

## Problem

EOPKG runtime warning:
1. `struct_x node count mismatch (got=B*C expected=C)`

Cause:
1. `PyG DataLoader` batched `struct_x` by concatenation across samples.
2. Structural tensors were passed to model from batched payload directly.

## Implemented Option A

Updated `src/application/use_cases/trainer.py`:
1. Added structural payload selector for forward:
   - `batch.to_data_list()[0]` is used as source of:
     - `struct_x`
     - `structural_edge_index`
     - `structural_edge_weight`
2. Added fallback reshape for `struct_x` when only batched tensor is available:
   - detect `[B*C, F]`
   - reshape to `[B, C, F]`
   - take `[0, :, :]`
3. Added runtime mixed-snapshot guard:
   - if batch contains more than one `stats_snapshot_version_idx` -> `UserWarning`
   - execution continues with first graph structural payload.

## Tests

Updated:
1. `tests/application/test_trainer_forward_stats_logging.py`

Added checks:
1. first-graph structural payload is used in `_data_to_contract`
2. mixed snapshot batch emits `UserWarning`

Validation commands:
1. `pytest tests/application/test_trainer_forward_stats_logging.py tests/domain/test_dynamic_graph_builder_tensor_purity.py tests/domain/test_dynamic_graph_builder_stats_matrix_mapping.py -v`
2. `pytest -m mvp1_regression -v`

All passed.
