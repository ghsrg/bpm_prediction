# MVP2.5 Stage4.2 - Snapshot Metadata Fallback Fix

Date: 2026-03-20  
Status: Completed

## Problem

Runtime logs showed:
1. `snapshot_versions=none`
2. `snapshot_as_of_ts=none`

even when structural stats tensors were present (`struct_x_batches > 0`).

## Root cause

`DynamicGraphBuilder` extracted snapshot metadata only from top-level:
1. `metadata.knowledge_version`
2. `metadata.as_of_ts`

For some payloads, snapshot identity was present in:
1. `metadata.stats_contract.identity.knowledge_version`
2. `metadata.stats_contract.identity.as_of_ts`

## Fix

Updated `src/domain/services/dynamic_graph_builder.py`:
1. Added fallback to `metadata.stats_contract.identity` when top-level snapshot fields are missing.
2. Kept contract purity by exposing scalar fields:
   - `stats_snapshot_version_seq`
   - `stats_snapshot_as_of_epoch`

## Tests

Updated:
1. `tests/domain/test_dynamic_graph_builder_stats_matrix_mapping.py`

Added coverage:
1. `test_builder_reads_snapshot_metadata_from_stats_contract_identity`

Executed:
1. `pytest tests/domain/test_dynamic_graph_builder_stats_matrix_mapping.py tests/application/test_trainer_forward_stats_logging.py -v`
2. `pytest tests/domain/test_dynamic_graph_builder_tensor_purity.py -v`

All passed.
