# MVP2.5 Stats Forward Verification Pack Report

Date: 2026-03-21  
Status: Completed

## Scope

Implemented a synthetic verification pack to increase confidence that:
1. `stats_index` values are mapped to correct node rows and edge pairs.
2. Snapshot metadata is propagated into forward contract.
3. `strict_asof` uses prefix last-event timestamp for repository lookup.

## Added tests

1. `tests/integration/test_stats_forward_verification_pack.py`
   - `test_stats_values_are_mapped_to_expected_nodes_edges_and_forward_contract`
   - `test_strict_asof_uses_prefix_last_event_timestamp`

## What is verified now

1. End-to-end synthetic path:
   - `stats_index -> DynamicGraphBuilder(struct_x, structural_edge_weight) -> Trainer DataLoader -> _data_to_contract -> forward contract`.
2. Node mapping correctness:
   - expected values for `Start/Approve/End` land in matching `struct_x` rows.
3. Edge mapping correctness:
   - expected weights land in matching `(src,dst)` edges via `structural_edge_index`.
4. Snapshot metadata correctness:
   - `k000321` and expected `as_of` timestamp are present in forward contract batch metadata.
5. Strict temporal routing:
   - `get_process_structure_as_of(..., as_of_ts=<prefix_last_event_ts>)` is called with correct timestamp.

## Test runs

Executed:
1. `.\.venv\Scripts\python.exe -m pytest tests/integration/test_stats_forward_verification_pack.py tests/domain/test_dynamic_graph_builder_stats_matrix_mapping.py tests/application/test_trainer_forward_stats_logging.py -v`
   - Result: `10 passed`
2. `.\.venv\Scripts\python.exe -m pytest -m mvp1_regression -v`
   - Result: `17 passed`

Additional check:
1. `py -3 tools/architecture_guard.py`
   - Result: failed due pre-existing BOM (`U+FEFF`) in `src/adapters/ingestion/xes_adapter.py` (not introduced by this task).

## Conclusion

Synthetic verification confidence for stats-to-forward mapping is now significantly stronger.
Remaining gap before full research-grade confidence is real-data alignment gating and snapshot-homogeneous batching policy.
