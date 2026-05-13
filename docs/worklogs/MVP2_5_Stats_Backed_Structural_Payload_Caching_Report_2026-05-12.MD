# Stats-Backed Structural Payload Caching Report

## Metadata

- `date`: 2026-05-12
- `status`: implemented-and-unit-verified
- `debt`: `stats_backed_structural_payload_caching`
- `plan`: `docs/worklogs/MVP2_5_Stats_Backed_Structural_Payload_Caching_Plan_2026-05-12.MD`
- `full_drift_validation`: pending user run on real `S_Str-USc-drift-loan`

---

## Summary

Implemented the two-stage fix for stats-backed structural drift graph building:

1. `snapshot-aware DTO/stats payload cache`
2. `structural payload deduplication`

The fix targets failures where full `eval_drift + structural_mode=true +
statistic_enabled=true` could overload Neo4j/driver connections or exhaust
RAM/virtual memory while building test graphs.

The model math was not changed. `ClassAwareStructuralScoring`,
`bilinear_with_prior`, structural auxiliary loss, and logits behavior are out of
scope for this change.

---

## Stage 1 - Snapshot-Aware Stats Payload Cache

### Changed

- `src/infrastructure/repositories/neo4j_knowledge_graph_repository.py`
  - Added lightweight snapshot timeline lookup.
  - Timeline queries now return identity fields only:
    - `knowledge_version`
    - `as_of_ts`
    - `kv_seq`
  - Heavy JSON fields are loaded separately by resolved snapshot identity.
  - Added bounded heavy payload cache keyed by:
    - `process_name`
    - `tenant_id`
    - `version_key`
    - `proc_def_id` or `__no_proc_def__`
    - `knowledge_version`
  - Added `cache_diagnostics()` counters:
    - `snapshot_timeline_loads`
    - `snapshot_timeline_cache_hits`
    - `snapshot_payload_loads`
    - `snapshot_payload_cache_hits`

- `src/domain/services/dynamic_graph_builder.py`
  - Added `cache_diagnostics()`.
  - Added topology cache hit/miss diagnostics.
  - Added resolved snapshot identity tracking.
  - Avoids retaining strict-asof DTOs in builder-level DTO cache by exact
    prefix timestamp; repository-level snapshot cache is now responsible for
    heavy stats payload reuse.

### Tests Added

- `tests/infrastructure/test_neo4j_knowledge_graph_repository.py`
  - `test_neo4j_snapshot_timeline_query_does_not_load_heavy_payloads`
  - `test_get_process_structure_asof_loads_payload_once_for_same_resolved_snapshot`

- `tests/domain/test_dynamic_graph_builder_cache.py`
  - `test_dynamic_builder_cache_diagnostics_tracks_unique_snapshot_identity`

---

## Stage 2 - Deduplicated Structural Payload Shards

### Changed

- `src/cli.py`
  - Added `GRAPH_DATASET_SHARD_FORMAT_DEDUP_STRUCTURAL`.
  - Added helpers:
    - `_structural_payload_key_from_data`
    - `_strip_structural_payload`
    - `_attach_structural_payload`
  - Sharded graph dataset cache now writes shard payloads as:

```python
{
    "schema": 2,
    "format": "dedup_structural_payloads",
    "graphs": [Data(...), ...],
    "structural_payloads": {
        "payload_key": {
            "struct_x": ...,
            "structural_edge_index": ...,
            "structural_edge_weight": ...,
            "struct_node_to_class_index": ...,
        }
    },
}
```

  - Per-prefix graphs store `structural_payload_key` only inside the shard file.
  - During rehydrate, structural tensors are restored and
    `structural_payload_key` is removed before the graph reaches PyG batching.
  - Added `GRAPH_DATASET_STRUCT_PAYLOADS` log line per split.
  - Cache metadata now includes structural payload counts per split.

- `src/application/use_cases/trainer.py`
  - `ShardedGraphDataset` now supports both:
    - legacy `list[Data]` shard files;
    - new `dedup_structural_payloads` shard files.

### Tests Added

- `tests/application/test_cli_graph_dataset_cache.py`
  - `test_sharded_cache_deduplicates_structural_payloads`
  - `test_iter_graphs_rehydrates_deduplicated_structural_payload`

- `tests/application/test_trainer_preloaded_data.py`
  - `test_sharded_graph_dataset_rehydrates_deduplicated_structural_payload`

---

## Documentation Updated

- `docs/current/project-state.md`
  - Added current runtime note for snapshot-aware payload caching and
    deduplicated structural shard payloads.

- `docs/GNN_RUNTIME_MVP2_5.MD`
  - Added `Stats-Backed Structural Payload Caching` section.

- `docs/current/architecture-debt.md`
  - Updated `stats_backed_structural_payload_caching` current behavior and
    linked this report.

- `docs/dissertation_changes.MD`
  - Added note that dissertation-grade structural drift experiments need
    runtime validity checks for stats-backed payload caching.

---

## Verification

### Stage 1

```powershell
.\.venv\Scripts\python.exe -m pytest tests/infrastructure/test_neo4j_knowledge_graph_repository.py tests/domain/test_dynamic_graph_builder_cache.py tests/integration/test_stats_forward_verification_pack.py -v
```

Result:

```text
15 passed
```

### Stage 2

```powershell
.\.venv\Scripts\python.exe -m pytest tests/application/test_cli_graph_dataset_cache.py tests/application/test_trainer_preloaded_data.py -v
```

Result:

```text
8 passed
```

```powershell
.\.venv\Scripts\python.exe -m pytest tests/integration/test_stats_forward_verification_pack.py tests/application/test_trainer_forward_stats_logging.py -v
```

Result:

```text
14 passed
```

### Focused Regression

```powershell
.\.venv\Scripts\python.exe -m pytest tests/infrastructure/test_neo4j_knowledge_graph_repository.py tests/domain/test_dynamic_graph_builder_cache.py tests/application/test_cli_graph_dataset_cache.py tests/application/test_trainer_preloaded_data.py tests/integration/test_stats_forward_verification_pack.py -v
```

Result:

```text
23 passed
```

### Architecture Guard

```powershell
.\.venv\Scripts\python.exe tools\architecture_guard.py
```

Result:

```text
[ARCH_GUARD] OK: architecture dependency rules satisfied.
```

---

## How To Validate On Real Drift Run

Run the failing preset again:

```text
S_Str-USc-drift-loan
```

Expected runtime signs:

- no Neo4j `defunct connection`;
- no system-wide low virtual memory event;
- test graph build completes;
- logs contain:

```text
GRAPH_DATASET_STRUCT_PAYLOADS split=test graphs=<many> shards=<n> structural_payloads=<bounded>
```

Healthy behavior: `structural_payloads` should be much lower than total graph
count and bounded by version/snapshot diversity inside shards.

---

## Remaining Risk

`stats_backed_structural_payload_caching` stays `active` in
`docs/current/architecture-debt.md` until a full real `S_Str-USc-drift-loan`
run confirms the system no longer hits Neo4j connection aborts or virtual-memory
exhaustion.

This implementation does not close `snapshot_homogeneous_batching`; it only
adds payload identities and deduplicated storage that are compatible with that
future batching work.
