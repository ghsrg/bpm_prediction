# MVP2.5 Stage 3.4 Report

Date: 2026-03-19  
Status: Completed and aligned with finalized storage policy

## 1. Finalized decisions implemented

1. Immutable snapshots: enabled.
2. No `:NodeStat` / `:EdgeStat` normalization: enabled. Stats are stored as JSON payloads.
3. `as_of` lookup: enabled in repositories and builder path.
4. No TTL: enforced (snapshots are retained).
5. Strict composite key: enabled for snapshot identity.
6. Process-level stats stored in same snapshot node as version-level stats: enabled.

## 2. Neo4j snapshot model (implemented)

`Neo4jKnowledgeGraphRepository` now supports full snapshot API:

1. `save_process_structure_snapshot(...)`
2. `get_process_structure_as_of(...)`

Snapshot identity uses strict key:
1. `tenant_id`
2. `process_name`
3. `version_key`
4. `proc_def_id`
5. `knowledge_version`

Payload is JSON-only (no stat node explosion):
1. `node_stats_json`
2. `edge_stats_json`
3. `gnn_features_json`
4. `stats_diagnostics_json`
5. `metadata_json`

Version linkage:
1. `(:ProcessVersion)-[:HAS_STATS_SNAPSHOT]->(:StatsSnapshot)`
2. `as_of` lookup returns nearest snapshot `<= as_of_ts`
3. If no snapshot exists, repository returns base structure DTO (graceful fallback)

## 3. Scope of Stage 3.4 implementation

1. Added command `sync-stats`:
   - `python main.py sync-stats --config ... --out ... [--as-of ...]`
   - file: `tools/sync_stats.py`
2. Tier A Camunda stats aggregation:
   - windows: `last_7d`, `last_30d`, `last_90d`, `all_time`
   - node metrics: `exec_count`, `duration_median`, `duration_p95`, `loop_density`, `parallel_token_variance`, `freshness_score`, `confidence_score`
   - edge metrics: `transition_probability`, `latency_median`, `latency_p95`
   - global metric: `resource_handover_entropy`
3. Added stats snapshot/as-of contract to:
   - `src/domain/ports/knowledge_graph_port.py`
   - file/in-memory/neo4j repositories
4. `DynamicGraphBuilder` updated to read mapped graph stats and build `struct_x` via `mapping.graph_feature_mapping`

## 4. Files updated (core)

1. `main.py`
2. `tools/sync_stats.py`
3. `src/domain/entities/process_structure.py`
4. `src/domain/ports/knowledge_graph_port.py`
5. `src/domain/services/dynamic_graph_builder.py`
6. `src/infrastructure/repositories/file_based_knowledge_graph_repository.py`
7. `src/infrastructure/repositories/in_memory_networkx_repository.py`
8. `src/infrastructure/repositories/neo4j_knowledge_graph_repository.py`
9. `src/cli.py`
10. `README.MD`

## 5. Tests

Added/updated:
1. `tests/application/test_sync_stats_tool.py`
2. `tests/infrastructure/test_file_based_knowledge_graph_repository.py::test_file_repo_snapshot_roundtrip_and_asof_lookup`
3. `tests/infrastructure/test_neo4j_knowledge_graph_repository.py::test_save_process_structure_snapshot_uses_composite_key_and_json_payload`
4. `tests/infrastructure/test_neo4j_knowledge_graph_repository.py::test_get_process_structure_asof_overlays_stats_snapshot_json`

Execution:
1. `./.venv/Scripts/python.exe -m pytest tests/ -v`  
   Result: `181 passed, 1 warning`
2. `./.venv/Scripts/python.exe -m pytest -m mvp1_regression -v`  
   Result: `17 passed, 164 deselected, 1 warning`

## 6. Run sequence

1. Sync topology:
```bash
python main.py sync-topology --config <topology_config.yaml> --out outputs/sync_topology.json
```
2. Sync stats snapshots:
```bash
python main.py sync-stats --config configs/experiments/mvp2_5_stage3_4_sync_stats_files.yaml --out outputs/sync_stats_files.json
```
3. Historical `as_of` snapshot:
```bash
python main.py sync-stats --config configs/experiments/mvp2_5_stage3_4_sync_stats_files.yaml --as-of 2026-03-12T00:00:00Z --out outputs/sync_stats_asof.json
```
4. Train with mapped graph stats:
```bash
python main.py --config configs/experiments/mvp2_5_stage3_4_eopkg_files.yaml
```
