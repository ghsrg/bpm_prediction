# MVP2.5 Phase 4.2 Report - Neo4j Repository (APOC-free)

Date: 2026-03-18  
Status: Completed

## 1. What was done

1. Fixed plan with explicit **APOC-free policy**.
   - Updated: `docs/worklogs/MVP2_5_Stage4_2_Plan.MD`
   - Rule now states: no `apoc.*`; version arrays are updated via pure Cypher `CASE WHEN`.

2. Implemented new Neo4j backend repository:
   - Added: `src/infrastructure/repositories/neo4j_knowledge_graph_repository.py`
   - Implements `IKnowledgeGraphPort`:
     - `save_process_structure(...)`
     - `get_process_structure(...)`
     - `list_versions(...)`
     - `get_graph_for_visualization(...)`
   - Uses pure Cypher only (no APOC).
   - Supports BPMN-rich semantics:
     - node labels: `:ProcessNode` + typed labels (`:Task`, `:Gateway`, `:StartEvent`, `:EndEvent`, `:CallActivity`, ...)
     - relationship type mapping from `edge_type` (e.g. `SEQUENCE_FLOW`, `BOUNDARY_ATTACHMENT`, ...)
     - version binding with `ProcessVersion` and `versions` arrays.

3. Integrated backend into repository factory:
   - Updated: `src/infrastructure/repositories/knowledge_graph_repository_factory.py`
   - `mapping.knowledge_graph.backend` now supports: `in_memory`, `file`, `neo4j`.
   - Added Neo4j config resolution via profile/env (from connection resolver).
   - Added fail-fast validation when Neo4j credentials are unresolved.

4. Added/updated tests:
   - Added: `tests/infrastructure/test_neo4j_knowledge_graph_repository.py`
   - Updated: `tests/infrastructure/test_knowledge_graph_repository_factory.py`
   - Verifies:
     - APOC-free query generation,
     - typed label/edge mapping,
     - DTO roundtrip reconstruction from read rows,
     - ambiguous version safety,
     - factory support for Neo4j backend.

5. Added dependency for Stage 4 runtime:
   - Updated: `requirements.txt`
   - Added: `neo4j>=5.20.0`

6. Added new bulk synchronization mode:
   - Added: `tools/sync_topology.py`
   - Added command routing in `main.py`:
     - `python main.py sync-topology --config ... --out ...`
   - Supports:
     - Camunda BPMN sync (files/sql source via `CamundaBpmnAdapter`)
     - XES directory sync (iterates all `*.xes` files in folder)
   - Camunda full-sync behavior:
     - Reads full process-definition catalog (or filtered subset).
     - Pulls BPMN for each definition.
     - Saves structure into KB per `proc_def_key` namespace.
     - Missing BPMN payload is logged as warning and recorded in quarantine summary.

7. Added tests for sync mode:
   - Added: `tests/application/test_sync_topology_tool.py`
   - Verifies:
     - Camunda files full-sync over multiple process definitions.
     - XES directory sync over multiple files.

8. Added tenant-aware namespace strategy for bulk sync:
   - Namespace format: `process_key@tenant_id` (when tenant exists).
   - If tenant is missing, fallback namespace remains `process_key`.
   - Updated adapter parsing/filtering:
     - `src/adapters/ingestion/camunda_bpmn_adapter.py`
     - supports `tenant_id` extraction and `tenant_filters`.
   - Updated SQL templates to export tenant:
     - `src/adapters/ingestion/camunda/sql/structure/procdef_catalog.sql`
     - `src/adapters/ingestion/camunda/sql/runtime/procdef_catalog.sql`
     - `src/adapters/ingestion/camunda/sql/manual/process_definitions.sql`

9. Added sync-topology config matrix and updated existing Camunda data configs with tenant fields:
   - New data configs:
     - `configs/data/mvp2_5_stage4_2_sync_camunda_files_file.yaml`
     - `configs/data/mvp2_5_stage4_2_sync_camunda_files_neo4j.yaml`
     - `configs/data/mvp2_5_stage4_2_sync_camunda_sql_file.yaml`
     - `configs/data/mvp2_5_stage4_2_sync_camunda_sql_neo4j.yaml`
     - `configs/data/mvp2_5_stage4_2_sync_xes_dir_file.yaml`
     - `configs/data/mvp2_5_stage4_2_sync_xes_dir_neo4j.yaml`
   - New experiment wrappers:
     - `configs/experiments/mvp2_5_stage4_2_sync_camunda_files_file.yaml`
     - `configs/experiments/mvp2_5_stage4_2_sync_camunda_files_neo4j.yaml`
     - `configs/experiments/mvp2_5_stage4_2_sync_camunda_sql_file.yaml`
     - `configs/experiments/mvp2_5_stage4_2_sync_camunda_sql_neo4j.yaml`
     - `configs/experiments/mvp2_5_stage4_2_sync_xes_dir_file.yaml`
     - `configs/experiments/mvp2_5_stage4_2_sync_xes_dir_neo4j.yaml`
   - Existing Camunda data configs now include:
     - `camunda_adapter.tenant_id`
     - `camunda_adapter.tenant_filters`

10. Updated operator docs:
   - Updated: `README.MD`
   - Added `sync-topology` commands, config matrix, and tenant namespace notes.

## 2. APOC-free guarantee

Repository implementation contains **no `apoc.*` usage**.  
`versions` updates are implemented with:
- `CASE WHEN ... IS NULL THEN [...]`
- `WHEN $version_key IN ... THEN ...`
- `ELSE ... + $version_key`

## 3. Test execution logs

### Targeted Phase 4.2 tests
Command:
```powershell
.\.venv\Scripts\python.exe -m pytest tests/infrastructure/test_neo4j_knowledge_graph_repository.py tests/infrastructure/test_knowledge_graph_repository_factory.py -v
```
Result:
- `11 passed`

### MVP1 regression shield
Command:
```powershell
.\.venv\Scripts\python.exe -m pytest -m mvp1_regression -v
```
Result:
- `17 passed, 157 deselected`

### Full test suite
Command:
```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -v
```
Result:
- `177 passed`

## 4. Notes

1. Neo4j backend is now available through the same `IKnowledgeGraphPort` interface as file/in-memory backends.
2. `visualize_topology` remains compatible because it reads through the port abstraction.
3. This phase is ready for Phase 4.3 enrichment/statistics layering without schema rewrite.

## 5. Sync-topology usage

### Single config run
```powershell
python main.py sync-topology --config configs/experiments/mvp2_5_stage4_2_eopkg_files.yaml --out outputs/sync_summary.json
```

### Camunda behavior
1. Source is determined by config:
   - `mapping.camunda_adapter.structure.bpmn_source: files | mssql`
2. Catalog is read and iterated for all process definitions (or `process_filters` subset).
3. Each definition is stored under `process_name = proc_def_key` to avoid `v1/v2/...` collisions across different processes.
4. Missing BPMN file/bytes does not crash sync:
   - warning in logs
   - quarantine record in summary JSON
5. Tenant support:
   - optional in config (`tenant_id`, `tenant_filters`)
   - included in catalog parsing from `TENANT_ID_`
   - used in namespace composition for Camunda sync

### XES behavior
1. If `data.log_path` (or `data.log_dir`) is a directory, tool scans all `*.xes` files recursively.
2. Each file becomes separate dataset namespace (`dataset_name = file stem`).
3. Topology is extracted and saved per file namespace.
