# MVP2.5 Stage 3.2 Pre-Report

## Status
Completed: WP1-WP7 (Decoupling Ingestion from Training)

## Implemented Changes

### WP1: File-Backed Repository
1. Added `FileBasedKnowledgeGraphRepository`:
   - `src/infrastructure/repositories/file_based_knowledge_graph_repository.py`
2. JSON persistence with schema header:
   - `schema_version: "1.0"`
   - repository metadata (`created_at_utc`, backend, process/version)
3. Atomic writes implemented:
   - write to `*.tmp`
   - `os.replace(tmp, target)`
4. Supports full `IKnowledgeGraphPort` contract:
   - `save_process_structure`
   - `get_process_structure`
   - `list_versions`
   - `get_graph_for_visualization`

### WP2: Backend Factory + Config Switching
1. Added repository factory:
   - `src/infrastructure/repositories/knowledge_graph_repository_factory.py`
2. Added config parser for `mapping.knowledge_graph`:
   - `backend: in_memory | file`
   - `path`
   - `strict_load`
   - `ingest_split`
3. Added config blocks with comments:
   - `configs/data/mvp2_5_stage3_1_camunda_files.yaml`
   - `configs/data/mvp2_5_stage3_1_camunda_sql.yaml`

### WP3: Dedicated Ingestion Mode
1. Added new command tool:
   - `tools/ingest_topology.py`
2. Added command routing in `main.py`:
   - `python main.py ingest-topology --config ...`
3. Behavior:
   - reads traces via configured adapter (`xes` / `camunda`)
   - applies fraction and split policy (`train|full`)
   - extracts topology via `TopologyExtractorService`
   - persists JSON artifacts via selected repository
   - writes summary JSON

### WP4/WP5: Remove Sync Extraction from Training Path
1. Removed synchronous topology extraction from `prepare_data()` in `src/cli.py`.
2. Training now only consumes repository-backed topology.
3. Added strict mode guard:
   - if `knowledge_graph.strict_load=true` and no artifacts found -> fail fast with clear message.
4. Updated `DynamicGraphBuilder`:
   - accepts `process_name`
   - queries repository with `version + process_name`
   - keeps safe fallback to version-only lookup for legacy compatibility

### Additional Wiring
1. `DynamicGraphBuilder` updated:
   - `src/domain/services/dynamic_graph_builder.py`
2. `cli.py` now uses repository factory (`build_knowledge_graph_repository`).

## Tests Added

### Infrastructure
1. `tests/infrastructure/test_file_based_knowledge_graph_repository.py`
   - roundtrip save/load
   - schema header verification
   - atomic write tmp cleanup
   - ambiguity behavior for version-only lookup
   - visualization frequency filtering
   - corrupted JSON graceful handling
2. `tests/infrastructure/test_knowledge_graph_repository_factory.py`
   - backend selection and defaults normalization

### Application/Flow
1. `tests/application/test_ingest_topology_tool.py`
   - ingest command creates JSON artifact from tiny XES
   - `prepare_data` uses prebuilt file artifacts
   - sync extraction is not called during training prep

## Validation Runs

### New tests
```bash
.venv\Scripts\python -m pytest tests/infrastructure/test_file_based_knowledge_graph_repository.py tests/infrastructure/test_knowledge_graph_repository_factory.py tests/application/test_ingest_topology_tool.py -v
```
Result: `10 passed`

### MVP1 regression shield
```bash
.venv\Scripts\python -m pytest -m mvp1_regression -v
```
Result: `17 passed`

### Model + dynamic graph unit tests
```bash
.venv\Scripts\python -m pytest tests/domain/test_eopkg_models_forward.py tests/domain/test_dynamic_graph_builder_masks.py tests/domain/test_dynamic_graph_builder_tensor_purity.py -v
```
Result: `10 passed`

### Full suite
```bash
.venv\Scripts\python -m pytest tests/ -q
```
Result: `141 passed, 1 warning`

## Live Ingestion Check (XES)

Executed:
```bash
.venv\Scripts\python main.  py ingest-topology --config outputs/stage3_2_ingest_demo/tiny_ingest.yaml --out outputs/stage3_2_ingest_demo/ingest_summary.json
```

Observed:
1. Artifact created:
   - `outputs/stage3_2_ingest_demo/knowledge/demo_xes/demo_xes/process_structure.json`
2. Summary created:
   - `outputs/stage3_2_ingest_demo/ingest_summary.json`
3. Summary content confirms:
   - `status=ok`
   - `adapter=xes`
   - `versions_saved=["demo_xes"]`
   - `knowledge_backend=file`

## Notes for Stage 3.2 readiness
1. Decoupling complete: training no longer computes topology.
2. File-backed storage behaves as graph DB simulation layer.
3. Graceful degradation preserved (fallback in graph builder when topology absent).
4. Config-driven backend switch (`in_memory` vs `file`) is operational.
5. Ready to plug BPMN/Camunda structure adapters into same ingestion pipeline and persist using same repository contract.
