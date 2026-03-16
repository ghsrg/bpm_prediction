# MVP2.5 Stage 3.2 Report

Date: 2026-03-16
Status: Completed
Scope: Camunda BPMN Structure Ingestion (`bpmn_xml_content` / BPMN files) -> `ProcessStructureDTO`

## 1. What Was Implemented

1. Implemented Stage 3.2 BPMN structure path for Camunda in `ingest-topology`.
2. Added dedicated Camunda BPMN adapter with dual source support:
   - `bpmn_source: files`
   - `bpmn_source: mssql`
3. Added BPMN parser service that builds structural DTO from BPMN XML.
4. Added strict Stage 3.2 call binding contract with unresolved-safe behavior and **no lazy fetch**.
5. Added required `inference_fallback_strategy` in `call_bindings`.
6. Extended DTO/repository persistence to store new structural sections.
7. Preserved backward compatibility for MVP1/MVP2 runtime training flow.

## 2. Key Architecture Decisions Implemented

1. Camunda structure source for Stage 3.2 is BPMN (not runtime logs).
2. One `ProcessStructureDTO` is created per `proc_def_id`.
3. `version_key` normalization is enforced as `vNN`.
4. Embedded `subProcess` is recursively flattened (`flattened-no-subprocess-node`).
5. `callActivity` is represented as supernode with explicit binding state.
6. If child process is out of filter/not found:
   - stays unresolved,
   - no lazy-fetch,
   - inference flags are persisted.
7. Added `inference_fallback_strategy` in call binding payload:
   - supported values: `skip | use_aggregated_stats | raise`
   - default: `use_aggregated_stats`.

## 3. Files Added

1. `src/domain/ports/camunda_bpmn_port.py`
2. `src/adapters/ingestion/camunda_bpmn_adapter.py`
3. `src/application/services/bpmn_structure_parser_service.py`
4. `src/adapters/ingestion/camunda/sql/structure/procdef_catalog.sql`
5. `src/adapters/ingestion/camunda/sql/structure/bpmn_bytes.sql`
6. `src/adapters/ingestion/camunda/sql/manual/process_definitions.sql`
7. `src/adapters/ingestion/camunda/sql/runtime/procdef_catalog.sql`
8. `src/adapters/ingestion/camunda/sql/runtime/bpmn_bytes.sql`
9. `tests/application/test_bpmn_structure_parser_service.py`
10. `tests/application/test_ingest_topology_camunda_bpmn.py`

## 4. Files Updated

1. `tools/ingest_topology.py`
   - Added routing: Camunda + `structure.source=bpmn` -> BPMN ingestion path.
   - Added summary fields for BPMN ingestion (`structure_source`, procdef stats, quarantine records).
2. `src/domain/entities/process_structure.py`
   - Extended DTO fields for Stage 3.2 (`proc_def_id`, `proc_def_key`, `deployment_id`, `nodes`, `edges`, `graph_topology`, `call_bindings`, `metadata`).
3. `src/infrastructure/repositories/file_based_knowledge_graph_repository.py`
   - Added serialization/deserialization for Stage 3.2 fields.
4. `src/domain/services/dynamic_graph_builder.py`
   - Added numeric/v-prefixed version fallback (`22` <-> `v22`) for robust lookup.
5. `src/cli.py`
   - Added explicit warning when topology artifact is missing, including `ingest-topology` and `structure_from_logs` fallback hint.
6. `configs/data/mvp2_5_stage3_1_camunda_files.yaml`
7. `configs/data/mvp2_5_stage3_1_camunda_sql.yaml`
   - Added `camunda_adapter.structure` block with defaults and comments.
8. `tests/infrastructure/test_file_based_knowledge_graph_repository.py`
9. `tests/domain/test_dynamic_graph_builder_version_fallback.py`
10. `tests/contracts/test_mvp2_dto_contract.py`
11. `src/adapters/ingestion/camunda/sql/manual/README.MD`

## 5. Stage 3.2 Call Binding Payload (Implemented)

`ProcessStructureDTO.call_bindings[call_activity_node_id]` now includes:

1. `called_element`
2. `binding_type`
3. `version_tag`
4. `version_number`
5. `resolved_child_proc_def_id`
6. `status` (`resolved` / `unresolved`)
7. `reason`
8. `requires_separate_inference` (always `true`)
9. `inference_trigger` (`child_process_root`)
10. `child_process_key`
11. `child_version_tag`
12. `inference_fallback_strategy` (`skip | use_aggregated_stats | raise`)

## 6. How To Run

## 6.1 Camunda BPMN ingestion from files

```bash
python main.py ingest-topology --config <experiment.yaml> --out outputs/<name>.json
```

Config requirements (`mapping.camunda_adapter.structure`):

1. `source: bpmn`
2. `structure_from_logs: false`
3. `bpmn_source: files`
4. `files.export_dir`
5. `files.catalog_file`
6. `files.bpmn_dir`

## 6.2 Camunda BPMN ingestion from MSSQL

```bash
python main.py ingest-topology --config <experiment.yaml> --out outputs/<name>.json
```

Config requirements:

1. `source: bpmn`
2. `bpmn_source: mssql`
3. `sql_dir: src/adapters/ingestion/camunda/sql/structure` (or `src/adapters/ingestion/camunda/sql/runtime`)
4. valid MSSQL connection string in structure/runtime mssql block

## 6.3 Explicit fallback ingestion from logs (legacy)

Set:

1. `mapping.camunda_adapter.structure.source: bpmn`
2. `mapping.camunda_adapter.structure.structure_from_logs: true`

Then run same `ingest-topology` command.

## 7. Test Results

## 7.1 Targeted Stage 3.2 tests

Command:

```bash
.venv\Scripts\python -m pytest tests/application/test_bpmn_structure_parser_service.py tests/application/test_ingest_topology_camunda_bpmn.py tests/infrastructure/test_file_based_knowledge_graph_repository.py tests/domain/test_dynamic_graph_builder_version_fallback.py tests/contracts/test_mvp2_dto_contract.py -v
```

Result: passed after one parser flattening fix.

## 7.2 Full test suite

Command:

```bash
.venv\Scripts\python -m pytest tests/ -v
```

Result:

1. `150 passed`
2. `1 warning`

## 7.3 MVP1 Regression Shield

Command:

```bash
.venv\Scripts\python -m pytest -m mvp1_regression -v
```

Result:

1. `17 passed`
2. `133 deselected`
3. `1 warning`

## 8. Notes

1. Existing `TopologyExtractorService.extract_from_bpmn()` remained untouched (still explicit `NotImplemented`) to avoid changing existing service semantics; Stage 3.2 uses the new dedicated BPMN parser/adapter path in `ingest-topology`.
2. Training/eval behavior stays backward compatible: missing structure does not hard-fail unless `knowledge_graph.strict_load=true`.
