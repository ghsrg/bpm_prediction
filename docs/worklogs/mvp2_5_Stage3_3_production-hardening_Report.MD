# MVP2.5 Stage 3.3 Production-Hardening Report

Date: 2026-03-17  
Status: Done  
Verdict: Ready for Stage 4 planning (Neo4j integration)

## Scope completed (WP1-WP5)

## WP1 & WP3: Contract freeze + deterministic semantics
Implemented:
1. Hardened `ProcessStructureDTO` validators in `src/domain/entities/process_structure.py`:
   - normalized and deduplicated `allowed_edges`,
   - validated and normalized `nodes`, `edges`, `call_bindings`,
   - deterministic ordering guarantees,
   - stable `graph_topology.allowed_edges` synchronization.
2. Hardened `BpmnStructureParserService` in `src/application/services/bpmn_structure_parser_service.py`:
   - deterministic node and edge ordering,
   - edge deduplication,
   - richer `graph_topology` for Neo4j-friendly mapping:
     - `node_count`, `edge_count`,
     - `gateway_node_ids`,
     - `event_subprocess_node_ids`,
     - cycle flags/counters.

## WP2 & WP5: Quarantine contract + dirty BPMN tests
Implemented:
1. Standardized parser quarantine payload (`source_hint` added, stable error-code set).
2. Extended Camunda BPMN ingest summary in `tools/ingest_topology.py`:
   - `quarantine_error_codes`,
   - stable quarantine records accumulation.
3. Added dirty BPMN fixtures:
   - `tests/fixtures/bpmn_dirty/invalid_xml.bpmn`
   - `tests/fixtures/bpmn_dirty/missing_ids.bpmn`
   - `tests/fixtures/bpmn_dirty/unknown_extensions.bpmn`
4. Added/updated tests:
   - `tests/application/test_bpmn_structure_parser_service.py` (dirty BPMN resilience + quarantine assertions),
   - `tests/application/test_ingest_topology_camunda_bpmn.py` (quarantine summary contract),
   - `tests/contracts/test_process_structure_contract_freeze.py`.

## WP4: SQL consistency pass
Implemented:
1. Added SQL consistency tests:
   - `tests/adapters/test_camunda_sql_templates_consistency.py`
2. Re-validated runtime SQL contract for Stage 3.2 parser:
   - `procdef_catalog.sql` aliases,
   - `bpmn_bytes.sql` uses `BA.BYTES_ AS bpmn_xml_content`.
3. Synced docs with contract expectations in `docs/ADAPTER_CAMUNDA_SQL.MD`.

## Repository and schema hardening
Implemented in `src/infrastructure/repositories/file_based_knowledge_graph_repository.py`:
1. Schema header bumped to `1.1`.
2. Reader compatibility set: `1.0`, `1.1`.
3. Header validation with warning on unknown schema.
4. Deterministic serialization of `nodes`, `edges`, `call_bindings`.

## Documentation updates
Updated:
1. `docs/DATA_MODEL_MVP2_5.MD` (frozen DTO + quarantine + schema compatibility matrix),
2. `docs/LLD_MVP2_5.MD` (low-level hardening behavior and failure matrix),
3. `docs/DATA_FLOWS_MVP2_5.MD` (explicit quarantine branch in ingest flow),
4. `docs/ADAPTER_CAMUNDA_SQL.MD` (SQL alias/column consistency requirements).

## Test execution

### 1) MVP1 regression shield
Command:
```bash
.venv\Scripts\python.exe -m pytest -m mvp1_regression -v
```
Result:
1. 17 passed
2. 0 failed

### 2) Full test suite
Command:
```bash
.venv\Scripts\python.exe -m pytest tests/ -v
```

Result:
1. 163 passed
2. 0 failed

## Notes
1. Existing MVP1/MVP2 behavior remains green.
2. Contract and diagnostics are now deterministic and ready for Neo4j mapping phase.
3. Quarantine behavior is resilient: parser never hard-crashes ingestion batch on dirty BPMN.
