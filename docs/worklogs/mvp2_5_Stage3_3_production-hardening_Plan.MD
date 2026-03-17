# MVP2.5 Stage 3.3 Production-Hardening Plan (Pre-Stage4)

Date: 2026-03-17  
Status: Planned  
Goal: stabilise BPMN ingestion + structure contracts to production level before Neo4j backend switch (Stage 4).

## 0. Scope and Intent

This plan is not a new feature wave.  
It is a hardening wave to remove operational uncertainty before Stage 4.

In-scope:
1. parser robustness,
2. strict contracts and diagnostics,
3. deterministic behavior under dirty BPMN data,
4. regression safety for MVP1 and MVP2.5.

Out-of-scope:
1. Neo4j implementation itself (Stage 4),
2. BPMS enrichment adapter,
3. new ML architecture changes.

## 1. Entry Criteria

Current baseline already exists:
1. offline `ingest-topology`,
2. file-backed knowledge repository,
3. Camunda BPMN path (`source=bpmn`) with fallback path (`structure_from_logs`),
4. working train/eval consumption via `IKnowledgeGraphPort`.

## 2. Target Outcomes (Hardening DoD)

After hardening, we must have:
1. stable `ProcessStructureDTO` contract for Stage 4 migration,
2. explicit machine-readable diagnostics/quarantine outputs,
3. deterministic parser behavior on malformed BPMN variants,
4. no hidden fallback surprises in topology-aware runs,
5. green regression shield and targeted Stage 3.2/3.3 tests.

## 3. Work Package Breakdown

## WP1. Contract Freeze and Validation Guards

Files:
1. `src/domain/entities/process_structure.py`
2. `src/application/services/bpmn_structure_parser_service.py`
3. `docs/DATA_MODEL_MVP2_5.MD`

Actions:
1. Freeze minimum required fields in DTO for production read/write.
2. Add strict runtime validation for required structure fields per proc_def/version.
3. Ensure unknown optional fields are tolerated (forward compatibility).
4. Add `schema_version` compatibility matrix in docs.

Tests:
1. valid DTO roundtrip remains stable,
2. missing required fields fail with explicit error,
3. unknown fields do not break deserialization.

## WP2. Quarantine and Diagnostics Contract

Files:
1. `tools/ingest_topology.py`
2. `src/application/services/bpmn_structure_parser_service.py`
3. `docs/LLD_MVP2_5.MD`

Actions:
1. Normalize quarantine record schema:
   - `proc_def_id`, `proc_def_key`, `version`, `error_code`, `error_message`, `source_hint`.
2. Persist deterministic summary diagnostics in ingest output:
   - `total_procdefs`, `parsed_procdefs`, `quarantined_procdefs`, `versions_saved`.
3. Standardize error codes (enum-like set) for parser failures.

Tests:
1. invalid XML goes to quarantine with stable error code,
2. unresolved callActivity binding is not quarantined (warning path),
3. ingest summary schema is stable and complete.

## WP3. BPMN Edge-Case Semantics Hardening

Files:
1. `src/application/services/bpmn_structure_parser_service.py`
2. `docs/ADAPTER_CAMUNDA_SQL.MD`
3. `docs/DATA_FLOWS_MVP2_5.MD`

Actions:
1. Verify and harden handling for:
   - boundary interrupting/non-interrupting,
   - event subprocess entry semantics,
   - embedded subprocess flattening invariants,
   - unresolved callActivity metadata completeness.
2. Enforce deterministic edge typing for downstream graph use.
3. Ensure no accidental runtime lazy-fetch behavior appears in parser path.

Tests:
1. boundary events produce expected edge typing,
2. event subprocess is represented without illegal incoming/outgoing contamination,
3. callActivity unresolved contract always contains fallback strategy and inference flags.

## WP4. Camunda SQL and Manual Export Consistency

Files:
1. `src/adapters/ingestion/camunda/sql/manual/*.sql`
2. `src/adapters/ingestion/camunda/sql/runtime/*.sql`
3. `src/adapters/ingestion/camunda/sql/manual/README.MD`
4. `docs/ADAPTER_CAMUNDA_SQL.MD`

Actions:
1. Final consistency pass for field names:
   - `BYTES_` usage,
   - `SUPER_PROCESS_INSTANCE_ID_` naming,
   - fully-qualified table prefixes.
2. Align manual export instructions with actual expected directories and filenames.
3. Add one canonical “minimal export set” profile for operators.

Tests:
1. file-mode adapter reads canonical exports without renaming hacks,
2. runtime SQL template smoke check for required columns.

## WP5. Regression and Non-Regression Shield Expansion

Files:
1. `tests/application/test_bpmn_structure_parser_service.py`
2. `tests/application/test_ingest_topology_camunda_bpmn.py`
3. `tests/infrastructure/test_file_based_knowledge_graph_repository.py`
4. `tests/infrastructure/test_knowledge_graph_repository_factory.py`
5. new tests for dirty BPMN corpus

Actions:
1. Add dirty BPMN fixtures:
   - malformed namespaces,
   - broken XML tail,
   - missing IDs,
   - unresolved call bindings.
2. Add deterministic assertions for diagnostics/quarantine.
3. Add performance smoke check (small upper bound on parser runtime for fixture set).

Mandatory test runs:
1. `pytest -m mvp1_regression -v`
2. targeted Stage 3.2/3.3 suite,
3. full `pytest tests/ -v` before Stage 4 branch.

## 4. Operational Hardening Checklist

Before Stage 4 starts:
1. `ingest-topology` produces stable summary JSON for both XES and Camunda BPMN.
2. `visualize-topology` works from repository artifacts, not raw source, by default.
3. Topology-aware configs fail-fast on invalid/missing backend settings.
4. One real Camunda export package is replayed end-to-end without manual code edits.
5. Documentation and SQL templates are synchronized.

## 5. Risks and Mitigation

1. Risk: Hidden parser behavior drift with new BPMN variants.
   - Mitigation: curated dirty BPMN regression fixtures + error code contract.
2. Risk: Quarantine format instability.
   - Mitigation: strict schema tests on ingest summary and quarantine records.
3. Risk: Stage 4 migration blocked by contract ambiguity.
   - Mitigation: DTO freeze + compatibility notes in docs.
4. Risk: Performance regressions in ingestion.
   - Mitigation: parser smoke time bounds on fixture corpus.

## 6. Deliverables

1. Hardened parser and ingestion behavior.
2. Extended test suite for dirty BPMN and diagnostics.
3. Updated docs:
   - `DATA_MODEL_MVP2_5.MD`
   - `LLD_MVP2_5.MD`
   - `ADAPTER_CAMUNDA_SQL.MD`
   - `DATA_FLOWS_MVP2_5.MD` (if flow details change)
4. Stage 3.3 hardening report with executed test logs and migration readiness verdict.

## 7. Stage 4 Readiness Gate

Proceed to Stage 4 only when all are true:
1. zero critical parser contract gaps,
2. zero unresolved documentation contradictions,
3. regression shield green,
4. ingestion diagnostics are deterministic and actionable,
5. file-backed repository schema is stable and migration-mappable to Neo4j nodes/edges.
