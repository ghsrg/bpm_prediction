# MVP2.5 Stage 3.2 Detailed Plan (Revised)

Date: 2026-03-16  
Status: Planned (Revalidated)  
Scope: Camunda 7 BPMN Structure Ingestion (`bpmn_xml_content` or BPMN files) -> `ProcessStructureDTO`

## 0. Sources of Truth

1. `docs/worklogs/mvp2_5_Stage3_Plan.MD`
2. `docs/TARGET_ARCHITECTURE.MD`
3. `AGENT_GUIDE.MD`
4. `docs/DATA_MODEL_MVP2_5.MD`
5. `docs/LLD_MVP2_5.MD`
6. `docs/ADAPTER_CAMUNDA_SQL.MD`

---

## 1. Stage 3.2 Objective

Build a version-safe, failure-tolerant **static process structure** extractor for Camunda that:
1. uses BPMN as the source of structure (not runtime logs),
2. creates one `ProcessStructureDTO` per `proc_def_id`,
3. preserves Camunda extensions and binding metadata,
4. flattens embedded subprocesses recursively to activity-level graph,
5. keeps `callActivity` as supernode with explicit resolved/unresolved status,
6. stays resilient on dirty BPMN and partial catalogs.

---

## 2. Fixed Decisions (Locked)

1. **Camunda structure source is BPMN only.**
   - No structure building from Camunda runtime logs in Stage 3.2.
2. **Files mode must be supported** (manual exports), because BPMN is not exported as CSV payload.
3. `version_key` format is **only** `vNN` (example: `v22`).
   - Process name is stored separately and must not be concatenated into version key.
4. **One DTO == one `proc_def_id`** (single executable process definition version).
5. Embedded `subProcess` must be **recursively inlined** into main activity graph.
   - Target mode: `flattened-no-subprocess-node`.
6. `callActivity` out-of-scope child process:
   - never lazy-fetch during ingest,
   - keep unresolved supernode with full metadata,
   - set inference flags for future chained inference.
7. If structure artifact is missing during train/eval:
   - do not crash by default,
   - show explicit instruction,
   - support explicit fallback build from logs via `structure_from_logs` switch.

---

## 3. In Scope / Out of Scope

## 3.1 In Scope

1. BPMN ingestion from `mssql` and `files` adapters.
2. ProcDef catalog fetch/load and call binding resolution within loaded catalog.
3. XML sanitize/parse/extension extraction.
4. `ProcessStructureDTO` v2 expansion for structure + bindings + topology metadata.
5. Persistence through `IKnowledgeGraphPort` (file/in-memory backend abstraction already exists).

## 3.2 Out of Scope

1. Runtime IG extraction (Stage 3.1).
2. Periodic runtime stats enrichment (Stage 3.3).
3. Recursive/chained inference execution engine (future stage).
4. Neo4j implementation (future migration stage).

---

## 4. Input Contracts for Camunda BPMN Ingestion

## 4.1 SQL mode (`bpmn_source: mssql`)

Adapter reads:
1. Process definition catalog (`proc_def_id`, `proc_def_key`, `version`, `version_tag`, `deployment_id`, `resource_name`, executable flag).
2. BPMN bytes/content by exact `proc_def_id`.

## 4.2 Files mode (`bpmn_source: files`)

Expected local source under `data/camunda_exports/bpmn/`:
1. `process_definitions.csv` (catalog rows).
2. BPMN files in separate folder (for example `bpmn_xml/<proc_def_id>.bpmn` or `.xml`).

Rules:
1. BPMN content is loaded from file path referenced by catalog row.
2. Missing BPMN file -> warning + quarantine record (non-fatal for whole batch).

---

## 5. Target Output Contract (`ProcessStructureDTO` Stage 3.2)

## 5.1 Identity and scoping

1. `process_name` stored separately (repository dimension).
2. `version` field stores `version_key` in format `vNN`.
3. One artifact per exact `proc_def_id`.

## 5.2 DTO extensions (additive)

```python
class ProcessStructureDTO(BaseModel):
    version: str  # vNN
    proc_def_id: str
    proc_def_key: str
    deployment_id: str | None = None

    nodes: list[dict] = []
    edges: list[dict] = []

    # backward-compatible core topology
    allowed_edges: list[tuple[str, str]] = []
    edge_statistics: dict[tuple[str, str], dict[str, float]] | None = None
    node_metadata: dict[str, dict[str, str]] | None = None

    graph_topology: dict[str, Any] = {}

    # dedicated binding section (source of truth for callActivity state)
    call_bindings: dict[str, dict] = {}
```

## 5.3 `call_bindings` payload (mandatory for callActivity)

Keyed by `callActivity` node id:
1. `called_element`
2. `binding_type` (`latest`, `version`, `versionTag`, `deployment`)
3. `version_tag`
4. `version_number`
5. `resolved_child_proc_def_id` (`null` when unresolved)
6. `status` (`resolved` | `unresolved`)
7. `reason` (`child_process_out_of_current_filter`, `child_not_found_in_catalog`, etc.)
8. `requires_separate_inference: true`
9. `inference_trigger: "child_process_root"`
10. `child_process_key`
11. `child_version_tag`

---

## 6. Architecture Components

## 6.1 Ports

1. `ICamundaBpmnPort`
   - `fetch_procdef_catalog(...)`
   - `fetch_bpmn_xml(proc_def_id)` or `load_bpmn_file(proc_def_id)`
2. `IKnowledgeGraphPort`
   - `save_process_structure(process_name, version, dto)`
   - `get_process_structure(process_name, version)`
3. `IQuarantinePort` (or repository-backed diagnostics sink)
   - `save_quarantine_record(...)`

## 6.2 Services

1. `BpmnStructureParserService`
2. `CallBindingResolverService`
3. `BpmnQuarantineService`

---

## 7. Stage 3.2 Algorithm

## 7.1 Step A - Catalog snapshot

1. Load proc definitions in current filter scope.
2. Build indexes by:
   - `proc_def_id`
   - `(proc_def_key, version)`
   - `(proc_def_key, version_tag)`
   - `(proc_def_key, deployment_id)`

## 7.2 Step B - Load BPMN payload

For each `proc_def_id`:
1. Load BPMN from configured source (`mssql` or `files`).
2. If payload missing/corrupt: quarantine + continue with next definition.

## 7.3 Step C - Parse and sanitize

1. Decode safely (`utf-8`, `utf-8-sig`).
2. Strip BOM/garbage when recoverable.
3. Namespace-aware parse with recover mode when configured.
4. Keep non-fatal warnings for partial parsing.

## 7.4 Step D - Build activity-level structure

1. Extract flow nodes and sequence flows.
2. Parse Camunda `extensionElements` (properties, listeners, IO, fields, impl attrs).
3. Parse MI/loop static metadata.
4. Parse boundary attachment and interrupting semantics.

## 7.5 Step E - Embedded subprocess handling (locked behavior)

1. Recursively inline embedded subprocess content.
2. Remove subprocess container node from final activity-level graph.
3. Preserve scope metadata (`scope_level`, `parent_scope_id`) on inlined nodes.
4. Keep topology equivalent to execution-level activity graph design.

## 7.6 Step F - callActivity binding resolution

1. Resolve only against current catalog snapshot.
2. No lazy-fetch, no out-of-band query during ingest.
3. If unresolved, keep supernode and fill full unresolved metadata in `call_bindings`.

## 7.7 Step G - Finalization and persistence

1. Build `allowed_edges` and `graph_topology`.
2. Set `cycles_detected` and diagnostic counters.
3. Persist DTO for (`process_name`, `version=vNN`, `proc_def_id`).

---

## 8. Train/Eval Runtime Behavior (No Surprise Rules)

1. Trainer/DynamicGraphBuilder loads structure via `IKnowledgeGraphPort`.
2. If structure artifact is missing:
   - warning with exact next action,
   - continue in baseline fallback unless strict mode enabled.
3. Required user message:
   - run ingest with BPMN source, or
   - explicitly enable fallback generation from logs using `structure_from_logs=true`.

---

## 9. Configuration Additions

```yaml
mapping:
  knowledge_graph:
    backend: "file"            # in_memory | file
    path: "data/knowledge_graph"
    strict_load: false

  camunda_adapter:
    process_name: "B2BContracts_ApproveProject"

    structure:
      enabled: true
      source: "bpmn"           # bpmn | logs
      structure_from_logs: false # explicit fallback only

      bpmn_source: "files"     # files | mssql
      files:
        export_dir: "data/camunda_exports/bpmn"
        catalog_file: "process_definitions.csv"
        bpmn_dir: "bpmn_xml"

      version_key_format: "vNN"
      subprocess_mode: "flattened-no-subprocess-node"

      call_activity:
        lazy_fetch: false
        unresolved_policy: "keep_supernode"
```

---

## 10. Test Plan (Updated)

## 10.1 BPMN source modes

1. `test_bpmn_ingest_from_files_success`
2. `test_bpmn_ingest_from_mssql_success`
3. `test_files_and_mssql_same_procdef_same_topology`

## 10.2 DTO identity and versioning

1. `test_one_dto_per_proc_def_id`
2. `test_version_key_vnn_only`
3. `test_process_name_not_concatenated_into_version`

## 10.3 Subprocess and callActivity semantics

1. `test_embedded_subprocess_flattened_no_container`
2. `test_call_activity_resolved_binding_saved_in_call_bindings`
3. `test_call_activity_unresolved_out_of_filter_no_lazy_fetch`
4. `test_call_activity_unresolved_has_inference_flags`

## 10.4 XML resilience

1. `test_invalid_xml_quarantined`
2. `test_missing_bpmn_file_quarantined_non_fatal`
3. `test_unknown_extensions_warn_not_fail`

## 10.5 Runtime fallback behavior

1. `test_train_warns_when_structure_missing`
2. `test_structure_from_logs_flag_explicit_fallback_path`

## 10.6 Regression gates

1. `pytest -m mvp1_regression -v`
2. MVP2 dual-encoder domain tests
3. `pytest tests/ -v`

---

## 11. Acceptance Criteria (DoD)

1. Camunda Stage 3.2 structure is built from BPMN, not logs.
2. Files mode and MSSQL mode are both supported and deterministic.
3. DTO identity is strict: one DTO per `proc_def_id`.
4. `version_key` uses `vNN` only.
5. Embedded subprocesses are recursively flattened into activity-level graph.
6. `callActivity` unresolved policy is explicit and non-fatal, with inference flags saved.
7. Train/eval provides clear guidance when structure artifact is missing.
8. Regression shield remains green.

---

## 12. Open Clarification (single item)

Please confirm final files-mode naming defaults:
1. `process_definitions.csv`
2. `bpmn_xml/<proc_def_id>.bpmn`

If you prefer different names, they will be configurable anyway; this affects only default templates and docs examples.