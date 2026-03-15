# MVP2.5 Stage 3.2 Detailed Plan

Date: 2026-03-13  
Status: Planned  
Scope: BPMN Structure Ingestion from `bpmn_xml_content` (Camunda 7 -> ProcessStructureDTO)

## 0. Sources of Truth

1. `mvp2_5_Stage3_Plan.MD` (Stage split, ports, failure tolerance).
2. `docs/TARGET_ARCHITECTURE.MD` (Clean/Hexagonal boundaries, adapter-service-port flow).
3. `AGENT_GUIDE.MD` (contract-first, resilience, no speculative architecture breaks).
4. `docs/DATA_MODEL_MVP2.MD` (current DTO baseline and backward compatibility constraints).

---

## 1. Stage 3.2 Objective

Build a version-safe, failure-tolerant static process graph extractor for Camunda BPMN XML that:
1. reliably anchors structure to runtime via `activity_def_id` (BPMN node id),
2. preserves Camunda-specific metadata/extensions,
3. handles callActivity binding resolution with explicit unresolved states,
4. survives dirty/legacy XML and produces usable partial structure when possible.

---

## 2. In Scope / Out of Scope

## 2.1 In Scope

1. SQL fetch of process definitions + BPMN bytes.
2. XML sanitization, namespace-aware parsing, structural extraction.
3. `ProcessStructureDTO` expansion with nodes/edges/metadata/topology.
4. callActivity binding resolver (`latest`, `versionTag`, `deployment`, `version`).
5. quarantine diagnostics and partial-graph continuation policy.

## 2.2 Out of Scope

1. Runtime history ingestion (Stage 3.1).
2. Periodic runtime statistics refresh (Stage 3.3).
3. Neo4j persistence implementation.

---

## 3. Critical Requirements Mapping

| Required nuance | Planned handling |
|---|---|
| Version-safe callActivity binding | Save `calledElement`, binding mode, `versionTag`, `resolved_proc_def_id`; unresolved stays in DTO + diagnostics. |
| Camunda extensions | Extract `asyncBefore/After`, properties, io mappings, listeners, fields, implementation attrs, form/task attrs. |
| SubProcess/Event SubProcess/CallActivity differences | Embedded as nested scope, event subprocess as event-triggered scope, callActivity as reference placeholder only. |
| MI/loopCharacteristics | Store static fields (`is_multi_instance`, `is_sequential_mi`, `loop_cardinality_expr`, completion behavior). |
| Boundary events | Create `boundary_attachment` edges and interrupting/non-interrupting semantics. |
| SequenceFlow conditions | Store raw condition + parsed complexity + default-flow flag. |
| Dirty XML resilience | Sanitize bytes, namespace normalization, tolerant parser path, quarantine only for hard-fail XML. |
| Failure policy | invalid XML -> quarantine+skip definition; missing elements -> soft warning + partial graph. |
| DTO critical fields | nodes/edges/metadata/graph_topology fully populated with cycle flag and call_bindings state. |

---

## 4. Target Output Contract (ProcessStructureDTO v2 for Stage 3.2)

## 4.1 DTO shape (planned extension)

```python
class ProcessStructureDTO(BaseModel):
    version: str
    deployment_id: str | None = None
    process_ids: list[str] = []

    nodes: list[dict] = []
    # node fields:
    # id, name, type, bpmn_tag, camunda_type,
    # extensions, is_multi_instance, is_sequential_mi, loop_cardinality_expr,
    # attached_to, subprocess_scope, is_event_subprocess, is_executable

    edges: list[dict] = []
    # edge fields:
    # source, target, edge_type,
    # condition_expr, condition_complexity,
    # is_default, is_interrupting

    call_bindings: dict[str, dict] = {}
    # keyed by callActivity node id:
    # called_element, binding_type, version_tag, version_number,
    # deployment_binding, resolved_proc_def_id, status, reason

    graph_topology: dict[str, Any] = {}
    # graph_topology fields:
    # allowed_edges: list[tuple[str, str]]
    # cycles_detected: bool

    # backward-compatible mirrors (optional during transition)
    allowed_edges: list[tuple[str, str]] = []
    cycles_detected: bool = False

    metadata: dict[str, Any] = {}
    # parse warnings, unresolved counts, cache key, parser mode, etc.

    edge_statistics: Optional[dict[tuple[str, str], dict[str, float]]] = None
```

## 4.2 Backward compatibility

1. Existing fields `version`, `allowed_edges`, `edge_statistics` remain valid.
2. New fields are additive; MVP1/MVP2 code that only reads `allowed_edges` will continue working.

---

## 5. Architecture Components for Stage 3.2

## 5.1 Ports

1. `ICamundaBpmnPort`
   - `fetch_procdef_catalog(process_filters, ...)`
   - `fetch_bpmn_xml(proc_def_id)`
2. `IKnowledgeGraphPort`
   - `save_process_structure(process_name, version_key, dto)`
3. `IQuarantinePort` (new or internal repository)
   - `save_bpmn_quarantine(record)`

## 5.2 Services

1. `BpmnStructureParserService`
   - XML sanitize + parse + DTO build.
2. `CallBindingResolverService`
   - resolves call bindings against process definition catalog.
3. `BpmnQuarantineService`
   - writes hard/soft parse diagnostics.

---

## 6. Stage 3.2 Algorithm (Step-by-Step)

## 6.1 Step A - ProcDef Catalog Snapshot

1. Read all candidate process definitions for scope.
2. Build in-memory indexes:
   - by `proc_def_id`,
   - by `(process_key, version)`,
   - by `(process_key, version_tag)`,
   - by `(process_key, deployment_id)`.

This snapshot is the source of truth for callActivity resolution.

## 6.2 Step B - Byte Sanitize and XML Safety

1. Decode `bpmn_xml_content` with guarded strategy (`utf-8`, `utf-8-sig`, fallback logging).
2. Remove BOM and trailing garbage outside root where possible.
3. Validate root + namespace map.
4. If parse hard-fails -> quarantine record and skip this definition.

Quarantine record includes:
1. `proc_def_id`
2. `deployment_id`
3. `error_type`
4. `error_message`
5. `raw_xml_snippet` (safe truncated)
6. `timestamp`

## 6.3 Step C - Namespace-Normalized Parse

1. Parse with strict namespace-aware XPath first.
2. If namespace issues detected:
   - run tolerant fallback path,
   - continue with warning if recoverable.
3. `isExecutable="false"` is not fatal; mark process metadata and continue.

## 6.4 Step D - Node Extraction

Extract all `FlowNode`-relevant tags:
1. tasks (user/service/script/businessRule/send/receive/manual)
2. events (start/intermediate/end/boundary)
3. gateways (exclusive/inclusive/parallel/event-based)
4. subprocesses (embedded/event subprocess)
5. callActivity

Per node store:
1. `id`, `name`, `bpmn_tag`, `type` (`bpmn_tag + camunda:type if present`)
2. `subprocess_scope`
3. `attached_to` (`attachedToRef` for boundary)
4. `is_event_subprocess`
5. `is_executable` inherited from process
6. camunda implementation attrs (`class`, `delegateExpression`, `external`, `formKey`, `candidateGroups`, etc.)

If required attr missing (e.g., node without id):
1. soft warning,
2. skip node,
3. continue graph build.

## 6.5 Step E - ExtensionElements Extraction (Critical)

For each node parse `<extensionElements>` separately:
1. `camunda:properties` -> key-value map
2. `camunda:inputOutput` -> inputs/outputs map
3. listeners:
   - execution listeners
   - task listeners
4. `camunda:field` list (name/expression/string)
5. unknown namespaces/custom elements:
   - keep as raw dict/xml chunk when safe,
   - otherwise skip with warning (non-fatal)

No extension data should silently disappear.

## 6.6 Step F - Loop/Multi-Instance Static Metadata

Parse loop characteristics from BPMN tags:
1. `standardLoopCharacteristics`
2. `multiInstanceLoopCharacteristics`

Store static fields only:
1. `is_multi_instance`
2. `is_sequential_mi`
3. `loop_cardinality_expr`
4. behavior/completion expression hints (`all/any/none` when derivable)

Do not mix with runtime `loopCounter` (Stage 3.1/3.3 concern).

## 6.7 Step G - Edge Extraction

1. Parse all `sequenceFlow` edges.
2. Store per edge:
   - `source`, `target`, `edge_type="sequence"`
   - `condition_expr` raw text
   - `is_default`
3. For gateways (especially XOR/Inclusive) keep condition fidelity.

Condition complexity parser (lightweight heuristic):
1. variable count
2. function-like token presence
3. expression length bucket

## 6.8 Step H - Boundary Attachment Semantics

For each `boundaryEvent` with `attachedToRef`:
1. add `boundary_attachment` edge (`attachedToRef -> boundaryEventId`)
2. set `is_interrupting` from `cancelActivity` semantics
3. annotate potential path type:
   - interrupting -> `edge_type="cancellation"` semantics on outgoing boundary path
   - non-interrupting -> `edge_type="fork"` semantics on outgoing boundary path

## 6.9 Step I - Embedded vs Event SubProcess vs CallActivity

1. Embedded SubProcess:
   - keep parent-child hierarchy by `subprocess_scope`
   - include nested nodes in same process context.
2. Event SubProcess:
   - mark as event-triggered scope (may have no incoming sequence flow).
3. CallActivity:
   - do not inline child process graph in Stage 3.2,
   - create placeholder node with call binding metadata only.

## 6.10 Step J - Call Binding Resolution

For every callActivity resolve using catalog and binding policy:
1. `latest`: highest version by `calledElement` key
2. `versionTag`: exact `(calledElement, versionTag)`
3. `deployment`: same deployment scope when possible
4. `version`: exact numeric version

Persist binding payload:
1. `called_element`
2. `binding_type`
3. `version_tag`
4. `version_number`
5. `resolved_proc_def_id` or `null`
6. `status`: `resolved` / `unresolved`
7. `reason` for unresolved

Unresolved binding policy:
1. mark unresolved in DTO,
2. write diagnostics,
3. do not fail whole process parsing.

## 6.11 Step K - Topology Finalization

1. Compute `allowed_edges` from sequence + structural semantics.
2. Detect cycles (`cycles_detected`).
3. Populate `graph_topology = {allowed_edges, cycles_detected}`.
4. Build metadata summary:
   - node count, edge count, unresolved calls, warnings count,
   - parser mode (`strict`/`recovered`), cache hit/miss.
5. Save DTO through `IKnowledgeGraphPort`.

---

## 7. Failure Policy and Quarantine

## 7.1 Hard-fail -> quarantine + skip definition

1. unrecoverable XML parse failure (invalid root/corrupt bytes)
2. catastrophic parser exception

## 7.2 Soft-fail -> warning + partial graph

1. missing optional/partial elements
2. missing node ids on subset of elements
3. unknown extension namespace/elements
4. unresolved call binding
5. non-executable process

## 7.3 Quarantine record schema

1. `proc_def_id`, `process_key`, `deployment_id`, `version`
2. `severity` (`error`/`warning`)
3. `error_code`
4. `error_message`
5. `xml_snippet`
6. `created_at`

---

## 8. Performance and Caching Plan

Cache key:
1. `(proc_def_id, version, deployment_id, xml_checksum)`

Cache policy:
1. if key unchanged -> reuse parsed `ProcessStructureDTO`
2. if checksum changed -> reparse and replace cache entry
3. support large models (`>500` nodes) with cache-first strategy

Operational metrics:
1. parse_time_ms
2. cache_hit_ratio
3. large_model_count
4. unresolved_call_binding_count

---

## 9. Configuration Additions for Stage 3.2

```yaml
camunda:
  bpmn:
    parser: "lxml"
    strict_namespaces: true
    allow_recover_parse: true
    cache_enabled: true
    cache_max_entries: 1000
    quarantine_on_error: true
    keep_unknown_extensions_raw: true
    call_binding_strict: false    # false => unresolved allowed with diagnostics
```

---

## 10. File-Level Implementation Plan

## 10.1 New files (planned)

1. `src/domain/entities/bpmn_structure_diagnostics.py`
2. `src/domain/services/call_binding_resolver.py`
3. `src/application/services/bpmn_structure_parser_service.py`
4. `src/adapters/ingestion/camunda_bpmn_adapter.py`
5. `src/adapters/ingestion/camunda/sql/structure/procdef_catalog.sql`
6. `src/adapters/ingestion/camunda/sql/structure/bpmn_bytes.sql`
7. `src/infrastructure/repositories/in_memory_quarantine_repository.py` (or shared diagnostics store)

## 10.2 Existing files to update

1. `src/domain/entities/process_structure.py`
   - extend DTO fields listed in section 4.
2. `src/domain/ports/knowledge_graph_port.py`
   - ensure save/get API supports extended metadata fields.
3. `src/cli.py`
   - wire parser config block for Stage 3.2.
4. `docs/DATA_MODEL_MVP2.MD`
   - update ProcessStructureDTO contract.
5. `docs/DATA_FLOWS_MVP2.MD`
   - add Stage 3.2 parsing and quarantine flow.

---

## 11. Test Plan for Stage 3.2

## 11.1 Call binding and version safety

1. `test_call_binding_latest_resolves`
2. `test_call_binding_version_tag_resolves`
3. `test_call_binding_deployment_resolves`
4. `test_call_binding_numeric_version_resolves`
5. `test_call_binding_unresolved_marked_not_failed`

## 11.2 Extensions extraction

1. `test_extract_async_flags`
2. `test_extract_camunda_properties`
3. `test_extract_input_output_mappings`
4. `test_extract_task_and_execution_listeners`
5. `test_unknown_extensions_preserved_or_warned`

## 11.3 Structural semantics

1. `test_embedded_subprocess_scope_hierarchy`
2. `test_event_subprocess_marked_without_incoming`
3. `test_call_activity_placeholder_only`
4. `test_boundary_attachment_edge_and_interrupting_flag`
5. `test_multi_instance_static_fields`
6. `test_sequence_flow_condition_and_default_flag`
7. `test_cycles_detected_flag`

## 11.4 XML resilience

1. `test_invalid_xml_quarantined`
2. `test_namespace_mismatch_recovered_with_warning`
3. `test_bom_and_trailing_garbage_sanitization`
4. `test_non_executable_process_not_failed`
5. `test_large_model_cache_hit`

## 11.5 Non-regression

1. `pytest -m mvp1_regression -v`
2. `pytest tests/domain/test_eopkg_models_forward.py tests/domain/test_eopkg_gatv2_dual_encoder.py -v`
3. `pytest tests/ -v`

---

## 12. Acceptance Criteria (Stage 3.2 DoD)

1. Structure parser outputs valid `ProcessStructureDTO` with all required node/edge/metadata fields.
2. callActivity bindings are version-safe and unresolved bindings are explicitly represented (never silent loss).
3. Camunda extensions are extracted without silent drop; unknown extensions do not crash parser.
4. Embedded/event subprocess and boundary semantics are represented in graph topology.
5. Dirty XML handling works with quarantine and partial-graph continuation policies.
6. Parser handles large models with cache support and stable runtime.
7. Existing MVP1 regression suite remains green.

---

## 13. Recommended Default Strategy for First Rollout

1. Parser mode: `strict_namespaces=true`, `allow_recover_parse=true`.
2. call binding mode: permissive (`call_binding_strict=false`) with mandatory diagnostics.
3. unknown extensions: keep raw payload when possible.
4. cache enabled by default.
5. unresolved call bindings tracked as warnings, not fatal errors.
