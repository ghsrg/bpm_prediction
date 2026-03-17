# LLD_MVP2_5.MD

Updated: 2026-03-17  
Status: ACTIVE

## 1. Purpose
Low-level behavior of MVP2.5 decoupled ingestion, file-backed structure storage, and graceful degradation in training/eval.

## 2. FileBasedKnowledgeGraphRepository

### 2.1 Responsibilities
1. Persist `ProcessStructureDTO` to JSON artifact.
2. Read artifact by `(process_name, version)` with version-only fallback only if unambiguous.
3. Serve visualization graph with edge-frequency filtering.

### 2.2 Atomic write
```text
payload = serialize(dto)
write payload -> target.tmp
os.replace(target.tmp, target)
```

### 2.3 Schema policy
1. Current writer header: `schema_version=1.1`
2. Reader supports: `1.0`, `1.1`
3. Unknown schema version logs warning and still attempts parse

## 3. BPMN parser hardening (Stage 3.3)
Service: `BpmnStructureParserService`

Deterministic output guarantees:
1. nodes sorted by scope and id
2. edges sorted by source-target-edge_type-id with deduplication
3. call bindings deterministic by key
4. `graph_topology` enriched with:
   - `allowed_edges`
   - `cycles_detected`, `cycle_count`
   - `node_count`, `edge_count`
   - `gateway_node_ids`
   - `event_subprocess_node_ids`

Quarantine contract:
1. parser returns `dto=None` + structured quarantine record for hard failures
2. quarantine includes `source_hint` for fast operational triage
3. standardized error codes:
   - `missing_bpmn_payload`
   - `empty_bpmn_payload`
   - `xml_parse_error`
   - `no_process_element`
   - `missing_proc_def_id` (ingest pre-parse)

## 4. ingest-topology summary contract (Camunda BPMN path)
Mandatory summary fields:
1. `total_procdefs`
2. `parsed_procdefs`
3. `quarantined_procdefs`
4. `quarantine_error_codes`
5. `quarantine_records`
6. `versions_saved`

This summary is deterministic and machine-readable for CI smoke checks.

## 5. DynamicGraphBuilder graceful degradation
1. Resolve structure version from prefix and version variants (`22` <-> `v22`).
2. Read via `IKnowledgeGraphPort`.
3. If structure exists, build `allowed_target_mask`.
4. If missing, keep structural tensors/mask as `None`; baseline path continues.
5. Strict mode (`mapping.knowledge_graph.strict_load=true`) fails fast when structure is expected but unavailable.

## 6. Failure matrix
| Component | Failure | Behavior |
|---|---|---|
| File repo write | disk IO error | explicit exception |
| File repo read | missing file | returns `None` |
| File repo read | corrupted JSON | returns `None` |
| BPMN parser | malformed XML | quarantine + continue batch |
| BPMN parser | node without id | warning + partial graph |
| Trainer graph build | structure missing | fallback unless strict mode |

## 7. Test coverage focus
1. DTO contract freeze validators.
2. Dirty BPMN resilience and quarantine correctness.
3. ingest summary stability for Camunda BPMN mode.
4. repository schema header/roundtrip compatibility.
5. full non-regression (`mvp1_regression` + full `tests/`).
