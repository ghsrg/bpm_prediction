# DATA_MODEL_MVP2_5.MD

Updated: 2026-03-17  
Status: ACTIVE (Contract Freeze for Stage 4 readiness)

## 1. Purpose
Canonical DTO contracts for MVP2.5 ingestion, structure persistence, runtime assembly, and model consumption.

## 2. Core DTOs

### 2.1 EventRecord and RawTrace
`EventRecord` and `RawTrace` stay backward-compatible with MVP1.  
`RawTrace.process_version` is still string-based.

### 2.2 ProcessEventDTO (Camunda runtime, Stage 3.1)
`src/domain/entities/process_event.py` is unchanged in shape.  
Important field for cleanup diagnostics: `removal_time: datetime | None`.

### 2.3 ProcessStructureDTO (Frozen, Stage 3.3 hardening)
Source: `src/domain/entities/process_structure.py`

Required:
1. `version: str` (non-empty, `vNN` for Camunda structure artifacts)
2. `allowed_edges: list[tuple[str, str]]` (normalized, sorted, deduplicated)

Optional (normalized when provided):
1. `edge_statistics: dict[(src,dst), dict[str,float]]`
2. `node_metadata: dict[node_id, {activity_name, activity_type}]`
3. `proc_def_id`, `proc_def_key`, `deployment_id`
4. `nodes: list[dict]`
5. `edges: list[dict]`
6. `graph_topology: dict`
7. `call_bindings: dict[call_node_id, dict]`
8. `metadata: dict`

Deterministic semantics:
1. `nodes` sorted by `(scope_level, parent_scope_id, id)`
2. `edges` sorted by `(source, target, edge_type, id)`
3. `call_bindings` sorted by call node id
4. `graph_topology.allowed_edges` always mirrors normalized `allowed_edges`

Node minimal requirements (validator-enforced):
1. `id`
2. `bpmn_tag` (or derived from `activity_type`)
3. `type` (fallback to `bpmn_tag` if omitted in older artifacts)

Edge minimal requirements:
1. `source`
2. `target`
3. `edge_type` (default `sequence`)

### 2.4 call_bindings payload (frozen minimum)
Each call binding includes:
1. `called_element`
2. `binding_type`
3. `version_tag`
4. `version_number`
5. `resolved_child_proc_def_id`
6. `status` (`resolved` or `unresolved`)
7. `reason`
8. `requires_separate_inference`
9. `inference_trigger` (`child_process_root`)
10. `child_process_key`
11. `child_version_tag`
12. `inference_fallback_strategy` (`skip | use_aggregated_stats | raise`)

### 2.5 RuntimeFetchDiagnosticsDTO
Health/fallback diagnostics for runtime ingestion:
1. `history_coverage_percent`
2. `cleaned_instances_percent`
3. `fallback_triggered`
4. `fallback_reason`
5. `warnings`

## 3. Quarantine record contract (BPMN parser)
Hard parse failures produce deterministic quarantine records with:
1. `proc_def_id`
2. `proc_def_key`
3. `deployment_id`
4. `version`
5. `severity`
6. `error_code`
7. `error_message`
8. `source_hint`
9. `xml_snippet`
10. `created_at`

Standard error codes:
1. `missing_bpmn_payload`
2. `empty_bpmn_payload`
3. `xml_parse_error`
4. `no_process_element`
5. `missing_proc_def_id` (ingest catalog-level)

## 4. File repository artifact schema
Path:
`data/knowledge_graph/<process_name>/<version>/process_structure.json`

Envelope:
1. `schema_version`
2. `repository_backend`
3. `created_at_utc`
4. `process_name`
5. `version`
6. `dto`

Schema compatibility:
1. writer schema: `1.1`
2. supported reader schemas: `1.0`, `1.1`
3. unknown schema is warned but still attempted for forward compatibility

## 5. GraphTensorContract extension status
MVP2 optional fields remain backward compatible:
1. `structural_edge_index`
2. `structural_edge_weight`
3. `version_emb_idx`
4. `allowed_target_mask`
5. `struct_x`

If absent, baseline path must remain valid.
