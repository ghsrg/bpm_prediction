# DATA_MODEL_MVP2_5.MD

Updated: 2026-03-17  
Status: ACTIVE

## 1. Purpose
Canonical DTO contracts for MVP2.5 ingestion, structure persistence, runtime assembly, and model consumption.

## 2. Core DTOs

### 2.1 EventRecord and RawTrace
`EventRecord` and `RawTrace` remain backward compatible with MVP1.

`RawTrace.process_version` is still string-based and defaults to `"1"` at DTO level, but Camunda Stage 3.2 structure artifacts are stored with `vNN` keys.

### 2.2 ProcessEventDTO (Camunda runtime, Stage 3.1)

Current code contract (`src/domain/entities/process_event.py`) uses:

| field | type | required | note |
|---|---|---:|---|
| `case_id` | `str` | yes | process instance id |
| `activity_def_id` | `str` | yes | BPMN activity definition id |
| `activity_name` | `str or None` | no | activity label |
| `activity_type` | `str or None` | no | userTask/gateway/etc |
| `proc_def_id` | `str or None` | no | process definition id |
| `proc_def_key` | `str or None` | no | process key |
| `proc_def_version` | `str or None` | no | version label from source |
| `task_id` | `str or None` | no | human task id |
| `act_inst_id` | `str or None` | no | activity instance id |
| `parent_act_inst_id` | `str or None` | no | subprocess scope link |
| `sequence_counter` | `int or None` | no | deterministic order helper |
| `execution_id` | `str or None` | no | execution tree id |
| `parent_execution_id` | `str or None` | no | execution hierarchy |
| `scope_depth` | `int or None` | no | execution scope depth |
| `is_concurrent` | `bool or None` | no | parallel token marker |
| `is_scope` | `bool or None` | no | scope marker |
| `is_event_scope` | `bool or None` | no | event scope marker |
| `execution_rev` | `int or None` | no | revision marker |
| `call_proc_inst_id` | `str or None` | no | call activity child case link |
| `called_element` | `str or None` | no | call binding key |
| `binding_type` | `str or None` | no | call binding mode |
| `version_tag` | `str or None` | no | call binding version tag |
| `version_number` | `str or None` | no | call binding version number |
| `resolved_child_proc_def_id` | `str or None` | no | resolved child process definition |
| `child_process_key` | `str or None` | no | child process key |
| `child_version` | `str or None` | no | child version marker |
| `start_time` | `datetime or None` | no | UTC recommended |
| `end_time` | `datetime or None` | no | UTC recommended |
| `duration_ms` | `float or None` | no | raw duration |
| `assignee` | `str or None` | no | assignee |
| `candidate_groups` | `List[str] or None` | no | candidate groups |
| `assigned_executor` | `str or None` | no | assigned executor |
| `executed_by` | `str or None` | no | actual executor |
| `potential_executor_users` | `List[str] or None` | no | optional |
| `potential_executor_groups` | `List[str] or None` | no | optional |
| `process_variables` | `Dict[str, Any] or None` | no | optional process vars |
| `call_activity_link` | `Dict[str, Any] or None` | no | optional call link payload |
| `removal_time` | `datetime or None` | no | cleanup diagnostics |
| `extra` | `Dict[str, Any]` | no | passthrough fields |

### 2.3 ProcessStructureDTO (Stage 3.2)

Current code contract (`src/domain/entities/process_structure.py`):

| field | type | required | note |
|---|---|---:|---|
| `version` | `str` | yes | normalized version key (`vNN` for Camunda) |
| `allowed_edges` | `List[Tuple[str,str]]` | yes | canonical transitions |
| `edge_statistics` | `Dict[(src,dst), Dict[str,float]] or None` | no | transition counts/stats |
| `node_metadata` | `Dict[str, Dict[str,str]] or None` | no | visualization labels/types |
| `proc_def_id` | `str or None` | no | strict structure identity |
| `proc_def_key` | `str or None` | no | process key |
| `deployment_id` | `str or None` | no | deployment identity |
| `nodes` | `List[Dict[str,Any]] or None` | no | detailed BPMN nodes |
| `edges` | `List[Dict[str,Any]] or None` | no | detailed BPMN edges |
| `graph_topology` | `Dict[str,Any] or None` | no | cycles/topology metadata |
| `call_bindings` | `Dict[str, Dict[str,Any]] or None` | no | callActivity binding map |
| `metadata` | `Dict[str,Any] or None` | no | parser diagnostics/warnings |

#### call_bindings payload (minimum expected)

For each call activity node id:
1. `called_element`
2. `binding_type`
3. `version_tag`
4. `version_number`
5. `resolved_child_proc_def_id`
6. `status` (`resolved` / `unresolved`)
7. `reason`
8. `requires_separate_inference` (bool)
9. `inference_trigger` (`child_process_root`)
10. `child_process_key`
11. `child_version_tag`
12. `inference_fallback_strategy` (`skip | use_aggregated_stats | raise`)

### 2.4 RuntimeFetchDiagnosticsDTO

Health/fallback diagnostics from runtime ingestion.

| field | type | required | note |
|---|---|---:|---|
| `history_coverage_percent` | `float or None` | yes | observed coverage |
| `cleaned_instances_percent` | `float or None` | yes | cleanup impact |
| `fallback_triggered` | `bool` | yes | fallback marker |
| `fallback_reason` | `str or None` | no | reason |
| `warnings` | `List[str]` | no | warnings |

## 3. File-based structure artifact schema

Repository path:
`data/knowledge_graph/<process_name>/<version>/process_structure.json`

Envelope fields:
1. `schema_version`
2. `repository_backend`
3. `created_at_utc`
4. `process_name`
5. `version`
6. `dto`

The `dto` block stores full `ProcessStructureDTO` including `call_bindings` and metadata.

## 4. GraphTensorContract extension status
MVP2 optional fields remain backward compatible:
1. `structural_edge_index`
2. `structural_edge_weight`
3. `version_emb_idx`
4. `allowed_target_mask`
5. `struct_x`

If absent, baseline path must remain valid.

## 5. Compatibility and versioning rules
1. JSON reader must ignore unknown fields.
2. Missing optional fields must not crash pipelines.
3. Ambiguous version-only lookup must return `None`.
4. For Camunda structure identity, use `(process_name, version=vNN, proc_def_id)` semantics.