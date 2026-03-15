# DATA_MODEL_MVP2_5.MD

Updated: 2026-03-15  
Status: ACTIVE

## 1. Purpose
Define canonical DTO contracts used in MVP2.5 ingestion, structure persistence, runtime assembly, and model training.

## 2. Core DTOs

### 2.1 EventRecord and RawTrace
`EventRecord` and `RawTrace` are inherited from MVP1 and remain backward compatible.

### 2.2 ProcessEventDTO (Camunda runtime)
Required for Stage 3.1 runtime ingestion.

| field | type | required | note |
|---|---|---:|---|
| `case_id` | `str` | yes | process instance id |
| `activity_id` | `str` | yes | BPMN activity definition id |
| `activity_instance_id` | `str` | yes | runtime activity instance id |
| `activity_type` | `str` | no | userTask gateway etc |
| `start_time` | `datetime or None` | no | UTC recommended |
| `end_time` | `datetime or None` | no | UTC recommended |
| `duration_ms` | `float or None` | no | raw duration |
| `task_id` | `str or None` | no | human task id |
| `execution_id` | `str or None` | no | execution tree link |
| `parent_execution_id` | `str or None` | no | execution hierarchy |
| `parent_activity_instance_id` | `str or None` | no | scope hierarchy |
| `call_proc_inst_id` | `str or None` | no | call activity link |
| `sequence_counter` | `int or None` | no | deterministic order helper |
| `loop_counter` | `int or None` | no | multi-instance iteration |
| `removal_time` | `datetime or None` | no | cleanup diagnostics |

### 2.3 ProcessStructureDTO
Used as topology artifact in knowledge repository.

| field | type | required | note |
|---|---|---:|---|
| `version` | `str` | yes | process version key |
| `allowed_edges` | `List[Tuple[str,str]]` | yes | canonical transitions |
| `edge_statistics` | `Dict[(src,dst), Dict[str,float]] or None` | no | count or stats |
| `node_metadata` | `Dict[str, Dict[str,str]] or None` | no | display metadata |

### 2.4 RuntimeFetchDiagnosticsDTO
Defines health and fallback diagnostics.

| field | type | required | note |
|---|---|---:|---|
| `history_coverage_percent` | `float` | yes | observed coverage |
| `cleaned_instances_percent` | `float` | yes | likely cleanup impact |
| `fallback_triggered` | `bool` | yes | assembler fallback marker |
| `fallback_reason` | `str or None` | no | reason string |
| `warnings` | `List[str]` | no | non-fatal warnings |

## 3. File-based structure artifact schema
Repository file path:
`data/knowledge_graph/<process_name>/<version>/process_structure.json`

Canonical JSON payload:
```json
{
  "schema_version": "1.0",
  "repository_backend": "file",
  "created_at_utc": "2026-03-15T08:00:00+00:00",
  "process_name": "bpi2012",
  "version": "bpi2012",
  "dto": {
    "version": "bpi2012",
    "allowed_edges": [["A", "B"], ["B", "C"]],
    "edge_statistics": [
      {"src": "A", "dst": "B", "stats": {"count": 120.0}},
      {"src": "B", "dst": "C", "stats": {"count": 95.0}}
    ],
    "node_metadata": {
      "A": {"activity_name": "Start", "activity_type": "startEvent"}
    }
  }
}
```

## 4. GraphTensorContract extension status
MVP2 optional fields remain optional and backward compatible:
- `structural_edge_index`
- `structural_edge_weight`
- `version_emb_idx`
- `allowed_target_mask`
- `struct_x`

If not present, baseline-compatible path must remain valid.

## 5. Versioning and compatibility
- Schema version is explicit (`schema_version`).
- Unknown fields in JSON should be ignored by reader.
- Missing optional fields must not crash train/eval pipelines.
