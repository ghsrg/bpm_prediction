# LLD_MVP2_5.MD

Updated: 2026-03-17  
Status: ACTIVE

## 1. Purpose
Low-level design for MVP2.5 decoupled ingestion, structure persistence, and runtime-safe model path.

## 2. FileBasedKnowledgeGraphRepository

### 2.1 Responsibilities
1. Persist `ProcessStructureDTO` as JSON artifact.
2. Read by `(process_name, version)`.
3. Support visualization graph retrieval with edge-frequency filter.

### 2.2 Atomic write algorithm
```text
payload = serialize(dto)
write payload -> target.tmp
(optional) fsync tmp
os.replace(target.tmp, target)
```

Properties:
1. no partial target writes,
2. idempotent overwrite per `(process_name, version)`,
3. crash-safe replacement on same filesystem.

### 2.3 Read behavior
1. strict lookup when process name is provided,
2. version-only lookup allowed only when unambiguous,
3. invalid JSON returns `None` (no hard crash).

## 3. Camunda BPMN Ingestion Path (Stage 3.2)

### 3.1 Entry routing
`tools/ingest_topology.py`:
1. if adapter is Camunda and `structure.source=bpmn` and `structure_from_logs=false` -> BPMN path,
2. otherwise legacy logs-based extraction path.

### 3.2 CamundaBpmnAdapter
Responsibilities:
1. fetch process definition catalog,
2. fetch BPMN payload by `proc_def_id`,
3. support `files` and `mssql` sources,
4. fallback SQL lookup across configured `sql_dir`, `sql/structure`, `sql/runtime`.

### 3.3 BpmnStructureParserService
Responsibilities:
1. sanitize XML,
2. parse BPMN nodes/edges,
3. flatten embedded subprocesses (`flattened-no-subprocess-node`),
4. build `call_bindings` with explicit resolved/unresolved state,
5. set `inference_fallback_strategy` in call bindings,
6. produce quarantine record on hard parse failure.

## 4. DynamicGraphBuilder graceful degradation

### 4.1 Structure resolution logic
1. resolve version from prefix,
2. attempt both numeric/v-prefixed variants (`22` <-> `v22`),
3. read structure from knowledge port,
4. if found -> build `allowed_target_mask`,
5. if missing -> keep structure tensors/mask as `None` and continue baseline path.

### 4.2 Mask construction
1. `allowed_mask = zeros(C, bool)`
2. set allowed next activities for last prefix activity,
3. unknown activity ids are skipped safely.

## 5. Strict load switch
Config:
- `mapping.knowledge_graph.strict_load: true`

Behavior:
1. if enabled and no structure versions exist for process -> fail fast with explicit message,
2. if disabled -> warn and continue baseline fallback.

## 6. Failure matrix
| component | failure | behavior |
|---|---|---|
| file repo write | disk error | explicit IO exception |
| file repo read | missing file | `None` |
| BPMN parser | invalid xml | quarantine record + continue |
| topology in train | structure missing | fallback unless strict_load |
| runtime fetch | cleanup reduced history | diagnostics warning + fallback graph |
| visualization | no transitions after filter | clear user-facing error |

## 7. Test focus
1. repository atomic save/read + extended DTO fields,
2. BPMN parser (flattening, call bindings, invalid XML quarantine),
3. ingestion routing for Camunda BPMN files mode,
4. version fallback in `DynamicGraphBuilder` (`22`/`v22`),
5. non-regression (`pytest tests/`, `pytest -m mvp1_regression`).