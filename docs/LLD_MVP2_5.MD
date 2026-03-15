# LLD_MVP2_5.MD

Updated: 2026-03-15  
Status: ACTIVE

## 1. Purpose
Low-level implementation design for MVP2.5 decoupled ingestion and runtime resilience.

## 2. FileBasedKnowledgeGraphRepository

### 2.1 Responsibilities
- Persist `ProcessStructureDTO` as JSON artifacts.
- Read by `process_name + version`.
- Provide visualization graph object with edge-frequency filtering.

### 2.2 Atomic write algorithm
Pseudo-code:
```text
payload = serialize(dto)
write payload to target.tmp
fsync tmp if available
os.replace(target.tmp, target)
```

Properties:
- crash-safe replacement semantics on same filesystem,
- no partially written target file,
- idempotent overwrite per `(process_name, version)`.

### 2.3 Read behavior
- strict key lookup when process name provided,
- version-only lookup allowed only if unambiguous,
- invalid JSON returns `None` instead of hard crash.

## 3. InstanceGraphAssemblerService behavior

### 3.1 Canonical assembly
Inputs:
- activity events,
- execution rows,
- variable rows,
- identity rows,
- optional process links.

Outputs:
- canonical graph dict,
- collapsed projection,
- diagnostics object.

### 3.2 Mode policy
- `activity-centric`: primary mode for stable and scalable runtime.
- `execution-centric`: optional refinement, may degrade to activity-centric when runtime execution coverage is insufficient.

### 3.3 Cleanup-aware fallback
When history coverage becomes too low:
1. log warning,
2. set diagnostics fallback flag,
3. return valid fallback graph (`mode=fallback_structure_only`) instead of raising runtime error.

## 4. DynamicGraphBuilder graceful degradation

### 4.1 Structure resolution logic
1. resolve process version from prefix,
2. query knowledge port,
3. if found: build `allowed_target_mask`,
4. if not found: keep structure fields `None` and continue baseline path.

### 4.2 Mask construction
Given last activity in prefix and class vocabulary:
- initialize `allowed_mask = zeros(C, dtype=bool)`,
- mark `True` for all allowed next activities present in vocabulary,
- unknown activity ids are skipped safely.

## 5. Strict load switch
Config:
- `mapping.knowledge_graph.strict_load: true`

Behavior:
- if enabled and no structure versions found for process, fail fast with explicit message,
- if disabled, continue in fallback baseline mode.

## 6. Failure matrix
| component | failure | behavior |
|---|---|---|
| file repository write | disk error | raise explicit IO error |
| file repository read | missing file | return `None` |
| topology in train | structure missing | fallback unless strict_load |
| runtime fetch | cleanup reduced history | diagnostics warning and fallback graph |
| visualization | no transitions after filter | clear user-facing error |

## 7. Test focus
- repository atomic save and read,
- ambiguous version lookup rules,
- train path does not call synchronous extraction,
- fallback path remains deterministic,
- strict load fail-fast behavior.
