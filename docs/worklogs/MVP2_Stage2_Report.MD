# MVP2 Stage 2 Report

Date: 2026-03-11  
Stage: Step 2 — Knowledge Port and TopologyExtractorService

## 1. Implemented Changes

### 1.1 New Knowledge Port

Created [knowledge_graph_port.py](C:\Users\korsr\PycharmProjects\bpm_prediction\src\application\ports\knowledge_graph_port.py):

1. `IKnowledgeGraphPort` (Protocol)
2. Method contract:
   - `get_process_structure(version: str) -> Optional[ProcessStructureDTO]`

### 1.2 New Topology Extractor Service

Created [topology_extractor_service.py](C:\Users\korsr\PycharmProjects\bpm_prediction\src\application\services\topology_extractor_service.py):

1. `TopologyExtractorService(IKnowledgeGraphPort)`
2. In-memory state:
   - `self._topology_registry: Dict[str, ProcessStructureDTO]`
3. Implemented `extract_from_logs(train_traces: List[RawTrace])`:
   - collects unique transitions `(e_i.activity_id -> e_{i+1}.activity_id)`
   - groups by `process_version`
   - stores normalized `ProcessStructureDTO` per version
4. Implemented `extract_from_bpmn(bpmn_data: Any)`:
   - raises:
   - `NotImplementedError("Will be implemented in Enterprise PoC for Camunda integration")`
5. Implemented `get_process_structure(version: str)`:
   - returns DTO or `None` for unknown version

## 2. Tests Added

Created [test_topology_extractor_service.py](C:\Users\korsr\PycharmProjects\bpm_prediction\tests\application\test_topology_extractor_service.py) with:

1. `test_extract_from_logs_builds_unique_edges_grouped_by_version_without_leakage`
   - checks unique edges per version
   - verifies no leakage from out-of-scope (holdout) transitions
2. `test_get_process_structure_returns_none_for_unknown_version`
3. `test_extract_from_bpmn_is_explicit_not_implemented_yet`

## 3. Verification Runs

### 3.1 Regression Shield

Command:

```bash
.\.venv\Scripts\python.exe -m pytest -m mvp1_regression -v
```

Result:
- `16 passed, 33 deselected, 1 warning`

### 3.2 Full test suite

Command:

```bash
.\.venv\Scripts\python.exe -m pytest tests/ -v
```

Result:
- `49 passed, 1 warning`

## 4. MVP1 Compatibility Status

1. MVP1 regression shield remains green.
2. New knowledge components are additive and not wired into MVP1 runtime path yet.
3. No behavior change in existing MVP1 training/evaluation code paths at this stage.

## 5. Can We Run Full MVP2 Validation Now?

Partially:

1. We can already validate knowledge extraction logic (logs -> versioned topology DTOs).
2. We can test BPMN-path contract behavior (explicit `NotImplementedError`).

Not yet full MVP2 pipeline:

1. Dynamic graph injection (`allowed_target_mask`, structural tensors in builder) is Step 3.
2. EOPKG model forward fusion is Step 4.
3. OOS/slicing metrics integration is Step 5.

