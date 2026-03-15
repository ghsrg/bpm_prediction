# MVP2.5 Stage 1 Report

Date: 2026-03-13  
Status: Completed

## 1. Scope Delivered

Implemented Stage 1 refactor with both approved architectural corrections:
1. Dynamic version fallback without hardcoded `"1"`.
2. Visualization graph retrieval through knowledge-graph port/repository.

No Camunda or Neo4j runtime implementation was added in this stage.

---

## 2. Implemented Architecture Changes

## 2.1 Domain Port
Created domain-level contract:
- `src/domain/ports/knowledge_graph_port.py`

Port methods:
- `save_process_structure(version, dto, process_name=None)`
- `get_process_structure(version, process_name=None)`
- `list_versions(process_name=None)`
- `get_graph_for_visualization(process_name, version_key, min_edge_frequency=0) -> nx.DiGraph`

Compatibility alias kept:
- `src/application/ports/knowledge_graph_port.py` now re-exports domain port.

## 2.2 In-Memory Repository (NetworkX)
Created:
- `src/infrastructure/repositories/in_memory_networkx_repository.py`

Behavior:
- Stores DTOs and `nx.DiGraph` objects in-memory.
- Supports process-scoped retrieval for visualization.
- Supports edge-frequency filtering in `get_graph_for_visualization(...)`.

## 2.3 TopologyExtractorService Refactor
Updated:
- `src/application/services/topology_extractor_service.py`

Changes:
- Now receives `IKnowledgeGraphPort` via DI.
- `extract_from_logs(...)` computes DTOs and writes via `save_process_structure(...)`.
- No internal topology registry as source of truth.
- Visualization path now reads graph via repository port method and renders DFG.

## 2.4 DynamicGraphBuilder Refactor
Updated:
- `src/domain/services/dynamic_graph_builder.py`

Changes:
- Reads topology via injected domain port.
- Uses `prefix.process_version` key (normalized), no hardcoded fallback to `"1"`.
- Existing mask and structural tensor generation logic preserved.

## 2.5 XESAdapter Dynamic Dataset Fallback
Updated:
- `src/adapters/ingestion/xes_adapter.py`
- `src/cli.py`

Changes:
- Adapter now accepts `dataset_name` from mapping config.
- Process version resolution priority is now:
  - event version -> trace version -> log version -> dataset_name -> filename -> default
- CLI now injects `dataset_name` into mapping (`xes_adapter.dataset_name`) using:
  - `data.dataset_name` or `data.dataset_label` or log filename stem.

## 2.6 Visualization Tool Wiring
Updated:
- `tools/visualize_topology.py`

Changes:
- Uses `InMemoryNetworkXRepository + TopologyExtractorService` via DI.
- `plot_topology(...)` is called with explicit `process_name` and version.
- Rendering consumes graph obtained from knowledge port (repository-backed).

---

## 3. Tests Added/Updated

New tests:
- `tests/infrastructure/test_in_memory_networkx_repository.py`
- `tests/domain/test_dynamic_graph_builder_version_fallback.py`
- `tests/adapters/test_xes_adapter_process_version_fallback.py`

Updated tests:
- `tests/application/test_topology_extractor_service.py`
- `tests/application/test_visualize_topology_fallback.py`
- `tests/domain/test_dynamic_graph_builder_masks.py`
- `tests/domain/test_dynamic_graph_builder_tensor_purity.py`

---

## 4. Test Execution Results

## 4.1 Stage 1 Targeted Tests
Command:
```bash
.\.venv\Scripts\python.exe -m pytest tests/infrastructure/test_in_memory_networkx_repository.py tests/application/test_topology_extractor_service.py tests/domain/test_dynamic_graph_builder_masks.py tests/domain/test_dynamic_graph_builder_tensor_purity.py tests/domain/test_dynamic_graph_builder_version_fallback.py tests/application/test_visualize_topology_fallback.py tests/adapters/test_xes_adapter_process_version_fallback.py -v
```
Result:
- `22 passed`
- `0 failed`

## 4.2 MVP1 Regression Shield
Command:
```bash
.\.venv\Scripts\python.exe -m pytest -m mvp1_regression -v
```
Result:
- `16 passed`
- `0 failed`

## 4.3 Model Unit Tests (Dual-Encoder/EOPKG)
Command:
```bash
.\.venv\Scripts\python.exe -m pytest tests/domain/test_eopkg_models_forward.py tests/domain/test_eopkg_gatv2_dual_encoder.py -v
```
Result:
- `9 passed`
- `0 failed`

---

## 5. Additional Notes

- Added Python cache ignore rules in `.gitignore` (`__pycache__/`, `*.pyc`) to keep stage diffs clean.
- Stage 1 output is ready for Stage 2 validation/hardening and subsequent Camunda/Neo4j adapter swap stages.
