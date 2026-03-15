# MVP2 General Implementation Plan

Updated: 2026-03-11  
Scope: Plan-first implementation for MVP2 Phase-1 (Zero-Shot Knowledge Injection) with strict MVP1 backward compatibility.

## 0. Preconditions and Ground Rules

1. Phase boundary:
   - MVP2 Phase-1 includes only `TopologyExtractorService + DynamicGraphBuilder + EOPKGFusionGNN`.
   - Critic/Semaphore/dynamic-beta are explicitly out of scope (MVP3).
2. Data-leakage policy:
   - Topology is extracted only from train traces.
3. Regression policy:
   - Existing MVP1 flows stay default and unchanged unless explicitly switched by config.
4. Validation policy:
   - Each step ships with focused tests + full `pytest tests/ -v` non-regression run.

---

## Step 1. DTO and Tensor Contract Extension

### 1.1 Files to create/update

1. Update [raw_trace.py](C:\Users\korsr\PycharmProjects\bpm_prediction\src\domain\entities\raw_trace.py)
2. Update [prefix_slice.py](C:\Users\korsr\PycharmProjects\bpm_prediction\src\domain\entities\prefix_slice.py)
3. Create `src/domain/entities/process_structure.py`
4. Update [tensor_contract.py](C:\Users\korsr\PycharmProjects\bpm_prediction\src\domain\entities\tensor_contract.py)

### 1.2 Classes/functions to add or modify

1. `RawTrace.process_version: str = "1"` (defaulted for compatibility and explicit contract).
2. `PrefixSlice.process_version: str = "1"` (same default policy).
3. New DTO `ProcessStructureDTO`:
   - `version: str`
   - `allowed_edges: List[Tuple[str, str]]`
   - `edge_statistics: Optional[Dict[Tuple[str, str], Dict[str, float]]] = None`
4. Extend `GraphTensorContract` with optional MVP2 fields:
   - `structural_edge_index`
   - `structural_edge_weight`
   - `version_emb_idx`
   - `allowed_target_mask`

### 1.3 MVP1 backward compatibility

1. All new tensor fields are optional.
2. Existing MVP1 builders/models continue to provide/consume only legacy fields.
3. Default `process_version="1"` avoids breakage for legacy traces/prefixes.

### 1.4 Tests for Step 1

1. `tests/contracts/test_mvp2_dto_contract.py`
   - `RawTrace` and `PrefixSlice` default `process_version` behavior.
   - `ProcessStructureDTO` validation.
2. `tests/contracts/test_graph_tensor_contract_mvp2_optional_fields.py`
   - legacy contract keys still accepted.
   - optional MVP2 fields can be present without breaking type/dtype assertions.

---

## Step 2. Knowledge Port and TopologyExtractorService (Oracle)

### 2.1 Files to create/update

1. Create `src/application/ports/knowledge_graph_port.py`
2. Create `src/application/services/topology_extractor_service.py`
3. Update composition points in [cli.py](C:\Users\korsr\PycharmProjects\bpm_prediction\src\cli.py) and trainer wiring config payloads as needed.

### 2.2 Classes/functions to add or modify

1. Interface `IKnowledgeGraphPort`:
   - `get_process_structure(version: str) -> Optional[ProcessStructureDTO]`
2. `TopologyExtractorService` implementing the port:
   - internal in-memory storage `{version: ProcessStructureDTO}`
   - `fit(train_traces: List[RawTrace]) -> None`
     - extract all unique transitions `(e_i.activity_id -> e_{i+1}.activity_id)` grouped by `process_version`
   - `get_process_structure(version: str)` lookup with `None` fallback

### 2.3 MVP1 backward compatibility

1. Service is not activated in MVP1 default path.
2. If config does not request structure mode, no KG port is required.

### 2.4 Tests for Step 2

1. `tests/application/test_topology_extractor_service.py`
   - `fit()` builds version-scoped unique edges correctly.
   - `get_process_structure()` returns `None` for unknown version.
   - no leakage: only passed train traces affect extracted topology.

---

## Step 3. DynamicGraphBuilder and Tensor Purity

### 3.1 Files to create/update

1. Create `src/domain/services/dynamic_graph_builder.py`
2. Update [graph_builder_base.py](C:\Users\korsr\PycharmProjects\bpm_prediction\src\domain\services\graph_builder_base.py) if additional abstract hooks are needed.
3. Update [cli.py](C:\Users\korsr\PycharmProjects\bpm_prediction\src\cli.py) to select `BaselineGraphBuilder` vs `DynamicGraphBuilder` by config.
4. Update [trainer.py](C:\Users\korsr\PycharmProjects\bpm_prediction\src\application\use_cases\trainer.py):
   - data-to-contract mapping for optional fields
   - loader payload fields (`allowed_target_mask`, `version_emb_idx`) with correct tensor shapes.

### 3.2 Classes/functions to add or modify

1. `DynamicGraphBuilder(feature_encoder, knowledge_graph_port, version_indexer)`
2. `build_graph(prefix: PrefixSlice) -> GraphTensorContract`:
   - build baseline tensors (`x_cat`, `x_num`, `edge_index`, etc.)
   - fetch `ProcessStructureDTO` by `prefix.process_version`
   - generate `allowed_target_mask` per LLD Algorithm 2:
     - `C = len(activity_vocab)`
     - `mask = torch.zeros(C, dtype=torch.bool)`
     - find allowed targets from `dto.allowed_edges` for last activity in prefix
     - map target labels to class indices via encoder vocab
   - set `version_emb_idx` from deterministic version->index mapping
   - optionally emit structural edges/weights (or `None` if unavailable)

### 3.3 MVP1 backward compatibility

1. `BaselineGraphBuilder` remains default.
2. `DynamicGraphBuilder` activated only when explicit MVP2 config flag is set.
3. If no structure exists for version, builder falls back to baseline tensors and leaves optional fields as `None`.

### 3.4 Tests for Step 3

1. `tests/domain/test_dynamic_graph_builder_masks.py`
   - mask generation correctness for known transitions.
   - unknown/empty topology yields zero mask or `None` per selected contract.
2. `tests/integration/test_dynamic_graph_builder_tensor_purity.py`
   - asserts only tensor/int fields, strict dtypes.
3. `tests/application/test_dynamic_graph_builder_mvp1_fallback.py`
   - confirms fallback behavior without topology.

---

## Step 4. EOPKGFusionGNN Model + Model Factory Integration

### 4.1 Files to create/update

1. Create `src/domain/models/eopkg_fusion_gnn.py`
2. Create model registry/factory utilities (if absent in current codebase), e.g.:
   - `src/domain/models/registry.py`
   - `src/domain/models/model_factory.py`
3. Update [cli.py](C:\Users\korsr\PycharmProjects\bpm_prediction\src\cli.py) model selection to use registry/factory.
4. Register baseline models into the same registry to preserve existing behavior.

### 4.2 Classes/functions to add or modify

1. `EOPKGFusionGNN(BaseGNN)`:
   - local observed context path (`H_obs`)
   - optional version embedding path (`V_emb`) from `version_emb_idx`
   - optional structural path (`H_struct`) using `structural_edge_index`
   - fusion (`concat + fusion_layer`) when structural/context present
   - strict fallback to observed-only path when structural input is absent
2. `@register_model("EOPKGFusionGNN")` and `ModelFactory.create(...)` wiring

### 4.3 MVP1 backward compatibility

1. Existing model names (`BaselineGCN`, `BaselineGATv2`) stay registered and default.
2. If MVP2 tensors are absent, `EOPKGFusionGNN.forward()` falls back to MVP1-like path.
3. Existing config files do not need changes for MVP1 runs.

### 4.4 Tests for Step 4

1. `tests/domain/test_eopkg_fusion_gnn_forward.py`
   - forward with only baseline tensors.
   - forward with structural tensors.
   - forward with version embedding only.
   - output shape and dtype invariants.
2. `tests/contracts/test_model_factory_registry.py`
   - baseline and EOPKGFusion models resolve via registry.
   - unknown model type raises clear error.

---

## Step 5. Evaluator Metrics: OOS + Slicing

### 5.1 Files to create/update

1. Update [trainer.py](C:\Users\korsr\PycharmProjects\bpm_prediction\src\application\use_cases\trainer.py) evaluation path (currently evaluator logic is embedded here).
2. Optionally extract evaluator logic into dedicated service for cleaner MVP2 extension (recommended but not required for first pass).

### 5.2 Classes/functions to add or modify

1. Extend test/eval aggregation with OOS rate:
   - `oos_rate = (~allowed_target_mask[torch.arange(B), y_hat]).float().mean().item()`
2. Add slicing metrics:
   - by `process_version`: `test_f1_v1`, `test_f1_v2`, ...
   - by prefix length bins:
     - `len_1_5`
     - `len_6_10`
     - `len_11_20`
     - `len_21_plus`
3. Ensure loader/contract path carries per-graph metadata required for slicing (version and prefix length derivation).

### 5.3 MVP1 backward compatibility

1. If `allowed_target_mask` is missing, OOS metric defaults to neutral behavior (e.g., skip or `0.0`) without errors.
2. Slicing keys appear only when corresponding metadata exists.
3. Legacy MVP1 metrics remain unchanged.

### 5.4 Tests for Step 5

1. `tests/application/test_evaluator_oos_metric.py`
   - synthetic batch verifies exact OOS formula result.
2. `tests/application/test_evaluator_slicing.py`
   - verifies version-based and length-bin metric keys/aggregation.
3. `tests/integration/test_eval_drift_mvp2_metrics_smoke.py`
   - in-memory drift evaluation with masks confirms OOS + slices are emitted.

---

## Cross-Step Regression Shield Execution

After each step:

1. `.\.venv\Scripts\python.exe -m pytest -m mvp1_regression -v`
2. `.\.venv\Scripts\python.exe -m pytest tests/ -v`
3. `py -3 tools/architecture_guard.py`

This keeps MVP1 continuously protected while MVP2 capabilities are introduced incrementally.

