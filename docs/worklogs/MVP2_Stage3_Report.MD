# MVP2 Stage 3 Report

Date: 2026-03-12  
Scope: DynamicGraphBuilder and Tensor Purity (Topology -> Tensor mask injection)

## 1. Implemented

### 1.1 New DynamicGraphBuilder

Created [dynamic_graph_builder.py](C:\Users\korsr\PycharmProjects\bpm_prediction\src\domain\services\dynamic_graph_builder.py):

1. Added `DynamicGraphBuilder(BaselineGraphBuilder)` with constructor:
   - `feature_encoder: FeatureEncoder`
   - `knowledge_port: IKnowledgeGraphPort`
2. `build_graph(prefix)` flow:
   - builds baseline MVP1 graph contract via `super().build_graph(prefix)`
   - fetches `ProcessStructureDTO` by `prefix.process_version`
   - generates `allowed_target_mask: torch.BoolTensor` of size `C` (`len(activity_vocab)`)
   - marks `True` for class indices allowed by topology (`last_prefix_activity -> next_activity`)
   - injects `contract["allowed_target_mask"] = mask`
3. Fallback:
   - if topology is absent for version, sets `allowed_target_mask = None`
   - baseline tensors remain unchanged (MVP1-compatible behavior)

## 2. Tests Added

### 2.1 Mask correctness and fallback

Created [test_dynamic_graph_builder_masks.py](C:\Users\korsr\PycharmProjects\bpm_prediction\tests\domain\test_dynamic_graph_builder_masks.py):

1. `test_dynamic_graph_builder_builds_allowed_mask_for_known_version`
   - checks mask shape `C`, dtype `torch.bool`
   - validates allowed transitions are correctly marked `True`
2. `test_dynamic_graph_builder_fallbacks_to_none_mask_for_unknown_version`
   - validates unknown version path returns `allowed_target_mask is None`

### 2.2 Tensor purity

Created [test_dynamic_graph_builder_tensor_purity.py](C:\Users\korsr\PycharmProjects\bpm_prediction\tests\domain\test_dynamic_graph_builder_tensor_purity.py):

1. `test_dynamic_graph_builder_contract_contains_only_tensors_and_scalars`
   - verifies contract values are tensors (or scalar `num_nodes`)
   - ensures no lists/strings/dicts are injected into tensor contract
2. `test_dynamic_graph_builder_mask_dtype_is_bool_when_present`
   - strict dtype assertion for `allowed_target_mask`

## 3. Verification

### 3.1 New tests (targeted)

Command:

```bash
.\.venv\Scripts\python.exe -m pytest tests/domain/test_dynamic_graph_builder_masks.py -v
```

Result:
- `2 passed`

Command:

```bash
.\.venv\Scripts\python.exe -m pytest tests/domain/test_dynamic_graph_builder_tensor_purity.py -v
```

Result:
- `2 passed`

### 3.2 Full suite

Command:

```bash
.\.venv\Scripts\python.exe -m pytest tests/ -v
```

Result:
- `60 passed, 1 warning`

### 3.3 MVP1 Regression Shield

Command:

```bash
.\.venv\Scripts\python.exe -m pytest -m mvp1_regression -v
```

Result:
- `16 passed, 44 deselected, 1 warning`

## 4. Run readiness

Stage 3 is ready for integration experiments:

1. Tensor mask injection (`allowed_target_mask`) works for known topology versions.
2. Unknown topology versions preserve fallback behavior (`None`) and do not break MVP1 paths.
3. Regression shield remains green.

