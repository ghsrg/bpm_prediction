# MVP2 Stage 1 Report

Date: 2026-03-11  
Stage: Step 1 — DTO and Tensor Contract Extension

## 1. What Was Implemented

### 1.1 DTO updates

1. Updated [raw_trace.py](C:\Users\korsr\PycharmProjects\bpm_prediction\src\domain\entities\raw_trace.py):
   - `process_version: str = "1"`
2. Updated [prefix_slice.py](C:\Users\korsr\PycharmProjects\bpm_prediction\src\domain\entities\prefix_slice.py):
   - `process_version: str = "1"`

### 1.2 New MVP2 DTO

1. Added [process_structure.py](C:\Users\korsr\PycharmProjects\bpm_prediction\src\domain\entities\process_structure.py):
   - `ProcessStructureDTO(version, allowed_edges, edge_statistics=None)`

### 1.3 Tensor contract extension (backward-compatible)

1. Updated [tensor_contract.py](C:\Users\korsr\PycharmProjects\bpm_prediction\src\domain\entities\tensor_contract.py):
   - Added optional/not-required MVP2 fields:
     - `structural_edge_index`
     - `structural_edge_weight`
     - `version_emb_idx`
     - `allowed_target_mask`
2. Existing MVP1 required keys were not changed.

## 2. Tests Added in Step 1

1. [test_mvp2_dto_contract.py](C:\Users\korsr\PycharmProjects\bpm_prediction\tests\contracts\test_mvp2_dto_contract.py)
   - Verifies `RawTrace`/`PrefixSlice` default `process_version == "1"`.
   - Verifies `ProcessStructureDTO` required/optional fields.
2. [test_graph_tensor_contract_mvp2_optional_fields.py](C:\Users\korsr\PycharmProjects\bpm_prediction\tests\contracts\test_graph_tensor_contract_mvp2_optional_fields.py)
   - Verifies BaselineGCN forward still works with extended MVP2 optional fields present.
   - Confirms tensor dtypes and output shape remain valid.

## 3. MVP1 Compatibility Status

Compatibility is preserved:

1. DTO defaults do not break existing constructors that already pass `process_version`.
2. `GraphTensorContract` keeps all MVP1 required keys untouched.
3. Optional MVP2 keys are additive and do not affect MVP1 model execution.

## 4. Verification Runs

### 4.1 Regression Shield

Command:

```bash
.\.venv\Scripts\python.exe -m pytest -m mvp1_regression -v
```

Result:
- `16 passed, 30 deselected`

### 4.2 Full test suite

Command:

```bash
.\.venv\Scripts\python.exe -m pytest tests/ -v
```

Result:
- `46 passed`

## 5. Diff Summary (Step 1 scope)

Modified:

1. `src/domain/entities/raw_trace.py`
2. `src/domain/entities/prefix_slice.py`
3. `src/domain/entities/tensor_contract.py`

Added:

1. `src/domain/entities/process_structure.py`
2. `tests/contracts/test_mvp2_dto_contract.py`
3. `tests/contracts/test_graph_tensor_contract_mvp2_optional_fields.py`

## 6. Can We Run Anything Meaningful Now?

Yes, for Step 1 scope:

1. Contract-level checks are fully runnable via tests (green).
2. MVP1 runtime non-regression is confirmed (Regression Shield green).

Not yet (expected at this stage):

1. Full MVP2 Zero-Shot behavior is not runnable until Step 2+ (knowledge port, topology extractor, dynamic graph builder, model fusion path).

