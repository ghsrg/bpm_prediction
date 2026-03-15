# MVP2 Sprint 3 Step 2 Report

Date: 2026-03-12  
Scope: Tensor contract extension for structural node features

## Done

1. Extended graph tensor contract with structural node features:
   - [tensor_contract.py](C:\Users\korsr\PycharmProjects\bpm_prediction\src\domain\entities\tensor_contract.py)
   - Added optional field:
     - `struct_x: Optional[torch.FloatTensor]`

2. Wired `struct_x` through trainer data conversion:
   - [trainer.py](C:\Users\korsr\PycharmProjects\bpm_prediction\src\application\use_cases\trainer.py)
   - `_build_loader(...)` now copies `contract["struct_x"]` into `Data` when present.
   - `_data_to_contract(...)` now restores `struct_x` back into `GraphTensorContract`.

3. Updated contract compatibility test:
   - [test_graph_tensor_contract_mvp2_optional_fields.py](C:\Users\korsr\PycharmProjects\bpm_prediction\tests\contracts\test_graph_tensor_contract_mvp2_optional_fields.py)
   - Added `struct_x` to extended contract assertion path.

## Notes

1. `struct_x` remains optional; MVP1 and existing pipelines stay compatible.
2. DynamicGraphBuilder can keep returning `struct_x=None` for now; model fallback will cover this path.

