# MVP2.5 EOPKGTopologyConditioned Stage 2 Foundation Report

- `date`: 2026-05-25
- `scope`: model-level dynamic candidate contract
- `model_type`: `EOPKGTopologyConditioned`
- `status`: implemented and targeted-tested

## Summary

Implemented the first Stage 2 slice for `EOPKGTopologyConditioned`: a
model-level dynamic candidate output contract.

The existing `forward(contract)` path remains backward-compatible and still
returns fixed label logits:

```text
forward(contract) -> logits [B, C_train]
```

The new Stage 2 path exposes topology-local candidates:

```text
forward_candidate(contract) -> CandidatePredictionOutput
```

## Runtime Contract

`CandidatePredictionOutput` contains:

```text
candidate_logits: FloatTensor [B, C_v]
candidate_class_index: LongTensor [C_v]
node_logits: FloatTensor [B, |V|]
node_to_candidate_index: LongTensor [|V|]
node_to_class_index: LongTensor [|V|]
fixed_label_logits: FloatTensor [B, C_train] | None
```

`candidate_class_index` maps the local dynamic candidate axis back to the
current global activity-class vocabulary. `map_targets_to_candidate_index()`
maps global targets into this local candidate axis and returns `-1` for targets
missing from the topology-local candidate set.

## What Changed

- Added `src/domain/entities/candidate_prediction.py`.
- Added `EOPKGTopologyConditioned.forward_candidate()`.
- Added dynamic candidate MIL pooling over valid `struct_node_to_class_index`
  classes.
- Kept `struct_node_to_class_index == -1` excluded from dynamic candidates.
- Kept duplicate activity labels pooled through the same configured
  `candidate_pooling` policy.
- Added `last_candidate_dynamic_count` and `last_candidate_class_index`
  diagnostics.
- Added `candidate_dynamic_count` to trainer metrics and selective trace
  attributes.

## Current Boundary

This is not yet full end-to-end dynamic candidate training.

Current:

```text
model can produce [B, C_v]
trainer/evaluator still consume [B, C_train]
```

Next required step:

```text
trainer/evaluator must consume CandidatePredictionOutput directly,
remap targets to local candidates,
compute candidate-level mask/OOS/calibration metrics,
and preserve fixed-label compatibility for old run comparisons.
```

## Tests

Targeted tests passed:

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\domain\test_eopkg_topology_conditioned.py -v
.\.venv-modern\Scripts\python.exe -m pytest tests\application\test_structural_trace_payload_builder.py -v
.\.venv-modern\Scripts\python.exe -m pytest tests\application\test_trainer_forward_stats_logging.py::test_trainer_logs_topology_conditioned_candidate_diagnostics -v
```

Full regression passed:

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\ -v
```

Result:

```text
484 passed, 3 warnings
```

Architecture guard passed:

```powershell
.\.venv-modern\Scripts\python.exe tools\architecture_guard.py
```

Result:

```text
[ARCH_GUARD] OK: architecture dependency rules satisfied.
```

## Research Interpretation

This closes the first technical gap between fixed-head structural fusion and
business-valid candidate scoring. The model can now represent a version-local
candidate set `C_v`, which is required for new/removed BPMN candidate handling.

It does not yet prove zero-shot benefit because the trainer and evaluator still
optimize/evaluate fixed-label logits by default.
