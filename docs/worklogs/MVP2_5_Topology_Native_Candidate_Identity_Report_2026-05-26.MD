# MVP2.5 Topology-Native Candidate Identity Report

## Scope

Implemented the first topology-native candidate identity block for
`EOPKGTopologyConditioned`.

The change addresses the audit finding that `candidate_id` was still partially
tied to the training activity vocabulary and could collapse future labels to
`<UNK>` or drop future topology edges before scoring.

## Implemented

- Extended `CandidatePredictionOutput` with:
  - `candidate_ids`;
  - `candidate_labels`;
  - `candidate_is_unseen`;
  - label-based target mask helper.
- Added candidate target helpers:
  - `candidate_target_mask_from_labels`;
  - `candidate_set_cross_entropy_from_mask`.
- Added raw `target_label` to baseline graph contracts.
- Updated `DynamicGraphBuilder` to emit topology-native candidate metadata:
  - `candidate_ids`;
  - `candidate_labels`;
  - `candidate_class_index`;
  - `candidate_is_unseen`;
  - `struct_node_to_candidate_index`;
  - `candidate_allowed_target_mask`.
- Updated topology disk cache schema and serialization to include candidate
  metadata.
- Updated `EOPKGTopologyConditioned.forward_candidate()` to pool candidate
  logits by topology candidate index instead of dropping `class_idx=-1`
  candidates.
- Added `training.candidate_identity_mode` with default `topology_native`.
- Removed the stale guard that rejected
  `candidate_batch_topology_policy=group_by_topology`.
- Propagated candidate metadata and `target_label` through trainer-built and
  CLI-built PyG `Data` payloads.
- Updated trace diagnostics with candidate IDs, labels, and unseen flags.
- Added forward/MLflow diagnostics for:
  - `candidate_unseen_candidate_rate`;
  - `candidate_unseen_target_rate`.
- Fixed `topology_conditioned` retention/rehearsal for `candidate_id` mode:
  version weights now apply to candidate-set cross entropy, while fixed-vocab
  `class_weights` are not applied to topology-native logits.

## Verification

Targeted tests run:

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\domain\test_dynamic_graph_builder_topology_native_candidates.py tests\domain\test_eopkg_topology_conditioned.py tests\application\test_candidate_target_mapping.py -v
```

Result:

```text
27 passed, 3 warnings
```

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\application\test_trainer_topology_conditioned_learning.py tests\application\test_trainer_forward_stats_logging.py -k "candidate_id or candidate_contract" -v
```

Result:

```text
12 passed, 24 deselected, 4 warnings
```

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\application\test_trainer_forward_stats_logging.py::test_trainer_logs_topology_conditioned_candidate_diagnostics tests\application\test_trainer_forward_stats_logging.py::test_forward_stats_logs_candidate_target_mapping_metrics -v
```

Result:

```text
2 passed, 3 warnings
```

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\application\test_candidate_target_mapping.py::test_candidate_set_cross_entropy_from_mask_supports_sample_weights tests\application\test_trainer_topology_conditioned_learning.py::test_candidate_id_retention_uses_candidate_loss_without_fixed_vocab_class_weights -v
```

Result:

```text
2 passed, 3 warnings
```

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\application\test_trainer_topology_conditioned_learning.py tests\application\test_candidate_target_mapping.py -v
```

Result:

```text
17 passed, 3 warnings
```

```powershell
.\.venv-modern\Scripts\python.exe tools\architecture_guard.py
```

Result:

```text
[ARCH_GUARD] OK: architecture dependency rules satisfied.
```

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\application\test_experiment_ui_catalog_fields.py::test_catalog_includes_candidate_contract_mode tests\application\test_trainer_forward_stats_logging.py::test_candidate_contract_mode_accepts_candidate_id_and_default_topology_policy -v
```

Result:

```text
2 passed, 3 warnings
```

Full test suite was intentionally not run per user request.

## What To Check In Experiments

Run `_EOPKGTC-UN` / `_EOPKGTC-UN-drift` style presets with:

```yaml
training.candidate_contract_mode: candidate_id
training.candidate_identity_mode: topology_native
training.candidate_batch_topology_policy: group_by_topology
```

Expected signs that the new contract is active:

```text
candidate_dynamic_count changes across versions when topology candidates differ
candidate_class_index contains -1 when a future candidate label is absent from train vocab
candidate_unseen_target_rate becomes visible on future versions
candidate_missing_target_rate remains near 0 for aligned XES/collapse runs
traces include candidate_ids/candidate_labels/candidate_is_unseen
```

## Remaining Limitations

- XES target supervision is still label-based, not exact BPMN node-id based.
- Duplicate labels are handled as a target set, not exact node identity.
- Unseen candidates are preserved, but semantic grounding is not implemented.
- Wrong-version negative loss remains skipped for candidate axes that may differ.
- Candidate-space ECE/F1 coverage is started through candidate diagnostics, but
  fixed-label MLflow charts remain compatibility views when unseen candidates
  exist.

## Next Step

Run a small drift experiment first and inspect:

```text
candidate_dynamic_count
candidate_class_index
candidate_missing_target_rate
candidate_unseen_target_rate
candidate_oos_rate
candidate_invalid_probability_mass
```

If unseen candidates appear but are never selected, proceed to semantic
candidate prototypes / retrieval-style scoring.
