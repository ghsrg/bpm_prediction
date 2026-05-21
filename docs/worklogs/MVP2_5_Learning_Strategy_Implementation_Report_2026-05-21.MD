# GNN Learning Strategy Implementation Report

## Summary

Implemented `training.learning_strategy=standard | topology_conditioned` as a
trainer-level methodology separate from `model.fusion_mode`.

`standard` preserves previous train/eval behavior.

`topology_conditioned` adds train-only topology dependency objectives:

1. `L_allowed_set`
2. `L_wrong_version_negative`
3. `L_drop_edges_negative`
4. `L_retention`

`experiment.mode=versioned_zero_shot` remains out of scope for this change.

## Implemented Files

Created:

- `src/application/services/learning_strategy_config.py`
- `src/application/services/topology_conditioned_learning.py`
- `src/application/services/topology_payload_pool.py`
- `tests/application/test_learning_strategy_config.py`
- `tests/application/test_topology_conditioned_learning.py`
- `tests/application/test_topology_payload_pool.py`
- `tests/application/test_trainer_topology_conditioned_learning.py`

Modified:

- `src/application/use_cases/trainer.py`
- `configs/ui/config_catalog.yaml`
- `tests/application/test_experiment_ui_catalog_fields.py`
- `docs/GNN_LEARNING_STRATEGY.MD`
- `docs/GNN_RUNTIME_MVP2_5.MD`
- `docs/EVF_MVP2_5.MD`
- `docs/current/project-state.md`
- `docs/current/architecture-debt.md`
- `docs/dissertation_changes.MD`

## Config Keys

Added to `configs/ui/config_catalog.yaml`:

- `training.learning_strategy`
- `training.topology_conditioning_wrong_version_negative_enabled`
- `training.topology_conditioning_wrong_version_negative_weight`
- `training.topology_conditioning_wrong_version_margin`
- `training.topology_conditioning_drop_edges_negative_enabled`
- `training.topology_conditioning_drop_edges_negative_weight`
- `training.topology_conditioning_drop_edges_margin`
- `training.topology_conditioning_drop_edges_ratio`
- `training.topology_conditioning_allowed_set_loss_enabled`
- `training.topology_conditioning_allowed_set_loss_weight`
- `training.topology_conditioning_retention_enabled`
- `training.topology_conditioning_retention_policy`
- `training.topology_conditioning_current_version_policy`
- `training.topology_conditioning_retention_recent_versions`
- `training.topology_conditioning_retention_recent_weight`
- `training.topology_conditioning_retention_old_weight`
- `training.topology_conditioning_retention_obsolete_weight`

## Runtime Behavior

### `standard`

No additional negative-topology forwards are executed.

### `topology_conditioned`

Requires `structural_mode=true` and a supported/allowed structural
`fusion_mode`.

During training only, the trainer can add:

- allowed-set loss on raw logits and `allowed_target_mask`;
- wrong-version topology negative from known train versions;
- same-version physical `drop_edges` negative;
- version-weighted CE/rehearsal retention.

For in-memory train datasets, topology-conditioned training uses
version-homogeneous mini-batches where possible so wrong-version negatives do
not get disabled by mixed batch composition.

During validation/test/eval_drift, these objectives are not applied.

## Diagnostics

Tracker metrics include:

- `train_topology_conditioned_allowed_set_loss`
- `train_topology_conditioned_wrong_version_ce`
- `train_topology_conditioned_wrong_version_ce_delta`
- `train_topology_conditioned_wrong_version_loss`
- `train_topology_conditioned_drop_edges_ce`
- `train_topology_conditioned_drop_edges_ce_delta`
- `train_topology_conditioned_drop_edges_loss`
- `train_topology_conditioned_retention_loss`
- `train_topology_conditioned_mixed_version_batches`
- `train_topology_conditioned_wrong_version_skipped_mixed_batches`

Interpretation:

- positive `wrong_version_ce_delta` means correct topology gives lower CE than a
  known wrong-version topology;
- positive `drop_edges_ce_delta` means the model reacts to local topology
  damage;
- mixed-batch counters must be checked before interpreting version-transition
  windows.

## Known Limitations

- `versioned_zero_shot` orchestration is not implemented in this change.
- Wrong-version negatives use one structural payload per batch. In-memory train
  datasets prefer version-homogeneous batches. If mixed-version batches still
  reach the trainer, this term is skipped with explicit diagnostics.
- Same-version corruption is physical `drop_edges`, not Soft Edge Drop.
- Soft Edge Drop is deferred until structural GNN forward consumes
  `structural_edge_weight`.
- Retention is version-weighted CE/rehearsal only; no distillation yet.

## Verification

TDD red checks:

- service modules initially failed with `ModuleNotFoundError`;
- trainer integration tests initially failed because topology-conditioned losses
  were not wired;
- catalog test initially failed because new config keys were absent.

Targeted verification passed:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\application\test_learning_strategy_config.py tests\application\test_topology_payload_pool.py tests\application\test_topology_conditioned_learning.py tests\application\test_trainer_topology_conditioned_learning.py tests\application\test_experiment_ui_catalog_fields.py -v
```

Result:

```text
18 passed
```

Full verification:

```powershell
.\.venv\Scripts\python.exe tools\architecture_guard.py
```

Result:

```text
[ARCH_GUARD] OK: architecture dependency rules satisfied.
```

```powershell
.\.venv\Scripts\python.exe -m pytest tests\ -v
```

Result:

```text
422 passed, 15 warnings
```

```powershell
.\.venv\Scripts\python.exe -m pytest -m mvp1_regression -v
```

Result:

```text
20 passed, 402 deselected, 11 warnings
```
