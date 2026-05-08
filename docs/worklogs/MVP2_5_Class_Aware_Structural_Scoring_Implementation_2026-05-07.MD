# MVP2.5 Class-Aware Structural Scoring Implementation

## Metadata

- `status`: implemented
- `date`: 2026-05-07
- `scope`: EOPKGGATv2 fusion, trainer loss, Experiment UI config catalog, docs
- `plan`: `docs/worklogs/MVP2_5_Structural_Logit_Scale_And_Aux_Loss_Plan_2026-05-07.MD`

---

## Summary

Implemented the two-component `ClassAwareStructuralScoring` update:

- bilinear prefix-to-structure node scoring;
- structural prior head;
- late node-to-class `LogSumExp` pooling through `struct_node_to_class_index`;
- per-sample LayerNorm for raw structural class logits;
- observed-logit scale alignment with detached observed magnitude;
- optional set-aware structural auxiliary loss in trainer;
- runtime diagnostics for structural contribution and auxiliary loss.

---

## Runtime Semantics

Final logits are now:

```text
final_logits = observed_logits + structural_logits
```

where:

```text
node_score(i, v) = q_i^T W_bilinear k_v + W_prior h_v
raw_class_score(i, j) = LogSumExp({node_score(i, v) | struct_node_to_class_index[v] = j})
structural_logits(i, :) = gamma * obs_scale(i) * LayerNorm(raw_class_score(i, :))
```

`obs_scale` is detached, so it aligns magnitude without routing structural loss
through the observed branch.

Auxiliary loss:

```text
loss = main_CE
     + training.structural_aux_loss_weight * L_struct_set
     + training.structural_aux_exact_loss_weight * L_struct_exact
```

`L_struct_set` uses the allowed target mask and always includes the exact target.
This keeps the objective compatible with parallel/ambiguous valid targets.

---

## Changed Files

- `src/domain/models/eopkg_models.py`
- `src/application/use_cases/trainer.py`
- `src/cli.py`
- `configs/ui/config_catalog.yaml`
- `tests/domain/test_eopkg_gatv2_dual_encoder.py`
- `tests/application/test_trainer_forward_stats_logging.py`
- `tests/application/test_cli_flatten_config.py`
- `docs/GNN_RUNTIME_MVP2_5.MD`
- `docs/current/project-state.md`
- `docs/dissertation_changes.MD`

---

## How To Configure

Recommended first experimental setting:

```yaml
model:
  fusion_mode: ClassAwareStructuralScoring
  structural_score_mode: bilinear_with_prior
  structural_logit_scale_init: 0.1
  structural_logit_scale_max: 2.0
  structural_observed_scale_min: 1.0
  structural_observed_scale_max: 10.0

training:
  structural_aux_loss_enabled: true
  structural_aux_loss_weight: 0.05
  structural_aux_exact_loss_weight: 0.01
```

---

## What To Check In Runs

In train/validation forward logs and MLflow metrics, check:

- `observed_logits_mean_abs`
- `structural_logits_mean_abs`
- `structural_logits_max_abs`
- `structural_to_observed_logit_ratio`
- `structural_aux_set_loss`
- `structural_aux_exact_loss`
- `structural_aux_total_loss`

Expected early behavior:

- structural ratio should no longer stay near `0.006-0.012` only because of
  scale mismatch;
- `structural_aux_set_loss` should move if structure receives signal;
- if `structural_to_observed_logit_ratio` dominates, reduce
  `model.structural_logit_scale_init` or `training.structural_aux_loss_weight`.

---

## Verification

Focused tests passed:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/domain/test_eopkg_gatv2_dual_encoder.py tests/application/test_trainer_forward_stats_logging.py tests/application/test_cli_flatten_config.py -v
```

Result:

```text
37 passed
```

Full verification passed:

```powershell
.\.venv\Scripts\python.exe tools\architecture_guard.py
.\.venv\Scripts\python.exe -m pytest -m mvp1_regression -v
.\.venv\Scripts\python.exe -m pytest tests/ -v
```

Result:

```text
[ARCH_GUARD] OK: architecture dependency rules satisfied.
20 passed, 289 deselected
309 passed, 12 warnings
```
