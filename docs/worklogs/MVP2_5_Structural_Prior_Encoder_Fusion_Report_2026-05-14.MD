# MVP2.5 StructuralPriorEncoder Fusion Implementation Report

## Summary

Implemented `model.fusion_mode=StructuralPriorEncoder` for `EOPKGGATv2`.

The mode is an etalon-like structural baseline:

```text
obs_context = ObservedGNN(prefix)
struct_context = mean(StructuralGNN(struct_x, structural_edge_index))
logits = Classifier(Fusion([obs_context || struct_context]))
```

It supports:

```yaml
model.structural_prior_pooling: mean
model.structural_prior_fusion: concat
model.structural_prior_fusion: gated_concat
model.structural_prior_gate_init_bias: -1.0
```

## Implemented Changes

### Runtime

- `src/domain/models/eopkg_models.py`
  - Added aliases:
    - `StructuralPriorEncoder`
    - `structural_prior_encoder`
    - `StructuralPrior`
    - `BPMNStructuralPrior`
  - Added config validation for:
    - `model.structural_prior_pooling`
    - `model.structural_prior_fusion`
    - `model.structural_prior_gate_init_bias`
  - Added `concat` fusion:

```text
fused = Fusion([obs_context || struct_context])
```

  - Added `gated_concat` fusion:

```text
gate = sigmoid(W_gate([obs_context || struct_context]) + b)
fused = Fusion([obs_context || gate * struct_context])
```

### CLI / Config

- `src/cli.py`
  - Passes new structural-prior keys into the model factory.

- `configs/ui/config_catalog.yaml`
  - Added `StructuralPriorEncoder` to the `model.fusion_mode` dropdown.
  - Added UI/config entries for:
    - `model.structural_prior_pooling`
    - `model.structural_prior_fusion`
    - `model.structural_prior_gate_init_bias`

### Diagnostics

- `src/application/use_cases/trainer.py`
  - Added forward-stat diagnostics:
    - `observed_context_mean_abs`
    - `structural_prior_context_mean_abs`
    - `structural_prior_to_observed_context_ratio`
    - `structural_prior_gate_mean`
    - `structural_prior_gate_max`
  - Metrics are logged to tracker with stage prefixes, for example:
    - `train_observed_context_mean_abs`
    - `train_structural_prior_context_mean_abs`
    - `train_structural_prior_to_observed_context_ratio`
    - `train_structural_prior_gate_mean`
    - `train_structural_prior_gate_max`

### Tests

- `tests/domain/test_eopkg_gatv2_dual_encoder.py`
  - Alias normalization tests.
  - Unsupported `structural_prior_fusion` validation.
  - Unsupported `structural_prior_pooling` validation.

- `tests/domain/test_eopkg_models_forward.py`
  - Forward coverage for `StructuralPriorEncoder`.
  - `concat` diagnostics coverage.
  - `gated_concat` gate diagnostics coverage.

- `tests/application/test_cli_flatten_config.py`
  - CLI model factory kwargs coverage.

- `tests/application/test_trainer_forward_stats_logging.py`
  - Trainer forward-stat logging coverage for structural-prior context and gate metrics.

## Documentation Updates

- `docs/GNN_RUNTIME_MVP2_5.MD`
  - Added `StructuralPriorEncoder Fusion` section.

- `docs/current/project-state.md`
  - Added current runtime note for `StructuralPriorEncoder`.

- `docs/dissertation_changes.MD`
  - Added dissertation note about this mode as a clean structural baseline.

## Verification

Focused tests executed during implementation and final verification:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/domain/test_eopkg_gatv2_dual_encoder.py -k "fusion_mode_aliases or structural_prior" -v
.\.venv\Scripts\python.exe -m pytest tests/domain/test_eopkg_models_forward.py -k "StructuralPriorEncoder or structural_prior_encoder" -v
.\.venv\Scripts\python.exe -m pytest tests/application/test_cli_flatten_config.py -v
.\.venv\Scripts\python.exe -m pytest tests/application/test_trainer_forward_stats_logging.py -v
.\.venv\Scripts\python.exe tools\architecture_guard.py
```

Results:

- `tests/domain/test_eopkg_models_forward.py tests/domain/test_eopkg_gatv2_dual_encoder.py tests/domain/test_eopkg_topology_state_encoder.py`: `53 passed`
- `tests/application/test_cli_flatten_config.py tests/application/test_trainer_forward_stats_logging.py`: `24 passed`
- `tools/architecture_guard.py`: `[ARCH_GUARD] OK`

Full regression was also executed:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -v
```

Result:

- `356 passed`
- `1 failed`: `tests/domain/test_dynamic_graph_builder_cache_policy.py::test_cache_policy_full_caches_dto_for_strict_asof`

The failing test is outside the `StructuralPriorEncoder` change set. This task
did not modify `src/domain/services/dynamic_graph_builder.py` or
`tests/domain/test_dynamic_graph_builder_cache_policy.py`. The failure is a
pre-existing/residual cache-policy issue where `repo.calls_asof == 5` instead of
the expected `1`.

## How To Test Experimentally

Recommended first train matrix:

```text
Base-UN-loan
S_Str-UNcPrior-loan
S_Str-UNcPriorGated-loan
```

If statistics are enabled:

```text
S_Str-UScPrior-loan
S_Str-UScPriorGated-loan
```

Recommended config snippets:

```yaml
model:
  fusion_mode: StructuralPriorEncoder
  structural_prior_pooling: mean
  structural_prior_fusion: concat
```

```yaml
model:
  fusion_mode: StructuralPriorEncoder
  structural_prior_pooling: mean
  structural_prior_fusion: gated_concat
  structural_prior_gate_init_bias: -1.0
```

Primary interpretation:

- If `concat` improves strict-F1/OOS versus Base-UN, structure helps as a representation-level prior.
- If `gated_concat` is more stable with similar or better strict-F1, use gated variant for drift experiments.
- If both fail while `ClassAwareStructuralScoring` helps, the structure is more useful as candidate ranking/logit prior than as a global representation context.
- If `structural_prior_to_observed_context_ratio` is near zero, the branch is ignored.
- If `structural_prior_gate_mean` quickly saturates near `1.0` and validation oscillates, structural context may be over-dominating.

## Notes

This feature is not intended to solve version-transfer by itself. It establishes
a clean structural baseline first. Drift transfer remains a separate
methodology/architecture question.
