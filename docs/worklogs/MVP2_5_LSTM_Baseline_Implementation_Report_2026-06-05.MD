# MVP2.5 LSTM Baseline Implementation Report

Date: 2026-06-05

## Scope

Implemented `model.type: LSTM_Baseline` as a fixed-vocabulary logs-only
comparison model for structural-drift experiments.

The model is intentionally not topology-native:

- it consumes only observed prefix tensors (`x_cat`, `x_num`, `batch`);
- it ignores structural tensors and graph edges;
- it returns fixed softmax logits `[B, C_train]`;
- future-version activities unseen during training remain outside its output
  head.

This matches the article baseline requirement: compare dynamic
topology-native candidate scoring against a classic fixed-vocabulary recurrent
PPM model.

## Implemented Changes

### Domain Model

- Added `src/domain/models/lstm_baseline.py`.
- Added `LSTMBaseline` with:
  - categorical feature embeddings;
  - numeric feature concatenation;
  - `LSTM` or `GRU` recurrent encoder;
  - `last_node` and `global_mean` prefix readout;
  - linear classifier over `C_train`.
- Registered `LSTM_Baseline` in `src/domain/models/factory.py`.

### CLI / Config

- Extended `src/cli.py::_build_model_factory_kwargs` with:
  - `model.recurrent_type`;
  - `model.recurrent_layers`;
  - `model.recurrent_bidirectional`.
- Added `configs/model/lstm_baseline.yaml`.
- Added UI catalog entries in `configs/ui/config_catalog.yaml`.

### Desktop UI Presets

Added active desktop UI presets to `outputs/ui/experiment_ui_presets.json`:

- `_LSTM-UN-f1-5`
- `_LSTM-UN-f1-5-drift`

They are cloned from the existing complex v1-v5 fixed-vocabulary baseline
scenario and changed to:

```yaml
model.type: LSTM_Baseline
model.model_label: LSTM
model.hidden_dim: 64
model.pooling_strategy: last_node
model.recurrent_type: lstm
model.recurrent_layers: 1
model.recurrent_bidirectional: false
experiment.structural_mode: false
experiment.statistic_enabled: false
experiment.mask_guided_enabled: false
training.learning_strategy: standard
```

The drift preset uses this checkpoint placeholder:

```text
checkpoints/_LSTM-UN-f1-5-42_loan_v1_v5_simulated_LSTM_best.pth
```

Adjust if the actual checkpoint filename differs after the first train run.

### Tests

Added or updated:

- `tests/domain/test_lstm_baseline.py`
- `tests/contracts/test_model_registry_mvp1.py`
- `tests/application/test_cli_flatten_config.py`
- `tests/application/test_experiment_ui_catalog_fields.py`
- `tests/integration/test_training_loops.py`

## Verification

Focused verification passed:

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\domain\test_lstm_baseline.py tests\contracts\test_model_registry_mvp1.py tests\application\test_cli_flatten_config.py::test_build_model_factory_kwargs_passes_lstm_baseline_config tests\application\test_experiment_ui_catalog_fields.py::test_catalog_includes_lstm_baseline_model_fields tests\integration\test_training_loops.py::test_lstm_baseline_isolated_forward_backward_fixed_vocabulary_smoke -v
```

Result:

```text
12 passed, 3 warnings
```

Architecture guard passed:

```powershell
.\.venv-modern\Scripts\python.exe tools\architecture_guard.py
```

Result:

```text
[ARCH_GUARD] OK: architecture dependency rules satisfied.
```

The warnings are the existing PyTorch/PyG deprecation warnings.

## Suggested Run Order

1. Train:

```text
preset_name: _LSTM-UN-f1-5
```

2. Update/check the drift preset checkpoint path if needed.

3. Evaluate drift:

```text
preset_name: _LSTM-UN-f1-5-drift
```

4. Compare against:

```text
_Base-UN-f02-5-drift
_EOPKG_SC-f1-5
_EOPKG_SC-imp-f1-5
```

Primary expected diagnostic behavior:

- LSTM may be competitive on stable seen-version segments.
- LSTM cannot predict activities outside `C_train`.
- Under v3-v5 candidate expansion, `strict_*`, OOS, calibration, and NLL should
  expose the fixed-vocabulary limitation.

## Known Limits

- No full `tests/ -v` run was executed in this session by request to keep the
  implementation focused.
- `LSTM_Baseline` currently treats the PyG batch as an ordered prefix sequence
  through `batch`; it does not use `edge_index`.
- Bidirectional recurrent mode exists for ablation but should not be used as
  the default causal next-activity baseline.

## Documentation Drift Check

Updated:

- `docs/current/project-state.md`
- `docs/GNN_RUNTIME_MVP2_5.MD`
- `configs/ui/config_catalog.yaml`
- `outputs/ui/experiment_ui_presets.json`
