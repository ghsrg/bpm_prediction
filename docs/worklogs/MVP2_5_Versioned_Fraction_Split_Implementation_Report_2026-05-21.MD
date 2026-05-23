# MVP2.5 Versioned Fraction and Split Implementation Report

## Summary

Implemented version-aware experiment data selection for topology-conditioned
training:

- `experiment.fraction_strategy: temporal | versioned`
- `experiment.split_strategy: temporal | none | versioned`
- `experiment.version_scope_policy: all | train_cut`

The default remains backward-compatible:

```yaml
experiment.fraction_strategy: temporal
experiment.split_strategy: temporal
experiment.version_scope_policy: all
```

## What Changed

### Versioned Fraction

`fraction_strategy=versioned` samples `experiment.fraction` independently per
`process_version`.

Example:

```text
v1: 100 traces, fraction=0.2 -> 20 traces
v2: 100 traces, fraction=0.2 -> 20 traces
```

For versioned fraction, each non-empty version keeps at least one trace when
`fraction > 0`.

### Versioned Split

`split_strategy=versioned` applies `experiment.split_ratio` inside each selected
version, then concatenates per-version train/val/test slices.

Example:

```yaml
experiment.split_ratio: [0.5, 0.2, 0.3]
```

For each version:

```text
50% -> train
20% -> validation
30% -> test
```

### Train-Cut Version Scope

`version_scope_policy=train_cut` prevents future-version leakage:

- `mode=train`: uses traces whose versions are known before the chronological
  `train_ratio` cut;
- `mode=eval_drift` / `eval_cross_dataset`: uses future tail versions after the
  cut.

This is intended for controlled topology-conditioned experiments where train
must include multiple known versions but must not learn from future versions.

## Recommended Fast Topology-Conditioned Config

```yaml
experiment:
  fraction: 0.2
  fraction_strategy: versioned
  split_strategy: versioned
  version_scope_policy: train_cut
  train_ratio: 0.5
  split_ratio: [0.5, 0.2, 0.3]
training:
  learning_strategy: topology_conditioned
```

Expected behavior:

- train/validation uses known versions before the train cut;
- future versions are not leaked into train;
- small fractions still preserve per-version coverage;
- `train_topology_conditioned_wrong_version_ce_delta` can be produced when at
  least two known train versions are present.

## Tests Added

Added coverage in `tests/application/test_cascade_data_split.py`:

- versioned fraction keeps representatives from each version;
- versioned micro split applies `split_ratio` per version;
- `train_cut` excludes future versions from train preparation;
- `train_cut` eval drift uses future tail only.

Added coverage in `tests/application/test_experiment_ui_catalog_fields.py`:

- UI catalog exposes `experiment.fraction_strategy`;
- UI catalog exposes `experiment.version_scope_policy`;
- UI catalog exposes `split_strategy=versioned`.

Updated `tests/contracts/test_mvp1_config_contract.py`:

- `versioned` is accepted as a valid split strategy.

## Verification

Targeted verification:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\application\test_cascade_data_split.py tests\application\test_experiment_ui_catalog_fields.py tests\contracts\test_mvp1_config_contract.py tests\application\test_cli_flatten_config.py -v
```

Result:

```text
30 passed, 11 warnings
```

Architecture guard:

```powershell
.\.venv\Scripts\python.exe tools\architecture_guard.py
```

Result:

```text
[ARCH_GUARD] OK: architecture dependency rules satisfied.
```

Full verification:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\ -v
```

Result:

```text
428 passed, 11 warnings
```

MVP1 regression:

```powershell
.\.venv\Scripts\python.exe -m pytest -m mvp1_regression -v
```

Result:

```text
20 passed, 408 deselected, 11 warnings
```

## Architecture Debt

Added active debt:

```text
split_fraction_policy_duplication
```

Reason: split/fraction behavior is currently mirrored in `src/cli.py` and
`src/application/use_cases/trainer.py`. Long-term target is one application
service for experiment trace selection.
