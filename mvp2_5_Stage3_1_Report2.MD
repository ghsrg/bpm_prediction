# MVP2.5 Stage 3.1 Report (v2)

Date: 2026-03-13
Status: Completed and validated

## Done in this iteration

1. Added 4 experiment-only configs (include data + model only):
- `configs/experiments/mvp2_5_stage3_1_baseline_files.yaml`
- `configs/experiments/mvp2_5_stage3_1_baseline_sql.yaml`
- `configs/experiments/mvp2_5_stage3_1_eopkg_files.yaml`
- `configs/experiments/mvp2_5_stage3_1_eopkg_sql.yaml`

2. Added dedicated data configs with mapping + features + policies:
- `configs/data/mvp2_5_stage3_1_camunda_files.yaml`
- `configs/data/mvp2_5_stage3_1_camunda_sql.yaml`

3. Added new IXESAdapter-compatible Camunda bridge:
- `src/adapters/ingestion/camunda_trace_adapter.py`

4. Updated CLI wiring to auto-select data adapter by config:
- `src/cli.py`
- Supports `mapping.adapter: "camunda"` (or presence of `mapping.camunda_adapter`)
- Keeps old XES flow unchanged
- `prepare_data` now accepts optional adapter and supports non-XES sources

5. Added tests for new adapter and CLI smoke path:
- `tests/adapters/test_camunda_trace_adapter.py`
- `tests/integration/test_cli_camunda_adapter_smoke.py`

## Config behavior

- Model configs remain unchanged and reusable:
  - `../model/gatv2_baseline.yaml`
  - `../model/gatv2_eopkg.yaml`
- New data configs contain comments for all new `camunda` fields and defaults.
- `version_key` is optional (`""` means all versions).

## How to run

1. Baseline + Files:
```bash
python main.py --config configs/experiments/mvp2_5_stage3_1_baseline_files.yaml
```

2. EOPKG + Files:
```bash
python main.py --config configs/experiments/mvp2_5_stage3_1_eopkg_files.yaml
```

3. Baseline + SQL:
```bash
python main.py --config configs/experiments/mvp2_5_stage3_1_baseline_sql.yaml
```

4. EOPKG + SQL:
```bash
python main.py --config configs/experiments/mvp2_5_stage3_1_eopkg_sql.yaml
```

## Validation results

1. New focused tests:
```bash
python -m pytest tests/adapters/test_camunda_trace_adapter.py tests/integration/test_cli_camunda_adapter_smoke.py -v
```
Result: `3 passed`

2. Full test suite:
```bash
python -m pytest tests/ -v
```
Result: `106 passed`

3. MVP1 regression shield:
```bash
python -m pytest -m mvp1_regression -v
```
Result: `16 passed`

4. Runtime smoke runs with new experiment configs:
- `mvp2_5_stage3_1_baseline_files.yaml`: finished successfully
- `mvp2_5_stage3_1_eopkg_files.yaml`: finished successfully

## Notes

1. SQL configs are ready; actual run requires valid `mssql.connection_string` and driver.
2. Trainer log message still says `Reading XES file...` for historical compatibility, but adapter selection is now data-driven.
