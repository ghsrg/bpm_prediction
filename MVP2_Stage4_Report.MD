# MVP2 Stage 4 Report

Date: 2026-03-12  
Scope: EOPKG models + model factory integration

## 1. Implemented

### 1.1 Model Factory

Created [factory.py](C:\Users\korsr\PycharmProjects\bpm_prediction\src\domain\models\factory.py):

1. Added registry API:
   - `register_model(name)`
   - `create_model(model_type, **kwargs)`
   - `get_registered_models()`
2. Registered baseline models:
   - `BaselineGCN`
   - `BaselineGATv2`
3. Wired EOPKG registration via import side effects from `eopkg_models.py`.

### 1.2 EOPKG model implementations

Created [eopkg_models.py](C:\Users\korsr\PycharmProjects\bpm_prediction\src\domain\models\eopkg_models.py):

1. Added shared base class `BaseEOPKGModel`:
   - local context encoding from `x_cat/x_num/edge_index`
   - pooling (`global_mean` / `last_node`)
   - structural fusion path (`structural_edge_index`, optional `structural_edge_weight`)
   - fallback when structural tensors are missing
2. Added concrete models:
   - `@register_model("EOPKGGCN") class EOPKGGCN`
   - `@register_model("EOPKGGATv2") class EOPKGGATv2`
3. Added critical warning on fallback:
   - `logger.warning("Structural tensors are missing in contract! Falling back to Baseline forward.")`
4. `allowed_target_mask` is not used in model forward (kept for evaluator stage).

### 1.3 CLI integration via factory

Updated [cli.py](C:\Users\korsr\PycharmProjects\bpm_prediction\src\cli.py):

1. Removed hardcoded model registry from CLI.
2. Model initialization now goes only through factory:
   - `create_model(model_type=..., ...)`
3. Kept backward-compatible config behavior:
   - `model.type`
   - fallback to `model.model_type`

## 2. Tests Added

Created [test_eopkg_models_forward.py](C:\Users\korsr\PycharmProjects\bpm_prediction\tests\domain\test_eopkg_models_forward.py):

1. Factory initialization checks for:
   - `EOPKGGCN`
   - `EOPKGGATv2`
2. Forward test with structural tensors:
   - model returns valid logits shape and dtype
3. Fallback test without structural tensors:
   - no crash
   - output shape consistent
   - warning is emitted

## 3. Verification

### 3.1 Targeted tests

Command:

```bash
.\.venv\Scripts\python.exe -m pytest tests/domain/test_eopkg_models_forward.py -v
```

Result:
- `6 passed`

### 3.2 Full test suite

Command:

```bash
.\.venv\Scripts\python.exe -m pytest tests/ -v
```

Result:
- `66 passed, 1 warning`

### 3.3 MVP1 Regression Shield

Command:

```bash
.\.venv\Scripts\python.exe -m pytest -m mvp1_regression -v
```

Result:
- `16 passed, 50 deselected, 1 warning`

================================================================ warnings summary =================================================================
.venv\Lib\site-packages\mlflow\utils\autologging_utils\versioning.py:6
  C:\Users\korsr\PycharmProjects\bpm_prediction\.venv\Lib\site-packages\mlflow\utils\autologging_utils\versioning.py:6: DeprecationWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html
    from pkg_resources import resource_filename

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html

## 4. Run readiness

Stage 4 is complete and safe:

1. Factory supports baseline + EOPKG model creation from config names.
2. EOPKG models consume structural tensors and support safe fallback for MVP1 contracts.
3. MVP1 non-regression shield remains green.

