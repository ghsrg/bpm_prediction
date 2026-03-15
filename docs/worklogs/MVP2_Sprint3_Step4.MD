# MVP2 Sprint 3 Step 4 Report

Date: 2026-03-12  
Scope: Dual-Encoder tests and validation surface for `EOPKGGATv2`

## Done

Added targeted tests for new architecture behavior:

1. [test_eopkg_gatv2_dual_encoder.py](C:\Users\korsr\PycharmProjects\bpm_prediction\tests\domain\test_eopkg_gatv2_dual_encoder.py)
   - validates attention weights are produced and stored
   - validates `struct_x: FloatTensor` path with projection (`struct_hidden_dim != hidden_dim`)
   - validates fallback without structural edges (warning + stable logits)

## Covered scenarios

1. Attention output and shape:
   - `last_cross_attn_weights` expected shape `[B, heads, 1, C]`
2. Structural feature projection:
   - `struct_x` with custom feature dimension is projected to structural hidden size
3. Critical fallback:
   - no structural graph -> baseline forward, no crash

## Verification

Commands:

```bash
.\.venv\Scripts\python.exe -m pytest tests/domain/test_eopkg_gatv2_dual_encoder.py -v
.\.venv\Scripts\python.exe -m pytest tests/domain/test_eopkg_models_forward.py -v
```

Results:

1. `test_eopkg_gatv2_dual_encoder.py`: `3 passed`
2. `test_eopkg_models_forward.py`: `6 passed`

Notes:

1. Torch emitted `LazyLinear` warning (expected due `nn.LazyLinear` usage in structural input projector).

