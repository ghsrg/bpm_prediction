# MVP2 Sprint 3 Step 3 Report

Date: 2026-03-12  
Scope: Dual-Encoder refactor for `EOPKGGATv2`

## Done

Refactored [eopkg_models.py](C:\Users\korsr\PycharmProjects\bpm_prediction\src\domain\models\eopkg_models.py) for `EOPKGGATv2` only:

1. Replaced old scalar structural shortcut with structural encoder branch.
2. Added new constructor args:
   - `struct_encoder_type`
   - `struct_hidden_dim`
   - `cross_attention_heads`
3. Added structural fallback embedding:
   - `self.struct_node_emb = nn.Embedding(num_classes, struct_hidden_dim)`
   - used when `contract.struct_x` is missing.
4. Added structural encoder selection:
   - `GATv2Conv` (default)
   - `GCNConv`
5. Added attention stack:
   - `self.cross_attention = nn.MultiheadAttention(..., batch_first=True)`
   - `Q = H_obs.unsqueeze(1)`; `K/V = H_norm` expanded to batch.
6. Added projection safety:
   - `struct_to_attn` for `struct_hidden_dim -> hidden_dim`
   - `attn_to_struct` for `hidden_dim -> struct_hidden_dim`
7. Added fusion head:
   - `Linear(hidden_dim + struct_hidden_dim, hidden_dim) + ReLU`
8. Added XAI hook:
   - `self.last_cross_attn_weights = attn_weights.detach()`
9. Preserved critical fallback:
   - if `structural_edge_index` absent -> warning + baseline logits path.

## Notes

1. `EOPKGGCN` was intentionally left on the previous logic (scope decision).
2. `struct_x` is treated as `FloatTensor`.
3. If `struct_x` has mismatched node count, model logs warning and falls back to `struct_node_emb`.

## Verification

Command:

```bash
.\.venv\Scripts\python.exe -m pytest tests/domain/test_eopkg_models_forward.py -v
```

Result:
- `6 passed`

