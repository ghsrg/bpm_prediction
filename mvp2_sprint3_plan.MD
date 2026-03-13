# MVP2 Sprint 3 Plan: EOPKGGATv2 -> Dual-Encoder + Soft Cross-Attention

Date: 2026-03-12  
Status: Planning only (no model code changes in this step)

## 1. Goal

Refactor `EOPKGGATv2` from the current scalar structural shortcut (`num_edges/mean_weight/density`) into a true dual-encoder architecture:

1. Observed branch (`G_obs`) stays GATv2-based.
2. Structural branch (`G_struct`) becomes graph-encoder-based.
3. Fusion is done via soft cross-attention (`Q=H_obs`, `K/V=H_norm`).
4. Keep strict backward compatibility and safe fallback behavior.

## 2. Current State (Gap Analysis)

Current [eopkg_models.py](C:\Users\korsr\PycharmProjects\bpm_prediction\src\domain\models\eopkg_models.py):

1. Uses `struct_projector` over 3 scalar stats.
2. Does not consume node-level structural features (`struct_x`).
3. Has no cross-attention block.
4. `GraphTensorContract` currently has:
   - `structural_edge_index`
   - `structural_edge_weight`
   - but **no explicit structural node tensor** (`struct_x`).

Implication: before dual-encoder can be production-true, we need minimal contract extension + staged fallback path.

## 3. Proposed Architecture (Target for Sprint 3)

## 3.1 Model Config Extension (`model:` in YAML)

Add new params in model config:

1. `struct_encoder_type: str = "GATv2Conv"`
2. `struct_hidden_dim: int = hidden_dim` (default to local hidden dim)
3. `cross_attention_heads: int = 4`

Backward-compatible defaults must preserve runnability for old configs.

## 3.2 Structural Encoder Branch

For `EOPKGGATv2`:

1. Remove `struct_projector`.
2. Add:
   - `self.struct_gnn` selected by `struct_encoder_type`
   - `self.struct_node_emb = nn.Embedding(num_classes, struct_hidden_dim)` as critical fallback.
3. Structural forward policy:
   - If `contract.struct_x` exists: use it as structural node features.
   - Else: use `self.struct_node_emb(torch.arange(num_classes))`.
   - Apply `self.struct_gnn` with `contract.structural_edge_index`.
   - Produce `H_norm` with shape `[num_classes, struct_hidden_dim]`.

## 3.3 Cross-Attention

Add:

1. `self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=cross_attention_heads, batch_first=True)`
2. If `struct_hidden_dim != hidden_dim`, add projection:
   - `self.struct_to_attn = nn.Linear(struct_hidden_dim, hidden_dim)`

Attention tensors:

1. `Q = H_obs.unsqueeze(1)` -> `[B, 1, hidden_dim]`
2. `K = V = H_norm_attn.unsqueeze(0).expand(B, -1, -1)` -> `[B, C, hidden_dim]`
3. `context = cross_attention(Q, K, V)[0].squeeze(1)` -> `[B, hidden_dim]`

## 3.4 Fusion Head

For classifier path:

1. `fusion_input = torch.cat([H_obs, context], dim=-1)` -> `[B, 2*hidden_dim]`
2. `self.fusion = nn.Sequential(nn.Linear(hidden_dim + hidden_dim, hidden_dim), nn.ReLU())`
3. `logits = self.classifier(self.fusion(fusion_input))`

Note: user request says `hidden_dim + struct_hidden_dim`; with cross-attention output in `hidden_dim`, practical fusion is `hidden_dim + hidden_dim`.  
If strict requirement is to concatenate with non-projected structural context, we can alternatively skip `struct_to_attn` and set attention embed dim to `struct_hidden_dim`, but then local branch needs projection too. Recommended path is projection to `hidden_dim` for stable shape contracts.

## 4. Data Contract & Builder Impact

## 4.1 Required Contract Extension

Extend [tensor_contract.py](C:\Users\korsr\PycharmProjects\bpm_prediction\src\domain\entities\tensor_contract.py) with:

1. `struct_x: Optional[torch.FloatTensor] = None` (or `torch.LongTensor` if embedding indices design is selected)

This is needed for future EOPKG node attributes (resources/policies/statistics).

## 4.2 DynamicGraphBuilder (Staged)

Stage A (Sprint 3 immediate):

1. Keep generating `structural_edge_index/weight`.
2. Leave `struct_x=None` so model uses `struct_node_emb` fallback.

Stage B (next sub-phase):

1. Build real `struct_x` from knowledge store:
   - node priors/statistics
   - resource distribution embeddings
   - policy features

## 5. Future EOPKG Graph DB Compatibility (Design Notes)

To support “graph DB with rich node/edge attrs” without rewrites:

1. Introduce adapter-level DTO for structural graph snapshot:
   - `nodes: [{activity_id, resource_stats, policy_flags, ...}]`
   - `edges: [{src, dst, weight, transition_freq, sla_penalty, ...}]`
2. Add structural feature assembler module:
   - maps DTO -> `struct_x`, `structural_edge_index`, `structural_edge_weight`
3. Keep model API stable: model only sees tensors, not DB objects.
4. Version graph schema explicitly (`kg_schema_version`) to avoid silent tensor drift.

## 6. Implementation Steps (Atomic)

## Step 1. Config plumbing

Files:

1. [cli.py](C:\Users\korsr\PycharmProjects\bpm_prediction\src\cli.py)
2. [factory.py](C:\Users\korsr\PycharmProjects\bpm_prediction\src\domain\models\factory.py) (if needed for constructor kwargs consistency)
3. MVP2 experiment configs under `configs/experiments/`

Tasks:

1. Read `model.struct_encoder_type`, `model.struct_hidden_dim`, `model.cross_attention_heads`.
2. Pass them into model constructor.
3. Keep defaults if missing.

Tests:

1. Config constructor test for defaults.
2. Factory init test with explicit overrides.

## Step 2. Contract extension (non-breaking)

Files:

1. [tensor_contract.py](C:\Users\korsr\PycharmProjects\bpm_prediction\src\domain\entities\tensor_contract.py)
2. [trainer.py](C:\Users\korsr\PycharmProjects\bpm_prediction\src\application\use_cases\trainer.py) (`_data_to_contract`)

Tasks:

1. Add optional `struct_x`.
2. Ensure absent field does not break MVP1 and existing MVP2 tests.

Tests:

1. Contract optional-field compatibility regression.

## Step 3. EOPKGGATv2 refactor to Dual-Encoder

Files:

1. [eopkg_models.py](C:\Users\korsr\PycharmProjects\bpm_prediction\src\domain\models\eopkg_models.py)

Tasks:

1. Remove legacy `struct_projector` path in EOPKGGATv2.
2. Add `struct_gnn`, `struct_node_emb`, `cross_attention`, `fusion`.
3. Add strict tensor shape assertions in debug mode.
4. Keep warning-based fallback if structural tensors are absent.

Tests:

1. Forward with real structural edge tensors.
2. Forward with `struct_x=None` fallback to embedding.
3. Shape validation for attention (batch-first).

## Step 4. EOPKGGCN parity decision

Decision point:

1. Either keep `EOPKGGCN` on legacy fusion temporarily.
2. Or align it to same dual-encoder interface.

Recommendation:

1. Keep interface parity now (accept same config args).
2. Full dual-encoder parity can be phase 3.2 if needed.

Tests:

1. Ensure factory + forward still pass for both EOPKG classes.

## Step 5. Forward smoke + non-regression

Run:

1. New targeted tests for dual-encoder forward.
2. `pytest tests/ -v`
3. `pytest -m mvp1_regression -v`

Acceptance:

1. No tensor shape mismatches.
2. No MVP1 regression.
3. EOPKG fallback path still works when structural inputs are partial.

## 7. Risks & Mitigations

1. **Risk:** attention shape mismatch (`hidden_dim` vs `struct_hidden_dim`).
   - Mitigation: explicit projection layer and unit tests with non-equal dims.
2. **Risk:** contract drift (`struct_x` absent in many pipelines).
   - Mitigation: mandatory fallback to `struct_node_emb`.
3. **Risk:** silent overfitting from always-available class embeddings.
   - Mitigation: dropout + ablation tests vs baseline.
4. **Risk:** future EOPKG node attributes incompatible with current tensorizer.
   - Mitigation: separate structural feature assembler module now (interface-first).

## 8. Open Clarifications Before Coding

1. Should dual-encoder refactor be applied only to `EOPKGGATv2` now, or both EOPKG models in same PR?
2. Preferred `struct_x` dtype:
   - float feature tensor (`FloatTensor`) or
   - categorical node IDs + embedding (`LongTensor`)?
3. For attention, do we enforce `struct_hidden_dim == hidden_dim` in config, or always use projection layer?
4. Should we expose attention weights as debug metrics/artifacts in this sprint?

