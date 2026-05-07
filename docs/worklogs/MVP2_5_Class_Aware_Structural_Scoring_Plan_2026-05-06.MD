# Class-Aware Structural Scoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make BPMN/topology structure affect the ranking of each candidate next activity directly, instead of being compressed into one global structural context vector.

**Architecture:** Keep the existing observed prefix encoder and structural topology encoder, but add a new class-aware fusion mode that produces one structural score per activity class. The final logits become `observed_logits + gated_structural_logits`, so the structural graph can promote or suppress concrete candidate activities.

**Tech Stack:** Python, PyTorch, PyTorch Geometric, `EOPKGGATv2`, MLflow metrics, YAML configs, pytest.

---

## Metadata

- `status`: plan
- `date`: 2026-05-06
- `scope`: MVP2.5 GNN runtime
- `primary_file`: `src/domain/models/eopkg_models.py`
- `related_docs`:
  - `docs/GNN_RUNTIME_MVP2_5.MD`
  - `docs/current/architecture-debt.md`
  - `configs/ui/config_catalog.yaml`
- `related_runs`:
  - `ef26fbcc4c644167a04dc82d9e46f197`: `Base-UN-drift-42`
  - `a5adf68e520943b2a27fac3a523d5d5c`: `S_Att-UNc-drift-42`
  - `4e970e1da36c4108bbc189506e470b4e`: `Base-UN-42`
  - `764b80165e23478a9fd0d770e7d6f0fe`: `S_Att-UNc-42`

---

## Problem Summary

Поточна `EOPKGGATv2` має structural node representations, але не використовує їх як per-candidate scoring layer.

Current flow:

```text
prefix graph -> observed GATv2 -> obs_context [B, H]

topology graph -> structural GNN -> h_node [|V|, D]

Attention mode:
  obs_context attends over h_node
  -> one context_struct [B, D]
  -> concat(obs_context, context_struct)
  -> classifier
  -> logits [B, C]

Concat mode:
  mean(h_node) -> one context_struct [B, D]
  -> classifier
  -> logits [B, C]
```

This means topology influences logits only indirectly through one compressed vector. It does not score candidate activity `j` through an explicit node-to-class projection from structural graph nodes to activity classes.

Observed symptom:

- `S_Att-UNc` improves non-strict/hybrid `macro_f1`.
- `S_Att-UNc` reduces OOS and improves `pred_in_mask_rate`.
- `S_Att-UNc` can be worse on `strict_macro_f1`.
- `strict_error_but_allowed_rate` increases, meaning the model often predicts a structurally allowed activity that is not the exact logged next activity.

Interpretation:

Структура вже допомагає моделі залишатися у допустимому процесному просторі, але поточна архітектура не дає структурі прямого механізму ранжувати конкретні activity classes.

---

## Current Code Evidence

Current structural attention in `src/domain/models/eopkg_models.py`:

```python
h_norm = self.struct_gnn(struct_nodes, structural_edge_index)
h_norm = self.activation(h_norm)
h_norm = self.dropout(h_norm)

query = obs_context.unsqueeze(1)  # [B, 1, hidden_dim]
kv = self.struct_to_attn(h_norm).unsqueeze(0).expand(batch_size, -1, -1)  # [B, C, hidden_dim]
attn_output, attn_weights = self.cross_attention(
    query=query,
    key=kv,
    value=kv,
    need_weights=True,
    average_attn_weights=False,
)
context_struct = self.attn_to_struct(attn_output.squeeze(1))  # [B, struct_hidden_dim]
fused = self.fusion(torch.cat([obs_context, context_struct], dim=1))
return self.classifier(fused)
```

The attention weights are per candidate, but their output is compressed to one vector before classification. Candidate-specific structural information is not preserved as one score per class.

---

## Design Alternatives

### Option A: Additive Class-Aware Structural Logits With Scaled Cosine

Formula:

```text
H_v = GNN_struct(X_s, E_s)                         # [|V|, D]
q_i = W_q o_i                                      # [D]
z_v = W_k H_v[v]                                   # [D]
l_obs(i, j) = classifier(o_i)[j]
l_node(i, v) = cosine(q_i, z_v) / tau
l_struct(i, j) = aggregate({l_node(i, v) | class(v) = j})
l_final(i, j) = l_obs(i, j) + gamma * l_struct(i, j)
```

Where:

- `o_i`: observed prefix context for sample `i`.
- `H_v[v]`: structural representation of BPMN/topology node `v`.
- `class(v)`: activity-vocabulary class index for structural node `v`; non-activity nodes use `-1`.
- `gamma`: learnable or config-controlled structural logit scale; initialize conservatively at `0.1`.
- `tau`: temperature, default `sqrt(D)` or a config scalar.

Pros:

- Directly aligns topology with class ranking.
- Minimal change to current model.
- Keeps current classifier and observed branch intact.
- Can be ablated by setting `gamma=0`.
- Works without stats because trainable structural node embeddings still encode topology after structural GNN.
- Handles `|V| != C` through an explicit node-to-class projection before adding structural logits to class logits.
- Computationally stable because L2-normalization bounds the structural score.
- Similar in spirit to contrastive scoring, so the structural branch is easy to reason about.

Cons:

- Less expressive than a bilinear interaction because the metric is fixed after projection.
- Requires a correct `struct_node_to_class_index` contract.
- Still cannot infer branch probabilities from topology alone.
- Parallel/ambiguous regions may remain strict-F1 hard without stats.

Recommendation: implement this first. Keep `gamma` / `structural_logit_scale`
initialized at `0.1` to protect the observed-logit baseline from early
structural degradation.

### Node-To-Class Projection Decision

`struct_gnn` is node-level and returns `[|V|, D]`, while next-activity
prediction is class-level and returns `[B, C]`. Therefore
`ClassAwareStructuralScoring` must not assume `|V| == C`.

Two projection options exist:

```text
Early Pooling:
  h_class_j = mean({h_node_v | class(v) = j})
  structural_logits = score(obs_context, h_class_j)
```

```text
Late Pooling:
  node_logits_v = score(obs_context, h_node_v)
  structural_logits_j = logsumexp({node_logits_v | class(v) = j})
```

Decision:

Use **Late Pooling** as the canonical implementation. It is more appropriate
for duplicate activity labels because different BPMN nodes can represent the
same activity label in different branches. The model should score those nodes
separately before the result is projected back to the activity vocabulary.

Early Pooling remains a possible future ablation, but should not be the default
because it merges branch-specific structural evidence before scoring.

### Scoring Escalation Decision

Implementation order:

```text
1. Scaled cosine structural scoring
2. Bilinear structural scoring
3. Candidate MLP structural scoring
```

Do not implement bilinear scoring in the first pass unless the user explicitly
changes the experiment goal. The first pass should answer the narrow question:
does topology help when it is wired into class-level logits without adding much
capacity?

Escalate from scaled cosine to bilinear only if:

- `structural_to_observed_logit_ratio` shows the structural branch is active but
  not dominant;
- OOS / mask-admissibility improves or remains stable;
- `strict_test_macro_f1` does not improve enough versus the structural baseline.

Bilinear scoring is the preferred second step because it learns a controlled
metric:

```text
l_node(i, v) = q_i^T W_bilinear z_v
```

This is more expressive than cosine similarity but less over-parameterized than
an MLP candidate scorer. Candidate MLP scoring is reserved for a later ablation
after bilinear has been tested.

### Option B: Bilinear Class-Aware Structural Scoring

Formula:

```text
q_i = W_q o_i
z_v = W_k H_v[v]
l_node(i, v) = q_i^T W_bilinear z_v
l_struct(i, j) = aggregate({l_node(i, v) | class(v) = j})
l_final(i, j) = l_obs(i, j) + gamma * l_struct(i, j)
```

Pros:

- Stronger interaction between prefix state and candidate structure.
- More expressive than scaled cosine because the similarity metric is learned.
- Still controlled: one bilinear matrix is easier to regularize and interpret than a full MLP scorer.
- Preserves observed-logit baseline and additive structural-logit semantics.

Cons:

- More parameters and overfitting risk than scaled cosine.
- Needs regularization and the same structural-logit contribution diagnostics.

Use only if Option A shows structural signal is present but strict-F1 does not
improve enough. Do not jump directly to MLP before trying bilinear scoring.

### Option C: Replace Classifier With Full Candidate MLP Scorer

Formula:

```text
l_node(i, v) = MLP([o_i, H_v[v], o_i * H_v[v]])
l_final(i, j) = aggregate({l_node(i, v) | class(v) = j})
```

Pros:

- Strongest interaction between prefix state and candidate structure.
- Can model non-linear relations that cosine/bilinear forms cannot.

Cons:

- Larger architectural change.
- Highest overfitting risk among the scoring options.
- Harder to compare with current baseline because classifier semantics change more deeply.

Use later only if both scaled cosine and bilinear scoring are insufficient.

### Option D: Keep Global Attention And Add Candidate Bias

Formula:

```text
context_struct = attention(o_i, H_node)
l_final = classifier([o_i, context_struct]) + bias_struct(H_node)
```

Pros:

- Smallest edit to current code.

Cons:

- `bias_struct(H_node)` is not prefix-conditioned.
- Does not solve the main problem: topology is still mostly global context.

Not recommended.

---

## Recommended Design

Add a new canonical `model.fusion_mode` value:

```yaml
model:
  fusion_mode: ClassAwareStructuralScoring
```

Normalize fusion mode names so old configs continue to work and new
experiments have precise semantics:

```text
Attention -> ClassMeanAttention
Concat -> ClassMeanConcat
struct_pool_concat -> ClassMeanConcat
ClassAwareAdditive -> ClassAwareStructuralScoring
ClassAwareAttention -> ClassAwareStructuralScoring
```

Naming rule:

`ClassAwareStructuralScoring` is not classical attention. It does not compute
one attended structural context vector from `Q=obs_context`, `K=h_node[v]`,
and `V=h_node[v]`. It computes prefix-conditioned structural logits for
structural nodes, projects them to candidate classes, and adds that vector to
observed logits.

Behavior:

1. Observed branch stays unchanged:

```text
obs_context = pool(local_gnn(prefix_graph))
observed_logits = classifier(obs_context)
```

2. Structural branch computes per-node structural embeddings:

```text
struct_nodes = struct_x if valid else struct_node_emb(arange(num_struct_nodes))
h_node = struct_gnn(struct_nodes, structural_edge_index)  # [|V|, D]
```

3. New class-aware scorer first produces `[B, |V|]` node logits:

```text
q = normalize(W_q(obs_context))          # [B, D]
k = normalize(W_k(h_node))               # [|V|, D]
node_structural_logits = q @ k.T         # [B, |V|]
```

4. Node logits are projected to class logits through `struct_node_to_class_index`:

```text
struct_node_to_class_index[v] = j         # activity node v maps to class j
struct_node_to_class_index[v] = -1        # gateway/start/end or unmapped node

structural_logits[:, j] =
  logsumexp({node_structural_logits[:, v] | struct_node_to_class_index[v] = j})
```

Default aggregation is **Late Pooling** with `logsumexp`. It scores each
structural node first and only then aggregates duplicate activity nodes into the
activity vocabulary class. This preserves branch-specific evidence better than
averaging node embeddings before scoring.

5. Final logits:

```text
logits = observed_logits + gamma * structural_logits
```

6. Mask-guided learning remains separate:

```text
effective_logits = apply_mask_policy(logits, allowed_target_mask)
loss = cross_entropy(effective_logits, y)
```

This is important: class-aware structural scoring is not a mask. It is a learned structural prior over classes.

---

## Implementation Tasks

### Task 1: Add Tests For Fusion Mode Normalization And New Mode

**Files:**

- Modify: `tests/domain/test_eopkg_gatv2_dual_encoder.py`

- [ ] **Step 1: Add test for shape and diagnostics**

Add:

```python
def test_eopkg_gatv2_class_aware_structural_scoring_returns_per_class_structural_logits():
    model = create_model(
        model_type="EOPKGGATv2",
        feature_layout=_layout(),
        hidden_dim=16,
        output_dim=7,
        struct_hidden_dim=16,
        cross_attention_heads=4,
        dropout=0.0,
        pooling_strategy="global_mean",
        fusion_mode="ClassAwareStructuralScoring",
    )
    contract = {
        **_base_contract(),
        "structural_edge_index": _struct_edge_index(),
        "struct_node_to_class_index": torch.arange(7, dtype=torch.long),
    }

    logits = model(contract)

    assert tuple(logits.shape) == (1, 7)
    assert model.last_cross_attn_weights is None
    assert model.last_structural_class_logits is not None
    assert tuple(model.last_structural_class_logits.shape) == (1, 7)
    assert not any(isinstance(param, UninitializedParameter) for param in model.parameters())
```

- [ ] **Step 1a: Add test for `|V| > C` duplicate activity projection**

Add a test where structural tensors contain more nodes than output classes:

```python
def test_eopkg_gatv2_class_aware_structural_scoring_projects_nodes_to_classes():
    model = create_model(
        model_type="EOPKGGATv2",
        feature_layout=_layout(),
        hidden_dim=16,
        output_dim=3,
        struct_hidden_dim=16,
        dropout=0.0,
        pooling_strategy="global_mean",
        fusion_mode="ClassAwareStructuralScoring",
    )
    contract = {
        **_base_contract(output_dim=3),
        "structural_edge_index": _struct_edge_index(num_nodes=6),
        "struct_node_to_class_index": torch.tensor([0, 1, 1, 2, -1, 2], dtype=torch.long),
    }

    logits = model(contract)

    assert tuple(logits.shape) == (1, 3)
    assert tuple(model.last_structural_node_logits.shape) == (1, 6)
    assert tuple(model.last_structural_class_logits.shape) == (1, 3)
```

- [ ] **Step 1b: Add test that missing node-to-class mapping fails when `|V| != C`**

Add:

```python
def test_eopkg_gatv2_class_aware_structural_scoring_requires_node_to_class_mapping():
    model = create_model(
        model_type="EOPKGGATv2",
        feature_layout=_layout(),
        hidden_dim=16,
        output_dim=3,
        struct_hidden_dim=16,
        dropout=0.0,
        pooling_strategy="global_mean",
        fusion_mode="ClassAwareStructuralScoring",
    )
    contract = {
        **_base_contract(output_dim=3),
        "structural_edge_index": _struct_edge_index(num_nodes=6),
    }

    with pytest.raises(ValueError, match="struct_node_to_class_index"):
        model(contract)
```

- [ ] **Step 2: Add test for fallback compatibility**

Add:

```python
def test_eopkg_gatv2_class_aware_structural_scoring_falls_back_without_structural_edges(
    caplog: pytest.LogCaptureFixture,
):
    model = create_model(
        model_type="EOPKGGATv2",
        feature_layout=_layout(),
        hidden_dim=16,
        output_dim=7,
        struct_hidden_dim=16,
        dropout=0.0,
        pooling_strategy="global_mean",
        fusion_mode="ClassAwareStructuralScoring",
    )

    with caplog.at_level(logging.WARNING):
        logits = model(_base_contract())

    assert tuple(logits.shape) == (1, 7)
    assert model.last_cross_attn_weights is None
    assert model.last_structural_class_logits is None
    assert "Structural tensors are missing in contract! Falling back to Baseline forward." in caplog.text
```

- [ ] **Step 3: Add tests that existing names normalize to canonical modes**

Add:

```python
@pytest.mark.parametrize(
    ("raw_fusion_mode", "expected"),
    [
        ("Attention", "class_mean_attention"),
        ("Concat", "class_mean_concat"),
        ("struct_pool_concat", "class_mean_concat"),
        ("ClassAwareAdditive", "class_aware_structural_scoring"),
        ("ClassAwareAttention", "class_aware_structural_scoring"),
        ("ClassAwareStructuralScoring", "class_aware_structural_scoring"),
    ],
)
def test_eopkg_gatv2_normalizes_fusion_mode_aliases(raw_fusion_mode: str, expected: str):
    model = create_model(
        model_type="EOPKGGATv2",
        feature_layout=_layout(),
        hidden_dim=16,
        output_dim=7,
        struct_hidden_dim=16,
        dropout=0.0,
        pooling_strategy="global_mean",
        fusion_mode=raw_fusion_mode,
    )

    assert model.fusion_mode == expected
```

- [ ] **Step 4: Existing behavior tests must stay compatible**

Existing tests for `Attention` and fallback must continue to pass without behavior changes.

- [ ] **Step 5: Run failing tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/domain/test_eopkg_gatv2_dual_encoder.py -v
```

Expected before implementation:

```text
FAILED ... AttributeError or ValueError for ClassAwareStructuralScoring
```

---

### Task 2: Implement Class-Aware Structural Scoring Mode

**Files:**

- Modify: `src/domain/models/eopkg_models.py`
- Modify: `src/domain/entities/tensor_contract.py`
- Modify: `src/domain/services/dynamic_graph_builder.py`
- Modify: contract/builder tests that validate optional structural fields.

- [ ] **Step 1: Extend `fusion_mode` alias map and canonical names**

Change:

```python
alias_map = {
    "attention": "attention",
    "concat": "concat",
    "concat_mlp": "attention",
    "struct_pool_concat": "concat",
}
```

To:

```python
alias_map = {
    "attention": "class_mean_attention",
    "classmeanattention": "class_mean_attention",
    "class_mean_attention": "class_mean_attention",
    "concat": "class_mean_concat",
    "classmeanconcat": "class_mean_concat",
    "class_mean_concat": "class_mean_concat",
    "concat_mlp": "class_mean_attention",
    "struct_pool_concat": "class_mean_concat",
    "classawareadditive": "class_aware_structural_scoring",
    "class_aware_additive": "class_aware_structural_scoring",
    "classawareattention": "class_aware_structural_scoring",
    "class_aware_attention": "class_aware_structural_scoring",
    "classawarestructuralscoring": "class_aware_structural_scoring",
    "class_aware_structural_scoring": "class_aware_structural_scoring",
    "class_aware": "class_aware_structural_scoring",
    "candidate_scoring": "class_aware_structural_scoring",
}
```

Then update validation:

```python
if self.fusion_mode not in {"class_mean_attention", "class_mean_concat", "class_aware_structural_scoring"}:
    raise ValueError(
        f"Unsupported model.fusion_mode '{self.fusion_mode}'. "
        "Available: ['ClassMeanAttention', 'ClassMeanConcat', 'ClassAwareStructuralScoring']"
    )
```

- [ ] **Step 1a: Add `struct_node_to_class_index` to the graph tensor contract**

Add an optional tensor field:

```text
struct_node_to_class_index: LongTensor[|V|]
```

Contract semantics:

- length equals the number of structural nodes used by `struct_x` and
  `structural_edge_index`;
- value `j >= 0` maps structural node `v` to activity-vocabulary class `j`;
- value `-1` marks gateways, events, and non-target nodes;
- values `>= output_dim` are invalid.

The dynamic graph builder/topology projection layer must populate this field
from structural node metadata and the active activity vocabulary. This field is
required for `ClassAwareStructuralScoring` whenever `|V| != C`.

- [ ] **Step 1b: Make structural node feature building node-count aware**

Update `_build_struct_node_features()` so it accepts `struct_x` with row count
`|V|`, not only `num_classes`.

Rules:

- if `struct_x` is present, use its actual row count as the structural node
  count;
- if `struct_x` is absent and structural edges imply `|V| == C`, the existing
  class embedding fallback may be used;
- if `struct_x` is absent and structural edges imply `|V| != C`, raise a clear
  error for `ClassAwareStructuralScoring` because there is no valid feature
  source for extra structural nodes;
- do not silently truncate or pad `struct_x`.

- [ ] **Step 2: Add scorer modules**

In `__init__`, after `self.cross_attention`:

```python
self.struct_query_proj = (
    nn.Identity() if hidden_dim == self.struct_hidden_dim else nn.Linear(hidden_dim, self.struct_hidden_dim)
)
self.struct_key_proj = nn.Identity()
self.structural_logit_scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
self.last_observed_logits: torch.Tensor | None = None
self.last_structural_node_logits: torch.Tensor | None = None
self.last_structural_class_logits: torch.Tensor | None = None
```

Rationale:

- `struct_query_proj` maps observed prefix context to structural embedding dimension.
- `struct_key_proj` is explicit for future extension.
- `structural_logit_scale` starts at `0.1` so the structural branch cannot dominate the observed baseline at initialization.
- `last_observed_logits`, `last_structural_node_logits`, and `last_structural_class_logits` allow contribution diagnostics and tests.

If a config key is added for this value, use:

```yaml
model:
  structural_logit_scale_init: 0.1
```

and add it to `configs/ui/config_catalog.yaml`.

- [ ] **Step 3: Add node-to-class projection helper**

Add:

```python
def _resolve_struct_node_to_class_index(
    self,
    *,
    contract: Mapping[str, Any],
    num_struct_nodes: int,
    num_classes: int,
    device: torch.device,
) -> torch.Tensor:
    node_to_class = contract.get("struct_node_to_class_index")
    if node_to_class is None:
        if num_struct_nodes == num_classes:
            return torch.arange(num_classes, dtype=torch.long, device=device)
        raise ValueError(
            "ClassAwareStructuralScoring requires struct_node_to_class_index "
            f"when num_struct_nodes={num_struct_nodes} differs from num_classes={num_classes}."
        )

    node_to_class = node_to_class.to(device=device, dtype=torch.long)
    if node_to_class.numel() != num_struct_nodes:
        raise ValueError(
            "struct_node_to_class_index length must match the number of structural nodes."
        )
    if torch.any(node_to_class >= num_classes):
        raise ValueError("struct_node_to_class_index contains class ids outside output_dim.")
    return node_to_class
```

Rules:

- `-1` means the node is not an activity class target and is ignored during
  class-logit aggregation.
- Gateways/events should normally be `-1`: they participate in message passing
  but are not direct prediction classes.
- If `|V| == C` and mapping is missing, use identity mapping only as a backward
  compatibility fallback.

- [ ] **Step 4: Add structural scoring helper with Late Pooling**

Add to `EOPKGGATv2`:

```python
def _class_aware_structural_logits(
    self,
    *,
    obs_context: torch.Tensor,
    h_node: torch.Tensor,
    node_to_class: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    query = self.struct_query_proj(obs_context)
    keys = self.struct_key_proj(h_node)
    query = torch.nn.functional.normalize(query, p=2, dim=1, eps=1e-12)
    keys = torch.nn.functional.normalize(keys, p=2, dim=1, eps=1e-12)
    node_logits = torch.matmul(query, keys.transpose(0, 1))  # [B, |V|]
    class_logits = torch.full(
        (query.size(0), num_classes),
        fill_value=torch.finfo(node_logits.dtype).min,
        dtype=node_logits.dtype,
        device=node_logits.device,
    )
    for class_idx in range(num_classes):
        node_mask = node_to_class == class_idx
        if torch.any(node_mask):
            class_logits[:, class_idx] = torch.logsumexp(node_logits[:, node_mask], dim=1)
    class_logits = torch.where(
        torch.isfinite(class_logits),
        class_logits,
        torch.zeros_like(class_logits),
    )
    scale = torch.clamp(self.structural_logit_scale, min=0.0, max=10.0)
    return node_logits * scale, class_logits * scale
```

- [ ] **Step 5: Route `forward()` by fusion mode**

Replace the current post-`h_norm` block with:

```python
if self.fusion_mode == "class_aware_structural_scoring":
    observed_logits = self.classifier(obs_context)
    node_to_class = self._resolve_struct_node_to_class_index(
        contract=contract,
        num_struct_nodes=h_norm.size(0),
        num_classes=observed_logits.size(1),
        device=h_norm.device,
    )
    structural_node_logits, structural_class_logits = self._class_aware_structural_logits(
        obs_context=obs_context,
        h_node=h_norm,
        node_to_class=node_to_class,
        num_classes=observed_logits.size(1),
    )
    self.last_cross_attn_weights = None
    self.last_observed_logits = observed_logits.detach()
    self.last_structural_node_logits = structural_node_logits.detach()
    self.last_structural_class_logits = structural_class_logits.detach()
    return observed_logits + structural_class_logits

if self.fusion_mode == "class_mean_concat":
    context_struct = h_norm.mean(dim=0, keepdim=True).expand(batch_size, -1)
    self.last_cross_attn_weights = None
    self.last_observed_logits = None
    self.last_structural_node_logits = None
    self.last_structural_class_logits = None
else:
    query = obs_context.unsqueeze(1)
    kv = self.struct_to_attn(h_norm).unsqueeze(0).expand(batch_size, -1, -1)
    attn_output, attn_weights = self.cross_attention(
        query=query,
        key=kv,
        value=kv,
        need_weights=True,
        average_attn_weights=False,
    )
    self.last_cross_attn_weights = attn_weights.detach()
    self.last_observed_logits = None
    self.last_structural_node_logits = None
    self.last_structural_class_logits = None
    context_struct = self.attn_to_struct(attn_output.squeeze(1))
fused = self.fusion(torch.cat([obs_context, context_struct], dim=1))
return self.classifier(fused)
```

- [ ] **Step 5: Clear diagnostics on fallback**

Every fallback branch that sets `last_cross_attn_weights = None` must also set:

```python
self.last_observed_logits = None
self.last_structural_node_logits = None
self.last_structural_class_logits = None
```

- [ ] **Step 6: Run unit tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/domain/test_eopkg_gatv2_dual_encoder.py -v
```

Expected:

```text
PASSED
```

---

### Task 3: Log Structural Logit Contribution Diagnostics

**Files:**

- Modify: `src/application/training/trainer.py`
- Optional modify: related trainer forward-stats tests.

- [ ] **Step 1: Extend forward-stats accumulator**

Add accumulator fields:

```python
observed_logits_abs_sum
observed_logits_count
structural_logits_abs_sum
structural_logits_abs_max
structural_logits_count
```

- [ ] **Step 2: Accumulate contribution metrics after each model forward**

After each `logits = self.model(contract)` call in train/eval/drift paths, read:

```python
observed_logits = getattr(self.model, "last_observed_logits", None)
structural_logits = getattr(self.model, "last_structural_class_logits", None)
```

If both tensors are present, accumulate absolute magnitudes. Use detached tensors
only. Do not make these diagnostics part of the loss graph.

- [ ] **Step 3: Log stage-level summary metrics**

At every existing forward-stats summary point, compute and log:

```text
observed_logits_mean_abs
structural_logits_mean_abs
structural_logits_max_abs
structural_to_observed_logit_ratio
```

Formula:

```text
observed_logits_mean_abs = mean(abs(observed_logits))
structural_logits_mean_abs = mean(abs(structural_logits))
structural_logits_max_abs = max(abs(structural_logits))
structural_to_observed_logit_ratio =
  structural_logits_mean_abs / max(observed_logits_mean_abs, 1e-12)
```

Required MLflow metric names:

```text
<stage>_observed_logits_mean_abs
<stage>_structural_logits_mean_abs
<stage>_structural_logits_max_abs
<stage>_structural_to_observed_logit_ratio
```

For drift windows also log:

```text
drift_window_observed_logits_mean_abs
drift_window_structural_logits_mean_abs
drift_window_structural_logits_max_abs
drift_window_structural_to_observed_logit_ratio
```

Interpretation:

```text
ratio near 0: structural branch is effectively silent.
ratio around 0.05-0.5: structural signal is present but not dominant.
ratio above 1.0: structural logits dominate observed logits and can damage the baseline.
```

- [ ] **Step 4: Add regression test for diagnostics**

Test that `ClassAwareStructuralScoring` emits the four metrics during validation
or a focused trainer forward-stats run.

---

### Task 4: Add Config Catalog, Experiment UI, And Runtime Documentation

**Files:**

- Modify: `configs/ui/config_catalog.yaml`
- Modify: `docs/GNN_RUNTIME_MVP2_5.MD`
- Optional modify: `docs/current/project-state.md` if runtime semantics are accepted after implementation.

- [ ] **Step 1: Add canonical enum values in config catalog**

Find `model.fusion_mode` and include:

```yaml
- ClassMeanAttention
- ClassMeanConcat
- ClassAwareStructuralScoring
```

Description should state:

```text
ClassMeanAttention is the backward-compatible old Attention mode that compresses structural nodes into one attention context.
ClassMeanConcat is the backward-compatible old Concat/struct_pool_concat mode that uses mean structural context.
ClassAwareStructuralScoring adds per-class structural logits to observed logits.
```

- [ ] **Step 2: Keep backward-compatible aliases in code and docs**

Document aliases:

```text
Attention -> ClassMeanAttention
Concat -> ClassMeanConcat
struct_pool_concat -> ClassMeanConcat
ClassAwareAdditive -> ClassAwareStructuralScoring
ClassAwareAttention -> ClassAwareStructuralScoring
```

- [ ] **Step 3: Add `model.fusion_mode` dropdown values to Experiment UI**

The Experiment UI is driven by `configs/ui/config_catalog.yaml`, so the enum
must expose the canonical values:

```yaml
enum:
  - ClassMeanAttention
  - ClassMeanConcat
  - ClassAwareStructuralScoring
```

If `tools/experiment_ui.py` has hardcoded help text or widget logic for this
field, update it to display the same three canonical choices.

- [ ] **Step 4: Make `model.pooling_strategy` a dropdown**

Find `model.pooling_strategy` in `configs/ui/config_catalog.yaml` and change:

```yaml
default: ''
enum: []
```

To:

```yaml
default: global_mean
enum:
  - global_mean
  - last_node
```

Description should state:

```text
global_mean pools all prefix node embeddings; last_node uses the final prefix node embedding as the process state.
```

- [ ] **Step 5: Make `model.graph_strategy` a readonly dropdown**

Find `model.graph_strategy` in `configs/ui/config_catalog.yaml` and change:

```yaml
default: ''
enum: []
```

To:

```yaml
default: sequential_prefix
enum:
  - sequential_prefix
ui:
  readonly: true
```

If the UI layer uses a different key for readonly/disabled controls, map this
intent to that existing UI convention.

Description should state:

```text
Currently only sequential_prefix is implemented: prefix events are represented as a sequential event graph.
```

Rationale:

`model.graph_strategy` is currently a future extension point. Exposing it as a
free text field or multi-choice control would imply supported graph-building
modes that do not exist yet.

- [ ] **Step 6: Update GNN runtime docs**

Add a short subsection:

```markdown
### ClassAwareStructuralScoring fusion

`ClassAwareStructuralScoring` computes a structural representation for each
structural node, projects node scores to activity classes, and adds a
prefix-conditioned structural score to each class logit:

`node_score_v = score(obs_context, h_node_v)`.

`struct_score_j = logsumexp({node_score_v | class(v) = j})`.

`logit_j = observed_logit_j + gamma * struct_score_j`.

This is not classical attention. It does not reduce structural nodes into one
context vector. It directly creates node-level structural scores and aggregates
them into one score per candidate class.

This mode is intended for next-activity prediction where topology should affect
candidate ranking directly. It is distinct from mask-guided logits: the structural
score is learned and dense, while `allowed_target_mask` remains an optional
constraint/diagnostic layer.
```

- [ ] **Step 7: Run documentation drift check**

Because this changes model runtime semantics and config keys:

```text
runtime_behavior_changed: update docs/GNN_RUNTIME_MVP2_5.MD
config_key_changed: update configs/ui/config_catalog.yaml
data_contract_changed: update docs/DATA_MODEL_MVP2_5.MD with struct_node_to_class_index
project-state update: only if this becomes accepted current behavior
```

---

### Task 5: Add Minimal Debug Experiment Configs

**Files:**

- Create: `configs/experiments/debug_base_un_fraction_001.yaml`
- Create: `configs/experiments/debug_s_classaware_unc_fraction_001.yaml`
- Optional create: `configs/experiments/debug_s_attention_unc_fraction_001.yaml`

Purpose:

Run a cheap `fraction=0.01` controlled comparison before full training.

Config constraints:

```yaml
experiment:
  mode: train
  fraction: 0.01
  train_ratio: 0.25
  structural_mode: true              # false for Base-UN
  statistic_enabled: false
  mask_guided_enabled: false
  cache_policy: off
  graph_dataset_cache_policy: off

mapping:
  graph_feature_mapping:
    topology_projection:
      gateway_mode: collapse_for_prediction
      diagnostics_enabled: true
      on_fail: warn

model:
  type: EOPKGGATv2
  fusion_mode: ClassAwareStructuralScoring   # only for structural run
  pooling_strategy: last_node        # compare with global_mean if needed

training:
  epochs: 10
  patience: 5
```

Do not use these configs as final evidence. They are diagnostic configs only.

---

### Task 6: Verification Protocol

**Files:**

- No new source file required.
- Use MLflow run metrics.

- [ ] **Step 1: Run unit tests**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/domain/test_eopkg_gatv2_dual_encoder.py -v
```

- [ ] **Step 2: Run focused regression**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/application/test_trainer_forward_stats_logging.py tests/domain/test_dynamic_graph_builder_masks.py -v
```

- [ ] **Step 3: Run debug train with fraction 0.01**

```powershell
.\.venv\Scripts\python.exe main.py --config configs/experiments/debug_base_un_fraction_001.yaml
.\.venv\Scripts\python.exe main.py --config configs/experiments/debug_s_classaware_unc_fraction_001.yaml
```

- [ ] **Step 4: Compare debug metrics**

Required metrics:

```text
strict_test_macro_f1
test_macro_f1
test_oos
test_target_in_mask_rate
test_pred_in_mask_rate
test_strict_error_but_allowed_rate
test_ambiguous_prefix_rate
strict_test_f1_mask_card_1
strict_test_f1_mask_card_2
strict_test_f1_mask_card_3_plus
```

Expected direction:

```text
S_ClassAware-UNc should not rely only on hybrid macro_f1.
It should improve or at least not materially degrade strict_test_macro_f1 versus S_Att-UNc.
It should reduce strict_error_but_allowed_rate versus S_Att-UNc.
It should keep OOS lower than Base-UN.
```

- [ ] **Step 5: Run full controlled ablation only after debug passes**

For each seed:

```text
Base-UN
S_Att-UNc
S_ClassAware-UNc
```

Keep:

```text
same dataset
same input features
same train_ratio
same split_strategy
same pooling_strategy
same mask_guided_enabled=false
same statistic_enabled=false
```

Only structural fusion mode changes.

---

## Acceptance Criteria

Implementation is acceptable only if all are true:

1. Existing `Attention` and `Concat` modes remain backward compatible.
2. New `ClassAwareStructuralScoring` mode returns logits with shape `[B, C]`.
3. New mode handles `|V| != C` through `struct_node_to_class_index`.
4. Duplicate activity nodes are aggregated to class logits with Late Pooling / `logsumexp`.
5. Non-target structural nodes use class index `-1` and affect predictions only through message passing.
6. New mode stores `last_structural_node_logits` with shape `[B, |V|]`.
7. New mode stores `last_structural_class_logits` with shape `[B, C]`.
8. New mode stores `last_observed_logits` so observed-vs-structural contribution can be measured.
9. `structural_logit_scale` starts conservatively at `0.1`, not `1.0`.
10. Missing structural tensors still fall back to observed-only classifier.
11. Unit tests pass.
12. Config catalog contains the new fusion mode.
13. Runtime docs explain the difference between:
   - global structural context,
   - class-aware structural logits,
   - mask-guided logits.
14. Training/eval/drift logs include:
   - `structural_logits_mean_abs`,
   - `structural_logits_max_abs`,
   - `observed_logits_mean_abs`,
   - `structural_to_observed_logit_ratio`.
15. Debug run shows whether strict-F1 issue is architectural or data/ambiguity-driven.
16. If scaled cosine fails, the next planned scorer is bilinear, not MLP.

---

## Research Interpretation After Implementation

If `ClassAwareStructuralScoring` improves strict-F1:

```text
Root cause confirmed: previous structure branch compressed topology too early.
Topology is useful when it directly scores candidate activities.
```

If OOS improves but strict-F1 still lags:

```text
Topology alone improves structural admissibility but cannot resolve exact branch choice.
Next step: inspect structural-to-observed logit ratio and error distribution.
If the structural branch is active and non-dominant, try bilinear structural
scoring before moving to MLP or adding stats-backed node/edge features.
```

If neither strict-F1 nor OOS improves:

```text
Investigate topology projection, activity identity ambiguity, duplicate labels,
or whether prefix representation is too weak. Do not add stats before proving
that structural topology is correctly wired.
```

---

## Known Limitations

1. **Topology is not behavior frequency.**  
   BPMN structure says what is possible, not what is likely.

2. **Strict-F1 and structural admissibility are different.**  
   A prediction can be process-valid and still fail exact next-activity matching.

3. **Parallel branches remain ambiguous.**  
   Without route identity or behavior statistics, exact target choice can remain underdetermined.

4. **Duplicate activity labels remain active architecture debt.**  
   `ClassAwareStructuralScoring` can score duplicate structural nodes separately
   and aggregate them into one class logit, but the supervised target is still
   label-level. It cannot know which duplicated BPMN node was the true target
   unless the dataset carries route/node identity.

5. **Stats can help later, but should not hide topology wiring defects.**  
   First validate that topology-only class-aware scoring behaves as expected.

---

## Self-Review

- `placeholder_scan`: no placeholders left.
- `scope_check`: one model-level architectural change plus config/docs/test updates.
- `type_consistency`: all proposed names map to current project concepts.
- `risk_check`: plan preserves current modes and adds new mode instead of changing current `Attention`.
- `documentation_check`: config catalog and GNN runtime docs are included.
