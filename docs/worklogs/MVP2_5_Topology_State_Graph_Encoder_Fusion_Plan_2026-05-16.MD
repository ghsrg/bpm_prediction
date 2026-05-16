# MVP2.5 TopologyStateGraphEncoder Fusion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans`
> or `executing-plans` to implement this plan task-by-task. Use
> `test-driven-development` for every implementation task and
> `verification-before-completion` before reporting completion. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `model.fusion_mode=TopologyStateGraphEncoder` as a structural
graph encoder baseline that mirrors the old etalon/article-style topology
input: prefix state is projected onto BPMN/EOPKG nodes, structural message
passing is performed on `structural_edge_index`, and a graph-level structural
context is classified.

**Architecture:** This mode is a pure topology-state graph readout, not a
late structural-logit scorer and not an observed-prefix plus pooled-structure
concat. It reuses the existing `struct_prefix_state_x` contract emitted by
`DynamicGraphBuilder`, reuses the topology-state input projection/gate, runs
the existing structural GNN on batched topology graphs, then applies mean
graph pooling and the existing `fusion + classifier` head with a zero observed
context. This avoids new checkpoint parameters for unrelated modes and gives a
clean structural graph baseline.

**Tech Stack:** Python, PyTorch, PyTorch Geometric, existing
`GraphTensorContract`, `DynamicGraphBuilder`, `EOPKGGATv2`, pytest, local venv
`.\.venv\Scripts\python.exe`.

---

## 1. Context And Rationale

### 1.1 Current fusion modes

Current `EOPKGGATv2` modes cover different integration levels:

1. `ClassMeanAttention`: old attention path; observed prefix context queries
   structural node states and receives one structural context vector.
2. `ClassMeanConcat`: old mean structural context path.
3. `ClassAwareStructuralScoring`: late class-aware structural logits are added
   to observed logits.
4. `TopologyStateEncoder`: early topology-state path, but it produces
   per-node logits and then pools them into class logits.
5. `StructuralPriorEncoder`: observed prefix encoder remains primary;
   structural node states are mean-pooled into one context and fused before the
   classifier.

The recent experiments indicate that `StructuralPriorEncoder` is technically
active but behaves too close to baseline because it compresses the entire
structure into one global prior after the observed prefix encoder. The older
`etalon` effect was stronger because structure was the graph on which node
features were propagated before graph-level classification.

### 1.2 Why `TopologyStateGraphEncoder`

`TopologyStateGraphEncoder` should test a more article/etalon-like baseline:

```text
structural node identity + optional stats + prefix-state features
  -> structural GNN over BPMN/EOPKG edges
  -> global_mean_pool over structural nodes
  -> classifier
```

This is different from `TopologyStateEncoder`:

```text
TopologyStateEncoder:
  topology nodes -> structural GNN -> node logits -> node-to-class pooling

TopologyStateGraphEncoder:
  topology nodes -> structural GNN -> graph-level context -> classifier
```

The graph-level readout avoids direct `LogSumExp` class inflation from duplicate
activity nodes. It does not solve duplicate activity identity ambiguity, but it
does not force duplicated BPMN nodes to become duplicated additive class logits.

### 1.3 Research interpretation

This mode is a controlled baseline for showing whether BPMN/EOPKG topology is
useful for same-version predictive monitoring when it is used as the actual
prediction graph. It is not expected to solve unseen-version drift transfer by
itself. A hard `structural_edge_index` can still overfit to the topology
spectrum of seen versions and may require later transfer methodology:
cross-attention, side-tuning, topology-as-soft-bias, or version-invariant
training objectives.

---

## 2. Non-Goals

- Do not change `DynamicGraphBuilder` data production. The required
  `struct_prefix_state_x`, `struct_x`, `structural_edge_index`, and
  `struct_node_to_class_index` fields already exist.
- Do not change the meaning of `TopologyStateEncoder`.
- Do not remove or rewrite `ClassAwareStructuralScoring`,
  `StructuralPriorEncoder`, `ClassMeanAttention`, or `ClassMeanConcat`.
- Do not solve duplicate activity identity ambiguity.
- Do not implement transfer learning, side-network tuning, or layerwise
  structural cross-attention in this task.
- Do not make this mode the default.
- Do not rebuild topology/stats from raw sources during train/eval/infer.

---

## 3. Target Config Contract

Canonical config:

```yaml
experiment:
  structural_mode: true

model:
  type: EOPKGGATv2
  fusion_mode: TopologyStateGraphEncoder
  topology_graph_pooling: mean
```

Shared topology-state controls:

```yaml
model:
  topology_state_beta: 0.5
  topology_state_beta_max: 2.0
  topology_state_gate_init_bias: -2.0
  topology_state_dropout: 0.2
  structural_stats_beta: 0.1
```

Semantics:

1. `experiment.structural_mode=false` must still force observed-only behavior.
2. `model.fusion_mode=TopologyStateGraphEncoder` requires structural tensors
   and `struct_prefix_state_x`.
3. Missing `struct_prefix_state_x` must raise a clear `ValueError`; silent
   fallback invalidates this ablation.
4. Initial `model.topology_graph_pooling` supports only `mean`.
5. `training.structural_aux_loss_enabled` must not affect this mode because it
   is only defined for `ClassAwareStructuralScoring` structural class logits.

Recommended aliases:

```text
TopologyStateGraphEncoder -> topology_state_graph_encoder
topology_state_graph_encoder -> topology_state_graph_encoder
TopologyGraphEncoder -> topology_state_graph_encoder
topology_graph_encoder -> topology_state_graph_encoder
StructuralGraphEncoder -> topology_state_graph_encoder
```

---

## 4. Data Contract

No new graph-builder field is required.

The mode consumes:

```text
structural_edge_index: LongTensor [2, E_struct]
structural_edge_weight: FloatTensor [E_struct] optional, currently not used by GATv2 structural encoder
struct_node_to_class_index: LongTensor [|V|] optional but expected for identity projection
struct_x: FloatTensor [|V|, F_struct] optional stats-backed features
struct_prefix_state_x: FloatTensor [|V|, 6] per graph or [B, |V|, 6] in batch
```

`struct_prefix_state_x` feature order remains:

```text
0 prefix_executed_count_log1p
1 prefix_was_executed
2 prefix_is_last_event
3 prefix_last_position_norm
4 prefix_recency_norm
5 prefix_is_active_after_complete
```

The graph-level encoder must accept `[V, 6]` and `[B, V, 6]`. The helper
`_build_topology_state_input()` already normalizes the single-graph case to
`[1, V, 6]`, validates row count and feature count, applies prefix projection,
gate, beta clamp, dropout, and sets prefix/gate diagnostics.

---

## 5. Model Design

### 5.1 Forward path

The target path for this mode:

```text
base_struct_v = LayerNorm(identity_v + structural_stats_beta * W_stats(struct_x_v))
prefix_v = W_prefix(struct_prefix_state_x_v)
gate_v = sigmoid(W_gate(struct_prefix_state_x_v) + topology_state_gate_init_bias)
h0_v = LayerNorm(base_struct_v + clamp(beta, 0, beta_max) * gate_v * prefix_v)
h_v = StructuralGNN(h0_v, structural_edge_index)
graph_context_i = mean_v(h_i,v)
logits_i = classifier(fusion([zeros_i || graph_context_i]))
```

Where:

```text
zeros_i: FloatTensor [hidden_dim]
graph_context_i: FloatTensor [struct_hidden_dim]
fusion input shape: [hidden_dim + struct_hidden_dim]
```

Using the existing `fusion + classifier` head is intentional:

1. It matches the existing `EOPKGGATv2` dimensional contract.
2. It avoids introducing new state-dict keys for unrelated modes.
3. It keeps checkpoint compatibility risk lower than adding a new
   `topology_graph_head`.
4. It still trains a graph-level classifier from structural graph context.

### 5.2 Observed branch handling

For `TopologyStateGraphEncoder`, observed IG node features must not participate
in prediction. The observed graph may only be used for fallback when structural
tensors are missing and the model is not in this strict mode.

Preferred implementation:

```python
if self.structural_mode and self.fusion_mode == "topology_state_graph_encoder":
    structural_edge_index = contract.get("structural_edge_index")
    if structural_edge_index is None or structural_edge_index.numel() == 0:
        raise ValueError("TopologyStateGraphEncoder requires structural_edge_index.")
    device = self._resolve_contract_device(contract)
    return self._forward_topology_state_graph(contract=contract, structural_edge_index=structural_edge_index, device=device)

# existing observed-first path for all other modes
x, edge_index, batch = self._encode_input(contract)
node_hidden = self._forward_local(x, edge_index)
obs_context = self._pool_nodes(node_hidden, batch)
...
```

This makes the baseline clean: changing `x_cat` or `x_num` while keeping
structural tensors and `struct_prefix_state_x` fixed must not change logits in
this mode.

### 5.3 Diagnostics

Keep existing topology-state diagnostics from `_build_topology_state_input()`:

```text
topology_state_prefix_mean_abs
topology_state_prefix_max_abs
topology_state_gate_mean
topology_state_gate_max
```

Add graph-specific diagnostics:

```text
topology_graph_context_mean_abs
topology_graph_context_max_abs
topology_graph_logits_mean_abs
topology_graph_entropy
```

Model attributes:

```python
self.last_topology_graph_context: torch.Tensor | None = None
self.last_topology_graph_logits: torch.Tensor | None = None
self.last_topology_graph_entropy: torch.Tensor | None = None
```

These diagnostics answer three questions:

1. Is the graph branch active?
2. Is the graph context numerically non-zero?
3. Is the classifier becoming overconfident or unstable?

---

## 6. Files To Modify

### Runtime

- Modify: `src/domain/models/eopkg_models.py`
  - Add fusion aliases and validation.
  - Add `topology_graph_pooling` constructor parameter.
  - Add graph-level topology forward helper.
  - Add graph diagnostics fields and clear logic.
  - Route `TopologyStateGraphEncoder` before observed encoder computation.

- Modify: `src/cli.py`
  - Pass `model.topology_graph_pooling` through `_build_model_factory_kwargs()`.

### Config / UI

- Modify: `configs/ui/config_catalog.yaml`
  - Add `TopologyStateGraphEncoder` to `model.fusion_mode` enum.
  - Add `model.topology_graph_pooling` with one enum value: `mean`.
  - Update descriptions for shared `topology_state_*` keys so they apply to
    `TopologyStateEncoder` and `TopologyStateGraphEncoder`.

### Tests

- Modify: `tests/domain/test_eopkg_topology_state_encoder.py`
  - Add alias tests.
  - Add graph-level forward and diagnostics tests.
  - Add pure-structural-path test showing observed `x_cat/x_num` changes do not
    affect logits.
  - Add missing `struct_prefix_state_x` fail-fast test.

- Modify: `tests/domain/test_eopkg_models_forward.py`
  - Add `TopologyStateGraphEncoder` to supported fusion modes.

- Modify: `tests/domain/test_eopkg_gatv2_dual_encoder.py`
  - Add alias normalization cases and unsupported pooling validation.

- Modify: `tests/application/test_cli_flatten_config.py`
  - Add factory kwargs forwarding test for `topology_graph_pooling`.

- Modify: `tests/application/test_trainer_forward_stats_logging.py`
  - Add trainer diagnostic logging coverage for graph-level topology metrics.

### Documentation

- Modify: `docs/GNN_RUNTIME_MVP2_5.MD`
  - Add `TopologyStateGraphEncoder` section.
  - Clarify difference from `TopologyStateEncoder` and `StructuralPriorEncoder`.

- Modify: `docs/current/project-state.md`
  - Update `gnn_runtime` summary.

- Modify: `docs/dissertation_changes.MD`
  - Add note that this is the etalon/article-like structural graph baseline.

- Create implementation report after execution:
  - `docs/worklogs/MVP2_5_Topology_State_Graph_Encoder_Fusion_Report_2026-05-16.MD`

---

## 7. Task 1: Add Failing Domain Tests For Mode Alias And Config Validation

**Files:**

- Modify: `tests/domain/test_eopkg_topology_state_encoder.py`
- Modify: `tests/domain/test_eopkg_gatv2_dual_encoder.py`

- [ ] **Step 1: Add alias cases to topology-state test file**

Extend the alias parameterization:

```python
@pytest.mark.parametrize(
    ("raw_fusion_mode", "expected"),
    [
        ("TopologyStateEncoder", "topology_state_encoder"),
        ("topology_state_encoder", "topology_state_encoder"),
        ("topology_state", "topology_state_encoder"),
        ("EarlyTopologyStateEncoder", "topology_state_encoder"),
        ("TopologyStateGraphEncoder", "topology_state_graph_encoder"),
        ("topology_state_graph_encoder", "topology_state_graph_encoder"),
        ("TopologyGraphEncoder", "topology_state_graph_encoder"),
        ("topology_graph_encoder", "topology_state_graph_encoder"),
        ("StructuralGraphEncoder", "topology_state_graph_encoder"),
    ],
)
def test_eopkg_gatv2_normalizes_topology_state_aliases(raw_fusion_mode: str, expected: str):
    model = create_model(
        model_type="EOPKGGATv2",
        feature_layout=_layout(),
        hidden_dim=16,
        output_dim=3,
        struct_hidden_dim=16,
        dropout=0.0,
        fusion_mode=raw_fusion_mode,
    )

    assert model.fusion_mode == expected
```

- [ ] **Step 2: Add unsupported pooling validation test**

Add to `tests/domain/test_eopkg_gatv2_dual_encoder.py`:

```python
def test_eopkg_gatv2_rejects_unknown_topology_graph_pooling():
    with pytest.raises(ValueError, match="Unsupported model.topology_graph_pooling"):
        create_model(
            model_type="EOPKGGATv2",
            feature_layout=_layout(),
            hidden_dim=16,
            output_dim=7,
            struct_hidden_dim=16,
            dropout=0.0,
            pooling_strategy="global_mean",
            fusion_mode="TopologyStateGraphEncoder",
            topology_graph_pooling="attention",
        )
```

- [ ] **Step 3: Run tests and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/domain/test_eopkg_topology_state_encoder.py::test_eopkg_gatv2_normalizes_topology_state_aliases tests/domain/test_eopkg_gatv2_dual_encoder.py::test_eopkg_gatv2_rejects_unknown_topology_graph_pooling -v
```

Expected:

```text
FAIL
Unsupported model.fusion_mode 'topology_state_graph_encoder'
or __init__ got an unexpected keyword argument 'topology_graph_pooling'
```

---

## 8. Task 2: Add Failing Forward Tests For Graph-Level Topology Encoder

**Files:**

- Modify: `tests/domain/test_eopkg_topology_state_encoder.py`

- [ ] **Step 1: Add graph-level forward test**

Add:

```python
def test_topology_state_graph_encoder_returns_graph_logits_and_diagnostics():
    model = create_model(
        model_type="EOPKGGATv2",
        feature_layout=_layout(),
        hidden_dim=16,
        output_dim=3,
        struct_hidden_dim=16,
        dropout=0.0,
        fusion_mode="TopologyStateGraphEncoder",
        topology_state_beta=0.5,
        topology_graph_pooling="mean",
    )

    logits = model(_topology_contract(batch_size=2))

    assert tuple(logits.shape) == (2, 3)
    assert model.last_topology_graph_context is not None
    assert tuple(model.last_topology_graph_context.shape) == (2, 16)
    assert model.last_topology_graph_logits is not None
    assert tuple(model.last_topology_graph_logits.shape) == (2, 3)
    assert model.last_topology_graph_entropy is not None
    assert model.last_topology_state_gate_mean is not None
    assert model.last_topology_state_gate_max is not None
    assert model.last_topology_state_node_logits is None
    assert model.last_topology_state_class_logits is None
```

- [ ] **Step 2: Add missing prefix-state fail-fast test**

Add:

```python
def test_topology_state_graph_encoder_requires_prefix_state_tensor():
    model = create_model(
        model_type="EOPKGGATv2",
        feature_layout=_layout(),
        hidden_dim=16,
        output_dim=3,
        struct_hidden_dim=16,
        dropout=0.0,
        fusion_mode="TopologyStateGraphEncoder",
    )
    contract = _topology_contract(batch_size=1)
    contract.pop("struct_prefix_state_x")

    with pytest.raises(ValueError, match="TopologyStateGraphEncoder requires struct_prefix_state_x"):
        model(contract)
```

- [ ] **Step 3: Add observed-input invariance test**

Add:

```python
def test_topology_state_graph_encoder_ignores_observed_graph_features():
    model = create_model(
        model_type="EOPKGGATv2",
        feature_layout=_layout(),
        hidden_dim=16,
        output_dim=3,
        struct_hidden_dim=16,
        dropout=0.0,
        fusion_mode="TopologyStateGraphEncoder",
    )
    model.eval()
    contract_a = _topology_contract(batch_size=2)
    contract_b = {key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in contract_a.items()}
    contract_b["x_cat"] = torch.flip(contract_b["x_cat"], dims=[0])
    contract_b["x_num"] = contract_b["x_num"] + 1000.0

    with torch.no_grad():
        logits_a = model(contract_a)
        logits_b = model(contract_b)

    assert torch.allclose(logits_a, logits_b, atol=1e-6, rtol=0.0)
```

- [ ] **Step 4: Add structural mode off behavior test**

Add:

```python
def test_topology_state_graph_encoder_structural_mode_off_uses_observed_only_path():
    model = create_model(
        model_type="EOPKGGATv2",
        feature_layout=_layout(),
        hidden_dim=16,
        output_dim=3,
        struct_hidden_dim=16,
        dropout=0.0,
        structural_mode=False,
        fusion_mode="TopologyStateGraphEncoder",
    )
    model.eval()

    with torch.no_grad():
        without_struct = model(_base_contract(batch_size=1))
        with_struct = model(_topology_contract(batch_size=1))

    assert tuple(without_struct.shape) == (1, 3)
    assert tuple(with_struct.shape) == (1, 3)
    assert torch.allclose(without_struct, with_struct, atol=1e-6, rtol=0.0)
```

- [ ] **Step 5: Run tests and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/domain/test_eopkg_topology_state_encoder.py -v
```

Expected:

```text
FAIL
TopologyStateGraphEncoder is not yet supported
```

---

## 9. Task 3: Implement `TopologyStateGraphEncoder` In `EOPKGGATv2`

**Files:**

- Modify: `src/domain/models/eopkg_models.py`

- [ ] **Step 1: Extend constructor signature**

Add parameter after `topology_state_class_pooling`:

```python
topology_graph_pooling: str = "mean",
```

- [ ] **Step 2: Extend aliases and allowed modes**

Add to `alias_map`:

```python
"topologystategraphencoder": "topology_state_graph_encoder",
"topology_state_graph_encoder": "topology_state_graph_encoder",
"topologygraphencoder": "topology_state_graph_encoder",
"topology_graph_encoder": "topology_state_graph_encoder",
"structuralgraphencoder": "topology_state_graph_encoder",
```

Add to `allowed_fusion_modes`:

```python
"topology_state_graph_encoder",
```

Update the error message available values to include:

```text
'TopologyStateGraphEncoder'
```

- [ ] **Step 3: Add pooling validation**

Add after topology-state class-pooling validation:

```python
self.topology_graph_pooling = str(topology_graph_pooling or "mean").strip().lower()
if self.topology_graph_pooling not in {"mean"}:
    raise ValueError(
        "Unsupported model.topology_graph_pooling "
        f"'{self.topology_graph_pooling}'. Available: ['mean']"
    )
```

- [ ] **Step 4: Add diagnostics fields**

Add in `__init__`:

```python
self.last_topology_graph_context: torch.Tensor | None = None
self.last_topology_graph_logits: torch.Tensor | None = None
self.last_topology_graph_entropy: torch.Tensor | None = None
```

Add in `_clear_structural_diagnostics()`:

```python
self.last_topology_graph_context = None
self.last_topology_graph_logits = None
self.last_topology_graph_entropy = None
```

- [ ] **Step 5: Add helper to resolve contract device**

Add method:

```python
@staticmethod
def _resolve_contract_device(contract: Mapping[str, Any]) -> torch.device:
    for key in ("x_num", "x_cat", "struct_x", "struct_prefix_state_x", "structural_edge_index"):
        value = contract.get(key)
        if isinstance(value, torch.Tensor):
            return value.device
    return torch.device("cpu")
```

- [ ] **Step 6: Add graph pooling helper**

Add:

```python
def _pool_topology_graph_context(self, h_topology: torch.Tensor) -> torch.Tensor:
    if self.topology_graph_pooling == "mean":
        return h_topology.mean(dim=1)
    raise ValueError(
        "Unsupported model.topology_graph_pooling "
        f"'{self.topology_graph_pooling}'. Available: ['mean']"
    )
```

- [ ] **Step 7: Add strict topology graph forward helper**

Add:

```python
def _forward_topology_state_graph(
    self,
    *,
    contract: GraphTensorContract,
    structural_edge_index: torch.Tensor,
    struct_nodes: torch.Tensor,
) -> torch.Tensor:
    prefix_state_x = contract.get("struct_prefix_state_x")
    if not isinstance(prefix_state_x, torch.Tensor):
        raise ValueError("TopologyStateGraphEncoder requires struct_prefix_state_x.")

    h0 = self._build_topology_state_input(
        base_struct=struct_nodes,
        prefix_state_x=prefix_state_x,
    )
    h_topology = self._run_topology_state_gnn(h0, structural_edge_index)
    graph_context = self._pool_topology_graph_context(h_topology)
    zero_obs = graph_context.new_zeros((int(graph_context.size(0)), self.hidden_dim))
    fused = self.fusion(torch.cat([zero_obs, graph_context], dim=1))
    logits = self.classifier(fused)
    probs = torch.softmax(logits, dim=1)
    entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=1).mean()

    self.last_cross_attn_weights = None
    self.last_observed_logits = None
    self.last_structural_node_logits = None
    self.last_structural_raw_class_logits = None
    self.last_structural_normalized_class_logits = None
    self.last_structural_observed_scale = None
    self.last_structural_class_logits = None
    self.last_topology_state_node_logits = None
    self.last_topology_state_class_logits = None
    self.last_topology_state_entropy = None
    self.last_topology_state_mean_class_cardinality = None
    self.last_topology_state_max_class_cardinality = None
    self.last_observed_context = None
    self.last_structural_prior_context = None
    self.last_structural_prior_gate = None
    self.last_topology_graph_context = graph_context.detach()
    self.last_topology_graph_logits = logits.detach()
    self.last_topology_graph_entropy = entropy.detach()
    return logits
```

- [ ] **Step 8: Add structural tensor preparation helper**

Extract the existing structural-node preparation logic from `forward()` into:

```python
def _prepare_structural_nodes(
    self,
    *,
    contract: GraphTensorContract,
    structural_edge_index: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    structural_edge_index_max = int(structural_edge_index.max().item())
    raw_struct_x = contract.get("struct_x")
    if isinstance(raw_struct_x, torch.Tensor):
        raw_struct_x_rows = int(raw_struct_x.size(0)) if raw_struct_x.dim() > 0 else 0
        if structural_edge_index_max >= raw_struct_x_rows:
            raise ValueError(
                "Structural edge index is out of bounds: "
                f"max_index={structural_edge_index_max} node_count={raw_struct_x_rows} "
                f"struct_x_shape={tuple(raw_struct_x.shape)}."
            )
        num_struct_nodes = raw_struct_x_rows
    elif structural_edge_index_max < self.num_classes:
        num_struct_nodes = self.num_classes
    else:
        num_struct_nodes = structural_edge_index_max + 1

    struct_nodes = self._build_struct_node_features(
        contract,
        device=device,
        num_struct_nodes=num_struct_nodes,
    )
    node_count = int(struct_nodes.size(0))
    if node_count <= 0:
        raise ValueError("Structural node tensor is empty.")
    if structural_edge_index_max >= node_count:
        raise ValueError(
            "Structural edge index is out of bounds: "
            f"max_index={structural_edge_index_max} node_count={node_count} "
            f"struct_x_shape={tuple(struct_nodes.shape)}."
        )
    return struct_nodes
```

Then update existing `forward()` to reuse it for all structural modes.

- [ ] **Step 9: Route graph mode before observed encoder**

At the beginning of `forward()`:

```python
if self.structural_mode and self.fusion_mode == "topology_state_graph_encoder":
    structural_edge_index = contract.get("structural_edge_index")
    if structural_edge_index is None or structural_edge_index.numel() == 0:
        raise ValueError("TopologyStateGraphEncoder requires structural_edge_index.")
    device = self._resolve_contract_device(contract)
    structural_edge_index = structural_edge_index.to(device=device, dtype=torch.long)
    struct_nodes = self._prepare_structural_nodes(
        contract=contract,
        structural_edge_index=structural_edge_index,
        device=device,
    )
    return self._forward_topology_state_graph(
        contract=contract,
        structural_edge_index=structural_edge_index,
        struct_nodes=struct_nodes,
    )
```

Keep the existing observed-first path for all other modes and for
`structural_mode=false`.

- [ ] **Step 10: Run focused domain tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/domain/test_eopkg_topology_state_encoder.py tests/domain/test_eopkg_models_forward.py tests/domain/test_eopkg_gatv2_dual_encoder.py -v
```

Expected:

```text
PASS
```

---

## 10. Task 4: Wire CLI Factory Config

**Files:**

- Modify: `src/cli.py`
- Modify: `tests/application/test_cli_flatten_config.py`

- [ ] **Step 1: Add failing CLI kwargs test**

Add:

```python
def test_build_model_factory_kwargs_passes_topology_graph_config():
    kwargs = _build_model_factory_kwargs(
        model_cfg={
            "type": "EOPKGGATv2",
            "hidden_dim": 16,
            "fusion_mode": "TopologyStateGraphEncoder",
            "topology_graph_pooling": "mean",
        },
        feature_layout=object(),
        output_dim=7,
    )

    assert kwargs["fusion_mode"] == "TopologyStateGraphEncoder"
    assert kwargs["topology_graph_pooling"] == "mean"
```

- [ ] **Step 2: Run test and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/application/test_cli_flatten_config.py::test_build_model_factory_kwargs_passes_topology_graph_config -v
```

Expected:

```text
FAIL
KeyError: 'topology_graph_pooling'
```

- [ ] **Step 3: Forward config key**

In `_build_model_factory_kwargs()`, add:

```python
"topology_graph_pooling": str(model_cfg.get("topology_graph_pooling", "mean")),
```

- [ ] **Step 4: Run CLI tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/application/test_cli_flatten_config.py -v
```

Expected:

```text
PASS
```

---

## 11. Task 5: Add Trainer Diagnostics For Graph-Level Topology Mode

**Files:**

- Modify: `src/application/use_cases/trainer.py`
- Modify: `tests/application/test_trainer_forward_stats_logging.py`

- [ ] **Step 1: Add diagnostic dummy model**

Add:

```python
class _TopologyGraphDiagnosticModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dummy = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.last_topology_graph_context: torch.Tensor | None = None
        self.last_topology_graph_logits: torch.Tensor | None = None
        self.last_topology_graph_entropy: torch.Tensor | None = None

    def forward(self, contract):
        batch = contract["batch"]
        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
        context = torch.tensor([[0.5, -0.25]], dtype=torch.float32, device=batch.device).repeat(num_graphs, 1)
        logits = torch.tensor([[1.0, -1.0]], dtype=torch.float32, device=batch.device).repeat(num_graphs, 1)
        self.last_topology_graph_context = context.detach()
        self.last_topology_graph_logits = logits.detach()
        self.last_topology_graph_entropy = torch.tensor(0.75, dtype=torch.float32, device=batch.device)
        return logits + (self.dummy * 0.0)
```

- [ ] **Step 2: Add failing tracker/log test**

Add:

```python
def test_trainer_logs_topology_graph_diagnostics(caplog):
    loader = DataLoader(
        [
            _sample(0, snapshot_idx=1, snapshot_epoch=0.0),
            _sample(1, snapshot_idx=1, snapshot_epoch=0.0),
        ],
        batch_size=2,
        shuffle=False,
    )
    tracker = _RecordingTracker()
    trainer = _make_trainer(model=_TopologyGraphDiagnosticModel(), tracker=tracker)

    caplog.set_level(logging.INFO)
    trainer._evaluate_test(loader, stage_label="inference")

    assert "topology_graph_context_mean_abs=" in caplog.text
    assert "topology_graph_context_max_abs=" in caplog.text
    assert "topology_graph_logits_mean_abs=" in caplog.text
    assert "topology_graph_entropy=" in caplog.text
    metric_names = {name for name, _, _ in tracker.metrics}
    assert "inference_topology_graph_context_mean_abs" in metric_names
    assert "inference_topology_graph_logits_mean_abs" in metric_names
    assert "inference_topology_graph_entropy" in metric_names
```

- [ ] **Step 3: Run test and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/application/test_trainer_forward_stats_logging.py::test_trainer_logs_topology_graph_diagnostics -v
```

Expected:

```text
FAIL
topology_graph_context_mean_abs is not in forward stats log
```

- [ ] **Step 4: Add bucket fields**

In `_new_forward_stats_bucket()`, add:

```python
"topology_graph_context_abs_sum": 0.0,
"topology_graph_context_abs_max": 0.0,
"topology_graph_context_count": 0,
"topology_graph_logits_abs_sum": 0.0,
"topology_graph_logits_count": 0,
"topology_graph_entropy_sum": 0.0,
"topology_graph_diag_batches": 0,
```

- [ ] **Step 5: Add accumulator**

Add:

```python
def _accumulate_topology_graph_stats(self, bucket: Dict[str, Any]) -> None:
    context = getattr(self.model, "last_topology_graph_context", None)
    logits = getattr(self.model, "last_topology_graph_logits", None)
    entropy = self._diagnostic_scalar(getattr(self.model, "last_topology_graph_entropy", None))
    if not isinstance(context, torch.Tensor) and not isinstance(logits, torch.Tensor) and entropy is None:
        return
    bucket["topology_graph_diag_batches"] = int(bucket.get("topology_graph_diag_batches", 0)) + 1
    if isinstance(context, torch.Tensor):
        safe = torch.abs(torch.nan_to_num(context.detach().float(), nan=0.0, posinf=1e6, neginf=-1e6))
        bucket["topology_graph_context_abs_sum"] = float(bucket.get("topology_graph_context_abs_sum", 0.0)) + float(safe.sum().item())
        bucket["topology_graph_context_count"] = int(bucket.get("topology_graph_context_count", 0)) + int(safe.numel())
        bucket["topology_graph_context_abs_max"] = max(
            float(bucket.get("topology_graph_context_abs_max", 0.0)),
            float(safe.max().item()),
        )
    if isinstance(logits, torch.Tensor):
        safe_logits = torch.abs(torch.nan_to_num(logits.detach().float(), nan=0.0, posinf=1e6, neginf=-1e6))
        bucket["topology_graph_logits_abs_sum"] = float(bucket.get("topology_graph_logits_abs_sum", 0.0)) + float(safe_logits.sum().item())
        bucket["topology_graph_logits_count"] = int(bucket.get("topology_graph_logits_count", 0)) + int(safe_logits.numel())
    if entropy is not None:
        bucket["topology_graph_entropy_sum"] = float(bucket.get("topology_graph_entropy_sum", 0.0)) + float(entropy)
```

- [ ] **Step 6: Call accumulator after forward**

Where `_accumulate_topology_state_stats()` and
`_accumulate_structural_prior_stats()` are called, add:

```python
self._accumulate_topology_graph_stats(forward_stats)
```

- [ ] **Step 7: Extend log summary and tracker metrics**

In `_log_forward_stats_summary()`, compute:

```python
topology_graph_context_count = int(bucket.get("topology_graph_context_count", 0))
topology_graph_logits_count = int(bucket.get("topology_graph_logits_count", 0))
topology_graph_diag_batches = int(bucket.get("topology_graph_diag_batches", 0))
topology_graph_context_mean_abs = (
    float(bucket.get("topology_graph_context_abs_sum", 0.0)) / float(topology_graph_context_count)
    if topology_graph_context_count > 0
    else 0.0
)
topology_graph_context_max_abs = float(bucket.get("topology_graph_context_abs_max", 0.0))
topology_graph_logits_mean_abs = (
    float(bucket.get("topology_graph_logits_abs_sum", 0.0)) / float(topology_graph_logits_count)
    if topology_graph_logits_count > 0
    else 0.0
)
topology_graph_entropy = (
    float(bucket.get("topology_graph_entropy_sum", 0.0)) / float(topology_graph_diag_batches)
    if topology_graph_diag_batches > 0
    else 0.0
)
```

Add these fields to the `logger.info()` format string and arguments:

```text
topology_graph_context_mean_abs=%.6f topology_graph_context_max_abs=%.6f
topology_graph_logits_mean_abs=%.6f topology_graph_entropy=%.6f
```

Add tracker metrics:

```python
if self.tracker is not None and topology_graph_diag_batches > 0:
    self.tracker.log_metric(f"{metric_prefix}_topology_graph_context_mean_abs", topology_graph_context_mean_abs)
    self.tracker.log_metric(f"{metric_prefix}_topology_graph_context_max_abs", topology_graph_context_max_abs)
    self.tracker.log_metric(f"{metric_prefix}_topology_graph_logits_mean_abs", topology_graph_logits_mean_abs)
    self.tracker.log_metric(f"{metric_prefix}_topology_graph_entropy", topology_graph_entropy)
```

- [ ] **Step 8: Run trainer diagnostics tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/application/test_trainer_forward_stats_logging.py -v
```

Expected:

```text
PASS
```

---

## 12. Task 6: Update Config Catalog

**Files:**

- Modify: `configs/ui/config_catalog.yaml`

- [ ] **Step 1: Add fusion enum value**

In `model.fusion_mode.enum`, add:

```yaml
- TopologyStateGraphEncoder
```

Update description:

```yaml
description: ClassMeanAttention keeps old Attention semantics; ClassMeanConcat keeps old Concat/struct_pool_concat semantics; ClassAwareStructuralScoring adds per-class structural logits to observed logits; TopologyStateEncoder projects observed prefix state onto topology nodes before structural message passing and pools node logits to classes; TopologyStateGraphEncoder projects observed prefix state onto topology nodes, performs topology message passing, then classifies a graph-level topology context; StructuralPriorEncoder uses etalon-like representation-level fusion with observed context plus pooled structural context before classifier.
```

- [ ] **Step 2: Add `model.topology_graph_pooling`**

Add after `model.topology_state_class_pooling`:

```yaml
  model.topology_graph_pooling:
    path: model.topology_graph_pooling
    section: model
    label: 'Model: Topology graph pooling'
    description: Graph-level pooling strategy for TopologyStateGraphEncoder; mean mirrors etalon/article-style global_mean_pool.
    affects: How structural node states become one graph-level topology context before classification.
    default: mean
    enum:
    - mean
    required_in_modes: []
    required_when:
      model.type: EOPKGGATv2
      model.fusion_mode: TopologyStateGraphEncoder
    ui:
      tab: model
      group: advanced
      priority: 100
      order: 1011
```

- [ ] **Step 3: Adjust shared topology-state config visibility**

For these keys:

```text
model.topology_state_beta
model.topology_state_beta_max
model.topology_state_gate_init_bias
model.topology_state_dropout
```

Update descriptions to say they apply to both:

```text
TopologyStateEncoder and TopologyStateGraphEncoder
```

If the UI catalog cannot express `model.fusion_mode in [TopologyStateEncoder,
TopologyStateGraphEncoder]`, make these keys visible for all `EOPKGGATv2`
advanced configs by keeping only:

```yaml
required_when:
  model.type: EOPKGGATv2
```

Do not duplicate the same config key under two catalog entries.

---

## 13. Task 7: Update Documentation

**Files:**

- Modify: `docs/GNN_RUNTIME_MVP2_5.MD`
- Modify: `docs/current/project-state.md`
- Modify: `docs/dissertation_changes.MD`

- [ ] **Step 1: Update runtime mode list**

In `docs/GNN_RUNTIME_MVP2_5.MD`, add:

```text
`model.fusion_mode=TopologyStateGraphEncoder` runs an etalon/article-like
structural graph baseline where prefix-state features are injected into
topology nodes, structural message passing is performed, and a graph-level
topology context is classified.
```

- [ ] **Step 2: Add dedicated runtime section**

Add section after `TopologyStateEncoder`:

```markdown
### 6.6 TopologyStateGraphEncoder Fusion

`TopologyStateGraphEncoder` is a structural graph encoder baseline. It uses the
same `struct_prefix_state_x` tensor as `TopologyStateEncoder`, but it does not
emit node logits and does not perform node-to-class pooling.

Runtime formula:

```text
base_v = structural_identity_v + structural_stats_beta * stats_projection(struct_x_v)
prefix_v = W_prefix struct_prefix_state_x_v
gate_v = sigmoid(W_gate struct_prefix_state_x_v + topology_state_gate_init_bias)
h0_v = LayerNorm(base_v + beta * gate_v * prefix_v)
h_topology_v = StructuralGNN(h0_v, structural_edge_index)
graph_context = mean_v(h_topology_v)
logits = Classifier(Fusion([0 || graph_context]))
```

This mode is closest to the old etalon/article-style GNN baseline: structure is
the graph on which node state is propagated before graph-level classification.
It should be used to test whether topology helps same-version predictive
monitoring before drawing conclusions about structural drift transfer.

Diagnostics:

1. `topology_state_prefix_mean_abs`
2. `topology_state_prefix_max_abs`
3. `topology_state_gate_mean`
4. `topology_state_gate_max`
5. `topology_graph_context_mean_abs`
6. `topology_graph_context_max_abs`
7. `topology_graph_logits_mean_abs`
8. `topology_graph_entropy`
```

Renumber later sections if needed.

- [ ] **Step 3: Update project state**

In `docs/current/project-state.md`, add `TopologyStateGraphEncoder` to the
`gnn_runtime` section as experimental but implemented once code is merged.

- [ ] **Step 4: Update dissertation changes**

Add a new entry:

```markdown
## 2026-05-16 - TopologyStateGraphEncoder Structural Graph Baseline

- `topic`: etalon/article-like structural graph baseline
- `impact`: critical for methodology
- `status`: planned/implemented after this plan execution
- `related_runtime`: `model.fusion_mode=TopologyStateGraphEncoder`

**Note (ukr):**

Додано окремий baseline, де BPMN/EOPKG topology є самим графом прогнозування:
стан префікса проектується на structural nodes, далі виконується structural GNN
message passing і graph-level readout. Це ближче до старого `etalon`, ніж
`StructuralPriorEncoder`, бо структура не стискається у mean context після
observed encoder, а бере участь у message passing до класифікації.

**Do not forget:**

- порівнювати `Base-UN`, `StructuralPriorEncoder`, `TopologyStateEncoder`,
  `TopologyStateGraphEncoder`, `ClassAwareStructuralScoring`;
- спочатку перевірити same-version користь структури, і лише після цього
  аналізувати transfer між версіями;
- не називати цей режим final drift solution: hard `structural_edge_index`
  може давати structural overfitting;
- показувати diagnostics `topology_graph_context_mean_abs`,
  `topology_graph_logits_mean_abs`, `topology_graph_entropy`.
```

---

## 14. Task 8: Verification Suite

**Files:**

- No source changes in this task; run verification only.

- [ ] **Step 1: Run focused model tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/domain/test_eopkg_topology_state_encoder.py tests/domain/test_eopkg_models_forward.py tests/domain/test_eopkg_gatv2_dual_encoder.py -v
```

Expected:

```text
PASS
```

- [ ] **Step 2: Run application/config tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/application/test_cli_flatten_config.py tests/application/test_trainer_forward_stats_logging.py -v
```

Expected:

```text
PASS
```

- [ ] **Step 3: Run contract and builder tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/contracts/test_graph_tensor_contract_mvp2_optional_fields.py tests/domain/test_dynamic_graph_builder_topology_state.py tests/application/test_trainer_topology_state_contract.py tests/application/test_trainer_preloaded_data.py -v
```

Expected:

```text
PASS
```

- [ ] **Step 4: Run architecture guard**

Run:

```powershell
.\.venv\Scripts\python.exe tools\architecture_guard.py
```

Expected:

```text
PASS / no violations
```

- [ ] **Step 5: Run broad tests if time allows**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -v
```

Expected:

```text
PASS
```

If an unrelated existing failure appears, document exact test id and error in
the implementation report instead of hiding it.

---

## 15. Recommended First Experiments

### 15.1 Same-version structural usefulness

First compare same split and seed:

```text
Base-UN-loan
S_Str-UNc-loan with fusion_mode=StructuralPriorEncoder
S_Str-UNc-loan with fusion_mode=TopologyStateEncoder
S_Str-UNc-loan with fusion_mode=TopologyStateGraphEncoder
S_Str-UNc-loan with fusion_mode=ClassAwareStructuralScoring
```

Primary metrics:

```text
strict_test_macro_f1
test_macro_f1
test_oos_rate / strict OOS metrics if available
confidence / calibration metrics if available
```

Graph-mode diagnostics:

```text
topology_graph_context_mean_abs
topology_graph_context_max_abs
topology_graph_logits_mean_abs
topology_graph_entropy
topology_state_gate_mean
topology_state_gate_max
```

Expected useful signal:

1. Same-version strict-F1 is above or competitive with `Base-UN`.
2. Topology graph logits/context are non-zero and stable.
3. Entropy does not collapse early to near zero unless accuracy also improves.

### 15.2 Stats-backed structural graph baseline

Then compare:

```text
TopologyStateGraphEncoder with structural_stats_beta=0.0
TopologyStateGraphEncoder with structural_stats_beta=0.1
TopologyStateGraphEncoder with structural_stats_beta=0.3
TopologyStateGraphEncoder with structural_stats_beta=1.0
```

This isolates topology identity from runtime/statistical enrichment.

### 15.3 Drift transfer check

Run drift only after same-version behavior is defensible:

```text
Train v1 -> drift test v1,v2,v3,v4
Train v1,v2 -> drift test v1,v2,v3,v4
```

Interpretation rules:

1. If same-version improves but unseen versions degrade, structure is useful but
   overfits topology; this supports the need for transfer methodology.
2. If same-version does not improve, current prefix-state features or graph
   readout are insufficient; do not proceed to strong drift claims.
3. If process-consistency metrics improve without strict-F1 improvement,
   report it separately from exact next-activity prediction.

---

## 16. Risks And Controls

### Risk: observed branch accidentally participates

Control:

- Add the invariance test where `x_cat/x_num` changes but structural tensors
  stay fixed and logits remain identical.

### Risk: duplicate activity ambiguity remains

Control:

- This mode avoids direct duplicate-node `LogSumExp` class inflation, but
  duplicate nodes still receive identical class-level prefix-state features.
  Keep `duplicate_activity_identity_ambiguity` as active debt.

### Risk: hard topology overfitting

Control:

- Treat the mode as a structural graph baseline and same-version usefulness
  test, not as the final drift solution.
- Report `F1(v_seen) - F1(v_unseen)` degradation separately.

### Risk: graph context becomes silent

Control:

- Log `topology_graph_context_mean_abs`,
  `topology_graph_logits_mean_abs`, and entropy.

### Risk: new config hidden in UI

Control:

- Add `TopologyStateGraphEncoder` to `configs/ui/config_catalog.yaml`.
- Add `model.topology_graph_pooling`.
- Make shared `topology_state_*` controls visible for this mode.

### Risk: stale graph cache

Control:

- Old graph cache entries without `struct_prefix_state_x` cannot run this mode.
  Rebuild graph caches for `TopologyStateGraphEncoder` experiments.

### Risk: checkpoint mismatch

Control:

- This new `fusion_mode` requires fresh training. Do not reuse checkpoints from
  `Base-UN`, `StructuralPriorEncoder`, `TopologyStateEncoder`, or
  `ClassAwareStructuralScoring`.
- The plan intentionally reuses existing trainable modules to avoid adding
  new state-dict keys for unrelated modes.

---

## 17. Implementation Report Template

After implementation, create:

```text
docs/worklogs/MVP2_5_Topology_State_Graph_Encoder_Fusion_Report_2026-05-16.MD
```

Report sections:

```markdown
# MVP2.5 TopologyStateGraphEncoder Fusion Implementation Report

## Summary

## Files Changed

## Runtime Semantics

## Config Keys

## Diagnostics

## Tests Run

## How To Run First Experiment

## Known Limits

## Documentation Drift Check
```

Include exact verification commands and results.

---

## 18. Self-Review

### Spec coverage

- The plan defines the new `fusion_mode`.
- The plan keeps existing contracts and graph builder unchanged.
- The plan distinguishes graph-level readout from node-to-class pooling.
- The plan includes tests for aliases, forward behavior, pure structural path,
  CLI config, trainer diagnostics, and docs.

### Placeholder scan

No `TBD`, `TODO`, or unspecified implementation steps are left.

### Type consistency

The plan consistently uses:

```text
model.fusion_mode=TopologyStateGraphEncoder
canonical internal name: topology_state_graph_encoder
model.topology_graph_pooling=mean
struct_prefix_state_x: [V, 6] or [B, V, 6]
last_topology_graph_context
last_topology_graph_logits
last_topology_graph_entropy
```

### Architecture consistency

The plan respects current clean architecture boundaries:

- model behavior stays in `src/domain/models/eopkg_models.py`;
- CLI only forwards config;
- trainer only logs diagnostics;
- graph builder data contract remains unchanged;
- docs and UI catalog are updated explicitly.
