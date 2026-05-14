# MVP2.5 Early TopologyStateEncoder Fusion Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans`
> or `executing-plans` to implement this plan task-by-task. Use
> `test-driven-development` for every implementation task and
> `verification-before-completion` before reporting completion.

**Goal:** Add `model.fusion_mode: TopologyStateEncoder` as an early
input-level fusion mode for `EOPKGGATv2` when
`experiment.structural_mode: true`, while preserving observed-only behavior
when `experiment.structural_mode: false`.

**Architecture:** Current structural modes are context-level or logit-level
fusion: the observed prefix is first encoded as an instance graph and EOPKG is
added later. `TopologyStateEncoder` changes the integration level: the observed
prefix state is projected onto EOPKG topology nodes before topology message
passing. This is an experimental ablation, not the final OOD-drift solution:
it tests whether early topology injection changes the degradation pattern and
helps explain structural overfitting / negative transfer.

**Tech Stack:** Python, PyTorch, PyTorch Geometric, existing
`GraphTensorContract`, `DynamicGraphBuilder`, `EOPKGGATv2`, pytest.

---

## 1. Problem Statement

Recent experiments showed that `ClassAwareStructuralScoring` participates in
training, but does not provide stable strict-F1 improvement under structural
drift. The likely root cause is integration level:

- `ClassMeanAttention` and `ClassMeanConcat` compress structure into a context
  vector after observed-prefix encoding.
- `ClassAwareStructuralScoring` adds structural class logits after observed
  context is already built.
- None of the current modes lets topology shape the prefix representation
  before message passing.

The older `etalon` implementation had a stronger structure signal because the
process topology itself was the prediction graph, and the observed prefix was
encoded as node state (`active_mask`, timestamps, runtime attributes) on that
same topology.

`TopologyStateEncoder` should test this hypothesis directly, but the expected
outcome must be interpreted carefully. A hard topology encoder can improve
seen-version metrics while worsening unseen-version drift because GNN message
passing is bound to `structural_edge_index` and can overfit to the training
topology spectrum.

---

## 1.1 Reviewer Feedback Resolution

This plan incorporates the external review as follows:

1. **Hard structural message passing risk: accepted.**
   `TopologyStateEncoder` is not treated as a final drift-generalization
   architecture. It is a controlled ablation for early fusion and structural
   overfitting. Future flexible alternatives remain `Layerwise Structural
   Cross-Attention`, side-tuning, or topology-as-soft-bias attention.
2. **Duplicate activity cardinality bias: accepted.**
   Raw `LogSumExp` class pooling is replaced by cardinality-normalized
   `LogMeanExp` by default.
3. **Lossy prefix compression: accepted.**
   The mode intentionally discards the observed instance-graph encoder for the
   first ablation. This must be reported as an information bottleneck and not
   compared as a capacity-equivalent replacement for Base-UN.
4. **Overconfidence risk: accepted.**
   The plan adds entropy diagnostics, cardinality diagnostics, conservative
   gated prefix-state injection, and explicit structural-overfitting
   interpretation rules.
5. **Topology version policy: clarified.**
   Runtime must use topology resolved from each prefix `process_version` /
   `as_of` snapshot. Even with correct v2 topology at inference, a model trained
   only on v1 can still suffer aggregation mismatch because learned GNN weights
   were optimized on v1 topology neighborhoods.

---

## 2. Non-Goals

- Do not implement the future Reliability Semaphore or dual trust architecture.
- Do not solve duplicate activity identity ambiguity in this task.
- Do not change `experiment.structural_mode` into an enum.
- Do not remove or rewrite `ClassMeanAttention`, `ClassMeanConcat`, or
  `ClassAwareStructuralScoring`.
- Do not make topology-state the default mode.
- Do not use raw BPMN/Neo4j reads in train/eval/infer. Runtime must keep using
  `IKnowledgeGraphPort` and prebuilt topology/stats artifacts.

---

## 3. Terminology

- `observed-only`: `experiment.structural_mode: false`; model ignores structural
  payloads and uses the observed instance graph only.
- `late fusion`: structural signal is added after observed-prefix encoding.
- `early fusion`: observed prefix state is added to topology nodes before graph
  message passing.
- `TopologyStateEncoder`: early fusion mode where EOPKG topology nodes carry
  prefix-state features.
- `struct_prefix_state_x`: per-sample tensor with observed-prefix state features
  for topology nodes.

---

## 4. Target Config Semantics

Canonical config:

```yaml
experiment:
  structural_mode: true

model:
  type: EOPKGGATv2
  fusion_mode: TopologyStateEncoder
```

Required behavior:

1. If `experiment.structural_mode: false`, CLI override sets
   `model.structural_mode: false`, and `EOPKGGATv2.forward()` returns the
   observed-only path regardless of `model.fusion_mode`.
2. If `experiment.structural_mode: true` and
   `model.fusion_mode: TopologyStateEncoder`, the model uses topology-state
   tensors for prediction.
3. If topology tensors or `struct_prefix_state_x` are missing in
   `TopologyStateEncoder`, fail fast with a clear `ValueError`. Silent fallback
   would invalidate the experiment.

Recommended aliases:

```text
TopologyStateEncoder -> topology_state_encoder
topology_state_encoder -> topology_state_encoder
topology_state -> topology_state_encoder
EarlyTopologyStateEncoder -> topology_state_encoder
```

---

## 5. Data Contract Design

Extend `GraphTensorContract` with one optional field:

```python
struct_prefix_state_x: NotRequired[torch.FloatTensor | None]
```

Shape:

```text
single graph: [V, F_state]
batched graph: [B, V, F_state]
```

Where:

- `V` is the number of structural topology nodes used by
  `structural_edge_index`;
- `B` is batch size;
- `F_state` is the number of prefix-state features;
- rows align with `struct_node_to_class_index`.

Initial prefix-state feature vector per topology node:

```text
0 prefix_executed_count_log1p
1 prefix_was_executed
2 prefix_is_last_event
3 prefix_last_position_norm
4 prefix_recency_norm
5 prefix_is_active_after_complete
```

Feature definitions:

```text
prefix_executed_count_log1p = log1p(number of prefix events mapped to this class)
prefix_was_executed = 1.0 if count > 0 else 0.0
prefix_is_last_event = 1.0 if this node/class matches the last prefix event else 0.0
prefix_last_position_norm = (last_position + 1) / prefix_length, or 0.0 if unseen
prefix_recency_norm = 1 / (1 + prefix_length - 1 - last_position), or 0.0 if unseen
prefix_is_active_after_complete = 1.0 if last event says this activity is active after completion
```

Duplicate activity caveat:

- Current projection is mostly class-level after `collapse_for_prediction`.
- If several topology nodes map to the same activity class, all such nodes
  receive the same prefix-state vector.
- This is acceptable for the first ablation and must be documented as
  compatible with, but not a solution to, `duplicate_activity_identity_ambiguity`.

---

## 6. Model Design

Add a new branch to `EOPKGGATv2`:

```text
if not structural_mode:
    observed-only forward
elif fusion_mode == topology_state_encoder:
    topology-state forward
else:
    existing structural modes
```

Topology-state forward:

```text
base_struct_node_features =
    LayerNorm(identity_embedding + structural_stats_beta * stats_projection(struct_x))

prefix_state_features =
    prefix_state_projection(struct_prefix_state_x)

prefix_gate =
    sigmoid(prefix_gate_projection(struct_prefix_state_x) + topology_state_gate_init_bias)

h0 =
    LayerNorm(base_struct_node_features + topology_state_beta * prefix_gate * prefix_state_features)

h_topology =
    struct_gnn(h0, structural_edge_index, structural_edge_weight if supported)

node_logits =
    topology_state_node_head(h_topology)  # [B, V]

class_logits =
    LogMeanExp(node_logits over nodes where struct_node_to_class_index == class_j)
```

New model parameters:

```yaml
model.topology_state_beta: 0.5
model.topology_state_beta_max: 2.0
model.topology_state_dropout: inherits model.dropout unless explicitly set
model.topology_state_gate_init_bias: -2.0
model.topology_state_class_pooling: logmeanexp
```

Pooling policy:

```text
logmeanexp(class_j) = logsumexp(node_logits_j) - log(number_of_nodes_for_class_j)
```

Default `logmeanexp` prevents a class logit from increasing only because a new
process version contains more duplicated topology nodes for the same activity
label. Supported values for controlled ablations:

```text
logmeanexp | mean | max | logsumexp
```

`logsumexp` is allowed only for explicit diagnostic runs because it is
cardinality-sensitive.

Implementation note:

- `struct_x` remains statistics/topology enrichment.
- `struct_prefix_state_x` remains observed-prefix state.
- Do not append prefix-state features into `struct_x`; keeping them separate
  preserves the stats contract and makes diagnostics easier.
- Clamp `topology_state_beta` to `[0, topology_state_beta_max]` in forward.
- Initialize gate bias negative so prefix-state injection starts conservative
  instead of immediately dominating topology identity.

Initial output policy:

- `TopologyStateEncoder` should return `class_logits` from topology-state
  branch only.
- Do not add observed logits as a residual in the first implementation. A
  residual would make the ablation harder to interpret.
- The observed-only branch is still available through
  `experiment.structural_mode: false`.

Diagnostics to expose on the model:

```python
last_topology_state_node_logits
last_topology_state_class_logits
last_topology_state_prefix_x
last_topology_state_prefix_mean_abs
last_topology_state_prefix_max_abs
last_topology_state_entropy
last_topology_state_mean_class_cardinality
last_topology_state_max_class_cardinality
last_topology_state_gate_mean
last_topology_state_gate_max
```

Trainer metric names:

```text
training_topology_state_prefix_mean_abs
validation_topology_state_prefix_mean_abs
test_topology_state_prefix_mean_abs
validation_topology_state_entropy
validation_topology_state_mean_class_cardinality
validation_topology_state_gate_mean
```

---

## 7. Batch Contract Design

Existing structural modes can use the first graph structural payload in a batch
because topology and stats payloads are often shared by snapshot identity.
`TopologyStateEncoder` cannot use that shortcut because `struct_prefix_state_x`
is different for every prefix.

Required batching behavior:

1. `DynamicGraphBuilder.build_graph()` emits `struct_prefix_state_x` per prefix.
2. `_build_loader()` copies `struct_prefix_state_x` into each PyG `Data`.
3. `_data_to_contract()` detects batched `struct_prefix_state_x`.
4. For batched data, contract should contain:

```text
struct_prefix_state_x: [B, V, F_state]
struct_x: [V, F_stats]                         # shared payload selected as today
structural_edge_index: [2, E]                  # shared payload selected as today
structural_edge_weight: [E]
struct_node_to_class_index: [V]
```

Preferred implementation:

- Use `data.to_data_list()` when available and stack each
  `item.struct_prefix_state_x`.
- Validate all items have the same `V` and `F_state`.
- If shapes differ in one batch, raise a clear error for
  `TopologyStateEncoder` and add a future debt note for snapshot-homogeneous
  batching.

Fallback implementation:

- If `to_data_list()` is unavailable, infer `V` from
  `struct_node_to_class_index.numel()` and reshape concatenated
  `struct_prefix_state_x` from `[B * V, F_state]` to `[B, V, F_state]`.

---

## 8. Files To Modify

### Domain Contract

- Modify: `src/domain/entities/tensor_contract.py`
  - Add `struct_prefix_state_x`.

### Graph Builder

- Modify: `src/domain/services/dynamic_graph_builder.py`
  - Add `_build_struct_prefix_state_x`.
  - Add prefix-state tensor to `contract`.
  - Keep topology/stats cache unchanged: prefix-state is per-prefix and must not
    be stored inside compiled topology cache.

### Trainer / PyG Bridge

- Modify: `src/application/use_cases/trainer.py`
  - Copy `struct_prefix_state_x` into `Data`.
  - Rehydrate from sharded/prebuilt datasets.
  - Convert batched `Data` to contract with `[B, V, F_state]`.
  - Add forward-stat diagnostics for topology-state prefix features.

### Model

- Modify: `src/domain/models/eopkg_models.py`
  - Add fusion aliases.
  - Add `topology_state_beta`, `topology_state_beta_max`,
    `topology_state_gate_init_bias`, `topology_state_class_pooling`, and
    optional dropout.
  - Add prefix-state projection and topology-state node head.
  - Add cardinality-normalized node-to-class pooling.
  - Add entropy/gate/cardinality diagnostics.
  - Add fail-fast validations.
  - Add topology-state forward path.

### CLI / Config Wiring

- Modify: `src/cli.py`
  - Pass `model.topology_state_beta`.
  - Pass `model.topology_state_beta_max`.
  - Pass `model.topology_state_gate_init_bias`.
  - Pass `model.topology_state_class_pooling`.
  - Pass `model.topology_state_dropout` if configured.
  - Bump graph dataset cache schema because new graph samples include
    `struct_prefix_state_x`.

- Modify: `configs/ui/config_catalog.yaml`
  - Add `TopologyStateEncoder` to `model.fusion_mode.enum`.
  - Add config keys:
    - `model.topology_state_beta`
    - `model.topology_state_beta_max`
    - `model.topology_state_gate_init_bias`
    - `model.topology_state_class_pooling`
    - `model.topology_state_dropout`

### Documentation

- Modify: `docs/GNN_RUNTIME_MVP2_5.MD`
  - Add `TopologyStateEncoder` as input-level fusion.
  - Explain that it is not the future Reliability Semaphore dual architecture.

- Modify: `docs/DATA_MODEL_MVP2_5.MD`
  - Add `struct_prefix_state_x` to tensor contract.

- Modify: `docs/current/project-state.md`
  - Update current GNN runtime summary after implementation.

- Modify: `docs/dissertation_changes.MD`
  - Add note that MVP2.5 now distinguishes context-level, logit-level, and
    input-level structure fusion.

- Optional if the implementation reveals a durable decision:
  - Create ADR: `docs/adr/00xx-topology-state-encoder-fusion.md`.

### Tests

- Create or modify:
  - `tests/domain/test_dynamic_graph_builder_topology_state.py`
  - `tests/domain/test_eopkg_topology_state_encoder.py`
  - `tests/application/test_trainer_topology_state_contract.py`
  - `tests/application/test_cli_flatten_config.py`
  - `tests/contracts/test_graph_tensor_contract_mvp2_optional_fields.py`

---

## 9. Implementation Tasks

### Task 1: Contract Field

**Files:**

- Modify: `src/domain/entities/tensor_contract.py`
- Modify: `tests/contracts/test_graph_tensor_contract_mvp2_optional_fields.py`

- [ ] **Step 1: Write failing contract test**

Add assertion that `GraphTensorContract` accepts:

```python
"struct_prefix_state_x": torch.tensor(
    [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
     [1.0, 1.0, 1.0, 1.0, 1.0, 0.0]],
    dtype=torch.float32,
)
```

Expected:

```text
contract["struct_prefix_state_x"].dtype == torch.float32
contract["struct_prefix_state_x"].shape == torch.Size([2, 6])
```

- [ ] **Step 2: Run failing test**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/contracts/test_graph_tensor_contract_mvp2_optional_fields.py -v
```

- [ ] **Step 3: Add contract field**

Add:

```python
struct_prefix_state_x: NotRequired[torch.FloatTensor | None]
```

- [ ] **Step 4: Run contract test**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/contracts/test_graph_tensor_contract_mvp2_optional_fields.py -v
```

Expected: PASS.

---

### Task 2: DynamicGraphBuilder Prefix-State Tensor

**Files:**

- Modify: `src/domain/services/dynamic_graph_builder.py`
- Create: `tests/domain/test_dynamic_graph_builder_topology_state.py`

- [ ] **Step 1: Write failing test for prefix-state features**

Construct a prefix:

```text
prefix events: Start, Approve, Approve
target: End
activity_vocab: <UNK>, Start, Approve, End
topology edges: Start -> Approve -> End
```

Expected:

```text
struct_prefix_state_x.shape == [num_classes, 6]
Start:
  count_log1p == log1p(1)
  was_executed == 1
  is_last == 0
Approve:
  count_log1p == log1p(2)
  was_executed == 1
  is_last == 1
  prefix_last_position_norm == 1.0
  prefix_recency_norm == 1.0
End:
  count_log1p == 0
  was_executed == 0
  is_last == 0
```

- [ ] **Step 2: Run failing test**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/domain/test_dynamic_graph_builder_topology_state.py -v
```

- [ ] **Step 3: Implement `_build_struct_prefix_state_x`**

Add helper:

```python
def _build_struct_prefix_state_x(
    self,
    *,
    prefix: PrefixSlice,
    activity_vocab: Dict[str, int],
    struct_node_to_class_index: torch.Tensor,
) -> torch.Tensor:
    prefix_length = max(int(len(prefix.prefix_events)), 1)
    num_nodes = int(struct_node_to_class_index.numel())
    state = torch.zeros((num_nodes, 6), dtype=torch.float32)
    counts_by_class: dict[int, int] = {}
    last_pos_by_class: dict[int, int] = {}
    # Fill counts_by_class and last_pos_by_class by resolving each prefix event.
    # Then copy the class-level state into every topology node row.
    return state
```

Rules:

- Resolve event activity via `feature_encoder.resolve_event_feature`.
- Count prefix events by activity class.
- Track last occurrence position per activity class.
- Resolve active candidates from last event using existing
  `_extract_active_candidates`.
- For node rows where `struct_node_to_class_index[row] < 0`, emit zeros.
- For duplicate nodes with the same class id, emit the same class-state vector.

- [ ] **Step 4: Attach tensor to contract**

After compiled topology is resolved:

```python
contract["struct_prefix_state_x"] = self._build_struct_prefix_state_x(
    prefix=prefix,
    activity_vocab=activity_vocab,
    struct_node_to_class_index=struct_node_to_class_index,
)
```

Attach only when structural topology exists.

- [ ] **Step 5: Run graph-builder tests**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/domain/test_dynamic_graph_builder_topology_state.py tests/domain/test_dynamic_graph_builder_masks.py tests/domain/test_dynamic_graph_builder_stats_matrix_mapping.py -v
```

Expected: PASS.

---

### Task 3: Trainer Data Bridge

**Files:**

- Modify: `src/application/use_cases/trainer.py`
- Create: `tests/application/test_trainer_topology_state_contract.py`

- [ ] **Step 1: Write failing test for loader payload**

Assert `_build_loader()` copies `struct_prefix_state_x` from contract into PyG
`Data`.

Expected:

```text
hasattr(data, "struct_prefix_state_x") is True
data.struct_prefix_state_x.shape == [V, 6]
```

- [ ] **Step 2: Write failing test for batched contract**

Create two `Data` samples with same structural topology but different
`struct_prefix_state_x`. Batch them with PyG `DataLoader`, call
`Trainer._data_to_contract(batch)`.

Expected:

```text
contract["struct_prefix_state_x"].shape == [2, V, 6]
contract["struct_prefix_state_x"][0] equals sample_a prefix state
contract["struct_prefix_state_x"][1] equals sample_b prefix state
```

- [ ] **Step 3: Run failing tests**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/application/test_trainer_topology_state_contract.py -v
```

- [ ] **Step 4: Copy field in `_build_loader`**

Add:

```python
struct_prefix_state_x = contract.get("struct_prefix_state_x")
if isinstance(struct_prefix_state_x, torch.Tensor):
    payload["struct_prefix_state_x"] = struct_prefix_state_x
```

- [ ] **Step 5: Add batch extraction helper**

Add helper:

```python
def _select_struct_prefix_state_for_forward(
    self,
    data: Data,
    *,
    graph_count: int,
    struct_node_count: int | None,
) -> torch.Tensor | None:
    # Prefer stacking per-graph tensors from data.to_data_list().
    # Fall back to reshaping a concatenated [B * V, F] tensor when V is known.
    return None
```

Behavior:

- Prefer `data.to_data_list()` and `torch.stack`.
- Validate same shape across samples.
- Fallback reshape from concatenated `[B * V, F]` using `struct_node_count`.

- [ ] **Step 6: Attach to contract in `_data_to_contract`**

Add:

```python
struct_prefix_state_x = self._select_struct_prefix_state_for_forward(
    data,
    graph_count=graph_count,
    struct_node_count=struct_node_count,
)
if isinstance(struct_prefix_state_x, torch.Tensor):
    contract["struct_prefix_state_x"] = struct_prefix_state_x.float()
```

- [ ] **Step 7: Run trainer bridge tests**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/application/test_trainer_topology_state_contract.py tests/application/test_trainer_preloaded_data.py tests/application/test_trainer_forward_stats_logging.py -v
```

Expected: PASS.

---

### Task 4: EOPKGGATv2 TopologyStateEncoder

**Files:**

- Modify: `src/domain/models/eopkg_models.py`
- Create: `tests/domain/test_eopkg_topology_state_encoder.py`
- Modify: `tests/domain/test_eopkg_models_forward.py`
- Modify: `tests/domain/test_eopkg_gatv2_dual_encoder.py`

- [ ] **Step 1: Write failing alias test**

Expected aliases:

```python
("TopologyStateEncoder", "topology_state_encoder")
("topology_state_encoder", "topology_state_encoder")
("topology_state", "topology_state_encoder")
("EarlyTopologyStateEncoder", "topology_state_encoder")
```

- [ ] **Step 2: Write failing forward-shape test**

Contract:

```text
x_cat/x_num/edge_index: valid observed graph
structural_edge_index: [2, E]
struct_x: [V, F_stats]
struct_node_to_class_index: [V]
struct_prefix_state_x: [B, V, 6]
```

Expected:

```text
logits.shape == [B, output_dim]
model.last_topology_state_class_logits.shape == [B, output_dim]
model.last_topology_state_node_logits.shape == [B, V]
model.last_topology_state_entropy is not None
model.last_topology_state_gate_mean is not None
```

- [ ] **Step 3: Write failing missing-prefix-state test**

When `structural_mode=True` and `fusion_mode=TopologyStateEncoder`, missing
`struct_prefix_state_x` raises:

```text
TopologyStateEncoder requires struct_prefix_state_x
```

- [ ] **Step 4: Write observed-only override test**

When `structural_mode=False` and `fusion_mode=TopologyStateEncoder`, the model
must ignore structural tensors and match observed-only output:

```text
logits_without_struct == logits_with_struct
```

- [ ] **Step 5: Run failing tests**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/domain/test_eopkg_topology_state_encoder.py tests/domain/test_eopkg_models_forward.py tests/domain/test_eopkg_gatv2_dual_encoder.py -v
```

- [ ] **Step 6: Add aliases and validation**

Extend alias map and allowed set:

```python
"topologystateencoder": "topology_state_encoder",
"topology_state_encoder": "topology_state_encoder",
"topology_state": "topology_state_encoder",
"earlytopologystateencoder": "topology_state_encoder",
```

Allowed set:

```python
{
    "class_mean_attention",
    "class_mean_concat",
    "class_aware_structural_scoring",
    "topology_state_encoder",
}
```

- [ ] **Step 7: Add modules**

Add:

```python
self.topology_state_beta = float(topology_state_beta)
self.topology_state_beta_max = float(topology_state_beta_max)
self.topology_state_gate_init_bias = float(topology_state_gate_init_bias)
self.topology_state_class_pooling = str(topology_state_class_pooling or "logmeanexp").strip().lower()
self.topology_state_proj: nn.Linear | None = None
self.topology_state_gate_proj: nn.Linear | None = None
self.topology_state_norm = nn.LayerNorm(self.struct_hidden_dim)
self.topology_state_node_head = nn.Linear(self.struct_hidden_dim, 1)
```

Validate:

```python
if self.topology_state_class_pooling not in {"logmeanexp", "mean", "max", "logsumexp"}:
    raise ValueError(
        "Unsupported model.topology_state_class_pooling "
        f"'{self.topology_state_class_pooling}'. "
        "Available: ['logmeanexp', 'mean', 'max', 'logsumexp']"
    )
```

- [ ] **Step 8: Add `_build_topology_state_features`**

Requirements:

- Accept prefix state as `[V, F]` or `[B, V, F]`.
- Project prefix-state features into `struct_hidden_dim`.
- Build a conservative learned gate from raw prefix-state features.
- Combine with base structural node features:

```python
gate = torch.sigmoid(self.topology_state_gate_proj(prefix_state_x) + self.topology_state_gate_init_bias)
beta = torch.clamp(
    torch.tensor(self.topology_state_beta, device=prefix_state_x.device),
    min=0.0,
    max=float(self.topology_state_beta_max),
)
h0 = self.topology_state_norm(
    base_struct.unsqueeze(0) + beta * gate * prefix_state_proj
)
```

- [ ] **Step 9: Add batched topology GNN helper**

Implement either:

1. initial safe loop over `B`; or
2. vectorized edge-index offset batching.

Recommended first implementation: vectorized batching because `eval_drift`
already runs large datasets.

Expected output:

```text
h_topology: [B, V, D]
```

- [ ] **Step 10: Add cardinality-safe node-to-class pooling**

Default pooling must be `logmeanexp`, not raw `logsumexp`:

```python
values = node_logits[:, node_mask]
if self.topology_state_class_pooling == "logmeanexp":
    class_logits[:, class_idx] = torch.logsumexp(values, dim=1) - math.log(int(values.size(1)))
elif self.topology_state_class_pooling == "mean":
    class_logits[:, class_idx] = values.mean(dim=1)
elif self.topology_state_class_pooling == "max":
    class_logits[:, class_idx] = values.max(dim=1).values
elif self.topology_state_class_pooling == "logsumexp":
    class_logits[:, class_idx] = torch.logsumexp(values, dim=1)
```

Ignore nodes with `struct_node_to_class_index == -1`.

Also compute:

```python
class_cardinalities = torch.bincount(valid_node_to_class, minlength=num_classes)
```

Store mean/max class cardinality diagnostics.

- [ ] **Step 11: Branch early in `forward`**

For `EOPKGGATv2.forward()`:

```python
x, edge_index, batch = self._encode_input(contract)
if not self.structural_mode:
    return observed-only path
if self.fusion_mode == "topology_state_encoder":
    return self._forward_topology_state(contract, device=x.device)
else:
    existing observed + structural late-fusion path
```

This avoids unnecessary observed GNN computation for topology-state mode.

- [ ] **Step 12: Run model tests**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/domain/test_eopkg_topology_state_encoder.py tests/domain/test_eopkg_models_forward.py tests/domain/test_eopkg_gatv2_dual_encoder.py -v
```

Expected: PASS.

---

### Task 5: CLI, Cache Schema, and UI Catalog

**Files:**

- Modify: `src/cli.py`
- Modify: `configs/ui/config_catalog.yaml`
- Modify: `tests/application/test_cli_flatten_config.py`

- [ ] **Step 1: Write failing CLI config test**

Add expected factory kwargs:

```python
model_cfg={
    "fusion_mode": "TopologyStateEncoder",
    "topology_state_beta": 0.5,
    "topology_state_beta_max": 2.0,
    "topology_state_gate_init_bias": -2.0,
    "topology_state_class_pooling": "logmeanexp",
    "topology_state_dropout": 0.0,
}
```

Expected:

```text
kwargs["fusion_mode"] == "TopologyStateEncoder"
kwargs["topology_state_beta"] == 0.5
kwargs["topology_state_beta_max"] == 2.0
kwargs["topology_state_gate_init_bias"] == -2.0
kwargs["topology_state_class_pooling"] == "logmeanexp"
kwargs["topology_state_dropout"] == 0.0
```

- [ ] **Step 2: Run failing CLI test**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/application/test_cli_flatten_config.py -v
```

- [ ] **Step 3: Pass new kwargs**

Add to `_build_model_factory_kwargs`:

```python
"topology_state_beta": float(model_cfg.get("topology_state_beta", 0.5)),
"topology_state_beta_max": float(model_cfg.get("topology_state_beta_max", 2.0)),
"topology_state_gate_init_bias": float(model_cfg.get("topology_state_gate_init_bias", -2.0)),
"topology_state_class_pooling": str(model_cfg.get("topology_state_class_pooling", "logmeanexp")),
"topology_state_dropout": float(model_cfg.get("topology_state_dropout", model_cfg.get("dropout", 0.2))),
```

- [ ] **Step 4: Bump graph dataset cache schema**

If `src/cli.py` contains `GRAPH_DATASET_CACHE_SCHEMA = 3`, bump to:

```python
GRAPH_DATASET_CACHE_SCHEMA = 4
```

Reason: prebuilt graph samples for topology-state mode now include
`struct_prefix_state_x`; old cache entries are not sufficient for this mode.

- [ ] **Step 5: Update catalog**

Add enum value:

```yaml
- TopologyStateEncoder
```

Add:

```yaml
model.topology_state_beta:
  path: model.topology_state_beta
  section: model
  label: "Model: Topology-state beta"
  description: Scales gated observed prefix-state projection before topology message passing in TopologyStateEncoder.
  affects: Strength of early observed-state injection into EOPKG topology.
  default: 0.5
  type: number

model.topology_state_beta_max:
  path: model.topology_state_beta_max
  section: model
  label: "Model: Topology-state beta max"
  description: Forward-time upper clamp for topology_state_beta.
  affects: Prevents over-amplification of prefix-state injection.
  default: 2.0
  type: number

model.topology_state_gate_init_bias:
  path: model.topology_state_gate_init_bias
  section: model
  label: "Model: Topology-state gate init bias"
  description: Initial bias for sigmoid gate over prefix-state projection.
  affects: Controls whether early topology-state fusion starts conservative or aggressive.
  default: -2.0
  type: number

model.topology_state_class_pooling:
  path: model.topology_state_class_pooling
  section: model
  label: "Model: Topology-state class pooling"
  description: Node-to-class pooling for TopologyStateEncoder; logmeanexp is cardinality-normalized and is the default.
  affects: Duplicate activity cardinality bias and calibration.
  default: logmeanexp
  enum:
    - logmeanexp
    - mean
    - max
    - logsumexp

model.topology_state_dropout:
  path: model.topology_state_dropout
  section: model
  label: "Model: Topology-state dropout"
  description: Dropout applied to prefix-state projection in TopologyStateEncoder.
  affects: Regularization of early topology-state fusion.
  default: null
  type: number
```

- [ ] **Step 6: Run config tests**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/application/test_cli_flatten_config.py -v
```

Expected: PASS.

---

### Task 6: Trainer Diagnostics

**Files:**

- Modify: `src/application/use_cases/trainer.py`
- Modify: `tests/application/test_trainer_forward_stats_logging.py`

- [ ] **Step 1: Write failing diagnostic test**

Use a small diagnostic model with:

```python
last_topology_state_prefix_mean_abs = torch.tensor(0.5)
last_topology_state_prefix_max_abs = torch.tensor(1.0)
last_topology_state_entropy = torch.tensor(1.2)
last_topology_state_mean_class_cardinality = torch.tensor(1.1)
last_topology_state_max_class_cardinality = torch.tensor(2.0)
last_topology_state_gate_mean = torch.tensor(0.12)
last_topology_state_gate_max = torch.tensor(0.35)
```

Expected logged metrics include:

```text
topology_state_prefix_mean_abs
topology_state_prefix_max_abs
topology_state_entropy
topology_state_mean_class_cardinality
topology_state_max_class_cardinality
topology_state_gate_mean
topology_state_gate_max
```

- [ ] **Step 2: Run failing test**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/application/test_trainer_forward_stats_logging.py -v
```

- [ ] **Step 3: Add diagnostic extraction**

Extend forward stats collection to inspect model attributes:

```python
last_topology_state_prefix_mean_abs
last_topology_state_prefix_max_abs
last_topology_state_class_logits
last_topology_state_entropy
last_topology_state_mean_class_cardinality
last_topology_state_max_class_cardinality
last_topology_state_gate_mean
last_topology_state_gate_max
```

Track by phase:

```text
training_
validation_
test_
```

- [ ] **Step 4: Run diagnostic tests**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/application/test_trainer_forward_stats_logging.py -v
```

Expected: PASS.

---

### Task 7: Documentation Updates

**Files:**

- Modify: `docs/GNN_RUNTIME_MVP2_5.MD`
- Modify: `docs/DATA_MODEL_MVP2_5.MD`
- Modify: `docs/current/project-state.md`
- Modify: `docs/dissertation_changes.MD`

- [ ] **Step 1: Update GNN runtime docs**

Add section:

```text
TopologyStateEncoder is an input-level structural fusion mode. It projects
observed prefix state onto EOPKG topology nodes before structural GNN message
passing. It is an experimental ablation for early structural fusion and
structural-overfitting analysis. It is not the future Reliability Semaphore
dual architecture and is not claimed as the final OOD-drift solution.
```

- [ ] **Step 2: Update data model docs**

Document:

```text
struct_prefix_state_x: [V, F_state] or [B, V, F_state]
```

List the six prefix-state features.

Document:

```text
topology_state_class_pooling=logmeanexp is the default to avoid duplicate-node
cardinality amplification.
```

- [ ] **Step 3: Update project state**

Mention:

```text
EOPKGGATv2 supports TopologyStateEncoder as experimental input-level fusion.
```

- [ ] **Step 4: Update dissertation changes**

Add note:

```text
MVP2.5 now distinguishes context-level, logit-level, and input-level EOPKG
fusion. TopologyStateEncoder should be described as early fusion, while the
Reliability Semaphore remains a later trust/competence layer.
```

Add warning:

```text
TopologyStateEncoder may improve seen-version metrics but can still show
negative transfer under structural drift because message passing remains bound
to hard topology edges.
```

---

### Task 8: Verification

**Files:**

- No direct file edits.

- [ ] **Step 1: Run focused test set**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/contracts/test_graph_tensor_contract_mvp2_optional_fields.py tests/domain/test_dynamic_graph_builder_topology_state.py tests/domain/test_eopkg_topology_state_encoder.py tests/domain/test_eopkg_models_forward.py tests/domain/test_eopkg_gatv2_dual_encoder.py tests/application/test_trainer_topology_state_contract.py tests/application/test_trainer_preloaded_data.py tests/application/test_trainer_forward_stats_logging.py tests/application/test_cli_flatten_config.py -v
```

Expected: PASS.

- [ ] **Step 2: Run architecture guard**

```powershell
.\.venv\Scripts\python.exe tools\architecture_guard.py
```

Expected:

```text
[ARCH_GUARD] OK
```

- [ ] **Step 3: Run regression pool if focused tests pass**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -v
```

Expected: PASS or only unrelated existing failures explicitly documented.

---

## 10. Experiment Validation Plan

After implementation, do not judge the mode from a full dataset immediately.
Use controlled small runs first.

### Smoke config

Use the same loan dataset and fraction:

```yaml
experiment:
  structural_mode: true
  fraction: 0.01

model:
  type: EOPKGGATv2
  fusion_mode: TopologyStateEncoder
  topology_state_beta: 0.5
  topology_state_class_pooling: logmeanexp
  topology_state_gate_init_bias: -2.0
```

Compare against:

```text
Base-UN fraction=0.01
ClassAwareStructuralScoring fraction=0.01
TopologyStateEncoder fraction=0.01
```

### Expected logs

Look for:

```text
fusion_mode=TopologyStateEncoder
Forward stats [validation]: topology_state_prefix_mean_abs=<non_zero_value>
Forward stats [validation]: topology_state_entropy=<finite_value>
Forward stats [validation]: topology_state_gate_mean=<finite_value>
struct_x_batches=<positive_count>
struct_edge_batches=<positive_count>
```

### Acceptance criteria

Minimum implementation validity:

- no fallback to observed-only when structural mode is true;
- no missing `struct_prefix_state_x`;
- train/validation/test run without tensor shape mismatch;
- `TopologyStateEncoder` metrics show non-zero prefix state signal;
- entropy is finite and does not collapse to near-zero from the first epochs;
- class cardinality diagnostics are present when duplicate mappings exist;
- `experiment.structural_mode: false` still reproduces observed-only behavior.

Research validity:

- compare strict-F1 and process-consistency metrics separately;
- compare entropy/confidence shifts to detect structural overconfidence;
- compare degradation by version:

```text
F1(v_seen) - F1(v_unseen)
```

- do not claim structure helps until repeated seeds support it;
- treat worse unseen-version strict-F1 as valid evidence of structural
  overfitting, not as an implementation failure, if implementation diagnostics
  are clean.

---

## 11. Risks And Controls

### Risk: Batch shape mismatch

Cause: different topology node counts in one batch.

Control:

- fail fast for topology-state mode;
- later reuse `snapshot_homogeneous_batching` or add a topology-homogeneous
  sampler if needed.

### Risk: Duplicate activity ambiguity

Cause: multiple BPMN nodes can map to one activity class.

Control:

- duplicate nodes receive same class-level prefix-state vector in this first
  implementation;
- default node-to-class pooling is `logmeanexp`, not raw `logsumexp`, so adding
  duplicated nodes does not automatically inflate class logits;
- log mean/max class cardinality to make duplicate-node effects visible;
- document limitation;
- keep `struct_node_to_class_index` so future node-level identity can be added.

### Risk: Hard topology negative transfer

Cause: `TopologyStateEncoder` still performs message passing over
`structural_edge_index`. If the model is trained on v1 and evaluated on v2,
even correct v2 topology at inference can create aggregation mismatch because
GNN weights were optimized for v1 neighborhoods.

Control:

- define this mode as an ablation for early-fusion and structural-overfitting
  analysis;
- log the topology/snapshot version already carried by runtime diagnostics;
- compare seen-version and unseen-version degradation separately;
- do not position this mode as the final OOD generalization method.

### Risk: Lossy prefix compression

Cause: the first `TopologyStateEncoder` implementation does not use the
observed instance-graph encoder and compresses the prefix into six handcrafted
state features.

Control:

- compare it against Base-UN as a lower-capacity early-fusion ablation, not as a
  capacity-equivalent replacement;
- report failures as evidence that observed IG dynamics are required;
- keep future `ObservedResidualTopologyState` or `Layerwise Structural
  Cross-Attention` out of this MVP task.

### Risk: Topology-state overconfidence

Cause: hard topology and prefix-state gates can make class logits too certain
on seen topology and wrong on unseen topology.

Control:

- log prediction entropy from topology-state logits;
- use conservative gate initialization;
- clamp `topology_state_beta`;
- use `logmeanexp` pooling by default.

### Risk: Silent fallback invalidates experiment

Cause: missing topology-state tensor.

Control:

- `TopologyStateEncoder` raises `ValueError` when structural mode is true and
  required tensors are missing.

### Risk: Confounded ablation

Cause: adding observed logits residual to topology-state logits.

Control:

- first implementation returns topology-state logits only;
- observed-only baseline remains available with `experiment.structural_mode:
  false`.

### Risk: Cache reuses old graph samples

Cause: graph dataset cache schema did not include `struct_prefix_state_x`.

Control:

- bump cache schema to 4;
- document that old graph cache must be rebuilt.

---

## 12. Recommended First Experiments After Implementation

1. `Base-UN-loan`, fraction `0.01`, seed fixed.
2. `S_ClassAware-UNc-loan`, fraction `0.01`, same seed.
3. `S_TopologyState-UNc-loan`, fraction `0.01`, same seed.
4. Repeat for `Train: v1 -> Test: v1-v4`.
5. Repeat for `Train: v1,v2 -> Test: v1-v4`.
6. Only after smoke validation, run 5 seeds.

Primary metrics:

- strict macro-F1;
- non-strict macro-F1;
- OOS / `pred_in_mask_rate`;
- confidence and calibration;
- prediction entropy;
- topology-state gate mean/max;
- duplicate class cardinality mean/max;
- degradation across versions.

Interpretation rule:

```text
TopologyStateEncoder validates the early-fusion mechanism if implementation
diagnostics are clean and its degradation pattern differs from late-fusion
modes. It validates the drift-generalization hypothesis only if it improves
unseen-version degradation or process-consistency without a large seen-version
penalty across repeated seeds.
```

Expected possible outcomes:

1. **Seen-version improvement, unseen-version degradation:** evidence of hard
   topology structural overfitting / negative transfer.
2. **Seen-version and unseen-version degradation:** prefix-state compression is
   too lossy or topology-state capacity is insufficient.
3. **Process-consistency improvement without strict-F1 gain:** topology helps
   constrain invalid behavior but does not solve exact next-activity ranking.
4. **Repeated-seed unseen-version improvement:** early topology-state fusion is
   a viable candidate for further refinement.

---

## 13. Documentation Drift Checklist For Implementation

The implementation PR/report must include:

- `Documentation drift check: updated docs/GNN_RUNTIME_MVP2_5.MD, docs/DATA_MODEL_MVP2_5.MD, docs/current/project-state.md, docs/dissertation_changes.MD, configs/ui/config_catalog.yaml`
- mention whether an ADR was created or explicitly deferred;
- mention cache schema bump and need to rebuild graph cache;
- mention whether `TopologyStateEncoder` is experimental.
- mention that raw `LogSumExp` pooling is not the default because it can
  amplify duplicate activity cardinality.
- mention that this mode is a controlled structural-overfitting ablation, not
  the final Cross-Attention/Side-Tuning solution for OOD drift.
