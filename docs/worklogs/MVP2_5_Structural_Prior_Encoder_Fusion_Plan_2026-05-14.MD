# MVP2.5 StructuralPriorEncoder Fusion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` or `executing-plans` to implement this plan task-by-task. Use TDD for every implementation task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `model.fusion_mode=StructuralPriorEncoder` as an etalon-like structural baseline where BPMN/EOPKG structure contributes a representation context before the final classifier, with `model.structural_prior_fusion=concat|gated_concat`.

**Architecture:** Keep the existing observed prefix encoder as the primary prediction path. Run the structural graph through the existing structural GNN, pool it into a structural context vector, and fuse `[obs_context || struct_context]` before the classifier. This intentionally avoids topology-only class logits and class-aware structural scoring; structure acts as a representation prior, similar to the old etalon flow.

**Tech Stack:** Python, PyTorch, PyTorch Geometric, YAML config catalog, pytest, local venv `.\.venv\Scripts\python.exe`.

---

## 1. Context And Rationale

### 1.1 Current modes

Current `EOPKGGATv2` supports:

- `ClassMeanAttention`: old attention path; compresses structural nodes into one attention-derived context vector.
- `ClassMeanConcat`: old concat path; mean-pools structural nodes and concatenates with observed context.
- `ClassAwareStructuralScoring`: adds per-class structural logits to observed logits.
- `TopologyStateEncoder`: experimental topology-only class-logit path using `struct_prefix_state_x`.

The unstable behavior observed in `TopologyStateEncoder` is expected because the mode bypasses the observed classifier path:

```python
if self.fusion_mode == "topology_state_encoder":
    return self._forward_topology_state(...)
```

Inside `_forward_topology_state`, `last_observed_logits` is explicitly cleared. This is not etalon-like.

### 1.2 Etalon-like target

The old etalon architecture behaved conceptually as:

```text
x + edge_index
  -> GNN message passing
  -> global_mean_pool
  -> concat(graph_context, doc_or_context_features)
  -> LayerNorm / Linear / ReLU / Dropout
  -> task_head
```

The MVP2.5 equivalent should be:

```text
observed prefix graph
  -> observed GNN
  -> obs_context

structural BPMN/EOPKG graph
  -> structural GNN
  -> struct_context

obs_context + struct_context
  -> fusion head
  -> classifier
```

This gives a clean structural baseline: structure participates in representation learning, not as a late mask and not as a separate class-logit generator.

---

## 2. Target Configuration Contract

### 2.1 New model config keys

```yaml
model:
  fusion_mode: StructuralPriorEncoder
  structural_prior_pooling: mean
  structural_prior_fusion: concat        # concat | gated_concat
  structural_prior_gate_init_bias: -1.0  # used only for gated_concat
```

### 2.2 Semantics

`model.fusion_mode: StructuralPriorEncoder`

- Enables etalon-like representation-level structural fusion.
- Requires `experiment.structural_mode=true`.
- Uses `structural_edge_index` and structural node features.
- Does not produce separate structural class logits.
- Does not use structural auxiliary loss.
- Does not use mask-guided logits unless training config enables mask policy separately.

`model.structural_prior_pooling: mean`

- Pools structural node embeddings into one context vector:

```text
struct_context = mean_v(h_struct_v)
```

- This matches etalon's `global_mean_pool` behavior.
- Only `mean` is in scope for the first implementation. Other pooling strategies are explicitly rejected until needed.

`model.structural_prior_fusion: concat`

- Exact baseline:

```text
fused = fusion_head([obs_context || struct_context])
logits = classifier(fused)
```

`model.structural_prior_fusion: gated_concat`

- Stabilized baseline:

```text
gate = sigmoid(W_gate([obs_context || struct_context]) + bias)
fused = fusion_head([obs_context || gate * struct_context])
logits = classifier(fused)
```

- Gate is diagnostic and regularizing. It should not be used to claim a pure etalon replication; it is a controlled variant.

---

## 3. Files To Modify

### Runtime

- `src/domain/models/eopkg_models.py`
  - Add config aliases and validation.
  - Add structural-prior fusion parameters.
  - Add helper methods for structural-prior pooling and fusion.
  - Add diagnostics fields.

- `src/cli.py`
  - Pass new model config keys from YAML into model factory.

### UI / Config

- `configs/ui/config_catalog.yaml`
  - Add `StructuralPriorEncoder` to `model.fusion_mode` enum.
  - Add `model.structural_prior_pooling`.
  - Add `model.structural_prior_fusion`.
  - Add `model.structural_prior_gate_init_bias`.

### Tests

- `tests/domain/test_eopkg_models_forward.py`
  - Add forward coverage for `StructuralPriorEncoder`.
  - Add concat and gated concat behavior tests.

- `tests/domain/test_eopkg_gatv2_dual_encoder.py`
  - Add alias normalization tests.
  - Add unsupported structural-prior config tests if this file remains the main EOPKG config test location.

- `tests/application/test_cli_flatten_config.py`
  - Add factory kwargs coverage for new config keys.

- Optional focused test file:
  - `tests/domain/test_eopkg_structural_prior_encoder.py`
  - Use this if adding the tests to existing large files becomes noisy.

### Documentation

- `docs/GNN_RUNTIME_MVP2_5.MD`
  - Add `StructuralPriorEncoder` section.
  - Clarify difference from `ClassMeanConcat`, `ClassAwareStructuralScoring`, and `TopologyStateEncoder`.

- `docs/current/project-state.md`
  - Update runtime feature summary after implementation.

- `docs/dissertation_changes.MD`
  - Add note that `StructuralPriorEncoder` is the etalon-like structural baseline for demonstrating structure as representation prior.

- Implementation report:
  - `docs/worklogs/MVP2_5_Structural_Prior_Encoder_Fusion_Report_2026-05-14.MD`

---

## 4. Task 1: Add Failing Tests For Fusion Mode Alias And Config Validation

**Files:**

- Modify: `tests/domain/test_eopkg_gatv2_dual_encoder.py`
- Modify: `tests/domain/test_eopkg_models_forward.py`

- [ ] **Step 1: Extend alias normalization test**

Add cases:

```python
("StructuralPriorEncoder", "structural_prior_encoder"),
("structural_prior_encoder", "structural_prior_encoder"),
("StructuralPrior", "structural_prior_encoder"),
("BPMNStructuralPrior", "structural_prior_encoder"),
```

Expected behavior:

```python
assert model.fusion_mode == expected
```

- [ ] **Step 2: Add unsupported structural-prior fusion test**

Test:

```python
def test_eopkg_gatv2_rejects_unknown_structural_prior_fusion():
    with pytest.raises(ValueError, match="Unsupported model.structural_prior_fusion"):
        create_model(
            model_type="EOPKGGATv2",
            feature_layout=_layout(),
            hidden_dim=16,
            output_dim=7,
            struct_hidden_dim=16,
            dropout=0.0,
            pooling_strategy="global_mean",
            fusion_mode="StructuralPriorEncoder",
            structural_prior_fusion="bad_mode",
        )
```

- [ ] **Step 3: Add unsupported structural-prior pooling test**

Test:

```python
def test_eopkg_gatv2_rejects_unknown_structural_prior_pooling():
    with pytest.raises(ValueError, match="Unsupported model.structural_prior_pooling"):
        create_model(
            model_type="EOPKGGATv2",
            feature_layout=_layout(),
            hidden_dim=16,
            output_dim=7,
            struct_hidden_dim=16,
            dropout=0.0,
            pooling_strategy="global_mean",
            fusion_mode="StructuralPriorEncoder",
            structural_prior_pooling="attention",
        )
```

- [ ] **Step 4: Run tests and verify failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/domain/test_eopkg_gatv2_dual_encoder.py::test_eopkg_gatv2_normalizes_fusion_mode_aliases tests/domain/test_eopkg_gatv2_dual_encoder.py::test_eopkg_gatv2_rejects_unknown_structural_prior_fusion tests/domain/test_eopkg_gatv2_dual_encoder.py::test_eopkg_gatv2_rejects_unknown_structural_prior_pooling -v
```

Expected before implementation:

- Alias test fails because `StructuralPriorEncoder` is unsupported.
- Validation tests fail because constructor does not accept or validate the new keys.

---

## 5. Task 2: Add Constructor Config And Alias Normalization

**Files:**

- Modify: `src/domain/models/eopkg_models.py`

- [ ] **Step 1: Add constructor parameters**

Add after topology-state config parameters:

```python
structural_prior_pooling: str = "mean",
structural_prior_fusion: str = "concat",
structural_prior_gate_init_bias: float = -1.0,
```

- [ ] **Step 2: Extend fusion alias map**

Add:

```python
"structuralpriorencoder": "structural_prior_encoder",
"structural_prior_encoder": "structural_prior_encoder",
"structuralprior": "structural_prior_encoder",
"structural_prior": "structural_prior_encoder",
"bpmnstructuralprior": "structural_prior_encoder",
"bpmn_structural_prior": "structural_prior_encoder",
```

Add `"structural_prior_encoder"` to `allowed_fusion_modes`.

Update error text to include:

```text
'StructuralPriorEncoder'
```

- [ ] **Step 3: Validate structural-prior config**

Add:

```python
self.structural_prior_pooling = str(structural_prior_pooling or "mean").strip().lower()
if self.structural_prior_pooling not in {"mean"}:
    raise ValueError(
        "Unsupported model.structural_prior_pooling "
        f"'{self.structural_prior_pooling}'. Available: ['mean']"
    )

self.structural_prior_fusion = str(structural_prior_fusion or "concat").strip().lower()
if self.structural_prior_fusion not in {"concat", "gated_concat"}:
    raise ValueError(
        "Unsupported model.structural_prior_fusion "
        f"'{self.structural_prior_fusion}'. Available: ['concat', 'gated_concat']"
    )

self.structural_prior_gate_init_bias = float(structural_prior_gate_init_bias)
```

- [ ] **Step 4: Add gate module and diagnostics fields**

Add after existing fusion modules:

```python
self.structural_prior_gate = nn.Linear(hidden_dim + self.struct_hidden_dim, self.struct_hidden_dim)
nn.init.constant_(self.structural_prior_gate.bias, self.structural_prior_gate_init_bias)
```

Add diagnostics:

```python
self.last_structural_prior_context: torch.Tensor | None = None
self.last_structural_prior_gate: torch.Tensor | None = None
self.last_observed_context: torch.Tensor | None = None
```

- [ ] **Step 5: Update `_clear_structural_diagnostics`**

Add:

```python
self.last_structural_prior_context = None
self.last_structural_prior_gate = None
self.last_observed_context = None
```

- [ ] **Step 6: Run alias/config tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/domain/test_eopkg_gatv2_dual_encoder.py -k "fusion_mode_aliases or structural_prior" -v
```

Expected:

- New alias and validation tests pass.
- Existing alias tests still pass.

---

## 6. Task 3: Implement `concat` StructuralPriorEncoder Forward Path

**Files:**

- Modify: `src/domain/models/eopkg_models.py`
- Modify: `tests/domain/test_eopkg_models_forward.py`

- [ ] **Step 1: Write failing forward test for `concat`**

Add:

```python
def test_eopkggatv2_structural_prior_encoder_concat_uses_observed_and_structural_context():
    torch.manual_seed(7)
    model = create_model(
        model_type="EOPKGGATv2",
        feature_layout=_feature_layout(),
        hidden_dim=16,
        output_dim=7,
        dropout=0.0,
        pooling_strategy="global_mean",
        fusion_mode="StructuralPriorEncoder",
        structural_prior_fusion="concat",
        structural_prior_pooling="mean",
    )
    model.eval()

    contract = _contract(with_struct=True)
    with torch.no_grad():
        logits = model(contract)

    assert tuple(logits.shape) == (1, 7)
    assert model.last_observed_context is not None
    assert model.last_structural_prior_context is not None
    assert model.last_structural_prior_gate is None
    assert model.last_observed_logits is None
    assert model.last_structural_class_logits is None
```

- [ ] **Step 2: Run test and verify failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/domain/test_eopkg_models_forward.py::test_eopkggatv2_structural_prior_encoder_concat_uses_observed_and_structural_context -v
```

Expected before implementation:

- Fails because `StructuralPriorEncoder` forward branch is not implemented.

- [ ] **Step 3: Add pooling helper**

Add method:

```python
def _pool_structural_prior_context(
    self,
    h_struct: torch.Tensor,
    *,
    batch_size: int,
) -> torch.Tensor:
    if self.structural_prior_pooling == "mean":
        return h_struct.mean(dim=0, keepdim=True).expand(batch_size, -1)
    raise ValueError(
        "Unsupported model.structural_prior_pooling "
        f"'{self.structural_prior_pooling}'. Available: ['mean']"
    )
```

- [ ] **Step 4: Add fusion helper**

Add method:

```python
def _forward_structural_prior(
    self,
    *,
    obs_context: torch.Tensor,
    h_struct: torch.Tensor,
) -> torch.Tensor:
    context_struct = self._pool_structural_prior_context(
        h_struct,
        batch_size=int(obs_context.size(0)),
    )
    self.last_observed_context = obs_context.detach()
    self.last_structural_prior_context = context_struct.detach()
    self.last_structural_prior_gate = None

    if self.structural_prior_fusion == "concat":
        fused_input = torch.cat([obs_context, context_struct], dim=1)
    elif self.structural_prior_fusion == "gated_concat":
        gate_input = torch.cat([obs_context, context_struct], dim=1)
        gate = torch.sigmoid(self.structural_prior_gate(gate_input))
        self.last_structural_prior_gate = gate.detach()
        fused_input = torch.cat([obs_context, gate * context_struct], dim=1)
    else:
        raise ValueError(
            "Unsupported model.structural_prior_fusion "
            f"'{self.structural_prior_fusion}'. Available: ['concat', 'gated_concat']"
        )

    fused = self.fusion(fused_input)
    return self.classifier(fused)
```

- [ ] **Step 5: Route forward branch**

After structural GNN produces `h_norm`, before class-aware branch:

```python
if self.fusion_mode == "structural_prior_encoder":
    self.last_cross_attn_weights = None
    self.last_observed_logits = None
    self.last_structural_node_logits = None
    self.last_structural_raw_class_logits = None
    self.last_structural_normalized_class_logits = None
    self.last_structural_observed_scale = None
    self.last_structural_class_logits = None
    return self._forward_structural_prior(obs_context=obs_context, h_struct=h_norm)
```

- [ ] **Step 6: Run concat forward test**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/domain/test_eopkg_models_forward.py::test_eopkggatv2_structural_prior_encoder_concat_uses_observed_and_structural_context -v
```

Expected:

- Pass.

---

## 7. Task 4: Implement And Test `gated_concat`

**Files:**

- Modify: `tests/domain/test_eopkg_models_forward.py`
- Modify: `src/domain/models/eopkg_models.py` if Task 3 did not already include gate branch.

- [ ] **Step 1: Write failing gated test**

Add:

```python
def test_eopkggatv2_structural_prior_encoder_gated_concat_logs_gate():
    torch.manual_seed(11)
    model = create_model(
        model_type="EOPKGGATv2",
        feature_layout=_feature_layout(),
        hidden_dim=16,
        output_dim=7,
        dropout=0.0,
        pooling_strategy="global_mean",
        fusion_mode="StructuralPriorEncoder",
        structural_prior_fusion="gated_concat",
        structural_prior_pooling="mean",
        structural_prior_gate_init_bias=-1.0,
    )
    model.eval()

    with torch.no_grad():
        logits = model(_contract(with_struct=True))

    assert tuple(logits.shape) == (1, 7)
    assert model.last_structural_prior_gate is not None
    assert model.last_structural_prior_gate.shape == torch.Size([1, 16])
    assert torch.all(model.last_structural_prior_gate >= 0.0)
    assert torch.all(model.last_structural_prior_gate <= 1.0)
```

- [ ] **Step 2: Run test and verify failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/domain/test_eopkg_models_forward.py::test_eopkggatv2_structural_prior_encoder_gated_concat_logs_gate -v
```

Expected before implementation:

- Fails if gate branch is missing or diagnostics are missing.

- [ ] **Step 3: Implement minimal gate path**

If not already implemented in Task 3, add:

```python
elif self.structural_prior_fusion == "gated_concat":
    gate_input = torch.cat([obs_context, context_struct], dim=1)
    gate = torch.sigmoid(self.structural_prior_gate(gate_input))
    self.last_structural_prior_gate = gate.detach()
    fused_input = torch.cat([obs_context, gate * context_struct], dim=1)
```

- [ ] **Step 4: Run gated test**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/domain/test_eopkg_models_forward.py::test_eopkggatv2_structural_prior_encoder_gated_concat_logs_gate -v
```

Expected:

- Pass.

---

## 8. Task 5: Pass Config From CLI To Model Factory

**Files:**

- Modify: `src/cli.py`
- Modify: `tests/application/test_cli_flatten_config.py`

- [ ] **Step 1: Write failing factory kwargs test**

In `tests/application/test_cli_flatten_config.py`, extend the existing model kwargs test or add:

```python
def test_build_model_factory_kwargs_passes_structural_prior_config():
    from src.cli import _build_model_factory_kwargs

    kwargs = _build_model_factory_kwargs(
        model_cfg={
            "type": "EOPKGGATv2",
            "hidden_dim": 16,
            "fusion_mode": "StructuralPriorEncoder",
            "structural_prior_pooling": "mean",
            "structural_prior_fusion": "gated_concat",
            "structural_prior_gate_init_bias": -1.5,
        },
        feature_layout=object(),
        output_dim=7,
    )

    assert kwargs["fusion_mode"] == "StructuralPriorEncoder"
    assert kwargs["structural_prior_pooling"] == "mean"
    assert kwargs["structural_prior_fusion"] == "gated_concat"
    assert kwargs["structural_prior_gate_init_bias"] == -1.5
```

- [ ] **Step 2: Run test and verify failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/application/test_cli_flatten_config.py::test_build_model_factory_kwargs_passes_structural_prior_config -v
```

Expected before implementation:

- Fails because new keys are not returned.

- [ ] **Step 3: Update `_build_model_factory_kwargs`**

Add:

```python
"structural_prior_pooling": str(model_cfg.get("structural_prior_pooling", "mean")),
"structural_prior_fusion": str(model_cfg.get("structural_prior_fusion", "concat")),
"structural_prior_gate_init_bias": float(model_cfg.get("structural_prior_gate_init_bias", -1.0)),
```

- [ ] **Step 4: Run CLI test**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/application/test_cli_flatten_config.py::test_build_model_factory_kwargs_passes_structural_prior_config -v
```

Expected:

- Pass.

---

## 9. Task 6: Add Trainer Diagnostics For StructuralPriorEncoder

**Files:**

- Modify: `src/application/use_cases/trainer.py`
- Test: `tests/application/test_trainer_forward_stats_logging.py`

- [ ] **Step 1: Add diagnostics test**

Add a focused fake model test if current trainer tests already use a fake model. Required metric names:

```text
training_structural_prior_context_mean_abs
training_observed_context_mean_abs
training_structural_prior_to_observed_context_ratio
training_structural_prior_gate_mean
training_structural_prior_gate_max
```

For `concat`, gate metrics may be absent or zero. For `gated_concat`, gate metrics must be logged.

Expected fake model fields:

```python
model.last_observed_context = torch.tensor([[2.0, -2.0]])
model.last_structural_prior_context = torch.tensor([[0.5, -0.5]])
model.last_structural_prior_gate = torch.tensor([[0.25, 0.75]])
```

- [ ] **Step 2: Run test and verify failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/application/test_trainer_forward_stats_logging.py -k "structural_prior" -v
```

Expected before implementation:

- Fails because trainer does not accumulate structural-prior diagnostics.

- [ ] **Step 3: Extend forward stats buckets**

Add bucket counters near existing structural/topology diagnostics:

```python
"observed_context_abs_sum": 0.0,
"observed_context_count": 0,
"structural_prior_context_abs_sum": 0.0,
"structural_prior_context_count": 0,
"structural_prior_gate_sum": 0.0,
"structural_prior_gate_count": 0,
"structural_prior_gate_max": 0.0,
```

- [ ] **Step 4: Add accumulator**

Add method:

```python
def _accumulate_structural_prior_stats(self, bucket: Dict[str, Any]) -> None:
    observed_context = getattr(self.model, "last_observed_context", None)
    struct_context = getattr(self.model, "last_structural_prior_context", None)
    gate = getattr(self.model, "last_structural_prior_gate", None)
    if isinstance(observed_context, torch.Tensor) and isinstance(struct_context, torch.Tensor):
        observed_abs = torch.nan_to_num(observed_context.detach().float(), nan=0.0, posinf=1e6, neginf=-1e6).abs()
        struct_abs = torch.nan_to_num(struct_context.detach().float(), nan=0.0, posinf=1e6, neginf=-1e6).abs()
        bucket["observed_context_abs_sum"] += float(observed_abs.sum().item())
        bucket["observed_context_count"] += int(observed_abs.numel())
        bucket["structural_prior_context_abs_sum"] += float(struct_abs.sum().item())
        bucket["structural_prior_context_count"] += int(struct_abs.numel())
    if isinstance(gate, torch.Tensor):
        gate_safe = torch.nan_to_num(gate.detach().float(), nan=0.0, posinf=1.0, neginf=0.0)
        bucket["structural_prior_gate_sum"] += float(gate_safe.sum().item())
        bucket["structural_prior_gate_count"] += int(gate_safe.numel())
        bucket["structural_prior_gate_max"] = max(
            float(bucket.get("structural_prior_gate_max", 0.0)),
            float(gate_safe.max().item()),
        )
```

- [ ] **Step 5: Call accumulator in train/validation/eval paths**

Where `_accumulate_forward_stats` currently calls structural/topology accumulators, add:

```python
self._accumulate_structural_prior_stats(forward_stats)
```

- [ ] **Step 6: Log metrics**

Compute:

```python
observed_context_mean_abs = observed_context_abs_sum / max(observed_context_count, 1)
structural_prior_context_mean_abs = structural_prior_context_abs_sum / max(structural_prior_context_count, 1)
structural_prior_to_observed_context_ratio = structural_prior_context_mean_abs / max(observed_context_mean_abs, 1e-12)
structural_prior_gate_mean = structural_prior_gate_sum / max(structural_prior_gate_count, 1)
```

Log via tracker with prefix:

```python
self.tracker.log_metric(f"{metric_prefix}_observed_context_mean_abs", observed_context_mean_abs)
self.tracker.log_metric(f"{metric_prefix}_structural_prior_context_mean_abs", structural_prior_context_mean_abs)
self.tracker.log_metric(f"{metric_prefix}_structural_prior_to_observed_context_ratio", structural_prior_to_observed_context_ratio)
self.tracker.log_metric(f"{metric_prefix}_structural_prior_gate_mean", structural_prior_gate_mean)
self.tracker.log_metric(f"{metric_prefix}_structural_prior_gate_max", structural_prior_gate_max)
```

- [ ] **Step 7: Run diagnostics tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/application/test_trainer_forward_stats_logging.py -v
```

Expected:

- Existing forward-stat logging tests pass.
- New structural-prior diagnostics test passes.

---

## 10. Task 7: Update UI Config Catalog

**Files:**

- Modify: `configs/ui/config_catalog.yaml`

- [ ] **Step 1: Add fusion enum**

Add to `model.fusion_mode.enum`:

```yaml
- StructuralPriorEncoder
```

Update description:

```text
StructuralPriorEncoder uses etalon-like representation-level fusion: observed context plus pooled structural context before classifier.
```

- [ ] **Step 2: Add `model.structural_prior_pooling`**

Add near fusion-related model keys:

```yaml
  model.structural_prior_pooling:
    path: model.structural_prior_pooling
    section: model
    label: 'Model: Structural prior pooling'
    description: Pooling strategy for structural node embeddings in StructuralPriorEncoder; mean mirrors etalon global_mean_pool.
    affects: How BPMN/EOPKG node states become one structural context vector before classifier fusion.
    default: mean
    enum:
    - mean
    required_in_modes: []
    required_when:
      model.type: EOPKGGATv2
      model.fusion_mode: StructuralPriorEncoder
    ui:
      tab: model
      group: advanced
      priority: 100
      order: 1012
```

- [ ] **Step 3: Add `model.structural_prior_fusion`**

```yaml
  model.structural_prior_fusion:
    path: model.structural_prior_fusion
    section: model
    label: 'Model: Structural prior fusion'
    description: Fusion operator for observed context and pooled structural context in StructuralPriorEncoder.
    affects: concat is the direct etalon-like baseline; gated_concat learns how strongly to pass structural context.
    default: concat
    enum:
    - concat
    - gated_concat
    required_in_modes: []
    required_when:
      model.type: EOPKGGATv2
      model.fusion_mode: StructuralPriorEncoder
    ui:
      tab: model
      group: advanced
      priority: 100
      order: 1013
```

- [ ] **Step 4: Add `model.structural_prior_gate_init_bias`**

```yaml
  model.structural_prior_gate_init_bias:
    path: model.structural_prior_gate_init_bias
    section: model
    label: 'Model: Structural prior gate init bias'
    description: Initial sigmoid gate bias for gated_concat; negative values start with conservative structural influence.
    affects: Initial strength of structural context in gated StructuralPriorEncoder.
    default: -1.0
    enum: []
    required_in_modes: []
    required_when:
      model.type: EOPKGGATv2
      model.fusion_mode: StructuralPriorEncoder
      model.structural_prior_fusion: gated_concat
    ui:
      tab: model
      group: advanced
      priority: 100
      order: 1014
```

- [ ] **Step 5: Run config/UI-related tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/application/test_cli_flatten_config.py -v
```

Expected:

- Pass.

---

## 11. Task 8: Add Optional Experiment Preset Names

**Files:**

- Modify only if presets are centrally defined in `configs/ui/config_catalog.yaml` or another catalog file.

- [ ] **Step 1: Add recommended preset labels**

Use names:

```text
S_Str-UNcPrior-loan
S_Str-UNcPriorGated-loan
S_Str-UScPrior-loan
S_Str-UScPriorGated-loan
```

Recommended configs:

```yaml
model:
  fusion_mode: StructuralPriorEncoder
  structural_prior_pooling: mean
  structural_prior_fusion: concat
```

and:

```yaml
model:
  fusion_mode: StructuralPriorEncoder
  structural_prior_pooling: mean
  structural_prior_fusion: gated_concat
  structural_prior_gate_init_bias: -1.0
```

- [ ] **Step 2: Keep drift preset separate**

Do not create drift presets until train checkpoints exist. Drift should reuse the trained checkpoint exactly:

```yaml
pipeline:
  mode: eval_drift
training:
  load_checkpoint: checkpoints/<matching-prior-run>_best.pth
```

---

## 12. Task 9: Documentation Updates

**Files:**

- Modify: `docs/GNN_RUNTIME_MVP2_5.MD`
- Modify: `docs/current/project-state.md`
- Modify: `docs/dissertation_changes.MD`
- Create: `docs/worklogs/MVP2_5_Structural_Prior_Encoder_Fusion_Report_2026-05-14.MD`

- [ ] **Step 1: Update GNN runtime docs**

Add section:

```markdown
### StructuralPriorEncoder Fusion

`StructuralPriorEncoder` is an etalon-like representation-level structural fusion mode.
It keeps the observed prefix encoder as the primary path, computes a pooled structural
context from EOPKG/BPMN tensors, and fuses both contexts before the classifier.

It differs from:

- `ClassMeanConcat`: legacy mean structural context without explicit etalon-baseline naming or structural-prior diagnostics;
- `ClassAwareStructuralScoring`: late structural class-logit additive scoring;
- `TopologyStateEncoder`: topology-state branch that directly emits class logits.

Formula:

```text
obs_context_i = ObservedGNN(prefix_i)
h_struct_v = StructuralGNN(h0_v, structural_edge_index)
struct_context = mean_v(h_struct_v)
logits_i = Classifier(Fusion([obs_context_i || struct_context]))
```

For `model.structural_prior_fusion=gated_concat`:

```text
gate_i = sigmoid(W_gate([obs_context_i || struct_context]) + b)
logits_i = Classifier(Fusion([obs_context_i || gate_i * struct_context]))
```
```

- [ ] **Step 2: Update project state**

Add short runtime note:

```markdown
- `EOPKGGATv2` supports `StructuralPriorEncoder` as an etalon-like structural baseline with `concat` and `gated_concat` fusion.
```

- [ ] **Step 3: Update dissertation changes**

Add:

```markdown
## StructuralPriorEncoder baseline

- Додано etalon-like baseline, де структура процесу використовується як representation-level prior перед classifier, а не як mask або окремі structural logits.
- Це важливо для дисертації: дозволяє чесно відокремити питання "чи структура корисна для прогнозування" від питання "чи переноситься ця користь між версіями процесу".
- `concat` відповідає найближчій аналогії з etalon; `gated_concat` є стабілізованою абляцією.
```

- [ ] **Step 4: Write implementation report**

Report must include:

- implemented files;
- tests run;
- recommended first experiments;
- interpretation guide for metrics.

---

## 13. Task 10: Full Verification

**Files:** no new code changes unless tests fail.

- [ ] **Step 1: Run focused model tests**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/domain/test_eopkg_models_forward.py tests/domain/test_eopkg_gatv2_dual_encoder.py tests/domain/test_eopkg_topology_state_encoder.py -v
```

Expected:

- Pass.
- Existing `TopologyStateEncoder` tests still pass.
- Existing `ClassAwareStructuralScoring` tests still pass.

- [ ] **Step 2: Run focused application tests**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/application/test_cli_flatten_config.py tests/application/test_trainer_forward_stats_logging.py -v
```

Expected:

- Pass.

- [ ] **Step 3: Run architecture guard**

```powershell
.\.venv\Scripts\python.exe tools\architecture_guard.py
```

Expected:

- Pass.

- [ ] **Step 4: Run full regression if focused tests pass**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -v
```

Expected:

- Pass or only known unrelated failures explicitly documented.

---

## 14. Recommended First Experiments

### 14.1 Purpose

Use these experiments to test whether structure helps in same-version prediction before using it as a drift-transfer mechanism.

### 14.2 Minimal train matrix

Use the same data split, fraction, seed, and gateway mode as baseline.

Run:

```text
Base-UN-loan
S_Str-UNcPrior-loan
S_Str-UNcPriorGated-loan
```

If statistics are enabled:

```text
S_Str-UScPrior-loan
S_Str-UScPriorGated-loan
```

### 14.3 Metrics to compare

Primary:

- `validation_strict_macro_f1`
- `test_strict_macro_f1`
- `test_macro_f1`
- `test_oos_rate`
- `test_ece`

Structural-prior diagnostics:

- `*_observed_context_mean_abs`
- `*_structural_prior_context_mean_abs`
- `*_structural_prior_to_observed_context_ratio`
- `*_structural_prior_gate_mean`
- `*_structural_prior_gate_max`

Interpretation:

- If `concat` improves F1/OOS versus Base-UN, structure is useful as representation prior.
- If `gated_concat` improves stability with similar or better F1, gate should be used for drift experiments.
- If both fail while ClassAware succeeds, structure helps mostly as candidate ranking/logit prior, not as global representation context.
- If structural context ratio is near zero, the branch is ignored.
- If gate is near one from early epochs and validation is unstable, structural context is probably over-dominating.

### 14.4 Drift transfer is not the first success criterion

Do not expect immediate v1 -> v2/v3/v4 transfer. This feature is intended to establish a clean structural baseline. Drift transfer remains a separate methodology problem.

---

## 15. Risks And Guardrails

### Risk: This duplicates `ClassMeanConcat`

`StructuralPriorEncoder + concat` is intentionally close to `ClassMeanConcat`, but it is not just an alias. It adds explicit etalon-like semantics, config visibility, diagnostics, and a paired `gated_concat` variant.

### Risk: Static structural context is too weak

If `struct_x` is absent and the structure is the same for all prefixes, structural context may behave like a process/version bias. This is acceptable for the first baseline. If it is too weak, the next controlled extension is prefix-state-conditioned structural context, not class logits.

### Risk: Gated concat hides structural failure

Gate metrics must be logged. If the gate collapses near zero, the model is rejecting structure. That is a valid experimental result.

### Risk: Structural payload batch selection debt

The current project has known debt around mixed structural payloads in one batch. This feature should not attempt to solve that debt. Experiments should use existing warnings and drift-window one-pass evaluation safeguards.

---

## 16. Self-Review

- Spec coverage: covers `StructuralPriorEncoder`, `concat`, `gated_concat`, config, UI, tests, docs, and experiment interpretation.
- Placeholder scan: no `TBD`/`TODO` placeholders are used as implementation instructions.
- Type consistency: new config keys consistently use:
  - `structural_prior_pooling`
  - `structural_prior_fusion`
  - `structural_prior_gate_init_bias`
- Scope control: does not implement cross-attention, side-tuning, or version-transfer methodology.

