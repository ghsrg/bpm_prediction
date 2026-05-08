# Structural Logit Scale And Auxiliary Loss Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `ClassAwareStructuralScoring` produce structural logits on a useful scale and give the structural branch its own training signal without breaking parallel/allowed-mask semantics.

**Architecture:** Keep `fusion_mode=ClassAwareStructuralScoring`, but replace pure scaled-cosine scoring with a two-component structural scorer: prefix-aware bilinear scoring plus structural prior. Normalize structural class logits per sample, rescale them against detached observed-logit magnitude, then train with the normal fused CE loss plus a small set-aware structural auxiliary loss.

**Tech Stack:** Python, PyTorch, PyTorch Geometric, `EOPKGGATv2`, `ModelTrainer`, MLflow metrics, YAML config catalog, pytest.

---

## Metadata

- `status`: plan
- `date`: 2026-05-07
- `scope`: MVP2.5 GNN runtime, training objective, Experiment UI config catalog
- `primary_files`:
  - `src/domain/models/eopkg_models.py`
  - `src/application/use_cases/trainer.py`
  - `configs/ui/config_catalog.yaml`
- `related_files`:
  - `tests/domain/test_eopkg_gatv2_dual_encoder.py`
  - `tests/application/test_trainer_forward_stats_logging.py`
  - `docs/GNN_RUNTIME_MVP2_5.MD`
  - `docs/current/project-state.md`
- `trigger`: observed scale mismatch

Observed diagnostics:

```text
validation_observed_logits_mean_abs: 8.2 - 36.22
validation_structural_logits_max_abs: 0.12 - 0.496
validation_structural_logits_mean_abs: 0.07 - 0.226
validation_structural_to_observed_logit_ratio: 0.006 - 0.012
```

Interpretation: structural logits are non-zero and growing, but their magnitude is too small to affect the fused CE loss materially.

---

## Target Semantics

Current final logit:

```text
final_logits = observed_logits + structural_logits
```

Target structural branch:

```text
q_i = W_q obs_context_i
k_v = W_k h_struct_v

bilinear_node_score(i, v) = q_i^T W_bilinear k_v
prior_node_score(v) = W_prior h_struct_v
raw_node_score(i, v) = bilinear_node_score(i, v) + prior_node_score(v)

raw_class_score(i, j) = LogSumExp({raw_node_score(i, v) | node_to_class[v] = j})
normalized_struct(i, :) = LayerNorm(raw_class_score(i, :))

obs_scale(i) = clamp(mean(abs(observed_logits_i)).detach(), min=1.0, max=10.0)
scaled_struct(i, :) = gamma * obs_scale(i) * normalized_struct(i, :)

final_logits = observed_logits + scaled_struct
```

Training loss:

```text
main_loss = CE(mask_policy(final_logits), y)

L_struct_set = -mean(logsumexp(log_softmax(scaled_struct)[allowed_mask_or_target]))
L_struct_exact = CE(scaled_struct, y)

loss = main_loss
     + structural_aux_loss_weight * L_struct_set
     + structural_aux_exact_loss_weight * L_struct_exact
```

Default weights:

```yaml
training.structural_aux_loss_enabled: true
training.structural_aux_loss_weight: 0.05
training.structural_aux_exact_loss_weight: 0.01
```

Rationale:

- `L_struct_set` gives structure a direct gradient while respecting allowed parallel alternatives.
- `L_struct_exact` is intentionally small; it nudges strict-F1 without forcing structure to overfit exact next labels in ambiguous regions.
- `obs_scale.detach()` aligns magnitude but prevents the structural branch from backpropagating through the observed-logit scale.

---

## File Structure

### `src/domain/models/eopkg_models.py`

Responsibility: model forward semantics and structural logit generation.

Add:

- `structural_score_mode`: default `bilinear_with_prior`.
- `structural_logit_scale_init`: default `0.1`.
- `structural_logit_scale_max`: default `2.0`.
- `structural_observed_scale_min`: default `1.0`.
- `structural_observed_scale_max`: default `10.0`.
- bilinear scorer and structural prior head.
- helper method for per-sample structural normalization and observed-scale alignment.
- diagnostic tensors:
  - `last_structural_raw_class_logits`
  - `last_structural_normalized_class_logits`
  - `last_structural_observed_scale`

### `src/application/use_cases/trainer.py`

Responsibility: training objective and runtime metrics.

Add:

- structural auxiliary loss config parsing.
- helper method for set-aware structural loss.
- helper method for exact structural loss.
- total loss composition inside `_run_epoch`.
- MLflow/log metrics:
  - `train_structural_aux_set_loss`
  - `train_structural_aux_exact_loss`
  - `train_structural_aux_total_loss`
  - same validation prefixes.

### `configs/ui/config_catalog.yaml`

Responsibility: Experiment UI discoverability.

Add config keys:

- `model.structural_score_mode`
- `model.structural_logit_scale_init`
- `model.structural_logit_scale_max`
- `model.structural_observed_scale_min`
- `model.structural_observed_scale_max`
- `training.structural_aux_loss_enabled`
- `training.structural_aux_loss_weight`
- `training.structural_aux_exact_loss_weight`

### Docs

Update:

- `docs/GNN_RUNTIME_MVP2_5.MD`
- `docs/current/project-state.md`

No data contract change is required because `struct_node_to_class_index` already exists.

---

## Task 1: Add Model-Level Scale Alignment Tests

**Files:**

- Modify: `tests/domain/test_eopkg_gatv2_dual_encoder.py`
- Modify: `src/domain/models/eopkg_models.py`

- [ ] **Step 1: Write failing test for observed-scale alignment**

Add this test to `tests/domain/test_eopkg_gatv2_dual_encoder.py`:

```python
def test_eopkg_gatv2_scales_structural_logits_against_observed_logits():
    model = EOPKGGATv2(
        feature_layout=_feature_layout(),
        hidden_dim=8,
        output_dim=3,
        dropout=0.0,
        structural_mode=True,
        fusion_mode="ClassAwareStructuralScoring",
        structural_logit_scale_init=0.1,
        structural_observed_scale_min=1.0,
        structural_observed_scale_max=10.0,
    )
    raw = torch.tensor([[-1.0, 0.0, 1.0]], dtype=torch.float32)
    observed = torch.tensor([[20.0, -10.0, 0.0]], dtype=torch.float32)

    scaled, normalized, observed_scale = model._scale_structural_class_logits(
        raw_class_logits=raw,
        observed_logits=observed,
    )

    assert observed_scale.shape == torch.Size([1, 1])
    assert float(observed_scale.item()) == pytest.approx(10.0, abs=1e-6)
    assert torch.allclose(normalized.mean(dim=1), torch.zeros(1), atol=1e-6)
    assert torch.allclose(normalized.std(dim=1, unbiased=False), torch.ones(1), atol=1e-6)
    assert float(scaled.abs().mean().item()) == pytest.approx(float(normalized.abs().mean().item()), rel=1e-5)
```

- [ ] **Step 2: Run test and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/domain/test_eopkg_gatv2_dual_encoder.py::test_eopkg_gatv2_scales_structural_logits_against_observed_logits -v
```

Expected:

```text
FAILED with AttributeError: 'EOPKGGATv2' object has no attribute '_scale_structural_class_logits'
```

- [ ] **Step 3: Implement model constructor parameters and scaling helper**

In `src/domain/models/eopkg_models.py`, extend `EOPKGGATv2.__init__` signature:

```python
structural_score_mode: str = "bilinear_with_prior",
structural_logit_scale_init: float = 0.1,
structural_logit_scale_max: float = 2.0,
structural_observed_scale_min: float = 1.0,
structural_observed_scale_max: float = 10.0,
```

Add attributes:

```python
self.structural_score_mode = str(structural_score_mode or "bilinear_with_prior").strip().lower()
if self.structural_score_mode not in {"bilinear_with_prior", "cosine"}:
    raise ValueError(
        "Unsupported model.structural_score_mode "
        f"'{self.structural_score_mode}'. Available: ['bilinear_with_prior', 'cosine']"
    )
self.structural_logit_scale_max = float(structural_logit_scale_max)
self.structural_observed_scale_min = float(structural_observed_scale_min)
self.structural_observed_scale_max = float(structural_observed_scale_max)
self.structural_logit_scale = nn.Parameter(torch.tensor(float(structural_logit_scale_init), dtype=torch.float32))
self.structural_class_norm = nn.LayerNorm(output_dim, elementwise_affine=False)
```

Add helper:

```python
def _scale_structural_class_logits(
    self,
    *,
    raw_class_logits: torch.Tensor,
    observed_logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    normalized = self.structural_class_norm(raw_class_logits)
    observed_scale = observed_logits.detach().abs().mean(dim=1, keepdim=True)
    observed_scale = torch.clamp(
        observed_scale,
        min=float(self.structural_observed_scale_min),
        max=float(self.structural_observed_scale_max),
    )
    gamma = torch.clamp(
        self.structural_logit_scale,
        min=0.0,
        max=float(self.structural_logit_scale_max),
    )
    scaled = normalized * observed_scale * gamma
    return scaled, normalized, observed_scale
```

- [ ] **Step 4: Run test and verify GREEN**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/domain/test_eopkg_gatv2_dual_encoder.py::test_eopkg_gatv2_scales_structural_logits_against_observed_logits -v
```

Expected:

```text
PASSED
```

---

## Task 2: Replace Pure Cosine With Bilinear Plus Prior Scoring

**Files:**

- Modify: `tests/domain/test_eopkg_gatv2_dual_encoder.py`
- Modify: `src/domain/models/eopkg_models.py`

- [ ] **Step 1: Write failing test for bilinear-plus-prior diagnostics**

Add this test:

```python
def test_eopkg_gatv2_class_aware_bilinear_plus_prior_exposes_raw_and_normalized_diagnostics():
    model = EOPKGGATv2(
        feature_layout=_feature_layout(),
        hidden_dim=8,
        output_dim=4,
        dropout=0.0,
        structural_mode=True,
        fusion_mode="ClassAwareStructuralScoring",
        structural_score_mode="bilinear_with_prior",
        structural_logit_scale_init=0.1,
    )
    contract = _contract(
        batch_size=2,
        num_nodes=5,
        output_dim=4,
        struct_x=torch.randn(6, 3, dtype=torch.float32),
        structural_edge_index=_struct_edge_index(num_nodes=6),
        struct_node_to_class_index=torch.tensor([0, 1, 1, 2, 3, -1], dtype=torch.long),
    )

    logits = model(contract)

    assert logits.shape == torch.Size([2, 4])
    assert isinstance(model.last_structural_raw_class_logits, torch.Tensor)
    assert isinstance(model.last_structural_normalized_class_logits, torch.Tensor)
    assert isinstance(model.last_structural_observed_scale, torch.Tensor)
    assert model.last_structural_raw_class_logits.shape == torch.Size([2, 4])
    assert model.last_structural_normalized_class_logits.shape == torch.Size([2, 4])
    assert model.last_structural_observed_scale.shape == torch.Size([2, 1])
```

- [ ] **Step 2: Run test and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/domain/test_eopkg_gatv2_dual_encoder.py::test_eopkg_gatv2_class_aware_bilinear_plus_prior_exposes_raw_and_normalized_diagnostics -v
```

Expected:

```text
FAILED with assertion that raw/normalized structural diagnostics are missing
```

- [ ] **Step 3: Implement bilinear-plus-prior node scoring**

In `EOPKGGATv2.__init__`, add:

```python
self.struct_bilinear = nn.Bilinear(self.struct_hidden_dim, self.struct_hidden_dim, 1, bias=False)
self.struct_prior_head = nn.Linear(self.struct_hidden_dim, 1)
self.last_structural_raw_class_logits: torch.Tensor | None = None
self.last_structural_normalized_class_logits: torch.Tensor | None = None
self.last_structural_observed_scale: torch.Tensor | None = None
```

Update `_clear_structural_diagnostics`:

```python
self.last_structural_raw_class_logits = None
self.last_structural_normalized_class_logits = None
self.last_structural_observed_scale = None
```

Update `_class_aware_structural_logits` to compute raw class logits first:

```python
query = self.struct_query_proj(obs_context)
keys = self.struct_key_proj(h_node)

if self.structural_score_mode == "cosine":
    q_norm = torch.nn.functional.normalize(query, p=2, dim=1, eps=1e-12)
    k_norm = torch.nn.functional.normalize(keys, p=2, dim=1, eps=1e-12)
    node_logits = torch.matmul(q_norm, k_norm.transpose(0, 1))
else:
    expanded_query = query.unsqueeze(1).expand(-1, keys.size(0), -1)
    expanded_keys = keys.unsqueeze(0).expand(query.size(0), -1, -1)
    bilinear_scores = self.struct_bilinear(expanded_query, expanded_keys).squeeze(-1)
    prior_scores = self.struct_prior_head(keys).transpose(0, 1)
    node_logits = bilinear_scores + prior_scores
```

Then keep late pooling:

```python
raw_class_logits = torch.full(
    (query.size(0), int(num_classes)),
    fill_value=torch.finfo(node_logits.dtype).min,
    dtype=node_logits.dtype,
    device=node_logits.device,
)
for class_idx in range(int(num_classes)):
    node_mask = node_to_class == class_idx
    if torch.any(node_mask):
        raw_class_logits[:, class_idx] = torch.logsumexp(node_logits[:, node_mask], dim=1)
raw_class_logits = torch.where(torch.isfinite(raw_class_logits), raw_class_logits, torch.zeros_like(raw_class_logits))
scaled_class_logits, normalized_class_logits, observed_scale = self._scale_structural_class_logits(
    raw_class_logits=raw_class_logits,
    observed_logits=obs_context.new_zeros((obs_context.size(0), int(num_classes))),
)
```

In this task, pass real `observed_logits` instead of the zero placeholder by changing `_class_aware_structural_logits` signature to:

```python
observed_logits: torch.Tensor,
```

and call:

```python
scaled_class_logits, normalized_class_logits, observed_scale = self._scale_structural_class_logits(
    raw_class_logits=raw_class_logits,
    observed_logits=observed_logits,
)
return node_logits, raw_class_logits, normalized_class_logits, observed_scale, scaled_class_logits
```

Update the class-aware forward branch:

```python
(
    structural_node_logits,
    structural_raw_class_logits,
    structural_normalized_class_logits,
    structural_observed_scale,
    structural_class_logits,
) = self._class_aware_structural_logits(
    obs_context=obs_context,
    h_node=h_norm,
    node_to_class=node_to_class,
    num_classes=observed_logits.size(1),
    observed_logits=observed_logits,
)
self.last_observed_logits = observed_logits.detach()
self.last_structural_node_logits = structural_node_logits.detach()
self.last_structural_raw_class_logits = structural_raw_class_logits.detach()
self.last_structural_normalized_class_logits = structural_normalized_class_logits.detach()
self.last_structural_observed_scale = structural_observed_scale.detach()
self.last_structural_class_logits = structural_class_logits.detach()
return observed_logits + structural_class_logits
```

- [ ] **Step 4: Run targeted model tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/domain/test_eopkg_gatv2_dual_encoder.py tests/domain/test_eopkg_models_forward.py -v
```

Expected:

```text
PASSED
```

---

## Task 3: Add Trainer Structural Auxiliary Loss Tests

**Files:**

- Modify: `tests/application/test_trainer_forward_stats_logging.py`
- Modify: `src/application/use_cases/trainer.py`

- [ ] **Step 1: Write failing unit test for set-aware structural loss**

Add this test:

```python
def test_structural_set_loss_rewards_any_allowed_target_candidate():
    trainer = _make_trainer(model=_ClassAwareDiagnosticModel(output_dim=3))
    structural_logits = torch.tensor([[0.0, 5.0, -5.0]], dtype=torch.float32)
    targets = torch.tensor([2], dtype=torch.long)
    allowed_mask = torch.tensor([[False, True, True]], dtype=torch.bool)

    set_loss = trainer._compute_structural_set_loss(
        structural_logits=structural_logits,
        targets=targets,
        allowed_target_mask=allowed_mask,
    )
    exact_loss = torch.nn.functional.cross_entropy(structural_logits, targets)

    assert float(set_loss.item()) < 0.01
    assert float(exact_loss.item()) > 9.0
```

Meaning: strict target is class `2`, but class `1` is also structurally allowed and has high score. Set-aware loss should not punish that as heavily as exact CE.

- [ ] **Step 2: Run test and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/application/test_trainer_forward_stats_logging.py::test_structural_set_loss_rewards_any_allowed_target_candidate -v
```

Expected:

```text
FAILED with AttributeError: 'ModelTrainer' object has no attribute '_compute_structural_set_loss'
```

- [ ] **Step 3: Implement set-aware structural loss helper**

In `src/application/use_cases/trainer.py`, import functional API if not already present:

```python
import torch.nn.functional as F
```

Add method:

```python
def _compute_structural_set_loss(
    self,
    *,
    structural_logits: torch.Tensor,
    targets: torch.Tensor,
    allowed_target_mask: torch.Tensor | None,
) -> torch.Tensor:
    if not isinstance(allowed_target_mask, torch.Tensor):
        return F.cross_entropy(structural_logits, targets)

    valid_mask = allowed_target_mask.to(device=structural_logits.device, dtype=torch.bool).clone()
    valid_mask.scatter_(1, targets.view(-1, 1).to(device=structural_logits.device), True)
    log_probs = F.log_softmax(structural_logits, dim=1)
    masked_log_probs = log_probs.masked_fill(~valid_mask, torch.finfo(log_probs.dtype).min)
    set_log_prob = torch.logsumexp(masked_log_probs, dim=1)
    return -set_log_prob.mean()
```

- [ ] **Step 4: Run test and verify GREEN**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/application/test_trainer_forward_stats_logging.py::test_structural_set_loss_rewards_any_allowed_target_candidate -v
```

Expected:

```text
PASSED
```

---

## Task 4: Compose Main CE With Structural Auxiliary Loss

**Files:**

- Modify: `tests/application/test_trainer_forward_stats_logging.py`
- Modify: `src/application/use_cases/trainer.py`

- [ ] **Step 1: Write failing trainer test for auxiliary loss logging**

Add this test:

```python
def test_trainer_adds_and_logs_structural_auxiliary_loss(caplog):
    model = _ClassAwareDiagnosticModel(output_dim=3)
    trainer = _make_trainer(
        model=model,
        config_overrides={
            "structural_aux_loss_enabled": True,
            "structural_aux_loss_weight": 0.05,
            "structural_aux_exact_loss_weight": 0.01,
        },
    )
    loader = [
        Data(
            x_cat=torch.zeros((2, 2), dtype=torch.long),
            x_num=torch.zeros((2, 1), dtype=torch.float32),
            edge_index=torch.tensor([[0], [1]], dtype=torch.long),
            edge_type=torch.zeros(1, dtype=torch.long),
            y=torch.tensor([2], dtype=torch.long),
            batch=torch.zeros(2, dtype=torch.long),
            allowed_target_mask=torch.tensor([[False, True, True]], dtype=torch.bool),
        )
    ]

    with caplog.at_level(logging.INFO):
        trainer._run_epoch(
            loader,
            optimizer=None,
            training=False,
            epoch_index=1,
            total_epochs=1,
        )

    text = caplog.text
    assert "structural_aux_set_loss=" in text
    assert "structural_aux_exact_loss=" in text
    assert "structural_aux_total_loss=" in text
```

Ensure `_ClassAwareDiagnosticModel` sets `last_structural_class_logits` to a non-zero tensor with shape `[B, C]`.

- [ ] **Step 2: Run test and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/application/test_trainer_forward_stats_logging.py::test_trainer_adds_and_logs_structural_auxiliary_loss -v
```

Expected:

```text
FAILED with AssertionError: 'structural_aux_set_loss=' not found
```

- [ ] **Step 3: Parse training config**

In `ModelTrainer.__init__`, add:

```python
self.structural_aux_loss_enabled = bool(config.get("structural_aux_loss_enabled", False))
self.structural_aux_loss_weight = float(config.get("structural_aux_loss_weight", 0.05))
self.structural_aux_exact_loss_weight = float(config.get("structural_aux_exact_loss_weight", 0.01))
```

In `_log_params`, add:

```python
self.tracker.log_param("structural_aux_loss_enabled", self.structural_aux_loss_enabled)
self.tracker.log_param("structural_aux_loss_weight", self.structural_aux_loss_weight)
self.tracker.log_param("structural_aux_exact_loss_weight", self.structural_aux_exact_loss_weight)
```

Guard `self.tracker` consistently with surrounding code.

- [ ] **Step 4: Add auxiliary loss to `_run_epoch`**

After:

```python
loss = self.criterion(effective_logits, targets)
```

add:

```python
structural_aux_set_loss = None
structural_aux_exact_loss = None
if self.structural_aux_loss_enabled:
    structural_logits = getattr(self.model, "last_structural_class_logits", None)
    if isinstance(structural_logits, torch.Tensor) and structural_logits.shape == effective_logits.shape:
        allowed_target_mask = contract.get("allowed_target_mask")
        structural_aux_set_loss = self._compute_structural_set_loss(
            structural_logits=structural_logits,
            targets=targets,
            allowed_target_mask=allowed_target_mask if isinstance(allowed_target_mask, torch.Tensor) else None,
        )
        structural_aux_exact_loss = F.cross_entropy(structural_logits, targets)
        loss = (
            loss
            + float(self.structural_aux_loss_weight) * structural_aux_set_loss
            + float(self.structural_aux_exact_loss_weight) * structural_aux_exact_loss
        )
```

Accumulate batch sums:

```python
if structural_aux_set_loss is not None:
    forward_stats["structural_aux_set_loss_sum"] = float(forward_stats.get("structural_aux_set_loss_sum", 0.0)) + float(structural_aux_set_loss.detach().cpu().item())
    forward_stats["structural_aux_exact_loss_sum"] = float(forward_stats.get("structural_aux_exact_loss_sum", 0.0)) + float(structural_aux_exact_loss.detach().cpu().item())
    forward_stats["structural_aux_loss_batches"] = int(forward_stats.get("structural_aux_loss_batches", 0)) + 1
```

- [ ] **Step 5: Log auxiliary loss summary**

In `_log_forward_stats_summary`, compute:

```python
structural_aux_batches = int(bucket.get("structural_aux_loss_batches", 0))
structural_aux_set_loss = (
    float(bucket.get("structural_aux_set_loss_sum", 0.0)) / float(structural_aux_batches)
    if structural_aux_batches > 0
    else 0.0
)
structural_aux_exact_loss = (
    float(bucket.get("structural_aux_exact_loss_sum", 0.0)) / float(structural_aux_batches)
    if structural_aux_batches > 0
    else 0.0
)
structural_aux_total_loss = (
    float(self.structural_aux_loss_weight) * structural_aux_set_loss
    + float(self.structural_aux_exact_loss_weight) * structural_aux_exact_loss
)
```

Append to log line:

```text
structural_aux_set_loss=%.6f structural_aux_exact_loss=%.6f structural_aux_total_loss=%.6f
```

Log tracker metrics:

```python
self.tracker.log_metric(f"{metric_prefix}_structural_aux_set_loss", structural_aux_set_loss)
self.tracker.log_metric(f"{metric_prefix}_structural_aux_exact_loss", structural_aux_exact_loss)
self.tracker.log_metric(f"{metric_prefix}_structural_aux_total_loss", structural_aux_total_loss)
```

Only log these metrics when `structural_aux_batches > 0`.

- [ ] **Step 6: Run trainer tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/application/test_trainer_forward_stats_logging.py -v
```

Expected:

```text
PASSED
```

---

## Task 5: Wire Model Config From CLI/Factory Path

**Files:**

- Modify: `src/cli.py`
- Modify: `tests/application/test_cli_flatten_config.py` or another existing CLI config test if it already covers model factory parameters

- [ ] **Step 1: Write failing test for model config propagation**

First extract model-factory keyword construction into a helper named
`_build_model_factory_kwargs`. Then add a focused test for that helper:

```python
def test_cli_passes_structural_scoring_config_to_model_factory(monkeypatch):
    from src.cli import _build_model_factory_kwargs

    model_cfg = {
        "type": "EOPKGGATv2",
        "hidden_dim": 64,
        "dropout": 0.2,
        "pooling_strategy": "last_node",
        "structural_mode": True,
        "struct_encoder_type": "GATv2Conv",
        "struct_hidden_dim": 64,
        "cross_attention_heads": 4,
        "fusion_mode": "ClassAwareStructuralScoring",
        "structural_score_mode": "bilinear_with_prior",
        "structural_logit_scale_init": 0.1,
        "structural_logit_scale_max": 2.0,
        "structural_observed_scale_min": 1.0,
        "structural_observed_scale_max": 10.0,
    }

    kwargs = _build_model_factory_kwargs(
        model_cfg=model_cfg,
        feature_layout=_minimal_feature_layout(),
        output_dim=37,
    )

    assert kwargs["model_type"] == "EOPKGGATv2"
    assert kwargs["structural_score_mode"] == "bilinear_with_prior"
    assert kwargs["structural_logit_scale_init"] == 0.1
    assert kwargs["structural_logit_scale_max"] == 2.0
    assert kwargs["structural_observed_scale_min"] == 1.0
    assert kwargs["structural_observed_scale_max"] == 10.0
```

Add this local helper inside the test file if it does not already exist:

```python
def _minimal_feature_layout():
    return FeatureLayout(
        cat_features={"concept:name": 4},
        cat_feature_names=["concept:name"],
        num_dim=0,
    )
```

Import `FeatureLayout` from the same module used by nearby CLI/model tests.

The test must fail before `_build_model_factory_kwargs` exists or before it
includes the new structural scoring keys.

- [ ] **Step 2: Run test and verify RED**

Run the selected test:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/application/test_cli_flatten_config.py::test_cli_passes_structural_scoring_config_to_model_factory -v
```

Expected:

```text
FAILED
```

- [ ] **Step 3: Implement model kwargs propagation**

In `src/cli.py`, add:

```python
def _build_model_factory_kwargs(
    *,
    model_cfg: Dict[str, Any],
    feature_layout: Any,
    output_dim: int,
) -> Dict[str, Any]:
    hidden_dim = int(model_cfg.get("hidden_dim", 64))
    return {
        "model_type": str(model_cfg.get("type", model_cfg.get("model_type", "BaselineGCN"))),
        "feature_layout": feature_layout,
        "hidden_dim": hidden_dim,
        "output_dim": int(output_dim),
        "dropout": float(model_cfg.get("dropout", 0.2)),
        "pooling_strategy": str(model_cfg.get("pooling_strategy", "global_mean")).strip().lower(),
        "structural_mode": model_cfg.get("structural_mode", True),
        "struct_encoder_type": str(model_cfg.get("struct_encoder_type", "GATv2Conv")),
        "struct_hidden_dim": int(model_cfg.get("struct_hidden_dim", hidden_dim)),
        "cross_attention_heads": int(model_cfg.get("cross_attention_heads", 4)),
        "fusion_mode": str(model_cfg.get("fusion_mode", "Attention")),
        "structural_score_mode": str(model_cfg.get("structural_score_mode", "bilinear_with_prior")),
        "structural_logit_scale_init": float(model_cfg.get("structural_logit_scale_init", 0.1)),
        "structural_logit_scale_max": float(model_cfg.get("structural_logit_scale_max", 2.0)),
        "structural_observed_scale_min": float(model_cfg.get("structural_observed_scale_min", 1.0)),
        "structural_observed_scale_max": float(model_cfg.get("structural_observed_scale_max", 10.0)),
    }
```

Replace the current inline `create_model` call with:

```python
model_kwargs = _build_model_factory_kwargs(
    model_cfg=model_cfg,
    feature_layout=feature_layout,
    output_dim=output_dim,
)
model_type = str(model_kwargs["model_type"])
model = create_model(**model_kwargs)
```

- [ ] **Step 4: Run CLI config tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/application/test_cli_flatten_config.py -v
```

Expected:

```text
PASSED
```

---

## Task 6: Update Experiment UI Config Catalog

**Files:**

- Modify: `configs/ui/config_catalog.yaml`
- Modify: `tests/application/test_experiment_ui_catalog_fields.py`

- [ ] **Step 1: Write failing catalog test**

Add assertions:

```python
def test_class_aware_structural_scoring_fields_are_in_catalog():
    catalog = _load_catalog()
    expected = {
        "model.structural_score_mode",
        "model.structural_logit_scale_init",
        "model.structural_logit_scale_max",
        "model.structural_observed_scale_min",
        "model.structural_observed_scale_max",
        "training.structural_aux_loss_enabled",
        "training.structural_aux_loss_weight",
        "training.structural_aux_exact_loss_weight",
    }
    assert expected.issubset(set(catalog.keys()))
```

- [ ] **Step 2: Run test and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/application/test_experiment_ui_catalog_fields.py::test_class_aware_structural_scoring_fields_are_in_catalog -v
```

Expected:

```text
FAILED
```

- [ ] **Step 3: Add config catalog entries**

Add entries with these defaults:

```yaml
model.structural_score_mode:
  path: model.structural_score_mode
  section: model
  label: 'Модель: Structural score mode'
  description: Scoring function used by ClassAwareStructuralScoring.
  type: enum
  default: bilinear_with_prior
  enum:
    - bilinear_with_prior
    - cosine

model.structural_logit_scale_init:
  path: model.structural_logit_scale_init
  section: model
  label: 'Модель: Structural logit scale init'
  description: Initial gamma for scaled structural logits.
  type: float
  default: 0.1

model.structural_logit_scale_max:
  path: model.structural_logit_scale_max
  section: model
  label: 'Модель: Structural logit scale max'
  description: Upper clamp for learnable structural gamma.
  type: float
  default: 2.0

model.structural_observed_scale_min:
  path: model.structural_observed_scale_min
  section: model
  label: 'Модель: Structural observed scale min'
  description: Lower clamp for detached observed-logit scale alignment.
  type: float
  default: 1.0

model.structural_observed_scale_max:
  path: model.structural_observed_scale_max
  section: model
  label: 'Модель: Structural observed scale max'
  description: Upper clamp for detached observed-logit scale alignment.
  type: float
  default: 10.0

training.structural_aux_loss_enabled:
  path: training.structural_aux_loss_enabled
  section: training
  label: 'Тренування: Structural auxiliary loss'
  description: Enables set-aware auxiliary loss for class-aware structural logits.
  type: bool
  default: false

training.structural_aux_loss_weight:
  path: training.structural_aux_loss_weight
  section: training
  label: 'Тренування: Structural set loss weight'
  description: Weight for set-aware structural auxiliary loss.
  type: float
  default: 0.05

training.structural_aux_exact_loss_weight:
  path: training.structural_aux_exact_loss_weight
  section: training
  label: 'Тренування: Structural exact loss weight'
  description: Small exact CE auxiliary weight for strict target nudging.
  type: float
  default: 0.01
```

- [ ] **Step 4: Run catalog tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/application/test_experiment_ui_catalog_fields.py -v
```

Expected:

```text
PASSED
```

---

## Task 7: Update Documentation

**Files:**

- Modify: `docs/GNN_RUNTIME_MVP2_5.MD`
- Modify: `docs/current/project-state.md`
- Optional Modify: `docs/current/architecture-debt.md` only if implementation reveals new unresolved debt.

- [ ] **Step 1: Update `docs/GNN_RUNTIME_MVP2_5.MD`**

Add a subsection under `ClassAwareStructuralScoring`:

```markdown
#### Scale-Aligned Structural Logits

`ClassAwareStructuralScoring` uses two structural score components:

```text
raw_node_score(i, v) = bilinear(obs_context_i, h_struct_v) + prior(h_struct_v)
raw_class_score(i, j) = LogSumExp({raw_node_score(i, v) | node_to_class[v] = j})
scaled_struct(i, :) = gamma * detached_observed_scale(i) * LayerNorm(raw_class_score(i, :))
final_logits = observed_logits + scaled_struct
```

`detached_observed_scale` aligns structural evidence with observed-logit magnitude
without allowing structural loss to reshape the observed branch through the scale.
```

- [ ] **Step 2: Document auxiliary loss**

Add:

```markdown
#### Structural Auxiliary Loss

Training may add:

```text
loss = CE(final_logits, y)
     + lambda_set * L_struct_set
     + lambda_exact * CE(structural_logits, y)
```

`L_struct_set` rewards probability mass assigned to any class in
`allowed_target_mask`, with the exact target forcibly included. This avoids
over-penalizing parallel alternatives while still giving structural logits a
direct gradient.
```

- [ ] **Step 3: Update `docs/current/project-state.md`**

Add one short status bullet:

```markdown
- `ClassAwareStructuralScoring` uses scale-aligned bilinear-plus-prior structural logits and optional set-aware structural auxiliary loss.
```

- [ ] **Step 4: Run documentation grep check**

Run:

```powershell
Select-String -Path docs\GNN_RUNTIME_MVP2_5.MD,docs\current\project-state.md -Pattern "Scale-Aligned Structural Logits|Structural Auxiliary Loss|bilinear-plus-prior"
```

Expected: each phrase appears at least once.

---

## Task 8: Verification Protocol

**Files:**

- No code changes in this task.

- [ ] **Step 1: Run focused model and trainer tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/domain/test_eopkg_gatv2_dual_encoder.py tests/domain/test_eopkg_models_forward.py tests/application/test_trainer_forward_stats_logging.py -v
```

Expected:

```text
PASSED
```

- [ ] **Step 2: Run catalog and contract tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/application/test_experiment_ui_catalog_fields.py tests/contracts/test_graph_tensor_contract_mvp2_optional_fields.py tests/domain/test_dynamic_graph_builder_masks.py -v
```

Expected:

```text
PASSED
```

- [ ] **Step 3: Run architecture guard**

Run:

```powershell
.\.venv\Scripts\python.exe tools\architecture_guard.py
```

Expected:

```text
[ARCH_GUARD] OK: architecture dependency rules satisfied.
```

- [ ] **Step 4: Run regression suite**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -v
```

Expected:

```text
PASSED
```

- [ ] **Step 5: Run one controlled loan experiment**

Use a non-research smoke config first. Required log expectations:

```text
Forward stats [train] contains structural_to_observed_logit_ratio
Forward stats [validation] contains structural_to_observed_logit_ratio
Forward stats contains structural_aux_set_loss, structural_aux_exact_loss, and structural_aux_total_loss
```

Acceptance range for initial experiment:

```text
validation_structural_to_observed_logit_ratio: 0.05 - 0.30
structural_aux_total_loss: finite and non-zero
numeric guard warnings: absent
```

If `structural_to_observed_logit_ratio > 0.5`, lower either:

```yaml
model.structural_logit_scale_init
training.structural_aux_loss_weight
```

If `structural_to_observed_logit_ratio < 0.03`, raise one of:

```yaml
model.structural_logit_scale_init: 0.2
training.structural_aux_loss_weight: 0.1
```

---

## Task 9: Implementation Report

**Files:**

- Create: `docs/worklogs/MVP2_5_Structural_Logit_Scale_And_Aux_Loss_Implementation_Report_2026-05-07.MD`

- [ ] **Step 1: Create report after verification**

Report must include:

```markdown
# Structural Logit Scale And Auxiliary Loss Implementation Report

## What Changed

## New Config Keys

## Where To Inspect Metrics

## Verification Results

## Experiment Caveats

## Next Recommended Experiment
```

- [ ] **Step 2: Include exact command outputs**

Record the final status of:

```powershell
.\.venv\Scripts\python.exe tools\architecture_guard.py
.\.venv\Scripts\python.exe -m pytest tests/ -v
```

- [ ] **Step 3: Include documentation drift check**

The report must list updated documentation files:

```text
docs/GNN_RUNTIME_MVP2_5.MD
docs/current/project-state.md
configs/ui/config_catalog.yaml
```

---

## Risks And Controls

### Risk: Structural Branch Dominates Observed Branch

Control:

```text
structural_logit_scale_max = 2.0
observed_scale_max = 10.0
structural_to_observed_logit_ratio monitored every epoch
```

### Risk: Auxiliary Loss Punishes Parallel Alternatives

Control:

```text
L_struct_set uses allowed_target_mask and forcibly includes exact target.
Exact structural CE has a small default weight: 0.01.
```

### Risk: Linear Prior Becomes Static Class Shortcut

Control:

```text
The prior is only one additive component.
The main node score remains prefix-aware through bilinear(obs_context, h_struct).
```

### Risk: More Hyperparameters Make Experiment UI Noisy

Control:

```text
Expose only scale/auxiliary weights required for debugging.
Keep defaults conservative.
Do not add MLP scorer in this iteration.
```

---

## Self-Review

- Spec coverage: covers scale mismatch, bilinear-plus-prior scoring, observed-scale normalization, set-aware auxiliary loss, Experiment UI config, docs, and verification.
- Placeholder scan: no unresolved placeholders are present.
- Type consistency: planned model diagnostics use tensor names under `last_structural_*`; trainer reads `last_structural_class_logits` for auxiliary loss and keeps existing diagnostics compatible.
- Scope check: this is one implementation unit. It does not include MLP scoring or simulator/debug dataset redesign.
