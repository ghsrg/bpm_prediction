# EOPKGTopologyConditioned Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `test-driven-development` for every implementation task. Use `verification-before-completion` before claiming completion. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `model.type=EOPKGTopologyConditioned` as a new model family for topology-conditioned candidate scoring.

**Architecture:** Stage 1 is MVP-compatible and returns label-level logits `[B, C_train]` through topology node scoring + MIL aggregation, so existing trainer/eval/checkpoint/MLflow code keeps working. Stage 2 introduces true dynamic candidate outputs `[B, C_v]` after Stage 1 proves useful.

**Tech Stack:** Python, PyTorch, PyTorch Geometric, existing `GraphTensorContract`, `ModelTrainer`, model registry in `src/domain/models/factory.py`, config catalog in `configs/ui/config_catalog.yaml`.

---

## Design Decision

Do **not** add this as another `fusion_mode` inside `EOPKGGATv2`.

Reason:

```text
EOPKGGATv2 fusion_mode = how topology modifies fixed-head logits [B, C_train]
EOPKGTopologyConditioned = topology defines candidate prototypes and scoring contract
```

The new model is a separate `model.type`:

```yaml
model:
  type: EOPKGTopologyConditioned
  observed_encoder: GATv2
  struct_encoder: GATv2
  candidate_scoring: cosine
  candidate_pooling: logmeanexp
```

First implementation supports only:

```text
observed_encoder: GATv2
struct_encoder: GATv2
candidate_scoring: cosine | bilinear
candidate_pooling: logmeanexp | mean | max
```

Unsupported values must fail-fast.

---

## Stage 1 Scope: MVP-Compatible Label-Level Candidate Scoring

Stage 1 keeps the existing trainer contract:

```text
forward(contract) -> logits [B, C_train]
loss = CE(logits, y)
```

Internally the model does candidate scoring over topology nodes:

```text
z_prefix = ObservedPrefixEncoder(prefix)                 # [B, D]
h_nodes = TopologyCandidateEncoder(struct_x, node_id_emb, edges) # [V, D]
score_node[b, v] = sim(z_prefix[b], h_nodes[v])           # [B, V]
score_class[b, c] = MIL_pool(score_node[b, v where node_to_class[v] == c])
```

For duplicate labels/classes:

```text
score_class_c = LogSumExp(score_node_v for v in V_c) - log(|V_c|)
```

This avoids duplicate-count score inflation while allowing the model to select
the best matching topology instance.

Stage 1 does **not** solve new unseen output classes yet. It establishes whether
candidate-prototype scoring improves known-label topology transfer and
diagnostics without rewriting the whole runtime.

### Stage 1 Guardrails

- The structural candidate encoder must not rely on numeric `struct_x` alone.
  Stage 1 must add minimal identity grounding through `struct_node_emb` for
  `struct_node_to_class_index >= 0` and `struct_non_target_emb` for gateways,
  starts, ends, and other non-target nodes.
- `struct_x` remains useful as topology/statistics features, but activity
  identity comes from `node_to_class` grounding. Full text/BPMN semantic
  prototypes are deferred to the dynamic candidate contract stage.
- The model must return clean class scores. It must not apply
  `allowed_target_mask` internally. Mask-guided loss, allowed-set loss, and
  masked inference remain trainer/evaluator responsibilities.
- Nodes with `struct_node_to_class_index == -1` participate in message passing
  but must be filtered out before class-level MIL pooling.
- Candidate temperature must be stable by default. Stage 1 uses fixed
  `candidate_temperature=0.1` unless `candidate_temperature_trainable=true` is
  explicitly enabled for a controlled ablation.

---

## Stage 2 Scope: Dynamic Candidate Contract

Stage 2 is explicitly out of first implementation. It will add:

```text
forward_candidate(contract) -> CandidatePredictionOutput
candidate_logits [B, C_v]
candidate_ids per version
candidate_to_label mapping
trainer/evaluator support for variable candidate dimensions
```

Stage 2 should start only after Stage 1 proves stable or its failure is
diagnostically understood.

---

## Files To Create Or Modify

### Create

- `src/domain/models/topology_conditioned.py`
  - new `EOPKGTopologyConditioned` model family;
  - reusable observed GATv2 encoder;
  - structural candidate encoder;
  - cosine/bilinear scoring;
  - MIL aggregation.

- `tests/domain/test_eopkg_topology_conditioned.py`
  - model registry tests;
  - forward shape tests;
  - duplicate-label MIL tests;
  - topology sensitivity tests;
  - unsupported config fail-fast tests.

### Modify

- `src/domain/models/factory.py`
  - import/register `topology_conditioned.py`.

- `src/cli.py`
  - pass new model config keys from `_build_model_factory_kwargs`.

- `configs/ui/config_catalog.yaml`
  - add model keys for observed/structural encoder, candidate scoring, pooling,
    and candidate temperature.

- `docs/GNN_RUNTIME_MVP2_5.MD`
  - document new model type and Stage 1 fixed-logit compatibility.

- `docs/GNN_LEARNING_METODOLOGY.MD`
  - mark Stage 1 implementation status when complete.

- `docs/current/project-state.md`
  - add runtime status summary.

- `docs/current/architecture-debt.md`
  - add `dynamic_candidate_contract` debt: Stage 1 still returns `[B, C_train]`,
    target is `[B, C_v]`.

---

## Task 1: Register Empty Model Skeleton

**Files:**

- Create: `tests/domain/test_eopkg_topology_conditioned.py`
- Create: `src/domain/models/topology_conditioned.py`
- Modify: `src/domain/models/factory.py`

- [ ] **Step 1: Write failing registry test**

Add to `tests/domain/test_eopkg_topology_conditioned.py`:

```python
from __future__ import annotations

import pytest
import torch

from src.domain.entities.feature_config import FeatureLayout
from src.domain.models.factory import create_model, get_registered_models


def _layout() -> FeatureLayout:
    return FeatureLayout(
        cat_features={"concept:name": 5},
        num_dim=2,
        cat_feature_names=["concept:name"],
        num_feature_names=["duration", "elapsed"],
    )


def test_topology_conditioned_model_is_registered():
    assert "EOPKGTopologyConditioned" in get_registered_models()
    model = create_model(
        "EOPKGTopologyConditioned",
        feature_layout=_layout(),
        hidden_dim=8,
        output_dim=5,
        dropout=0.0,
    )
    assert model.__class__.__name__ == "EOPKGTopologyConditioned"
```

- [ ] **Step 2: Run RED**

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\domain\test_eopkg_topology_conditioned.py::test_topology_conditioned_model_is_registered -v
```

Expected: fail because model is not registered/importable.

- [ ] **Step 3: Implement minimal skeleton**

Create `src/domain/models/topology_conditioned.py`:

```python
"""Topology-conditioned candidate scoring models."""

from __future__ import annotations

import torch
from torch import nn

from src.domain.entities.feature_config import FeatureLayout
from src.domain.entities.tensor_contract import GraphTensorContract
from src.domain.models.base_gnn import BaseGNN
from src.domain.models.factory import register_model


@register_model("EOPKGTopologyConditioned")
class EOPKGTopologyConditioned(BaseGNN):
    """Topology-conditioned candidate scoring model, Stage 1 compatible head."""

    def __init__(
        self,
        feature_layout: FeatureLayout,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.2,
        pooling_strategy: str = "last_node",
        observed_encoder: str = "GATv2",
        struct_encoder: str = "GATv2",
        candidate_scoring: str = "cosine",
        candidate_pooling: str = "logmeanexp",
        candidate_temperature_init: float = 0.1,
        candidate_temperature_min: float = 0.05,
        candidate_temperature_max: float = 10.0,
        candidate_temperature_trainable: bool = False,
    ) -> None:
        super().__init__()
        self.feature_layout = feature_layout
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.dropout_p = float(dropout)
        self.pooling_strategy = str(pooling_strategy)
        self.observed_encoder = str(observed_encoder or "GATv2").strip().lower()
        self.struct_encoder = str(struct_encoder or "GATv2").strip().lower()
        self.candidate_scoring = str(candidate_scoring or "cosine").strip().lower()
        self.candidate_pooling = str(candidate_pooling or "logmeanexp").strip().lower()
        self.candidate_temperature_min = float(candidate_temperature_min)
        self.candidate_temperature_max = float(candidate_temperature_max)
        temperature = torch.tensor(float(candidate_temperature_init), dtype=torch.float32)
        if bool(candidate_temperature_trainable):
            self.candidate_temperature = nn.Parameter(temperature)
        else:
            self.register_buffer("candidate_temperature", temperature)

    def forward(self, contract: GraphTensorContract) -> torch.Tensor:
        raise NotImplementedError("EOPKGTopologyConditioned forward is not implemented yet.")
```

Modify `src/domain/models/factory.py`:

```python
from src.domain.models import topology_conditioned as _topology_conditioned  # noqa: E402,F401
```

- [ ] **Step 4: Run GREEN**

Same command as Step 2. Expected: pass.

---

## Task 2: Validate Config Fail-Fast Rules

**Files:**

- Modify: `tests/domain/test_eopkg_topology_conditioned.py`
- Modify: `src/domain/models/topology_conditioned.py`

- [ ] **Step 1: Write failing tests**

Add:

```python
@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("observed_encoder", "GCN", "observed_encoder"),
        ("struct_encoder", "GCN", "struct_encoder"),
        ("candidate_scoring", "mlp", "candidate_scoring"),
        ("candidate_pooling", "sum", "candidate_pooling"),
    ],
)
def test_topology_conditioned_rejects_unsupported_options(key: str, value: str, message: str):
    kwargs = {
        "feature_layout": _layout(),
        "hidden_dim": 8,
        "output_dim": 5,
        "dropout": 0.0,
        key: value,
    }
    with pytest.raises(ValueError, match=message):
        create_model("EOPKGTopologyConditioned", **kwargs)
```

- [ ] **Step 2: Run RED**

Expected: fail because constructor accepts unsupported values.

- [ ] **Step 3: Implement validation**

In `__init__`:

```python
if self.observed_encoder not in {"gatv2"}:
    raise ValueError("Unsupported model.observed_encoder for EOPKGTopologyConditioned. Available: ['GATv2']")
if self.struct_encoder not in {"gatv2"}:
    raise ValueError("Unsupported model.struct_encoder for EOPKGTopologyConditioned. Available: ['GATv2']")
if self.candidate_scoring not in {"cosine", "bilinear"}:
    raise ValueError("Unsupported model.candidate_scoring. Available: ['cosine', 'bilinear']")
if self.candidate_pooling not in {"logmeanexp", "mean", "max"}:
    raise ValueError("Unsupported model.candidate_pooling. Available: ['logmeanexp', 'mean', 'max']")
if self.candidate_temperature_min <= 0.0:
    raise ValueError("model.candidate_temperature_min must be positive.")
if self.candidate_temperature_max < self.candidate_temperature_min:
    raise ValueError("model.candidate_temperature_max must be >= candidate_temperature_min.")
```

- [ ] **Step 4: Run GREEN**

Expected: all Task 2 tests pass.

---

## Task 3: Implement Observed Encoder Path

**Files:**

- Modify: `tests/domain/test_eopkg_topology_conditioned.py`
- Modify: `src/domain/models/topology_conditioned.py`

- [ ] **Step 1: Write failing observed-context test**

Add helper contract:

```python
def _contract() -> dict:
    return {
        "x_cat": torch.tensor([[1], [2], [3]], dtype=torch.long),
        "x_num": torch.tensor([[0.1, 0.2], [0.2, 0.3], [0.3, 0.4]], dtype=torch.float32),
        "edge_index": torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        "edge_type": torch.zeros((2,), dtype=torch.long),
        "batch": torch.zeros((3,), dtype=torch.long),
        "y": torch.tensor([2], dtype=torch.long),
        "num_nodes": 3,
        "struct_x": torch.eye(5, 3, dtype=torch.float32),
        "structural_edge_index": torch.tensor([[1, 2, 3], [2, 3, 4]], dtype=torch.long),
        "structural_edge_weight": torch.ones((3,), dtype=torch.float32),
        "struct_node_to_class_index": torch.tensor([0, 1, 2, 3, 4], dtype=torch.long),
        "allowed_target_mask": torch.tensor([[False, True, True, False, False]], dtype=torch.bool),
    }


def test_topology_conditioned_forward_returns_fixed_label_logits_stage1():
    model = create_model(
        "EOPKGTopologyConditioned",
        feature_layout=_layout(),
        hidden_dim=8,
        output_dim=5,
        dropout=0.0,
        pooling_strategy="last_node",
    )
    logits = model(_contract())
    assert logits.shape == torch.Size([1, 5])
```

- [ ] **Step 2: Run RED**

Expected: fail because forward not implemented.

- [ ] **Step 3: Implement observed encoder**

Use `GATv2Conv` + `global_mean_pool`, following existing `EOPKGGATv2` observed path but keep the file focused.

Required methods:

```python
_encode_observed_input(contract) -> (x, edge_index, batch)
_encode_observed_context(contract) -> z_prefix [B, D]
```

Temporary `forward` may return zeros with correct shape after encoding:

```python
obs = self._encode_observed_context(contract)
return obs.new_zeros((obs.size(0), self.output_dim))
```

- [ ] **Step 4: Run GREEN**

Expected: forward shape test passes.

---

## Task 4: Implement Structural Candidate Encoder

**Files:**

- Modify: `tests/domain/test_eopkg_topology_conditioned.py`
- Modify: `src/domain/models/topology_conditioned.py`

- [ ] **Step 1: Write failing tests**

Add tests requiring:

```text
structural_edge_index
struct_node_to_class_index
```

Missing either field must raise `ValueError`.

Add tests proving identity grounding:

```text
changing struct_node_to_class_index changes h_nodes even when struct_x is identical
non-target nodes with class index -1 receive struct_non_target_emb
```

- [ ] **Step 2: Run RED**

Expected: fail because forward ignores structure.

- [ ] **Step 3: Implement structural encoder**

Add:

```text
struct_node_emb: Embedding(output_dim, hidden_dim)
struct_non_target_emb: Parameter(hidden_dim)
struct_input_proj: lazy Linear for struct_x
struct_fusion: add or gated add between projected struct_x and node identity embedding
struct_conv1: GATv2Conv(hidden_dim, hidden_dim)
struct_norm: LayerNorm(hidden_dim)
```

Required methods:

```python
_node_identity_features(node_to_class, device) -> h_id [V,D]
_struct_input(contract, device) -> (h0 [V,D], node_to_class [V])
_encode_candidates(contract, device) -> (h_nodes [V,D], node_to_class [V])
```

Implementation rule:

```text
target node h_id[v] = struct_node_emb[node_to_class[v]]
non-target node h_id[v] = struct_non_target_emb
h0 = LayerNorm(struct_input_proj(struct_x) + h_id)
```

- [ ] **Step 4: Run GREEN**

Expected: structural payload tests and forward shape pass.

---

## Task 5: Implement Cosine/Bilinear Scoring And MIL Aggregation

**Files:**

- Modify: `tests/domain/test_eopkg_topology_conditioned.py`
- Modify: `src/domain/models/topology_conditioned.py`

- [ ] **Step 1: Write failing MIL test**

Test direct method:

```python
def test_topology_conditioned_mil_logmeanexp_normalizes_duplicate_count():
    model = create_model(
        "EOPKGTopologyConditioned",
        feature_layout=_layout(),
        hidden_dim=8,
        output_dim=3,
        dropout=0.0,
        candidate_pooling="logmeanexp",
    )
    node_scores = torch.tensor([[1.0, 1.0, 0.0, -1.0]], dtype=torch.float32)
    node_to_class = torch.tensor([1, 1, 2, -1], dtype=torch.long)
    class_scores = model._pool_node_scores_to_classes(node_scores, node_to_class)
    assert class_scores.shape == torch.Size([1, 3])
    assert torch.isneginf(class_scores[0, 0])
    assert class_scores[0, 1].item() == pytest.approx(1.0, abs=1e-6)
    assert class_scores[0, 2].item() == pytest.approx(0.0, abs=1e-6)
```

Add explicit `-1` isolation test:

```python
def test_topology_conditioned_pooling_ignores_non_target_minus_one_nodes():
    model = create_model(
        "EOPKGTopologyConditioned",
        feature_layout=_layout(),
        hidden_dim=8,
        output_dim=3,
        dropout=0.0,
        candidate_pooling="max",
    )
    node_scores = torch.tensor([[100.0, 0.25]], dtype=torch.float32)
    node_to_class = torch.tensor([-1, 2], dtype=torch.long)
    class_scores = model._pool_node_scores_to_classes(node_scores, node_to_class)
    assert torch.isneginf(class_scores[0, 0])
    assert torch.isneginf(class_scores[0, 1])
    assert class_scores[0, 2].item() == pytest.approx(0.25, abs=1e-6)
```

Add clean-logits mask contract test:

```python
def test_topology_conditioned_forward_does_not_apply_allowed_target_mask():
    batch = _batch()
    masked = dict(batch)
    masked["allowed_target_mask"] = torch.zeros_like(batch["allowed_target_mask"])
    logits_without_mask = model(_contract(batch))
    logits_with_mask = model(_contract(masked))
    assert torch.allclose(logits_with_mask, logits_without_mask)
```

- [ ] **Step 2: Run RED**

Expected: fail because pooling method does not exist.

- [ ] **Step 3: Implement scoring**

Add:

```text
_score_candidates(obs_context, candidate_embeddings) -> node_scores [B,V]
_pool_node_scores_to_classes(node_scores, node_to_class) -> class_scores [B,C_train]
```

Cosine scoring:

```text
scores = normalize(z_prefix) @ normalize(h_nodes).T
scores = scores / clamp(candidate_temperature)
```

Temperature policy:

```text
default candidate_temperature_init = 0.1
default candidate_temperature_trainable = false
candidate_temperature_trainable = true is an ablation, not the canonical mode
if trainable, clamp to [candidate_temperature_min, candidate_temperature_max]
```

Bilinear scoring:

```text
scores[b,v] = Bilinear(z_prefix[b], h_nodes[v])
```

`forward`:

```text
obs = _encode_observed_context(contract)
candidates, node_to_class = _encode_candidates(contract, obs.device)
node_scores = _score_candidates(obs, candidates)
class_scores = _pool_node_scores_to_classes(node_scores, node_to_class)
return nan_to_num(class_scores, neginf=-1e6)
```

Pooling rule:

```text
valid = node_to_class >= 0
pool only node_scores[:, valid] into classes
never pass -1 into scatter/gather as an index
```

- [ ] **Step 4: Run GREEN**

Expected: MIL test and forward shape pass.

---

## Task 6: Add Candidate Diagnostics

**Files:**

- Modify: `tests/domain/test_eopkg_topology_conditioned.py`
- Modify: `src/domain/models/topology_conditioned.py`

- [ ] **Step 1: Write failing diagnostic test**

Assert after forward:

```text
last_candidate_node_score_mean_abs
last_candidate_class_score_mean_abs
last_duplicate_candidate_count_max
last_candidate_temperature
last_candidate_temperature_trainable
```

- [ ] **Step 2: Run RED**

Expected: diagnostics missing.

- [ ] **Step 3: Implement diagnostics**

Record detached tensors after scoring:

```text
mean abs node scores
mean abs finite class scores
max duplicate candidates per class
current clamped candidate temperature
temperature trainable flag
```

If `candidate_temperature_trainable=true`, trainer-level gradient diagnostics
should additionally log the temperature gradient norm when the generic gradient
diagnostic path is enabled. This is not required for canonical Stage 1 runs
because temperature is fixed by default.

- [ ] **Step 4: Run GREEN**

Expected: diagnostics test passes.

---

## Task 7: CLI Config Forwarding

**Files:**

- Modify: `tests/application/test_cli_flatten_config.py`
- Modify: `src/cli.py`

- [ ] **Step 1: Write failing config test**

Add test that `_build_model_factory_kwargs` forwards:

```text
observed_encoder
struct_encoder
candidate_scoring
candidate_pooling
candidate_temperature_init
candidate_temperature_min
candidate_temperature_max
candidate_temperature_trainable
```

- [ ] **Step 2: Run RED**

Expected: missing kwargs.

- [ ] **Step 3: Forward config keys**

Add keys in `_build_model_factory_kwargs`.

- [ ] **Step 4: Run GREEN**

Expected: config test passes.

---

## Task 8: Config Catalog And UI Surface

**Files:**

- Modify: `configs/ui/config_catalog.yaml`
- Modify: `tests/application/test_experiment_ui_catalog_fields.py`

- [ ] **Step 1: Write failing catalog test**

Assert catalog includes:

```text
model.observed_encoder
model.struct_encoder
model.candidate_scoring
model.candidate_pooling
model.candidate_temperature_init
model.candidate_temperature_trainable
```

- [ ] **Step 2: Run RED**

Expected: missing catalog fields.

- [ ] **Step 3: Add catalog entries**

Add under model/topology-conditioned group with:

```text
active_when: model.type == EOPKGTopologyConditioned
```

Use `level=experiment_run` for `candidate_scoring` and `candidate_pooling`;
use `level=advanced` for encoder and temperature fields.

- [ ] **Step 4: Run GREEN**

Expected: catalog test passes.

---

## Task 9: Runtime Docs

**Files:**

- Modify: `docs/GNN_RUNTIME_MVP2_5.MD`
- Modify: `docs/GNN_LEARNING_METODOLOGY.MD`
- Modify: `docs/current/project-state.md`
- Modify: `docs/current/architecture-debt.md`

- [ ] **Step 1: Update runtime docs**

Add `EOPKGTopologyConditioned` to model registry and describe:

```text
separate model family, not fusion_mode
Stage 1 returns [B, C_train]
internally computes [B, |V|] node candidate scores
MIL pooling via struct_node_to_class_index
```

- [ ] **Step 2: Update methodology doc**

Mark:

```text
Level 2 Stage 1: implemented as label-level MIL candidate scoring
Level 2 Stage 2: dynamic candidate contract planned
```

- [ ] **Step 3: Update architecture debt**

Add:

```text
dynamic_candidate_contract
current_behavior: Stage 1 still returns [B, C_train]
target_state: version-specific candidate outputs [B, C_v]
```

- [ ] **Step 4: Documentation self-check**

```powershell
rg -n "EOPKGTopologyConditioned|dynamic_candidate_contract|candidate_scoring" docs AGENTS.MD configs\ui\config_catalog.yaml
```

Expected: docs and catalog mention new model consistently.

---

## Task 10: Verification

**Files:** no code changes.

- [ ] **Step 1: Run targeted model tests**

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\domain\test_eopkg_topology_conditioned.py -v
```

- [ ] **Step 2: Run model regression tests**

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\domain\test_eopkg_models_forward.py tests\domain\test_eopkg_gatv2_dual_encoder.py tests\domain\test_eopkg_struct_xattn.py -v
```

- [ ] **Step 3: Run application config/UI tests**

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\application\test_cli_flatten_config.py tests\application\test_experiment_ui_catalog_fields.py -v
```

- [ ] **Step 4: Run architecture guard**

```powershell
.\.venv-modern\Scripts\python.exe tools\architecture_guard.py
```

Expected:

```text
[ARCH_GUARD] OK: architecture dependency rules satisfied.
```

- [ ] **Step 5: Run full test suite**

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\ -v
```

Expected: all pass. Existing external-library deprecation warnings are acceptable
if no new warnings are introduced.

---

## First Experiment After Implementation

Create one preset only after tests pass:

```yaml
preset_name: _TCG-UN
project: mvp2_6-test
experiment:
  structural_mode: true
  mask_guided_enabled: false
  statistic_enabled: false
  train_ratio: 0.25
  fraction: 0.3
  fraction_strategy: versioned
  split_strategy: versioned
  version_scope_policy: train_cut
model:
  type: EOPKGTopologyConditioned
  observed_encoder: GATv2
  struct_encoder: GATv2
  candidate_scoring: cosine
  candidate_pooling: logmeanexp
  pooling_strategy: last_node
training:
  learning_strategy: standard
```

Compare against:

```text
_Base-UN
_XAt-UNcnt
_CAS-UN
```

Only after standard learning works, test:

```yaml
training.learning_strategy: topology_conditioned
```

---

## Expected Research Interpretation

If Stage 1 improves known-label route-change drift but does not handle new v4
candidate failures, that is expected and useful. It proves candidate scoring is
better aligned with topology, while justifying Stage 2 dynamic candidate output.

If Stage 1 performs like baseline, inspect:

```text
last_candidate_node_score_mean_abs
last_candidate_class_score_mean_abs
last_duplicate_candidate_count_max
candidate_temperature
OOS / ECE / NLL
changed_transition_zone errors
```

If duplicate count is high and strict-F1 is unstable, prioritize node-identity
or MIL audit before adding semantic prototypes.

---

## Plan Self-Review

- No dynamic `[B, C_v]` trainer changes are included in Stage 1; this is intentional to avoid a high-risk rewrite.
- The new model is not a `fusion_mode`; docs and config must keep that distinction.
- MIL aggregation is required in Stage 1 because duplicate topology nodes already exist as known architecture debt.
- Multi-task duration/resource/risk heads are intentionally excluded until candidate scoring is stable.
- All behavior changes are covered by tests before implementation.
