# LSTM_Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `model.type=LSTM_Baseline` as a fixed-vocabulary sequence baseline for publication-grade comparison against `BaselineGATv2` and `EOPKGTopologyConditioned`.

**Architecture:** Implement one new domain model that consumes the existing `GraphTensorContract`, treats graph nodes as the ordered prefix sequence, and returns fixed-vocabulary logits `[B, C_train]`. The model must not use structure, masks, candidate ids, or topology-native scoring; it is an external PPM-style sequence baseline inside the same train/eval/drift pipeline.

**Tech Stack:** Python, PyTorch `nn.LSTM` / `nn.GRU`, existing `FeatureLayout`, existing `GraphTensorContract`, current `ModelTrainer`, `create_model()` registry, desktop UI config catalog.

---

## 1. Scope And Non-Goals

### In Scope

- Add `LSTM_Baseline` registered model type.
- Support both LSTM and GRU through one parameter:

```yaml
model.type: LSTM_Baseline
model.recurrent_type: lstm   # lstm | gru
```

- Keep prediction space fixed to `C_train`, like classical sequence PPM baselines.
- Reuse the existing graph-builder output; no new dataset format.
- Make train/eval/eval_drift work through the existing trainer.
- Add model-level, CLI-level, config-catalog, and minimal integration tests.
- Update GNN runtime docs and active UI presets after implementation.

### Out Of Scope

- Transformer baseline.
- Topology-native LSTM.
- Attention over candidate structures.
- New metrics.
- New loss functions.
- Any change to `EOPKGTopologyConditioned`.

### Research Interpretation

`LSTM_Baseline` is not meant to beat `EOPKGTopologyConditioned` under unseen candidate drift. It is a classical sequence/fixed-vocabulary baseline that answers reviewer criticism: the paper compares against a standard PPM-style neural model, not only against an internal logs-only GNN ablation.

For the article, the key comparison is not "LSTM is weak". The expected trade-off is:

1. On stable known-version segments (`v1` / known train versions), LSTM may match or exceed GNN strict-F1 because recurrent models are strong at learning linear prefix patterns.
2. Under structural drift, LSTM remains constrained by its fixed softmax head over `C_train`.
3. If future versions introduce new activities, removed activities, or changed routing, LSTM cannot produce logits for unseen candidate nodes; this should appear as higher OOS, worse future-version strict-F1, and poorer calibration compared with topology-native `C^(v)` scoring.

This is the intended evidence for the article: static-vocabulary sequence models can be competitive before drift, but they do not solve the cold-start candidate-space expansion problem.

---

## 2. File Map

### Create

- `src/domain/models/lstm_baseline.py`
  - New sequence model implementing `BaseGNN`.
  - Converts node-level graph tensors into padded per-graph sequences.
  - Uses dynamic categorical embeddings and numeric features like existing GNN baselines.
  - Returns fixed-label logits `[B, output_dim]`.

- `tests/domain/test_lstm_baseline.py`
  - Unit tests for LSTM and GRU forward pass, variable-length batching, registry, and unsupported config validation.

- `configs/model/lstm_baseline.yaml`
  - Minimal reusable model config.

### Modify

- `src/domain/models/factory.py`
  - Import and register `LSTM_Baseline`.

- `src/cli.py`
  - Forward recurrent model config keys through `_build_model_factory_kwargs()`.

- `configs/ui/config_catalog.yaml`
  - Add `LSTM_Baseline` to `model.type` enum.
  - Add `model.recurrent_type`, `model.recurrent_layers`, `model.recurrent_bidirectional`.

- `tests/contracts/test_model_registry_mvp1.py`
  - Include `LSTM_Baseline` in basic forward-contract regression.

- `tests/application/test_cli_flatten_config.py`
  - Add config-forwarding test for recurrent model kwargs.

- `tests/application/test_experiment_ui_catalog_fields.py`
  - Assert UI catalog contains recurrent fields and model enum.

- `docs/GNN_RUNTIME_MVP2_5.MD`
  - Document `LSTM_Baseline` as fixed-vocabulary sequence baseline.

- `docs/current/project-state.md`
  - Add `LSTM_Baseline` to implemented model families after implementation.

- `outputs/ui/experiment_ui_presets.json`
  - Add publication presets after implementation:
    - `_LSTM-UN-f1-5`
    - `_LSTM-UN-f1-5-drift`
  - Use the same dataset/split/fraction/seed family as current article runs.

---

## 3. Design Contract

### Input Contract

`LSTM_Baseline.forward(contract)` reads:

```python
x_cat = contract["x_cat"]        # [N, num_cat]
x_num = contract["x_num"]        # [N, num_num]
batch = contract["batch"]        # [N], graph id per node
```

It intentionally ignores:

```python
edge_index
allowed_target_mask
structural_edge_index
struct_x
candidate_ids
```

### Sequence Semantics

The existing graph builder emits prefix nodes in event order. PyG batching concatenates graphs while preserving per-graph node order. Therefore, the LSTM sequence for graph `g` is:

```python
seq_g = x[batch == g]
```

The plan must not sort by timestamps inside the model. Sorting belongs to graph construction.

### Output Contract

The model returns:

```python
logits: torch.Tensor  # shape [batch_size, output_dim]
```

This matches `BaselineGATv2` and the current trainer.

### Fair Baseline Defaults

Recommended config:

```yaml
model:
  type: LSTM_Baseline
  recurrent_type: lstm
  hidden_dim: 64
  recurrent_layers: 1
  recurrent_bidirectional: false
  dropout: 0.2
  pooling_strategy: last_node
```

`pooling_strategy` support:

- `last_node`: use final hidden state / last valid sequence step.
- `global_mean`: mean-pool recurrent outputs over valid steps.

For publication runs, use `last_node`.

Architecture description for the paper:

```text
Embedding layer -> 1-layer LSTM hidden size 64 -> Dense -> Softmax over C_train
```

The output dimensionality must equal the training activity vocabulary size:

```python
output_dim = len(activity_vocab)
```

This is the critical limitation: if training on known versions yields, for example, `|C_train| = 39`, the LSTM softmax head has exactly 39 outputs. If the drift dataset later reaches a larger current-version candidate space, for example `|C^(v5)| = 53`, the LSTM baseline cannot predict the new classes without changing the classifier head or retraining in an expanded vocabulary.

---

## 4. Task 1: Add Failing Domain Tests

**Files:**

- Create: `tests/domain/test_lstm_baseline.py`

- [ ] **Step 1: Create tests for LSTM/GRU forward behavior**

Add:

```python
from __future__ import annotations

import pytest
import torch

from src.domain.entities.feature_config import FeatureLayout
from src.domain.models.factory import create_model, get_registered_models


def _layout() -> FeatureLayout:
    return FeatureLayout(
        cat_features={"concept:name": 8, "org:resource": 4},
        cat_feature_names=["concept:name", "org:resource"],
        num_dim=3,
    )


def _contract() -> dict:
    return {
        "x_cat": torch.tensor(
            [
                [1, 2],
                [2, 1],
                [3, 1],
                [4, 0],
                [5, 2],
            ],
            dtype=torch.long,
        ),
        "x_num": torch.tensor(
            [
                [0.1, 0.2, 0.3],
                [0.2, 0.3, 0.4],
                [0.3, 0.4, 0.5],
                [1.0, 0.0, 0.1],
                [1.1, 0.1, 0.2],
            ],
            dtype=torch.float32,
        ),
        "edge_index": torch.tensor([[0, 1, 3], [1, 2, 4]], dtype=torch.long),
        "batch": torch.tensor([0, 0, 0, 1, 1], dtype=torch.long),
        "y": torch.tensor([1, 2], dtype=torch.long),
        "num_nodes": 5,
    }


@pytest.mark.parametrize("recurrent_type", ["lstm", "gru"])
def test_lstm_baseline_forward_returns_fixed_vocab_logits(recurrent_type: str):
    model = create_model(
        "LSTM_Baseline",
        feature_layout=_layout(),
        hidden_dim=16,
        output_dim=5,
        dropout=0.0,
        pooling_strategy="last_node",
        recurrent_type=recurrent_type,
        recurrent_layers=1,
        recurrent_bidirectional=False,
    )

    logits = model(_contract())

    assert tuple(logits.shape) == (2, 5)
    assert logits.dtype == torch.float32
    assert torch.isfinite(logits).all()


def test_lstm_baseline_supports_global_mean_pooling():
    model = create_model(
        "LSTM_Baseline",
        feature_layout=_layout(),
        hidden_dim=16,
        output_dim=5,
        dropout=0.0,
        pooling_strategy="global_mean",
        recurrent_type="lstm",
        recurrent_layers=1,
        recurrent_bidirectional=False,
    )

    logits = model(_contract())

    assert tuple(logits.shape) == (2, 5)


def test_lstm_baseline_is_registered():
    assert "LSTM_Baseline" in get_registered_models()


def test_lstm_baseline_rejects_unknown_recurrent_type():
    with pytest.raises(ValueError, match="Unsupported model.recurrent_type"):
        create_model(
            "LSTM_Baseline",
            feature_layout=_layout(),
            hidden_dim=16,
            output_dim=5,
            recurrent_type="rnn",
        )


def test_lstm_baseline_rejects_unknown_pooling_strategy():
    model = create_model(
        "LSTM_Baseline",
        feature_layout=_layout(),
        hidden_dim=16,
        output_dim=5,
        pooling_strategy="attention",
        recurrent_type="lstm",
    )

    with pytest.raises(ValueError, match="Unsupported pooling_strategy"):
        model(_contract())
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\domain\test_lstm_baseline.py -v
```

Expected:

```text
ValueError: Unsupported model.type 'LSTM_Baseline'
```

---

## 5. Task 2: Implement `LSTM_Baseline`

**Files:**

- Create: `src/domain/models/lstm_baseline.py`
- Modify: `src/domain/models/factory.py`

- [ ] **Step 1: Create model implementation**

Create `src/domain/models/lstm_baseline.py`:

```python
"""Sequence baseline model for fixed-vocabulary PPM comparison."""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from src.domain.entities.feature_config import FeatureLayout
from src.domain.entities.tensor_contract import GraphTensorContract
from src.domain.models.base_gnn import BaseGNN


class LSTMBaseline(BaseGNN):
    """LSTM/GRU fixed-vocabulary baseline over ordered prefix nodes."""

    def __init__(
        self,
        feature_layout: FeatureLayout,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.2,
        pooling_strategy: str = "last_node",
        recurrent_type: str = "lstm",
        recurrent_layers: int = 1,
        recurrent_bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.feature_layout = feature_layout
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.dropout_rate = float(dropout)
        self.pooling_strategy = str(pooling_strategy).strip().lower()
        self.recurrent_type = str(recurrent_type).strip().lower()
        self.recurrent_layers = int(recurrent_layers)
        self.recurrent_bidirectional = bool(recurrent_bidirectional)

        if self.recurrent_type not in {"lstm", "gru"}:
            raise ValueError("Unsupported model.recurrent_type. Available: ['lstm', 'gru']")
        if self.recurrent_layers < 1:
            raise ValueError("model.recurrent_layers must be >= 1.")

        self.embeddings = nn.ModuleDict()
        self.embedding_dims: Dict[str, int] = {}
        total_emb_dim = 0
        for name in feature_layout.cat_feature_names:
            vocab_size = int(feature_layout.cat_features[name])
            emb_dim = max(2, min(50, int(6 * (vocab_size ** 0.25))))
            self.embeddings[name] = nn.Embedding(vocab_size, emb_dim)
            self.embedding_dims[name] = emb_dim
            total_emb_dim += emb_dim

        input_dim = total_emb_dim + int(feature_layout.num_dim)
        recurrent_dropout = self.dropout_rate if self.recurrent_layers > 1 else 0.0
        recurrent_cls = nn.LSTM if self.recurrent_type == "lstm" else nn.GRU
        self.recurrent = recurrent_cls(
            input_size=input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.recurrent_layers,
            batch_first=True,
            dropout=recurrent_dropout,
            bidirectional=self.recurrent_bidirectional,
        )
        directions = 2 if self.recurrent_bidirectional else 1
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.classifier = nn.Linear(self.hidden_dim * directions, self.output_dim)

    def forward(self, contract: GraphTensorContract) -> torch.Tensor:
        x_cat = contract["x_cat"]
        x_num = contract["x_num"]
        batch = contract["batch"]

        emb_parts = []
        for col, name in enumerate(self.feature_layout.cat_feature_names):
            emb_parts.append(self.embeddings[name](x_cat[:, col]))
        if emb_parts:
            x = torch.cat([*emb_parts, x_num], dim=1)
        else:
            x = x_num

        if batch.numel() != x.size(0):
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        if x.size(0) == 0:
            return x.new_zeros((0, self.output_dim))

        sequences = self._split_by_batch(x=x, batch=batch)
        lengths = torch.tensor([seq.size(0) for seq in sequences], dtype=torch.long, device=x.device)
        padded = pad_sequence(sequences, batch_first=True)
        packed = pack_padded_sequence(
            padded,
            lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_out, _ = self.recurrent(packed)
        outputs, _ = pad_packed_sequence(packed_out, batch_first=True)

        pooled = self._pool_sequences(outputs=outputs, lengths=lengths)
        pooled = self.dropout(pooled)
        return self.classifier(pooled)

    @staticmethod
    def _split_by_batch(x: torch.Tensor, batch: torch.Tensor) -> list[torch.Tensor]:
        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
        return [x[batch == graph_idx] for graph_idx in range(num_graphs)]

    def _pool_sequences(self, outputs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        if self.pooling_strategy == "last_node":
            row_idx = torch.arange(outputs.size(0), device=outputs.device)
            col_idx = lengths.to(device=outputs.device) - 1
            return outputs[row_idx, col_idx]

        if self.pooling_strategy == "global_mean":
            mask = (
                torch.arange(outputs.size(1), device=outputs.device)
                .unsqueeze(0)
                .lt(lengths.to(device=outputs.device).unsqueeze(1))
            )
            masked = outputs * mask.unsqueeze(-1).to(dtype=outputs.dtype)
            return masked.sum(dim=1) / lengths.to(device=outputs.device, dtype=outputs.dtype).unsqueeze(1)

        raise ValueError(f"Unsupported pooling_strategy '{self.pooling_strategy}'.")
```

- [ ] **Step 2: Register model in factory**

Modify `src/domain/models/factory.py`:

```python
from src.domain.models.baseline_gat import BaselineGATv2  # noqa: E402
from src.domain.models.baseline_gcn import BaselineGCN  # noqa: E402
from src.domain.models.lstm_baseline import LSTMBaseline  # noqa: E402

register_model("BaselineGCN")(BaselineGCN)
register_model("BaselineGATv2")(BaselineGATv2)
register_model("LSTM_Baseline")(LSTMBaseline)
```

- [ ] **Step 3: Run domain tests**

Run:

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\domain\test_lstm_baseline.py -v
```

Expected:

```text
5 passed
```

---

## 6. Task 3: Extend Existing Model Registry Contract Tests

**Files:**

- Modify: `tests/contracts/test_model_registry_mvp1.py`

- [ ] **Step 1: Add LSTM baseline to contract test imports**

Change imports:

```python
from src.domain.models.baseline_gat import BaselineGATv2
from src.domain.models.baseline_gcn import BaselineGCN
from src.domain.models.lstm_baseline import LSTMBaseline
```

- [ ] **Step 2: Include class in parametrized test**

Change:

```python
@pytest.mark.parametrize("model_cls", [BaselineGCN, BaselineGATv2])
```

to:

```python
@pytest.mark.parametrize("model_cls", [BaselineGCN, BaselineGATv2, LSTMBaseline])
```

- [ ] **Step 3: Run registry contract test**

Run:

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\contracts\test_model_registry_mvp1.py -v
```

Expected:

```text
3 passed
```

---

## 7. Task 4: Wire CLI Model Config

**Files:**

- Modify: `src/cli.py`
- Modify: `tests/application/test_cli_flatten_config.py`

- [ ] **Step 1: Write failing CLI forwarding test**

Append to `tests/application/test_cli_flatten_config.py`:

```python
def test_build_model_factory_kwargs_passes_lstm_baseline_config():
    kwargs = _build_model_factory_kwargs(
        model_cfg={
            "type": "LSTM_Baseline",
            "hidden_dim": 32,
            "dropout": 0.15,
            "pooling_strategy": "last_node",
            "recurrent_type": "gru",
            "recurrent_layers": 2,
            "recurrent_bidirectional": True,
        },
        feature_layout=object(),
        output_dim=7,
    )

    assert kwargs["model_type"] == "LSTM_Baseline"
    assert kwargs["recurrent_type"] == "gru"
    assert kwargs["recurrent_layers"] == 2
    assert kwargs["recurrent_bidirectional"] is True
```

- [ ] **Step 2: Run test and verify it fails**

Run:

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\application\test_cli_flatten_config.py::test_build_model_factory_kwargs_passes_lstm_baseline_config -v
```

Expected:

```text
KeyError: 'recurrent_type'
```

- [ ] **Step 3: Forward recurrent config in `_build_model_factory_kwargs()`**

In `src/cli.py`, add these keys to the returned dict:

```python
"recurrent_type": str(model_cfg.get("recurrent_type", "lstm")).strip().lower(),
"recurrent_layers": int(model_cfg.get("recurrent_layers", 1)),
"recurrent_bidirectional": _as_bool(model_cfg.get("recurrent_bidirectional"), default=False),
```

- [ ] **Step 4: Run test**

Run:

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\application\test_cli_flatten_config.py::test_build_model_factory_kwargs_passes_lstm_baseline_config -v
```

Expected:

```text
1 passed
```

---

## 8. Task 5: Add Config Catalog Fields

**Files:**

- Modify: `configs/ui/config_catalog.yaml`
- Modify: `tests/application/test_experiment_ui_catalog_fields.py`

- [ ] **Step 1: Add failing catalog test**

Append to `tests/application/test_experiment_ui_catalog_fields.py`:

```python
def test_lstm_baseline_fields_are_exposed_in_catalog():
    catalog_path = Path("configs/ui/config_catalog.yaml")
    catalog = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))["fields"]

    assert "LSTM_Baseline" in catalog["model.type"]["enum"]
    for field_name in [
        "model.recurrent_type",
        "model.recurrent_layers",
        "model.recurrent_bidirectional",
    ]:
        assert field_name in catalog
        assert catalog[field_name]["required_when"]["model.type"] == "LSTM_Baseline"
```

- [ ] **Step 2: Run test and verify it fails**

Run:

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\application\test_experiment_ui_catalog_fields.py::test_lstm_baseline_fields_are_exposed_in_catalog -v
```

Expected:

```text
AssertionError: assert 'LSTM_Baseline' in ...
```

- [ ] **Step 3: Add model type enum**

In `configs/ui/config_catalog.yaml`, under `model.type.enum`, add:

```yaml
    - LSTM_Baseline
```

- [ ] **Step 4: Add recurrent fields**

Add fields in the `model` section:

```yaml
  model.recurrent_type:
    affects: Selects sequence baseline recurrent cell.
    default: lstm
    description: Recurrent cell for LSTM_Baseline fixed-vocabulary sequence baseline.
    enum:
    - lstm
    - gru
    label: 'Модель: recurrent type'
    path: model.recurrent_type
    required_in_modes: []
    required_when:
      model.type: LSTM_Baseline
    section: model
    ui:
      group: advanced
      order: 1110
      priority: 50
      tab: model
  model.recurrent_layers:
    affects: Depth of recurrent sequence baseline.
    default: 1
    description: Number of recurrent layers for LSTM_Baseline.
    enum: []
    label: 'Модель: recurrent layers'
    path: model.recurrent_layers
    required_in_modes: []
    required_when:
      model.type: LSTM_Baseline
    section: model
    ui:
      group: advanced
      order: 1120
      priority: 50
      tab: model
  model.recurrent_bidirectional:
    affects: Uses bidirectional recurrent encoder; not recommended for causal online inference.
    default: false
    description: Enables bidirectional LSTM/GRU for ablation only. Keep false for causal PPM comparison.
    enum:
    - 'false'
    - 'true'
    label: 'Модель: bidirectional recurrent'
    path: model.recurrent_bidirectional
    required_in_modes: []
    required_when:
      model.type: LSTM_Baseline
    section: model
    ui:
      group: advanced
      order: 1130
      priority: 50
      tab: model
```

- [ ] **Step 5: Run catalog test**

Run:

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\application\test_experiment_ui_catalog_fields.py::test_lstm_baseline_fields_are_exposed_in_catalog -v
```

Expected:

```text
1 passed
```

---

## 9. Task 6: Add Reusable Model Config

**Files:**

- Create: `configs/model/lstm_baseline.yaml`

- [ ] **Step 1: Create config**

Create:

```yaml
model:
  type: LSTM_Baseline
  model_label: LSTM
  hidden_dim: 64
  dropout: 0.2
  pooling_strategy: last_node
  recurrent_type: lstm
  recurrent_layers: 1
  recurrent_bidirectional: false
```

- [ ] **Step 2: Validate YAML loads through existing config loader**

Run:

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\application\test_cli_flatten_config.py::test_build_model_factory_kwargs_passes_lstm_baseline_config -v
```

Expected:

```text
1 passed
```

---

## 10. Task 7: Trainer Smoke Test

**Files:**

- Modify: `tests/integration/test_training_loops.py` or create `tests/integration/test_lstm_baseline_training_smoke.py`

- [ ] **Step 1: Add minimal training smoke test**

Create `tests/integration/test_lstm_baseline_training_smoke.py`:

```python
from __future__ import annotations

import torch

from src.domain.entities.feature_config import FeatureLayout
from src.domain.models.factory import create_model


def test_lstm_baseline_backward_pass_on_batch():
    layout = FeatureLayout(
        cat_features={"concept:name": 8, "org:resource": 4},
        cat_feature_names=["concept:name", "org:resource"],
        num_dim=3,
    )
    model = create_model(
        "LSTM_Baseline",
        feature_layout=layout,
        hidden_dim=16,
        output_dim=5,
        dropout=0.0,
        pooling_strategy="last_node",
        recurrent_type="lstm",
        recurrent_layers=1,
        recurrent_bidirectional=False,
    )
    contract = {
        "x_cat": torch.tensor([[1, 2], [2, 1], [3, 1], [4, 0]], dtype=torch.long),
        "x_num": torch.randn(4, 3),
        "edge_index": torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
        "batch": torch.tensor([0, 0, 0, 0], dtype=torch.long),
        "y": torch.tensor([2], dtype=torch.long),
    }

    logits = model(contract)
    loss = torch.nn.functional.cross_entropy(logits, contract["y"])
    loss.backward()

    assert torch.isfinite(loss)
    assert any(param.grad is not None for param in model.parameters())
```

- [ ] **Step 2: Run smoke test**

Run:

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\integration\test_lstm_baseline_training_smoke.py -v
```

Expected:

```text
1 passed
```

---

## 11. Task 8: Documentation Updates

**Files:**

- Modify: `docs/GNN_RUNTIME_MVP2_5.MD`
- Modify: `docs/current/project-state.md`
- Optional after presets: `Шаблон_статті_EOPKG_структурний_дрейф.MD`

- [ ] **Step 1: Update registered model list in GNN runtime**

In `docs/GNN_RUNTIME_MVP2_5.MD`, add `LSTM_Baseline` to the registered model list:

```markdown
6. `LSTM_Baseline`
```

- [ ] **Step 2: Add LSTM_Baseline runtime subsection**

Add a concise subsection near baseline models:

```markdown
### 6.x LSTM_Baseline

`LSTM_Baseline` is a fixed-vocabulary sequence baseline for publication
comparison with classical predictive process monitoring models. It consumes the
same `GraphTensorContract` as GNN baselines but ignores graph edges and
structural payloads. Prefix nodes are interpreted in their existing event order
inside each PyG graph batch.

Supported config:

```yaml
model.type: LSTM_Baseline
model.recurrent_type: lstm   # lstm | gru
model.recurrent_layers: 1
model.recurrent_bidirectional: false
model.pooling_strategy: last_node
```

This model predicts over the fixed training vocabulary `C_train`; it is not
topology-native and cannot produce logits for unseen future-version candidates.
```

- [ ] **Step 3: Update project-state model families**

In `docs/current/project-state.md`, add `LSTM_Baseline` under `gnn_runtime.model_families`.

- [ ] **Step 4: Update article draft only after first successful run**

Do not claim LSTM results before running experiments. Add only this text after the first successful train/drift run confirms the metric pattern:

```markdown
The external sequence baseline uses `LSTM_Baseline`, a fixed-vocabulary recurrent PPM model that consumes ordered prefixes and predicts through a softmax head over `C_train`. This baseline represents the classical sequence-learning formulation: it can be competitive on known-version segments, but its output layer is fixed to activities observed during training and therefore cannot directly produce logits for activities introduced in later process versions.
```

Add this interpretation to the results section only after runs exist:

```markdown
On the known-version segment, LSTM may show strong strict-F1 because recurrent models learn stable sequential patterns efficiently. After structural drift, the same fixed-vocabulary property becomes a limitation: future-version candidates outside `C_train` cannot be selected by the softmax head. Therefore, high OOS or sharp future-version F1 degradation for LSTM is interpreted as evidence for the need to move prediction from a static training vocabulary to topology-native `C^(v)` candidate ranking.
```

---

## 12. Task 9: UI Presets For Publication Runs

**Files:**

- Modify: `outputs/ui/experiment_ui_presets.json`

- [ ] **Step 1: Add train preset `_LSTM-UN-f1-5`**

Clone the current `_Base-UN-f1-drift-42` or article baseline train preset values, then set:

```json
{
  "vars": {
    "preset_name": "_LSTM-UN-f1-5",
    "experiment_name": "_LSTM-UN-f1-5-42",
    "mode": "train",
    "fraction": "1",
    "fraction_strategy": "versioned",
    "split_strategy": "versioned",
    "train_ratio": "0.38",
    "version_scope_policy": "train_cut",
    "gateway_mode": "collapse_for_prediction"
  },
  "model_form": {
    "model.type": "LSTM_Baseline",
    "model.model_label": "LSTM",
    "model.hidden_dim": "64",
    "model.dropout": "0.2",
    "model.pooling_strategy": "last_node",
    "model.recurrent_type": "lstm",
    "model.recurrent_layers": "1",
    "model.recurrent_bidirectional": "false"
  },
  "general_experiment_form": {
    "experiment.structural_mode": "false",
    "experiment.statistic_enabled": "false",
    "experiment.mask_guided_enabled": "false",
    "experiment.load_checkpoint": ""
  },
  "general_training_form": {
    "training.learning_strategy": "standard",
    "training.dynamic_candidate_contract_enabled": "false",
    "training.candidate_contract_mode": "fixed_label"
  }
}
```

Preserve all existing dataset, XES, tracking, and feature settings from the baseline preset.

- [ ] **Step 2: Add drift preset `_LSTM-UN-f1-5-drift`**

Clone `_LSTM-UN-f1-5`, then set:

```json
{
  "vars": {
    "preset_name": "_LSTM-UN-f1-5-drift",
    "experiment_name": "_LSTM-UN-f1-5-drift-last-42",
    "mode": "eval_drift",
    "train_ratio": "0"
  },
  "general_experiment_form": {
    "experiment.load_checkpoint": "checkpoints/_LSTM-UN-f1-5-42_loan_v1_v5_simulated_LSTM_best.pth"
  }
}
```

Adjust checkpoint path after the first train run if actual checkpoint naming differs.

- [ ] **Step 3: Validate JSON**

Run:

```powershell
.\.venv-modern\Scripts\python.exe -c "import json; json.load(open('outputs/ui/experiment_ui_presets.json', encoding='utf-8')); print('JSON OK')"
```

Expected:

```text
JSON OK
```

---

## 13. Task 10: Verification

- [ ] **Step 1: Focused tests**

Run:

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\domain\test_lstm_baseline.py tests\contracts\test_model_registry_mvp1.py tests\application\test_cli_flatten_config.py::test_build_model_factory_kwargs_passes_lstm_baseline_config tests\application\test_experiment_ui_catalog_fields.py::test_lstm_baseline_fields_are_exposed_in_catalog tests\integration\test_lstm_baseline_training_smoke.py -v
```

Expected:

```text
all selected tests passed
```

- [ ] **Step 2: Architecture guard**

Run:

```powershell
.\.venv-modern\Scripts\python.exe tools\architecture_guard.py
```

Expected:

```text
[ARCH_GUARD] OK
```

- [ ] **Step 3: MVP1 regression**

Run:

```powershell
.\.venv-modern\Scripts\python.exe -m pytest -m mvp1_regression -v
```

Expected:

```text
all selected tests passed
```

- [ ] **Step 4: Optional full suite**

Run only when time permits:

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\ -v
```

Expected:

```text
all tests passed
```

---

## 14. Manual Experiment Validation

After implementation:

1. Run `_LSTM-UN-f1-5` train.
2. Confirm MLflow params:

```text
model_type = LSTMBaseline
model.recurrent_type = lstm
model.pooling_strategy = last_node
```

3. Run `_LSTM-UN-f1-5-drift`.
4. Compare against:

```text
_Base-UN-f1-drift
_EOPKG_SC-f1-5
_EOPKG_SC-imp-f1-5
```

5. Interpret:

- If LSTM is strong on `v1/v2` but degrades on `v3..v5`, it supports the fixed-vocabulary drift argument.
- If LSTM beats `BaselineGATv2`, keep both: LSTM is external PPM baseline, GATv2 is architectural ablation.
- If LSTM beats EPOKG on `strict_macro_f1` but has higher OOS/NLL/ECE, the article should report that the topology-native method improves candidate-space validity/calibration, not universal exact next-event accuracy.
- If LSTM is strongest before drift and then collapses on new-version segments, this is not a problem for the hypothesis. It is the expected static-vocabulary trade-off and should be described explicitly.
- In drift figures, include LSTM as a third line. The most important panels for this comparison are `drift_window_strict_macro_f1`, `drift_window_macro_f1`, `drift_window_test_oos`, `drift_window_test_ece`, and `drift_window_test_set_nll`.
- Report current-version candidate growth beside the model results, for example `|C_train|` versus `|C^(v5)|`, so the reader can directly connect OOS behavior to vocabulary expansion.

---

## 15. Self-Review Checklist

- [ ] Plan covers model implementation, factory registration, CLI config, UI catalog, docs, presets, tests.
- [ ] New model does not import adapters or infrastructure.
- [ ] New model preserves `BaseGNN.forward(contract) -> logits [B, C_train]`.
- [ ] LSTM/GRU baseline ignores topology by design.
- [ ] No topology-native metrics are required for LSTM baseline.
- [ ] Publication interpretation is clear: external sequence baseline, not replacement for GNN ablation.
- [ ] No claims about LSTM results are added before actual runs.
