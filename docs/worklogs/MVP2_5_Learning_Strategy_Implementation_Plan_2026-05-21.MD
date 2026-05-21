# GNN Learning Strategy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `training.learning_strategy=standard | topology_conditioned` for existing GNN training, without implementing `experiment.mode=versioned_zero_shot` in this step.

**Architecture:** Keep `fusion_mode` as the model forward mechanism and add `learning_strategy` as trainer-level methodology. The implementation must not change model forward contracts; it adds config validation, topology-conditioned losses, version-weighted CE, version-safe negative topology sampling from prebuilt tensors, diagnostics, and config/UI/docs sync.

**Tech Stack:** Python 3.11 project venv, PyTorch, PyTorch Geometric `Data` / `DataLoader`, existing `ModelTrainer`, `configs/ui/config_catalog.yaml`, pytest.

---

## Scope

### In Scope

- Add `training.learning_strategy`.
- Preserve current behavior for `training.learning_strategy=standard`.
- Add fail-fast guard:

```text
learning_strategy=topology_conditioned requires experiment/model structural_mode=true
```

- Add topology-conditioned train-time losses:

```text
L_total =
  L_ce_correct
  + lambda_wrong * L_wrong_version_negative
  + lambda_drop * L_drop_edges_negative
  + lambda_allowed * L_allowed_set
  + lambda_retention * L_retention
```

- Add topology payload pool built from already prebuilt train graph tensors.
- Add `drop_edges` corruption only.
- Add wrong-version negative only from known versions available in current train data.
- Add train-time allowed-set loss, independent from `experiment.mask_guided_enabled`.
- Add version-weighted CE/rehearsal as first `L_retention` implementation.
- Add MLflow/log diagnostics for generic `topology_conditioned` metrics.
- Add `configs/ui/config_catalog.yaml` entries.
- Update docs after implementation.

### Out Of Scope

- Do not implement `experiment.mode=versioned_zero_shot`.
- Do not implement future-tail eval orchestration.
- Do not implement checkpoint smoothing/min-epoch policy yet.
- Do not implement `rewire_edges`.
- Do not implement distillation / teacher checkpoint retention.
- Do not change EOPKG model forward semantics.
- Do not make future topology available to train or negatives.

---

## File Structure

### Create

- `src/application/services/learning_strategy_config.py`
  - Parse and validate `training.learning_strategy`.
  - Define support levels for fusion modes.
  - Hold topology-conditioned coefficients.

- `src/application/services/topology_conditioned_learning.py`
  - Pure tensor/service functions for:
    - allowed-set loss;
    - version-weighted CE;
    - topology-conditioned margin loss;
    - stats accumulation helpers.

- `src/application/services/topology_payload_pool.py`
  - Build reusable structural payloads from prebuilt graph `Data`.
  - Select wrong-version payloads without repository/raw source calls.
  - Drop structural edges without mutating original contract.

- `tests/application/test_learning_strategy_config.py`
  - Unit tests for config parsing, support levels, and fail-fast rules.

- `tests/application/test_topology_payload_pool.py`
  - Unit tests for topology payload collection, wrong-version selection, and edge drop.

- `tests/application/test_topology_conditioned_learning.py`
  - Unit tests for allowed-set loss, version weights, and margin losses.

- `tests/application/test_trainer_topology_conditioned_learning.py`
  - Trainer integration tests using dummy models and small `Data` samples.

### Modify

- `src/application/use_cases/trainer.py`
  - Parse learning strategy config.
  - Build topology payload pool from train data.
  - Apply topology-conditioned losses inside `_run_epoch`.
  - Log generic learning strategy diagnostics.

- `src/cli.py`
  - Validate config before creating the trainer or during config flattening.
  - Ensure `training.learning_strategy` is preserved in MLflow params.

- `configs/ui/config_catalog.yaml`
  - Add all new `training.*` keys.

- `tests/application/test_cli_flatten_config.py`
  - Verify config forwarding / validation behavior.

- `tests/application/test_experiment_ui_catalog_fields.py`
  - Verify catalog includes new keys.

- `docs/GNN_LEARNING_STRATEGY.MD`
  - Update status from `design` to `implemented` after implementation.

- `docs/GNN_RUNTIME_MVP2_5.MD`
  - Add short cross-reference: forward unchanged, learning strategy lives in trainer.

- `docs/EVF_MVP2_5.MD`
  - Add evaluation interpretation for `standard` vs `topology_conditioned`.

- `docs/current/project-state.md`
  - Update current state after implementation.

---

## Config Keys

Add these keys under `training`:

```yaml
training.learning_strategy: standard

training.topology_conditioning_wrong_version_negative_enabled: true
training.topology_conditioning_wrong_version_negative_weight: 0.10
training.topology_conditioning_wrong_version_margin: 0.20

training.topology_conditioning_drop_edges_negative_enabled: true
training.topology_conditioning_drop_edges_negative_weight: 0.05
training.topology_conditioning_drop_edges_margin: 0.10
training.topology_conditioning_drop_edges_ratio: 0.15

training.topology_conditioning_allowed_set_loss_enabled: true
training.topology_conditioning_allowed_set_loss_weight: 0.05

training.topology_conditioning_retention_enabled: true
training.topology_conditioning_retention_policy: version_decay
training.topology_conditioning_current_version_policy: latest_in_train
training.topology_conditioning_retention_recent_versions: 1
training.topology_conditioning_retention_recent_weight: 0.50
training.topology_conditioning_retention_old_weight: 0.15
training.topology_conditioning_retention_obsolete_weight: 0.00
```

Do not add nested `topology_conditioning:` in the first implementation. The current trainer consumes flat `training` config values, so flat keys keep UI and CLI compatibility simple.

---

## Task 1: Learning Strategy Config Contract

**Files:**

- Create: `src/application/services/learning_strategy_config.py`
- Create: `tests/application/test_learning_strategy_config.py`
- Modify: `src/application/use_cases/trainer.py`

### Steps

- [ ] **Step 1: Write failing tests for parsing defaults**

Add to `tests/application/test_learning_strategy_config.py`:

```python
import pytest

from src.application.services.learning_strategy_config import (
    FusionSupportLevel,
    LearningStrategyConfig,
)


def test_learning_strategy_defaults_to_standard():
    cfg = LearningStrategyConfig.from_training_config({})

    assert cfg.learning_strategy == "standard"
    assert cfg.is_topology_conditioned is False
    assert cfg.allowed_set_loss_enabled is False
    assert cfg.wrong_version_negative_enabled is False
    assert cfg.drop_edges_negative_enabled is False
```

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\application\test_learning_strategy_config.py -v
```

Expected: fails because module does not exist.

- [ ] **Step 2: Implement minimal config dataclass**

Create `src/application/services/learning_strategy_config.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping


def _as_bool(raw: Any, *, default: bool = False) -> bool:
    if isinstance(raw, bool):
        return raw
    if raw is None:
        return bool(default)
    text = str(raw).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return bool(default)


class FusionSupportLevel(str, Enum):
    SUPPORTED = "supported"
    ALLOWED_BUT_WEAK = "allowed_but_weak"
    FORBIDDEN = "forbidden"


@dataclass(frozen=True)
class LearningStrategyConfig:
    learning_strategy: str = "standard"
    wrong_version_negative_enabled: bool = False
    wrong_version_negative_weight: float = 0.10
    wrong_version_margin: float = 0.20
    drop_edges_negative_enabled: bool = False
    drop_edges_negative_weight: float = 0.05
    drop_edges_margin: float = 0.10
    drop_edges_ratio: float = 0.15
    allowed_set_loss_enabled: bool = False
    allowed_set_loss_weight: float = 0.05
    retention_enabled: bool = False
    retention_recent_versions: int = 1
    retention_recent_weight: float = 0.50
    retention_old_weight: float = 0.15
    retention_obsolete_weight: float = 0.0

    @property
    def is_topology_conditioned(self) -> bool:
        return self.learning_strategy == "topology_conditioned"

    @classmethod
    def from_training_config(cls, config: Mapping[str, Any]) -> "LearningStrategyConfig":
        strategy = str(config.get("learning_strategy", "standard")).strip().lower()
        if strategy not in {"standard", "topology_conditioned"}:
            raise ValueError(
                "Unsupported training.learning_strategy "
                f"'{strategy}'. Expected 'standard' or 'topology_conditioned'."
            )
        conditioned = strategy == "topology_conditioned"
        return cls(
            learning_strategy=strategy,
            wrong_version_negative_enabled=_as_bool(
                config.get("topology_conditioning_wrong_version_negative_enabled"),
                default=conditioned,
            ),
            wrong_version_negative_weight=float(
                config.get("topology_conditioning_wrong_version_negative_weight", 0.10)
            ),
            wrong_version_margin=float(config.get("topology_conditioning_wrong_version_margin", 0.20)),
            drop_edges_negative_enabled=_as_bool(
                config.get("topology_conditioning_drop_edges_negative_enabled"),
                default=conditioned,
            ),
            drop_edges_negative_weight=float(
                config.get("topology_conditioning_drop_edges_negative_weight", 0.05)
            ),
            drop_edges_margin=float(config.get("topology_conditioning_drop_edges_margin", 0.10)),
            drop_edges_ratio=min(
                1.0,
                max(0.0, float(config.get("topology_conditioning_drop_edges_ratio", 0.15))),
            ),
            allowed_set_loss_enabled=_as_bool(
                config.get("topology_conditioning_allowed_set_loss_enabled"),
                default=conditioned,
            ),
            allowed_set_loss_weight=float(config.get("topology_conditioning_allowed_set_loss_weight", 0.05)),
            retention_enabled=_as_bool(
                config.get("topology_conditioning_retention_enabled"),
                default=conditioned,
            ),
            retention_recent_versions=max(
                0,
                int(config.get("topology_conditioning_retention_recent_versions", 1)),
            ),
            retention_recent_weight=float(config.get("topology_conditioning_retention_recent_weight", 0.50)),
            retention_old_weight=float(config.get("topology_conditioning_retention_old_weight", 0.15)),
            retention_obsolete_weight=float(
                config.get("topology_conditioning_retention_obsolete_weight", 0.0)
            ),
        )
```

- [ ] **Step 3: Run parsing test**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\application\test_learning_strategy_config.py -v
```

Expected: pass.

- [ ] **Step 4: Add support-level and fail-fast tests**

Append tests:

```python
def test_topology_conditioned_enables_default_components():
    cfg = LearningStrategyConfig.from_training_config({"learning_strategy": "topology_conditioned"})

    assert cfg.is_topology_conditioned is True
    assert cfg.allowed_set_loss_enabled is True
    assert cfg.wrong_version_negative_enabled is True
    assert cfg.drop_edges_negative_enabled is True
    assert cfg.retention_enabled is True


def test_unsupported_learning_strategy_fails_fast():
    with pytest.raises(ValueError, match="Unsupported training.learning_strategy"):
        LearningStrategyConfig.from_training_config({"learning_strategy": "unknown"})
```

Add to `learning_strategy_config.py`:

```python
SUPPORTED_FUSION_MODES = {
    "struct_xattn",
    "class_aware_structural_scoring",
    "topology_state_encoder",
    "topology_state_graph_encoder",
}

ALLOWED_BUT_WEAK_FUSION_MODES = {
    "class_mean_attention",
    "class_mean_concat",
    "structural_prior_encoder",
}


def resolve_fusion_support_level(*, structural_mode: bool, fusion_mode: str) -> FusionSupportLevel:
    if not structural_mode:
        return FusionSupportLevel.FORBIDDEN
    normalized = str(fusion_mode or "").strip().lower()
    if normalized in SUPPORTED_FUSION_MODES:
        return FusionSupportLevel.SUPPORTED
    if normalized in ALLOWED_BUT_WEAK_FUSION_MODES:
        return FusionSupportLevel.ALLOWED_BUT_WEAK
    return FusionSupportLevel.ALLOWED_BUT_WEAK
```

Add tests:

```python
from src.application.services.learning_strategy_config import resolve_fusion_support_level


def test_resolve_fusion_support_level_for_supported_mode():
    assert (
        resolve_fusion_support_level(structural_mode=True, fusion_mode="struct_xattn")
        == FusionSupportLevel.SUPPORTED
    )


def test_resolve_fusion_support_level_for_weak_mode():
    assert (
        resolve_fusion_support_level(structural_mode=True, fusion_mode="class_mean_concat")
        == FusionSupportLevel.ALLOWED_BUT_WEAK
    )


def test_resolve_fusion_support_level_for_structural_mode_false():
    assert (
        resolve_fusion_support_level(structural_mode=False, fusion_mode="struct_xattn")
        == FusionSupportLevel.FORBIDDEN
    )
```

- [ ] **Step 5: Wire config into trainer constructor**

Modify `src/application/use_cases/trainer.py` imports:

```python
from src.application.services.learning_strategy_config import (
    FusionSupportLevel,
    LearningStrategyConfig,
    resolve_fusion_support_level,
)
```

In `ModelTrainer.__init__`, after existing training config parsing:

```python
self.learning_strategy_config = LearningStrategyConfig.from_training_config(config)
self.learning_strategy = self.learning_strategy_config.learning_strategy
self.learning_strategy_support_level = resolve_fusion_support_level(
    structural_mode=_as_bool(getattr(self.model, "structural_mode", False), default=False),
    fusion_mode=str(getattr(self.model, "fusion_mode", "")),
)
if (
    self.learning_strategy_config.is_topology_conditioned
    and self.learning_strategy_support_level == FusionSupportLevel.FORBIDDEN
):
    raise ValueError(
        "training.learning_strategy=topology_conditioned requires "
        "experiment.structural_mode=true / model.structural_mode=true."
    )
if (
    self.learning_strategy_config.is_topology_conditioned
    and self.struct_xattn_contrastive_enabled
):
    raise ValueError(
        "training.struct_xattn_contrastive_enabled cannot be combined with "
        "training.learning_strategy=topology_conditioned. Use the generic "
        "topology_conditioning_* loss keys instead."
    )
if self.learning_strategy_config.is_topology_conditioned and self.learning_strategy_support_level == FusionSupportLevel.ALLOWED_BUT_WEAK:
    logger.warning(
        "training.learning_strategy=topology_conditioned is enabled for fusion_mode=%s, "
        "which has learning_strategy_support_level=allowed_but_weak.",
        getattr(self.model, "fusion_mode", ""),
    )
if self.tracker is not None:
    self.tracker.log_param("training.learning_strategy", self.learning_strategy)
    self.tracker.log_param("training.learning_strategy_support_level", self.learning_strategy_support_level.value)
```

- [ ] **Step 6: Add trainer validation tests**

Add to `tests/application/test_learning_strategy_config.py` or a trainer-specific test file:

```python
import torch
from torch import nn

from src.application.use_cases.trainer import ModelTrainer


class _DummyAdapter:
    def read(self, file_path, mapping_config):
        return iter([])


class _DummyPrefixPolicy:
    def generate_slices(self, trace):
        return []


class _DummyGraphBuilder:
    def build_graph(self, prefix):
        raise RuntimeError("not used")


class _DummyModel(nn.Module):
    def __init__(self, *, structural_mode: bool, fusion_mode: str):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.structural_mode = structural_mode
        self.fusion_mode = fusion_mode

    def forward(self, contract):
        return torch.zeros((1, 2), dtype=torch.float32)


def test_topology_conditioned_requires_structural_mode_true():
    with pytest.raises(ValueError, match="requires .*structural_mode=true"):
        ModelTrainer(
            xes_adapter=_DummyAdapter(),
            prefix_policy=_DummyPrefixPolicy(),
            graph_builder=_DummyGraphBuilder(),
            model=_DummyModel(structural_mode=False, fusion_mode="struct_xattn"),
            log_path="in_memory.xes",
            config={
                "device": "cpu",
                "show_progress": False,
                "tqdm_disable": True,
                "learning_strategy": "topology_conditioned",
            },
        )
```

- [ ] **Step 7: Run focused tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\application\test_learning_strategy_config.py -v
```

Expected: pass.

---

## Task 2: Topology Payload Pool

**Files:**

- Create: `src/application/services/topology_payload_pool.py`
- Create: `tests/application/test_topology_payload_pool.py`

### Steps

- [ ] **Step 1: Write failing tests for collecting payloads**

Create `tests/application/test_topology_payload_pool.py`:

```python
import torch
from torch_geometric.data import Data

from src.application.services.topology_payload_pool import TopologyPayloadPool


def _sample(version_idx: int, edges: torch.Tensor) -> Data:
    return Data(
        x_cat=torch.zeros((1, 0), dtype=torch.long),
        x_num=torch.ones((1, 1), dtype=torch.float32),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
        edge_type=torch.zeros((0,), dtype=torch.long),
        y=torch.tensor([1], dtype=torch.long),
        batch=torch.tensor([0], dtype=torch.long),
        process_version_idx=torch.tensor([version_idx], dtype=torch.long),
        struct_x=torch.ones((3, 2), dtype=torch.float32) * float(version_idx + 1),
        structural_edge_index=edges,
        structural_edge_weight=torch.ones(edges.size(1), dtype=torch.float32),
        struct_node_to_class_index=torch.tensor([0, 1, 2], dtype=torch.long),
    )


def test_payload_pool_collects_one_payload_per_version():
    pool = TopologyPayloadPool.from_dataset(
        [
            _sample(0, torch.tensor([[0, 1], [1, 2]], dtype=torch.long)),
            _sample(1, torch.tensor([[0, 2], [2, 1]], dtype=torch.long)),
        ],
        idx_to_version={0: "v1", 1: "v2"},
    )

    assert pool.known_versions == ("v1", "v2")
    assert pool.payload_for_version("v1").struct_x.shape == torch.Size([3, 2])
    assert pool.payload_for_version("v2").structural_edge_index.shape == torch.Size([2, 2])
```

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\application\test_topology_payload_pool.py -v
```

Expected: fails because module does not exist.

- [ ] **Step 2: Implement payload pool dataclasses**

Create `src/application/services/topology_payload_pool.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import torch

from src.domain.entities.tensor_contract import GraphTensorContract


@dataclass(frozen=True)
class TopologyPayload:
    version_label: str
    struct_x: torch.Tensor | None
    structural_edge_index: torch.Tensor
    structural_edge_weight: torch.Tensor | None
    struct_node_to_class_index: torch.Tensor | None


@dataclass(frozen=True)
class TopologyPayloadPool:
    payloads: Mapping[str, TopologyPayload]

    @property
    def known_versions(self) -> tuple[str, ...]:
        return tuple(sorted(self.payloads.keys()))

    def payload_for_version(self, version_label: str) -> TopologyPayload:
        return self.payloads[version_label]

    @classmethod
    def from_dataset(
        cls,
        dataset: Iterable[object],
        *,
        idx_to_version: Mapping[int, str],
    ) -> "TopologyPayloadPool":
        payloads: dict[str, TopologyPayload] = {}
        for data in dataset:
            version_idx = _extract_version_idx(data)
            if version_idx is None:
                continue
            version_label = str(idx_to_version.get(version_idx, f"v{version_idx}"))
            if version_label in payloads:
                continue
            edge_index = getattr(data, "structural_edge_index", None)
            if not isinstance(edge_index, torch.Tensor) or edge_index.dim() != 2:
                continue
            payloads[version_label] = TopologyPayload(
                version_label=version_label,
                struct_x=_clone_optional_tensor(getattr(data, "struct_x", None)),
                structural_edge_index=edge_index.detach().clone(),
                structural_edge_weight=_clone_optional_tensor(getattr(data, "structural_edge_weight", None)),
                struct_node_to_class_index=_clone_optional_tensor(
                    getattr(data, "struct_node_to_class_index", None)
                ),
            )
        return cls(payloads=payloads)

    def wrong_version_payload(self, current_version: str) -> TopologyPayload | None:
        for version_label in self.known_versions:
            if version_label != current_version:
                return self.payload_for_version(version_label)
        return None


def _extract_version_idx(data: object) -> int | None:
    raw = getattr(data, "process_version_idx", None)
    if isinstance(raw, torch.Tensor) and raw.numel() > 0:
        return int(raw.view(-1)[0].detach().cpu().item())
    return None


def _clone_optional_tensor(value: object) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    return None


def replace_structural_payload(
    contract: GraphTensorContract,
    payload: TopologyPayload,
) -> GraphTensorContract:
    replaced: GraphTensorContract = dict(contract)  # type: ignore[assignment]
    if payload.struct_x is not None:
        replaced["struct_x"] = payload.struct_x.to(device=_contract_device(contract)).clone()
    replaced["structural_edge_index"] = payload.structural_edge_index.to(device=_contract_device(contract)).clone()
    if payload.structural_edge_weight is not None:
        replaced["structural_edge_weight"] = payload.structural_edge_weight.to(device=_contract_device(contract)).clone()
    if payload.struct_node_to_class_index is not None:
        replaced["struct_node_to_class_index"] = payload.struct_node_to_class_index.to(device=_contract_device(contract)).clone()
    return replaced


def _contract_device(contract: GraphTensorContract) -> torch.device:
    for key in ("x_num", "x_cat", "y", "batch"):
        value = contract.get(key)
        if isinstance(value, torch.Tensor):
            return value.device
    return torch.device("cpu")
```

- [ ] **Step 3: Add edge-drop tests and implementation**

Append to test file:

```python
from src.application.services.topology_payload_pool import drop_structural_edges


def test_drop_structural_edges_does_not_mutate_original_contract():
    data = _sample(0, torch.tensor([[0, 1, 1, 2], [1, 2, 0, 0]], dtype=torch.long))
    contract = {
        "x_num": data.x_num,
        "y": data.y,
        "batch": data.batch,
        "structural_edge_index": data.structural_edge_index,
        "structural_edge_weight": data.structural_edge_weight,
    }
    original = contract["structural_edge_index"].clone()

    dropped = drop_structural_edges(contract, drop_ratio=1.0, seed=123)

    assert torch.equal(contract["structural_edge_index"], original)
    assert dropped["structural_edge_index"].size(1) < original.size(1)
    assert dropped["structural_edge_index"].size(1) >= 1
    assert dropped["structural_edge_weight"].numel() == dropped["structural_edge_index"].size(1)
```

Add to `topology_payload_pool.py`:

```python
def drop_structural_edges(
    contract: GraphTensorContract,
    *,
    drop_ratio: float,
    seed: int,
) -> GraphTensorContract:
    corrupted: GraphTensorContract = dict(contract)  # type: ignore[assignment]
    edge_index = corrupted.get("structural_edge_index")
    if not isinstance(edge_index, torch.Tensor) or edge_index.dim() != 2 or int(edge_index.size(1)) <= 1:
        return corrupted
    edge_count = int(edge_index.size(1))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    keep_prob = 1.0 - min(1.0, max(0.0, float(drop_ratio)))
    keep_mask = torch.rand(edge_count, generator=generator, device="cpu") < keep_prob
    if not torch.any(keep_mask):
        keep_mask[0] = True
    keep_mask = keep_mask.to(device=edge_index.device, dtype=torch.bool)
    corrupted["structural_edge_index"] = edge_index[:, keep_mask].clone()
    edge_weight = corrupted.get("structural_edge_weight")
    if isinstance(edge_weight, torch.Tensor) and int(edge_weight.numel()) == edge_count:
        corrupted["structural_edge_weight"] = edge_weight[keep_mask].clone()
    return corrupted
```

- [ ] **Step 4: Run payload pool tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\application\test_topology_payload_pool.py -v
```

Expected: pass.

---

## Task 3: Topology-Conditioned Loss Helpers

**Files:**

- Create: `src/application/services/topology_conditioned_learning.py`
- Create: `tests/application/test_topology_conditioned_learning.py`

### Steps

- [ ] **Step 1: Write failing tests for allowed-set loss**

Create `tests/application/test_topology_conditioned_learning.py`:

```python
import torch

from src.application.services.topology_conditioned_learning import (
    allowed_set_loss,
    margin_negative_loss,
    version_weights,
)


def test_allowed_set_loss_rewards_probability_inside_allowed_mask():
    logits = torch.tensor([[0.0, 8.0, 7.0]], dtype=torch.float32)
    targets = torch.tensor([2], dtype=torch.long)
    allowed = torch.tensor([[False, True, True]], dtype=torch.bool)

    loss = allowed_set_loss(logits=logits, targets=targets, allowed_target_mask=allowed)

    assert float(loss.item()) < 0.01


def test_allowed_set_loss_fails_fast_without_mask():
    logits = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
    targets = torch.tensor([1], dtype=torch.long)

    try:
        allowed_set_loss(logits=logits, targets=targets, allowed_target_mask=None)
    except ValueError as exc:
        assert "allowed_target_mask" in str(exc)
    else:
        raise AssertionError("expected ValueError")
```

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\application\test_topology_conditioned_learning.py -v
```

Expected: fails because module does not exist.

- [ ] **Step 2: Implement allowed-set and margin losses**

Create `src/application/services/topology_conditioned_learning.py`:

```python
from __future__ import annotations

from typing import Mapping

import torch
import torch.nn.functional as F


def allowed_set_loss(
    *,
    logits: torch.Tensor,
    targets: torch.Tensor,
    allowed_target_mask: torch.Tensor | None,
) -> torch.Tensor:
    if not isinstance(allowed_target_mask, torch.Tensor):
        raise ValueError("topology_conditioned allowed-set loss requires allowed_target_mask.")
    if allowed_target_mask.dim() != 2 or allowed_target_mask.size(0) != logits.size(0):
        raise ValueError("allowed_target_mask must have shape [batch, num_classes].")
    valid_mask = allowed_target_mask.to(device=logits.device, dtype=torch.bool).clone()
    valid_mask.scatter_(1, targets.view(-1, 1).to(device=logits.device), True)
    log_probs = F.log_softmax(logits, dim=1)
    masked_log_probs = log_probs.masked_fill(~valid_mask, torch.finfo(log_probs.dtype).min)
    return -torch.logsumexp(masked_log_probs, dim=1).mean()


def margin_negative_loss(
    *,
    correct_ce: torch.Tensor,
    negative_ce: torch.Tensor,
    margin: float,
    max_loss: float | None = None,
) -> torch.Tensor:
    loss = F.relu(float(margin) + correct_ce - negative_ce)
    if max_loss is not None and float(max_loss) > 0.0:
        loss = torch.clamp(loss, max=float(max_loss))
    return loss
```

- [ ] **Step 3: Add tests and implementation for version weights**

Append tests:

```python
def test_margin_negative_loss_is_positive_when_negative_is_too_good():
    loss = margin_negative_loss(
        correct_ce=torch.tensor(0.7),
        negative_ce=torch.tensor(0.6),
        margin=0.2,
    )

    assert float(loss.item()) > 0.0


def test_version_weights_apply_decay_from_latest_version():
    weights = version_weights(
        version_indices=torch.tensor([0, 1, 2, 2], dtype=torch.long),
        latest_version_idx=2,
        recent_versions=1,
        recent_weight=0.5,
        old_weight=0.15,
        obsolete_weight=0.0,
    )

    assert weights.tolist() == [0.15, 0.5, 1.0, 1.0]
```

Append implementation:

```python
def version_weights(
    *,
    version_indices: torch.Tensor,
    latest_version_idx: int,
    recent_versions: int,
    recent_weight: float,
    old_weight: float,
    obsolete_weight: float,
) -> torch.Tensor:
    idx = version_indices.view(-1).long()
    latest = int(latest_version_idx)
    distance = latest - idx
    weights = torch.full(idx.shape, float(old_weight), dtype=torch.float32, device=idx.device)
    weights = torch.where(distance <= 0, torch.ones_like(weights), weights)
    if int(recent_versions) > 0:
        recent_mask = (distance > 0) & (distance <= int(recent_versions))
        weights = torch.where(recent_mask, torch.full_like(weights, float(recent_weight)), weights)
    obsolete_mask = distance > max(1, int(recent_versions) + 1)
    weights = torch.where(obsolete_mask, torch.full_like(weights, float(obsolete_weight)), weights)
    return weights


def weighted_cross_entropy(
    *,
    logits: torch.Tensor,
    targets: torch.Tensor,
    sample_weights: torch.Tensor | None,
    class_weights: torch.Tensor | None,
) -> torch.Tensor:
    per_sample = F.cross_entropy(
        logits,
        targets,
        weight=class_weights.to(device=logits.device) if isinstance(class_weights, torch.Tensor) else None,
        reduction="none",
    )
    if not isinstance(sample_weights, torch.Tensor):
        return per_sample.mean()
    weights = sample_weights.to(device=logits.device, dtype=per_sample.dtype).view(-1)
    denominator = torch.clamp(weights.sum(), min=torch.finfo(per_sample.dtype).eps)
    return (per_sample * weights).sum() / denominator
```

- [ ] **Step 4: Run loss helper tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\application\test_topology_conditioned_learning.py -v
```

Expected: pass.

---

## Task 4: Trainer Integration For `topology_conditioned`

**Files:**

- Modify: `src/application/use_cases/trainer.py`
- Create: `tests/application/test_trainer_topology_conditioned_learning.py`

### Steps

- [ ] **Step 1: Write failing trainer integration test for allowed-set loss**

Create `tests/application/test_trainer_topology_conditioned_learning.py`:

```python
from __future__ import annotations

from typing import Iterator

import torch
from torch import nn
from torch.optim import Adam
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from src.application.use_cases.trainer import ModelTrainer
from src.domain.entities.raw_trace import RawTrace


class _DummyAdapter:
    def read(self, file_path: str, mapping_config: dict) -> Iterator[RawTrace]:
        return iter([])


class _DummyPrefixPolicy:
    def generate_slices(self, trace: RawTrace):
        return []


class _DummyGraphBuilder:
    def build_graph(self, prefix):
        raise RuntimeError("not used")


class _RecordingTracker:
    def __init__(self) -> None:
        self.metrics: list[tuple[str, float, int | None]] = []
        self.params: dict[str, object] = {}

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        self.metrics.append((key, float(value), step))

    def log_param(self, key: str, value: object) -> None:
        self.params[key] = value


class _TopologyConditionedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.structural_mode = True
        self.fusion_mode = "class_aware_structural_scoring"
        self.calls = 0

    def forward(self, contract):
        self.calls += 1
        batch = contract["batch"]
        num_graphs = int(batch.max().item()) + 1 if batch.numel() else 1
        edge_index = contract.get("structural_edge_index")
        edge_count = int(edge_index.size(1)) if isinstance(edge_index, torch.Tensor) else 0
        if edge_count >= 3:
            logits = torch.tensor([[-2.0, 4.0, 3.0]], dtype=torch.float32, device=batch.device)
        else:
            logits = torch.tensor([[4.0, -2.0, 1.0]], dtype=torch.float32, device=batch.device)
        return logits.repeat(num_graphs, 1) + (self.weight * 0.0)


def _sample(version_idx: int = 0) -> Data:
    return Data(
        x_cat=torch.zeros((1, 0), dtype=torch.long),
        x_num=torch.ones((1, 1), dtype=torch.float32),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
        edge_type=torch.zeros((0,), dtype=torch.long),
        y=torch.tensor([2], dtype=torch.long),
        batch=torch.tensor([0], dtype=torch.long),
        process_version_idx=torch.tensor([version_idx], dtype=torch.long),
        struct_x=torch.ones((3, 2), dtype=torch.float32),
        structural_edge_index=torch.tensor([[0, 1, 1], [1, 2, 0]], dtype=torch.long),
        structural_edge_weight=torch.ones(3, dtype=torch.float32),
        struct_node_to_class_index=torch.tensor([0, 1, 2], dtype=torch.long),
        allowed_target_mask=torch.tensor([[False, True, True]], dtype=torch.bool),
    )


def _trainer(tracker: _RecordingTracker | None = None) -> ModelTrainer:
    trainer = ModelTrainer(
        xes_adapter=_DummyAdapter(),
        prefix_policy=_DummyPrefixPolicy(),
        graph_builder=_DummyGraphBuilder(),
        model=_TopologyConditionedModel(),
        log_path="in_memory.xes",
        config={
            "mode": "train",
            "device": "cpu",
            "show_progress": False,
            "tqdm_disable": True,
            "learning_strategy": "topology_conditioned",
            "topology_conditioning_allowed_set_loss_enabled": True,
            "topology_conditioning_allowed_set_loss_weight": 0.05,
            "topology_conditioning_drop_edges_negative_enabled": False,
            "topology_conditioning_wrong_version_negative_enabled": False,
            "topology_conditioning_retention_enabled": False,
        },
        tracker=tracker,
        prepared_data={"idx_to_version": {0: "v1"}},
    )
    trainer.criterion = nn.CrossEntropyLoss()
    return trainer


def test_topology_conditioned_allowed_set_loss_logs_metrics():
    tracker = _RecordingTracker()
    trainer = _trainer(tracker=tracker)
    loader = DataLoader([_sample()], batch_size=1, shuffle=False)

    trainer._run_epoch(loader, optimizer=None, training=True, epoch_index=1)

    metric_names = {name for name, _, _ in tracker.metrics}
    assert "train_topology_conditioned_allowed_set_loss" in metric_names
    assert tracker.params["training.learning_strategy"] == "topology_conditioned"
```

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\application\test_trainer_topology_conditioned_learning.py -v
```

Expected: fails because trainer has no generic metrics yet.

- [ ] **Step 2: Add imports and trainer state**

Modify `src/application/use_cases/trainer.py` imports:

```python
from src.application.services.topology_conditioned_learning import (
    allowed_set_loss,
    margin_negative_loss,
    version_weights,
    weighted_cross_entropy,
)
from src.application.services.topology_payload_pool import (
    TopologyPayloadPool,
    drop_structural_edges,
    replace_structural_payload,
)
```

In `__init__`, add:

```python
self._topology_payload_pool: TopologyPayloadPool | None = None
self._latest_train_version_idx: int | None = None
```

- [ ] **Step 3: Add topology-conditioned stats keys**

Modify `_new_forward_stats_accumulator()`:

```python
"topology_conditioned_allowed_set_loss_sum": 0.0,
"topology_conditioned_wrong_version_ce_sum": 0.0,
"topology_conditioned_drop_edges_ce_sum": 0.0,
"topology_conditioned_correct_ce_sum": 0.0,
"topology_conditioned_wrong_version_delta_sum": 0.0,
"topology_conditioned_drop_edges_delta_sum": 0.0,
"topology_conditioned_wrong_version_loss_sum": 0.0,
"topology_conditioned_drop_edges_loss_sum": 0.0,
"topology_conditioned_retention_weight_sum": 0.0,
"topology_conditioned_retention_weight_count": 0,
"topology_conditioned_batches": 0,
"topology_conditioned_wrong_version_batches": 0,
"topology_conditioned_drop_edges_batches": 0,
"topology_conditioned_allowed_set_batches": 0,
"topology_conditioned_mixed_version_batches": 0,
```

- [ ] **Step 4: Implement helper methods in trainer**

Add methods near the existing `_compute_structural_set_loss`:

```python
def _is_topology_conditioned_active(self, *, training: bool) -> bool:
    return bool(training and self.learning_strategy_config.is_topology_conditioned)


def _batch_version_indices(self, data: Data) -> torch.Tensor | None:
    if not hasattr(data, "process_version_idx"):
        return None
    value = data.process_version_idx
    if isinstance(value, torch.Tensor) and value.numel() > 0:
        return value.view(-1).long().to(device=self.device)
    return None


def _batch_primary_version_label(self, data: Data) -> str | None:
    version_indices = self._batch_version_indices(data)
    if not isinstance(version_indices, torch.Tensor) or version_indices.numel() == 0:
        return None
    primary_idx = int(version_indices.view(-1)[0].detach().cpu().item())
    return self._idx_to_version.get(primary_idx, f"v{primary_idx}")


def _batch_has_mixed_versions(self, data: Data) -> bool:
    version_indices = self._batch_version_indices(data)
    if not isinstance(version_indices, torch.Tensor) or version_indices.numel() <= 1:
        return False
    return int(torch.unique(version_indices.detach()).numel()) > 1
```

- [ ] **Step 5: Apply allowed-set and retention losses**

Inside `_run_epoch`, after `targets` and `allowed_mask` are available and before `loss` is finalized, replace:

```python
loss = self.criterion(effective_logits, targets)
base_ce_loss = loss
```

with:

```python
sample_weights = None
if self._is_topology_conditioned_active(training=training) and self.learning_strategy_config.retention_enabled:
    version_indices = self._batch_version_indices(data)
    if isinstance(version_indices, torch.Tensor) and version_indices.numel() == int(targets.numel()):
        latest_idx = self._latest_train_version_idx
        if latest_idx is None:
            latest_idx = int(version_indices.max().detach().cpu().item())
        sample_weights = version_weights(
            version_indices=version_indices,
            latest_version_idx=int(latest_idx),
            recent_versions=self.learning_strategy_config.retention_recent_versions,
            recent_weight=self.learning_strategy_config.retention_recent_weight,
            old_weight=self.learning_strategy_config.retention_old_weight,
            obsolete_weight=self.learning_strategy_config.retention_obsolete_weight,
        )
        forward_stats["topology_conditioned_retention_weight_sum"] = float(
            forward_stats.get("topology_conditioned_retention_weight_sum", 0.0)
        ) + float(sample_weights.detach().sum().cpu().item())
        forward_stats["topology_conditioned_retention_weight_count"] = int(
            forward_stats.get("topology_conditioned_retention_weight_count", 0)
        ) + int(sample_weights.numel())

if self._is_topology_conditioned_active(training=training):
    loss = weighted_cross_entropy(
        logits=effective_logits,
        targets=targets,
        sample_weights=sample_weights,
        class_weights=self.class_weights,
    )
else:
    loss = self.criterion(effective_logits, targets)
base_ce_loss = loss

if (
    self._is_topology_conditioned_active(training=training)
    and self.learning_strategy_config.allowed_set_loss_enabled
):
    topology_allowed_loss = allowed_set_loss(
        logits=logits,
        targets=targets,
        allowed_target_mask=allowed_mask,
    )
    loss = loss + (
        float(self.learning_strategy_config.allowed_set_loss_weight)
        * topology_allowed_loss
    )
    forward_stats["topology_conditioned_allowed_set_loss_sum"] = float(
        forward_stats.get("topology_conditioned_allowed_set_loss_sum", 0.0)
    ) + float(topology_allowed_loss.detach().cpu().item())
    forward_stats["topology_conditioned_allowed_set_batches"] = int(
        forward_stats.get("topology_conditioned_allowed_set_batches", 0)
    ) + 1
```

Important: `allowed_set_loss` uses raw `logits`, not `effective_logits`, so it remains separate from mask-guided inference penalties.

- [ ] **Step 6: Run allowed-set trainer test**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\application\test_trainer_topology_conditioned_learning.py::test_topology_conditioned_allowed_set_loss_logs_metrics -v
```

Expected: pass after metric logging is implemented in Step 8 below. If it still fails only because metric logging is missing, continue to Step 8 before rerun.

- [ ] **Step 7: Add negative loss integration**

Inside `_run_epoch`, after allowed-set loss and before optimizer step:

```python
if self._is_topology_conditioned_active(training=training):
    forward_stats["topology_conditioned_correct_ce_sum"] = float(
        forward_stats.get("topology_conditioned_correct_ce_sum", 0.0)
    ) + float(base_ce_loss.detach().cpu().item())
    forward_stats["topology_conditioned_batches"] = int(
        forward_stats.get("topology_conditioned_batches", 0)
    ) + 1

    if self._batch_has_mixed_versions(data):
        forward_stats["topology_conditioned_mixed_version_batches"] = int(
            forward_stats.get("topology_conditioned_mixed_version_batches", 0)
        ) + 1

    if (
        self.learning_strategy_config.drop_edges_negative_enabled
        and float(self.learning_strategy_config.drop_edges_negative_weight) > 0.0
    ):
        dropped_contract = drop_structural_edges(
            contract,
            drop_ratio=self.learning_strategy_config.drop_edges_ratio,
            seed=int(self.seed) + (int(epoch_index) * 100_003) + int(batch_idx),
        )
        dropped_logits = self.model(dropped_contract)
        if not self._is_finite_tensor(dropped_logits):
            dropped_logits = torch.nan_to_num(dropped_logits, nan=0.0, posinf=1e6, neginf=-1e6)
        dropped_ce = weighted_cross_entropy(
            logits=dropped_logits,
            targets=targets,
            sample_weights=sample_weights,
            class_weights=self.class_weights,
        )
        dropped_loss = margin_negative_loss(
            correct_ce=base_ce_loss,
            negative_ce=dropped_ce,
            margin=self.learning_strategy_config.drop_edges_margin,
        )
        loss = loss + (
            float(self.learning_strategy_config.drop_edges_negative_weight)
            * dropped_loss
        )
        forward_stats["topology_conditioned_drop_edges_ce_sum"] = float(
            forward_stats.get("topology_conditioned_drop_edges_ce_sum", 0.0)
        ) + float(dropped_ce.detach().cpu().item())
        forward_stats["topology_conditioned_drop_edges_delta_sum"] = float(
            forward_stats.get("topology_conditioned_drop_edges_delta_sum", 0.0)
        ) + float((dropped_ce - base_ce_loss).detach().cpu().item())
        forward_stats["topology_conditioned_drop_edges_loss_sum"] = float(
            forward_stats.get("topology_conditioned_drop_edges_loss_sum", 0.0)
        ) + float(dropped_loss.detach().cpu().item())
        forward_stats["topology_conditioned_drop_edges_batches"] = int(
            forward_stats.get("topology_conditioned_drop_edges_batches", 0)
        ) + 1

    if (
        self.learning_strategy_config.wrong_version_negative_enabled
        and float(self.learning_strategy_config.wrong_version_negative_weight) > 0.0
        and self._topology_payload_pool is not None
        and not self._batch_has_mixed_versions(data)
    ):
        current_version = self._batch_primary_version_label(data)
        wrong_payload = (
            self._topology_payload_pool.wrong_version_payload(current_version)
            if current_version is not None
            else None
        )
        if wrong_payload is not None:
            wrong_contract = replace_structural_payload(contract, wrong_payload)
            wrong_logits = self.model(wrong_contract)
            if not self._is_finite_tensor(wrong_logits):
                wrong_logits = torch.nan_to_num(wrong_logits, nan=0.0, posinf=1e6, neginf=-1e6)
            wrong_ce = weighted_cross_entropy(
                logits=wrong_logits,
                targets=targets,
                sample_weights=sample_weights,
                class_weights=self.class_weights,
            )
            wrong_loss = margin_negative_loss(
                correct_ce=base_ce_loss,
                negative_ce=wrong_ce,
                margin=self.learning_strategy_config.wrong_version_margin,
            )
            loss = loss + (
                float(self.learning_strategy_config.wrong_version_negative_weight)
                * wrong_loss
            )
            forward_stats["topology_conditioned_wrong_version_ce_sum"] = float(
                forward_stats.get("topology_conditioned_wrong_version_ce_sum", 0.0)
            ) + float(wrong_ce.detach().cpu().item())
            forward_stats["topology_conditioned_wrong_version_delta_sum"] = float(
                forward_stats.get("topology_conditioned_wrong_version_delta_sum", 0.0)
            ) + float((wrong_ce - base_ce_loss).detach().cpu().item())
            forward_stats["topology_conditioned_wrong_version_loss_sum"] = float(
                forward_stats.get("topology_conditioned_wrong_version_loss_sum", 0.0)
            ) + float(wrong_loss.detach().cpu().item())
            forward_stats["topology_conditioned_wrong_version_batches"] = int(
                forward_stats.get("topology_conditioned_wrong_version_batches", 0)
            ) + 1
```

This intentionally skips wrong-version negatives for mixed-version batches. `versioned_zero_shot` later should introduce version-homogeneous or operational samplers.

- [ ] **Step 8: Log topology-conditioned metrics**

Modify `_log_forward_stats` near existing structural aux / StructXAttn metric code.

Compute:

```python
tc_batches = int(bucket.get("topology_conditioned_batches", 0))
tc_allowed_batches = int(bucket.get("topology_conditioned_allowed_set_batches", 0))
tc_wrong_batches = int(bucket.get("topology_conditioned_wrong_version_batches", 0))
tc_drop_batches = int(bucket.get("topology_conditioned_drop_edges_batches", 0))
tc_allowed_loss = (
    float(bucket.get("topology_conditioned_allowed_set_loss_sum", 0.0)) / float(tc_allowed_batches)
    if tc_allowed_batches > 0
    else 0.0
)
tc_correct_ce = (
    float(bucket.get("topology_conditioned_correct_ce_sum", 0.0)) / float(tc_batches)
    if tc_batches > 0
    else 0.0
)
tc_wrong_ce = (
    float(bucket.get("topology_conditioned_wrong_version_ce_sum", 0.0)) / float(tc_wrong_batches)
    if tc_wrong_batches > 0
    else 0.0
)
tc_drop_ce = (
    float(bucket.get("topology_conditioned_drop_edges_ce_sum", 0.0)) / float(tc_drop_batches)
    if tc_drop_batches > 0
    else 0.0
)
tc_wrong_delta = (
    float(bucket.get("topology_conditioned_wrong_version_delta_sum", 0.0)) / float(tc_wrong_batches)
    if tc_wrong_batches > 0
    else 0.0
)
tc_drop_delta = (
    float(bucket.get("topology_conditioned_drop_edges_delta_sum", 0.0)) / float(tc_drop_batches)
    if tc_drop_batches > 0
    else 0.0
)
```

Log to tracker when `tc_batches > 0`:

```python
self.tracker.log_metric(f"{metric_prefix}_topology_conditioned_correct_ce", tc_correct_ce)
self.tracker.log_metric(f"{metric_prefix}_topology_conditioned_allowed_set_loss", tc_allowed_loss)
self.tracker.log_metric(f"{metric_prefix}_topology_conditioned_wrong_version_ce", tc_wrong_ce)
self.tracker.log_metric(f"{metric_prefix}_topology_conditioned_drop_edges_ce", tc_drop_ce)
self.tracker.log_metric(f"{metric_prefix}_topology_conditioned_wrong_version_ce_delta", tc_wrong_delta)
self.tracker.log_metric(f"{metric_prefix}_topology_conditioned_drop_edges_ce_delta", tc_drop_delta)
```

Also add the same values to the `logger.info("Forward stats...")` line or add a second concise line:

```python
logger.info(
    "Learning strategy stats [%s]: topology_conditioned_batches=%d "
    "correct_ce=%.6f allowed_set_loss=%.6f wrong_version_ce_delta=%.6f "
    "drop_edges_ce_delta=%.6f mixed_version_batches=%d",
    stage_label,
    tc_batches,
    tc_correct_ce,
    tc_allowed_loss,
    tc_wrong_delta,
    tc_drop_delta,
    int(bucket.get("topology_conditioned_mixed_version_batches", 0)),
)
```

- [ ] **Step 9: Rerun topology-conditioned trainer test**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\application\test_trainer_topology_conditioned_learning.py -v
```

Expected: pass.

---

## Task 5: Build Topology Payload Pool From Train Dataset

**Files:**

- Modify: `src/application/use_cases/trainer.py`
- Modify: `tests/application/test_trainer_topology_conditioned_learning.py`

### Steps

- [ ] **Step 1: Add failing integration test for wrong-version negative**

Append to `tests/application/test_trainer_topology_conditioned_learning.py`:

```python
def test_topology_conditioned_wrong_version_negative_uses_known_payload_pool():
    tracker = _RecordingTracker()
    trainer = ModelTrainer(
        xes_adapter=_DummyAdapter(),
        prefix_policy=_DummyPrefixPolicy(),
        graph_builder=_DummyGraphBuilder(),
        model=_TopologyConditionedModel(),
        log_path="in_memory.xes",
        config={
            "mode": "train",
            "device": "cpu",
            "show_progress": False,
            "tqdm_disable": True,
            "learning_strategy": "topology_conditioned",
            "topology_conditioning_allowed_set_loss_enabled": False,
            "topology_conditioning_drop_edges_negative_enabled": False,
            "topology_conditioning_wrong_version_negative_enabled": True,
            "topology_conditioning_wrong_version_negative_weight": 0.10,
            "topology_conditioning_wrong_version_margin": 0.05,
            "topology_conditioning_retention_enabled": False,
        },
        tracker=tracker,
        prepared_data={"idx_to_version": {0: "v1", 1: "v2"}},
    )
    trainer.criterion = nn.CrossEntropyLoss()
    train_dataset = [
        _sample(version_idx=0),
        _sample(version_idx=1),
    ]
    trainer._prepare_learning_strategy_for_train_dataset(train_dataset)
    loader = DataLoader([train_dataset[0]], batch_size=1, shuffle=False)

    trainer._run_epoch(loader, optimizer=None, training=True, epoch_index=1)

    metric_names = {name for name, _, _ in tracker.metrics}
    assert "train_topology_conditioned_wrong_version_ce_delta" in metric_names
```

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\application\test_trainer_topology_conditioned_learning.py::test_topology_conditioned_wrong_version_negative_uses_known_payload_pool -v
```

Expected: fails because `_prepare_learning_strategy_for_train_dataset` does not exist.

- [ ] **Step 2: Implement trainer preparation helper**

Add to `ModelTrainer`:

```python
def _prepare_learning_strategy_for_train_dataset(self, train_dataset: Iterable[Data]) -> None:
    if not self.learning_strategy_config.is_topology_conditioned:
        self._topology_payload_pool = None
        self._latest_train_version_idx = None
        return
    versions: list[int] = []
    for graph in train_dataset:
        if hasattr(graph, "process_version_idx"):
            raw = graph.process_version_idx
            if isinstance(raw, torch.Tensor) and raw.numel() > 0:
                versions.append(int(raw.view(-1)[0].detach().cpu().item()))
    self._latest_train_version_idx = max(versions) if versions else None
    self._topology_payload_pool = TopologyPayloadPool.from_dataset(
        train_dataset,
        idx_to_version=self._idx_to_version,
    )
    if self._topology_payload_pool.known_versions:
        logger.info(
            "Learning strategy topology payload pool: known_versions=%s",
            ",".join(self._topology_payload_pool.known_versions),
        )
```

- [ ] **Step 3: Call helper after train graph dataset is built**

In `ModelTrainer._run_train_pipeline()`, after `train_loader`, `val_loader`,
and `test_loader` are created and before `self._log_topology_metrics_and_artifacts()`,
add:

```python
self._prepare_learning_strategy_for_train_dataset(train_loader.dataset)
```

This location covers both paths already present in `_run_train_pipeline()`:

```python
if has_prebuilt:
    train_loader = self._build_loader_from_dataset(prebuilt_datasets.get("train"), shuffle=True)
    val_loader = self._build_loader_from_dataset(prebuilt_datasets.get("val"), shuffle=False)
    test_loader = self._build_loader_from_dataset(prebuilt_datasets.get("test"), shuffle=False)
else:
    train_loader = self._build_loader(split_data.train, shuffle=True)
    val_loader = self._build_loader(split_data.val, shuffle=False)
    test_loader = self._build_loader(split_data.test, shuffle=False)

self._prepare_learning_strategy_for_train_dataset(train_loader.dataset)
self._log_topology_metrics_and_artifacts()
```

- [ ] **Step 4: Run wrong-version test**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\application\test_trainer_topology_conditioned_learning.py::test_topology_conditioned_wrong_version_negative_uses_known_payload_pool -v
```

Expected: pass.

---

## Task 6: Config Catalog And CLI Validation

**Files:**

- Modify: `configs/ui/config_catalog.yaml`
- Modify: `tests/application/test_experiment_ui_catalog_fields.py`
- Modify: `tests/application/test_cli_flatten_config.py`

### Steps

- [ ] **Step 1: Add failing catalog test**

In `tests/application/test_experiment_ui_catalog_fields.py`, add:

```python
def test_config_catalog_includes_learning_strategy_keys(catalog):
    keys = catalog["fields"].keys() if "fields" in catalog else catalog.keys()

    assert "training.learning_strategy" in keys
    assert "training.topology_conditioning_allowed_set_loss_enabled" in keys
    assert "training.topology_conditioning_wrong_version_negative_enabled" in keys
    assert "training.topology_conditioning_drop_edges_negative_enabled" in keys
    assert "training.topology_conditioning_retention_enabled" in keys
```

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\application\test_experiment_ui_catalog_fields.py -v
```

Expected: fail until catalog is updated.

- [ ] **Step 2: Add catalog entries**

Add entries near existing `training.struct_xattn_*` fields in `configs/ui/config_catalog.yaml`:

```yaml
  training.learning_strategy:
    path: training.learning_strategy
    section: training
    label: learning_strategy
    type: enum
    default: standard
    choices:
      - standard
      - topology_conditioned
    description: Selects trainer-level learning methodology. standard preserves current behavior; topology_conditioned enables topology-conditioned train-time losses.
    affects: Training loss composition, topology negative sampling, retention weighting, and diagnostics.
```

Add analogous entries for each key listed in the Config Keys section. Use:

- `type: bool` for enabled flags;
- `type: float` for weights/margins/ratios;
- `type: integer` for `retention_recent_versions`;
- `visible_if` or existing catalog condition format tied to `training.learning_strategy: topology_conditioned` if supported by current catalog style.

- [ ] **Step 3: Add CLI validation test**

In `tests/application/test_cli_flatten_config.py`, add:

```python
def test_apply_experiment_switch_overrides_preserves_learning_strategy():
    config = {
        "experiment": {"structural_mode": "true"},
        "model": {"type": "EOPKGGATv2"},
        "training": {"learning_strategy": "topology_conditioned"},
    }

    patched = _apply_experiment_switch_overrides(config)

    assert patched["training"]["learning_strategy"] == "topology_conditioned"
    assert patched["model"]["structural_mode"] is True
```

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\application\test_cli_flatten_config.py::test_apply_experiment_switch_overrides_preserves_learning_strategy -v
```

Expected: pass without code change if config is already preserved. If it fails, adjust `_apply_experiment_switch_overrides` without changing existing switch behavior.

- [ ] **Step 4: Run catalog/CLI tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\application\test_experiment_ui_catalog_fields.py tests\application\test_cli_flatten_config.py -v
```

Expected: pass.

---

## Task 7: Backward Compatibility And Regression Tests

**Files:**

- Modify only if tests reveal defects.

### Steps

- [ ] **Step 1: Verify standard strategy does not activate new losses**

Add to `tests/application/test_trainer_topology_conditioned_learning.py`:

```python
def test_standard_learning_strategy_does_not_run_topology_conditioned_losses():
    tracker = _RecordingTracker()
    trainer = ModelTrainer(
        xes_adapter=_DummyAdapter(),
        prefix_policy=_DummyPrefixPolicy(),
        graph_builder=_DummyGraphBuilder(),
        model=_TopologyConditionedModel(),
        log_path="in_memory.xes",
        config={
            "mode": "train",
            "device": "cpu",
            "show_progress": False,
            "tqdm_disable": True,
            "learning_strategy": "standard",
        },
        tracker=tracker,
        prepared_data={"idx_to_version": {0: "v1"}},
    )
    trainer.criterion = nn.CrossEntropyLoss()
    loader = DataLoader([_sample()], batch_size=1, shuffle=False)

    trainer._run_epoch(loader, optimizer=None, training=True, epoch_index=1)

    metric_names = {name for name, _, _ in tracker.metrics}
    assert all("topology_conditioned" not in name for name in metric_names)
```

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\application\test_trainer_topology_conditioned_learning.py::test_standard_learning_strategy_does_not_run_topology_conditioned_losses -v
```

Expected: pass.

- [ ] **Step 2: Verify legacy StructXAttn contrastive tests still pass**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\application\test_trainer_struct_xattn_training.py -v
```

Expected: pass.

- [ ] **Step 3: Verify model forward tests remain unchanged**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\domain\test_eopkg_models_forward.py tests\domain\test_eopkg_struct_xattn.py -v
```

Expected: pass.

- [ ] **Step 4: Verify MVP1 regression**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -m mvp1_regression -v
```

Expected: pass.

---

## Task 8: Documentation Update

**Files:**

- Modify: `docs/GNN_LEARNING_STRATEGY.MD`
- Modify: `docs/GNN_RUNTIME_MVP2_5.MD`
- Modify: `docs/EVF_MVP2_5.MD`
- Modify: `docs/current/project-state.md`
- Modify: `docs/dissertation_changes.MD`

### Steps

- [ ] **Step 1: Update `docs/GNN_LEARNING_STRATEGY.MD` status**

Change:

```markdown
- `status`: design
```

to:

```markdown
- `status`: implemented
```

Add an `Implementation Notes` section:

```markdown
## Implementation Notes

- `standard` preserves legacy trainer behavior.
- `topology_conditioned` is implemented in `ModelTrainer`, not in model forward.
- `wrong_version_topology` uses prebuilt structural tensors collected from train data.
- `drop_edges` is the only implemented edge corruption policy.
- `versioned_zero_shot` remains a separate future experiment mode.
```

- [ ] **Step 2: Update `docs/GNN_RUNTIME_MVP2_5.MD`**

Add a short section near training semantics:

```markdown
### Learning Strategy Boundary

`fusion_mode` controls forward-time structural fusion. `training.learning_strategy`
controls trainer-time methodology. `training.learning_strategy=topology_conditioned`
does not add a new model forward path; it adds train-time topology-conditioned
losses around the existing `fusion_mode`.
```

- [ ] **Step 3: Update `docs/EVF_MVP2_5.MD`**

Add evaluation interpretation:

```markdown
For learning-strategy experiments, compare:

1. `standard + no mask`
2. `standard + mask`
3. `topology_conditioned + no mask`
4. `topology_conditioned + mask`

This separates constraint effect from learning effect.
```

- [ ] **Step 4: Update `docs/current/project-state.md`**

Replace planned wording:

```text
The planned training.learning_strategy=standard...
```

with implemented wording:

```text
training.learning_strategy=standard preserves current behavior, while
training.learning_strategy=topology_conditioned adds trainer-level topology-conditioned losses...
```

- [ ] **Step 5: Update `docs/dissertation_changes.MD`**

Add one entry:

```markdown
## 2026-05-21 - Implemented GNN Learning Strategy Contract

- `topic`: implementation of `training.learning_strategy`
- `impact`: critical methodology implementation
- `status`: implemented

**Note (ukr):**

Реалізація відокремлює спосіб подачі topology (`fusion_mode`) від методології
навчання (`learning_strategy`). Це важливо для дисертації, бо дозволяє окремо
оцінювати архітектурний ефект fusion і навчальний ефект topology-conditioned
losses.
```

---

## Task 9: Full Verification

**Files:**

- No planned file changes.

### Steps

- [ ] **Step 1: Run focused application/domain tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\application\test_learning_strategy_config.py tests\application\test_topology_payload_pool.py tests\application\test_topology_conditioned_learning.py tests\application\test_trainer_topology_conditioned_learning.py tests\application\test_trainer_struct_xattn_training.py tests\domain\test_eopkg_models_forward.py -v
```

Expected: pass.

- [ ] **Step 2: Run architecture guard**

Run:

```powershell
.\.venv\Scripts\python.exe tools\architecture_guard.py
```

Expected: `[ARCH_GUARD] OK`.

- [ ] **Step 3: Run full test suite**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\ -v
```

Expected: pass.

- [ ] **Step 4: Create implementation report**

Create:

```text
docs/worklogs/MVP2_5_Learning_Strategy_Implementation_Report_2026-05-21.MD
```

Include:

- implemented files;
- config keys;
- tests run and results;
- known limitations:
  - `versioned_zero_shot` not implemented;
  - wrong-version negative skips mixed-version batches;
  - only `drop_edges` corruption implemented;
  - retention is version-weighted CE/rehearsal, no distillation.

---

## Implementation Notes And Risks

### Risk: Wrong-Version Negative And Mixed Batches

Current `ModelTrainer._data_to_contract()` uses one structural payload for a batch. Therefore `wrong_version_topology` must skip mixed-version batches in this implementation. This is acceptable for first trainer-level support and must be logged through `topology_conditioned_mixed_version_batches`.

Future `versioned_zero_shot` should add version-homogeneous or operational samplers.

### Risk: Mask-Guided Interference

`L_allowed_set` must use raw `logits`, not mask-guided `effective_logits`. Otherwise it becomes indistinguishable from `experiment.mask_guided_enabled`.

### Risk: Legacy StructXAttn Contrastive Duplication

Do not run legacy `struct_xattn_contrastive_enabled` together with `learning_strategy=topology_conditioned`. Fail-fast to avoid double negative-topology objectives.

### Risk: Class Weights

If `self.class_weights` exists, `weighted_cross_entropy()` must pass it into `F.cross_entropy(..., reduction="none")`. This preserves class-weight behavior for topology-conditioned runs.

### Risk: ShardedGraphDataset Iteration

Building a topology payload pool by iterating a large sharded dataset can be expensive. First implementation may rely on materialized train data. If train data is sharded, collect only one payload per version and stop when all known versions are found, or log a warning and disable wrong-version negatives while keeping drop-edge and allowed-set losses active.

---

## Self-Review Checklist

- [ ] `standard` path preserves current behavior.
- [ ] `topology_conditioned` fails fast when `structural_mode=false`.
- [ ] `allowed_set_loss_enabled=true` fails fast when `allowed_target_mask` is missing.
- [ ] `drop_edges` does not mutate original contract.
- [ ] `wrong_version_topology` uses only payloads from prebuilt train data.
- [ ] Future topology is not introduced anywhere in this plan.
- [ ] Legacy `struct_xattn_contrastive_enabled` still works for `standard`.
- [ ] New config keys are represented in `configs/ui/config_catalog.yaml`.
- [ ] Docs are updated after implementation.
- [ ] Full tests and architecture guard pass before completion.
