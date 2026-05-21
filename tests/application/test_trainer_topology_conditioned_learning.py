from __future__ import annotations

from typing import Iterator

import torch
from torch import nn
from torch.optim import Adam

from src.application.use_cases.trainer import ModelTrainer
from src.domain.entities.raw_trace import RawTrace


class _DummyAdapter:
    def read(self, file_path: str, mapping_config: dict) -> Iterator[RawTrace]:
        _ = file_path
        _ = mapping_config
        return iter([])


class _DummyPrefixPolicy:
    def generate_slices(self, trace: RawTrace):
        _ = trace
        return []


class _DummyGraphBuilder:
    def build_graph(self, prefix):
        _ = prefix
        raise RuntimeError("Not used in this test.")


class _RecordingTracker:
    def __init__(self) -> None:
        self.metrics: list[tuple[str, float, int | None]] = []
        self.params: dict[str, object] = {}

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        self.metrics.append((key, float(value), step))

    def log_param(self, key: str, value: object) -> None:
        self.params[key] = value


class _TopologySensitiveModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dummy = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.fusion_mode = "class_aware_structural_scoring"
        self.structural_mode = True
        self.calls = 0

    def forward(self, contract):
        self.calls += 1
        batch = contract["batch"]
        graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else int(contract["y"].view(-1).numel())
        edge_index = contract.get("structural_edge_index")
        edge_count = int(edge_index.size(1)) if isinstance(edge_index, torch.Tensor) and edge_index.dim() == 2 else 0
        logits = torch.tensor([[-1.0, 2.0]], dtype=torch.float32, device=batch.device)
        if edge_count < 3:
            logits = torch.tensor([[2.0, -1.0]], dtype=torch.float32, device=batch.device)
        self.last_observed_logits = logits.repeat(graphs, 1).detach()
        self.last_structural_class_logits = torch.zeros_like(self.last_observed_logits)
        return logits.repeat(graphs, 1) + (self.dummy * 0.0)


def _sample(version_idx: int, *, edge_count: int = 4):
    from torch_geometric.data import Data

    edges = torch.tensor([[0, 1, 1, 2], [1, 2, 3, 3]], dtype=torch.long)[:, :edge_count]
    return Data(
        x_cat=torch.zeros((1, 0), dtype=torch.long),
        x_num=torch.ones((1, 1), dtype=torch.float32),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
        edge_type=torch.zeros((0,), dtype=torch.long),
        y=torch.tensor([1], dtype=torch.long),
        num_nodes=1,
        allowed_target_mask=torch.tensor([[True, True]], dtype=torch.bool),
        struct_x=torch.full((4, 2), float(version_idx), dtype=torch.float32),
        structural_edge_index=edges,
        structural_edge_weight=torch.ones(edge_count, dtype=torch.float32),
        struct_node_to_class_index=torch.tensor([0, 1, 0, 1], dtype=torch.long),
        process_version_idx=torch.tensor([version_idx], dtype=torch.long),
    )


def _trainer(*, model: nn.Module, tracker: _RecordingTracker | None = None, config: dict | None = None):
    base_config = {
        "mode": "train",
        "device": "cpu",
        "show_progress": False,
        "tqdm_disable": True,
        "batch_size": 1,
        "learning_strategy": "topology_conditioned",
        "topology_conditioning_allowed_set_loss_enabled": True,
        "topology_conditioning_allowed_set_loss_weight": 0.10,
        "topology_conditioning_wrong_version_negative_enabled": True,
        "topology_conditioning_wrong_version_negative_weight": 0.10,
        "topology_conditioning_wrong_version_margin": 0.05,
        "topology_conditioning_drop_edges_negative_enabled": True,
        "topology_conditioning_drop_edges_negative_weight": 0.10,
        "topology_conditioning_drop_edges_margin": 0.05,
        "topology_conditioning_drop_edges_ratio": 1.0,
        "topology_conditioning_retention_enabled": True,
        "topology_conditioning_retention_recent_weight": 0.50,
        "topology_conditioning_retention_old_weight": 0.25,
        "topology_conditioning_retention_obsolete_weight": 0.0,
    }
    if config:
        base_config.update(config)
    trainer = ModelTrainer(
        xes_adapter=_DummyAdapter(),
        prefix_policy=_DummyPrefixPolicy(),
        graph_builder=_DummyGraphBuilder(),
        model=model,
        log_path="in_memory.xes",
        config=base_config,
        tracker=tracker,
        prepared_data={"idx_to_version": {1: "v1", 2: "v2"}},
    )
    trainer.criterion = nn.CrossEntropyLoss()
    return trainer


def test_topology_conditioned_training_logs_losses_and_uses_negative_forwards():
    tracker = _RecordingTracker()
    model = _TopologySensitiveModel()
    trainer = _trainer(model=model, tracker=tracker)
    loader = trainer._build_loader_from_dataset([_sample(1, edge_count=4), _sample(2, edge_count=2)], shuffle=False)
    optimizer = Adam(model.parameters(), lr=0.01)

    train_loss, *_ = trainer._run_epoch(loader, optimizer=optimizer, training=True)

    metric_names = {name for name, _, _ in tracker.metrics}
    assert train_loss > 0.0
    assert model.calls > 2
    assert "train_topology_conditioned_allowed_set_loss" in metric_names
    assert "train_topology_conditioned_wrong_version_ce_delta" in metric_names
    assert "train_topology_conditioned_drop_edges_ce_delta" in metric_names
    assert "train_topology_conditioned_retention_loss" in metric_names


def test_standard_learning_strategy_does_not_add_negative_forwards():
    model = _TopologySensitiveModel()
    trainer = _trainer(
        model=model,
        config={
            "learning_strategy": "standard",
            "topology_conditioning_wrong_version_negative_enabled": True,
            "topology_conditioning_drop_edges_negative_enabled": True,
        },
    )
    loader = trainer._build_loader_from_dataset([_sample(1), _sample(2)], shuffle=False)
    optimizer = Adam(model.parameters(), lr=0.01)

    trainer._run_epoch(loader, optimizer=optimizer, training=True)

    assert model.calls == 2


def test_mixed_version_batch_skips_wrong_version_loss_with_metric():
    tracker = _RecordingTracker()
    model = _TopologySensitiveModel()
    trainer = _trainer(model=model, tracker=tracker, config={"batch_size": 2})
    loader = trainer._build_loader_from_dataset([_sample(1), _sample(2)], shuffle=False)
    optimizer = Adam(model.parameters(), lr=0.01)

    trainer._run_epoch(loader, optimizer=optimizer, training=True)

    metrics = {name: value for name, value, _ in tracker.metrics}
    assert metrics["train_topology_conditioned_mixed_version_batches"] == 1.0
    assert metrics["train_topology_conditioned_wrong_version_skipped_mixed_batches"] == 1.0


def test_topology_conditioned_train_loader_prefers_version_homogeneous_batches():
    model = _TopologySensitiveModel()
    trainer = _trainer(model=model, config={"batch_size": 2})
    loader = trainer._build_loader_from_dataset(
        [_sample(1), _sample(2), _sample(1), _sample(2)],
        shuffle=True,
    )

    for batch in loader:
        versions = batch.process_version_idx.view(-1).long().tolist()
        assert len(set(versions)) == 1
