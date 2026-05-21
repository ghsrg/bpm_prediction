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

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        self.metrics.append((key, float(value), step))


class _StructXAttnContrastiveModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dummy = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.fusion_mode = "struct_xattn"
        self.structural_mode = True
        self.calls = 0
        self.last_struct_xattn_context_mean_abs: torch.Tensor | None = None
        self.last_struct_xattn_delta_mean_abs: torch.Tensor | None = None
        self.last_struct_xattn_to_observed_ratio: torch.Tensor | None = None
        self.last_struct_xattn_attention_entropy: torch.Tensor | None = None
        self.last_struct_xattn_scale: torch.Tensor | None = None
        self.last_struct_xattn_gate_mean: torch.Tensor | None = None

    def forward(self, contract):
        self.calls += 1
        batch = contract["batch"]
        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
        edge_index = contract.get("structural_edge_index")
        edge_count = int(edge_index.size(1)) if isinstance(edge_index, torch.Tensor) and edge_index.dim() == 2 else 0
        if edge_count >= 4:
            base = torch.tensor([[-1.0, 1.0]], dtype=torch.float32, device=batch.device)
        else:
            base = torch.tensor([[1.0, -1.0]], dtype=torch.float32, device=batch.device)
        self.last_struct_xattn_context_mean_abs = torch.tensor(0.25, dtype=torch.float32, device=batch.device)
        self.last_struct_xattn_delta_mean_abs = torch.tensor(0.10, dtype=torch.float32, device=batch.device)
        self.last_struct_xattn_to_observed_ratio = torch.tensor(0.20, dtype=torch.float32, device=batch.device)
        self.last_struct_xattn_attention_entropy = torch.tensor(1.10, dtype=torch.float32, device=batch.device)
        self.last_struct_xattn_scale = torch.tensor(0.10, dtype=torch.float32, device=batch.device)
        self.last_struct_xattn_gate_mean = torch.tensor(0.15, dtype=torch.float32, device=batch.device)
        return base.repeat(num_graphs, 1) + (self.dummy * 0.0)


def _sample() -> Data:
    return Data(
        x_cat=torch.zeros((1, 0), dtype=torch.long),
        x_num=torch.ones((1, 1), dtype=torch.float32),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
        edge_type=torch.zeros((0,), dtype=torch.long),
        y=torch.tensor([1], dtype=torch.long),
        batch=torch.tensor([0], dtype=torch.long),
        num_nodes=1,
        struct_x=torch.ones((4, 2), dtype=torch.float32),
        structural_edge_index=torch.tensor([[0, 1, 1, 2], [1, 2, 3, 3]], dtype=torch.long),
        structural_edge_weight=torch.ones(4, dtype=torch.float32),
        struct_node_to_class_index=torch.tensor([0, 1, 2, 3], dtype=torch.long),
    )


def _trainer(*, model: nn.Module, tracker: _RecordingTracker | None = None, config: dict | None = None) -> ModelTrainer:
    base_config = {
        "mode": "train",
        "device": "cpu",
        "show_progress": False,
        "tqdm_disable": True,
        "struct_xattn_contrastive_enabled": True,
        "struct_xattn_contrastive_weight": 0.10,
        "struct_xattn_contrastive_margin": 0.05,
        "struct_xattn_contrastive_temperature": 0.10,
        "struct_xattn_contrastive_max_loss": 0.50,
        "struct_xattn_contrastive_warmup_epochs": 0,
        "struct_xattn_corruption_policy": "edge_drop",
        "struct_xattn_corruption_edge_drop_prob": 1.0,
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
    )
    trainer.criterion = nn.CrossEntropyLoss()
    return trainer


def test_struct_xattn_corruption_does_not_mutate_original_contract():
    trainer = _trainer(model=_StructXAttnContrastiveModel())
    contract = trainer._data_to_contract(_sample())
    original_edges = contract["structural_edge_index"].clone()

    corrupted = trainer._build_corrupted_structural_contract(contract, epoch_index=1, batch_idx=1)

    assert torch.equal(contract["structural_edge_index"], original_edges)
    assert corrupted is not contract
    assert corrupted["structural_edge_index"].size(1) < original_edges.size(1)
    assert corrupted["structural_edge_index"].size(1) >= 1


def test_struct_xattn_contrastive_loss_runs_only_during_training_and_logs_metrics():
    tracker = _RecordingTracker()
    model = _StructXAttnContrastiveModel()
    trainer = _trainer(model=model, tracker=tracker)
    loader = DataLoader([_sample()], batch_size=1, shuffle=False)
    optimizer = Adam(model.parameters(), lr=0.01)

    train_loss, *_ = trainer._run_epoch(loader, optimizer=optimizer, training=True)

    assert train_loss > 0.0
    assert model.calls == 2
    metric_names = {name for name, _, _ in tracker.metrics}
    assert "train_struct_xattn_correct_vs_corrupt_ce_delta" in metric_names
    assert "train_struct_xattn_contrastive_loss" in metric_names
    assert "train_struct_xattn_delta_mean_abs" in metric_names

    model.calls = 0
    trainer._run_epoch(loader, optimizer=None, training=False)
    assert model.calls == 1


def test_struct_xattn_contrastive_warmup_disables_corrupted_forward():
    model = _StructXAttnContrastiveModel()
    trainer = _trainer(
        model=model,
        config={"struct_xattn_contrastive_warmup_epochs": 5},
    )
    loader = DataLoader([_sample()], batch_size=1, shuffle=False)
    optimizer = Adam(model.parameters(), lr=0.01)

    trainer._run_epoch(loader, optimizer=optimizer, training=True, epoch_index=1)

    assert model.calls == 1
