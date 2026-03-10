from __future__ import annotations

from typing import Iterator

import torch
from torch import nn
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


class _ConstantZeroModel(nn.Module):
    """Always predicts class 0 to ensure preds are independent from labels."""

    def forward(self, contract):
        batch = contract["batch"]
        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
        logits = torch.zeros((num_graphs, 2), dtype=torch.float32, device=batch.device)
        logits[:, 0] = 5.0
        return logits


class _ConstantOneModel(nn.Module):
    """Always predicts class 1 to validate single-class window metrics."""

    def forward(self, contract):
        batch = contract["batch"]
        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
        logits = torch.zeros((num_graphs, 2), dtype=torch.float32, device=batch.device)
        logits[:, 1] = 5.0
        return logits


def test_eval_drift_predictions_are_model_outputs_not_labels(caplog):
    trainer = ModelTrainer(
        xes_adapter=_DummyAdapter(),
        prefix_policy=_DummyPrefixPolicy(),
        graph_builder=_DummyGraphBuilder(),
        model=_ConstantZeroModel(),
        log_path="in_memory.xes",
        config={
            "mode": "eval_drift",
            "device": "cpu",
            "show_progress": False,
            "tqdm_disable": True,
            "experiment_config": {"drift_window_size": 5, "drift_window_sliding": 2},
        },
    )

    sample_1 = Data(
        x_cat=torch.zeros((1, 0), dtype=torch.long),
        x_num=torch.ones((1, 1), dtype=torch.float32),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
        edge_type=torch.zeros((0,), dtype=torch.long),
        y=torch.tensor([1], dtype=torch.long),
        num_nodes=1,
    )
    sample_2 = Data(
        x_cat=torch.zeros((1, 0), dtype=torch.long),
        x_num=torch.ones((1, 1), dtype=torch.float32),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
        edge_type=torch.zeros((0,), dtype=torch.long),
        y=torch.tensor([1], dtype=torch.long),
        num_nodes=1,
    )
    loader = DataLoader([sample_1, sample_2], batch_size=2, shuffle=False)

    caplog.set_level("INFO")
    metrics = trainer._evaluate_test(loader)

    assert metrics["test_accuracy"] == 0.0
    assert metrics["test_macro_f1"] == 0.0
    assert "Drift debug first batch y_true[:5]=[1, 1] y_pred[:5]=[0, 0]" in caplog.text


def test_eval_metrics_single_class_window_can_reach_one():
    trainer = ModelTrainer(
        xes_adapter=_DummyAdapter(),
        prefix_policy=_DummyPrefixPolicy(),
        graph_builder=_DummyGraphBuilder(),
        model=_ConstantOneModel(),
        log_path="in_memory.xes",
        config={
            "mode": "eval_drift",
            "device": "cpu",
            "show_progress": False,
            "tqdm_disable": True,
            "experiment_config": {"drift_window_size": 5},
        },
    )

    sample_1 = Data(
        x_cat=torch.zeros((1, 0), dtype=torch.long),
        x_num=torch.ones((1, 1), dtype=torch.float32),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
        edge_type=torch.zeros((0,), dtype=torch.long),
        y=torch.tensor([1], dtype=torch.long),
        num_nodes=1,
    )
    sample_2 = Data(
        x_cat=torch.zeros((1, 0), dtype=torch.long),
        x_num=torch.ones((1, 1), dtype=torch.float32),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
        edge_type=torch.zeros((0,), dtype=torch.long),
        y=torch.tensor([1], dtype=torch.long),
        num_nodes=1,
    )
    loader = DataLoader([sample_1, sample_2], batch_size=2, shuffle=False)
    metrics = trainer._evaluate_test(loader)

    assert metrics["test_accuracy"] == 1.0
    assert metrics["test_macro_f1"] == 1.0
