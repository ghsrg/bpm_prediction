from __future__ import annotations

from typing import Iterator, Sequence

import torch
from torch import nn
from torch.optim import Adam
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from src.application.use_cases.trainer import ModelTrainer, SplitData
from src.domain.entities.feature_config import FeatureLayout
from src.domain.entities.raw_trace import RawTrace
from src.domain.models.baseline_gcn import BaselineGCN
from src.domain.services.baseline_graph_builder import BaselineGraphBuilder
from src.domain.services.feature_encoder import FeatureEncoder
from src.domain.services.prefix_policy import PrefixPolicy


def _make_stress_contract() -> dict[str, torch.Tensor | int]:
    # Graph batch: [2 nodes] + [5 nodes] + [1 isolated node]
    x_num = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    x_cat = torch.zeros((8, 0), dtype=torch.long)
    edge_index = torch.tensor(
        [
            [0, 2, 3, 4, 5],
            [1, 3, 4, 5, 6],
        ],
        dtype=torch.long,
    )
    edge_type = torch.zeros(edge_index.shape[1], dtype=torch.long)
    batch = torch.tensor([0, 0, 1, 1, 1, 1, 1, 2], dtype=torch.long)
    y = torch.tensor([0, 1, 2], dtype=torch.long)
    return {
        "x_cat": x_cat,
        "x_num": x_num,
        "edge_index": edge_index,
        "edge_type": edge_type,
        "y": y,
        "batch": batch,
        "num_nodes": int(x_num.size(0)),
    }


def _to_data(contract: dict[str, torch.Tensor | int]) -> Data:
    return Data(
        x_cat=contract["x_cat"],
        x_num=contract["x_num"],
        edge_index=contract["edge_index"],
        edge_type=contract["edge_type"],
        y=contract["y"],
        num_nodes=int(contract["num_nodes"]),
    )


class _InMemoryAdapter:
    def __init__(self, traces: Sequence[RawTrace]) -> None:
        self._traces = traces

    def read(self, file_path: str, mapping_config: dict) -> Iterator[RawTrace]:
        _ = file_path
        _ = mapping_config
        return iter(self._traces)


def test_gnn_isolated_forward_backward_stress_math_and_edge_cases():
    torch.manual_seed(7)
    contract = _make_stress_contract()

    model = BaselineGCN(
        feature_layout=FeatureLayout(cat_features={}, cat_feature_names=[], num_dim=3),
        hidden_dim=16,
        output_dim=3,
        dropout=0.0,
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.05)

    losses: list[float] = []
    for _ in range(50):
        model.train()
        optimizer.zero_grad()
        logits = model(contract)
        loss = criterion(logits, contract["y"])
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().item()))

    assert losses[-1] < 0.05

    model.eval()
    with torch.no_grad():
        preds = torch.argmax(model(contract), dim=1)
    assert torch.equal(preds, contract["y"])


def test_trainer_smoke_wiring_with_in_memory_dataloader(
    tmp_path,
    monkeypatch,
    mock_feature_configs,
    mock_raw_trace,
):
    traces = [mock_raw_trace, mock_raw_trace, mock_raw_trace]
    prefix_policy = PrefixPolicy()
    feature_encoder = FeatureEncoder(feature_configs=mock_feature_configs, traces=traces)
    graph_builder = BaselineGraphBuilder(feature_encoder=feature_encoder)

    contracts: list[dict[str, torch.Tensor | int]] = []
    for trace in traces:
        for prefix in prefix_policy.generate_slices(trace):
            contracts.append(graph_builder.build_graph(prefix))
    assert contracts

    graph_data = [_to_data(contract) for contract in contracts]
    train_loader = DataLoader(graph_data, batch_size=4, shuffle=True)
    val_loader = DataLoader(graph_data[: max(1, len(graph_data) // 2)], batch_size=2, shuffle=False)
    test_loader = DataLoader(graph_data[-max(1, len(graph_data) // 2) :], batch_size=2, shuffle=False)

    model = BaselineGCN(
        feature_layout=feature_encoder.feature_layout,
        hidden_dim=16,
        output_dim=len(feature_encoder.categorical_vocabs[feature_encoder.activity_feature_name]),
        dropout=0.0,
    )

    trainer = ModelTrainer(
        xes_adapter=_InMemoryAdapter(traces),
        prefix_policy=prefix_policy,
        graph_builder=graph_builder,
        model=model,
        log_path="in_memory.xes",
        config={
            "epochs": 2,
            "batch_size": 4,
            "learning_rate": 0.01,
            "device": "cpu",
            "show_progress": False,
            "tqdm_disable": True,
            "tqdm_leave": False,
            "patience": 5,
            "delta": 0.0,
            "checkpoint_dir": str(tmp_path),
            "experiment_config": {
                "name": "pytest_training_smoke",
                "fraction": 1.0,
                "split_strategy": "temporal",
                "train_ratio": 0.7,
                "split_ratio": [0.6, 0.2, 0.2],
            },
        },
        tracker=None,
    )
    trainer.criterion = nn.CrossEntropyLoss()

    loaders = iter([train_loader, val_loader, test_loader])
    monkeypatch.setattr(trainer, "_build_loader", lambda traces_arg, shuffle: next(loaders))

    results = trainer._run_train_pipeline(
        split_data=SplitData(train=[mock_raw_trace], val=[mock_raw_trace], test=[mock_raw_trace]),
        checkpoint=None,
        start_epoch=0,
        best_val_loss=float("inf"),
        best_epoch=0,
    )

    assert isinstance(results, dict)
    assert "history" in results
    assert results["history"]
    assert "train_loss" in results["history"][0]
    assert "test_metrics" in results and isinstance(results["test_metrics"], dict)
