from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterator
import logging
import math

import pytest
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


class _TrainableBinaryModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.logit_bias = nn.Parameter(torch.tensor([1.5, -1.5], dtype=torch.float32))

    def forward(self, contract):
        batch = contract["batch"]
        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
        return self.logit_bias.unsqueeze(0).repeat(num_graphs, 1)


class _NaNLogitModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dummy = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

    def forward(self, contract):
        batch = contract["batch"]
        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
        logits = torch.full((num_graphs, 4), float("nan"), dtype=torch.float32, device=batch.device)
        return logits + (self.dummy * 0.0)


def _sample(y_value: int, *, snapshot_idx: int, snapshot_epoch: float) -> Data:
    return Data(
        x_cat=torch.zeros((1, 0), dtype=torch.long),
        x_num=torch.ones((1, 1), dtype=torch.float32),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
        edge_type=torch.zeros((0,), dtype=torch.long),
        y=torch.tensor([y_value], dtype=torch.long),
        num_nodes=1,
        struct_x=torch.tensor([[1.0], [0.5]], dtype=torch.float32),
        structural_edge_index=torch.tensor([[0], [1]], dtype=torch.long),
        structural_edge_weight=torch.tensor([1.0], dtype=torch.float32),
        stats_snapshot_version_idx=torch.tensor([snapshot_idx], dtype=torch.long),
        stats_snapshot_as_of_epoch=torch.tensor([snapshot_epoch], dtype=torch.float64),
    )


def _sample_with_missing_asof(y_value: int, *, snapshot_idx: int, snapshot_epoch: float, missing_asof: bool) -> Data:
    payload = _sample(y_value=y_value, snapshot_idx=snapshot_idx, snapshot_epoch=snapshot_epoch)
    payload.stats_missing_asof_snapshot = torch.tensor([1 if missing_asof else 0], dtype=torch.long)
    payload.stats_allowed = torch.tensor([0 if missing_asof else 1], dtype=torch.long)
    return payload


def _sample_with_projection_summary(y_value: int, *, aligned: bool, skipped_edges: int, missing_vocab: int) -> Data:
    payload = _sample(
        y_value=y_value,
        snapshot_idx=7,
        snapshot_epoch=float(datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc).timestamp()),
    )
    payload.topology_projection_aligned = torch.tensor([1 if aligned else 0], dtype=torch.long)
    payload.topology_projection_projected_edge_count = torch.tensor([3], dtype=torch.long)
    payload.topology_projection_source_path_count = torch.tensor([4], dtype=torch.long)
    payload.topology_projection_skipped_edge_count = torch.tensor([skipped_edges], dtype=torch.long)
    payload.topology_projection_missing_vocab_count = torch.tensor([missing_vocab], dtype=torch.long)
    payload.topology_projection_duplicate_label_count = torch.tensor([0], dtype=torch.long)
    payload.topology_projection_missing_node_metadata = torch.tensor([0], dtype=torch.long)
    return payload


def test_trainer_logs_forward_stats_for_train_inference_and_drift(caplog):
    snapshot_epoch = float(datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc).timestamp())
    loader = DataLoader(
        [
            _sample(0, snapshot_idx=7, snapshot_epoch=snapshot_epoch),
            _sample(1, snapshot_idx=7, snapshot_epoch=snapshot_epoch),
        ],
        batch_size=2,
        shuffle=False,
    )

    trainer = ModelTrainer(
        xes_adapter=_DummyAdapter(),
        prefix_policy=_DummyPrefixPolicy(),
        graph_builder=_DummyGraphBuilder(),
        model=_TrainableBinaryModel(),
        log_path="in_memory.xes",
        config={
            "mode": "train",
            "device": "cpu",
            "show_progress": False,
            "tqdm_disable": True,
            "batch_size": 2,
            "experiment_config": {"drift_window_size": 2},
        },
    )
    trainer.criterion = nn.CrossEntropyLoss()
    trainer._idx_to_stats_snapshot_version[7] = "k000777"
    trainer._stats_snapshot_version_to_idx["k000777"] = 7

    optimizer = Adam(trainer.model.parameters(), lr=0.01)

    caplog.set_level(logging.INFO)
    trainer._run_epoch(loader, optimizer=optimizer, training=True)
    trainer._evaluate_test(loader, stage_label="inference")
    trainer.mode = "eval_drift"
    trainer._evaluate_test(loader, stage_label="eval_drift")

    assert "Forward stats [train]:" in caplog.text
    assert "Forward stats [inference]:" in caplog.text
    assert "Forward stats [eval_drift]:" in caplog.text
    assert "snapshot_versions=k000777" in caplog.text
    assert "snapshot_as_of_ts=2026-03-20T12:00:00+00:00" in caplog.text
    assert "missing_asof_snapshot_batches=0 missing_asof_snapshot[true/false]=0/0" in caplog.text


def test_trainer_logs_run_profile_banner(caplog):
    trainer = ModelTrainer(
        xes_adapter=_DummyAdapter(),
        prefix_policy=_DummyPrefixPolicy(),
        graph_builder=_DummyGraphBuilder(),
        model=_TrainableBinaryModel(),
        log_path="in_memory.xes",
        config={
            "mode": "train",
            "device": "cpu",
            "show_progress": False,
            "tqdm_disable": True,
            "mapping_config": {"adapter": "xes"},
            "model_config": {"type": "EOPKGGATv2"},
            "run_profile": {
                "adapter_kind": "xes",
                "model_family": "eopkg",
                "graph_features_enabled": True,
                "node_feature_count": 4,
                "edge_weight_enabled": True,
                "global_process_stats_forward_enabled": False,
                "stats_quality_gate_enabled": True,
                "stats_time_policy": "strict_asof",
                "on_missing_asof_snapshot": "disable_stats",
                "xes_use_classifier": False,
            },
            "data_config": {"dataset_label": "demo_ds"},
        },
    )

    caplog.set_level(logging.INFO)
    trainer._log_run_context()

    assert "TRAINER_PROFILE mode=train model=EOPKGGATv2 model_family=eopkg adapter=xes dataset=demo_ds" in caplog.text
    assert "TRAINER_PROFILE forward struct_nodes=on(node_features=4) struct_edges=on" in caplog.text
    assert "TRAINER_PROFILE xes use_classifier=False" in caplog.text
    assert "TRAINER_CHECKS forward_stats_summary=on mixed_snapshot_guard=on missing_asof_policy=disable_stats" in caplog.text


def test_trainer_forward_stats_logs_missing_asof_counters(caplog):
    snapshot_epoch = float(datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc).timestamp())
    loader = DataLoader(
        [
            _sample_with_missing_asof(0, snapshot_idx=7, snapshot_epoch=snapshot_epoch, missing_asof=True),
            _sample_with_missing_asof(1, snapshot_idx=7, snapshot_epoch=snapshot_epoch, missing_asof=False),
        ],
        batch_size=2,
        shuffle=False,
    )
    trainer = ModelTrainer(
        xes_adapter=_DummyAdapter(),
        prefix_policy=_DummyPrefixPolicy(),
        graph_builder=_DummyGraphBuilder(),
        model=_TrainableBinaryModel(),
        log_path="in_memory.xes",
        config={"mode": "train", "device": "cpu", "show_progress": False, "tqdm_disable": True},
    )
    trainer.criterion = nn.CrossEntropyLoss()
    optimizer = Adam(trainer.model.parameters(), lr=0.01)

    caplog.set_level(logging.INFO)
    trainer._run_epoch(loader, optimizer=optimizer, training=True)
    assert "missing_asof_snapshot_batches=1 missing_asof_snapshot[true/false]=1/1" in caplog.text


def test_trainer_forward_stats_logs_topology_projection_summary(caplog):
    loader = DataLoader(
        [
            _sample_with_projection_summary(0, aligned=True, skipped_edges=0, missing_vocab=0),
            _sample_with_projection_summary(1, aligned=False, skipped_edges=2, missing_vocab=1),
        ],
        batch_size=2,
        shuffle=False,
    )
    trainer = ModelTrainer(
        xes_adapter=_DummyAdapter(),
        prefix_policy=_DummyPrefixPolicy(),
        graph_builder=_DummyGraphBuilder(),
        model=_TrainableBinaryModel(),
        log_path="in_memory.xes",
        config={"mode": "train", "device": "cpu", "show_progress": False, "tqdm_disable": True},
    )
    trainer.criterion = nn.CrossEntropyLoss()
    optimizer = Adam(trainer.model.parameters(), lr=0.01)

    caplog.set_level(logging.INFO)
    trainer._run_epoch(loader, optimizer=optimizer, training=True)

    assert "topology_projection_aligned[true/false]=1/1" in caplog.text
    assert "topology_projection_skipped_edges=2" in caplog.text
    assert "topology_projection_missing_vocab=1" in caplog.text


def test_data_to_contract_uses_first_graph_structural_payload_from_batch():
    snapshot_epoch = float(datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc).timestamp())
    sample_a = Data(
        x_cat=torch.zeros((1, 0), dtype=torch.long),
        x_num=torch.ones((1, 1), dtype=torch.float32),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
        edge_type=torch.zeros((0,), dtype=torch.long),
        y=torch.tensor([0], dtype=torch.long),
        num_nodes=1,
        struct_x=torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float32),
        structural_edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        structural_edge_weight=torch.tensor([0.2, 0.8], dtype=torch.float32),
        stats_snapshot_version_idx=torch.tensor([10], dtype=torch.long),
        stats_snapshot_as_of_epoch=torch.tensor([snapshot_epoch], dtype=torch.float64),
    )
    sample_b = Data(
        x_cat=torch.zeros((1, 0), dtype=torch.long),
        x_num=torch.ones((1, 1), dtype=torch.float32),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
        edge_type=torch.zeros((0,), dtype=torch.long),
        y=torch.tensor([1], dtype=torch.long),
        num_nodes=1,
        struct_x=torch.tensor([[9.0], [9.0], [9.0]], dtype=torch.float32),
        structural_edge_index=torch.tensor([[0], [2]], dtype=torch.long),
        structural_edge_weight=torch.tensor([1.0], dtype=torch.float32),
        stats_snapshot_version_idx=torch.tensor([10], dtype=torch.long),
        stats_snapshot_as_of_epoch=torch.tensor([snapshot_epoch], dtype=torch.float64),
    )
    loader = DataLoader([sample_a, sample_b], batch_size=2, shuffle=False)
    batch = next(iter(loader))

    trainer = ModelTrainer(
        xes_adapter=_DummyAdapter(),
        prefix_policy=_DummyPrefixPolicy(),
        graph_builder=_DummyGraphBuilder(),
        model=_TrainableBinaryModel(),
        log_path="in_memory.xes",
        config={"mode": "train", "device": "cpu", "show_progress": False, "tqdm_disable": True},
    )
    contract = trainer._data_to_contract(batch)

    assert isinstance(contract.get("struct_x"), torch.Tensor)
    assert contract["struct_x"].shape == torch.Size([3, 1])
    assert torch.allclose(contract["struct_x"], sample_a.struct_x)
    assert isinstance(contract.get("structural_edge_index"), torch.Tensor)
    assert torch.equal(contract["structural_edge_index"], sample_a.structural_edge_index)
    assert isinstance(contract.get("structural_edge_weight"), torch.Tensor)
    assert torch.allclose(contract["structural_edge_weight"], sample_a.structural_edge_weight)


def test_data_to_contract_warns_when_batch_has_mixed_snapshot_versions():
    snapshot_epoch = float(datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc).timestamp())
    sample_a = _sample(0, snapshot_idx=7, snapshot_epoch=snapshot_epoch)
    sample_b = _sample(1, snapshot_idx=8, snapshot_epoch=snapshot_epoch)
    loader = DataLoader([sample_a, sample_b], batch_size=2, shuffle=False)
    batch = next(iter(loader))

    trainer = ModelTrainer(
        xes_adapter=_DummyAdapter(),
        prefix_policy=_DummyPrefixPolicy(),
        graph_builder=_DummyGraphBuilder(),
        model=_TrainableBinaryModel(),
        log_path="in_memory.xes",
        config={"mode": "train", "device": "cpu", "show_progress": False, "tqdm_disable": True},
    )
    trainer._idx_to_stats_snapshot_version[7] = "k000007"
    trainer._idx_to_stats_snapshot_version[8] = "k000008"

    with pytest.warns(UserWarning, match="Mixed stats snapshots in one batch detected"):
        contract = trainer._data_to_contract(batch)

    assert contract.get("stats_snapshot_versions") is not None
    assert "k000007" in contract["stats_snapshot_versions"]
    assert "k000008" in contract["stats_snapshot_versions"]


def test_trainer_numeric_guard_sanitizes_nan_logits_in_eval():
    snapshot_epoch = float(datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc).timestamp())
    loader = DataLoader(
        [
            _sample(0, snapshot_idx=7, snapshot_epoch=snapshot_epoch),
            _sample(1, snapshot_idx=7, snapshot_epoch=snapshot_epoch),
        ],
        batch_size=2,
        shuffle=False,
    )
    trainer = ModelTrainer(
        xes_adapter=_DummyAdapter(),
        prefix_policy=_DummyPrefixPolicy(),
        graph_builder=_DummyGraphBuilder(),
        model=_NaNLogitModel(),
        log_path="in_memory.xes",
        config={"mode": "train", "device": "cpu", "show_progress": False, "tqdm_disable": True},
    )
    metrics = trainer._evaluate_test(loader, stage_label="inference")
    assert math.isfinite(float(metrics["test_macro_f1"]))
    assert math.isfinite(float(metrics["test_top3_accuracy"]))
    assert math.isfinite(float(metrics["test_ece"]))


def test_trainer_numeric_guard_keeps_run_epoch_finite_with_nan_logits():
    snapshot_epoch = float(datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc).timestamp())
    loader = DataLoader(
        [
            _sample(0, snapshot_idx=7, snapshot_epoch=snapshot_epoch),
            _sample(1, snapshot_idx=7, snapshot_epoch=snapshot_epoch),
        ],
        batch_size=2,
        shuffle=False,
    )
    trainer = ModelTrainer(
        xes_adapter=_DummyAdapter(),
        prefix_policy=_DummyPrefixPolicy(),
        graph_builder=_DummyGraphBuilder(),
        model=_NaNLogitModel(),
        log_path="in_memory.xes",
        config={"mode": "train", "device": "cpu", "show_progress": False, "tqdm_disable": True},
    )
    trainer.criterion = nn.CrossEntropyLoss()
    optimizer = Adam(trainer.model.parameters(), lr=0.01)

    avg_loss, macro_f1, weighted_f1, _ = trainer._run_epoch(loader, optimizer=optimizer, training=True)
    assert math.isfinite(float(avg_loss))
    assert math.isfinite(float(macro_f1))
    assert math.isfinite(float(weighted_f1))
