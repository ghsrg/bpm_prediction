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
from torch.utils.data import Dataset

from src.application.use_cases.trainer import ModelTrainer, ShardedGraphDataset, _TopologyHomogeneousBatchSampler
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


class _ClassAwareDiagnosticModel(nn.Module):
    def __init__(self, output_dim: int = 2) -> None:
        super().__init__()
        self.dummy = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.output_dim = int(output_dim)
        self.last_observed_logits: torch.Tensor | None = None
        self.last_structural_class_logits: torch.Tensor | None = None

    def forward(self, contract):
        batch = contract["batch"]
        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
        if self.output_dim == 2:
            observed_base = torch.tensor([[2.0, -2.0]], dtype=torch.float32, device=batch.device)
            structural_base = torch.tensor([[0.25, -0.75]], dtype=torch.float32, device=batch.device)
        else:
            observed_base = torch.zeros((1, self.output_dim), dtype=torch.float32, device=batch.device)
            structural_base = torch.linspace(
                -0.5,
                0.5,
                steps=self.output_dim,
                dtype=torch.float32,
                device=batch.device,
            ).view(1, -1)
        observed = observed_base.repeat(num_graphs, 1)
        structural = structural_base.repeat(num_graphs, 1)
        self.last_observed_logits = observed.detach()
        self.last_structural_class_logits = structural.detach()
        return observed + structural + (self.dummy * 0.0)


class _TopologyStateDiagnosticModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dummy = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.last_topology_state_prefix_mean_abs: torch.Tensor | None = None
        self.last_topology_state_prefix_max_abs: torch.Tensor | None = None
        self.last_topology_state_entropy: torch.Tensor | None = None
        self.last_topology_state_mean_class_cardinality: torch.Tensor | None = None
        self.last_topology_state_max_class_cardinality: torch.Tensor | None = None
        self.last_topology_state_gate_mean: torch.Tensor | None = None
        self.last_topology_state_gate_max: torch.Tensor | None = None

    def forward(self, contract):
        batch = contract["batch"]
        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
        self.last_topology_state_prefix_mean_abs = torch.tensor(0.25, dtype=torch.float32, device=batch.device)
        self.last_topology_state_prefix_max_abs = torch.tensor(1.5, dtype=torch.float32, device=batch.device)
        self.last_topology_state_entropy = torch.tensor(0.75, dtype=torch.float32, device=batch.device)
        self.last_topology_state_mean_class_cardinality = torch.tensor(1.25, dtype=torch.float32, device=batch.device)
        self.last_topology_state_max_class_cardinality = torch.tensor(2.0, dtype=torch.float32, device=batch.device)
        self.last_topology_state_gate_mean = torch.tensor(0.1, dtype=torch.float32, device=batch.device)
        self.last_topology_state_gate_max = torch.tensor(0.2, dtype=torch.float32, device=batch.device)
        logits = torch.tensor([[1.0, -1.0]], dtype=torch.float32, device=batch.device).repeat(num_graphs, 1)
        return logits + (self.dummy * 0.0)


class _TopologyGraphDiagnosticModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dummy = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.last_topology_graph_context: torch.Tensor | None = None
        self.last_topology_graph_logits: torch.Tensor | None = None
        self.last_topology_graph_entropy: torch.Tensor | None = None

    def forward(self, contract):
        batch = contract["batch"]
        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
        context = torch.tensor([[0.5, -0.25]], dtype=torch.float32, device=batch.device).repeat(num_graphs, 1)
        logits = torch.tensor([[1.0, -1.0]], dtype=torch.float32, device=batch.device).repeat(num_graphs, 1)
        self.last_topology_graph_context = context.detach()
        self.last_topology_graph_logits = logits.detach()
        self.last_topology_graph_entropy = torch.tensor(0.75, dtype=torch.float32, device=batch.device)
        return logits + (self.dummy * 0.0)


class _StructuralPriorDiagnosticModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dummy = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.last_observed_context: torch.Tensor | None = None
        self.last_structural_prior_context: torch.Tensor | None = None
        self.last_structural_prior_gate: torch.Tensor | None = None

    def forward(self, contract):
        batch = contract["batch"]
        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
        observed = torch.tensor([[2.0, -2.0]], dtype=torch.float32, device=batch.device).repeat(num_graphs, 1)
        structural = torch.tensor([[0.5, -0.5]], dtype=torch.float32, device=batch.device).repeat(num_graphs, 1)
        gate = torch.tensor([[0.25, 0.75]], dtype=torch.float32, device=batch.device).repeat(num_graphs, 1)
        self.last_observed_context = observed.detach()
        self.last_structural_prior_context = structural.detach()
        self.last_structural_prior_gate = gate.detach()
        logits = torch.tensor([[1.0, -1.0]], dtype=torch.float32, device=batch.device).repeat(num_graphs, 1)
        return logits + (self.dummy * 0.0)


class _TopologyConditionedCandidateDiagnosticModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dummy = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.last_candidate_node_score_mean_abs: float | None = None
        self.last_candidate_class_score_mean_abs: float | None = None
        self.last_duplicate_candidate_count_max: int | None = None
        self.last_candidate_temperature: float | None = None
        self.last_candidate_temperature_trainable: bool | None = None
        self.last_candidate_prediction_entropy: float | None = None
        self.last_candidate_target_score: float | None = None
        self.last_candidate_pred_score: float | None = None
        self.last_candidate_score_gap: float | None = None
        self.last_candidate_dynamic_count: int | None = None
        self.last_candidate_is_unseen: list[bool] | None = None

    def forward(self, contract):
        batch = contract["batch"]
        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
        self.last_candidate_node_score_mean_abs = 0.33
        self.last_candidate_class_score_mean_abs = 0.22
        self.last_duplicate_candidate_count_max = 2
        self.last_candidate_temperature = 0.1
        self.last_candidate_temperature_trainable = False
        self.last_candidate_prediction_entropy = 1.2
        self.last_candidate_target_score = 0.7
        self.last_candidate_pred_score = 0.9
        self.last_candidate_score_gap = 0.2
        self.last_candidate_dynamic_count = 2
        self.last_candidate_is_unseen = [False, True]
        logits = torch.tensor([[1.0, -1.0]], dtype=torch.float32, device=batch.device).repeat(num_graphs, 1)
        return logits + (self.dummy * 0.0)


class _RecordingTracker:
    def __init__(self) -> None:
        self.metrics: list[tuple[str, float, int | None]] = []

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        self.metrics.append((key, float(value), step))


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


def _make_trainer(
    *,
    model: nn.Module,
    config_overrides: dict | None = None,
    tracker: _RecordingTracker | None = None,
) -> ModelTrainer:
    config = {"mode": "train", "device": "cpu", "show_progress": False, "tqdm_disable": True}
    if config_overrides:
        config.update(config_overrides)
    trainer = ModelTrainer(
        xes_adapter=_DummyAdapter(),
        prefix_policy=_DummyPrefixPolicy(),
        graph_builder=_DummyGraphBuilder(),
        model=model,
        log_path="in_memory.xes",
        config=config,
        tracker=tracker,
    )
    trainer.criterion = nn.CrossEntropyLoss()
    return trainer


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


def test_trainer_logs_class_aware_structural_logit_contribution_metrics(caplog):
    snapshot_epoch = float(datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc).timestamp())
    loader = DataLoader(
        [
            _sample(0, snapshot_idx=7, snapshot_epoch=snapshot_epoch),
            _sample(1, snapshot_idx=7, snapshot_epoch=snapshot_epoch),
        ],
        batch_size=2,
        shuffle=False,
    )
    tracker = _RecordingTracker()
    trainer = ModelTrainer(
        xes_adapter=_DummyAdapter(),
        prefix_policy=_DummyPrefixPolicy(),
        graph_builder=_DummyGraphBuilder(),
        model=_ClassAwareDiagnosticModel(),
        log_path="in_memory.xes",
        config={"mode": "train", "device": "cpu", "show_progress": False, "tqdm_disable": True},
        tracker=tracker,
    )
    trainer.criterion = nn.CrossEntropyLoss()
    optimizer = Adam(trainer.model.parameters(), lr=0.01)

    caplog.set_level(logging.INFO)
    trainer._run_epoch(loader, optimizer=optimizer, training=True)

    assert "observed_logits_mean_abs=2.000000" in caplog.text
    assert "structural_logits_mean_abs=0.500000" in caplog.text
    assert "structural_logits_max_abs=0.750000" in caplog.text
    assert "structural_to_observed_logit_ratio=0.250000" in caplog.text
    metric_names = {key for key, _, _ in tracker.metrics}
    assert "train_observed_logits_mean_abs" in metric_names
    assert "train_structural_logits_mean_abs" in metric_names
    assert "train_structural_logits_max_abs" in metric_names
    assert "train_structural_to_observed_logit_ratio" in metric_names


def test_trainer_logs_topology_state_diagnostics(caplog):
    snapshot_epoch = float(datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc).timestamp())
    loader = DataLoader(
        [
            _sample(0, snapshot_idx=7, snapshot_epoch=snapshot_epoch),
            _sample(1, snapshot_idx=7, snapshot_epoch=snapshot_epoch),
        ],
        batch_size=2,
        shuffle=False,
    )
    tracker = _RecordingTracker()
    trainer = _make_trainer(model=_TopologyStateDiagnosticModel(), tracker=tracker)
    optimizer = Adam(trainer.model.parameters(), lr=0.01)

    caplog.set_level(logging.INFO)
    trainer._run_epoch(loader, optimizer=optimizer, training=True)

    assert "topology_state_prefix_mean_abs=0.250000" in caplog.text
    assert "topology_state_prefix_max_abs=1.500000" in caplog.text
    assert "topology_state_entropy=0.750000" in caplog.text
    assert "topology_state_mean_class_cardinality=1.250000" in caplog.text
    assert "topology_state_max_class_cardinality=2.000000" in caplog.text
    assert "topology_state_gate_mean=0.100000" in caplog.text
    assert "topology_state_gate_max=0.200000" in caplog.text
    metric_names = {key for key, _, _ in tracker.metrics}
    assert "train_topology_state_entropy" in metric_names
    assert "train_topology_state_gate_mean" in metric_names


def test_trainer_logs_topology_graph_diagnostics(caplog):
    snapshot_epoch = float(datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc).timestamp())
    loader = DataLoader(
        [
            _sample(0, snapshot_idx=7, snapshot_epoch=snapshot_epoch),
            _sample(1, snapshot_idx=7, snapshot_epoch=snapshot_epoch),
        ],
        batch_size=2,
        shuffle=False,
    )
    tracker = _RecordingTracker()
    trainer = _make_trainer(model=_TopologyGraphDiagnosticModel(), tracker=tracker)

    caplog.set_level(logging.INFO)
    trainer._evaluate_test(loader, stage_label="inference")

    assert "topology_graph_context_mean_abs=0.375000" in caplog.text
    assert "topology_graph_context_max_abs=0.500000" in caplog.text
    assert "topology_graph_logits_mean_abs=1.000000" in caplog.text
    assert "topology_graph_entropy=0.750000" in caplog.text
    metric_names = {key for key, _, _ in tracker.metrics}
    assert "inference_topology_graph_context_mean_abs" in metric_names
    assert "inference_topology_graph_logits_mean_abs" in metric_names
    assert "inference_topology_graph_entropy" in metric_names


def test_trainer_logs_structural_prior_diagnostics(caplog):
    snapshot_epoch = float(datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc).timestamp())
    loader = DataLoader(
        [
            _sample(0, snapshot_idx=7, snapshot_epoch=snapshot_epoch),
            _sample(1, snapshot_idx=7, snapshot_epoch=snapshot_epoch),
        ],
        batch_size=2,
        shuffle=False,
    )
    tracker = _RecordingTracker()
    trainer = _make_trainer(model=_StructuralPriorDiagnosticModel(), tracker=tracker)
    optimizer = Adam(trainer.model.parameters(), lr=0.01)

    caplog.set_level(logging.INFO)
    trainer._run_epoch(loader, optimizer=optimizer, training=True)

    assert "observed_context_mean_abs=2.000000" in caplog.text
    assert "structural_prior_context_mean_abs=0.500000" in caplog.text
    assert "structural_prior_to_observed_context_ratio=0.250000" in caplog.text
    assert "structural_prior_gate_mean=0.500000" in caplog.text
    assert "structural_prior_gate_max=0.750000" in caplog.text
    metric_names = {key for key, _, _ in tracker.metrics}
    assert "train_observed_context_mean_abs" in metric_names
    assert "train_structural_prior_context_mean_abs" in metric_names
    assert "train_structural_prior_to_observed_context_ratio" in metric_names
    assert "train_structural_prior_gate_mean" in metric_names
    assert "train_structural_prior_gate_max" in metric_names


def test_trainer_logs_topology_conditioned_candidate_diagnostics(caplog):
    snapshot_epoch = float(datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc).timestamp())
    loader = DataLoader(
        [
            _sample(0, snapshot_idx=7, snapshot_epoch=snapshot_epoch),
            _sample(1, snapshot_idx=7, snapshot_epoch=snapshot_epoch),
        ],
        batch_size=2,
        shuffle=False,
    )
    tracker = _RecordingTracker()
    trainer = _make_trainer(model=_TopologyConditionedCandidateDiagnosticModel(), tracker=tracker)
    optimizer = Adam(trainer.model.parameters(), lr=0.01)

    caplog.set_level(logging.INFO)
    trainer._run_epoch(loader, optimizer=optimizer, training=True)

    assert "candidate_node_score_mean_abs=0.330000" in caplog.text
    assert "candidate_class_score_mean_abs=0.220000" in caplog.text
    assert "duplicate_candidate_count_max=2.000000" in caplog.text
    assert "candidate_temperature=0.100000" in caplog.text
    assert "candidate_prediction_entropy=1.200000" in caplog.text
    assert "candidate_score_gap=0.200000" in caplog.text
    assert "candidate_dynamic_count=2.000000" in caplog.text
    assert "candidate_unseen_candidate_rate=0.500000" in caplog.text
    metric_names = {key for key, _, _ in tracker.metrics}
    assert "train_candidate_node_score_mean_abs" in metric_names
    assert "train_candidate_class_score_mean_abs" in metric_names
    assert "train_duplicate_candidate_count_max" in metric_names
    assert "train_candidate_temperature" in metric_names
    assert "train_candidate_prediction_entropy" in metric_names
    assert "train_candidate_score_gap" in metric_names
    assert "train_candidate_dynamic_count" in metric_names
    assert "train_candidate_unseen_candidate_rate" in metric_names


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


def test_trainer_parses_structural_aux_loss_enabled_string_false():
    trainer = _make_trainer(
        model=_ClassAwareDiagnosticModel(output_dim=3),
        config_overrides={"structural_aux_loss_enabled": "false"},
    )

    assert trainer.structural_aux_loss_enabled is False


def test_trainer_adds_and_logs_structural_auxiliary_loss(caplog):
    tracker = _RecordingTracker()
    trainer = _make_trainer(
        model=_ClassAwareDiagnosticModel(output_dim=3),
        config_overrides={
            "structural_aux_loss_enabled": True,
            "structural_aux_loss_weight": 0.05,
            "structural_aux_exact_loss_weight": 0.01,
        },
        tracker=tracker,
    )
    loader = [
        Data(
            x_cat=torch.zeros((2, 0), dtype=torch.long),
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

    assert "structural_aux_set_loss=" in caplog.text
    assert "structural_aux_exact_loss=" in caplog.text
    assert "structural_aux_total_loss=" in caplog.text
    metric_names = {key for key, _, _ in tracker.metrics}
    assert "validation_structural_aux_set_loss" in metric_names
    assert "validation_structural_aux_exact_loss" in metric_names
    assert "validation_structural_aux_total_loss" in metric_names


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
        struct_node_to_class_index=torch.tensor([0, 1, 2], dtype=torch.long),
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
        struct_node_to_class_index=torch.tensor([2, 1, 0], dtype=torch.long),
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
    assert isinstance(contract.get("struct_node_to_class_index"), torch.Tensor)
    assert torch.equal(contract["struct_node_to_class_index"], sample_a.struct_node_to_class_index)


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
    assert trainer.candidate_contract_mode == "fixed_label"


def test_candidate_contract_mode_defaults_to_fixed_projection_when_dynamic_enabled():
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
            "dynamic_candidate_contract_enabled": True,
        },
    )

    assert trainer.candidate_contract_mode == "fixed_projection"


def test_candidate_contract_mode_accepts_candidate_id_and_default_topology_policy():
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
            "candidate_contract_mode": "candidate_id",
        },
    )

    assert trainer.candidate_contract_mode == "candidate_id"
    assert trainer.candidate_batch_topology_policy == "single_topology_required"


def test_forward_stats_logs_candidate_target_mapping_metrics():
    tracker = _RecordingTracker()
    trainer = _make_trainer(model=_TrainableBinaryModel(), tracker=tracker)

    trainer._log_forward_stats_summary(
        "train",
        {
            "batches": 2,
            "candidate_target_mapping_batches": 2,
            "candidate_target_in_candidate_set_rate_sum": 2.0,
            "candidate_missing_target_rate_sum": 0.0,
            "candidate_target_duplicate_count_max_sum": 3.0,
            "candidate_target_set_logit_variance_mean_sum": 0.4,
            "candidate_target_set_entropy_mean_sum": 1.2,
            "candidate_unseen_target_rate_sum": 0.5,
        },
    )

    metrics = {name: value for name, value, _ in tracker.metrics}
    assert metrics["train_candidate_target_in_candidate_set_rate"] == pytest.approx(1.0)
    assert metrics["train_candidate_missing_target_rate"] == pytest.approx(0.0)
    assert metrics["train_candidate_target_duplicate_count_max"] == pytest.approx(1.5)
    assert metrics["train_candidate_target_set_logit_variance_mean"] == pytest.approx(0.2)
    assert metrics["train_candidate_target_set_entropy_mean"] == pytest.approx(0.6)
    assert metrics["train_candidate_unseen_target_rate"] == pytest.approx(0.25)


def test_topology_homogeneous_sampler_groups_by_version_and_snapshot_when_shuffling():
    snapshot_epoch = float(datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc).timestamp())
    samples = [
        _sample(0, snapshot_idx=7, snapshot_epoch=snapshot_epoch),
        _sample(1, snapshot_idx=8, snapshot_epoch=snapshot_epoch),
        _sample(0, snapshot_idx=7, snapshot_epoch=snapshot_epoch),
        _sample(1, snapshot_idx=8, snapshot_epoch=snapshot_epoch),
    ]
    samples[0].process_version_idx = torch.tensor([0], dtype=torch.long)
    samples[1].process_version_idx = torch.tensor([0], dtype=torch.long)
    samples[2].process_version_idx = torch.tensor([1], dtype=torch.long)
    samples[3].process_version_idx = torch.tensor([0], dtype=torch.long)

    sampler = _TopologyHomogeneousBatchSampler(samples, batch_size=2, shuffle=True, seed=42)

    for batch_indices in sampler:
        identities = {
            (
                int(samples[idx].process_version_idx.view(-1)[0].item()),
                int(samples[idx].stats_snapshot_version_idx.view(-1)[0].item()),
            )
            for idx in batch_indices
        }
        assert len(identities) == 1


def test_topology_homogeneous_sampler_preserves_order_when_not_shuffling():
    snapshot_epoch = float(datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc).timestamp())
    samples = [
        _sample(0, snapshot_idx=7, snapshot_epoch=snapshot_epoch),
        _sample(1, snapshot_idx=7, snapshot_epoch=snapshot_epoch),
        _sample(0, snapshot_idx=8, snapshot_epoch=snapshot_epoch),
        _sample(1, snapshot_idx=8, snapshot_epoch=snapshot_epoch),
        _sample(0, snapshot_idx=7, snapshot_epoch=snapshot_epoch),
    ]
    for sample in samples:
        sample.process_version_idx = torch.tensor([0], dtype=torch.long)

    sampler = _TopologyHomogeneousBatchSampler(samples, batch_size=3, shuffle=False, seed=42)

    assert list(sampler) == [[0, 1], [2, 3], [4]]


def test_candidate_id_dataloader_emits_topology_homogeneous_batches():
    snapshot_epoch = float(datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc).timestamp())
    samples = [
        _sample(0, snapshot_idx=7, snapshot_epoch=snapshot_epoch),
        _sample(1, snapshot_idx=8, snapshot_epoch=snapshot_epoch),
        _sample(0, snapshot_idx=7, snapshot_epoch=snapshot_epoch),
        _sample(1, snapshot_idx=8, snapshot_epoch=snapshot_epoch),
    ]
    samples[0].process_version_idx = torch.tensor([0], dtype=torch.long)
    samples[1].process_version_idx = torch.tensor([0], dtype=torch.long)
    samples[2].process_version_idx = torch.tensor([1], dtype=torch.long)
    samples[3].process_version_idx = torch.tensor([0], dtype=torch.long)

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
            "candidate_contract_mode": "candidate_id",
            "batch_size": 2,
        },
    )

    loader = trainer._create_data_loader(samples, shuffle=True)

    for batch in loader:
        identities = set(
            zip(
                batch.process_version_idx.view(-1).long().tolist(),
                batch.stats_snapshot_version_idx.view(-1).long().tolist(),
            )
        )
        assert len(identities) == 1


def test_candidate_id_dataloader_groups_dataset_sources_by_topology():
    class _DatasetSource(Dataset):
        def __init__(self, items):
            self._items = list(items)

        def __len__(self):
            return len(self._items)

        def __getitem__(self, index):
            return self._items[index]

    snapshot_epoch = float(datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc).timestamp())
    samples = [
        _sample(0, snapshot_idx=80, snapshot_epoch=snapshot_epoch),
        _sample(1, snapshot_idx=80, snapshot_epoch=snapshot_epoch),
        _sample(0, snapshot_idx=80, snapshot_epoch=snapshot_epoch),
        _sample(1, snapshot_idx=80, snapshot_epoch=snapshot_epoch),
    ]
    samples[0].process_version_idx = torch.tensor([0], dtype=torch.long)
    samples[1].process_version_idx = torch.tensor([1], dtype=torch.long)
    samples[2].process_version_idx = torch.tensor([0], dtype=torch.long)
    samples[3].process_version_idx = torch.tensor([1], dtype=torch.long)

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
            "candidate_contract_mode": "candidate_id",
            "batch_size": 2,
        },
    )

    loader = trainer._create_data_loader_from_source(_DatasetSource(samples), shuffle=True)

    assert isinstance(loader.batch_sampler, _TopologyHomogeneousBatchSampler)
    observed = []
    for batch in loader:
        versions = tuple(int(v) for v in batch.process_version_idx.view(-1).long().tolist())
        snapshots = tuple(int(v) for v in batch.stats_snapshot_version_idx.view(-1).long().tolist())
        observed.append((versions, snapshots))

    assert sorted(observed) == [
        ((0, 0), (80, 80)),
        ((1, 1), (80, 80)),
    ]


def test_topology_sampler_uses_dataset_topology_index_without_loading_graphs():
    class _IndexedDataset(Dataset):
        def __len__(self):
            return 4

        def __getitem__(self, index):
            raise AssertionError("topology sampler should not load full graph items when topology_index is available")

        def topology_index(self):
            return ("v:0|s:80", "v:1|s:80", "v:0|s:80", "v:1|s:80")

    sampler = _TopologyHomogeneousBatchSampler(_IndexedDataset(), batch_size=2, shuffle=True, seed=42)

    observed = sorted(sorted(batch) for batch in sampler)
    assert observed == [[0, 2], [1, 3]]


def test_sharded_dataset_topology_index_uses_segments_without_loading_files(tmp_path):
    dataset = ShardedGraphDataset.from_payload(
        {
            "entry_dir": str(tmp_path),
            "shards": [
                {
                    "path": "missing.pt",
                    "count": 4,
                    "topology_segments": [
                        {"key": "v:0|s:80", "count": 2},
                        {"key": "v:1|s:80", "count": 2},
                    ],
                }
            ],
        }
    )

    assert dataset.topology_index() == ("v:0|s:80", "v:0|s:80", "v:1|s:80", "v:1|s:80")


def test_candidate_id_mode_rejects_mixed_process_version_batch():
    snapshot_epoch = float(datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc).timestamp())
    sample_a = _sample(0, snapshot_idx=7, snapshot_epoch=snapshot_epoch)
    sample_b = _sample(1, snapshot_idx=7, snapshot_epoch=snapshot_epoch)
    sample_a.process_version_idx = torch.tensor([0], dtype=torch.long)
    sample_b.process_version_idx = torch.tensor([1], dtype=torch.long)
    loader = DataLoader([sample_a, sample_b], batch_size=2, shuffle=False)
    batch = next(iter(loader))

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
            "candidate_contract_mode": "candidate_id",
        },
    )

    with pytest.raises(ValueError, match="candidate_id.*mixed topology"):
        trainer._data_to_contract(batch)


def test_candidate_id_mode_rejects_mixed_stats_snapshot_batch():
    snapshot_epoch = float(datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc).timestamp())
    sample_a = _sample(0, snapshot_idx=7, snapshot_epoch=snapshot_epoch)
    sample_b = _sample(1, snapshot_idx=8, snapshot_epoch=snapshot_epoch)
    sample_a.process_version_idx = torch.tensor([0], dtype=torch.long)
    sample_b.process_version_idx = torch.tensor([0], dtype=torch.long)
    loader = DataLoader([sample_a, sample_b], batch_size=2, shuffle=False)
    batch = next(iter(loader))

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
            "candidate_contract_mode": "candidate_id",
        },
    )

    with pytest.raises(ValueError, match="candidate_id.*mixed topology"):
        trainer._data_to_contract(batch)


def test_candidate_id_mode_accepts_homogeneous_topology_batch():
    snapshot_epoch = float(datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc).timestamp())
    sample_a = _sample(0, snapshot_idx=7, snapshot_epoch=snapshot_epoch)
    sample_b = _sample(1, snapshot_idx=7, snapshot_epoch=snapshot_epoch)
    sample_a.process_version_idx = torch.tensor([0], dtype=torch.long)
    sample_b.process_version_idx = torch.tensor([0], dtype=torch.long)
    loader = DataLoader([sample_a, sample_b], batch_size=2, shuffle=False)
    batch = next(iter(loader))

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
            "candidate_contract_mode": "candidate_id",
        },
    )

    contract = trainer._data_to_contract(batch)

    assert contract.get("batch_topology_unique_count") == 1


def test_fixed_projection_mode_does_not_reject_mixed_topology_batch():
    snapshot_epoch = float(datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc).timestamp())
    sample_a = _sample(0, snapshot_idx=7, snapshot_epoch=snapshot_epoch)
    sample_b = _sample(1, snapshot_idx=8, snapshot_epoch=snapshot_epoch)
    sample_a.process_version_idx = torch.tensor([0], dtype=torch.long)
    sample_b.process_version_idx = torch.tensor([1], dtype=torch.long)
    loader = DataLoader([sample_a, sample_b], batch_size=2, shuffle=False)
    batch = next(iter(loader))

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
            "candidate_contract_mode": "fixed_projection",
        },
    )

    with pytest.warns(UserWarning, match="Mixed stats snapshots in one batch detected"):
        contract = trainer._data_to_contract(batch)

    assert contract.get("batch_topology_unique_count") is None


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


def test_trainer_target_labels_extraction_collated_by_pyg():
    trainer = ModelTrainer(
        xes_adapter=_DummyAdapter(),
        prefix_policy=_DummyPrefixPolicy(),
        graph_builder=_DummyGraphBuilder(),
        model=_TrainableBinaryModel(),
        log_path="in_memory.xes",
        config={"mode": "train", "device": "cpu", "show_progress": False, "tqdm_disable": True},
    )
    
    # Simulate a collated target_label list from PyG DataLoader for batch_size=2
    contract = {
        "target_label": ["Approve", "Decline"],
        "candidate_labels": [("Approve", "Decline", "End"), ("Approve", "Decline", "End")],
        "candidate_ids": [("node1", "node2", "node3"), ("node1", "node2", "node3")],
    }
    
    targets = torch.tensor([0, 1], dtype=torch.long)
    
    extracted_targets = trainer._target_labels_from_contract(contract, targets)
    assert extracted_targets == ["Approve", "Decline"]
