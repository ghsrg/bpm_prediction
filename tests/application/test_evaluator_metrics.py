from __future__ import annotations

import warnings
import numpy as np
import pytest
import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.exceptions import UndefinedMetricWarning

from src.domain.entities.candidate_prediction import CandidatePredictionOutput
from src.application.use_cases.trainer import ModelTrainer


def test_oos_calculation():
    y_hat = torch.tensor([0, 1, 2, 1], dtype=torch.long)
    allowed_mask = torch.tensor(
        [
            [True, False, False],   # pred=0 -> in-sequence
            [False, False, True],   # pred=1 -> OOS
            [False, False, True],   # pred=2 -> in-sequence
            [True, True, False],    # pred=1 -> in-sequence
        ],
        dtype=torch.bool,
    )

    oos_flags = ModelTrainer._compute_oos_flags(y_hat, allowed_mask)
    oos_rate = float(oos_flags.mean().item())
    assert oos_rate == 0.25


def test_evaluator_slicing_logic():
    y_true = np.asarray([0, 1, 0, 1, 2, 2], dtype=np.int64)
    y_pred = np.asarray([0, 1, 1, 1, 2, 0], dtype=np.int64)
    prefix_lengths = np.asarray([2, 7, 12, 25, 4, 9], dtype=np.int64)
    versions = ["v1", "v1", "v2", "v2", "v1", "v2"]

    metrics_no_oos = ModelTrainer._compute_sliced_metrics(
        y_true=y_true,
        y_pred=y_pred,
        oos_flags=None,
        target_in_mask_flags=None,
        pred_in_mask_flags=None,
        strict_error_but_allowed_flags=None,
        mask_cardinality=None,
        prefix_lengths=prefix_lengths,
        versions=versions,
    )
    assert "test_f1_len_1_5" in metrics_no_oos
    assert "test_f1_len_6_10" in metrics_no_oos
    assert "test_f1_len_11_20" in metrics_no_oos
    assert "test_f1_len_21_plus" in metrics_no_oos
    assert "test_f1_v1" in metrics_no_oos
    assert "test_f1_v2" in metrics_no_oos
    assert "test_oos_v1" not in metrics_no_oos
    assert "test_oos_len_1_5" not in metrics_no_oos

    oos_flags = np.asarray([0.0, 1.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float32)
    metrics_with_oos = ModelTrainer._compute_sliced_metrics(
        y_true=y_true,
        y_pred=y_pred,
        oos_flags=oos_flags,
        target_in_mask_flags=None,
        pred_in_mask_flags=None,
        strict_error_but_allowed_flags=None,
        mask_cardinality=None,
        prefix_lengths=prefix_lengths,
        versions=versions,
    )
    assert "test_oos_len_1_5" in metrics_with_oos
    assert "test_oos_v1" in metrics_with_oos
    assert metrics_with_oos["test_oos_v1"] >= 0.0
    assert metrics_with_oos["test_oos_v1"] <= 1.0


def test_evaluator_mask_cardinality_slices_and_rates():
    y_true = np.asarray([0, 1, 1, 2, 2], dtype=np.int64)
    y_pred = np.asarray([0, 0, 1, 2, 0], dtype=np.int64)
    oos_flags = np.asarray([0.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float32)
    target_in_mask_flags = np.asarray([1.0, 1.0, 1.0, 1.0, 0.0], dtype=np.float32)
    pred_in_mask_flags = np.asarray([1.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float32)
    strict_error_but_allowed_flags = np.asarray([0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    mask_cardinality = np.asarray([1, 2, 2, 3, 1], dtype=np.float32)

    metrics = ModelTrainer._compute_sliced_metrics(
        y_true=y_true,
        y_pred=y_pred,
        oos_flags=oos_flags,
        target_in_mask_flags=target_in_mask_flags,
        pred_in_mask_flags=pred_in_mask_flags,
        strict_error_but_allowed_flags=strict_error_but_allowed_flags,
        mask_cardinality=mask_cardinality,
        prefix_lengths=None,
        versions=None,
    )

    assert "test_f1_mask_card_1" in metrics
    assert "test_f1_mask_card_2" in metrics
    assert "test_f1_mask_card_3_plus" in metrics
    assert "test_target_in_mask_rate_mask_card_2" in metrics
    assert "test_pred_in_mask_rate_mask_card_2" in metrics
    assert "test_strict_error_but_allowed_rate_mask_card_2" in metrics
    assert metrics["test_strict_error_but_allowed_rate_mask_card_2"] == 0.5


class _DummyAdapter:
    def read(self, file_path: str, mapping_config: dict):
        _ = file_path
        _ = mapping_config
        return iter([])


class _DummyPrefixPolicy:
    def generate_slices(self, trace):
        _ = trace
        return []


class _DummyGraphBuilder:
    def build_graph(self, prefix):
        _ = prefix
        raise RuntimeError("Not used in this test.")


class _ConstantClassZero3(nn.Module):
    def forward(self, contract):
        batch = contract["batch"]
        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
        logits = torch.zeros((num_graphs, 3), dtype=torch.float32, device=batch.device)
        logits[:, 0] = 5.0
        return logits


class _ParamTracker:
    def __init__(self) -> None:
        self.params = {}

    def log_param(self, key, value):
        self.params[key] = value


class _DynamicCandidateModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dummy = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.output_dim = 3
        self.forward_called = False
        self.forward_candidate_called = False

    def forward(self, contract):
        self.forward_called = True
        batch = contract["batch"]
        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
        return torch.zeros((num_graphs, self.output_dim), dtype=torch.float32, device=batch.device)

    def forward_candidate(self, contract):
        self.forward_candidate_called = True
        batch = contract["batch"]
        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
        candidate_logits = torch.tensor([[0.0, 5.0]], dtype=torch.float32, device=batch.device).repeat(num_graphs, 1)
        candidate_logits = candidate_logits + (self.dummy * 0.0)
        return CandidatePredictionOutput(
            candidate_logits=candidate_logits,
            candidate_class_index=torch.tensor([1, 2], dtype=torch.long, device=batch.device),
            node_logits=candidate_logits,
            node_to_candidate_index=torch.tensor([0, 1], dtype=torch.long, device=batch.device),
            node_to_class_index=torch.tensor([1, 2], dtype=torch.long, device=batch.device),
        )


class _UnseenDynamicCandidateModel(_DynamicCandidateModel):
    def forward_candidate(self, contract):
        self.forward_candidate_called = True
        batch = contract["batch"]
        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
        candidate_logits = torch.tensor([[0.0, 5.0]], dtype=torch.float32, device=batch.device).repeat(num_graphs, 1)
        candidate_logits = candidate_logits + (self.dummy * 0.0)
        return CandidatePredictionOutput(
            candidate_logits=candidate_logits,
            candidate_class_index=torch.tensor([1, -1], dtype=torch.long, device=batch.device),
            node_logits=candidate_logits,
            node_to_candidate_index=torch.tensor([0, 1], dtype=torch.long, device=batch.device),
            node_to_class_index=torch.tensor([1, -1], dtype=torch.long, device=batch.device),
            candidate_ids=("node_known", "node_new"),
            candidate_labels=("known_task", "new_task"),
            candidate_is_unseen=torch.tensor([False, True], dtype=torch.bool, device=batch.device),
        )


def test_evaluate_test_reports_stage2_mask_metrics():
    trainer = ModelTrainer(
        xes_adapter=_DummyAdapter(),
        prefix_policy=_DummyPrefixPolicy(),
        graph_builder=_DummyGraphBuilder(),
        model=_ConstantClassZero3(),
        log_path="in_memory.xes",
        config={
            "mode": "eval_drift",
            "device": "cpu",
            "show_progress": False,
            "tqdm_disable": True,
            "experiment_config": {"drift_window_size": 5},
        },
    )

    samples = [
        Data(
            x_cat=torch.zeros((1, 0), dtype=torch.long),
            x_num=torch.ones((1, 1), dtype=torch.float32),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            edge_type=torch.zeros((0,), dtype=torch.long),
            y=torch.tensor([0], dtype=torch.long),
            num_nodes=1,
            allowed_target_mask=torch.tensor([[True, False, False]], dtype=torch.bool),
        ),
        Data(
            x_cat=torch.zeros((1, 0), dtype=torch.long),
            x_num=torch.ones((1, 1), dtype=torch.float32),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            edge_type=torch.zeros((0,), dtype=torch.long),
            y=torch.tensor([1], dtype=torch.long),
            num_nodes=1,
            allowed_target_mask=torch.tensor([[True, True, False]], dtype=torch.bool),
        ),
        Data(
            x_cat=torch.zeros((1, 0), dtype=torch.long),
            x_num=torch.ones((1, 1), dtype=torch.float32),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            edge_type=torch.zeros((0,), dtype=torch.long),
            y=torch.tensor([1], dtype=torch.long),
            num_nodes=1,
            allowed_target_mask=torch.tensor([[False, True, False]], dtype=torch.bool),
        ),
    ]
    loader = DataLoader(samples, batch_size=3, shuffle=False)
    metrics = trainer._evaluate_test(loader)

    assert metrics["test_target_in_mask_rate"] == pytest.approx(1.0)
    assert metrics["test_pred_in_mask_rate"] == pytest.approx(2.0 / 3.0)
    assert metrics["test_strict_error_but_allowed_rate"] == pytest.approx(1.0 / 3.0)
    assert metrics["test_ambiguous_prefix_rate"] == pytest.approx(1.0 / 3.0)
    assert metrics["test_mask_coverage"] == pytest.approx(1.0)
    assert metrics["test_oos"] == pytest.approx(1.0 / 3.0)
    assert metrics["strict_test_accuracy"] == pytest.approx(1.0 / 3.0)
    assert metrics["test_accuracy"] == pytest.approx(2.0 / 3.0)
    assert metrics["test_set_hit_rate_ambiguous"] == pytest.approx(1.0)
    assert metrics["test_set_nll"] >= 0.0


def test_evaluate_test_can_consume_dynamic_candidate_contract():
    model = _DynamicCandidateModel()
    trainer = ModelTrainer(
        xes_adapter=_DummyAdapter(),
        prefix_policy=_DummyPrefixPolicy(),
        graph_builder=_DummyGraphBuilder(),
        model=model,
        log_path="in_memory.xes",
        config={
            "mode": "eval_drift",
            "device": "cpu",
            "show_progress": False,
            "tqdm_disable": True,
            "dynamic_candidate_contract_enabled": True,
        },
    )

    samples = [
        Data(
            x_cat=torch.zeros((1, 0), dtype=torch.long),
            x_num=torch.ones((1, 1), dtype=torch.float32),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            edge_type=torch.zeros((0,), dtype=torch.long),
            y=torch.tensor([2], dtype=torch.long),
            num_nodes=1,
        ),
        Data(
            x_cat=torch.zeros((1, 0), dtype=torch.long),
            x_num=torch.ones((1, 1), dtype=torch.float32),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            edge_type=torch.zeros((0,), dtype=torch.long),
            y=torch.tensor([2], dtype=torch.long),
            num_nodes=1,
        ),
    ]

    metrics = trainer._evaluate_test(DataLoader(samples, batch_size=2, shuffle=False))

    assert model.forward_candidate_called is True
    assert model.forward_called is False
    assert metrics["strict_test_accuracy"] == pytest.approx(1.0)
    assert metrics["strict_test_macro_f1"] == pytest.approx(1.0)


def test_train_epoch_can_consume_dynamic_candidate_contract():
    model = _DynamicCandidateModel()
    trainer = ModelTrainer(
        xes_adapter=_DummyAdapter(),
        prefix_policy=_DummyPrefixPolicy(),
        graph_builder=_DummyGraphBuilder(),
        model=model,
        log_path="in_memory.xes",
        config={
            "mode": "train",
            "device": "cpu",
            "show_progress": False,
            "tqdm_disable": True,
            "dynamic_candidate_contract_enabled": True,
        },
    )
    trainer.criterion = nn.CrossEntropyLoss()
    samples = [
        Data(
            x_cat=torch.zeros((1, 0), dtype=torch.long),
            x_num=torch.ones((1, 1), dtype=torch.float32),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            edge_type=torch.zeros((0,), dtype=torch.long),
            y=torch.tensor([2], dtype=torch.long),
            num_nodes=1,
        )
    ]

    loss, macro_f1, _, _ = trainer._run_epoch(
        DataLoader(samples, batch_size=1, shuffle=False),
        optimizer=torch.optim.Adam(model.parameters(), lr=0.01),
        training=True,
    )

    assert model.forward_candidate_called is True
    assert model.forward_called is False
    assert loss >= 0.0
    assert macro_f1 == pytest.approx(1.0)


def test_train_epoch_candidate_id_uses_candidate_set_loss_without_fixed_projection():
    model = _DynamicCandidateModel()
    trainer = ModelTrainer(
        xes_adapter=_DummyAdapter(),
        prefix_policy=_DummyPrefixPolicy(),
        graph_builder=_DummyGraphBuilder(),
        model=model,
        log_path="in_memory.xes",
        config={
            "mode": "train",
            "device": "cpu",
            "show_progress": False,
            "tqdm_disable": True,
            "candidate_contract_mode": "candidate_id",
            "candidate_missing_target_fail_threshold": 1.0,
        },
    )
    trainer.criterion = nn.CrossEntropyLoss()
    samples = [
        Data(
            x_cat=torch.zeros((1, 0), dtype=torch.long),
            x_num=torch.ones((1, 1), dtype=torch.float32),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            edge_type=torch.zeros((0,), dtype=torch.long),
            y=torch.tensor([2], dtype=torch.long),
            num_nodes=1,
            process_version_idx=torch.tensor([0], dtype=torch.long),
            stats_snapshot_version_idx=torch.tensor([1], dtype=torch.long),
        )
    ]

    loss, macro_f1, _, _ = trainer._run_epoch(
        DataLoader(samples, batch_size=1, shuffle=False),
        optimizer=torch.optim.Adam(model.parameters(), lr=0.01),
        training=True,
    )

    assert model.forward_candidate_called is True
    assert model.forward_called is False
    assert loss >= 0.0
    assert macro_f1 == pytest.approx(1.0)


def test_train_epoch_candidate_id_topology_native_f1_uses_target_label_for_unseen_candidate():
    model = _UnseenDynamicCandidateModel()
    trainer = ModelTrainer(
        xes_adapter=_DummyAdapter(),
        prefix_policy=_DummyPrefixPolicy(),
        graph_builder=_DummyGraphBuilder(),
        model=model,
        log_path="in_memory.xes",
        config={
            "mode": "train",
            "device": "cpu",
            "show_progress": False,
            "tqdm_disable": True,
            "candidate_contract_mode": "candidate_id",
            "candidate_identity_mode": "topology_native",
            "candidate_missing_target_fail_threshold": 1.0,
        },
    )
    samples = [
        Data(
            x_cat=torch.zeros((1, 0), dtype=torch.long),
            x_num=torch.ones((1, 1), dtype=torch.float32),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            edge_type=torch.zeros((0,), dtype=torch.long),
            y=torch.tensor([0], dtype=torch.long),
            num_nodes=1,
            target_label="new_task",
            process_version_idx=torch.tensor([0], dtype=torch.long),
            stats_snapshot_version_idx=torch.tensor([1], dtype=torch.long),
        )
    ]

    loss, macro_f1, weighted_f1, _ = trainer._run_epoch(
        DataLoader(samples, batch_size=1, shuffle=False),
        optimizer=torch.optim.Adam(model.parameters(), lr=0.01),
        training=True,
    )

    assert loss >= 0.0
    assert macro_f1 == pytest.approx(1.0)
    assert weighted_f1 == pytest.approx(1.0)


def test_train_epoch_candidate_id_topology_native_f1_matches_xes_lifecycle_target_alias():
    model = _UnseenDynamicCandidateModel()
    trainer = ModelTrainer(
        xes_adapter=_DummyAdapter(),
        prefix_policy=_DummyPrefixPolicy(),
        graph_builder=_DummyGraphBuilder(),
        model=model,
        log_path="in_memory.xes",
        config={
            "mode": "train",
            "device": "cpu",
            "show_progress": False,
            "tqdm_disable": True,
            "candidate_contract_mode": "candidate_id",
            "candidate_identity_mode": "topology_native",
            "candidate_missing_target_fail_threshold": 1.0,
        },
    )
    samples = [
        Data(
            x_cat=torch.zeros((1, 0), dtype=torch.long),
            x_num=torch.ones((1, 1), dtype=torch.float32),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            edge_type=torch.zeros((0,), dtype=torch.long),
            y=torch.tensor([0], dtype=torch.long),
            num_nodes=1,
            target_label="new_task+COMPLETE",
            process_version_idx=torch.tensor([0], dtype=torch.long),
            stats_snapshot_version_idx=torch.tensor([1], dtype=torch.long),
        )
    ]

    loss, macro_f1, weighted_f1, _ = trainer._run_epoch(
        DataLoader(samples, batch_size=1, shuffle=False),
        optimizer=torch.optim.Adam(model.parameters(), lr=0.01),
        training=True,
    )

    assert loss >= 0.0
    assert macro_f1 == pytest.approx(1.0)
    assert weighted_f1 == pytest.approx(1.0)


def test_dry_run_candidate_id_uses_candidate_forward_path():
    model = _DynamicCandidateModel()
    trainer = ModelTrainer(
        xes_adapter=_DummyAdapter(),
        prefix_policy=_DummyPrefixPolicy(),
        graph_builder=_DummyGraphBuilder(),
        model=model,
        log_path="in_memory.xes",
        config={
            "mode": "train",
            "device": "cpu",
            "show_progress": False,
            "tqdm_disable": True,
            "candidate_contract_mode": "candidate_id",
        },
    )
    samples = [
        Data(
            x_cat=torch.zeros((1, 0), dtype=torch.long),
            x_num=torch.ones((1, 1), dtype=torch.float32),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            edge_type=torch.zeros((0,), dtype=torch.long),
            y=torch.tensor([2], dtype=torch.long),
            num_nodes=1,
            process_version_idx=torch.tensor([0], dtype=torch.long),
            stats_snapshot_version_idx=torch.tensor([1], dtype=torch.long),
        )
    ]

    trainer._perform_dry_run(DataLoader(samples, batch_size=1, shuffle=False), context_label="train")

    assert model.forward_candidate_called is True
    assert model.forward_called is False


def test_evaluate_test_candidate_id_reports_global_metrics_from_candidate_predictions():
    model = _DynamicCandidateModel()
    trainer = ModelTrainer(
        xes_adapter=_DummyAdapter(),
        prefix_policy=_DummyPrefixPolicy(),
        graph_builder=_DummyGraphBuilder(),
        model=model,
        log_path="in_memory.xes",
        config={
            "mode": "eval_drift",
            "device": "cpu",
            "show_progress": False,
            "tqdm_disable": True,
            "candidate_contract_mode": "candidate_id",
        },
    )
    samples = [
        Data(
            x_cat=torch.zeros((1, 0), dtype=torch.long),
            x_num=torch.ones((1, 1), dtype=torch.float32),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            edge_type=torch.zeros((0,), dtype=torch.long),
            y=torch.tensor([2], dtype=torch.long),
            num_nodes=1,
            process_version_idx=torch.tensor([0], dtype=torch.long),
            stats_snapshot_version_idx=torch.tensor([1], dtype=torch.long),
        )
    ]

    metrics = trainer._evaluate_test(DataLoader(samples, batch_size=1, shuffle=False))

    assert model.forward_candidate_called is True
    assert model.forward_called is False
    assert metrics["strict_test_accuracy"] == pytest.approx(1.0)
    assert metrics["candidate_target_in_candidate_set_rate"] == pytest.approx(1.0)


def test_evaluate_test_candidate_id_topology_native_primary_metrics_use_unseen_candidate_space():
    model = _UnseenDynamicCandidateModel()
    trainer = ModelTrainer(
        xes_adapter=_DummyAdapter(),
        prefix_policy=_DummyPrefixPolicy(),
        graph_builder=_DummyGraphBuilder(),
        model=model,
        log_path="in_memory.xes",
        config={
            "mode": "eval_drift",
            "device": "cpu",
            "show_progress": False,
            "tqdm_disable": True,
            "candidate_contract_mode": "candidate_id",
            "candidate_identity_mode": "topology_native",
        },
    )
    samples = [
        Data(
            x_cat=torch.zeros((1, 0), dtype=torch.long),
            x_num=torch.ones((1, 1), dtype=torch.float32),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            edge_type=torch.zeros((0,), dtype=torch.long),
            y=torch.tensor([0], dtype=torch.long),
            num_nodes=1,
            target_label="new_task",
            process_version_idx=torch.tensor([0], dtype=torch.long),
            stats_snapshot_version_idx=torch.tensor([1], dtype=torch.long),
        )
    ]

    metrics = trainer._evaluate_test(DataLoader(samples, batch_size=1, shuffle=False))

    assert metrics["strict_test_accuracy"] == pytest.approx(1.0)
    assert metrics["strict_test_macro_f1"] == pytest.approx(1.0)
    assert metrics["test_accuracy"] == pytest.approx(1.0)
    assert metrics["fixed_label_strict_test_accuracy"] == pytest.approx(0.0)
    assert metrics["fixed_label_strict_test_macro_f1"] == pytest.approx(0.0)
    assert metrics["candidate_target_in_candidate_set_rate"] == pytest.approx(1.0)


def test_evaluate_test_candidate_id_reports_candidate_flow_metrics():
    model = _DynamicCandidateModel()
    trainer = ModelTrainer(
        xes_adapter=_DummyAdapter(),
        prefix_policy=_DummyPrefixPolicy(),
        graph_builder=_DummyGraphBuilder(),
        model=model,
        log_path="in_memory.xes",
        config={
            "mode": "eval_drift",
            "device": "cpu",
            "show_progress": False,
            "tqdm_disable": True,
            "candidate_contract_mode": "candidate_id",
        },
    )
    samples = [
        Data(
            x_cat=torch.zeros((1, 0), dtype=torch.long),
            x_num=torch.ones((1, 1), dtype=torch.float32),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            edge_type=torch.zeros((0,), dtype=torch.long),
            y=torch.tensor([2], dtype=torch.long),
            num_nodes=1,
            allowed_target_mask=torch.tensor([[False, True, False]], dtype=torch.bool),
            process_version_idx=torch.tensor([0], dtype=torch.long),
            stats_snapshot_version_idx=torch.tensor([1], dtype=torch.long),
        )
    ]

    metrics = trainer._evaluate_test(DataLoader(samples, batch_size=1, shuffle=False))

    assert metrics["candidate_oos_rate"] == pytest.approx(1.0)
    assert metrics["candidate_invalid_probability_mass"] > 0.5
    assert metrics["candidate_valid_invalid_logit_margin"] < 0.0


def test_evaluate_test_uses_meaningful_topk_when_class_count_is_three():
    trainer = ModelTrainer(
        xes_adapter=_DummyAdapter(),
        prefix_policy=_DummyPrefixPolicy(),
        graph_builder=_DummyGraphBuilder(),
        model=_ConstantClassZero3(),
        log_path="in_memory.xes",
        config={
            "mode": "eval_drift",
            "device": "cpu",
            "show_progress": False,
            "tqdm_disable": True,
        },
    )
    samples = [
        Data(
            x_cat=torch.zeros((1, 0), dtype=torch.long),
            x_num=torch.ones((1, 1), dtype=torch.float32),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            edge_type=torch.zeros((0,), dtype=torch.long),
            y=torch.tensor([0], dtype=torch.long),
            num_nodes=1,
        ),
        Data(
            x_cat=torch.zeros((1, 0), dtype=torch.long),
            x_num=torch.ones((1, 1), dtype=torch.float32),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            edge_type=torch.zeros((0,), dtype=torch.long),
            y=torch.tensor([1], dtype=torch.long),
            num_nodes=1,
        ),
    ]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        metrics = trainer._evaluate_test(DataLoader(samples, batch_size=2, shuffle=False))

    assert not any(isinstance(item.message, UndefinedMetricWarning) for item in caught)
    assert 0.0 <= float(metrics["test_top3_accuracy"]) <= 1.0


def test_mask_guided_policy_hard_when_reliable_and_soft_when_not():
    trainer = ModelTrainer(
        xes_adapter=_DummyAdapter(),
        prefix_policy=_DummyPrefixPolicy(),
        graph_builder=_DummyGraphBuilder(),
        model=_ConstantClassZero3(),
        log_path="in_memory.xes",
        config={
            "mode": "train",
            "device": "cpu",
            "show_progress": False,
            "tqdm_disable": True,
            "mask_guided_enabled": True,
            "mask_guided_hard_threshold": 1.0,
            "mask_guided_min_samples_for_hard": 1,
        },
    )

    assert trainer._resolve_mask_guided_policy(
        training=True,
        batch_target_in_mask_rate=1.0,
        batch_samples=4,
    ) == "hard"
    assert trainer._resolve_mask_guided_policy(
        training=True,
        batch_target_in_mask_rate=0.99,
        batch_samples=4,
    ) == "soft"

    trainer._mask_guided_reliability_rate = 1.0
    trainer._mask_guided_reliability_samples = 10
    assert trainer._resolve_mask_guided_policy(
        training=False,
        batch_target_in_mask_rate=None,
        batch_samples=4,
    ) == "hard"

    trainer._mask_guided_reliability_rate = 0.95
    trainer._mask_guided_reliability_samples = 10
    assert trainer._resolve_mask_guided_policy(
        training=False,
        batch_target_in_mask_rate=None,
        batch_samples=4,
    ) == "soft"


def test_mask_guided_policy_can_force_hard_in_eval_without_reliability_state():
    trainer = ModelTrainer(
        xes_adapter=_DummyAdapter(),
        prefix_policy=_DummyPrefixPolicy(),
        graph_builder=_DummyGraphBuilder(),
        model=_ConstantClassZero3(),
        log_path="in_memory.xes",
        config={
            "mode": "eval_drift",
            "device": "cpu",
            "show_progress": False,
            "tqdm_disable": True,
            "mask_guided_enabled": True,
            "mask_guided_policy": "hard",
            "mask_guided_apply_in_eval": True,
        },
    )

    assert trainer._resolve_mask_guided_policy(
        training=False,
        batch_target_in_mask_rate=None,
        batch_samples=4,
    ) == "hard"


def test_trainer_logs_configured_tracking_model_type_for_mask_baseline():
    tracker = _ParamTracker()
    trainer = ModelTrainer(
        xes_adapter=_DummyAdapter(),
        prefix_policy=_DummyPrefixPolicy(),
        graph_builder=_DummyGraphBuilder(),
        model=_ConstantClassZero3(),
        log_path="in_memory.xes",
        tracker=tracker,
        config={
            "mode": "eval_drift",
            "device": "cpu",
            "show_progress": False,
            "tqdm_disable": True,
            "tracking_model_type": "BaselineGATv2Mask",
            "mask_guided_enabled": True,
            "mask_guided_policy": "hard",
            "mask_guided_apply_in_eval": True,
        },
    )

    trainer._log_params()

    assert tracker.params["model_type"] == "BaselineGATv2Mask"


def test_mask_guided_logits_hard_and_soft_behaviour():
    trainer = ModelTrainer(
        xes_adapter=_DummyAdapter(),
        prefix_policy=_DummyPrefixPolicy(),
        graph_builder=_DummyGraphBuilder(),
        model=_ConstantClassZero3(),
        log_path="in_memory.xes",
        config={
            "mode": "train",
            "device": "cpu",
            "show_progress": False,
            "tqdm_disable": True,
            "mask_guided_enabled": True,
            "mask_guided_soft_penalty": 10.0,
        },
    )

    logits = torch.tensor([[5.0, 0.0, 0.0]], dtype=torch.float32)
    mask = torch.tensor([[False, True, True]], dtype=torch.bool)

    hard_logits = trainer._apply_mask_guided_logits(
        logits=logits,
        allowed_mask=mask,
        policy="hard",
    )
    hard_pred = int(torch.argmax(hard_logits, dim=1).item())
    assert hard_pred in {1, 2}

    soft_logits = trainer._apply_mask_guided_logits(
        logits=logits,
        allowed_mask=mask,
        policy="soft",
    )
    soft_pred = int(torch.argmax(soft_logits, dim=1).item())
    assert soft_pred in {1, 2}


def test_mask_guided_hard_post_filter_changes_fixed_vocab_prediction_to_allowed_class():
    trainer = ModelTrainer(
        xes_adapter=_DummyAdapter(),
        prefix_policy=_DummyPrefixPolicy(),
        graph_builder=_DummyGraphBuilder(),
        model=_ConstantClassZero3(),
        log_path="in_memory.xes",
        config={
            "mode": "eval_drift",
            "device": "cpu",
            "show_progress": False,
            "tqdm_disable": True,
            "mask_guided_enabled": True,
            "mask_guided_policy": "hard",
            "mask_guided_apply_in_eval": True,
        },
    )

    logits = torch.tensor([[9.0, 1.0, 0.5]], dtype=torch.float32)
    allowed_mask = torch.tensor([[False, True, True]], dtype=torch.bool)
    policy = trainer._resolve_mask_guided_policy(
        training=False,
        batch_target_in_mask_rate=None,
        batch_samples=1,
    )

    effective_logits = trainer._apply_mask_guided_logits(
        logits=logits,
        allowed_mask=allowed_mask,
        policy=policy,
    )

    assert int(torch.argmax(logits, dim=1).item()) == 0
    assert int(torch.argmax(effective_logits, dim=1).item()) == 1


def test_evaluate_test_fixed_head_primary_metrics_use_target_label_for_unseen_future_activity():
    class _KnownOnlyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.output_dim = 3

        def forward(self, contract):
            batch = contract["batch"]
            num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
            logits = torch.full((num_graphs, 3), -5.0, dtype=torch.float32, device=batch.device)
            logits[:, 0] = 5.0
            return logits

    trainer = ModelTrainer(
        xes_adapter=_DummyAdapter(),
        prefix_policy=_DummyPrefixPolicy(),
        graph_builder=_DummyGraphBuilder(),
        model=_KnownOnlyModel(),
        log_path="in_memory.xes",
        config={
            "mode": "eval_drift",
            "device": "cpu",
            "show_progress": False,
            "tqdm_disable": True,
        },
        prepared_data={
            "reverse_activity_vocab": {
                0: "<UNK>",
                1: "known_task",
                2: "other_task",
            }
        },
    )
    sample = Data(
        x_cat=torch.zeros((1, 0), dtype=torch.long),
        x_num=torch.ones((1, 1), dtype=torch.float32),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
        edge_type=torch.zeros((0,), dtype=torch.long),
        y=torch.tensor([0], dtype=torch.long),
        num_nodes=1,
        target_label="new_future_task",
        process_version_idx=torch.tensor([4], dtype=torch.long),
    )

    metrics = trainer._evaluate_test(DataLoader([sample], batch_size=1, shuffle=False))

    assert metrics["strict_test_accuracy"] == pytest.approx(0.0)
    assert metrics["strict_test_macro_f1"] == pytest.approx(0.0)
    assert metrics["test_ece"] > 0.99
    assert metrics["test_set_nll"] > 20.0
    assert metrics["fixed_label_strict_test_accuracy"] == pytest.approx(1.0)
    assert metrics["fixed_label_test_ece"] < 0.01
    assert metrics["fixed_label_test_set_nll"] < 0.01
