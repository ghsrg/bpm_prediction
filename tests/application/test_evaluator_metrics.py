from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

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
