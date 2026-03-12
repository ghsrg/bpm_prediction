from __future__ import annotations

import numpy as np
import torch

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
        prefix_lengths=prefix_lengths,
        versions=versions,
    )
    assert "test_oos_len_1_5" in metrics_with_oos
    assert "test_oos_v1" in metrics_with_oos
    assert metrics_with_oos["test_oos_v1"] >= 0.0
    assert metrics_with_oos["test_oos_v1"] <= 1.0

