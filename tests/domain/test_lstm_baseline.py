from __future__ import annotations

import pytest
import torch

from src.domain.entities.feature_config import FeatureLayout
from src.domain.models.factory import create_model, get_registered_models


def _layout() -> FeatureLayout:
    return FeatureLayout(
        cat_features={"concept:name": 8, "org:resource": 4},
        cat_feature_names=["concept:name", "org:resource"],
        num_dim=3,
    )


def _contract() -> dict:
    return {
        "x_cat": torch.tensor(
            [
                [1, 2],
                [2, 1],
                [3, 1],
                [4, 0],
                [5, 2],
            ],
            dtype=torch.long,
        ),
        "x_num": torch.tensor(
            [
                [0.1, 0.2, 0.3],
                [0.2, 0.3, 0.4],
                [0.3, 0.4, 0.5],
                [1.0, 0.0, 0.1],
                [1.1, 0.1, 0.2],
            ],
            dtype=torch.float32,
        ),
        "edge_index": torch.tensor([[0, 1, 3], [1, 2, 4]], dtype=torch.long),
        "batch": torch.tensor([0, 0, 0, 1, 1], dtype=torch.long),
        "y": torch.tensor([1, 2], dtype=torch.long),
        "num_nodes": 5,
    }


@pytest.mark.parametrize("recurrent_type", ["lstm", "gru"])
def test_lstm_baseline_forward_returns_fixed_vocab_logits(recurrent_type: str):
    model = create_model(
        "LSTM_Baseline",
        feature_layout=_layout(),
        hidden_dim=16,
        output_dim=5,
        dropout=0.0,
        pooling_strategy="last_node",
        recurrent_type=recurrent_type,
        recurrent_layers=1,
        recurrent_bidirectional=False,
    )

    logits = model(_contract())

    assert tuple(logits.shape) == (2, 5)
    assert logits.dtype == torch.float32
    assert torch.isfinite(logits).all()


def test_lstm_baseline_supports_global_mean_pooling():
    model = create_model(
        "LSTM_Baseline",
        feature_layout=_layout(),
        hidden_dim=16,
        output_dim=5,
        dropout=0.0,
        pooling_strategy="global_mean",
        recurrent_type="lstm",
        recurrent_layers=1,
        recurrent_bidirectional=False,
    )

    logits = model(_contract())

    assert tuple(logits.shape) == (2, 5)


def test_lstm_baseline_is_registered():
    assert "LSTM_Baseline" in get_registered_models()


def test_lstm_baseline_rejects_unknown_recurrent_type():
    with pytest.raises(ValueError, match="Unsupported model.recurrent_type"):
        create_model(
            "LSTM_Baseline",
            feature_layout=_layout(),
            hidden_dim=16,
            output_dim=5,
            recurrent_type="rnn",
        )


def test_lstm_baseline_rejects_unknown_pooling_strategy():
    model = create_model(
        "LSTM_Baseline",
        feature_layout=_layout(),
        hidden_dim=16,
        output_dim=5,
        pooling_strategy="attention",
        recurrent_type="lstm",
    )

    with pytest.raises(ValueError, match="Unsupported pooling_strategy"):
        model(_contract())
