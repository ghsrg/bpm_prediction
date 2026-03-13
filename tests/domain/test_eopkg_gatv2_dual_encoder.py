from __future__ import annotations

import logging

import pytest
import torch
from torch.nn.parameter import UninitializedParameter

from src.domain.entities.feature_config import FeatureLayout
from src.domain.models.factory import create_model


def _layout() -> FeatureLayout:
    return FeatureLayout(
        cat_features={"concept:name": 16, "org:resource": 8},
        cat_feature_names=["concept:name", "org:resource"],
        num_dim=3,
    )


def _base_contract() -> dict:
    return {
        "x_cat": torch.tensor([[1, 2], [2, 1], [3, 0]], dtype=torch.long),
        "x_num": torch.tensor(
            [[0.1, 0.2, 0.3], [0.0, 1.0, 2.0], [0.5, 0.7, 0.9]],
            dtype=torch.float32,
        ),
        "edge_index": torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        "edge_type": torch.zeros((2,), dtype=torch.long),
        "y": torch.tensor([1], dtype=torch.long),
        "batch": torch.tensor([0, 0, 0], dtype=torch.long),
        "num_nodes": 3,
    }


def _struct_edge_index() -> torch.Tensor:
    return torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)


def test_eopkg_gatv2_dual_encoder_returns_attention_weights():
    model = create_model(
        model_type="EOPKGGATv2",
        feature_layout=_layout(),
        hidden_dim=16,
        output_dim=7,
        struct_hidden_dim=16,
        cross_attention_heads=4,
        dropout=0.0,
        pooling_strategy="global_mean",
    )
    contract = {
        **_base_contract(),
        "structural_edge_index": _struct_edge_index(),
    }
    logits = model(contract)

    assert tuple(logits.shape) == (1, 7)
    assert model.last_cross_attn_weights is not None
    assert tuple(model.last_cross_attn_weights.shape) == (1, 4, 1, 7)
    assert not any(isinstance(param, UninitializedParameter) for param in model.parameters())


def test_eopkg_gatv2_accepts_struct_x_float_and_projects_if_needed():
    model = create_model(
        model_type="EOPKGGATv2",
        feature_layout=_layout(),
        hidden_dim=16,
        output_dim=7,
        struct_hidden_dim=12,
        cross_attention_heads=4,
        dropout=0.0,
        pooling_strategy="global_mean",
    )
    contract = {
        **_base_contract(),
        "structural_edge_index": _struct_edge_index(),
        "struct_x": torch.randn(7, 5, dtype=torch.float32),
    }
    logits = model(contract)

    assert tuple(logits.shape) == (1, 7)
    assert model.last_cross_attn_weights is not None
    assert tuple(model.last_cross_attn_weights.shape) == (1, 4, 1, 7)
    assert not any(isinstance(param, UninitializedParameter) for param in model.parameters())


def test_eopkg_gatv2_falls_back_without_structural_edges(caplog: pytest.LogCaptureFixture):
    model = create_model(
        model_type="EOPKGGATv2",
        feature_layout=_layout(),
        hidden_dim=16,
        output_dim=7,
        struct_hidden_dim=16,
        cross_attention_heads=4,
        dropout=0.0,
        pooling_strategy="global_mean",
    )
    with caplog.at_level(logging.WARNING):
        logits = model(_base_contract())

    assert tuple(logits.shape) == (1, 7)
    assert model.last_cross_attn_weights is None
    assert "Structural tensors are missing in contract! Falling back to Baseline forward." in caplog.text
    assert not any(isinstance(param, UninitializedParameter) for param in model.parameters())
