from __future__ import annotations

import logging

import pytest
import torch

from src.domain.entities.feature_config import FeatureLayout
from src.domain.models.factory import create_model, get_registered_models


def _feature_layout() -> FeatureLayout:
    return FeatureLayout(
        cat_features={"concept:name": 10, "org:resource": 6},
        cat_feature_names=["concept:name", "org:resource"],
        num_dim=3,
    )


def _contract(*, with_struct: bool) -> dict:
    contract = {
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
    if with_struct:
        contract["structural_edge_index"] = torch.tensor([[0, 1, 1], [1, 2, 3]], dtype=torch.long)
        contract["structural_edge_weight"] = torch.tensor([5.0, 2.0, 1.0], dtype=torch.float32)
    return contract


@pytest.mark.parametrize("model_type", ["EOPKGGCN", "EOPKGGATv2"])
def test_eopkg_models_are_initialized_via_factory(model_type: str):
    model = create_model(
        model_type=model_type,
        feature_layout=_feature_layout(),
        hidden_dim=16,
        output_dim=7,
        dropout=0.0,
        pooling_strategy="global_mean",
    )
    assert model.__class__.__name__ == model_type
    assert model_type in get_registered_models()


@pytest.mark.parametrize("model_type", ["EOPKGGCN", "EOPKGGATv2"])
def test_eopkg_forward_with_structural_tensors(model_type: str):
    model = create_model(
        model_type=model_type,
        feature_layout=_feature_layout(),
        hidden_dim=16,
        output_dim=7,
        dropout=0.0,
        pooling_strategy="global_mean",
    )
    logits = model(_contract(with_struct=True))
    assert isinstance(logits, torch.Tensor)
    assert logits.dtype == torch.float32
    assert tuple(logits.shape) == (1, 7)


@pytest.mark.parametrize("fusion_mode", ["Attention", "Concat", "concat_mlp", "struct_pool_concat"])
def test_eopkggatv2_forward_supports_fusion_modes(fusion_mode: str):
    model = create_model(
        model_type="EOPKGGATv2",
        feature_layout=_feature_layout(),
        hidden_dim=16,
        output_dim=7,
        dropout=0.0,
        pooling_strategy="global_mean",
        fusion_mode=fusion_mode,
    )
    logits = model(_contract(with_struct=True))
    assert isinstance(logits, torch.Tensor)
    assert tuple(logits.shape) == (1, 7)


@pytest.mark.parametrize("model_type", ["EOPKGGCN", "EOPKGGATv2"])
def test_eopkg_fallback_without_structural_tensors_logs_warning_and_keeps_shape(
    model_type: str, caplog: pytest.LogCaptureFixture
):
    model = create_model(
        model_type=model_type,
        feature_layout=_feature_layout(),
        hidden_dim=16,
        output_dim=7,
        dropout=0.0,
        pooling_strategy="global_mean",
    )

    with caplog.at_level(logging.WARNING):
        logits = model(_contract(with_struct=False))

    assert tuple(logits.shape) == (1, 7)
    assert "Structural tensors are missing in contract! Falling back to Baseline forward." in caplog.text


@pytest.mark.parametrize("model_type", ["EOPKGGCN", "EOPKGGATv2"])
def test_eopkg_structural_mode_off_ignores_structural_branch(model_type: str):
    model = create_model(
        model_type=model_type,
        feature_layout=_feature_layout(),
        hidden_dim=16,
        output_dim=7,
        dropout=0.0,
        pooling_strategy="global_mean",
        structural_mode=False,
    )
    model.eval()

    with torch.no_grad():
        logits_without_struct = model(_contract(with_struct=False))
        logits_with_struct = model(_contract(with_struct=True))

    assert tuple(logits_without_struct.shape) == (1, 7)
    assert tuple(logits_with_struct.shape) == (1, 7)
    assert torch.allclose(logits_without_struct, logits_with_struct, atol=1e-6, rtol=0.0)


def test_eopkggatv2_rejects_unknown_fusion_mode():
    with pytest.raises(ValueError, match="Unsupported model.fusion_mode"):
        create_model(
            model_type="EOPKGGATv2",
            feature_layout=_feature_layout(),
            hidden_dim=16,
            output_dim=7,
            dropout=0.0,
            pooling_strategy="global_mean",
            fusion_mode="unknown_mode",
        )
