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


def _struct_edge_index(num_nodes: int = 7) -> torch.Tensor:
    src = torch.arange(num_nodes, dtype=torch.long)
    dst = torch.roll(src, shifts=-1)
    return torch.stack([src, dst], dim=0)


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


def test_eopkg_gatv2_rejects_out_of_bounds_structural_edge_index():
    model = create_model(
        model_type="EOPKGGATv2",
        feature_layout=_layout(),
        hidden_dim=16,
        output_dim=3,
        struct_hidden_dim=16,
        cross_attention_heads=4,
        dropout=0.0,
        pooling_strategy="global_mean",
    )
    contract = {
        **_base_contract(),
        "structural_edge_index": torch.tensor([[0, 4], [1, 2]], dtype=torch.long),
        "struct_x": torch.randn(3, 5, dtype=torch.float32),
    }

    with pytest.raises(ValueError, match="Structural edge index is out of bounds"):
        model(contract)


def test_eopkg_gatv2_class_aware_structural_scoring_returns_per_class_structural_logits():
    model = create_model(
        model_type="EOPKGGATv2",
        feature_layout=_layout(),
        hidden_dim=16,
        output_dim=7,
        struct_hidden_dim=16,
        cross_attention_heads=4,
        dropout=0.0,
        pooling_strategy="global_mean",
        fusion_mode="ClassAwareStructuralScoring",
    )
    contract = {
        **_base_contract(),
        "structural_edge_index": _struct_edge_index(),
        "struct_node_to_class_index": torch.arange(7, dtype=torch.long),
    }

    logits = model(contract)

    assert tuple(logits.shape) == (1, 7)
    assert model.last_cross_attn_weights is None
    assert model.last_observed_logits is not None
    assert tuple(model.last_observed_logits.shape) == (1, 7)
    assert model.last_structural_node_logits is not None
    assert tuple(model.last_structural_node_logits.shape) == (1, 7)
    assert model.last_structural_class_logits is not None
    assert tuple(model.last_structural_class_logits.shape) == (1, 7)
    assert not any(isinstance(param, UninitializedParameter) for param in model.parameters())


def test_eopkg_gatv2_scales_structural_logits_against_observed_logits():
    model = create_model(
        model_type="EOPKGGATv2",
        feature_layout=_layout(),
        hidden_dim=8,
        output_dim=3,
        dropout=0.0,
        structural_mode=True,
        fusion_mode="ClassAwareStructuralScoring",
        structural_logit_scale_init=0.1,
        structural_observed_scale_min=1.0,
        structural_observed_scale_max=10.0,
    )
    raw = torch.tensor([[-1.0, 0.0, 1.0]], dtype=torch.float32)
    observed = torch.tensor([[20.0, -10.0, 0.0]], dtype=torch.float32)

    scaled, normalized, observed_scale = model._scale_structural_class_logits(
        raw_class_logits=raw,
        observed_logits=observed,
    )

    assert observed_scale.shape == torch.Size([1, 1])
    assert float(observed_scale.item()) == pytest.approx(10.0, abs=1e-6)
    assert torch.allclose(normalized.mean(dim=1), torch.zeros(1), atol=1e-6)
    assert torch.allclose(normalized.std(dim=1, unbiased=False), torch.ones(1), atol=1e-6)
    assert float(scaled.abs().mean().item()) == pytest.approx(float(normalized.abs().mean().item()), rel=1e-5)


def test_eopkg_gatv2_class_aware_structural_scoring_projects_nodes_to_classes():
    model = create_model(
        model_type="EOPKGGATv2",
        feature_layout=_layout(),
        hidden_dim=16,
        output_dim=3,
        struct_hidden_dim=16,
        dropout=0.0,
        pooling_strategy="global_mean",
        fusion_mode="ClassAwareStructuralScoring",
    )
    contract = {
        **_base_contract(),
        "structural_edge_index": _struct_edge_index(num_nodes=6),
        "struct_x": torch.randn(6, 5, dtype=torch.float32),
        "struct_node_to_class_index": torch.tensor([0, 1, 1, 2, -1, 2], dtype=torch.long),
    }

    logits = model(contract)

    assert tuple(logits.shape) == (1, 3)
    assert model.last_cross_attn_weights is None
    assert model.last_structural_node_logits is not None
    assert tuple(model.last_structural_node_logits.shape) == (1, 6)
    assert model.last_structural_class_logits is not None
    assert tuple(model.last_structural_class_logits.shape) == (1, 3)


def test_eopkg_gatv2_class_aware_bilinear_plus_prior_exposes_raw_and_normalized_diagnostics():
    model = create_model(
        model_type="EOPKGGATv2",
        feature_layout=_layout(),
        hidden_dim=8,
        output_dim=4,
        dropout=0.0,
        structural_mode=True,
        fusion_mode="ClassAwareStructuralScoring",
        structural_score_mode="bilinear_with_prior",
        structural_logit_scale_init=0.1,
    )
    contract = {
        **_base_contract(),
        "structural_edge_index": _struct_edge_index(num_nodes=6),
        "struct_x": torch.randn(6, 3, dtype=torch.float32),
        "struct_node_to_class_index": torch.tensor([0, 1, 1, 2, 3, -1], dtype=torch.long),
    }

    logits = model(contract)

    assert logits.shape == torch.Size([1, 4])
    assert isinstance(model.last_structural_raw_class_logits, torch.Tensor)
    assert isinstance(model.last_structural_normalized_class_logits, torch.Tensor)
    assert isinstance(model.last_structural_observed_scale, torch.Tensor)
    assert model.last_structural_raw_class_logits.shape == torch.Size([1, 4])
    assert model.last_structural_normalized_class_logits.shape == torch.Size([1, 4])
    assert model.last_structural_observed_scale.shape == torch.Size([1, 1])


def test_eopkg_gatv2_class_aware_structural_logits_keep_gradient_for_auxiliary_loss():
    model = create_model(
        model_type="EOPKGGATv2",
        feature_layout=_layout(),
        hidden_dim=8,
        output_dim=4,
        dropout=0.0,
        structural_mode=True,
        fusion_mode="ClassAwareStructuralScoring",
        structural_score_mode="bilinear_with_prior",
        structural_logit_scale_init=0.1,
    )
    contract = {
        **_base_contract(),
        "structural_edge_index": _struct_edge_index(num_nodes=6),
        "struct_x": torch.randn(6, 3, dtype=torch.float32),
        "struct_node_to_class_index": torch.tensor([0, 1, 1, 2, 3, -1], dtype=torch.long),
    }

    model(contract)

    assert isinstance(model.last_structural_class_logits, torch.Tensor)
    assert model.last_structural_class_logits.requires_grad


def test_eopkg_gatv2_class_aware_structural_scoring_requires_node_to_class_mapping():
    model = create_model(
        model_type="EOPKGGATv2",
        feature_layout=_layout(),
        hidden_dim=16,
        output_dim=3,
        struct_hidden_dim=16,
        dropout=0.0,
        pooling_strategy="global_mean",
        fusion_mode="ClassAwareStructuralScoring",
    )
    contract = {
        **_base_contract(),
        "structural_edge_index": _struct_edge_index(num_nodes=6),
        "struct_x": torch.randn(6, 5, dtype=torch.float32),
    }

    with pytest.raises(ValueError, match="struct_node_to_class_index"):
        model(contract)


def test_eopkg_gatv2_class_aware_structural_scoring_falls_back_without_structural_edges(
    caplog: pytest.LogCaptureFixture,
):
    model = create_model(
        model_type="EOPKGGATv2",
        feature_layout=_layout(),
        hidden_dim=16,
        output_dim=7,
        struct_hidden_dim=16,
        dropout=0.0,
        pooling_strategy="global_mean",
        fusion_mode="ClassAwareStructuralScoring",
    )

    with caplog.at_level(logging.WARNING):
        logits = model(_base_contract())

    assert tuple(logits.shape) == (1, 7)
    assert model.last_cross_attn_weights is None
    assert model.last_observed_logits is None
    assert model.last_structural_node_logits is None
    assert model.last_structural_class_logits is None
    assert "Structural tensors are missing in contract! Falling back to Baseline forward." in caplog.text


@pytest.mark.parametrize(
    ("raw_fusion_mode", "expected"),
    [
        ("Attention", "class_mean_attention"),
        ("Concat", "class_mean_concat"),
        ("struct_pool_concat", "class_mean_concat"),
        ("ClassAwareAdditive", "class_aware_structural_scoring"),
        ("ClassAwareAttention", "class_aware_structural_scoring"),
        ("ClassAwareStructuralScoring", "class_aware_structural_scoring"),
    ],
)
def test_eopkg_gatv2_normalizes_fusion_mode_aliases(raw_fusion_mode: str, expected: str):
    model = create_model(
        model_type="EOPKGGATv2",
        feature_layout=_layout(),
        hidden_dim=16,
        output_dim=7,
        struct_hidden_dim=16,
        dropout=0.0,
        pooling_strategy="global_mean",
        fusion_mode=raw_fusion_mode,
    )

    assert model.fusion_mode == expected
