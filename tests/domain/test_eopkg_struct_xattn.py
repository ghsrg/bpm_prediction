from __future__ import annotations

import logging

import pytest
import torch

from src.domain.entities.feature_config import FeatureLayout
from src.domain.models.factory import create_model


def _feature_layout() -> FeatureLayout:
    return FeatureLayout(
        cat_features={"concept:name": 10, "org:resource": 6},
        cat_feature_names=["concept:name", "org:resource"],
        num_dim=3,
    )


def _contract(*, with_struct: bool = True) -> dict:
    contract = {
        "x_cat": torch.tensor([[1, 2], [2, 1], [3, 0], [4, 1]], dtype=torch.long),
        "x_num": torch.tensor(
            [
                [0.1, 0.2, 0.3],
                [0.0, 1.0, 2.0],
                [0.5, 0.7, 0.9],
                [1.0, 0.2, 0.4],
            ],
            dtype=torch.float32,
        ),
        "edge_index": torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
        "edge_type": torch.zeros((3,), dtype=torch.long),
        "y": torch.tensor([1], dtype=torch.long),
        "batch": torch.tensor([0, 0, 0, 0], dtype=torch.long),
        "num_nodes": 4,
    }
    if with_struct:
        contract["struct_x"] = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.5, 0.5, 0.5],
            ],
            dtype=torch.float32,
        )
        contract["structural_edge_index"] = torch.tensor(
            [[0, 1, 1, 2], [1, 2, 3, 3]],
            dtype=torch.long,
        )
        contract["structural_edge_weight"] = torch.ones(4, dtype=torch.float32)
        contract["struct_node_to_class_index"] = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    return contract


def _model(**kwargs):
    model_kwargs = {
        "model_type": "EOPKGGATv2",
        "feature_layout": _feature_layout(),
        "hidden_dim": 16,
        "output_dim": 7,
        "dropout": 0.0,
        "pooling_strategy": "global_mean",
        "fusion_mode": "StructXAttn",
        "struct_xattn_heads": 4,
        "struct_xattn_layers": "post_conv2",
    }
    model_kwargs.update(kwargs)
    return create_model(
        **model_kwargs,
    )


def test_struct_xattn_forward_shape_and_diagnostics():
    torch.manual_seed(17)
    model = _model()
    model.eval()

    with torch.no_grad():
        logits = model(_contract())

    assert tuple(logits.shape) == (1, 7)
    assert model.fusion_mode == "struct_xattn"
    assert model.last_struct_xattn_context_mean_abs is not None
    assert model.last_struct_xattn_delta_mean_abs is not None
    assert model.last_struct_xattn_to_observed_ratio is not None
    assert model.last_struct_xattn_raw_context_mean_abs is not None
    assert model.last_struct_xattn_pre_norm_delta_mean_abs is not None
    assert model.last_struct_xattn_post_norm_delta_mean_abs is not None
    assert model.last_struct_xattn_raw_to_observed_ratio is not None
    assert model.last_struct_xattn_pre_norm_to_observed_ratio is not None
    assert model.last_struct_xattn_post_norm_to_observed_ratio is not None
    assert model.last_struct_xattn_attention_entropy is not None
    assert model.last_struct_xattn_scale is not None
    assert model.last_struct_xattn_gate_mean is not None
    assert model.last_struct_xattn_l1_delta_mean_abs is None
    assert model.last_struct_xattn_l2_delta_mean_abs is not None


def test_struct_xattn_structural_mode_false_matches_observed_path():
    torch.manual_seed(23)
    model = _model(structural_mode=False)
    model.eval()

    with torch.no_grad():
        logits_without_struct = model(_contract(with_struct=False))
        logits_with_struct = model(_contract(with_struct=True))

    assert torch.allclose(logits_without_struct, logits_with_struct, atol=1e-6, rtol=0.0)
    assert model.last_struct_xattn_delta_mean_abs is None


def test_struct_xattn_missing_structure_falls_back_to_observed(caplog: pytest.LogCaptureFixture):
    torch.manual_seed(29)
    model = _model()
    model.eval()

    with caplog.at_level(logging.WARNING), torch.no_grad():
        logits = model(_contract(with_struct=False))

    assert tuple(logits.shape) == (1, 7)
    assert "Structural tensors are missing in contract! Falling back to Baseline forward." in caplog.text
    assert model.last_struct_xattn_delta_mean_abs is None


def test_struct_xattn_changing_structure_changes_logits():
    torch.manual_seed(31)
    model = _model()
    model.eval()
    base_contract = _contract()
    changed_contract = _contract()
    changed_contract["struct_x"] = torch.flip(changed_contract["struct_x"], dims=[0])

    with torch.no_grad():
        logits_base = model(base_contract)
        logits_changed = model(changed_contract)

    assert not torch.allclose(logits_base, logits_changed, atol=1e-6, rtol=0.0)


def test_struct_xattn_after_each_conv_records_layer_metrics():
    torch.manual_seed(37)
    model = _model(struct_xattn_layers="after_each_conv")
    model.eval()

    with torch.no_grad():
        logits = model(_contract())

    assert tuple(logits.shape) == (1, 7)
    assert model.last_struct_xattn_l1_delta_mean_abs is not None
    assert model.last_struct_xattn_l2_delta_mean_abs is not None
    assert model.last_struct_xattn_l1_attention_entropy is not None
    assert model.last_struct_xattn_l2_attention_entropy is not None
