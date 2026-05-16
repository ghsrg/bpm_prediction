from __future__ import annotations

import pytest
import torch

from src.domain.entities.feature_config import FeatureLayout
from src.domain.models.factory import create_model


def _layout() -> FeatureLayout:
    return FeatureLayout(
        cat_features={"concept:name": 16, "org:resource": 8},
        cat_feature_names=["concept:name", "org:resource"],
        num_dim=3,
    )


def _base_contract(*, batch_size: int = 2) -> dict:
    x_cat = torch.tensor([[1, 2], [2, 1]], dtype=torch.long)[:batch_size]
    x_num = torch.tensor([[0.1, 0.2, 0.3], [0.0, 1.0, 2.0]], dtype=torch.float32)[:batch_size]
    return {
        "x_cat": x_cat,
        "x_num": x_num,
        "edge_index": torch.zeros((2, 0), dtype=torch.long),
        "edge_type": torch.zeros((0,), dtype=torch.long),
        "y": torch.arange(batch_size, dtype=torch.long),
        "batch": torch.arange(batch_size, dtype=torch.long),
        "num_nodes": batch_size,
    }


def _topology_contract(*, batch_size: int = 2) -> dict:
    return {
        **_base_contract(batch_size=batch_size),
        "structural_edge_index": torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
        "structural_edge_weight": torch.ones((3,), dtype=torch.float32),
        "struct_x": torch.randn(4, 3, dtype=torch.float32),
        "struct_node_to_class_index": torch.tensor([0, 1, 1, 2], dtype=torch.long),
        "struct_prefix_state_x": torch.randn(batch_size, 4, 6, dtype=torch.float32),
    }


@pytest.mark.parametrize(
    ("raw_fusion_mode", "expected"),
    [
        ("TopologyStateEncoder", "topology_state_encoder"),
        ("topology_state_encoder", "topology_state_encoder"),
        ("topology_state", "topology_state_encoder"),
        ("EarlyTopologyStateEncoder", "topology_state_encoder"),
        ("TopologyStateGraphEncoder", "topology_state_graph_encoder"),
        ("topology_state_graph_encoder", "topology_state_graph_encoder"),
        ("TopologyGraphEncoder", "topology_state_graph_encoder"),
        ("topology_graph_encoder", "topology_state_graph_encoder"),
        ("StructuralGraphEncoder", "topology_state_graph_encoder"),
    ],
)
def test_eopkg_gatv2_normalizes_topology_state_aliases(raw_fusion_mode: str, expected: str):
    model = create_model(
        model_type="EOPKGGATv2",
        feature_layout=_layout(),
        hidden_dim=16,
        output_dim=3,
        struct_hidden_dim=16,
        dropout=0.0,
        fusion_mode=raw_fusion_mode,
    )

    assert model.fusion_mode == expected


def test_topology_state_encoder_returns_class_logits_and_diagnostics():
    model = create_model(
        model_type="EOPKGGATv2",
        feature_layout=_layout(),
        hidden_dim=16,
        output_dim=3,
        struct_hidden_dim=16,
        dropout=0.0,
        fusion_mode="TopologyStateEncoder",
        topology_state_beta=0.5,
        topology_state_class_pooling="logmeanexp",
    )

    logits = model(_topology_contract(batch_size=2))

    assert tuple(logits.shape) == (2, 3)
    assert model.last_topology_state_node_logits is not None
    assert tuple(model.last_topology_state_node_logits.shape) == (2, 4)
    assert model.last_topology_state_class_logits is not None
    assert tuple(model.last_topology_state_class_logits.shape) == (2, 3)
    assert model.last_topology_state_entropy is not None
    assert model.last_topology_state_gate_mean is not None
    assert model.last_topology_state_mean_class_cardinality is not None
    assert model.last_topology_state_max_class_cardinality is not None


def test_topology_state_encoder_trainable_parameters_exist_before_first_forward():
    model = create_model(
        model_type="EOPKGGATv2",
        feature_layout=_layout(),
        hidden_dim=16,
        output_dim=3,
        struct_hidden_dim=16,
        dropout=0.0,
        fusion_mode="TopologyStateEncoder",
    )

    parameters = dict(model.named_parameters())

    assert "topology_state_proj.weight" in parameters
    assert "topology_state_gate_proj.weight" in parameters
    assert parameters["topology_state_proj.weight"].shape == torch.Size([16, 6])
    assert parameters["topology_state_gate_proj.weight"].shape == torch.Size([16, 6])


def test_topology_state_encoder_requires_prefix_state_tensor():
    model = create_model(
        model_type="EOPKGGATv2",
        feature_layout=_layout(),
        hidden_dim=16,
        output_dim=3,
        struct_hidden_dim=16,
        dropout=0.0,
        fusion_mode="TopologyStateEncoder",
    )
    contract = _topology_contract(batch_size=1)
    contract.pop("struct_prefix_state_x")

    with pytest.raises(ValueError, match="TopologyStateEncoder requires struct_prefix_state_x"):
        model(contract)


def test_topology_state_encoder_structural_mode_off_uses_observed_only_path():
    model = create_model(
        model_type="EOPKGGATv2",
        feature_layout=_layout(),
        hidden_dim=16,
        output_dim=3,
        struct_hidden_dim=16,
        dropout=0.0,
        structural_mode=False,
        fusion_mode="TopologyStateEncoder",
    )
    model.eval()

    with torch.no_grad():
        without_struct = model(_base_contract(batch_size=1))
        with_struct = model(_topology_contract(batch_size=1))

    assert tuple(without_struct.shape) == (1, 3)
    assert tuple(with_struct.shape) == (1, 3)
    assert torch.allclose(without_struct, with_struct, atol=1e-6, rtol=0.0)


def test_topology_state_graph_encoder_returns_graph_logits_and_diagnostics():
    model = create_model(
        model_type="EOPKGGATv2",
        feature_layout=_layout(),
        hidden_dim=16,
        output_dim=3,
        struct_hidden_dim=16,
        dropout=0.0,
        fusion_mode="TopologyStateGraphEncoder",
        topology_state_beta=0.5,
        topology_graph_pooling="mean",
    )

    logits = model(_topology_contract(batch_size=2))

    assert tuple(logits.shape) == (2, 3)
    assert model.last_topology_graph_context is not None
    assert tuple(model.last_topology_graph_context.shape) == (2, 16)
    assert model.last_topology_graph_logits is not None
    assert tuple(model.last_topology_graph_logits.shape) == (2, 3)
    assert model.last_topology_graph_entropy is not None
    assert model.last_topology_state_gate_mean is not None
    assert model.last_topology_state_gate_max is not None
    assert model.last_topology_state_node_logits is None
    assert model.last_topology_state_class_logits is None


def test_topology_state_graph_encoder_requires_prefix_state_tensor():
    model = create_model(
        model_type="EOPKGGATv2",
        feature_layout=_layout(),
        hidden_dim=16,
        output_dim=3,
        struct_hidden_dim=16,
        dropout=0.0,
        fusion_mode="TopologyStateGraphEncoder",
    )
    contract = _topology_contract(batch_size=1)
    contract.pop("struct_prefix_state_x")

    with pytest.raises(ValueError, match="TopologyStateGraphEncoder requires struct_prefix_state_x"):
        model(contract)


def test_topology_state_graph_encoder_ignores_observed_graph_features():
    model = create_model(
        model_type="EOPKGGATv2",
        feature_layout=_layout(),
        hidden_dim=16,
        output_dim=3,
        struct_hidden_dim=16,
        dropout=0.0,
        fusion_mode="TopologyStateGraphEncoder",
    )
    model.eval()
    contract_a = _topology_contract(batch_size=2)
    contract_b = {
        key: value.clone() if isinstance(value, torch.Tensor) else value
        for key, value in contract_a.items()
    }
    contract_b["x_cat"] = torch.flip(contract_b["x_cat"], dims=[0])
    contract_b["x_num"] = contract_b["x_num"] + 1000.0

    with torch.no_grad():
        logits_a = model(contract_a)
        logits_b = model(contract_b)

    assert torch.allclose(logits_a, logits_b, atol=1e-6, rtol=0.0)


def test_topology_state_graph_encoder_structural_mode_off_uses_observed_only_path():
    model = create_model(
        model_type="EOPKGGATv2",
        feature_layout=_layout(),
        hidden_dim=16,
        output_dim=3,
        struct_hidden_dim=16,
        dropout=0.0,
        structural_mode=False,
        fusion_mode="TopologyStateGraphEncoder",
    )
    model.eval()

    with torch.no_grad():
        without_struct = model(_base_contract(batch_size=1))
        with_struct = model(_topology_contract(batch_size=1))

    assert tuple(without_struct.shape) == (1, 3)
    assert tuple(with_struct.shape) == (1, 3)
    assert torch.allclose(without_struct, with_struct, atol=1e-6, rtol=0.0)
