from __future__ import annotations

import pytest
import torch

from src.domain.entities.feature_config import FeatureLayout
from src.domain.models.factory import create_model, get_registered_models


def _layout() -> FeatureLayout:
    return FeatureLayout(
        cat_features={"concept:name": 5},
        num_dim=2,
        cat_feature_names=["concept:name"],
    )


def _batch() -> dict:
    return {
        "x_cat": torch.tensor([[1], [2], [3]], dtype=torch.long),
        "x_num": torch.tensor([[0.1, 0.2], [0.2, 0.3], [0.3, 0.4]], dtype=torch.float32),
        "edge_index": torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        "edge_type": torch.zeros((2,), dtype=torch.long),
        "batch": torch.zeros((3,), dtype=torch.long),
        "y": torch.tensor([2], dtype=torch.long),
        "num_nodes": 3,
        "struct_x": torch.eye(5, 3, dtype=torch.float32),
        "structural_edge_index": torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long),
        "structural_edge_weight": torch.ones((4,), dtype=torch.float32),
        "struct_node_to_class_index": torch.tensor([0, 1, 2, 3, 4], dtype=torch.long),
        "allowed_target_mask": torch.tensor([[False, True, True, False, False]], dtype=torch.bool),
    }


def _model(**kwargs):
    defaults = {
        "feature_layout": _layout(),
        "hidden_dim": 8,
        "output_dim": 5,
        "dropout": 0.0,
        "pooling_strategy": "last_node",
    }
    defaults.update(kwargs)
    return create_model("EOPKGTopologyConditioned", **defaults)


def test_topology_conditioned_model_is_registered():
    assert "EOPKGTopologyConditioned" in get_registered_models()
    model = _model()
    assert model.__class__.__name__ == "EOPKGTopologyConditioned"


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("observed_encoder", "GCN", "observed_encoder"),
        ("struct_encoder", "GCN", "struct_encoder"),
        ("candidate_scoring", "mlp", "candidate_scoring"),
        ("candidate_pooling", "sum", "candidate_pooling"),
    ],
)
def test_topology_conditioned_rejects_unsupported_options(key: str, value: str, message: str):
    with pytest.raises(ValueError, match=message):
        _model(**{key: value})


def test_topology_conditioned_forward_returns_fixed_label_logits_stage1():
    logits = _model()(_batch())
    assert logits.shape == torch.Size([1, 5])


@pytest.mark.parametrize("missing_key", ["structural_edge_index", "struct_node_to_class_index"])
def test_topology_conditioned_requires_structural_payload(missing_key: str):
    batch = _batch()
    batch.pop(missing_key)
    with pytest.raises(ValueError, match=missing_key):
        _model()(batch)


def test_topology_conditioned_forward_can_start_without_struct_x_when_identity_mapping_exists():
    batch = _batch()
    batch.pop("struct_x")
    logits = _model()(batch)
    assert logits.shape == torch.Size([1, 5])


def test_topology_conditioned_identity_grounding_changes_candidate_embeddings():
    model = _model()
    batch = _batch()
    h1, _ = model._encode_candidates(batch, torch.device("cpu"))
    changed = dict(batch)
    changed["struct_node_to_class_index"] = torch.tensor([4, 3, 2, 1, 0], dtype=torch.long)
    h2, _ = model._encode_candidates(changed, torch.device("cpu"))
    assert not torch.allclose(h1, h2)


def test_topology_conditioned_non_target_identity_grounding_uses_shared_embedding():
    model = _model()
    batch = _batch()
    batch["struct_x"] = torch.zeros((4, 3), dtype=torch.float32)
    batch["struct_node_to_class_index"] = torch.tensor([-1, -1, 1, 2], dtype=torch.long)
    h_id = model._node_identity_features(batch["struct_node_to_class_index"], torch.device("cpu"))
    assert torch.allclose(h_id[0], h_id[1])
    assert not torch.allclose(h_id[0], h_id[2])


def test_topology_conditioned_mil_logmeanexp_normalizes_duplicate_count():
    model = _model(output_dim=3, candidate_pooling="logmeanexp")
    node_scores = torch.tensor([[1.0, 1.0, 0.0, -1.0]], dtype=torch.float32)
    node_to_class = torch.tensor([1, 1, 2, -1], dtype=torch.long)
    class_scores = model._pool_node_scores_to_classes(node_scores, node_to_class)
    assert class_scores.shape == torch.Size([1, 3])
    assert torch.isneginf(class_scores[0, 0])
    assert class_scores[0, 1].item() == pytest.approx(1.0, abs=1e-6)
    assert class_scores[0, 2].item() == pytest.approx(0.0, abs=1e-6)


def test_topology_conditioned_pooling_ignores_non_target_minus_one_nodes():
    model = _model(output_dim=3, candidate_pooling="max")
    node_scores = torch.tensor([[100.0, 0.25]], dtype=torch.float32)
    node_to_class = torch.tensor([-1, 2], dtype=torch.long)
    class_scores = model._pool_node_scores_to_classes(node_scores, node_to_class)
    assert torch.isneginf(class_scores[0, 0])
    assert torch.isneginf(class_scores[0, 1])
    assert class_scores[0, 2].item() == pytest.approx(0.25, abs=1e-6)


def test_topology_conditioned_forward_does_not_apply_allowed_target_mask():
    model = _model()
    batch = _batch()
    masked = dict(batch)
    masked["allowed_target_mask"] = torch.zeros_like(batch["allowed_target_mask"])
    logits_without_mask = model(batch)
    logits_with_mask = model(masked)
    assert torch.allclose(logits_with_mask, logits_without_mask)


def test_topology_conditioned_records_candidate_diagnostics_after_forward():
    model = _model()
    logits = model(_batch())
    assert logits.shape == torch.Size([1, 5])
    assert model.last_candidate_node_score_mean_abs >= 0.0
    assert model.last_candidate_class_score_mean_abs >= 0.0
    assert model.last_duplicate_candidate_count_max == 1
    assert model.last_candidate_temperature == pytest.approx(0.1)
    assert model.last_candidate_temperature_trainable is False
    assert model.last_candidate_prediction_entropy >= 0.0
    assert model.last_candidate_target_score is not None


def test_topology_conditioned_forward_candidate_returns_dynamic_candidate_space():
    model = _model(output_dim=6, candidate_pooling="logmeanexp")
    batch = _batch()
    batch["struct_x"] = torch.eye(5, 3, dtype=torch.float32)
    batch["struct_node_to_class_index"] = torch.tensor([-1, 1, 1, 4, -1], dtype=torch.long)
    batch["structural_edge_index"] = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)

    output = model.forward_candidate(batch)

    assert output.candidate_logits.shape == torch.Size([1, 2])
    assert output.candidate_class_index.tolist() == [1, 4]
    assert output.node_to_candidate_index.tolist() == [-1, 0, 0, 1, -1]
    assert output.fixed_label_logits.shape == torch.Size([1, 6])


def test_topology_conditioned_candidate_output_maps_global_targets_to_local_candidates():
    model = _model(output_dim=6)
    batch = _batch()
    batch["struct_node_to_class_index"] = torch.tensor([-1, 1, 1, 4, -1], dtype=torch.long)
    batch["y"] = torch.tensor([4, 2], dtype=torch.long)

    output = model.forward_candidate(batch)

    assert output.map_targets_to_candidate_index(torch.tensor([1, 4], dtype=torch.long)).tolist() == [0, 1]
    assert output.map_targets_to_candidate_index(batch["y"]).tolist() == [1, -1]


def test_topology_conditioned_forward_candidate_uses_topology_candidate_axis_with_unseen_label():
    model = _model(output_dim=6)
    batch = _batch()
    batch["struct_node_to_class_index"] = torch.tensor([-1, 1, 4], dtype=torch.long)
    batch["struct_node_to_candidate_index"] = torch.tensor([0, 1, 2], dtype=torch.long)
    batch["candidate_class_index"] = torch.tensor([-1, 1, 4], dtype=torch.long)
    batch["candidate_ids"] = ("node_c", "node_a", "node_b")
    batch["candidate_labels"] = ("C", "A", "B")
    batch["candidate_is_unseen"] = torch.tensor([True, False, False], dtype=torch.bool)
    batch["structural_edge_index"] = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    batch["struct_x"] = torch.eye(3, 3, dtype=torch.float32)

    output = model.forward_candidate(batch)

    assert output.candidate_logits.shape == torch.Size([1, 3])
    assert output.candidate_class_index.tolist() == [-1, 1, 4]
    assert output.candidate_ids == ("node_c", "node_a", "node_b")
    assert output.candidate_is_unseen.tolist() == [True, False, False]
    assert output.map_target_labels_to_candidate_mask(["C"]).tolist() == [[True, False, False]]


def test_topology_conditioned_forward_remains_fixed_label_compatible_after_stage2():
    model = _model(output_dim=6)
    batch = _batch()
    batch["struct_node_to_class_index"] = torch.tensor([-1, 1, 1, 4, -1], dtype=torch.long)

    logits = model(batch)

    assert logits.shape == torch.Size([1, 6])
    assert model.last_candidate_dynamic_count == 2


def test_topology_conditioned_candidate_tuple_unpacks_nested_pyg_collation():
    model = _model(output_dim=6)
    contract = {
        "candidate_labels": [("A", "B", "C"), ("A", "B", "C")],
        "candidate_ids": [("node_a", "node_b", "node_c"), ("node_a", "node_b", "node_c")]
    }
    labels = model._candidate_tuple(contract, "candidate_labels")
    ids = model._candidate_tuple(contract, "candidate_ids")
    
    assert labels == ("A", "B", "C")
    assert ids == ("node_a", "node_b", "node_c")
