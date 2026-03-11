from __future__ import annotations

import pytest
import torch

from src.domain.services.baseline_graph_builder import BaselineGraphBuilder
from src.domain.services.feature_encoder import FeatureEncoder
from src.domain.services.prefix_policy import PrefixPolicy
from src.domain.services.schema_resolver import SchemaResolver


@pytest.mark.mvp1_regression
def test_mvp1_graph_tensor_contract_shape_and_types(mock_feature_configs, mock_raw_trace):
    encoder = FeatureEncoder(
        feature_configs=mock_feature_configs,
        traces=[mock_raw_trace],
        schema_resolver=SchemaResolver(fallback_keys=("legacy_activity",)),
    )
    graph_builder = BaselineGraphBuilder(feature_encoder=encoder)
    prefix_policy = PrefixPolicy()

    slices = prefix_policy.generate_slices(mock_raw_trace)
    assert slices, "Expected at least one prefix slice for contract check."

    contract = graph_builder.build_graph(slices[0])

    assert set(contract.keys()) == {"x_cat", "x_num", "edge_index", "edge_type", "y", "batch", "num_nodes"}
    assert isinstance(contract["x_cat"], torch.Tensor) and contract["x_cat"].dtype == torch.long
    assert isinstance(contract["x_num"], torch.Tensor) and contract["x_num"].dtype == torch.float32
    assert isinstance(contract["edge_index"], torch.Tensor) and contract["edge_index"].dtype == torch.long
    assert isinstance(contract["edge_type"], torch.Tensor) and contract["edge_type"].dtype == torch.long
    assert isinstance(contract["y"], torch.Tensor) and contract["y"].dtype == torch.long
    assert isinstance(contract["batch"], torch.Tensor) and contract["batch"].dtype == torch.long
    assert isinstance(contract["num_nodes"], int)

    # MVP1 contract must not depend on MVP2 structural fields.
    for forbidden in ("structural_edge_index", "structural_edge_weight", "allowed_target_mask", "version_emb_idx"):
        assert forbidden not in contract

