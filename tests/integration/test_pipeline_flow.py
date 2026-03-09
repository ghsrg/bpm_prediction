import torch

from src.domain.services.baseline_graph_builder import BaselineGraphBuilder
from src.domain.services.feature_encoder import FeatureEncoder
from src.domain.services.prefix_policy import PrefixPolicy


def test_pipeline_flow_builds_expected_tensor_shapes_and_types(mock_feature_configs, mock_raw_trace):
    encoder = FeatureEncoder(feature_configs=mock_feature_configs, traces=[mock_raw_trace])
    prefix_slices = PrefixPolicy().generate_slices(mock_raw_trace)
    prefix = prefix_slices[-1]

    contract = BaselineGraphBuilder(feature_encoder=encoder).build_graph(prefix)

    assert isinstance(contract["x_cat"], torch.Tensor)
    assert isinstance(contract["x_num"], torch.Tensor)
    assert isinstance(contract["edge_index"], torch.Tensor)
    assert isinstance(contract["edge_type"], torch.Tensor)
    assert isinstance(contract["y"], torch.Tensor)
    assert isinstance(contract["batch"], torch.Tensor)

    assert contract["x_cat"].dtype == torch.long
    assert contract["x_num"].dtype == torch.float32
    assert contract["edge_index"].dtype == torch.long
    assert contract["edge_type"].dtype == torch.long
    assert contract["y"].dtype == torch.long
    assert contract["batch"].dtype == torch.long

    assert contract["x_cat"].shape == (2, 2)
    assert contract["x_num"].shape == (2, 1)
    assert contract["edge_index"].shape == (2, 1)
    assert contract["edge_type"].shape == (1,)
    assert contract["y"].shape == (1,)
    assert contract["batch"].shape == (2,)
    assert contract["num_nodes"] == 2

