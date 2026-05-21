import torch
from torch_geometric.data import Data

from src.application.services.topology_payload_pool import (
    TopologyPayloadPool,
    drop_structural_edges,
    replace_structural_payload,
)


def _graph(version_idx: int, *, edge_count: int = 3) -> Data:
    edges = torch.tensor(
        [[0, 1, 2, 3], [1, 2, 3, 0]],
        dtype=torch.long,
    )[:, :edge_count]
    graph = Data(
        x_cat=torch.zeros((2, 1), dtype=torch.long),
        x_num=torch.zeros((2, 1), dtype=torch.float32),
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
        y=torch.tensor([0], dtype=torch.long),
        struct_x=torch.full((4, 2), float(version_idx), dtype=torch.float32),
        structural_edge_index=edges,
        structural_edge_weight=torch.arange(1, edge_count + 1, dtype=torch.float32),
        struct_node_to_class_index=torch.tensor([0, 1, 2, 3], dtype=torch.long),
        process_version_idx=torch.tensor([version_idx], dtype=torch.long),
    )
    return graph


def test_payload_pool_collects_one_payload_per_version_and_selects_wrong_version():
    pool = TopologyPayloadPool.from_dataset(
        [_graph(1), _graph(2), _graph(1)],
        idx_to_version={1: "v1", 2: "v2"},
    )

    assert pool.version_labels == ["v1", "v2"]
    wrong = pool.wrong_version_payload("v1")

    assert wrong is not None
    assert wrong.version_label == "v2"
    assert torch.allclose(wrong.struct_x, torch.full((4, 2), 2.0))


def test_replace_structural_payload_does_not_mutate_original_contract():
    pool = TopologyPayloadPool.from_dataset([_graph(1), _graph(2)], idx_to_version={1: "v1", 2: "v2"})
    wrong = pool.wrong_version_payload("v1")
    contract = {
        "struct_x": _graph(1).struct_x,
        "structural_edge_index": _graph(1).structural_edge_index,
        "structural_edge_weight": _graph(1).structural_edge_weight,
        "struct_node_to_class_index": _graph(1).struct_node_to_class_index,
    }

    replaced = replace_structural_payload(contract, wrong)

    assert torch.allclose(contract["struct_x"], torch.full((4, 2), 1.0))
    assert torch.allclose(replaced["struct_x"], torch.full((4, 2), 2.0))
    assert replaced["struct_x"] is not wrong.struct_x


def test_drop_structural_edges_keeps_at_least_one_edge_and_syncs_weights():
    contract = {
        "structural_edge_index": _graph(1, edge_count=3).structural_edge_index,
        "structural_edge_weight": _graph(1, edge_count=3).structural_edge_weight,
    }

    dropped = drop_structural_edges(contract, drop_ratio=1.0, seed=7)

    assert dropped["structural_edge_index"].shape[1] == 1
    assert dropped["structural_edge_weight"].numel() == 1
    assert contract["structural_edge_index"].shape[1] == 3

