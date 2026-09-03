from __future__ import annotations

import torch
from torch_geometric.data import Data

from src.domain.entities.feature_config import FeatureLayout


def brg_feature_layout() -> FeatureLayout:
    return FeatureLayout(
        cat_features={"concept:name": 4},
        num_dim=2,
        cat_feature_names=["concept:name"],
    )


def brg_mock_graph_contract() -> dict:
    return {
        "x_cat": torch.tensor([[1], [2], [3]], dtype=torch.long),
        "x_num": torch.tensor([[0.0, 0.1], [0.1, 0.2], [0.2, 0.3]], dtype=torch.float32),
        "edge_index": torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        "edge_type": torch.zeros((2,), dtype=torch.long),
        "batch": torch.zeros((3,), dtype=torch.long),
        "y": torch.tensor([2], dtype=torch.long),
        "num_nodes": 3,
        "struct_x": torch.eye(4, 3, dtype=torch.float32),
        "structural_edge_index": torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        "structural_edge_weight": torch.ones((2,), dtype=torch.float32),
        "struct_node_to_class_index": torch.tensor([0, 1, 2, 3], dtype=torch.long),
        "struct_node_to_candidate_index": torch.tensor([0, 1, 2, 3], dtype=torch.long),
        "candidate_class_index": torch.tensor([0, 1, 2, 3], dtype=torch.long),
        "candidate_is_unseen": torch.tensor([False, False, False, False], dtype=torch.bool),
        "candidate_ids": ("node_start", "node_b", "node_c", "node_end"),
        "candidate_labels": ("Start", "B", "C", "End"),
        "candidate_allowed_target_mask": torch.tensor([False, True, True, False], dtype=torch.bool),
        "allowed_target_mask": torch.tensor([[False, True, True, False]], dtype=torch.bool),
        "target_label": "C",
    }


def brg_mock_uniform_data() -> Data:
    contract = brg_mock_graph_contract()
    return Data(
        y=contract["y"],
        target_label=contract["target_label"],
        candidate_allowed_target_mask=contract["candidate_allowed_target_mask"].unsqueeze(0),
        candidate_ids=contract["candidate_ids"],
        candidate_labels=contract["candidate_labels"],
        prefix_len=torch.tensor([2]),
        process_version_idx=torch.tensor([1]),
        trace_idx=torch.tensor([0]),
        prefix_idx=torch.tensor([0]),
        trace_start_ts=torch.tensor([10.0]),
        trace_end_ts=torch.tensor([20.0]),
    )
