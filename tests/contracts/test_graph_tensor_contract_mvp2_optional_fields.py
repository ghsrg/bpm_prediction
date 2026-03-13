from __future__ import annotations

import torch

from src.domain.entities.feature_config import FeatureLayout
from src.domain.models.baseline_gcn import BaselineGCN


def test_mvp2_optional_contract_fields_do_not_break_mvp1_model_forward():
    model = BaselineGCN(
        feature_layout=FeatureLayout(cat_features={"concept:name": 4}, cat_feature_names=["concept:name"], num_dim=1),
        hidden_dim=8,
        output_dim=3,
        dropout=0.0,
        pooling_strategy="global_mean",
    )

    base_contract = {
        "x_cat": torch.tensor([[1], [2]], dtype=torch.long),
        "x_num": torch.tensor([[0.1], [0.2]], dtype=torch.float32),
        "edge_index": torch.tensor([[0], [1]], dtype=torch.long),
        "edge_type": torch.zeros((1,), dtype=torch.long),
        "y": torch.tensor([1], dtype=torch.long),
        "batch": torch.tensor([0, 0], dtype=torch.long),
        "num_nodes": 2,
    }

    extended_contract = {
        **base_contract,
        "struct_x": torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.float32),
        "structural_edge_index": torch.tensor([[0], [1]], dtype=torch.long),
        "structural_edge_weight": torch.tensor([1.0], dtype=torch.float32),
        "version_emb_idx": torch.tensor([0], dtype=torch.long),
        "allowed_target_mask": torch.tensor([[True, False, True]], dtype=torch.bool),
    }

    logits_base = model(base_contract)
    logits_extended = model(extended_contract)

    assert isinstance(logits_base, torch.Tensor)
    assert isinstance(logits_extended, torch.Tensor)
    assert logits_base.dtype == torch.float32
    assert logits_extended.dtype == torch.float32
    assert tuple(logits_base.shape) == (1, 3)
    assert tuple(logits_extended.shape) == (1, 3)
    assert extended_contract["struct_x"].dtype == torch.float32
    assert extended_contract["allowed_target_mask"].dtype == torch.bool
