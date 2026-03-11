from __future__ import annotations

import pytest
import torch

from src.domain.entities.feature_config import FeatureLayout
from src.domain.models.baseline_gat import BaselineGATv2
from src.domain.models.baseline_gcn import BaselineGCN


@pytest.mark.mvp1_regression
@pytest.mark.parametrize("model_cls", [BaselineGCN, BaselineGATv2])
def test_mvp1_baseline_models_forward_contract(model_cls):
    feature_layout = FeatureLayout(
        cat_features={"concept:name": 8, "org:resource": 4},
        cat_feature_names=["concept:name", "org:resource"],
        num_dim=3,
    )
    model = model_cls(
        feature_layout=feature_layout,
        hidden_dim=16,
        output_dim=5,
        dropout=0.0,
        pooling_strategy="global_mean",
    )

    contract = {
        "x_cat": torch.tensor([[1, 2], [2, 1]], dtype=torch.long),
        "x_num": torch.tensor([[0.1, 0.2, 0.3], [0.0, 1.0, 2.0]], dtype=torch.float32),
        "edge_index": torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        "edge_type": torch.zeros((2,), dtype=torch.long),
        "y": torch.tensor([1], dtype=torch.long),
        "batch": torch.tensor([0, 0], dtype=torch.long),
        "num_nodes": 2,
    }
    logits = model(contract)

    assert isinstance(logits, torch.Tensor)
    assert logits.dtype == torch.float32
    assert tuple(logits.shape) == (1, 5)

