"""Baseline graph builder for MVP1 observed-graph tensors."""

# Відповідно до:
# - AGENT_GUIDE.MD -> розділ 2 (MVP1 без EPOKG/critic), розділ 4 (GraphTensorContract)
# - LLD_MVP1.MD -> розділ 4 (Feature Engineering), розділ 5 (лінійний edge_index)
# - DATA_FLOWS_MVP1.MD -> розділ 2.3 (IGraphBuilder контракт)

from __future__ import annotations

from typing import List

import torch

from src.application.ports.graph_builder_port import IGraphBuilder
from src.domain.entities.prefix_slice import PrefixSlice
from src.domain.entities.tensor_contract import GraphTensorContract
from src.domain.services.feature_encoder import FeatureEncoder


class BaselineGraphBuilder(IGraphBuilder):
    """Graph builder that emits split categorical/numeric tensors (x_cat + x_num)."""

    def __init__(self, feature_encoder: FeatureEncoder) -> None:
        self.feature_encoder = feature_encoder
        self.feature_layout = feature_encoder.feature_layout
        self.input_dim = self.feature_layout.num_dim + len(self.feature_layout.cat_feature_names)

    def build_graph(self, prefix: PrefixSlice) -> GraphTensorContract:
        """Convert PrefixSlice into GraphTensorContract for baseline GNN training."""
        cat_rows: List[torch.Tensor] = []
        num_rows: List[torch.Tensor] = []

        for event in prefix.prefix_events:
            encoded = self.feature_encoder.encode_event(event_extra=event.extra)
            cat_rows.append(torch.tensor(encoded.cat_indices, dtype=torch.long))
            num_rows.append(torch.tensor(encoded.num_values, dtype=torch.float32))

        if cat_rows:
            x_cat = torch.stack(cat_rows, dim=0)
        else:
            x_cat = torch.zeros((0, len(self.feature_layout.cat_feature_names)), dtype=torch.long)

        if num_rows:
            x_num = torch.stack(num_rows, dim=0)
        else:
            x_num = torch.zeros((0, self.feature_layout.num_dim), dtype=torch.float32)

        num_nodes = x_num.shape[0]
        if num_nodes >= 2:
            src = torch.arange(0, num_nodes - 1, dtype=torch.long)
            dst = torch.arange(1, num_nodes, dtype=torch.long)
            edge_index = torch.stack([src, dst], dim=0)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        edge_type = torch.zeros(edge_index.shape[1], dtype=torch.long)
        batch = torch.zeros(num_nodes, dtype=torch.long)

        target_feature = self.feature_encoder.activity_feature_name
        activity_vocab = self.feature_encoder.categorical_vocabs.get(target_feature, {"<UNK>": 0})
        target_token = str(prefix.target_event.extra.get(target_feature, prefix.target_event.activity_id))
        y_index = activity_vocab.get(target_token, 0)
        y = torch.tensor([y_index], dtype=torch.long)

        return GraphTensorContract(
            x_cat=x_cat,
            x_num=x_num,
            edge_index=edge_index,
            edge_type=edge_type,
            y=y,
            batch=batch,
        )
