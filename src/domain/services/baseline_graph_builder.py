"""Baseline graph builder for MVP1 observed-graph tensors."""

# Відповідно до:
# - AGENT_GUIDE.MD -> розділ 2 (MVP1 без EPOKG/critic), розділ 4 (GraphTensorContract)
# - LLD_MVP1.MD -> розділ 4 (Feature Engineering, OOV->0, Z-score), розділ 5 (лінійний edge_index)
# - DATA_FLOWS_MVP1.MD -> розділ 2.3 (IGraphBuilder контракт)

from __future__ import annotations

from typing import Dict, List

import torch

from src.application.ports.graph_builder_port import IGraphBuilder
from src.domain.entities.prefix_slice import PrefixSlice
from src.domain.entities.tensor_contract import GraphTensorContract


class BaselineGraphBuilder(IGraphBuilder):
    """Stateless builder with fixed vocabularies and normalization statistics."""

    def __init__(
        self,
        activity_vocab: Dict[str, int],
        resource_vocab: Dict[str, int],
        normalization_stats: Dict[str, Dict[str, float]],
        unk_index: int = 0,
    ) -> None:
        self.activity_vocab = activity_vocab
        self.resource_vocab = resource_vocab
        self.normalization_stats = normalization_stats
        self.unk_index = unk_index

        self.num_activities = len(activity_vocab)
        self.num_resources = len(resource_vocab)
        self.input_dim = self.num_activities + self.num_resources + 3

    def build_graph(self, prefix: PrefixSlice) -> GraphTensorContract:
        """Convert PrefixSlice into GraphTensorContract for MVP1 baseline GNN."""
        node_vectors: List[torch.Tensor] = []

        for event in prefix.prefix_events:
            activity_idx = self.activity_vocab.get(event.activity_id, self.unk_index)
            resource_idx = self.resource_vocab.get(event.resource_id, self.unk_index)

            x_act = self._one_hot(activity_idx, self.num_activities)
            x_res = self._one_hot(resource_idx, self.num_resources)

            time_features = torch.tensor(
                [
                    self._zscore(event.duration, "duration"),
                    self._zscore(event.time_since_case_start, "time_since_case_start"),
                    self._zscore(event.time_since_previous_event, "time_since_previous_event"),
                ],
                dtype=torch.float32,
            )

            x_node = torch.cat([x_act, x_res, time_features])
            node_vectors.append(x_node)

        if node_vectors:
            x = torch.stack(node_vectors, dim=0)
        else:
            x = torch.zeros((0, self.input_dim), dtype=torch.float32)

        num_nodes = x.shape[0]
        if num_nodes >= 2:
            src = torch.arange(0, num_nodes - 1, dtype=torch.long)
            dst = torch.arange(1, num_nodes, dtype=torch.long)
            edge_index = torch.stack([src, dst], dim=0)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        edge_type = torch.zeros(edge_index.shape[1], dtype=torch.long)
        batch = torch.zeros(num_nodes, dtype=torch.long)

        y_index = self.activity_vocab.get(prefix.target_event.activity_id, self.unk_index)
        y = torch.tensor([y_index], dtype=torch.long)

        return GraphTensorContract(
            x=x,
            edge_index=edge_index,
            edge_type=edge_type,
            y=y,
            batch=batch,
        )

    def _one_hot(self, index: int, size: int) -> torch.Tensor:
        """Encode categorical index to one-hot float vector with UNK fallback."""
        vector = torch.zeros(size, dtype=torch.float32)
        safe_index = index if 0 <= index < size else self.unk_index
        if 0 <= safe_index < size:
            vector[safe_index] = 1.0
        return vector

    def _zscore(self, value: float, feature_name: str) -> float:
        """Apply Z-score normalization; return 0.0 when sigma is zero or undefined."""
        feature_stats = self.normalization_stats.get(feature_name, {})
        mu = float(feature_stats.get("mu", 0.0))
        sigma = float(feature_stats.get("sigma", 0.0))
        if sigma == 0.0:
            return 0.0
        return (float(value) - mu) / sigma
