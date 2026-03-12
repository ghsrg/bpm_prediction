"""Dynamic graph builder with MVP2 topology-to-tensor injection."""

from __future__ import annotations

import torch

from src.application.ports.knowledge_graph_port import IKnowledgeGraphPort
from src.domain.entities.prefix_slice import PrefixSlice
from src.domain.entities.tensor_contract import GraphTensorContract
from src.domain.services.baseline_graph_builder import BaselineGraphBuilder
from src.domain.services.feature_encoder import FeatureEncoder


class DynamicGraphBuilder(BaselineGraphBuilder):
    """Extends baseline tensors with version-scoped allowed-target mask."""

    def __init__(self, feature_encoder: FeatureEncoder, knowledge_port: IKnowledgeGraphPort) -> None:
        super().__init__(feature_encoder=feature_encoder)
        self.knowledge_port = knowledge_port

    def build_graph(self, prefix: PrefixSlice) -> GraphTensorContract:
        """Build baseline contract and inject OOS mask when topology is available."""
        contract = super().build_graph(prefix)

        dto = self.knowledge_port.get_process_structure(str(prefix.process_version))
        if dto is None or not prefix.prefix_events:
            contract["allowed_target_mask"] = None
            return contract

        target_feature = self.feature_encoder.activity_feature_name
        activity_vocab = self.feature_encoder.categorical_vocabs.get(target_feature, {"<UNK>": 0})
        num_classes = len(activity_vocab)
        allowed_mask = torch.zeros(num_classes, dtype=torch.bool)

        last_activity = str(prefix.prefix_events[-1].activity_id)
        for src, dst in dto.allowed_edges:
            if str(src) != last_activity:
                continue
            dst_idx = activity_vocab.get(str(dst))
            if dst_idx is not None:
                allowed_mask[dst_idx] = True

        contract["allowed_target_mask"] = allowed_mask
        return contract

