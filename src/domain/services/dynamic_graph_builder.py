"""Dynamic graph builder with MVP2 topology-to-tensor injection."""

from __future__ import annotations

import torch

from src.domain.entities.prefix_slice import PrefixSlice
from src.domain.entities.tensor_contract import GraphTensorContract
from src.domain.ports.knowledge_graph_port import IKnowledgeGraphPort
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

        version_key = str(prefix.process_version).strip() or "default"
        dto = self.knowledge_port.get_process_structure(version_key)
        if dto is None or not prefix.prefix_events:
            contract["allowed_target_mask"] = None
            return contract

        target_feature = self.feature_encoder.activity_feature_name
        activity_vocab = self.feature_encoder.categorical_vocabs.get(target_feature, {"<UNK>": 0})
        num_classes = len(activity_vocab)
        allowed_mask = torch.zeros(num_classes, dtype=torch.bool)
        structural_src: list[int] = []
        structural_dst: list[int] = []
        structural_weight: list[float] = []
        edge_stats = dto.edge_statistics or {}

        last_activity = str(
            self.feature_encoder.resolve_event_feature(
                event_extra=prefix.prefix_events[-1].extra,
                feature_name=target_feature,
                default=prefix.prefix_events[-1].activity_id,
            )
        )
        for src, dst in dto.allowed_edges:
            src_token = str(src).strip()
            dst_token = str(dst).strip()
            src_idx = activity_vocab.get(src_token)
            dst_idx = activity_vocab.get(dst_token)
            if src_idx is not None and dst_idx is not None:
                structural_src.append(int(src_idx))
                structural_dst.append(int(dst_idx))
                stats = edge_stats.get((src, dst), {})
                structural_weight.append(float(stats.get("count", 1.0)))

            if str(src) != last_activity:
                continue
            dst_token = str(dst)
            dst_idx = activity_vocab.get(dst_token)
            if dst_idx is None:
                dst_idx = activity_vocab.get(dst_token.strip())
            if dst_idx is not None:
                allowed_mask[dst_idx] = True

        if structural_src:
            contract["structural_edge_index"] = torch.tensor([structural_src, structural_dst], dtype=torch.long)
            contract["structural_edge_weight"] = torch.tensor(structural_weight, dtype=torch.float32)
        else:
            contract["structural_edge_index"] = torch.zeros((2, 0), dtype=torch.long)
            contract["structural_edge_weight"] = torch.zeros((0,), dtype=torch.float32)
        contract["allowed_target_mask"] = allowed_mask
        return contract
