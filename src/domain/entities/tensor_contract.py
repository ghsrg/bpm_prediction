"""Domain tensor DTO contract for graph-based model input."""

# Відповідно до:
# - AGENT_GUIDE.MD -> розділ 4 (GraphTensorContract)
# - LLD_MVP1.MD -> розділ 5.2 (структура GraphTensorContract)
# - DATA_MODEL_MVP1.MD -> розділ 6.2 (тензорний контракт Domain)

from __future__ import annotations

from typing import NotRequired, TypedDict

import torch


class GraphTensorContract(TypedDict):
    """Typed contract for graph tensors consumed by baseline GNN."""

    x_cat: torch.LongTensor
    x_num: torch.FloatTensor
    edge_index: torch.LongTensor
    edge_type: torch.LongTensor
    y: torch.LongTensor
    batch: torch.LongTensor
    num_nodes: int
    struct_x: NotRequired[torch.FloatTensor | None]
    structural_edge_index: NotRequired[torch.LongTensor | None]
    structural_edge_weight: NotRequired[torch.FloatTensor | None]
    version_emb_idx: NotRequired[torch.LongTensor | None]
    allowed_target_mask: NotRequired[torch.BoolTensor | None]
    stats_snapshot_version_seq: NotRequired[int | None]
    stats_snapshot_as_of_epoch: NotRequired[float | None]
    stats_allowed: NotRequired[bool | None]
    stats_missing_asof_snapshot: NotRequired[bool | None]
    stats_snapshot_versions: NotRequired[list[str] | None]
    stats_snapshot_as_of_ts_batch: NotRequired[list[str] | None]
    stats_missing_asof_snapshot_batch: NotRequired[list[bool] | None]
    topology_projection_aligned: NotRequired[bool | None]
    topology_projection_projected_edge_count: NotRequired[int | None]
    topology_projection_source_path_count: NotRequired[int | None]
    topology_projection_skipped_edge_count: NotRequired[int | None]
    topology_projection_missing_vocab_count: NotRequired[int | None]
    topology_projection_duplicate_label_count: NotRequired[int | None]
    topology_projection_missing_node_metadata: NotRequired[bool | None]
    topology_projection_aligned_batch: NotRequired[list[bool] | None]
    topology_projection_projected_edge_count_batch: NotRequired[list[int] | None]
    topology_projection_source_path_count_batch: NotRequired[list[int] | None]
    topology_projection_skipped_edge_count_batch: NotRequired[list[int] | None]
    topology_projection_missing_vocab_count_batch: NotRequired[list[int] | None]
    topology_projection_duplicate_label_count_batch: NotRequired[list[int] | None]
    topology_projection_missing_node_metadata_batch: NotRequired[list[bool] | None]
