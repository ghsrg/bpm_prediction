"""Domain tensor DTO contract for graph-based model input."""

# Відповідно до:
# - AGENT_GUIDE.MD -> розділ 4 (GraphTensorContract)
# - LLD_MVP1.MD -> розділ 5.2 (структура GraphTensorContract)
# - DATA_MODEL_MVP1.MD -> розділ 6.2 (тензорний контракт Domain)

from __future__ import annotations

from typing import TypedDict

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
