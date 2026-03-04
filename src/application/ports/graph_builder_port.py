"""Application port contract for graph construction from prefix slices."""

# Відповідно до:
# - AGENT_GUIDE.MD -> розділ 2 (Application залежить від абстракцій)
# - DATA_FLOWS_MVP1.MD -> розділ 2.3 (IGraphBuilder контракт)
# - LLD_MVP1.MD -> розділ 5 (Graph Tensor Construction)

from __future__ import annotations

from typing import Protocol

from src.domain.entities.prefix_slice import PrefixSlice
from src.domain.entities.tensor_contract import GraphTensorContract


class IGraphBuilder(Protocol):
    """Port for transforming a PrefixSlice into GraphTensorContract."""

    def build_graph(self, prefix: PrefixSlice) -> GraphTensorContract:
        """Build graph tensors for one prefix-target sample."""
        ...
