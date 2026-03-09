"""Abstract graph-builder base for MVP1 and future dynamic structural builders."""

# Відповідно до:
# - DATA_FLOWS_MVP1.MD -> IGraphBuilder контракт
# - TARGET_ARCHITECTURE.MD -> підготовка extension point для Dynamic Structural Layer (MVP2)

from __future__ import annotations

from abc import ABC, abstractmethod

from src.domain.entities.prefix_slice import PrefixSlice
from src.domain.entities.tensor_contract import GraphTensorContract


class GraphBuilderBase(ABC):
    """Abstract graph builder extension point.

    MVP1 uses BaselineGraphBuilder. MVP2 can introduce DynamicStructuralGraphBuilder
    (EOPKG-aware) without changing trainer orchestration contract.
    """

    @abstractmethod
    def build_graph(self, prefix: PrefixSlice) -> GraphTensorContract:
        """Build graph tensors from one prefix slice."""
        raise NotImplementedError
