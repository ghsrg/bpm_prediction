"""Abstract base interface for MVP1 GNN classifiers."""

# Відповідно до:
# - AGENT_GUIDE.MD -> розділ 2 (MVP1 без Critic/Reliability) і розділ 4 (GraphTensorContract)
# - LLD_MVP1.MD -> розділ 5.2 (тензорний контракт) та 6.2 (єдиний forward-контракт)
# - DATA_FLOWS_MVP1.MD -> розділ 2.4 (порт взаємодії з моделлю)

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn

from src.domain.entities.tensor_contract import GraphTensorContract


class BaseGNN(nn.Module, ABC):
    """Unified interface for graph classifiers in MVP1."""

    @abstractmethod
    def forward(self, contract: GraphTensorContract) -> torch.Tensor:
        """Return activity logits with shape [batch_size, num_classes]."""
        raise NotImplementedError
