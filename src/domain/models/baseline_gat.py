"""GATv2 baseline model for MVP1 next-activity prediction."""

# Відповідно до:
# - AGENT_GUIDE.MD -> розділ 2 (MVP1 baseline без Critic/Reliability)
# - LLD_MVP1.MD -> розділ 6.1 (Message Passing + global_mean_pool + task head)
# - DATA_MODEL_MVP1.MD -> розділ 6.2 (GraphTensorContract)

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import GATv2Conv, global_mean_pool

from src.domain.entities.tensor_contract import GraphTensorContract
from src.domain.models.base_gnn import BaseGNN


class BaselineGATv2(BaseGNN):
    """2-layer GATv2 baseline with multi-head attention on first layer."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        # Шар 1: heads=4, concat=True => вихід hidden_dim * 4.
        self.conv1 = GATv2Conv(input_dim, hidden_dim, heads=4, concat=True, dropout=dropout)
        # Шар 2: вхід має бути hidden_dim * 4, heads=1 => вихід hidden_dim.
        self.conv2 = GATv2Conv(hidden_dim * 4, hidden_dim, heads=1, concat=True, dropout=dropout)

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        # Класифікаційна голова повертає logits для activity classes.
        self.classifier = nn.Linear(hidden_dim, output_dim)

    def forward(self, contract: GraphTensorContract) -> torch.Tensor:
        """Compute logits [batch_size, num_classes] for one PyG batch."""
        x = contract["x"]
        edge_index = contract["edge_index"]
        batch = contract["batch"]

        # Перший attention-block з multi-head репрезентацією.
        x = self.conv1(x, edge_index)
        x = self.activation(x)
        x = self.dropout(x)

        # Другий attention-block проєктує назад до hidden_dim.
        x = self.conv2(x, edge_index)
        x = self.activation(x)
        x = self.dropout(x)

        # Graph-level readout через mean pooling за індексами batch.
        pooled = global_mean_pool(x, batch)
        # Лише logits активності (MVP1 класифікація).
        logits = self.classifier(pooled)
        return logits
