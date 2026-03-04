"""DeepGCN baseline model for MVP1 next-activity prediction."""

# Відповідно до:
# - AGENT_GUIDE.MD -> розділ 2 (MVP1 baseline, без Critic/Reliability)
# - LLD_MVP1.MD -> розділ 6.1 (Message Passing + global_mean_pool + classification head)
# - DATA_MODEL_MVP1.MD -> розділ 6.2 (GraphTensorContract fields)

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import GCNConv, global_mean_pool

from src.domain.entities.tensor_contract import GraphTensorContract
from src.domain.models.base_gnn import BaseGNN


class BaselineGCN(BaseGNN):
    """3-layer GCN with selective residual connections for MVP1 baseline."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        # Шар 1: перетворення input_dim -> hidden_dim (без residual через різну розмірність).
        self.conv1 = GCNConv(input_dim, hidden_dim)
        # Шари 2-3: hidden_dim -> hidden_dim (дозволяють skip connection).
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        # Класифікаційна голова повертає logits для activity classes.
        self.classifier = nn.Linear(hidden_dim, output_dim)

    def forward(self, contract: GraphTensorContract) -> torch.Tensor:
        """Compute logits [batch_size, num_classes] for one PyG batch."""
        x = contract["x"]
        edge_index = contract["edge_index"]
        batch = contract["batch"]

        # Перший message-passing блок без residual (розмірність змінюється).
        x = self.conv1(x, edge_index)
        x = self.activation(x)
        x = self.dropout(x)

        # Другий блок: hidden_dim -> hidden_dim.
        residual = x
        x = self.conv2(x, edge_index)
        x = self.activation(x)
        x = self.dropout(x)
        # Residual додається лише при збігу розмірностей.
        if x.shape == residual.shape:
            x = x + residual

        # Третій блок: hidden_dim -> hidden_dim.
        residual = x
        x = self.conv3(x, edge_index)
        x = self.activation(x)
        x = self.dropout(x)
        # Residual додається лише при збігу розмірностей.
        if x.shape == residual.shape:
            x = x + residual

        # Graph-level readout через середнє по вузлах кожного графа в батчі.
        pooled = global_mean_pool(x, batch)
        # Лише logits активності (без time-regression у MVP1).
        logits = self.classifier(pooled)
        return logits
