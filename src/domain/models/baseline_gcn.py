"""DeepGCN baseline model for MVP1 next-activity prediction."""

# Відповідно до:
# - AGENT_GUIDE.MD -> розділ 2 (MVP1 baseline)
# - LLD_MVP1.MD -> розділ 6.1 (Message Passing + global_mean_pool + head)
# - DATA_MODEL_MVP1.MD -> розділ 6.2 (GraphTensorContract)

from __future__ import annotations

from typing import Dict

import torch
from torch import nn
from torch_geometric.nn import GCNConv, global_mean_pool

from src.domain.entities.feature_config import FeatureLayout
from src.domain.entities.tensor_contract import GraphTensorContract
from src.domain.models.base_gnn import BaseGNN


class BaselineGCN(BaseGNN):
    """3-layer GCN with per-feature embeddings for categorical inputs."""

    def __init__(self, feature_layout: FeatureLayout, hidden_dim: int, output_dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.feature_layout = feature_layout

        self.embeddings = nn.ModuleDict()
        self.embedding_dims: Dict[str, int] = {}
        total_emb_dim = 0

        for name in feature_layout.cat_feature_names:
            vocab_size = int(feature_layout.cat_features[name])
            emb_dim = max(2, min(50, int(6 * (vocab_size ** 0.25))))
            self.embeddings[name] = nn.Embedding(vocab_size, emb_dim)
            self.embedding_dims[name] = emb_dim
            total_emb_dim += emb_dim

        input_dim = total_emb_dim + feature_layout.num_dim

        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(hidden_dim, output_dim)

    def forward(self, contract: GraphTensorContract) -> torch.Tensor:
        """Compute logits [batch_size, num_classes] for one PyG batch."""
        x_cat = contract["x_cat"]
        x_num = contract["x_num"]
        edge_index = contract["edge_index"]
        batch = contract["batch"]

        emb_parts = []
        for col, name in enumerate(self.feature_layout.cat_feature_names):
            emb_parts.append(self.embeddings[name](x_cat[:, col]))

        if emb_parts:
            x = torch.cat([*emb_parts, x_num], dim=1)
        else:
            x = x_num

        # Гарантуємо консистентність batch-вектора для pool/readout.
        if batch.numel() != x.size(0):
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        x = self.conv1(x, edge_index)
        x = self.activation(x)
        x = self.dropout(x)

        residual = x
        x = self.conv2(x, edge_index)
        x = self.activation(x)
        x = self.dropout(x)
        if x.shape == residual.shape:
            x = x + residual

        residual = x
        x = self.conv3(x, edge_index)
        x = self.activation(x)
        x = self.dropout(x)
        if x.shape == residual.shape:
            x = x + residual

        pooled = global_mean_pool(x, batch)
        return self.classifier(pooled)
