"""EOPKG model variants with structural tensor fusion and baseline fallback."""

from __future__ import annotations

import logging
from typing import Dict

import torch
from torch import nn
from torch_geometric.nn import GATv2Conv, GCNConv, global_mean_pool

from src.domain.entities.feature_config import FeatureLayout
from src.domain.entities.tensor_contract import GraphTensorContract
from src.domain.models.base_gnn import BaseGNN
from src.domain.models.factory import register_model


logger = logging.getLogger(__name__)


class BaseEOPKGModel(BaseGNN):
    """Shared EOPKG logic: local encoder + optional structural fusion."""

    def __init__(
        self,
        feature_layout: FeatureLayout,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.2,
        pooling_strategy: str = "global_mean",
        structural_mode: bool | str = True,
    ) -> None:
        super().__init__()
        self.feature_layout = feature_layout
        self.pooling_strategy = pooling_strategy
        self.hidden_dim = hidden_dim
        self.structural_mode = self._to_bool(structural_mode, default=True)

        self.embeddings = nn.ModuleDict()
        self.embedding_dims: Dict[str, int] = {}
        total_emb_dim = 0
        for name in feature_layout.cat_feature_names:
            vocab_size = int(feature_layout.cat_features[name])
            emb_dim = max(2, min(50, int(6 * (vocab_size ** 0.25))))
            self.embeddings[name] = nn.Embedding(vocab_size, emb_dim)
            self.embedding_dims[name] = emb_dim
            total_emb_dim += emb_dim
        self.input_dim = total_emb_dim + feature_layout.num_dim

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.struct_projector = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )
        self.classifier = nn.Linear(hidden_dim, output_dim)
        self._warned_missing_struct = False

    @staticmethod
    def _to_bool(value: bool | str, *, default: bool) -> bool:
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off"}:
            return False
        return bool(default)

    def _encode_input(self, contract: GraphTensorContract) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_cat = contract["x_cat"]
        x_num = contract["x_num"]
        edge_index = contract["edge_index"]
        batch = contract["batch"]

        emb_parts = []
        for col, name in enumerate(self.feature_layout.cat_feature_names):
            emb_parts.append(self.embeddings[name](x_cat[:, col]))
        x = torch.cat([*emb_parts, x_num], dim=1) if emb_parts else x_num

        if batch.numel() != x.size(0):
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        return x, edge_index, batch

    def _pool_nodes(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        if self.pooling_strategy == "global_mean":
            return global_mean_pool(x, batch)
        if self.pooling_strategy == "last_node":
            if x.size(0) == 0:
                return x.new_zeros((0, x.size(1)))
            num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
            counts = torch.bincount(batch, minlength=num_graphs)
            last_indices = torch.cumsum(counts, dim=0) - 1
            return x[last_indices]
        raise ValueError(f"Unsupported pooling_strategy '{self.pooling_strategy}'.")

    def _build_struct_context(self, contract: GraphTensorContract, batch_size: int, device: torch.device) -> torch.Tensor | None:
        structural_edge_index = contract.get("structural_edge_index")
        if structural_edge_index is None or structural_edge_index.numel() == 0:
            return None

        structural_edge_weight = contract.get("structural_edge_weight")
        if structural_edge_weight is None or structural_edge_weight.numel() == 0:
            structural_edge_weight = torch.ones(
                structural_edge_index.shape[1], dtype=torch.float32, device=device
            )
        else:
            structural_edge_weight = structural_edge_weight.to(device=device, dtype=torch.float32)

        num_struct_edges = float(structural_edge_index.shape[1])
        mean_weight = float(structural_edge_weight.mean().item()) if structural_edge_weight.numel() > 0 else 1.0
        max_index = int(structural_edge_index.max().item()) + 1 if structural_edge_index.numel() > 0 else 1
        density = float(num_struct_edges / max(max_index, 1))

        stats = torch.tensor(
            [[num_struct_edges, mean_weight, density]],
            dtype=torch.float32,
            device=device,
        ).expand(batch_size, -1)
        return self.struct_projector(stats)

    def _forward_local(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, contract: GraphTensorContract) -> torch.Tensor:
        x, edge_index, batch = self._encode_input(contract)
        node_hidden = self._forward_local(x, edge_index)
        obs_context = self._pool_nodes(node_hidden, batch)
        if not self.structural_mode:
            return self.classifier(obs_context)

        struct_context = self._build_struct_context(contract, batch_size=obs_context.shape[0], device=obs_context.device)
        if struct_context is None:
            if not self._warned_missing_struct:
                logger.warning("Structural tensors are missing in contract! Falling back to Baseline forward.")
                self._warned_missing_struct = True
            return self.classifier(obs_context)

        fused = self.fusion(torch.cat([obs_context, struct_context], dim=1))
        return self.classifier(fused)


@register_model("EOPKGGCN")
class EOPKGGCN(BaseEOPKGModel):
    """GCN-based EOPKG model with optional structural fusion."""

    def __init__(
        self,
        feature_layout: FeatureLayout,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.2,
        pooling_strategy: str = "global_mean",
        structural_mode: bool | str = True,
    ) -> None:
        super().__init__(
            feature_layout=feature_layout,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout=dropout,
            pooling_strategy=pooling_strategy,
            structural_mode=structural_mode,
        )
        self.conv1 = GCNConv(self.input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

    def _forward_local(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = self.activation(x)
        x = self.dropout(x)
        return x


@register_model("EOPKGGATv2")
class EOPKGGATv2(BaseEOPKGModel):
    """Dual-encoder EOPKG GATv2 with soft cross-attention fusion."""

    def __init__(
        self,
        feature_layout: FeatureLayout,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.2,
        pooling_strategy: str = "global_mean",
        structural_mode: bool | str = True,
        struct_encoder_type: str = "GATv2Conv",
        struct_hidden_dim: int | None = None,
        cross_attention_heads: int = 4,
    ) -> None:
        super().__init__(
            feature_layout=feature_layout,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout=dropout,
            pooling_strategy=pooling_strategy,
            structural_mode=structural_mode,
        )
        self.num_classes = int(output_dim)
        self.struct_hidden_dim = int(struct_hidden_dim if struct_hidden_dim is not None else hidden_dim)
        self.struct_encoder_type = str(struct_encoder_type or "GATv2Conv")
        self.cross_attention_heads = int(cross_attention_heads)

        self.conv1 = GATv2Conv(self.input_dim, hidden_dim, heads=4, concat=True, dropout=dropout)
        self.conv2 = GATv2Conv(hidden_dim * 4, hidden_dim, heads=1, concat=True, dropout=dropout)
        self.struct_node_emb = nn.Embedding(self.num_classes, self.struct_hidden_dim)
        self.struct_input_proj: nn.Linear | None = None
        self.struct_gnn = self._build_struct_encoder(dropout=dropout)
        self.struct_to_attn = (
            nn.Identity() if self.struct_hidden_dim == hidden_dim else nn.Linear(self.struct_hidden_dim, hidden_dim)
        )
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=self.cross_attention_heads,
            batch_first=True,
        )
        self.attn_to_struct = (
            nn.Identity() if self.struct_hidden_dim == hidden_dim else nn.Linear(hidden_dim, self.struct_hidden_dim)
        )
        self.fusion = nn.Sequential(nn.Linear(hidden_dim + self.struct_hidden_dim, hidden_dim), nn.ReLU())
        self.last_cross_attn_weights: torch.Tensor | None = None

    def _build_struct_encoder(self, dropout: float) -> nn.Module:
        if self.struct_encoder_type == "GATv2Conv":
            return GATv2Conv(
                self.struct_hidden_dim,
                self.struct_hidden_dim,
                heads=1,
                concat=False,
                dropout=dropout,
            )
        if self.struct_encoder_type == "GCNConv":
            return GCNConv(self.struct_hidden_dim, self.struct_hidden_dim)
        raise ValueError(
            f"Unsupported model.struct_encoder_type '{self.struct_encoder_type}'. "
            "Available: ['GATv2Conv', 'GCNConv']"
        )

    def _build_struct_node_features(self, contract: GraphTensorContract, device: torch.device) -> torch.Tensor:
        struct_x = contract.get("struct_x")
        if isinstance(struct_x, torch.Tensor):
            struct_x = struct_x.to(device=device, dtype=torch.float32)
            if struct_x.dim() == 1:
                struct_x = struct_x.unsqueeze(-1)
            if struct_x.size(0) != self.num_classes:
                logger.warning(
                    "struct_x node count mismatch (got=%d expected=%d). Falling back to structural embeddings.",
                    int(struct_x.size(0)),
                    self.num_classes,
                )
            else:
                if struct_x.size(1) != self.struct_hidden_dim:
                    in_dim = int(struct_x.size(1))
                    if self.struct_input_proj is None or int(self.struct_input_proj.in_features) != in_dim:
                        self.struct_input_proj = nn.Linear(in_dim, self.struct_hidden_dim).to(device)
                    return self.struct_input_proj(struct_x)
                return struct_x

        indices = torch.arange(self.num_classes, device=device, dtype=torch.long)
        return self.struct_node_emb(indices)

    def _forward_local(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = self.activation(x)
        x = self.dropout(x)
        return x

    def forward(self, contract: GraphTensorContract) -> torch.Tensor:
        x, edge_index, batch = self._encode_input(contract)
        node_hidden = self._forward_local(x, edge_index)
        obs_context = self._pool_nodes(node_hidden, batch)
        batch_size = int(obs_context.shape[0])
        if not self.structural_mode:
            self.last_cross_attn_weights = None
            return self.classifier(obs_context)

        structural_edge_index = contract.get("structural_edge_index")
        if structural_edge_index is None or structural_edge_index.numel() == 0:
            if not self._warned_missing_struct:
                logger.warning("Structural tensors are missing in contract! Falling back to Baseline forward.")
                self._warned_missing_struct = True
            self.last_cross_attn_weights = None
            return self.classifier(obs_context)

        structural_edge_index = structural_edge_index.to(device=obs_context.device, dtype=torch.long)
        struct_nodes = self._build_struct_node_features(contract, device=obs_context.device)
        node_count = int(struct_nodes.size(0))
        if node_count <= 0:
            if not self._warned_missing_struct:
                logger.warning("Structural node tensor is empty. Falling back to Baseline forward.")
                self._warned_missing_struct = True
            self.last_cross_attn_weights = None
            return self.classifier(obs_context)
        if int(structural_edge_index.max().item()) >= node_count:
            structural_edge_index = structural_edge_index % node_count
        h_norm = self.struct_gnn(struct_nodes, structural_edge_index)
        h_norm = self.activation(h_norm)
        h_norm = self.dropout(h_norm)

        query = obs_context.unsqueeze(1)  # [B, 1, hidden_dim]
        kv = self.struct_to_attn(h_norm).unsqueeze(0).expand(batch_size, -1, -1)  # [B, C, hidden_dim]
        attn_output, attn_weights = self.cross_attention(
            query=query,
            key=kv,
            value=kv,
            need_weights=True,
            average_attn_weights=False,
        )
        self.last_cross_attn_weights = attn_weights.detach()

        context_struct = self.attn_to_struct(attn_output.squeeze(1))  # [B, struct_hidden_dim]
        fused = self.fusion(torch.cat([obs_context, context_struct], dim=1))
        return self.classifier(fused)
