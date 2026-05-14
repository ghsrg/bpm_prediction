"""EOPKG model variants with structural tensor fusion and baseline fallback."""

from __future__ import annotations

import math
import logging
from typing import Any, Dict, Mapping

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
        fusion_mode: str = "concat_mlp",
        structural_score_mode: str = "bilinear_with_prior",
        structural_logit_scale_init: float = 0.1,
        structural_logit_scale_max: float = 2.0,
        structural_observed_scale_min: float = 1.0,
        structural_observed_scale_max: float = 10.0,
        structural_stats_beta: float = 0.1,
        topology_state_beta: float = 0.5,
        topology_state_beta_max: float = 2.0,
        topology_state_gate_init_bias: float = -2.0,
        topology_state_class_pooling: str = "logmeanexp",
        topology_state_dropout: float | None = None,
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
        raw_fusion_mode = str(fusion_mode or "attention").strip().lower()
        alias_map = {
            "attention": "class_mean_attention",
            "classmeanattention": "class_mean_attention",
            "class_mean_attention": "class_mean_attention",
            "concat": "class_mean_concat",
            "classmeanconcat": "class_mean_concat",
            "class_mean_concat": "class_mean_concat",
            # Backward compatibility aliases.
            "concat_mlp": "class_mean_attention",
            "struct_pool_concat": "class_mean_concat",
            "classawareadditive": "class_aware_structural_scoring",
            "class_aware_additive": "class_aware_structural_scoring",
            "classawareattention": "class_aware_structural_scoring",
            "class_aware_attention": "class_aware_structural_scoring",
            "classawarestructuralscoring": "class_aware_structural_scoring",
            "class_aware_structural_scoring": "class_aware_structural_scoring",
            "class_aware": "class_aware_structural_scoring",
            "candidate_scoring": "class_aware_structural_scoring",
            "topologystateencoder": "topology_state_encoder",
            "topology_state_encoder": "topology_state_encoder",
            "topology_state": "topology_state_encoder",
            "earlytopologystateencoder": "topology_state_encoder",
        }
        self.fusion_mode = alias_map.get(raw_fusion_mode, raw_fusion_mode)
        allowed_fusion_modes = {
            "class_mean_attention",
            "class_mean_concat",
            "class_aware_structural_scoring",
            "topology_state_encoder",
        }
        if self.fusion_mode not in allowed_fusion_modes:
            raise ValueError(
                f"Unsupported model.fusion_mode '{self.fusion_mode}'. "
                "Available: ['ClassMeanAttention', 'ClassMeanConcat', "
                "'ClassAwareStructuralScoring', 'TopologyStateEncoder']"
            )
        self.structural_score_mode = str(structural_score_mode or "bilinear_with_prior").strip().lower()
        if self.structural_score_mode not in {"bilinear_with_prior", "cosine"}:
            raise ValueError(
                "Unsupported model.structural_score_mode "
                f"'{self.structural_score_mode}'. Available: ['bilinear_with_prior', 'cosine']"
            )
        self.structural_logit_scale_max = float(structural_logit_scale_max)
        self.structural_observed_scale_min = float(structural_observed_scale_min)
        self.structural_observed_scale_max = float(structural_observed_scale_max)
        self.structural_stats_beta = float(structural_stats_beta)
        self.topology_state_beta = float(topology_state_beta)
        self.topology_state_beta_max = float(topology_state_beta_max)
        self.topology_state_gate_init_bias = float(topology_state_gate_init_bias)
        self.topology_state_class_pooling = str(topology_state_class_pooling or "logmeanexp").strip().lower()
        if self.topology_state_class_pooling not in {"logmeanexp", "mean", "max", "logsumexp"}:
            raise ValueError(
                "Unsupported model.topology_state_class_pooling "
                f"'{self.topology_state_class_pooling}'. "
                "Available: ['logmeanexp', 'mean', 'max', 'logsumexp']"
            )
        self.topology_state_dropout_p = float(dropout if topology_state_dropout is None else topology_state_dropout)

        self.conv1 = GATv2Conv(self.input_dim, hidden_dim, heads=4, concat=True, dropout=dropout)
        self.conv2 = GATv2Conv(hidden_dim * 4, hidden_dim, heads=1, concat=True, dropout=dropout)
        self.struct_node_emb = nn.Embedding(self.num_classes, self.struct_hidden_dim)
        self.struct_non_target_emb = nn.Parameter(torch.zeros(self.struct_hidden_dim, dtype=torch.float32))
        self.struct_input_proj: nn.Linear | None = None
        self.struct_input_norm = nn.LayerNorm(self.struct_hidden_dim)
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
        self.struct_query_proj = (
            nn.Identity() if hidden_dim == self.struct_hidden_dim else nn.Linear(hidden_dim, self.struct_hidden_dim)
        )
        self.struct_key_proj = nn.Identity()
        self.structural_logit_scale = nn.Parameter(torch.tensor(float(structural_logit_scale_init), dtype=torch.float32))
        self.structural_class_norm = nn.LayerNorm(output_dim, elementwise_affine=False)
        self.struct_bilinear = nn.Bilinear(self.struct_hidden_dim, self.struct_hidden_dim, 1, bias=False)
        self.struct_prior_head = nn.Linear(self.struct_hidden_dim, 1)
        self.topology_state_proj = nn.Linear(6, self.struct_hidden_dim)
        self.topology_state_gate_proj = nn.Linear(6, self.struct_hidden_dim)
        self.topology_state_norm = nn.LayerNorm(self.struct_hidden_dim)
        self.topology_state_dropout = nn.Dropout(p=self.topology_state_dropout_p)
        self.topology_state_node_head = nn.Linear(self.struct_hidden_dim, 1)
        self.fusion = nn.Sequential(nn.Linear(hidden_dim + self.struct_hidden_dim, hidden_dim), nn.ReLU())
        self.last_cross_attn_weights: torch.Tensor | None = None
        self.last_observed_logits: torch.Tensor | None = None
        self.last_structural_node_logits: torch.Tensor | None = None
        self.last_structural_raw_class_logits: torch.Tensor | None = None
        self.last_structural_normalized_class_logits: torch.Tensor | None = None
        self.last_structural_observed_scale: torch.Tensor | None = None
        self.last_structural_class_logits: torch.Tensor | None = None
        self.last_topology_state_node_logits: torch.Tensor | None = None
        self.last_topology_state_class_logits: torch.Tensor | None = None
        self.last_topology_state_prefix_x: torch.Tensor | None = None
        self.last_topology_state_prefix_mean_abs: torch.Tensor | None = None
        self.last_topology_state_prefix_max_abs: torch.Tensor | None = None
        self.last_topology_state_entropy: torch.Tensor | None = None
        self.last_topology_state_mean_class_cardinality: torch.Tensor | None = None
        self.last_topology_state_max_class_cardinality: torch.Tensor | None = None
        self.last_topology_state_gate_mean: torch.Tensor | None = None
        self.last_topology_state_gate_max: torch.Tensor | None = None

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

    def _build_struct_identity_features(
        self,
        contract: Mapping[str, Any],
        *,
        device: torch.device,
        num_struct_nodes: int,
    ) -> torch.Tensor:
        node_count = int(num_struct_nodes)
        node_to_class = contract.get("struct_node_to_class_index")
        if isinstance(node_to_class, torch.Tensor) and int(node_to_class.numel()) == node_count:
            node_to_class = node_to_class.to(device=device, dtype=torch.long).reshape(-1)
            safe_class_ids = torch.clamp(node_to_class, min=0, max=max(self.num_classes - 1, 0))
            identity = self.struct_node_emb(safe_class_ids)
            non_target_mask = node_to_class < 0
            if torch.any(non_target_mask):
                identity = identity.clone()
                identity[non_target_mask] = self.struct_non_target_emb.to(device=device, dtype=identity.dtype)
            return identity

        node_ids = torch.arange(node_count, device=device, dtype=torch.long)
        safe_class_ids = torch.clamp(node_ids, max=max(self.num_classes - 1, 0))
        identity = self.struct_node_emb(safe_class_ids)
        overflow_mask = node_ids >= self.num_classes
        if torch.any(overflow_mask):
            identity = identity.clone()
            identity[overflow_mask] = self.struct_non_target_emb.to(device=device, dtype=identity.dtype)
        return identity

    def _build_struct_node_features(
        self,
        contract: GraphTensorContract,
        device: torch.device,
        *,
        num_struct_nodes: int | None = None,
    ) -> torch.Tensor:
        struct_x = contract.get("struct_x")
        if isinstance(struct_x, torch.Tensor):
            struct_x = struct_x.to(device=device, dtype=torch.float32)
            if struct_x.dim() == 1:
                struct_x = struct_x.unsqueeze(-1)
            if num_struct_nodes is not None and int(struct_x.size(0)) != int(num_struct_nodes):
                raise ValueError(
                    "struct_x row count must match structural node count: "
                    f"struct_x_rows={int(struct_x.size(0))} num_struct_nodes={int(num_struct_nodes)}."
                )
            node_count = int(num_struct_nodes) if num_struct_nodes is not None else int(struct_x.size(0))
            if struct_x.size(1) != self.struct_hidden_dim:
                in_dim = int(struct_x.size(1))
                if self.struct_input_proj is None or int(self.struct_input_proj.in_features) != in_dim:
                    self.struct_input_proj = nn.Linear(in_dim, self.struct_hidden_dim).to(device)
                stats_features = self.struct_input_proj(struct_x)
            else:
                stats_features = struct_x
            identity_features = self._build_struct_identity_features(
                contract,
                device=device,
                num_struct_nodes=node_count,
            )
            beta = float(self.structural_stats_beta)
            return self.struct_input_norm(identity_features + beta * stats_features)

        if num_struct_nodes is not None and int(num_struct_nodes) != self.num_classes:
            raise ValueError(
                "struct_x is required when structural node count differs from output_dim: "
                f"num_struct_nodes={int(num_struct_nodes)} output_dim={self.num_classes}."
            )

        indices = torch.arange(self.num_classes, device=device, dtype=torch.long)
        return self.struct_node_emb(indices)

    def _resolve_struct_node_to_class_index(
        self,
        *,
        contract: Mapping[str, Any],
        num_struct_nodes: int,
        num_classes: int,
        device: torch.device,
    ) -> torch.Tensor:
        node_to_class = contract.get("struct_node_to_class_index")
        if node_to_class is None:
            if int(num_struct_nodes) == int(num_classes):
                return torch.arange(num_classes, dtype=torch.long, device=device)
            raise ValueError(
                "ClassAwareStructuralScoring requires struct_node_to_class_index "
                f"when num_struct_nodes={num_struct_nodes} differs from num_classes={num_classes}."
            )

        node_to_class = node_to_class.to(device=device, dtype=torch.long).reshape(-1)
        if int(node_to_class.numel()) != int(num_struct_nodes):
            raise ValueError("struct_node_to_class_index length must match the number of structural nodes.")
        if torch.any(node_to_class >= int(num_classes)):
            raise ValueError("struct_node_to_class_index contains class ids outside output_dim.")
        if torch.any(node_to_class < -1):
            raise ValueError("struct_node_to_class_index may contain -1 or non-negative class ids only.")
        return node_to_class

    def _scale_structural_class_logits(
        self,
        *,
        raw_class_logits: torch.Tensor,
        observed_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        normalized = self.structural_class_norm(raw_class_logits)
        observed_scale = observed_logits.detach().abs().mean(dim=1, keepdim=True)
        observed_scale = torch.clamp(
            observed_scale,
            min=float(self.structural_observed_scale_min),
            max=float(self.structural_observed_scale_max),
        )
        gamma = torch.clamp(
            self.structural_logit_scale,
            min=0.0,
            max=float(self.structural_logit_scale_max),
        )
        scaled = normalized * observed_scale * gamma
        return scaled, normalized, observed_scale

    def _class_aware_structural_logits(
        self,
        *,
        obs_context: torch.Tensor,
        h_node: torch.Tensor,
        node_to_class: torch.Tensor,
        num_classes: int,
        observed_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        query = self.struct_query_proj(obs_context)
        keys = self.struct_key_proj(h_node)
        if self.structural_score_mode == "cosine":
            query_norm = torch.nn.functional.normalize(query, p=2, dim=1, eps=1e-12)
            keys_norm = torch.nn.functional.normalize(keys, p=2, dim=1, eps=1e-12)
            node_logits = torch.matmul(query_norm, keys_norm.transpose(0, 1))
        else:
            expanded_query = query.unsqueeze(1).expand(-1, keys.size(0), -1)
            expanded_keys = keys.unsqueeze(0).expand(query.size(0), -1, -1)
            bilinear_scores = self.struct_bilinear(expanded_query, expanded_keys).squeeze(-1)
            prior_scores = self.struct_prior_head(keys).transpose(0, 1)
            node_logits = bilinear_scores + prior_scores
        raw_class_logits = torch.full(
            (query.size(0), int(num_classes)),
            fill_value=torch.finfo(node_logits.dtype).min,
            dtype=node_logits.dtype,
            device=node_logits.device,
        )
        for class_idx in range(int(num_classes)):
            node_mask = node_to_class == class_idx
            if torch.any(node_mask):
                raw_class_logits[:, class_idx] = torch.logsumexp(node_logits[:, node_mask], dim=1)
        raw_class_logits = torch.where(
            torch.isfinite(raw_class_logits),
            raw_class_logits,
            torch.zeros_like(raw_class_logits),
        )
        scaled_class_logits, normalized_class_logits, observed_scale = self._scale_structural_class_logits(
            raw_class_logits=raw_class_logits,
            observed_logits=observed_logits,
        )
        return node_logits, raw_class_logits, normalized_class_logits, observed_scale, scaled_class_logits

    def _clear_structural_diagnostics(self) -> None:
        self.last_cross_attn_weights = None
        self.last_observed_logits = None
        self.last_structural_node_logits = None
        self.last_structural_raw_class_logits = None
        self.last_structural_normalized_class_logits = None
        self.last_structural_observed_scale = None
        self.last_structural_class_logits = None
        self.last_topology_state_node_logits = None
        self.last_topology_state_class_logits = None
        self.last_topology_state_prefix_x = None
        self.last_topology_state_prefix_mean_abs = None
        self.last_topology_state_prefix_max_abs = None
        self.last_topology_state_entropy = None
        self.last_topology_state_mean_class_cardinality = None
        self.last_topology_state_max_class_cardinality = None
        self.last_topology_state_gate_mean = None
        self.last_topology_state_gate_max = None

    def _build_topology_state_input(
        self,
        *,
        base_struct: torch.Tensor,
        prefix_state_x: torch.Tensor,
    ) -> torch.Tensor:
        if prefix_state_x.dim() == 2:
            prefix_state_x = prefix_state_x.unsqueeze(0)
        if prefix_state_x.dim() != 3:
            raise ValueError("TopologyStateEncoder requires struct_prefix_state_x with shape [V, F] or [B, V, F].")
        prefix_state_x = prefix_state_x.to(device=base_struct.device, dtype=torch.float32)
        if int(prefix_state_x.size(1)) != int(base_struct.size(0)):
            raise ValueError(
                "struct_prefix_state_x row count must match structural node count: "
                f"prefix_rows={int(prefix_state_x.size(1))} num_struct_nodes={int(base_struct.size(0))}."
            )
        in_dim = int(prefix_state_x.size(2))
        if in_dim != int(self.topology_state_proj.in_features):
            raise ValueError(
                "TopologyStateEncoder requires struct_prefix_state_x feature dimension 6: "
                f"got={in_dim}."
            )
        prefix_features = self.topology_state_dropout(self.topology_state_proj(prefix_state_x))
        gate = torch.sigmoid(self.topology_state_gate_proj(prefix_state_x) + float(self.topology_state_gate_init_bias))
        beta = max(0.0, min(float(self.topology_state_beta), float(self.topology_state_beta_max)))
        h0 = self.topology_state_norm(base_struct.unsqueeze(0) + float(beta) * gate * prefix_features)
        self.last_topology_state_prefix_x = prefix_state_x.detach()
        self.last_topology_state_prefix_mean_abs = prefix_state_x.detach().abs().mean()
        self.last_topology_state_prefix_max_abs = prefix_state_x.detach().abs().max()
        self.last_topology_state_gate_mean = gate.detach().mean()
        self.last_topology_state_gate_max = gate.detach().max()
        return h0

    def _run_topology_state_gnn(self, h0: torch.Tensor, structural_edge_index: torch.Tensor) -> torch.Tensor:
        batch_size, node_count, dim = int(h0.size(0)), int(h0.size(1)), int(h0.size(2))
        flat = h0.reshape(batch_size * node_count, dim)
        if structural_edge_index.numel() == 0:
            return flat.reshape(batch_size, node_count, dim)
        edge_index = structural_edge_index.to(device=h0.device, dtype=torch.long)
        offsets = (torch.arange(batch_size, device=h0.device, dtype=torch.long) * node_count).view(batch_size, 1, 1)
        batched_edge_index = (edge_index.unsqueeze(0) + offsets).permute(1, 0, 2).reshape(2, -1)
        h = self.struct_gnn(flat, batched_edge_index)
        h = self.activation(h)
        h = self.dropout(h)
        return h.reshape(batch_size, node_count, self.struct_hidden_dim)

    def _pool_topology_node_logits(
        self,
        *,
        node_logits: torch.Tensor,
        node_to_class: torch.Tensor,
        num_classes: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        class_logits = torch.zeros(
            (int(node_logits.size(0)), int(num_classes)),
            dtype=node_logits.dtype,
            device=node_logits.device,
        )
        valid = node_to_class >= 0
        cardinalities = torch.bincount(
            node_to_class[valid].to(dtype=torch.long),
            minlength=int(num_classes),
        ).to(device=node_logits.device, dtype=torch.float32)
        for class_idx in range(int(num_classes)):
            node_mask = node_to_class == class_idx
            if not torch.any(node_mask):
                continue
            values = node_logits[:, node_mask]
            if self.topology_state_class_pooling == "logmeanexp":
                class_logits[:, class_idx] = torch.logsumexp(values, dim=1) - math.log(int(values.size(1)))
            elif self.topology_state_class_pooling == "mean":
                class_logits[:, class_idx] = values.mean(dim=1)
            elif self.topology_state_class_pooling == "max":
                class_logits[:, class_idx] = values.max(dim=1).values
            elif self.topology_state_class_pooling == "logsumexp":
                class_logits[:, class_idx] = torch.logsumexp(values, dim=1)
        return class_logits, cardinalities

    def _forward_topology_state(
        self,
        *,
        contract: GraphTensorContract,
        structural_edge_index: torch.Tensor,
        struct_nodes: torch.Tensor,
        node_to_class: torch.Tensor,
    ) -> torch.Tensor:
        prefix_state_x = contract.get("struct_prefix_state_x")
        if not isinstance(prefix_state_x, torch.Tensor):
            raise ValueError("TopologyStateEncoder requires struct_prefix_state_x.")
        h0 = self._build_topology_state_input(
            base_struct=struct_nodes,
            prefix_state_x=prefix_state_x,
        )
        h_topology = self._run_topology_state_gnn(h0, structural_edge_index)
        node_logits = self.topology_state_node_head(h_topology).squeeze(-1)
        class_logits, cardinalities = self._pool_topology_node_logits(
            node_logits=node_logits,
            node_to_class=node_to_class.to(device=node_logits.device, dtype=torch.long),
            num_classes=self.num_classes,
        )
        probs = torch.softmax(class_logits, dim=1)
        entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=1).mean()
        self.last_cross_attn_weights = None
        self.last_observed_logits = None
        self.last_structural_node_logits = None
        self.last_structural_raw_class_logits = None
        self.last_structural_normalized_class_logits = None
        self.last_structural_observed_scale = None
        self.last_structural_class_logits = None
        self.last_topology_state_node_logits = node_logits
        self.last_topology_state_class_logits = class_logits
        self.last_topology_state_entropy = entropy.detach()
        nonzero_cardinalities = cardinalities[cardinalities > 0]
        if nonzero_cardinalities.numel() > 0:
            self.last_topology_state_mean_class_cardinality = nonzero_cardinalities.mean().detach()
            self.last_topology_state_max_class_cardinality = nonzero_cardinalities.max().detach()
        else:
            self.last_topology_state_mean_class_cardinality = cardinalities.new_tensor(0.0)
            self.last_topology_state_max_class_cardinality = cardinalities.new_tensor(0.0)
        return class_logits

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
            self._clear_structural_diagnostics()
            return self.classifier(obs_context)

        structural_edge_index = contract.get("structural_edge_index")
        if structural_edge_index is None or structural_edge_index.numel() == 0:
            if not self._warned_missing_struct:
                logger.warning("Structural tensors are missing in contract! Falling back to Baseline forward.")
                self._warned_missing_struct = True
            self._clear_structural_diagnostics()
            return self.classifier(obs_context)

        structural_edge_index = structural_edge_index.to(device=obs_context.device, dtype=torch.long)
        structural_edge_index_max = int(structural_edge_index.max().item())
        raw_struct_x = contract.get("struct_x")
        if isinstance(raw_struct_x, torch.Tensor):
            raw_struct_x_rows = int(raw_struct_x.size(0)) if raw_struct_x.dim() > 0 else 0
            if structural_edge_index_max >= raw_struct_x_rows:
                raise ValueError(
                    "Structural edge index is out of bounds: "
                    f"max_index={structural_edge_index_max} node_count={raw_struct_x_rows} "
                    f"struct_x_shape={tuple(raw_struct_x.shape)}."
                )
        elif (
            self.fusion_mode not in {"class_aware_structural_scoring", "topology_state_encoder"}
            and structural_edge_index_max >= self.num_classes
        ):
            raise ValueError(
                "Structural edge index is out of bounds: "
                f"max_index={structural_edge_index_max} node_count={self.num_classes} "
                "struct_x_shape=None."
            )
        if isinstance(raw_struct_x, torch.Tensor):
            num_struct_nodes = int(raw_struct_x.size(0)) if raw_struct_x.dim() > 0 else 0
        elif structural_edge_index_max < self.num_classes:
            num_struct_nodes = self.num_classes
        else:
            num_struct_nodes = int(structural_edge_index.max().item()) + 1 if structural_edge_index.numel() > 0 else None
        struct_nodes = self._build_struct_node_features(
            contract,
            device=obs_context.device,
            num_struct_nodes=num_struct_nodes,
        )
        node_count = int(struct_nodes.size(0))
        if node_count <= 0:
            if not self._warned_missing_struct:
                logger.warning("Structural node tensor is empty. Falling back to Baseline forward.")
                self._warned_missing_struct = True
            self._clear_structural_diagnostics()
            return self.classifier(obs_context)
        if structural_edge_index_max >= node_count:
            raise ValueError(
                "Structural edge index is out of bounds: "
                f"max_index={structural_edge_index_max} node_count={node_count} "
                f"struct_x_shape={tuple(struct_nodes.shape)}."
            )
        if self.fusion_mode == "topology_state_encoder":
            node_to_class = self._resolve_struct_node_to_class_index(
                contract=contract,
                num_struct_nodes=struct_nodes.size(0),
                num_classes=self.num_classes,
                device=struct_nodes.device,
            )
            return self._forward_topology_state(
                contract=contract,
                structural_edge_index=structural_edge_index,
                struct_nodes=struct_nodes,
                node_to_class=node_to_class,
            )

        h_norm = self.struct_gnn(struct_nodes, structural_edge_index)
        h_norm = self.activation(h_norm)
        h_norm = self.dropout(h_norm)

        if self.fusion_mode == "class_aware_structural_scoring":
            observed_logits = self.classifier(obs_context)
            node_to_class = self._resolve_struct_node_to_class_index(
                contract=contract,
                num_struct_nodes=h_norm.size(0),
                num_classes=observed_logits.size(1),
                device=h_norm.device,
            )
            (
                structural_node_logits,
                structural_raw_class_logits,
                structural_normalized_class_logits,
                structural_observed_scale,
                structural_class_logits,
            ) = self._class_aware_structural_logits(
                obs_context=obs_context,
                h_node=h_norm,
                node_to_class=node_to_class,
                num_classes=observed_logits.size(1),
                observed_logits=observed_logits,
            )
            self.last_cross_attn_weights = None
            self.last_observed_logits = observed_logits
            self.last_structural_node_logits = structural_node_logits
            self.last_structural_raw_class_logits = structural_raw_class_logits
            self.last_structural_normalized_class_logits = structural_normalized_class_logits
            self.last_structural_observed_scale = structural_observed_scale
            self.last_structural_class_logits = structural_class_logits
            return observed_logits + structural_class_logits

        if self.fusion_mode == "class_mean_concat":
            # Structural context is a pooled summary of structural node states.
            context_struct = h_norm.mean(dim=0, keepdim=True).expand(batch_size, -1)
            self._clear_structural_diagnostics()
        else:
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
            self.last_observed_logits = None
            self.last_structural_node_logits = None
            self.last_structural_raw_class_logits = None
            self.last_structural_normalized_class_logits = None
            self.last_structural_observed_scale = None
            self.last_structural_class_logits = None
            context_struct = self.attn_to_struct(attn_output.squeeze(1))  # [B, struct_hidden_dim]
        fused = self.fusion(torch.cat([obs_context, context_struct], dim=1))
        return self.classifier(fused)
