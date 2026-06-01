"""Topology-conditioned candidate scoring model family."""

from __future__ import annotations

import math
from typing import Dict, Any

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATv2Conv, global_mean_pool

from src.domain.entities.candidate_prediction import CandidatePredictionOutput
from src.domain.entities.feature_config import FeatureLayout
from src.domain.entities.tensor_contract import GraphTensorContract
from src.domain.models.base_gnn import BaseGNN
from src.domain.models.factory import register_model


@register_model("EOPKGTopologyConditioned")
class EOPKGTopologyConditioned(BaseGNN):
    """Stage-1 topology-conditioned candidate scorer with fixed-label logits."""

    def __init__(
        self,
        feature_layout: FeatureLayout,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.2,
        pooling_strategy: str = "last_node",
        observed_encoder: str = "GATv2",
        struct_encoder: str = "GATv2",
        candidate_scoring: str = "cosine",
        candidate_pooling: str = "logmeanexp",
        candidate_temperature_init: float = 0.1,
        candidate_temperature_min: float = 0.05,
        candidate_temperature_max: float = 2.0,
        candidate_temperature_trainable: bool = False,
        topology_conditioning_mode: str = "static_candidates",
        impulse_activation_enabled: bool = False,
        impulse_state_channels: list[str] | tuple[str, ...] | None = None,
        impulse_scale_init: float = 0.1,
        impulse_scale_max: float = 2.0,
        impulse_gnn_layers: int = 1,
        impulse_residual_h0: bool = True,
    ) -> None:
        super().__init__()
        self.feature_layout = feature_layout
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.dropout_p = float(dropout)
        self.pooling_strategy = str(pooling_strategy)
        self.observed_encoder = str(observed_encoder or "GATv2").strip().lower()
        self.struct_encoder = str(struct_encoder or "GATv2").strip().lower()
        self.candidate_scoring = str(candidate_scoring or "cosine").strip().lower()
        self.candidate_pooling = str(candidate_pooling or "logmeanexp").strip().lower()
        self.structural_mode = True
        self.fusion_mode = "TopologyConditionedCandidateScoring"
        self.candidate_scoring_mode = self.candidate_scoring
        self.candidate_temperature_min = float(candidate_temperature_min)
        self.candidate_temperature_max = float(candidate_temperature_max)
        self.candidate_temperature_trainable = bool(candidate_temperature_trainable)
        requested_mode = str(topology_conditioning_mode or "static_candidates").strip().lower()
        self.impulse_activation_enabled = bool(impulse_activation_enabled) or requested_mode == "impulse_activation_routing"
        self.topology_conditioning_mode = (
            "impulse_activation_routing" if self.impulse_activation_enabled else requested_mode
        )
        self.impulse_state_channels = tuple(
            impulse_state_channels
            or (
                "prefix_executed_count_log1p",
                "was_executed",
                "is_last_event",
                "last_position_norm",
                "prefix_recency_norm",
                "active_after_complete",
            )
        )
        self.impulse_scale_max = float(impulse_scale_max)
        self.impulse_gnn_layers = int(impulse_gnn_layers)
        self.impulse_residual_h0 = bool(impulse_residual_h0)

        self._validate_config()

        self.embeddings = nn.ModuleDict()
        self.embedding_dims: Dict[str, int] = {}
        total_emb_dim = 0
        for name in feature_layout.cat_feature_names:
            vocab_size = int(feature_layout.cat_features[name])
            emb_dim = max(2, min(50, int(6 * (vocab_size**0.25))))
            self.embeddings[name] = nn.Embedding(vocab_size, emb_dim)
            self.embedding_dims[name] = emb_dim
            total_emb_dim += emb_dim
        input_dim = total_emb_dim + feature_layout.num_dim

        self.observed_conv1 = GATv2Conv(input_dim, self.hidden_dim, heads=4, concat=True, dropout=dropout)
        self.observed_conv2 = GATv2Conv(self.hidden_dim * 4, self.hidden_dim, heads=1, concat=True, dropout=dropout)

        self.struct_input_proj = nn.LazyLinear(self.hidden_dim)
        self.struct_node_emb = nn.Embedding(self.output_dim, self.hidden_dim)
        self.struct_non_target_emb = nn.Parameter(torch.zeros(self.hidden_dim))
        self.struct_norm = nn.LayerNorm(self.hidden_dim)
        self.struct_conv1 = GATv2Conv(self.hidden_dim, self.hidden_dim, heads=1, concat=True, dropout=dropout)
        self.impulse_proj = nn.LazyLinear(self.hidden_dim)
        self.impulse_norm = nn.LayerNorm(self.hidden_dim)
        self.impulse_scale_raw = nn.Parameter(torch.tensor(float(impulse_scale_init), dtype=torch.float32))

        self.hash_vocab_size = 1000
        self.unseen_grounding_emb = nn.Embedding(self.hash_vocab_size, self.hidden_dim)
        nn.init.normal_(self.unseen_grounding_emb.weight, std=0.01)

        self.bilinear = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

        temperature = torch.tensor(float(candidate_temperature_init), dtype=torch.float32)
        if self.candidate_temperature_trainable:
            self.candidate_temperature = nn.Parameter(temperature)
        else:
            self.register_buffer("candidate_temperature", temperature)

        self._reset_candidate_diagnostics()

    def _validate_config(self) -> None:
        if self.observed_encoder not in {"gatv2"}:
            raise ValueError("Unsupported model.observed_encoder for EOPKGTopologyConditioned. Available: ['GATv2']")
        if self.struct_encoder not in {"gatv2"}:
            raise ValueError("Unsupported model.struct_encoder for EOPKGTopologyConditioned. Available: ['GATv2']")
        if self.candidate_scoring not in {"cosine", "bilinear"}:
            raise ValueError("Unsupported model.candidate_scoring. Available: ['cosine', 'bilinear']")
        if self.candidate_pooling not in {"logmeanexp", "mean", "max"}:
            raise ValueError("Unsupported model.candidate_pooling. Available: ['logmeanexp', 'mean', 'max']")
        if self.topology_conditioning_mode not in {"static_candidates", "impulse_activation_routing"}:
            raise ValueError(
                "Unsupported model.topology_conditioning_mode. "
                "Available: ['static_candidates', 'impulse_activation_routing']"
            )
        if self.impulse_scale_max <= 0.0:
            raise ValueError("model.impulse_scale_max must be positive.")
        if self.impulse_gnn_layers < 1 or self.impulse_gnn_layers > 2:
            raise ValueError("model.impulse_gnn_layers must be 1 or 2 for the initial implementation.")
        if self.candidate_temperature_min <= 0.0:
            raise ValueError("model.candidate_temperature_min must be positive.")
        if self.candidate_temperature_max < self.candidate_temperature_min:
            raise ValueError("model.candidate_temperature_max must be >= candidate_temperature_min.")

    def forward(self, contract: GraphTensorContract) -> torch.Tensor:
        """Return fixed-label logits [B, C_train] through topology node scoring."""
        self._reset_candidate_diagnostics()
        obs_context = self._encode_observed_context(contract)
        candidates, node_to_class = self._encode_candidates(contract, obs_context.device)
        node_scores = self._score_candidates(obs_context, candidates)
        class_scores = self._pool_node_scores_to_classes(node_scores, node_to_class)
        class_scores = torch.nan_to_num(class_scores, neginf=-1e6)
        self._record_candidate_diagnostics(
            node_scores=node_scores,
            class_scores=class_scores,
            node_to_class=node_to_class,
            targets=contract.get("y"),
        )
        return class_scores

    def _reset_candidate_diagnostics(self) -> None:
        self.last_candidate_node_score_mean_abs = None
        self.last_candidate_class_score_mean_abs = None
        self.last_duplicate_candidate_count_max = None
        self.last_candidate_temperature = None
        self.last_candidate_temperature_trainable = self.candidate_temperature_trainable
        self.last_candidate_prediction_entropy = None
        self.last_candidate_target_score = None
        self.last_candidate_pred_score = None
        self.last_candidate_score_gap = None
        self.last_candidate_dynamic_count = None
        self.last_candidate_class_index = None
        self.last_candidate_ids = None
        self.last_candidate_labels = None
        self.last_candidate_is_unseen = None
        self.last_candidate_logits = None
        self.last_topology_conditioning_mode = self.topology_conditioning_mode
        self.last_impulse_activation_mean_abs = None
        self.last_impulse_activation_max_abs = None
        self.last_impulse_to_base_node_ratio = None
        self.last_impulse_gnn_oversmoothing_ratio = None
        self.last_candidate_unseen_score_mean = None
        self.last_candidate_seen_score_mean = None
        self.last_candidate_seen_unseen_score_gap = None
        self.last_structural_edge_count = None

    def _encode_observed_input(self, contract: GraphTensorContract) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

    def _encode_observed_context(self, contract: GraphTensorContract) -> torch.Tensor:
        x, edge_index, batch = self._encode_observed_input(contract)
        x = self.observed_conv1(x, edge_index)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.observed_conv2(x, edge_index)
        x = self.activation(x)
        x = self.dropout(x)
        return self._pool_nodes(x, batch)

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

    def _node_identity_features(
        self,
        node_to_class: torch.Tensor,
        device: torch.device,
        contract: GraphTensorContract | None = None,
    ) -> torch.Tensor:
        node_to_class = node_to_class.to(device=device, dtype=torch.long)
        h_id = self.struct_non_target_emb.to(device=device).unsqueeze(0).expand(node_to_class.numel(), -1).clone()
        valid = (node_to_class >= 0) & (node_to_class < self.output_dim)
        if bool(valid.any()):
            h_id[valid] = self.struct_node_emb(node_to_class[valid])

        if contract is not None:
            struct_node_to_candidate = contract.get("struct_node_to_candidate_index")
            candidate_ids = self._candidate_tuple(contract, "candidate_ids")
            if isinstance(struct_node_to_candidate, torch.Tensor) and len(candidate_ids) > 0:
                unseen_mask = (node_to_class == -1)
                if unseen_mask.any():
                    candidate_indices = struct_node_to_candidate[unseen_mask].tolist()
                    hash_indices = []
                    import hashlib
                    for i, cand_idx in enumerate(candidate_indices):
                        if 0 <= cand_idx < len(candidate_ids):
                            cand_id = str(candidate_ids[cand_idx])
                        else:
                            cand_id = f"fallback_{i}"
                        h_val = int(hashlib.md5(cand_id.encode('utf-8')).hexdigest(), 16) % self.hash_vocab_size
                        hash_indices.append(h_val)
                    hash_tensor = torch.tensor(hash_indices, dtype=torch.long, device=device)
                    h_id[unseen_mask] = h_id[unseen_mask] + self.unseen_grounding_emb(hash_tensor)
        return h_id

    def _struct_base_input(self, contract: GraphTensorContract, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        node_to_class = contract.get("struct_node_to_class_index")
        if not isinstance(node_to_class, torch.Tensor):
            raise ValueError("EOPKGTopologyConditioned requires struct_node_to_class_index.")
        node_to_class = node_to_class.to(device=device, dtype=torch.long)
        identity = self._node_identity_features(node_to_class, device, contract)
        struct_x = contract.get("struct_x")
        if not isinstance(struct_x, torch.Tensor):
            return self.struct_norm(identity), node_to_class

        struct_x = struct_x.to(device=device, dtype=torch.float32)
        if struct_x.size(0) != node_to_class.numel():
            raise ValueError("struct_x and struct_node_to_class_index must describe the same number of nodes.")
        projected = self.struct_input_proj(struct_x)
        return projected + identity, node_to_class

    def _struct_input(self, contract: GraphTensorContract, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        base, node_to_class = self._struct_base_input(contract, device)
        return self.struct_norm(base), node_to_class

    def _encode_candidates(
        self,
        contract: GraphTensorContract,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        structural_edge_index = contract.get("structural_edge_index")
        if not isinstance(structural_edge_index, torch.Tensor):
            raise ValueError("EOPKGTopologyConditioned requires structural_edge_index.")
        if self.topology_conditioning_mode == "impulse_activation_routing":
            return self._encode_impulse_candidates(contract, device)
        h0, node_to_class = self._struct_input(contract, device)
        edge_index = structural_edge_index.to(device=device, dtype=torch.long)
        h = self.struct_conv1(h0, edge_index)
        h = self.activation(h)
        h = self.dropout(h)
        return self.struct_norm(h), node_to_class

    def _impulse_scale(self) -> torch.Tensor:
        return torch.clamp(self.impulse_scale_raw, min=0.0, max=float(self.impulse_scale_max))

    def _encode_impulse_candidates(
        self,
        contract: GraphTensorContract,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        structural_edge_index = contract.get("structural_edge_index")
        if not isinstance(structural_edge_index, torch.Tensor):
            raise ValueError("EOPKGTopologyConditioned requires structural_edge_index.")
        prefix_state = contract.get("struct_prefix_state_x")
        if not isinstance(prefix_state, torch.Tensor):
            raise ValueError("impulse_activation_routing requires struct_prefix_state_x.")

        base_raw, node_to_class = self._struct_base_input(contract, device)
        prefix_state = prefix_state.to(device=device, dtype=torch.float32)
        if prefix_state.dim() == 2:
            prefix_state = prefix_state.unsqueeze(0)
        if prefix_state.dim() != 3:
            raise ValueError("struct_prefix_state_x must have shape [V, F] or [B, V, F].")
        if prefix_state.size(1) != node_to_class.numel():
            raise ValueError("struct_prefix_state_x node dimension must match struct_node_to_class_index.")

        base = self.struct_norm(base_raw).unsqueeze(0).expand(prefix_state.size(0), -1, -1)
        impulse = self.impulse_norm(self.impulse_proj(prefix_state))
        h0 = base + self._impulse_scale().to(device=device, dtype=base.dtype) * impulse

        edge_index = structural_edge_index.to(device=device, dtype=torch.long)
        propagated = []
        for row in range(h0.size(0)):
            h = h0[row]
            for _ in range(self.impulse_gnn_layers):
                h = self.struct_conv1(h, edge_index)
                h = self.activation(h)
                h = self.dropout(h)
            if self.impulse_residual_h0:
                h = h + h0[row]
            propagated.append(h)
        h_nodes = self.struct_norm(torch.stack(propagated, dim=0))

        self._record_impulse_diagnostics(
            base=base,
            impulse=impulse,
            h0=h0,
            h_nodes=h_nodes,
            edge_index=edge_index,
        )
        return h_nodes, node_to_class

    @staticmethod
    def _candidate_tuple(contract: GraphTensorContract, key: str) -> tuple[str, ...]:
        raw = contract.get(key)  # type: ignore[typeddict-item]
        if raw is None:
            return ()
        if isinstance(raw, (list, tuple)) and len(raw) > 0 and isinstance(raw[0], (list, tuple)):
            raw = raw[0]
        if isinstance(raw, str):
            return (raw,)
        if isinstance(raw, (list, tuple)):
            return tuple(str(item) for item in raw)
        return ()

    def _candidate_class_index(self, contract: GraphTensorContract, node_to_class: torch.Tensor) -> torch.LongTensor:
        raw = contract.get("candidate_class_index")
        if isinstance(raw, torch.Tensor):
            return raw.to(device=node_to_class.device, dtype=torch.long).view(-1)
        valid = node_to_class[(node_to_class >= 0) & (node_to_class < self.output_dim)]
        return torch.unique(valid, sorted=True)

    def _candidate_unseen(self, contract: GraphTensorContract, candidate_class_index: torch.Tensor) -> torch.BoolTensor:
        raw = contract.get("candidate_is_unseen")
        if isinstance(raw, torch.Tensor):
            return raw.to(device=candidate_class_index.device, dtype=torch.bool).view(-1)
        return candidate_class_index < 0

    def _fixed_logits_from_candidates(
        self,
        candidate_logits: torch.Tensor,
        candidate_class_index: torch.Tensor,
    ) -> torch.Tensor:
        fixed = candidate_logits.new_full((candidate_logits.size(0), self.output_dim), float("-inf"))
        class_index = candidate_class_index.to(device=candidate_logits.device, dtype=torch.long).view(-1)
        valid = (class_index >= 0) & (class_index < self.output_dim)
        if not bool(valid.any()):
            return torch.nan_to_num(fixed, neginf=-1e6)

        valid_logits = candidate_logits[:, valid]
        valid_classes = class_index[valid]

        import math
        for class_idx in torch.unique(valid_classes).tolist():
            class_int = int(class_idx)
            values = valid_logits[:, valid_classes == class_int]
            if self.candidate_pooling == "logmeanexp":
                fixed[:, class_int] = torch.logsumexp(values, dim=1) - math.log(values.size(1))
            elif self.candidate_pooling == "mean":
                fixed[:, class_int] = values.mean(dim=1)
            elif self.candidate_pooling == "max":
                fixed[:, class_int] = values.max(dim=1).values

        return torch.nan_to_num(fixed, neginf=-1e6)

    def _temperature(self) -> torch.Tensor:
        return torch.clamp(
            self.candidate_temperature,
            min=float(self.candidate_temperature_min),
            max=float(self.candidate_temperature_max),
        )

    def _score_candidates(self, obs_context: torch.Tensor, candidate_embeddings: torch.Tensor) -> torch.Tensor:
        if self.candidate_scoring == "cosine":
            if candidate_embeddings.dim() == 3:
                scores = torch.sum(
                    F.normalize(obs_context, dim=1).unsqueeze(1)
                    * F.normalize(candidate_embeddings, dim=2),
                    dim=2,
                )
                return scores / self._temperature().to(device=obs_context.device, dtype=obs_context.dtype)
            scores = F.normalize(obs_context, dim=1) @ F.normalize(candidate_embeddings, dim=1).T
            return scores / self._temperature().to(device=obs_context.device, dtype=obs_context.dtype)

        transformed = self.bilinear(obs_context)
        if candidate_embeddings.dim() == 3:
            return torch.sum(transformed.unsqueeze(1) * candidate_embeddings, dim=2)
        return transformed @ candidate_embeddings.T

    def _record_impulse_diagnostics(
        self,
        *,
        base: torch.Tensor,
        impulse: torch.Tensor,
        h0: torch.Tensor,
        h_nodes: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> None:
        with torch.no_grad():
            safe_impulse = torch.abs(torch.nan_to_num(impulse.detach().float(), nan=0.0, posinf=1e6, neginf=-1e6))
            safe_base = torch.abs(torch.nan_to_num(base.detach().float(), nan=0.0, posinf=1e6, neginf=-1e6))
            delta = torch.abs(torch.nan_to_num((h_nodes - h0).detach().float(), nan=0.0, posinf=1e6, neginf=-1e6))
            h0_abs = torch.abs(torch.nan_to_num(h0.detach().float(), nan=0.0, posinf=1e6, neginf=-1e6))
            self.last_impulse_activation_mean_abs = float(safe_impulse.mean().item())
            self.last_impulse_activation_max_abs = float(safe_impulse.max().item())
            self.last_impulse_to_base_node_ratio = float(safe_impulse.mean().item() / max(float(safe_base.mean().item()), 1e-12))
            self.last_impulse_gnn_oversmoothing_ratio = float(delta.mean().item() / max(float(h0_abs.mean().item()), 1e-12))
            self.last_structural_edge_count = int(edge_index.size(1))

    def _pool_node_scores_to_classes(self, node_scores: torch.Tensor, node_to_class: torch.Tensor) -> torch.Tensor:
        node_to_class = node_to_class.to(device=node_scores.device, dtype=torch.long)
        valid = (node_to_class >= 0) & (node_to_class < self.output_dim)
        class_scores = node_scores.new_full((node_scores.size(0), self.output_dim), float("-inf"))
        if not bool(valid.any()):
            return class_scores

        valid_scores = node_scores[:, valid]
        valid_classes = node_to_class[valid]
        for class_idx in torch.unique(valid_classes).tolist():
            class_int = int(class_idx)
            values = valid_scores[:, valid_classes == class_int]
            if self.candidate_pooling == "logmeanexp":
                class_scores[:, class_int] = torch.logsumexp(values, dim=1) - math.log(values.size(1))
            elif self.candidate_pooling == "mean":
                class_scores[:, class_int] = values.mean(dim=1)
            elif self.candidate_pooling == "max":
                class_scores[:, class_int] = values.max(dim=1).values
        return class_scores

    def forward_candidate(self, contract: GraphTensorContract) -> CandidatePredictionOutput:
        """Return version/topology-local candidate logits `[B, C_v]`.

        This is the Stage-2 model-level contract. The current trainer still uses
        `forward()` and fixed-label logits, while this method exposes the
        topology-defined candidate axis needed by the next trainer/evaluator
        implementation step.
        """

        self._reset_candidate_diagnostics()
        obs_context = self._encode_observed_context(contract)
        candidates, node_to_class = self._encode_candidates(contract, obs_context.device)
        node_scores = self._score_candidates(obs_context, candidates)
        candidate_logits, candidate_class_index, node_to_candidate_index = self._pool_node_scores_to_dynamic_candidates(
            node_scores,
            node_to_class,
            contract=contract,
        )
        fixed_label_logits = self._fixed_logits_from_candidates(candidate_logits, candidate_class_index)
        self._record_candidate_diagnostics(
            node_scores=node_scores,
            class_scores=fixed_label_logits,
            node_to_class=node_to_class,
            targets=contract.get("y"),
        )
        self.last_candidate_dynamic_count = int(candidate_class_index.numel())
        self.last_candidate_class_index = candidate_class_index.detach().cpu().tolist()
        candidate_ids = self._candidate_tuple(contract, "candidate_ids")
        candidate_labels = self._candidate_tuple(contract, "candidate_labels")
        candidate_is_unseen = self._candidate_unseen(contract, candidate_class_index)
        self.last_candidate_ids = list(candidate_ids)
        self.last_candidate_labels = list(candidate_labels)
        self.last_candidate_is_unseen = candidate_is_unseen.detach().cpu().tolist()
        self.last_candidate_logits = candidate_logits.detach().cpu()
        self._record_seen_unseen_candidate_diagnostics(candidate_logits, candidate_is_unseen)
        return CandidatePredictionOutput(
            candidate_logits=candidate_logits,
            candidate_class_index=candidate_class_index,
            node_logits=node_scores,
            node_to_candidate_index=node_to_candidate_index,
            node_to_class_index=node_to_class,
            fixed_label_logits=fixed_label_logits,
            candidate_ids=candidate_ids,
            candidate_labels=candidate_labels,
            candidate_is_unseen=candidate_is_unseen,
        )

    def _record_seen_unseen_candidate_diagnostics(
        self,
        candidate_logits: torch.Tensor,
        candidate_is_unseen: torch.Tensor | None,
    ) -> None:
        if not isinstance(candidate_is_unseen, torch.Tensor) or candidate_is_unseen.numel() <= 0:
            return
        with torch.no_grad():
            unseen = candidate_is_unseen.to(device=candidate_logits.device, dtype=torch.bool).view(-1)
            finite_logits = torch.nan_to_num(candidate_logits.detach().float(), nan=0.0, posinf=1e6, neginf=-1e6)
            if bool(unseen.any()):
                self.last_candidate_unseen_score_mean = float(finite_logits[:, unseen].mean().item())
            if bool((~unseen).any()):
                self.last_candidate_seen_score_mean = float(finite_logits[:, ~unseen].mean().item())
            if self.last_candidate_seen_score_mean is not None and self.last_candidate_unseen_score_mean is not None:
                self.last_candidate_seen_unseen_score_gap = float(
                    self.last_candidate_seen_score_mean - self.last_candidate_unseen_score_mean
                )

    def _pool_node_scores_to_dynamic_candidates(
        self,
        node_scores: torch.Tensor,
        node_to_class: torch.Tensor,
        *,
        contract: GraphTensorContract | None = None,
    ) -> tuple[torch.Tensor, torch.LongTensor, torch.LongTensor]:
        node_to_class = node_to_class.to(device=node_scores.device, dtype=torch.long)
        raw_node_to_candidate = contract.get("struct_node_to_candidate_index") if contract is not None else None
        if isinstance(raw_node_to_candidate, torch.Tensor):
            node_to_candidate_index = raw_node_to_candidate.to(device=node_scores.device, dtype=torch.long).view(-1)
            candidate_class_index = self._candidate_class_index(contract, node_to_class)  # type: ignore[arg-type]
            valid = (node_to_candidate_index >= 0) & (node_to_candidate_index < int(candidate_class_index.numel()))
        else:
            valid = (node_to_class >= 0) & (node_to_class < self.output_dim)
            candidate_class_index = torch.unique(node_to_class[valid], sorted=True)
            node_to_candidate_index = torch.full_like(node_to_class, -1, dtype=torch.long)
            for local_idx, class_idx in enumerate(candidate_class_index.tolist()):
                node_to_candidate_index[(node_to_class == int(class_idx)) & valid] = int(local_idx)
        candidate_logits = node_scores.new_empty((node_scores.size(0), int(candidate_class_index.numel())))
        if candidate_class_index.numel() == 0:
            return candidate_logits, candidate_class_index, node_to_candidate_index

        for local_idx in range(int(candidate_class_index.numel())):
            class_mask = valid & (node_to_candidate_index == int(local_idx))
            if not bool(class_mask.any()):
                candidate_logits[:, local_idx] = float("-inf")
                continue
            values = node_scores[:, class_mask]
            if self.candidate_pooling == "logmeanexp":
                candidate_logits[:, local_idx] = torch.logsumexp(values, dim=1) - math.log(values.size(1))
            elif self.candidate_pooling == "mean":
                candidate_logits[:, local_idx] = values.mean(dim=1)
            elif self.candidate_pooling == "max":
                candidate_logits[:, local_idx] = values.max(dim=1).values

        return candidate_logits, candidate_class_index, node_to_candidate_index

    def _record_candidate_diagnostics(
        self,
        *,
        node_scores: torch.Tensor,
        class_scores: torch.Tensor,
        node_to_class: torch.Tensor,
        targets: torch.Tensor | None,
    ) -> None:
        with torch.no_grad():
            finite = torch.isfinite(class_scores)
            probs = torch.softmax(class_scores, dim=1)
            pred = torch.argmax(probs, dim=1)
            entropy = -torch.sum(probs * torch.log(torch.clamp(probs, min=1e-12)), dim=1)

            valid = node_to_class[(node_to_class >= 0) & (node_to_class < self.output_dim)]
            max_duplicate = 0
            if valid.numel() > 0:
                counts = torch.bincount(valid.detach().cpu(), minlength=self.output_dim)
                max_duplicate = int(counts.max().item())

            self.last_candidate_node_score_mean_abs = float(torch.abs(node_scores.detach()).mean().item())
            self.last_candidate_class_score_mean_abs = (
                float(torch.abs(class_scores.detach()[finite]).mean().item()) if bool(finite.any()) else 0.0
            )
            self.last_duplicate_candidate_count_max = max_duplicate
            self.last_candidate_temperature = float(self._temperature().detach().cpu().item())
            self.last_candidate_temperature_trainable = self.candidate_temperature_trainable
            self.last_candidate_prediction_entropy = float(entropy.mean().item())
            dynamic_classes = torch.unique(valid, sorted=True)
            self.last_candidate_dynamic_count = int(dynamic_classes.numel())
            self.last_candidate_class_index = dynamic_classes.detach().cpu().tolist()
            if isinstance(targets, torch.Tensor) and targets.numel() > 0:
                flat_targets = targets.detach().to(device=class_scores.device, dtype=torch.long).view(-1)
                rows = torch.arange(min(class_scores.size(0), flat_targets.numel()), device=class_scores.device)
                target_values = class_scores[rows, flat_targets[: rows.numel()].clamp(0, self.output_dim - 1)]
                pred_values = class_scores[rows, pred[: rows.numel()]]
                self.last_candidate_target_score = float(target_values.mean().item())
                self.last_candidate_pred_score = float(pred_values.mean().item())
                self.last_candidate_score_gap = float((pred_values - target_values).mean().item())

    def load_state_dict(self, state_dict: dict[str, Any], strict: bool = True):
        """Restore model state with backward compatibility for grounding embeddings."""
        key = "unseen_grounding_emb.weight"
        if key not in state_dict and hasattr(self, "unseen_grounding_emb"):
            state_dict[key] = self.unseen_grounding_emb.weight.data
        return super().load_state_dict(state_dict, strict=strict)
