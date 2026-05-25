"""Topology-conditioned candidate scoring model family."""

from __future__ import annotations

import math
from typing import Dict

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
        candidate_temperature_max: float = 10.0,
        candidate_temperature_trainable: bool = False,
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
        self.candidate_temperature_min = float(candidate_temperature_min)
        self.candidate_temperature_max = float(candidate_temperature_max)
        self.candidate_temperature_trainable = bool(candidate_temperature_trainable)

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

    def _node_identity_features(self, node_to_class: torch.Tensor, device: torch.device) -> torch.Tensor:
        node_to_class = node_to_class.to(device=device, dtype=torch.long)
        h_id = self.struct_non_target_emb.to(device=device).unsqueeze(0).expand(node_to_class.numel(), -1).clone()
        valid = (node_to_class >= 0) & (node_to_class < self.output_dim)
        if bool(valid.any()):
            h_id[valid] = self.struct_node_emb(node_to_class[valid])
        return h_id

    def _struct_input(self, contract: GraphTensorContract, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        node_to_class = contract.get("struct_node_to_class_index")
        if not isinstance(node_to_class, torch.Tensor):
            raise ValueError("EOPKGTopologyConditioned requires struct_node_to_class_index.")
        node_to_class = node_to_class.to(device=device, dtype=torch.long)
        identity = self._node_identity_features(node_to_class, device)
        struct_x = contract.get("struct_x")
        if not isinstance(struct_x, torch.Tensor):
            return self.struct_norm(identity), node_to_class

        struct_x = struct_x.to(device=device, dtype=torch.float32)
        if struct_x.size(0) != node_to_class.numel():
            raise ValueError("struct_x and struct_node_to_class_index must describe the same number of nodes.")
        projected = self.struct_input_proj(struct_x)
        return self.struct_norm(projected + identity), node_to_class

    def _encode_candidates(
        self,
        contract: GraphTensorContract,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        structural_edge_index = contract.get("structural_edge_index")
        if not isinstance(structural_edge_index, torch.Tensor):
            raise ValueError("EOPKGTopologyConditioned requires structural_edge_index.")
        h0, node_to_class = self._struct_input(contract, device)
        edge_index = structural_edge_index.to(device=device, dtype=torch.long)
        h = self.struct_conv1(h0, edge_index)
        h = self.activation(h)
        h = self.dropout(h)
        return self.struct_norm(h), node_to_class

    def _temperature(self) -> torch.Tensor:
        return torch.clamp(
            self.candidate_temperature,
            min=float(self.candidate_temperature_min),
            max=float(self.candidate_temperature_max),
        )

    def _score_candidates(self, obs_context: torch.Tensor, candidate_embeddings: torch.Tensor) -> torch.Tensor:
        if self.candidate_scoring == "cosine":
            scores = F.normalize(obs_context, dim=1) @ F.normalize(candidate_embeddings, dim=1).T
            return scores / self._temperature().to(device=obs_context.device, dtype=obs_context.dtype)

        transformed = self.bilinear(obs_context)
        return transformed @ candidate_embeddings.T

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
        )
        fixed_label_logits = self._pool_node_scores_to_classes(node_scores, node_to_class)
        fixed_label_logits = torch.nan_to_num(fixed_label_logits, neginf=-1e6)
        self._record_candidate_diagnostics(
            node_scores=node_scores,
            class_scores=fixed_label_logits,
            node_to_class=node_to_class,
            targets=contract.get("y"),
        )
        self.last_candidate_dynamic_count = int(candidate_class_index.numel())
        self.last_candidate_class_index = candidate_class_index.detach().cpu().tolist()
        return CandidatePredictionOutput(
            candidate_logits=candidate_logits,
            candidate_class_index=candidate_class_index,
            node_logits=node_scores,
            node_to_candidate_index=node_to_candidate_index,
            node_to_class_index=node_to_class,
            fixed_label_logits=fixed_label_logits,
        )

    def _pool_node_scores_to_dynamic_candidates(
        self,
        node_scores: torch.Tensor,
        node_to_class: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.LongTensor, torch.LongTensor]:
        node_to_class = node_to_class.to(device=node_scores.device, dtype=torch.long)
        valid = (node_to_class >= 0) & (node_to_class < self.output_dim)
        candidate_class_index = torch.unique(node_to_class[valid], sorted=True)
        node_to_candidate_index = torch.full_like(node_to_class, -1, dtype=torch.long)
        candidate_logits = node_scores.new_empty((node_scores.size(0), int(candidate_class_index.numel())))
        if candidate_class_index.numel() == 0:
            return candidate_logits, candidate_class_index, node_to_candidate_index

        valid_scores = node_scores[:, valid]
        valid_classes = node_to_class[valid]
        for local_idx, class_idx in enumerate(candidate_class_index.tolist()):
            class_int = int(class_idx)
            class_mask = valid_classes == class_int
            node_to_candidate_index[(node_to_class == class_int) & valid] = int(local_idx)
            values = valid_scores[:, class_mask]
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
