"""Fixed-vocabulary recurrent baseline for next-activity prediction."""

from __future__ import annotations

from typing import Dict, Sequence

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

from src.domain.entities.feature_config import FeatureLayout
from src.domain.entities.tensor_contract import GraphTensorContract
from src.domain.models.base_gnn import BaseGNN


class LSTMBaseline(BaseGNN):
    """LSTM/GRU logs-only baseline with a fixed activity-vocabulary head."""

    def __init__(
        self,
        feature_layout: FeatureLayout,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.2,
        pooling_strategy: str = "last_node",
        recurrent_type: str = "lstm",
        recurrent_layers: int = 1,
        recurrent_bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.feature_layout = feature_layout
        self.pooling_strategy = pooling_strategy
        self.recurrent_type = recurrent_type.lower().strip()
        self.recurrent_layers = int(recurrent_layers)
        self.recurrent_bidirectional = bool(recurrent_bidirectional)

        if self.recurrent_type not in {"lstm", "gru"}:
            raise ValueError(f"Unsupported model.recurrent_type '{recurrent_type}'.")
        if self.recurrent_layers < 1:
            raise ValueError("model.recurrent_layers must be >= 1.")

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
        rnn_cls = nn.LSTM if self.recurrent_type == "lstm" else nn.GRU
        recurrent_dropout = dropout if self.recurrent_layers > 1 else 0.0
        self.recurrent = rnn_cls(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=self.recurrent_layers,
            batch_first=True,
            dropout=recurrent_dropout,
            bidirectional=self.recurrent_bidirectional,
        )
        recurrent_output_dim = hidden_dim * (2 if self.recurrent_bidirectional else 1)
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(recurrent_output_dim, output_dim)

    def forward(self, contract: GraphTensorContract) -> torch.Tensor:
        """Compute fixed-vocabulary logits [batch_size, num_classes]."""
        x_cat = contract["x_cat"]
        x_num = contract["x_num"]
        batch = contract["batch"]

        emb_parts = []
        for col, name in enumerate(self.feature_layout.cat_feature_names):
            emb_parts.append(self.embeddings[name](x_cat[:, col]))
        x = torch.cat([*emb_parts, x_num], dim=1) if emb_parts else x_num

        if batch.numel() != x.size(0):
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        sequences = self._split_by_batch(x=x, batch=batch)
        if not sequences:
            return x.new_zeros((0, self.classifier.out_features))

        lengths = torch.tensor([seq.size(0) for seq in sequences], dtype=torch.long, device=x.device)
        padded = pad_sequence(sequences, batch_first=True)
        packed = pack_padded_sequence(
            padded,
            lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, _ = self.recurrent(packed)
        outputs, _ = pad_packed_sequence(packed_output, batch_first=True)
        pooled = self._pool_sequence_outputs(outputs=outputs, lengths=lengths)
        return self.classifier(self.dropout(pooled))

    @staticmethod
    def _split_by_batch(x: torch.Tensor, batch: torch.Tensor) -> list[torch.Tensor]:
        if x.size(0) == 0:
            return []
        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
        return [x[batch == graph_idx] for graph_idx in range(num_graphs) if bool((batch == graph_idx).any())]

    def _pool_sequence_outputs(self, outputs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        if self.pooling_strategy == "last_node":
            row_idx = torch.arange(outputs.size(0), device=outputs.device)
            last_idx = lengths.to(outputs.device) - 1
            return outputs[row_idx, last_idx]

        if self.pooling_strategy == "global_mean":
            mask = (
                torch.arange(outputs.size(1), device=outputs.device).unsqueeze(0)
                < lengths.to(outputs.device).unsqueeze(1)
            )
            masked = outputs * mask.unsqueeze(-1).to(outputs.dtype)
            return masked.sum(dim=1) / lengths.to(outputs.device).clamp_min(1).unsqueeze(1).to(outputs.dtype)

        raise ValueError(f"Unsupported pooling_strategy '{self.pooling_strategy}'.")
