"""Candidate-scoring output contracts."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CandidatePredictionOutput:
    """Dynamic topology candidate prediction output.

    `candidate_logits` is indexed by the local candidate axis `[0, C_v)`.
    `candidate_class_index` maps that local axis back to global activity-class
    indexes used by the current fixed-label vocabulary.
    """

    candidate_logits: torch.Tensor
    candidate_class_index: torch.LongTensor
    node_logits: torch.Tensor
    node_to_candidate_index: torch.LongTensor
    node_to_class_index: torch.LongTensor
    fixed_label_logits: torch.Tensor | None = None

    @property
    def candidate_count(self) -> int:
        return int(self.candidate_class_index.numel())

    def map_targets_to_candidate_index(self, targets: torch.Tensor, missing_value: int = -1) -> torch.LongTensor:
        """Map global class targets to local dynamic candidate indexes."""

        flat_targets = targets.detach().to(device=self.candidate_class_index.device, dtype=torch.long).view(-1)
        mapped = torch.full_like(flat_targets, int(missing_value), dtype=torch.long)
        for local_idx, global_idx in enumerate(self.candidate_class_index.tolist()):
            mapped[flat_targets == int(global_idx)] = int(local_idx)
        return mapped
