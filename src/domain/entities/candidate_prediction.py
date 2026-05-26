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

    def map_targets_to_candidate_mask(self, targets: torch.Tensor) -> torch.BoolTensor:
        """Map global class targets to a boolean mask over local candidates."""

        flat_targets = targets.detach().to(device=self.candidate_class_index.device, dtype=torch.long).view(-1)
        class_index = self.candidate_class_index.to(device=flat_targets.device, dtype=torch.long).view(1, -1)
        return class_index.eq(flat_targets.view(-1, 1))

    def map_targets_to_candidate_index(self, targets: torch.Tensor, missing_value: int = -1) -> torch.LongTensor:
        """Map global class targets to local dynamic candidate indexes."""

        mask = self.map_targets_to_candidate_mask(targets)
        mapped = torch.full((mask.size(0),), int(missing_value), dtype=torch.long, device=mask.device)
        has_one = mask.sum(dim=1) == 1
        if bool(has_one.any()):
            mapped[has_one] = torch.argmax(mask[has_one].long(), dim=1)
        return mapped
