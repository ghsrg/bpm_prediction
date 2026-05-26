from __future__ import annotations

import pytest
import torch

from src.application.services.candidate_target_mapping import (
    candidate_level_accuracy,
    candidate_set_cross_entropy,
    candidate_target_summary,
)
from src.domain.entities.candidate_prediction import CandidatePredictionOutput


def _output() -> CandidatePredictionOutput:
    return CandidatePredictionOutput(
        candidate_logits=torch.tensor([[0.2, 1.0, 0.7], [1.1, 0.1, 0.2]], dtype=torch.float32),
        candidate_class_index=torch.tensor([4, 5, 5], dtype=torch.long),
        node_logits=torch.tensor([[0.2, 1.0, 0.7], [1.1, 0.1, 0.2]], dtype=torch.float32),
        node_to_candidate_index=torch.tensor([0, 1, 2], dtype=torch.long),
        node_to_class_index=torch.tensor([4, 5, 5], dtype=torch.long),
    )


def test_candidate_output_maps_class_targets_to_candidate_mask_with_duplicates():
    mask = _output().map_targets_to_candidate_mask(torch.tensor([5, 4], dtype=torch.long))

    assert mask.tolist() == [[False, True, True], [True, False, False]]


def test_candidate_set_cross_entropy_accepts_duplicate_target_candidates():
    output = _output()
    loss = candidate_set_cross_entropy(output.candidate_logits, output.map_targets_to_candidate_mask(torch.tensor([5, 4])))

    assert loss.item() >= 0.0
    assert torch.isfinite(loss)


def test_candidate_target_summary_reports_missing_and_duplicate_spread():
    output = _output()
    targets = torch.tensor([5, 9], dtype=torch.long)
    mask = output.map_targets_to_candidate_mask(targets)

    summary = candidate_target_summary(output.candidate_logits, mask)

    assert summary["target_in_candidate_set_rate"] == pytest.approx(0.5)
    assert summary["missing_target_rate"] == pytest.approx(0.5)
    assert summary["target_duplicate_count_max"] == 2
    assert summary["target_set_logit_variance_mean"] >= 0.0


def test_candidate_level_accuracy_maps_predictions_back_to_global_classes():
    output = _output()

    assert candidate_level_accuracy(output.candidate_logits, output.candidate_class_index, torch.tensor([5, 4])) == pytest.approx(1.0)
