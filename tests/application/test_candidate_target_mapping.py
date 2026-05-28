from __future__ import annotations

import pytest
import torch

from src.application.services.candidate_target_mapping import (
    candidate_level_accuracy,
    candidate_set_cross_entropy_from_mask,
    candidate_set_cross_entropy,
    candidate_target_mask_from_labels,
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


def test_candidate_target_mask_from_labels_matches_unseen_candidate_label():
    mask = candidate_target_mask_from_labels(
        target_labels=["C", "B"],
        candidate_labels=("A", "C", "B"),
        device=torch.device("cpu"),
    )

    assert mask.tolist() == [[False, True, False], [False, False, True]]


def test_candidate_set_cross_entropy_from_mask_trains_unseen_candidate_label():
    logits = torch.tensor([[0.0, 2.0, 0.0]], requires_grad=True)
    target_mask = torch.tensor([[False, True, False]])

    loss = candidate_set_cross_entropy_from_mask(logits, target_mask)
    loss.backward()

    assert logits.grad is not None


def test_candidate_set_cross_entropy_from_mask_supports_sample_weights():
    logits = torch.tensor([[3.0, 0.0], [0.0, 3.0]], dtype=torch.float32)
    target_mask = torch.tensor([[True, False], [True, False]], dtype=torch.bool)
    weights = torch.tensor([10.0, 1.0], dtype=torch.float32)

    loss = candidate_set_cross_entropy_from_mask(logits, target_mask, sample_weights=weights)

    assert loss.item() < 0.4


def test_candidate_output_maps_target_labels_to_candidate_mask():
    output = CandidatePredictionOutput(
        candidate_logits=torch.zeros(1, 3),
        candidate_class_index=torch.tensor([1, -1, 2], dtype=torch.long),
        node_logits=torch.zeros(1, 3),
        node_to_candidate_index=torch.tensor([0, 1, 2], dtype=torch.long),
        node_to_class_index=torch.tensor([1, -1, 2], dtype=torch.long),
        candidate_ids=("node_a", "node_c", "node_b"),
        candidate_labels=("A", "C", "B"),
    )

    assert output.map_target_labels_to_candidate_mask(["C"]).tolist() == [[False, True, False]]
    assert output.map_target_labels_to_candidate_mask(["node_c"]).tolist() == [[False, True, False]]


def test_candidate_target_mask_from_labels_matches_id_or_label():
    mask = candidate_target_mask_from_labels(
        target_labels=["t_approve_loan", "Approve Loan"],
        candidate_labels=("Add Notes", "Approve Loan", "Assess Eligibility"),
        candidate_ids=("t_add_notes", "t_approve_loan", "t_assess_eligibility"),
        device=torch.device("cpu"),
    )
    assert mask.tolist() == [[False, True, False], [False, True, False]]
