import torch
import torch.nn.functional as F
import pytest

from src.application.services.topology_conditioned_learning import (
    allowed_set_mass_leakage,
    allowed_set_loss,
    candidate_allowed_flow_summary,
    candidate_allowed_mask_from_global,
    candidate_topology_flow_penalty_loss,
    margin_negative_loss,
    version_weighted_cross_entropy,
)


def test_allowed_set_loss_rewards_probability_mass_inside_allowed_set():
    logits = torch.tensor([[4.0, 3.0, -2.0]], dtype=torch.float32)
    targets = torch.tensor([1], dtype=torch.long)
    allowed = torch.tensor([[True, True, False]])

    loss = allowed_set_loss(logits, targets, allowed)
    exact_loss = F.cross_entropy(logits, targets)

    assert loss < exact_loss


def test_allowed_set_loss_forces_target_allowed_when_mask_misses_target():
    logits = torch.tensor([[4.0, 3.0, -2.0]], dtype=torch.float32)
    targets = torch.tensor([1], dtype=torch.long)
    allowed = torch.tensor([[True, False, False]])

    loss = allowed_set_loss(logits, targets, allowed)

    assert torch.isfinite(loss)
    assert loss < F.cross_entropy(logits, targets)


def test_allowed_set_mass_leakage_measures_probability_outside_allowed_set():
    logits = torch.tensor([[4.0, 3.0, -2.0]], dtype=torch.float32)
    targets = torch.tensor([1], dtype=torch.long)
    allowed = torch.tensor([[True, True, False]])

    leakage = allowed_set_mass_leakage(logits, targets, allowed)
    probs = torch.softmax(logits, dim=1)

    assert leakage == torch.sum(probs[:, 2])


def test_margin_negative_loss_penalizes_when_wrong_ce_is_not_worse():
    correct_ce = torch.tensor(0.50)
    wrong_ce = torch.tensor(0.55)

    loss = margin_negative_loss(correct_ce=correct_ce, negative_ce=wrong_ce, margin=0.20)

    assert loss.item() == pytest.approx(0.15)


def test_version_weighted_cross_entropy_applies_per_sample_weights():
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]], dtype=torch.float32)
    targets = torch.tensor([0, 1], dtype=torch.long)
    weights = torch.tensor([1.0, 0.25], dtype=torch.float32)

    loss = version_weighted_cross_entropy(logits, targets, sample_weights=weights)
    per_sample = F.cross_entropy(logits, targets, reduction="none")

    assert loss == torch.sum(per_sample * weights) / torch.sum(weights)


def test_candidate_allowed_mask_projects_global_mask_to_candidate_axis():
    allowed = torch.tensor([[True, False, True, False]])
    candidate_class_index = torch.tensor([2, 0, 3])

    mask = candidate_allowed_mask_from_global(allowed, candidate_class_index)

    assert mask.tolist() == [[True, True, False]]


def test_candidate_allowed_mask_rejects_out_of_range_class_index():
    allowed = torch.ones((1, 2), dtype=torch.bool)
    candidate_class_index = torch.tensor([0, 3])

    with pytest.raises(ValueError, match="candidate_class_index"):
        candidate_allowed_mask_from_global(allowed, candidate_class_index)


def test_candidate_allowed_flow_summary_reports_invalid_mass_and_oos():
    logits = torch.tensor([[3.0, 1.0, 4.0]], dtype=torch.float32)
    candidate_allowed_mask = torch.tensor([[True, False, False]])

    summary = candidate_allowed_flow_summary(logits, candidate_allowed_mask)

    assert summary["candidate_oos_rate"] == 1.0
    assert summary["candidate_invalid_probability_mass"] > 0.5
    assert summary["candidate_valid_invalid_logit_margin"] < 0.0


def test_candidate_topology_flow_penalty_invalid_probability_mass_is_bounded_and_differentiable():
    logits = torch.tensor([[4.0, 0.0, 3.0]], dtype=torch.float32, requires_grad=True)
    candidate_allowed_mask = torch.tensor([[True, False, False]])

    loss = candidate_topology_flow_penalty_loss(
        logits,
        candidate_allowed_mask,
        penalty_type="invalid_probability_mass",
        margin=0.1,
    )

    assert 0.0 <= float(loss.detach().item()) <= 1.0
    loss.backward()
    assert logits.grad is not None


def test_candidate_topology_flow_penalty_margin_penalizes_invalid_max():
    logits = torch.tensor([[1.0, 3.0]], dtype=torch.float32, requires_grad=True)
    candidate_allowed_mask = torch.tensor([[True, False]])

    loss = candidate_topology_flow_penalty_loss(
        logits,
        candidate_allowed_mask,
        penalty_type="margin",
        margin=0.5,
    )

    assert float(loss.detach().item()) > 0.0
