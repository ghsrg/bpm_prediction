import torch
import torch.nn.functional as F
import pytest

from src.application.services.topology_conditioned_learning import (
    allowed_set_loss,
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
