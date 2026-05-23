"""Pure tensor helpers for topology-conditioned training objectives."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def allowed_set_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    allowed_target_mask: torch.Tensor,
) -> torch.Tensor:
    """Negative log probability assigned to the allowed target set."""
    if not isinstance(allowed_target_mask, torch.Tensor):
        raise ValueError("allowed_set_loss requires allowed_target_mask.")
    mask = allowed_target_mask.to(device=logits.device, dtype=torch.bool).clone()
    target_index = targets.to(device=logits.device, dtype=torch.long).view(-1, 1)
    if mask.dim() != 2 or mask.size(0) != logits.size(0) or mask.size(1) != logits.size(1):
        raise ValueError(
            "allowed_target_mask shape must match logits: "
            f"mask={tuple(mask.shape)} logits={tuple(logits.shape)}."
        )
    mask.scatter_(1, target_index, True)
    log_probs = F.log_softmax(logits, dim=1)
    masked_log_probs = log_probs.masked_fill(~mask, torch.finfo(log_probs.dtype).min)
    return -torch.logsumexp(masked_log_probs, dim=1).mean()


def allowed_set_mass_leakage(
    logits: torch.Tensor,
    targets: torch.Tensor,
    allowed_target_mask: torch.Tensor,
) -> torch.Tensor:
    """Mean softmax probability mass assigned outside the allowed target set."""
    if not isinstance(allowed_target_mask, torch.Tensor):
        raise ValueError("allowed_set_mass_leakage requires allowed_target_mask.")
    mask = allowed_target_mask.to(device=logits.device, dtype=torch.bool).clone()
    target_index = targets.to(device=logits.device, dtype=torch.long).view(-1, 1)
    if mask.dim() != 2 or mask.size(0) != logits.size(0) or mask.size(1) != logits.size(1):
        raise ValueError(
            "allowed_target_mask shape must match logits: "
            f"mask={tuple(mask.shape)} logits={tuple(logits.shape)}."
        )
    mask.scatter_(1, target_index, True)
    probs = torch.softmax(logits, dim=1)
    return (probs * (~mask).to(dtype=probs.dtype)).sum(dim=1).mean()


def version_weighted_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    sample_weights: torch.Tensor,
    class_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Cross entropy with per-sample version/rehearsal weights."""
    weights = sample_weights.to(device=logits.device, dtype=logits.dtype).view(-1)
    if weights.numel() != targets.numel():
        raise ValueError("sample_weights length must match targets length.")
    ce = F.cross_entropy(
        logits,
        targets.to(device=logits.device, dtype=torch.long),
        weight=class_weights.to(device=logits.device, dtype=logits.dtype) if isinstance(class_weights, torch.Tensor) else None,
        reduction="none",
    )
    denom = weights.sum().clamp_min(torch.finfo(weights.dtype).eps)
    return torch.sum(ce * weights) / denom


def margin_negative_loss(
    *,
    correct_ce: torch.Tensor,
    negative_ce: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    """Margin loss that requires negative topology CE to be worse than correct CE."""
    return torch.relu(correct_ce - negative_ce + float(margin))
