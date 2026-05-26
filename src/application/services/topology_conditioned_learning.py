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


def candidate_allowed_mask_from_global(
    allowed_target_mask: torch.Tensor,
    candidate_class_index: torch.Tensor,
) -> torch.Tensor:
    """Project global allowed class mask `[B, C_train]` to candidate axis `[B, C_v]`."""

    if not isinstance(allowed_target_mask, torch.Tensor):
        raise ValueError("allowed_target_mask must be a tensor.")
    if allowed_target_mask.dim() != 2:
        raise ValueError("allowed_target_mask must have shape [B, C_train].")
    if not isinstance(candidate_class_index, torch.Tensor):
        raise ValueError("candidate_class_index must be a tensor.")

    class_index = candidate_class_index.to(device=allowed_target_mask.device, dtype=torch.long).view(-1)
    if class_index.numel() <= 0:
        raise ValueError("candidate_class_index must not be empty.")

    valid = class_index >= 0
    projected = torch.zeros(
        (allowed_target_mask.size(0), class_index.numel()),
        device=allowed_target_mask.device,
        dtype=torch.bool,
    )
    if not bool(valid.any()):
        return projected

    max_index = int(class_index[valid].max().detach().cpu().item())
    if max_index >= int(allowed_target_mask.size(1)):
        raise ValueError(
            "candidate_class_index contains values outside allowed_target_mask class axis: "
            f"max={max_index} classes={int(allowed_target_mask.size(1))}."
        )
    projected[:, valid] = allowed_target_mask.to(dtype=torch.bool)[:, class_index[valid]]
    return projected


def candidate_allowed_flow_summary(
    candidate_logits: torch.Tensor,
    candidate_allowed_mask: torch.Tensor,
) -> dict[str, float]:
    """Summarize candidate-space probability/logit mass outside current topology flow."""

    if candidate_logits.dim() != 2:
        raise ValueError("candidate_logits must have shape [B, C_v].")
    mask = candidate_allowed_mask.to(device=candidate_logits.device, dtype=torch.bool)
    if mask.shape != candidate_logits.shape:
        raise ValueError("candidate_allowed_mask must have the same shape as candidate_logits.")

    probs = torch.softmax(candidate_logits, dim=1)
    invalid = ~mask
    invalid_mass = probs.masked_fill(~invalid, 0.0).sum(dim=1)
    valid_mass = probs.masked_fill(~mask, 0.0).sum(dim=1)

    pred = torch.argmax(candidate_logits, dim=1)
    pred_allowed = mask.gather(1, pred.view(-1, 1)).view(-1)

    valid_logits = candidate_logits.masked_fill(~mask, float("-inf"))
    invalid_logits = candidate_logits.masked_fill(~invalid, float("-inf"))
    zeros = candidate_logits.new_zeros(candidate_logits.size(0))
    valid_max = torch.where(mask.any(dim=1), valid_logits.max(dim=1).values, zeros)
    invalid_max = torch.where(invalid.any(dim=1), invalid_logits.max(dim=1).values, zeros)
    margin = valid_max - invalid_max

    return {
        "candidate_oos_rate": float((~pred_allowed).float().mean().detach().cpu().item())
        if pred_allowed.numel()
        else 0.0,
        "candidate_invalid_probability_mass": float(invalid_mass.mean().detach().cpu().item())
        if invalid_mass.numel()
        else 0.0,
        "candidate_valid_probability_mass": float(valid_mass.mean().detach().cpu().item())
        if valid_mass.numel()
        else 0.0,
        "candidate_valid_invalid_logit_margin": float(margin.mean().detach().cpu().item())
        if margin.numel()
        else 0.0,
    }


def candidate_topology_flow_penalty_loss(
    candidate_logits: torch.Tensor,
    candidate_allowed_mask: torch.Tensor,
    *,
    penalty_type: str = "invalid_probability_mass",
    margin: float = 0.1,
) -> torch.Tensor:
    """Train-time penalty for assigning score/probability to invalid topology candidates."""

    if candidate_logits.dim() != 2:
        raise ValueError("candidate_logits must have shape [B, C_v].")
    mask = candidate_allowed_mask.to(device=candidate_logits.device, dtype=torch.bool)
    if mask.shape != candidate_logits.shape:
        raise ValueError("candidate_allowed_mask must have the same shape as candidate_logits.")

    penalty_type = str(penalty_type).strip().lower()
    invalid = ~mask
    if penalty_type == "invalid_probability_mass":
        probs = torch.softmax(candidate_logits, dim=1)
        return probs.masked_fill(~invalid, 0.0).sum(dim=1).mean()

    if penalty_type == "margin":
        valid_logits = candidate_logits.masked_fill(~mask, float("-inf"))
        invalid_logits = candidate_logits.masked_fill(~invalid, float("-inf"))
        zeros = candidate_logits.new_zeros(candidate_logits.size(0))
        valid_max = torch.where(mask.any(dim=1), valid_logits.max(dim=1).values, zeros)
        invalid_max = torch.where(invalid.any(dim=1), invalid_logits.max(dim=1).values, zeros)
        return torch.relu(invalid_max - valid_max + float(margin)).mean()

    raise ValueError("penalty_type must be invalid_probability_mass or margin.")


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
