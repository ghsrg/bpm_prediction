"""Helpers for topology-local candidate targets."""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn.functional as F


def candidate_set_cross_entropy(
    candidate_logits: torch.Tensor,
    target_candidate_mask: torch.Tensor,
    *,
    missing_target_penalty: float = 0.0,
) -> torch.Tensor:
    """Set-aware CE where any candidate matching the target class is correct."""

    if candidate_logits.dim() != 2:
        raise ValueError("candidate_logits must have shape [B, C_v].")
    mask = target_candidate_mask.to(device=candidate_logits.device, dtype=torch.bool)
    if mask.shape != candidate_logits.shape:
        raise ValueError("target_candidate_mask must have the same shape as candidate_logits.")
    log_probs = F.log_softmax(candidate_logits, dim=1)
    has_target = mask.any(dim=1)
    if not bool(has_target.any()):
        return candidate_logits.sum() * 0.0 + float(missing_target_penalty)
    masked_log_probs = log_probs.masked_fill(~mask, float("-inf"))
    per_row_loss = -torch.logsumexp(masked_log_probs[has_target], dim=1)
    loss = per_row_loss.mean()
    missing_rate = 1.0 - float(has_target.float().mean().detach().cpu().item())
    if missing_rate > 0.0 and float(missing_target_penalty) > 0.0:
        loss = loss + candidate_logits.new_tensor(float(missing_target_penalty) * missing_rate)
    return loss


def candidate_target_summary(candidate_logits: torch.Tensor, target_candidate_mask: torch.Tensor) -> Dict[str, Any]:
    mask = target_candidate_mask.to(device=candidate_logits.device, dtype=torch.bool)
    has_target = mask.any(dim=1)
    target_counts = mask.sum(dim=1).to(dtype=torch.float32)
    summary: Dict[str, Any] = {
        "target_in_candidate_set_rate": float(has_target.float().mean().detach().cpu().item()) if mask.size(0) else 0.0,
        "missing_target_rate": float((~has_target).float().mean().detach().cpu().item()) if mask.size(0) else 0.0,
        "target_duplicate_count_max": int(target_counts.max().detach().cpu().item()) if target_counts.numel() else 0,
        "target_set_logit_variance_mean": 0.0,
        "target_set_entropy_mean": 0.0,
    }
    variances = []
    entropies = []
    for row_logits, row_mask in zip(candidate_logits.detach(), mask):
        values = row_logits[row_mask]
        if int(values.numel()) <= 0:
            continue
        variances.append(torch.var(values.float(), unbiased=False))
        probs = torch.softmax(values.float(), dim=0)
        entropies.append(-(probs * torch.log(probs.clamp_min(1e-12))).sum())
    if variances:
        summary["target_set_logit_variance_mean"] = float(torch.stack(variances).mean().cpu().item())
    if entropies:
        summary["target_set_entropy_mean"] = float(torch.stack(entropies).mean().cpu().item())
    return summary


def candidate_predictions_to_global(candidate_logits: torch.Tensor, candidate_class_index: torch.Tensor) -> torch.Tensor:
    pred_candidate = torch.argmax(candidate_logits, dim=1).long()
    class_index = candidate_class_index.to(device=candidate_logits.device, dtype=torch.long)
    return class_index[pred_candidate]


def candidate_level_accuracy(
    candidate_logits: torch.Tensor,
    candidate_class_index: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    pred_global = candidate_predictions_to_global(candidate_logits, candidate_class_index)
    target = targets.to(device=pred_global.device, dtype=torch.long).view(-1)
    return float((pred_global == target).float().mean().detach().cpu().item()) if target.numel() else 0.0
