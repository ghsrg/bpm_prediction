"""Helpers for topology-local candidate targets."""

from __future__ import annotations

from typing import Any, Dict, Sequence

import torch
import torch.nn.functional as F


def candidate_set_cross_entropy(
    candidate_logits: torch.Tensor,
    target_candidate_mask: torch.Tensor,
    *,
    missing_target_penalty: float = 0.0,
    sample_weights: torch.Tensor | None = None,
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
    if isinstance(sample_weights, torch.Tensor):
        weights = sample_weights.to(device=candidate_logits.device, dtype=candidate_logits.dtype).view(-1)
        if weights.numel() != candidate_logits.size(0):
            raise ValueError("sample_weights length must match candidate_logits rows.")
        active_weights = weights[has_target]
        denom = active_weights.sum().clamp_min(torch.finfo(active_weights.dtype).eps)
        loss = torch.sum(per_row_loss * active_weights) / denom
    else:
        loss = per_row_loss.mean()
    missing_rate = 1.0 - float(has_target.float().mean().detach().cpu().item())
    if missing_rate > 0.0 and float(missing_target_penalty) > 0.0:
        loss = loss + candidate_logits.new_tensor(float(missing_target_penalty) * missing_rate)
    return loss


def candidate_target_mask_from_labels(
    *,
    target_labels: Sequence[str],
    candidate_labels: Sequence[str],
    candidate_ids: Sequence[str] | None = None,
    device: torch.device,
) -> torch.BoolTensor:
    """Build `[B, C_v]` target mask from raw labels and topology candidate labels/ids."""

    normalized_candidates = [str(label).strip() for label in candidate_labels]
    normalized_ids = [str(i).strip() for i in candidate_ids] if candidate_ids is not None else []
    rows: list[list[bool]] = []
    for raw_label in target_labels:
        target = str(raw_label).strip()
        row = []
        for idx, candidate in enumerate(normalized_candidates):
            match_lbl = (candidate == target)
            match_id = (normalized_ids[idx] == target) if normalized_ids else False
            row.append(match_lbl or match_id)
        rows.append(row)
    return torch.tensor(rows, dtype=torch.bool, device=device)


def candidate_set_cross_entropy_from_mask(
    candidate_logits: torch.Tensor,
    target_candidate_mask: torch.Tensor,
    *,
    missing_target_penalty: float = 0.0,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Set-aware CE over topology-local candidates."""

    return candidate_set_cross_entropy(
        candidate_logits,
        target_candidate_mask,
        missing_target_penalty=missing_target_penalty,
        sample_weights=sample_weights,
    )


def candidate_set_predictions(candidate_logits: torch.Tensor) -> torch.LongTensor:
    """Return local candidate predictions for `[B, C_v]` logits."""

    if candidate_logits.dim() != 2:
        raise ValueError("candidate_logits must have shape [B, C_v].")
    return torch.argmax(candidate_logits, dim=1).long()


def candidate_set_correct(
    candidate_logits: torch.Tensor,
    target_candidate_mask: torch.Tensor,
) -> torch.FloatTensor:
    """Return per-row exact candidate-space correctness."""

    if candidate_logits.dim() != 2:
        raise ValueError("candidate_logits must have shape [B, C_v].")
    mask = target_candidate_mask.to(device=candidate_logits.device, dtype=torch.bool)
    if mask.shape != candidate_logits.shape:
        raise ValueError("target_candidate_mask must have the same shape as candidate_logits.")
    if candidate_logits.size(0) <= 0:
        return candidate_logits.new_zeros((0,), dtype=torch.float32)
    pred = candidate_set_predictions(candidate_logits)
    row_ids = torch.arange(candidate_logits.size(0), device=candidate_logits.device)
    return mask[row_ids, pred].to(dtype=torch.float32)


def candidate_set_accuracy(candidate_logits: torch.Tensor, target_candidate_mask: torch.Tensor) -> float:
    """Exact next-activity accuracy in topology-local candidate space."""

    correct = candidate_set_correct(candidate_logits, target_candidate_mask)
    return float(correct.mean().detach().cpu().item()) if correct.numel() else 0.0


def candidate_set_nll(candidate_logits: torch.Tensor, target_candidate_mask: torch.Tensor) -> torch.Tensor:
    """NLL over the probability mass of all candidates matching the target."""

    if candidate_logits.dim() != 2:
        raise ValueError("candidate_logits must have shape [B, C_v].")
    mask = target_candidate_mask.to(device=candidate_logits.device, dtype=torch.bool)
    if mask.shape != candidate_logits.shape:
        raise ValueError("target_candidate_mask must have the same shape as candidate_logits.")
    log_probs = F.log_softmax(candidate_logits, dim=1)
    has_target = mask.any(dim=1)
    if not bool(has_target.any()):
        return candidate_logits.sum() * 0.0
    masked_log_probs = log_probs.masked_fill(~mask, float("-inf"))
    return -torch.logsumexp(masked_log_probs[has_target], dim=1).mean()


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
    masked_logits = candidate_logits.clone()
    class_index = candidate_class_index.to(device=candidate_logits.device, dtype=torch.long)
    unseen_mask = (class_index < 0)
    if unseen_mask.any():
        masked_logits[:, unseen_mask] = float("-inf")
    pred_candidate = torch.argmax(masked_logits, dim=1).long()
    return class_index[pred_candidate]


def candidate_level_accuracy(
    candidate_logits: torch.Tensor,
    candidate_class_index: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    pred_global = candidate_predictions_to_global(candidate_logits, candidate_class_index)
    target = targets.to(device=pred_global.device, dtype=torch.long).view(-1)
    return float((pred_global == target).float().mean().detach().cpu().item()) if target.numel() else 0.0
