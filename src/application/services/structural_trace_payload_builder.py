"""Build compact JSON-safe structural prediction trace payloads."""

from __future__ import annotations

from typing import Any

import torch

from src.application.ports.trace_recorder_port import StructuralTraceEvent


def build_structural_prediction_trace_event(
    *,
    stage: str,
    global_index: int,
    contract: dict[str, Any],
    logits: torch.Tensor,
    effective_logits: torch.Tensor,
    probs: torch.Tensor,
    targets: torch.Tensor,
    predictions: torch.Tensor,
    model: Any,
    reverse_activity_vocab: dict[int, str],
    row_index: int,
    reason: str,
    top_k: int = 5,
) -> StructuralTraceEvent:
    payload = build_structural_prediction_trace_payload(
        stage=stage,
        global_index=global_index,
        contract=contract,
        logits=logits,
        effective_logits=effective_logits,
        probs=probs,
        targets=targets,
        predictions=predictions,
        model=model,
        reverse_activity_vocab=reverse_activity_vocab,
        row_index=row_index,
        reason=reason,
        top_k=top_k,
    )
    attributes = build_structural_prediction_trace_attributes(payload)
    return StructuralTraceEvent(
        name="structural_prediction_debug",
        inputs={
            "sample": payload["sample"],
            "run": payload["run"],
            "snapshot": payload["snapshot"],
            "contract": payload["contract"],
            "mask": payload["mask"],
        },
        outputs={
            "prediction": payload["prediction"],
            "diagnostics": payload["diagnostics"],
            "reason": payload["reason"],
        },
        attributes=attributes,
    )


def build_structural_prediction_trace_payload(
    *,
    stage: str,
    global_index: int,
    contract: dict[str, Any],
    logits: torch.Tensor,
    effective_logits: torch.Tensor,
    probs: torch.Tensor,
    targets: torch.Tensor,
    predictions: torch.Tensor,
    model: Any,
    reverse_activity_vocab: dict[int, str],
    row_index: int,
    reason: str,
    top_k: int = 5,
) -> dict[str, Any]:
    row = int(row_index)
    target_index = tensor_int_at(targets, row)
    pred_index = tensor_int_at(predictions, row)
    confidence = tensor_float_at(probs, row, pred_index)
    strict_correct = bool(target_index == pred_index)
    mask_info = _mask_payload(
        contract=contract,
        row_index=row,
        target_index=target_index,
        pred_index=pred_index,
        reverse_activity_vocab=reverse_activity_vocab,
        top_k=top_k,
    )
    prediction = {
        "target_index": int(target_index),
        "target_label": label_for_index(target_index, reverse_activity_vocab),
        "pred_index": int(pred_index),
        "pred_label": label_for_index(pred_index, reverse_activity_vocab),
        "confidence": confidence,
        "strict_correct": strict_correct,
        "top_k": top_k_payload(probs, reverse_activity_vocab, k=top_k, row_index=row),
    }
    prediction.update(_candidate_prediction_payload(contract=contract, model=model, row_index=row, top_k=top_k))
    process_version = _batch_value(contract.get("process_version_labels"), row, default=None)
    if process_version is None:
        process_version = _batch_value(contract.get("version_labels"), row, default="__unknown__")
    prefix_last_activity_index = _tensor_attr_at(contract.get("prefix_last_activity_idx"), row)
    prefix_last_activity = (
        label_for_index(prefix_last_activity_index, reverse_activity_vocab)
        if prefix_last_activity_index is not None and int(prefix_last_activity_index) >= 0
        else "__unknown__"
    )

    model_type = str(model.__class__.__name__)
    fusion_mode = str(getattr(model, "fusion_mode", ""))
    structural_mode = bool(getattr(model, "structural_mode", False))
    if model_type == "EOPKGTopologyConditioned":
        structural_mode = True
        if not fusion_mode:
            fusion_mode = "TopologyConditionedCandidateScoring"

    payload = {
        "schema_version": "1.0",
        "stage": str(stage),
        "sample": {
            "global_index": int(global_index),
            "trace_idx": _tensor_attr_at(contract.get("trace_idx"), row),
            "prefix_idx": _tensor_attr_at(contract.get("prefix_idx"), row),
            "prefix_len": _tensor_attr_at(contract.get("prefix_len"), row),
            "prefix_last_activity_index": prefix_last_activity_index,
            "prefix_last_activity": prefix_last_activity,
            "process_version": str(process_version or "__unknown__"),
        },
        "run": {
            "model_type": model_type,
            "fusion_mode": fusion_mode,
            "fusion_family": "topology_conditioned" if model_type == "EOPKGTopologyConditioned" else fusion_mode,
            "structural_mode": structural_mode,
            "topology_conditioning_mode": _json_safe(getattr(model, "topology_conditioning_mode", None)),
            "candidate_identity_mode": _json_safe(contract.get("candidate_identity_mode")),
            "candidate_scoring_mode": _json_safe(
                getattr(model, "candidate_scoring_mode", getattr(model, "candidate_scoring", None))
            ),
            "candidate_pooling": _json_safe(getattr(model, "candidate_pooling", None)),
        },
        "snapshot": _snapshot_payload(contract, row),
        "contract": _contract_payload(contract),
        "prediction": prediction,
        "mask": mask_info,
        "diagnostics": _diagnostics_payload(
            model=model,
            logits=logits,
            effective_logits=effective_logits,
            probs=probs,
            row_index=row,
            target_index=target_index,
            pred_index=pred_index,
        ),
        "reason": str(reason),
    }
    return payload


def build_structural_prediction_trace_attributes(payload: dict[str, Any]) -> dict[str, Any]:
    prediction = payload.get("prediction", {})
    mask = payload.get("mask", {})
    run = payload.get("run", {})
    sample = payload.get("sample", {})
    snapshot = payload.get("snapshot", {})
    diagnostics = payload.get("diagnostics", {})
    generic = diagnostics.get("generic", {}) if isinstance(diagnostics, dict) else {}
    class_aware = diagnostics.get("class_aware_structural_scoring", {}) if isinstance(diagnostics, dict) else {}
    struct_xattn = diagnostics.get("struct_xattn", {}) if isinstance(diagnostics, dict) else {}
    topology_conditioned = (
        diagnostics.get("topology_conditioned_candidate_scoring", {}) if isinstance(diagnostics, dict) else {}
    )

    attrs: dict[str, Any] = {
        "stage": str(payload.get("stage", "")),
        "reason": str(payload.get("reason", "")),
        "model_type": str(run.get("model_type", "")),
        "fusion_mode": str(run.get("fusion_mode", "")),
        "fusion_family": str(run.get("fusion_family", "")),
        "structural_mode": bool(run.get("structural_mode", False)),
        "topology_conditioning_mode": str(run.get("topology_conditioning_mode", "")),
        "candidate_identity_mode": str(run.get("candidate_identity_mode", "")),
        "process_version": str(sample.get("process_version", "__unknown__")),
        "prefix_last_activity": str(sample.get("prefix_last_activity", "__unknown__")),
        "stats_snapshot_version": str(snapshot.get("stats_snapshot_version", "")),
        "strict_correct": bool(prediction.get("strict_correct", False)),
        "target_index": int(prediction.get("target_index", -1)),
        "pred_index": int(prediction.get("pred_index", -1)),
        "confidence": _safe_attr_float(prediction.get("confidence")),
        "target_in_mask": _optional_bool(mask.get("target_in_mask")),
        "pred_in_mask": _optional_bool(mask.get("prediction_in_mask")),
        "prediction_in_mask": _optional_bool(mask.get("prediction_in_mask")),
        "strict_error_but_allowed": _optional_bool(mask.get("strict_error_but_allowed")),
        "mask_cardinality": _safe_attr_float(mask.get("mask_cardinality")),
        "prediction_entropy": _safe_attr_float(generic.get("prediction_entropy")),
    }
    if run.get("candidate_scoring_mode") is not None:
        attrs["candidate_scoring_mode"] = str(run.get("candidate_scoring_mode", ""))
    if run.get("candidate_pooling") is not None:
        attrs["candidate_pooling"] = str(run.get("candidate_pooling", ""))
    if sample.get("prefix_last_activity_index") is not None:
        attrs["prefix_last_activity_index"] = int(sample.get("prefix_last_activity_index", -1))
    if "structural_to_observed_logit_ratio" in class_aware:
        attrs["structural_to_observed_logit_ratio"] = _safe_attr_float(
            class_aware.get("structural_to_observed_logit_ratio")
        )
    if "struct_xattn_attention_entropy" in struct_xattn:
        attrs["struct_xattn_attention_entropy"] = _safe_attr_float(
            struct_xattn.get("struct_xattn_attention_entropy")
        )
    for key in (
        "struct_xattn_raw_to_observed_ratio",
        "struct_xattn_pre_norm_to_observed_ratio",
        "struct_xattn_post_norm_to_observed_ratio",
    ):
        if key in struct_xattn:
            attrs[key] = _safe_attr_float(struct_xattn.get(key))
    for key in (
        "candidate_temperature",
        "candidate_prediction_entropy",
        "candidate_score_gap",
        "candidate_node_score_mean_abs",
        "candidate_class_score_mean_abs",
        "duplicate_candidate_count_max",
        "candidate_dynamic_count",
        "impulse_activation_mean_abs",
        "impulse_activation_max_abs",
        "impulse_to_base_node_ratio",
        "impulse_gnn_oversmoothing_ratio",
        "candidate_unseen_score_mean",
        "candidate_seen_score_mean",
        "candidate_seen_unseen_score_gap",
    ):
        if key in topology_conditioned:
            attrs[key] = _safe_attr_float(topology_conditioned.get(key))
    return {str(key): value for key, value in attrs.items() if value is not None}


def top_k_payload(
    probs: torch.Tensor,
    reverse_vocab: dict[int, str],
    *,
    k: int,
    row_index: int = 0,
) -> list[dict[str, Any]]:
    safe = _safe_tensor(probs)
    if safe.dim() == 1:
        row = safe
    else:
        row = safe[int(row_index)]
    count = max(1, min(int(k), int(row.numel())))
    values, indices = torch.topk(row, k=count)
    result: list[dict[str, Any]] = []
    for idx, value in zip(indices.cpu().tolist(), values.cpu().tolist()):
        class_idx = int(idx)
        result.append(
            {
                "index": class_idx,
                "label": label_for_index(class_idx, reverse_vocab),
                "probability": _round_float(float(value)),
            }
        )
    return result


def label_for_index(index: int, reverse_vocab: dict[int, str]) -> str:
    return str(reverse_vocab.get(int(index), f"__class_{int(index)}__"))


def tensor_scalar(value: torch.Tensor | int | float | None) -> float | None:
    if isinstance(value, (int, float)):
        return _round_float(float(value))
    if not isinstance(value, torch.Tensor) or value.numel() <= 0:
        return None
    safe = _safe_tensor(value)
    return _round_float(float(safe.mean().item()))


def tensor_int_at(value: torch.Tensor, row: int) -> int:
    safe = value.detach().cpu().view(-1)
    return int(safe[int(row)].item())


def tensor_float_at(value: torch.Tensor, row: int, column: int | None = None) -> float:
    safe = _safe_tensor(value)
    if safe.dim() == 1:
        return _round_float(float(safe[int(row)].item()))
    if column is None:
        return _round_float(float(safe[int(row)].mean().item()))
    return _round_float(float(safe[int(row), int(column)].item()))


def _diagnostics_payload(
    *,
    model: Any,
    logits: torch.Tensor,
    effective_logits: torch.Tensor,
    probs: torch.Tensor,
    row_index: int,
    target_index: int,
    pred_index: int,
) -> dict[str, Any]:
    row_probs = _safe_tensor(probs)[int(row_index)]
    entropy = -torch.sum(row_probs * torch.log(torch.clamp(row_probs, min=1e-12)))
    diagnostics: dict[str, Any] = {
        "generic": {
            "final_logits_mean_abs": tensor_scalar(torch.abs(_safe_tensor(effective_logits))),
            "raw_logits_mean_abs": tensor_scalar(torch.abs(_safe_tensor(logits))),
            "prediction_entropy": _round_float(float(entropy.item())),
        }
    }

    observed_logits = getattr(model, "last_observed_logits", None)
    structural_logits = getattr(model, "last_structural_class_logits", None)
    if isinstance(observed_logits, torch.Tensor) and isinstance(structural_logits, torch.Tensor):
        observed = _safe_tensor(observed_logits)
        structural = _safe_tensor(structural_logits)
        if observed.dim() == 2 and structural.dim() == 2 and observed.size(0) > row_index and structural.size(0) > row_index:
            observed_mean_abs = float(torch.abs(observed).mean().item())
            structural_mean_abs = float(torch.abs(structural).mean().item())
            diagnostics["class_aware_structural_scoring"] = {
                "observed_logits_mean_abs": _round_float(observed_mean_abs),
                "structural_logits_mean_abs": _round_float(structural_mean_abs),
                "structural_to_observed_logit_ratio": _round_float(
                    structural_mean_abs / max(observed_mean_abs, 1e-12)
                ),
                "target_observed_logit": tensor_float_at(observed, row_index, target_index),
                "target_structural_logit": tensor_float_at(structural, row_index, target_index),
                "pred_observed_logit": tensor_float_at(observed, row_index, pred_index),
                "pred_structural_logit": tensor_float_at(structural, row_index, pred_index),
            }

    struct_xattn_values = {
        "struct_xattn_raw_context_mean_abs": tensor_scalar(
            getattr(model, "last_struct_xattn_raw_context_mean_abs", None)
        ),
        "struct_xattn_pre_norm_delta_mean_abs": tensor_scalar(
            getattr(model, "last_struct_xattn_pre_norm_delta_mean_abs", None)
        ),
        "struct_xattn_post_norm_delta_mean_abs": tensor_scalar(
            getattr(model, "last_struct_xattn_post_norm_delta_mean_abs", None)
        ),
        "struct_xattn_raw_to_observed_ratio": tensor_scalar(
            getattr(model, "last_struct_xattn_raw_to_observed_ratio", None)
        ),
        "struct_xattn_pre_norm_to_observed_ratio": tensor_scalar(
            getattr(model, "last_struct_xattn_pre_norm_to_observed_ratio", None)
        ),
        "struct_xattn_post_norm_to_observed_ratio": tensor_scalar(
            getattr(model, "last_struct_xattn_post_norm_to_observed_ratio", None)
        ),
        "struct_xattn_to_observed_ratio": tensor_scalar(
            getattr(model, "last_struct_xattn_to_observed_ratio", None)
        ),
        "struct_xattn_attention_entropy": tensor_scalar(
            getattr(model, "last_struct_xattn_attention_entropy", None)
        ),
        "struct_xattn_gate_mean": tensor_scalar(getattr(model, "last_struct_xattn_gate_mean", None)),
    }
    struct_xattn_values = {key: value for key, value in struct_xattn_values.items() if value is not None}
    if struct_xattn_values:
        diagnostics["struct_xattn"] = struct_xattn_values
    topology_conditioned_values = {
        "candidate_node_score_mean_abs": tensor_scalar(
            getattr(model, "last_candidate_node_score_mean_abs", None)
        ),
        "candidate_class_score_mean_abs": tensor_scalar(
            getattr(model, "last_candidate_class_score_mean_abs", None)
        ),
        "duplicate_candidate_count_max": tensor_scalar(
            getattr(model, "last_duplicate_candidate_count_max", None)
        ),
        "candidate_temperature": tensor_scalar(getattr(model, "last_candidate_temperature", None)),
        "candidate_temperature_trainable": _json_safe(
            getattr(model, "last_candidate_temperature_trainable", None)
        ),
        "candidate_prediction_entropy": tensor_scalar(
            getattr(model, "last_candidate_prediction_entropy", None)
        ),
        "candidate_target_score": tensor_scalar(getattr(model, "last_candidate_target_score", None)),
        "candidate_pred_score": tensor_scalar(getattr(model, "last_candidate_pred_score", None)),
        "candidate_score_gap": tensor_scalar(getattr(model, "last_candidate_score_gap", None)),
        "candidate_dynamic_count": tensor_scalar(getattr(model, "last_candidate_dynamic_count", None)),
        "candidate_class_index": _json_safe(getattr(model, "last_candidate_class_index", None)),
        "candidate_ids": _json_safe(getattr(model, "last_candidate_ids", None)),
        "candidate_labels": _json_safe(getattr(model, "last_candidate_labels", None)),
        "candidate_is_unseen": _json_safe(getattr(model, "last_candidate_is_unseen", None)),
        "impulse_activation_mean_abs": tensor_scalar(getattr(model, "last_impulse_activation_mean_abs", None)),
        "impulse_activation_max_abs": tensor_scalar(getattr(model, "last_impulse_activation_max_abs", None)),
        "impulse_to_base_node_ratio": tensor_scalar(getattr(model, "last_impulse_to_base_node_ratio", None)),
        "impulse_gnn_oversmoothing_ratio": tensor_scalar(
            getattr(model, "last_impulse_gnn_oversmoothing_ratio", None)
        ),
        "candidate_unseen_score_mean": tensor_scalar(getattr(model, "last_candidate_unseen_score_mean", None)),
        "candidate_seen_score_mean": tensor_scalar(getattr(model, "last_candidate_seen_score_mean", None)),
        "candidate_seen_unseen_score_gap": tensor_scalar(
            getattr(model, "last_candidate_seen_unseen_score_gap", None)
        ),
    }
    topology_conditioned_values = {
        key: value for key, value in topology_conditioned_values.items() if value is not None
    }
    if topology_conditioned_values:
        diagnostics["topology_conditioned_candidate_scoring"] = topology_conditioned_values
    return diagnostics


def _mask_payload(
    *,
    contract: dict[str, Any],
    row_index: int,
    target_index: int,
    pred_index: int,
    reverse_activity_vocab: dict[int, str],
    top_k: int,
) -> dict[str, Any]:
    raw_mask = contract.get("allowed_target_mask")
    if not isinstance(raw_mask, torch.Tensor) or raw_mask.numel() <= 0:
        return {
            "target_in_mask": None,
            "prediction_in_mask": None,
            "strict_error_but_allowed": None,
            "mask_cardinality": None,
            "allowed_top_k": [],
        }
    mask = raw_mask.detach().bool().cpu()
    if mask.dim() == 1:
        row = mask
    else:
        row = mask[int(row_index)]
    target_in_mask = bool(row[int(target_index)].item()) if int(target_index) < int(row.numel()) else False
    prediction_in_mask = bool(row[int(pred_index)].item()) if int(pred_index) < int(row.numel()) else False
    allowed_indices = [int(idx) for idx, value in enumerate(row.tolist()) if bool(value)]
    return {
        "target_in_mask": target_in_mask,
        "prediction_in_mask": prediction_in_mask,
        "strict_error_but_allowed": bool(target_index != pred_index and prediction_in_mask),
        "mask_cardinality": int(len(allowed_indices)),
        "allowed_top_k": [
            {"index": idx, "label": label_for_index(idx, reverse_activity_vocab)}
            for idx in allowed_indices[: max(1, int(top_k))]
        ],
    }


def _snapshot_payload(contract: dict[str, Any], row_index: int) -> dict[str, Any]:
    return {
        "stats_snapshot_version": _batch_value(contract.get("stats_snapshot_versions"), row_index, default=None),
        "stats_snapshot_as_of_ts": _batch_value(contract.get("stats_snapshot_as_of_ts_batch"), row_index, default=None),
        "stats_allowed": _batch_value(contract.get("stats_allowed_batch"), row_index, default=None),
        "stats_missing_asof_snapshot": _batch_value(
            contract.get("stats_missing_asof_snapshot_batch"),
            row_index,
            default=None,
        ),
    }


def _contract_payload(contract: dict[str, Any]) -> dict[str, Any]:
    x_cat = contract.get("x_cat")
    edge_index = contract.get("edge_index")
    struct_x = contract.get("struct_x")
    structural_edge_index = contract.get("structural_edge_index")
    return {
        "observed_node_count": _tensor_rows(x_cat),
        "observed_edge_count": _edge_count(edge_index),
        "struct_node_count": _tensor_rows(struct_x),
        "struct_edge_count": _edge_count(structural_edge_index),
        "struct_feature_dim": _tensor_cols(struct_x),
        "has_allowed_target_mask": isinstance(contract.get("allowed_target_mask"), torch.Tensor),
        "candidate_ids": _json_safe(contract.get("candidate_ids")),
        "candidate_labels": _json_safe(contract.get("candidate_labels")),
        "has_candidate_allowed_target_mask": isinstance(contract.get("candidate_allowed_target_mask"), torch.Tensor),
    }


def _candidate_prediction_payload(
    *,
    contract: dict[str, Any],
    model: Any,
    row_index: int,
    top_k: int,
) -> dict[str, Any]:
    logits = getattr(model, "last_candidate_logits", None)
    ids = getattr(model, "last_candidate_ids", None)
    labels = getattr(model, "last_candidate_labels", None)
    if not isinstance(logits, torch.Tensor) or logits.numel() <= 0:
        return {}
    safe = _safe_tensor(logits)
    if safe.dim() == 1:
        row = safe
    else:
        row = safe[min(int(row_index), int(safe.size(0)) - 1)]
    ids_list = [str(item) for item in ids] if isinstance(ids, (list, tuple)) else []
    labels_list = [str(item) for item in labels] if isinstance(labels, (list, tuple)) else []
    count = max(1, min(int(top_k), int(row.numel())))
    values, indices = torch.topk(row, k=count)
    top_ids = []
    top_labels = []
    for idx in indices.cpu().tolist():
        top_ids.append(ids_list[int(idx)] if int(idx) < len(ids_list) else f"__candidate_{int(idx)}__")
        top_labels.append(labels_list[int(idx)] if int(idx) < len(labels_list) else f"__candidate_{int(idx)}__")
    payload: dict[str, Any] = {
        "top_k_candidate_ids": top_ids,
        "top_k_candidate_labels": top_labels,
        "top_k_candidate_logits": [_round_float(float(v)) for v in values.cpu().tolist()],
    }
    prefix_state = contract.get("struct_prefix_state_x")
    node_to_candidate = contract.get("struct_node_to_candidate_index")
    if isinstance(prefix_state, torch.Tensor) and isinstance(node_to_candidate, torch.Tensor):
        state = prefix_state.detach().cpu()
        if state.dim() == 3:
            state = state[min(int(row_index), int(state.size(0)) - 1)]
        if state.dim() == 2 and state.size(1) > 2:
            last_node = int(torch.argmax(state[:, 2].float()).item())
            node_to_candidate_flat = node_to_candidate.detach().cpu().view(-1)
            if last_node < int(node_to_candidate_flat.numel()):
                local_idx = int(node_to_candidate_flat[last_node].item())
                if 0 <= local_idx < len(ids_list):
                    payload["last_event_candidate_id"] = ids_list[local_idx]
    target_label = _batch_value(contract.get("target_label"), row_index, default=None)
    if target_label is not None:
        target = str(target_label)
        payload["target_candidate_ids"] = [
            ids_list[idx]
            for idx, label in enumerate(labels_list)
            if label == target or (idx < len(ids_list) and ids_list[idx] == target)
        ]
    return payload


def _safe_tensor(value: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(value.detach().float().cpu(), nan=0.0, posinf=1e6, neginf=-1e6)


def _tensor_rows(value: Any) -> int:
    return int(value.size(0)) if isinstance(value, torch.Tensor) and value.dim() >= 1 else 0


def _tensor_cols(value: Any) -> int:
    return int(value.size(1)) if isinstance(value, torch.Tensor) and value.dim() >= 2 else 0


def _edge_count(value: Any) -> int:
    return int(value.size(1)) if isinstance(value, torch.Tensor) and value.dim() >= 2 else 0


def _tensor_attr_at(value: Any, row_index: int) -> int | None:
    if not isinstance(value, torch.Tensor) or value.numel() <= 0:
        return None
    flat = value.detach().cpu().view(-1)
    if int(row_index) >= int(flat.numel()):
        return None
    return int(flat[int(row_index)].item())


def _batch_value(value: Any, row_index: int, *, default: Any) -> Any:
    if isinstance(value, list):
        if int(row_index) < len(value):
            return _json_safe(value[int(row_index)])
        return default
    if isinstance(value, torch.Tensor):
        return _tensor_attr_at(value, row_index)
    if value is not None:
        return _json_safe(value)
    return default


def _json_safe(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return tensor_scalar(value)
        return _safe_tensor(value).tolist()
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value)


def _safe_attr_float(value: Any) -> float | int | None:
    if value is None:
        return None
    try:
        return _round_float(float(value))
    except (TypeError, ValueError):
        return None


def _round_float(value: float) -> float:
    return round(float(value), 6)
