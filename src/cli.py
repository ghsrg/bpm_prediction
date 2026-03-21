"""Composition Root for MVP1 training pipeline."""

# Р’С–РґРїРѕРІС–РґРЅРѕ РґРѕ:
# - AGENT_GUIDE.MD -> СЂРѕР·РґС–Р» 2 (Clean Architecture, MVP1 scope) С– СЂРѕР·РґС–Р» 5 (РїРѕСЃР»С–РґРѕРІРЅС–СЃС‚СЊ РїРѕС‚РѕРєС–РІ)
# - ARCHITECTURE_RULES.MD -> СЂРѕР·РґС–Р» 2-4 (Application orchestration С‡РµСЂРµР· РїРѕСЂС‚Рё)
# - DATA_FLOWS_MVP1.MD -> СЂРѕР·РґС–Р» 3.1 (Training Pipeline)
# - EVF_MVP1.MD -> СЂРѕР·РґС–Р» 3 (Strict Temporal Split) С– СЂРѕР·РґС–Р» 5 (MLflow tracking)

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import math
from pathlib import Path
import logging
import random
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm

from src.adapters.ingestion.camunda_trace_adapter import CamundaTraceAdapter
from src.adapters.ingestion.xes_adapter import XESAdapter
from src.application.ports.xes_adapter_port import IXESAdapter
from src.application.use_cases.trainer import ModelTrainer
from src.domain.entities.raw_trace import RawTrace
from src.domain.entities.feature_config import parse_feature_configs
from src.domain.models.factory import create_model
from src.domain.services.dynamic_graph_builder import DynamicGraphBuilder
from src.domain.services.feature_encoder import FeatureEncoder
from src.domain.services.prefix_policy import PrefixPolicy
from src.domain.services.schema_resolver import SchemaResolver
from src.infrastructure.config.yaml_loader import load_yaml_with_includes
from src.infrastructure.repositories.knowledge_graph_repository_factory import (
    build_knowledge_graph_repository,
    get_knowledge_graph_settings,
)
from src.infrastructure.tracking.mlflow_tracker import MLflowTracker


def _resolve_config_path(config_arg: str) -> Path:
    """Resolve config path; if filename only, lookup in ./configs."""
    path = Path(config_arg)
    if path.exists():
        return path

    fallback = Path("configs") / config_arg
    if fallback.exists():
        return fallback

    raise FileNotFoundError(f"Config file not found: {config_arg}")


def load_yaml_config(config_arg: str) -> Dict[str, Any]:
    """Load runtime configuration from YAML file with recursive include support."""
    config_path = _resolve_config_path(config_arg)

    # Р—Р°РІР°РЅС‚Р°Р¶СѓС”РјРѕ РєРѕРЅС„С–Рі С–Р· РїС–РґС‚СЂРёРјРєРѕСЋ include/deep-merge РґР»СЏ РјРѕРґСѓР»СЊРЅРёС… playbook-С–РІ.
    loaded = load_yaml_with_includes(config_path)

    # РќРѕСЂРјР°Р»С–Р·СѓС”РјРѕ СЃРµРєС†С–СЋ experiment РґР»СЏ РјР°Р№Р±СѓС‚РЅСЊРѕРіРѕ СЂРѕСѓС‚РёРЅРіСѓ СЂРµР¶РёРјС–РІ (train/eval/infer).
    experiment_cfg = loaded.get("experiment")
    if experiment_cfg is None:
        loaded["experiment"] = {"mode": "train"}
    elif isinstance(experiment_cfg, dict):
        experiment_cfg.setdefault("mode", "train")
    else:
        raise ValueError("Config key 'experiment' must be a mapping if provided.")

    return loaded


def _flatten_config_dict(config: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Flatten nested config dict into dotted-key mapping."""
    flat: Dict[str, Any] = {}
    for key, value in config.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(_flatten_config_dict(value, prefix=full_key))
            continue
        if isinstance(value, (list, tuple, set)):
            flat[full_key] = ",".join(str(item) for item in value)
            continue
        flat[full_key] = "None" if value is None else value
    return flat


def _truncate_param_value(value: Any, max_len: int = 480) -> Any:
    """Ensure MLflow param value stays within safe length limits."""
    if isinstance(value, (int, float, bool)) and not isinstance(value, bool):
        return value
    text = str(value)
    if len(text) <= max_len:
        return text
    return f"{text[:max_len]}...[truncated]"


def _iter_mlflow_param_blocks(config: Dict[str, Any]) -> Iterable[tuple[str, Any]]:
    """Yield only config blocks that should be logged as params."""
    if "seed" in config:
        yield ("seed", config["seed"])
    for section in ("experiment", "data", "model", "training"):
        if section in config:
            yield (section, config[section])


def _build_mlflow_params(config: Dict[str, Any], max_value_len: int = 480) -> Dict[str, Any]:
    """Build sanitized flattened params subset for MLflow logging."""
    params: Dict[str, Any] = {}
    for prefix, payload in _iter_mlflow_param_blocks(config):
        if isinstance(payload, dict):
            flat = _flatten_config_dict(payload, prefix=prefix)
            for key, value in flat.items():
                params[key] = _truncate_param_value(value, max_len=max_value_len)
        else:
            params[prefix] = _truncate_param_value(payload, max_len=max_value_len)
    return params


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across random/numpy/torch backends."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _extract_base_vocabularies(feature_encoder: FeatureEncoder) -> Tuple[Dict[str, int], Dict[str, int], str]:
    """Extract activity/resource vocabularies from feature encoder artifacts."""
    activity_feature = feature_encoder.activity_feature_name
    resource_feature = feature_encoder.role_to_feature.get("resource", "org:resource")
    activity_vocab = feature_encoder.categorical_vocabs.get(activity_feature, {"<UNK>": 0})
    resource_vocab = feature_encoder.categorical_vocabs.get(resource_feature, {"<UNK>": 0})
    return activity_vocab, resource_vocab, activity_feature


def _build_normalization_stats() -> Dict[str, Dict[str, float]]:
    """Create MVP1 placeholder normalization stats (mu=0, sigma=1)."""
    return {
        "duration": {"mu": 0.0, "sigma": 1.0},
        "time_since_case_start": {"mu": 0.0, "sigma": 1.0},
        "time_since_previous_event": {"mu": 0.0, "sigma": 1.0},
    }


def _parse_split_ratio(experiment_cfg: Dict[str, Any]) -> Tuple[float, float, float]:
    """Parse split ratio with strict validation for train/val/test proportions."""
    raw_ratio = experiment_cfg.get("split_ratio", [0.7, 0.2, 0.1])
    if not isinstance(raw_ratio, Sequence) or isinstance(raw_ratio, (str, bytes)) or len(raw_ratio) != 3:
        raise ValueError("experiment.split_ratio must be a 3-item list [train, val, test].")

    ratios = [float(item) for item in raw_ratio]
    if any(item < 0.0 for item in ratios):
        raise ValueError("experiment.split_ratio must contain non-negative values.")

    ratio_sum = sum(ratios)
    if not math.isclose(ratio_sum, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(f"experiment.split_ratio must sum to 1.0, got {ratio_sum:.6f}")

    return ratios[0], ratios[1], ratios[2]


def _resolve_dataset_name(data_cfg: Dict[str, Any], log_path: str) -> str:
    """Resolve stable dataset name used as process_version fallback in adapters."""
    candidates = [
        data_cfg.get("dataset_name"),
        data_cfg.get("dataset_label"),
    ]
    if str(log_path).strip():
        candidates.append(Path(log_path).stem)
    for candidate in candidates:
        value = str(candidate).strip() if candidate is not None else ""
        if value:
            return value
    return "default_dataset"


def _inject_dataset_name_mapping(mapping_cfg: Dict[str, Any], dataset_name: str) -> Dict[str, Any]:
    """Attach dataset_name to xes_adapter mapping block without mutating input config."""
    result = dict(mapping_cfg)
    xes_cfg_raw = result.get("xes_adapter", {})
    xes_cfg = dict(xes_cfg_raw) if isinstance(xes_cfg_raw, dict) else {}
    xes_cfg.setdefault("dataset_name", dataset_name)
    result["xes_adapter"] = xes_cfg
    camunda_cfg_raw = result.get("camunda_adapter", {})
    camunda_cfg = dict(camunda_cfg_raw) if isinstance(camunda_cfg_raw, dict) else {}
    camunda_cfg.setdefault("dataset_name", dataset_name)
    camunda_cfg.setdefault("process_name", dataset_name)
    result["camunda_adapter"] = camunda_cfg
    result.setdefault("dataset_name", dataset_name)
    return result


def _resolve_trace_adapter_kind(mapping_cfg: Dict[str, Any]) -> str:
    """Resolve ingestion adapter kind from mapping config."""
    adapter_name = str(mapping_cfg.get("adapter", "")).strip().lower()
    if adapter_name:
        return adapter_name
    if isinstance(mapping_cfg.get("camunda_adapter"), dict):
        return "camunda"
    return "xes"


def _build_trace_adapter(mapping_cfg: Dict[str, Any]) -> IXESAdapter:
    """Construct trace adapter by mapping config without affecting model wiring."""
    kind = _resolve_trace_adapter_kind(mapping_cfg)
    if kind == "camunda":
        return CamundaTraceAdapter()
    return XESAdapter()


def _apply_fraction(traces: Sequence[RawTrace], fraction: float) -> List[RawTrace]:
    """Apply chronological fraction on traces after temporal ordering."""
    traces_with_events = [trace for trace in traces if trace.events]
    ordered = sorted(traces_with_events, key=lambda tr: tr.events[0].timestamp)

    if fraction >= 1.0:
        return ordered
    if fraction <= 0.0:
        raise ValueError("experiment.fraction must be within (0.0, 1.0].")

    keep_count = max(1, int(len(ordered) * fraction))
    return ordered[:keep_count]


def _strict_temporal_split(
    traces: Sequence[RawTrace],
    split_ratio: Tuple[float, float, float],
    split_strategy: str,
) -> Tuple[List[RawTrace], List[RawTrace], List[RawTrace]]:
    """Apply strict chronological split by configured strategy and ratio."""
    if split_strategy == "time":
        split_strategy = "temporal"
    if split_strategy not in {"temporal", "none"}:
        raise ValueError(f"Unsupported experiment.split_strategy '{split_strategy}'.")

    train_ratio, val_ratio, _ = split_ratio
    traces_with_events = [trace for trace in traces if trace.events]
    ordered = sorted(traces_with_events, key=lambda tr: tr.events[0].timestamp)

    total = len(ordered)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    return list(ordered[:train_end]), list(ordered[train_end:val_end]), list(ordered[val_end:])


def _apply_cascade_prepare(
    traces: Sequence[RawTrace],
    *,
    mode: str,
    split_strategy: str,
    train_ratio: float,
    fraction: float,
) -> List[RawTrace]:
    """Apply cascade filter in contract order: temporal sort -> macro split -> fraction."""
    if split_strategy not in {"temporal", "none"}:
        raise ValueError("experiment.split_strategy must be 'temporal' or 'none'.")
    if train_ratio < 0.0 or train_ratio > 1.0:
        raise ValueError("experiment.train_ratio must be within [0.0, 1.0].")
    if fraction <= 0.0 or fraction > 1.0:
        raise ValueError("experiment.fraction must be within (0.0, 1.0].")

    traces_with_events = [trace for trace in traces if trace.events]
    if split_strategy == "temporal":
        ordered = sorted(traces_with_events, key=lambda tr: tr.events[0].timestamp)
    else:
        ordered = list(traces_with_events)

    split_idx = int(len(ordered) * train_ratio)
    mode_key = str(mode).strip().lower()
    if mode_key == "train":
        macro = ordered[:split_idx]
    elif mode_key in {"eval_drift", "eval_cross_dataset"}:
        macro = ordered[split_idx:]
    else:
        macro = ordered

    if fraction >= 1.0:
        return macro
    keep_count = int(len(macro) * fraction)
    return macro[:keep_count]


def _safe_iso_to_epoch(value: Any) -> float | None:
    """Parse ISO datetime string into UTC epoch seconds; return None on invalid values."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    normalized = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return float(parsed.timestamp())


def _build_graph_dataset(
    traces: Sequence[RawTrace],
    prefix_policy: PrefixPolicy,
    graph_builder: DynamicGraphBuilder,
    version_to_idx: Dict[str, int],
    stats_snapshot_version_to_idx: Dict[str, int],
    *,
    show_progress: bool,
    tqdm_disable: bool,
    desc: str,
) -> List[Data]:
    """Convert traces into a list of PyG Data graphs via prefix slicing + graph builder."""
    dataset: List[Data] = []
    iterator = tqdm(
        traces,
        desc=desc,
        unit="trace",
        leave=False,
        disable=(not show_progress) or tqdm_disable,
    )
    for idx, trace in enumerate(iterator, start=1):
        for prefix_slice in prefix_policy.generate_slices(trace):
            contract = graph_builder.build_graph(prefix_slice)
            version_label = str(prefix_slice.process_version)
            version_idx = version_to_idx.setdefault(version_label, len(version_to_idx))
            prefix_len = int(len(prefix_slice.prefix_events))
            payload: Dict[str, Any] = {
                "x_cat": contract["x_cat"],
                "x_num": contract["x_num"],
                "edge_index": contract["edge_index"],
                "edge_type": contract["edge_type"],
                "y": contract["y"],
                "num_nodes": int(contract["num_nodes"]),
                "prefix_len": torch.tensor([prefix_len], dtype=torch.long),
                "process_version_idx": torch.tensor([version_idx], dtype=torch.long),
            }
            allowed_mask = contract.get("allowed_target_mask")
            if isinstance(allowed_mask, torch.Tensor):
                payload["allowed_target_mask"] = (
                    allowed_mask.unsqueeze(0) if allowed_mask.dim() == 1 else allowed_mask
                )
            struct_x = contract.get("struct_x")
            if isinstance(struct_x, torch.Tensor):
                payload["struct_x"] = struct_x
            structural_edge_index = contract.get("structural_edge_index")
            if isinstance(structural_edge_index, torch.Tensor):
                payload["structural_edge_index"] = structural_edge_index
            structural_edge_weight = contract.get("structural_edge_weight")
            if isinstance(structural_edge_weight, torch.Tensor):
                payload["structural_edge_weight"] = structural_edge_weight
            stats_allowed = contract.get("stats_allowed")
            if isinstance(stats_allowed, bool):
                payload["stats_allowed"] = torch.tensor([1 if stats_allowed else 0], dtype=torch.long)
            snapshot_seq_raw = contract.get("stats_snapshot_version_seq")
            if isinstance(snapshot_seq_raw, (int, float)) and not isinstance(snapshot_seq_raw, bool):
                snapshot_idx = int(snapshot_seq_raw)
                snapshot_version = f"k{snapshot_idx:06d}"
                stats_snapshot_version_to_idx.setdefault(snapshot_version, snapshot_idx)
                payload["stats_snapshot_version_idx"] = torch.tensor([snapshot_idx], dtype=torch.long)
            else:
                # Backward compatibility for contracts that still expose text metadata.
                stats_snapshot_version = contract.get("stats_snapshot_version")
                if stats_snapshot_version is not None:
                    snapshot_version = str(stats_snapshot_version).strip()
                    if snapshot_version:
                        snapshot_idx = stats_snapshot_version_to_idx.setdefault(snapshot_version, len(stats_snapshot_version_to_idx))
                        payload["stats_snapshot_version_idx"] = torch.tensor([snapshot_idx], dtype=torch.long)

            snapshot_as_of_epoch = None
            snapshot_as_of_epoch_raw = contract.get("stats_snapshot_as_of_epoch")
            if isinstance(snapshot_as_of_epoch_raw, (int, float)) and not isinstance(snapshot_as_of_epoch_raw, bool):
                snapshot_as_of_epoch = float(snapshot_as_of_epoch_raw)
            if snapshot_as_of_epoch is None:
                snapshot_as_of_ts = contract.get("stats_snapshot_as_of_ts")
                snapshot_as_of_epoch = _safe_iso_to_epoch(snapshot_as_of_ts)
            if snapshot_as_of_epoch is not None:
                payload["stats_snapshot_as_of_epoch"] = torch.tensor([snapshot_as_of_epoch], dtype=torch.float64)
            dataset.append(
                Data(**payload)
            )
        if idx % 1000 == 0:
            iterator.set_postfix({"graphs": len(dataset)})
    return dataset


def _resolve_model_family(model_type: str) -> str:
    text = str(model_type).strip().lower()
    if text.startswith("eopkg"):
        return "eopkg"
    if text.startswith("baseline"):
        return "baseline"
    return "custom"


def _summarize_graph_feature_mapping(graph_feature_mapping: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(graph_feature_mapping, dict):
        return {
            "enabled": False,
            "node_feature_count": 0,
            "node_scopes": [],
            "edge_weight_enabled": False,
            "edge_weight_metric": "none",
            "stats_quality_gate_enabled": False,
            "stats_quality_gate_on_fail": "none",
        }

    enabled = bool(graph_feature_mapping.get("enabled", False))
    node_raw = graph_feature_mapping.get("node_numeric", [])
    node_specs = [item for item in node_raw if isinstance(item, dict)] if isinstance(node_raw, list) else []
    node_scopes = sorted(
        {
            str(item.get("scope", "version")).strip() or "version"
            for item in node_specs
            if str(item.get("metric", "")).strip()
        }
    )

    edge_cfg = graph_feature_mapping.get("edge_weight", {})
    edge_weight_enabled = isinstance(edge_cfg, dict) and bool(str(edge_cfg.get("metric", "")).strip())
    edge_weight_metric = str(edge_cfg.get("metric", "")).strip() if isinstance(edge_cfg, dict) else ""

    quality_gate_cfg = graph_feature_mapping.get("stats_quality_gate", {})
    stats_quality_gate_enabled = bool(quality_gate_cfg.get("enabled", False)) if isinstance(quality_gate_cfg, dict) else False
    stats_quality_gate_on_fail = (
        str(quality_gate_cfg.get("on_fail", "ignore_with_warning")).strip()
        if isinstance(quality_gate_cfg, dict)
        else "ignore_with_warning"
    )

    return {
        "enabled": enabled,
        "node_feature_count": int(len(node_specs)),
        "node_scopes": node_scopes,
        "edge_weight_enabled": bool(edge_weight_enabled),
        "edge_weight_metric": edge_weight_metric or "none",
        "stats_quality_gate_enabled": bool(stats_quality_gate_enabled),
        "stats_quality_gate_on_fail": stats_quality_gate_on_fail or "ignore_with_warning",
    }


def prepare_data(config: Dict[str, Any], trace_adapter: IXESAdapter | None = None) -> Dict[str, Any]:
    """Prepare shared data artifacts for CLI and inspector without logic duplication."""
    logger = logging.getLogger(__name__)
    data_cfg = config.get("data", {})
    experiment_cfg = config.get("experiment", {})
    model_cfg = config.get("model", {})
    training_cfg = config.get("training", {})
    mapping_cfg_raw = config.get("mapping", {})
    adapter = trace_adapter or _build_trace_adapter(mapping_cfg_raw)
    adapter_kind = _resolve_trace_adapter_kind(mapping_cfg_raw)

    log_path = str(data_cfg.get("log_path", "")).strip()
    if adapter_kind == "xes" and not log_path:
        raise ValueError("Config must define data.log_path for XES adapter")
    if not log_path:
        log_path = "__adapter_source__"
    dataset_name = _resolve_dataset_name(data_cfg, log_path)
    mapping_cfg = _inject_dataset_name_mapping(mapping_cfg_raw, dataset_name)

    fraction = float(experiment_cfg.get("fraction", 1.0))
    split_strategy = str(experiment_cfg.get("split_strategy", "temporal")).strip().lower()
    if split_strategy == "time":
        split_strategy = "temporal"
    split_ratio = _parse_split_ratio(experiment_cfg)
    train_ratio = float(experiment_cfg.get("train_ratio", 0.7))
    mode = str(config.get("experiment", {}).get("mode", "train")).strip().lower()
    if mode == "eval_cross_dataset":
        split_ratio = (0.0, 0.0, 1.0)

    prefix_policy = PrefixPolicy()

    logger.info("Preparing data artifacts for mode=%s...", mode)
    logger.info("Reading events for preparation via adapter=%s...", adapter.__class__.__name__)
    traces = list(adapter.read(log_path, mapping_cfg))
    logger.info("Event read finished: traces=%d", len(traces))
    traces = _apply_cascade_prepare(
        traces,
        mode=mode,
        split_strategy=split_strategy,
        train_ratio=train_ratio,
        fraction=fraction,
    )
    logger.info("Applied cascade filter (mode=%s, fraction=%.4f) -> traces=%d", mode, fraction, len(traces))
    data_num_traces = len(traces)
    data_num_events = sum(len(trace.events) for trace in traces)

    feature_configs = parse_feature_configs(config)
    schema_resolver = SchemaResolver()
    policy_cfg = config.get("policies", {})
    logger.info("Fitting feature encoder from %d traces...", len(traces))
    feature_encoder = FeatureEncoder(
        feature_configs=feature_configs,
        traces=traces,
        state_dict=config.get("encoder_state"),
        schema_resolver=schema_resolver,
        policy_config=policy_cfg,
    )
    logger.info("Feature encoder ready.")
    feature_layout = feature_encoder.feature_layout

    activity_vocab, resource_vocab, activity_feature = _extract_base_vocabularies(feature_encoder)
    reverse_activity_vocab = {idx: key for key, idx in activity_vocab.items()}
    reverse_resource_vocab = {idx: key for key, idx in resource_vocab.items()}

    train_traces, val_traces, test_traces = _strict_temporal_split(traces, split_ratio, split_strategy)
    knowledge_repo = build_knowledge_graph_repository(config)
    knowledge_cfg = get_knowledge_graph_settings(config)
    available_versions = knowledge_repo.list_versions(process_name=dataset_name)
    if bool(knowledge_cfg.get("strict_load", False)):
        if not available_versions:
            raise ValueError(
                "knowledge_graph.strict_load=true but no topology artifacts were found for "
                f"process '{dataset_name}'. Run 'main.py ingest-topology --config ...' first."
            )
    elif not available_versions:
        logger.warning(
            "No topology artifacts found for process '%s'. Training will continue in baseline fallback mode. "
            "Run 'python main.py ingest-topology --config <experiment.yaml>' to build structure from BPMN, "
            "or set mapping.camunda_adapter.structure.structure_from_logs=true for explicit logs-based fallback ingestion.",
            dataset_name,
        )
    logger.info(
        "Knowledge graph backend=%s, strict_load=%s, available_versions=%s",
        str(knowledge_cfg.get("backend", "in_memory")),
        bool(knowledge_cfg.get("strict_load", False)),
        len(available_versions),
    )
    normalization_stats = _build_normalization_stats()
    graph_feature_mapping_raw = mapping_cfg.get("graph_feature_mapping", {})
    graph_feature_mapping = (
        dict(graph_feature_mapping_raw) if isinstance(graph_feature_mapping_raw, dict) else {}
    )
    feature_summary = _summarize_graph_feature_mapping(graph_feature_mapping)
    stats_time_policy = str(experiment_cfg.get("stats_time_policy", "latest")).strip().lower() or "latest"
    model_type = str(model_cfg.get("type", model_cfg.get("model_type", "unknown_model"))).strip() or "unknown_model"
    model_family = _resolve_model_family(model_type)
    xes_cfg = mapping_cfg.get("xes_adapter", {})
    xes_use_classifier = None
    xes_activity_key = ""
    xes_version_key = ""
    if isinstance(xes_cfg, dict):
        xes_use_classifier = bool(xes_cfg.get("use_classifier", True))
        xes_activity_key = str(xes_cfg.get("activity_key", "concept:name")).strip() or "concept:name"
        xes_version_key = str(xes_cfg.get("version_key", "concept:version")).strip() or "concept:version"
    run_profile = {
        "mode": mode,
        "adapter_kind": adapter_kind,
        "dataset_name": dataset_name,
        "model_type": model_type,
        "model_family": model_family,
        "graph_features_enabled": bool(feature_summary.get("enabled", False)),
        "node_feature_count": int(feature_summary.get("node_feature_count", 0)),
        "node_scopes": list(feature_summary.get("node_scopes", [])),
        "edge_weight_enabled": bool(feature_summary.get("edge_weight_enabled", False)),
        "edge_weight_metric": str(feature_summary.get("edge_weight_metric", "none")),
        "stats_quality_gate_enabled": bool(feature_summary.get("stats_quality_gate_enabled", False)),
        "stats_quality_gate_on_fail": str(feature_summary.get("stats_quality_gate_on_fail", "ignore_with_warning")),
        "global_process_stats_forward_enabled": False,
        "stats_time_policy": stats_time_policy,
        "knowledge_backend": str(knowledge_cfg.get("backend", "in_memory")),
        "knowledge_strict_load": bool(knowledge_cfg.get("strict_load", False)),
        "knowledge_versions_count": int(len(available_versions)),
        "xes_use_classifier": xes_use_classifier,
        "xes_activity_key": xes_activity_key or None,
        "xes_version_key": xes_version_key or None,
    }
    logger.info("========== RUN PROFILE ==========")
    logger.info(
        "RUN_PROFILE mode=%s model=%s model_family=%s adapter=%s dataset=%s stats_time_policy=%s",
        mode,
        model_type,
        model_family,
        adapter_kind,
        dataset_name,
        stats_time_policy,
    )
    logger.info(
        "RUN_PROFILE structure backend=%s strict_load=%s versions=%d graph_features=%s node_features=%d node_scopes=%s edge_weight=%s edge_metric=%s global_process_forward=%s",
        run_profile["knowledge_backend"],
        run_profile["knowledge_strict_load"],
        run_profile["knowledge_versions_count"],
        "on" if run_profile["graph_features_enabled"] else "off",
        run_profile["node_feature_count"],
        ",".join(str(item) for item in run_profile["node_scopes"]) or "none",
        "on" if run_profile["edge_weight_enabled"] else "off",
        run_profile["edge_weight_metric"],
        "on" if run_profile["global_process_stats_forward_enabled"] else "off",
    )
    if adapter_kind == "xes":
        logger.info(
            "RUN_PROFILE xes use_classifier=%s activity_key=%s version_key=%s",
            bool(xes_use_classifier),
            xes_activity_key or "concept:name",
            xes_version_key or "concept:version",
        )
    logger.info(
        "RUN_PROFILE checks alignment_guard=%s quality_guard=%s forward_stats_summary=%s",
        "manual",
        "on" if run_profile["stats_quality_gate_enabled"] else "off",
        "on",
    )
    logger.info("=================================")
    graph_builder = DynamicGraphBuilder(
        feature_encoder=feature_encoder,
        knowledge_port=knowledge_repo,
        process_name=dataset_name,
        graph_feature_mapping=graph_feature_mapping,
        stats_time_policy=stats_time_policy,
    )
    show_progress = bool(training_cfg.get("show_progress", True))
    tqdm_disable = bool(training_cfg.get("tqdm_disable", False))

    logger.info(
        "Building graph datasets: train_traces=%d, val_traces=%d, test_traces=%d",
        len(train_traces),
        len(val_traces),
        len(test_traces),
    )
    version_to_idx: Dict[str, int] = {}
    stats_snapshot_version_to_idx: Dict[str, int] = {}
    train_dataset = _build_graph_dataset(
        train_traces,
        prefix_policy,
        graph_builder,
        version_to_idx,
        stats_snapshot_version_to_idx,
        show_progress=show_progress,
        tqdm_disable=tqdm_disable,
        desc="Build train graphs",
    )
    val_dataset = _build_graph_dataset(
        val_traces,
        prefix_policy,
        graph_builder,
        version_to_idx,
        stats_snapshot_version_to_idx,
        show_progress=show_progress,
        tqdm_disable=tqdm_disable,
        desc="Build val graphs",
    )
    test_dataset = _build_graph_dataset(
        test_traces,
        prefix_policy,
        graph_builder,
        version_to_idx,
        stats_snapshot_version_to_idx,
        show_progress=show_progress,
        tqdm_disable=tqdm_disable,
        desc="Build test graphs",
    )
    idx_to_version = {idx: version for version, idx in version_to_idx.items()}
    idx_to_stats_snapshot_version = {
        idx: snapshot_version for snapshot_version, idx in stats_snapshot_version_to_idx.items()
    }
    logger.info(
        "Graph datasets ready: train_graphs=%d, val_graphs=%d, test_graphs=%d",
        len(train_dataset),
        len(val_dataset),
        len(test_dataset),
    )

    return {
        "log_path": log_path,
        "mapping_config": mapping_cfg,
        "data_config": {
            "dataset_name": dataset_name,
            "dataset_label": str(data_cfg.get("dataset_label", "unknown_data")),
        },
        "experiment_split_config": {
            "fraction": fraction,
            "split_strategy": split_strategy,
            "split_ratio": list(split_ratio),
            "train_ratio": train_ratio,
        },
        "data_metrics": {
            "data_num_traces": int(data_num_traces),
            "data_num_events": int(data_num_events),
            "vocab_activity_size": int(len(activity_vocab)),
            "vocab_resource_size": int(len(resource_vocab)),
        },
        "activity_vocab": activity_vocab,
        "resource_vocab": resource_vocab,
        "reverse_activity_vocab": reverse_activity_vocab,
        "reverse_resource_vocab": reverse_resource_vocab,
        "feature_configs": feature_configs,
        "feature_layout": feature_layout,
        "feature_encoder": feature_encoder,
        "activity_feature": activity_feature,
        "extra_vocabs": feature_encoder.categorical_vocabs,
        "ordered_extra_keys": [cfg.name for cfg in feature_configs],
        "normalization_stats": normalization_stats,
        "input_dim": feature_layout.num_dim,
        "graph_builder": graph_builder,
        "prefix_policy": prefix_policy,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
        "prepared_traces": list(traces),
        "train_traces": list(train_traces),
        "val_traces": list(val_traces),
        "test_traces": list(test_traces),
        "idx_to_version": idx_to_version,
        "version_to_idx": version_to_idx,
        "idx_to_stats_snapshot_version": idx_to_stats_snapshot_version,
        "stats_snapshot_version_to_idx": stats_snapshot_version_to_idx,
        "run_profile": run_profile,
    }


def _compute_class_weights(train_dataset: Sequence[Data], num_classes: int, device: torch.device) -> torch.Tensor:
    """Compute inverse-frequency class weights with 0.0 for absent classes."""
    counts = np.zeros(num_classes, dtype=np.float64)
    for sample in train_dataset:
        y_idx = int(sample.y.view(-1)[0].item())
        if 0 <= y_idx < num_classes:
            counts[y_idx] += 1.0

    total_samples = float(np.sum(counts))
    weights = np.zeros(num_classes, dtype=np.float32)
    if total_samples > 0.0:
        for idx in range(num_classes):
            if counts[idx] > 0.0:
                weights[idx] = float(total_samples / (num_classes * counts[idx]))
            else:
                weights[idx] = 0.0

    return torch.tensor(weights, dtype=torch.float32, device=device)


def main() -> None:
    """Parse CLI args, wire dependencies, and run model training."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Run MVP1 next-activity training pipeline.")
    parser.add_argument("--config", default="configs/experiments/experiment.yaml", help="YAML experiment config path or filename.")
    args = parser.parse_args()

    config_path = _resolve_config_path(args.config)
    config = load_yaml_config(args.config)

    seed = int(config.get("seed", 42))
    set_seed(seed)

    model_cfg = config.get("model", {})
    training_cfg = config.get("training", {})
    experiment_cfg = config.get("experiment", {})
    tracking_cfg = config.get("tracking", {})
    mode = str(experiment_cfg.get("mode", "train")).strip().lower()

    exp_name = str(experiment_cfg.get("name", "Run")).strip() or "Run"
    ds_label = str(config.get("data", {}).get("dataset_label", experiment_cfg.get("dataset", "unknown_data"))).strip() or "unknown_data"
    model_label_for_name = str(model_cfg.get("model_label", model_cfg.get("type", "unknown_model"))).strip() or "unknown_model"
    full_run_name = f"{exp_name}_{ds_label}_{model_label_for_name}"

    checkpoint_run_name = str(training_cfg.get("checkpoint_run_name", full_run_name)).strip() or full_run_name
    checkpoint_dir = str(training_cfg.get("checkpoint_dir", "checkpoints")).strip() or "checkpoints"
    checkpoint_override = str(training_cfg.get("checkpoint_path", "")).strip()
    eval_load_checkpoint = str(experiment_cfg.get("load_checkpoint", "")).strip()

    retrain = bool(training_cfg.get("retrain", False))
    resume_train_mode = (mode == "train") and (not retrain)

    if mode.startswith("eval_"):
        if not eval_load_checkpoint:
            raise ValueError("Р”Р»СЏ eval СЂРµР¶РёРјС–РІ РЅРµРѕР±С…С–РґРЅРѕ РІРєР°Р·Р°С‚Рё С€Р»СЏС… РґРѕ С‡РµРєРїРѕС–РЅС‚Сѓ РІ experiment.load_checkpoint")
        resolved_checkpoint_path = eval_load_checkpoint

    else:
        resolved_checkpoint_path = checkpoint_override or str(Path(checkpoint_dir) / f"{checkpoint_run_name}_best.pth")

    should_early_load_checkpoint = mode.startswith("eval_") or (resume_train_mode and Path(resolved_checkpoint_path).exists())

    if should_early_load_checkpoint:
        if mode.startswith("eval_") and not Path(resolved_checkpoint_path).exists():
            raise ValueError(f"Checkpoint required for mode '{mode}' was not found: {resolved_checkpoint_path}")

        checkpoint_payload = torch.load(resolved_checkpoint_path, map_location="cpu")
        if not isinstance(checkpoint_payload, dict):
            raise ValueError("Checkpoint payload must be a dictionary.")
        model_state_dict = checkpoint_payload.get("model_state_dict")
        if model_state_dict is None:
            raise ValueError("Checkpoint payload does not contain required key 'model_state_dict'.")
        encoder_state = checkpoint_payload.get("encoder_state")
        if encoder_state is None:
            raise ValueError(
                "Checkpoint does not contain 'encoder_state'. "
                "Cross-dataset/drift evaluation requires frozen encoder vocabularies."
            )
        config["encoder_state"] = encoder_state

    if mode == "eval_drift":
        drift_window_size = int(experiment_cfg.get("drift_window_size", 500))
        if drift_window_size <= 0:
            raise ValueError("experiment.drift_window_size must be a positive integer.")
        drift_window_sliding = int(experiment_cfg.get("drift_window_sliding", 0) or 0)
        if drift_window_sliding < 0:
            raise ValueError("experiment.drift_window_sliding must be a non-negative integer.")
        experiment_cfg = {
            **experiment_cfg,
            "drift_window_size": drift_window_size,
            "drift_window_sliding": drift_window_sliding,
        }
        config["experiment"] = experiment_cfg

    trace_adapter = _build_trace_adapter(config.get("mapping", {}))
    prepared = prepare_data(config, trace_adapter=trace_adapter)
    activity_vocab = prepared["activity_vocab"]
    resource_vocab = prepared["resource_vocab"]

    if not mode.startswith("eval_"):
        logger.info("Built vocabularies: activity_vocab=%d, resource_vocab=%d", len(activity_vocab), len(resource_vocab))

    # input_dim = Р±Р°Р·РѕРІС– one-hot + С‡Р°СЃРѕРІС– С„С–С‡С– + typed extra-С„С–С‡С– (СѓР·РіРѕРґР¶РµРЅРѕ Р· graph_builder).
    hidden_dim = int(model_cfg.get("hidden_dim", 64))
    output_dim = len(activity_vocab)
    dropout = float(model_cfg.get("dropout", 0.2))
    pooling_strategy = str(model_cfg.get("pooling_strategy", "global_mean")).strip().lower()
    feature_layout = prepared["feature_layout"]

    model_type = str(model_cfg.get("type", model_cfg.get("model_type", "BaselineGCN")))
    model = create_model(
        model_type=model_type,
        feature_layout=feature_layout,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        dropout=dropout,
        pooling_strategy=pooling_strategy,
        struct_encoder_type=str(model_cfg.get("struct_encoder_type", "GATv2Conv")),
        struct_hidden_dim=int(model_cfg.get("struct_hidden_dim", hidden_dim)),
        cross_attention_heads=int(model_cfg.get("cross_attention_heads", 4)),
    )
    logger.info("Initialized model via factory: type=%s", model_type)

    device = torch.device(str(training_cfg.get("device", "cpu")))
    class_weights = _compute_class_weights(prepared["train_dataset"], output_dim, device)

    md_label = model_label_for_name

    tracker = None
    if bool(tracking_cfg.get("enabled", False)):
        tracker = MLflowTracker(
            experiment_name=str(experiment_cfg.get("project", "DefaultExperiment")),
            run_name=full_run_name,
            tracking_uri=tracking_cfg.get("uri"),
        )
        mlflow_params = _build_mlflow_params(config, max_value_len=480)
        tracker.log_params(mlflow_params)
        logger.info("Logged MLflow params from selected blocks: %d entries.", len(mlflow_params))

    trainer_experiment_cfg = dict(experiment_cfg)
    trainer_experiment_cfg.update(prepared.get("experiment_split_config", {}))
    trainer_experiment_cfg["name"] = full_run_name

    trainer_config: Dict[str, Any] = {
        **training_cfg,
        "mapping_config": prepared["mapping_config"],
        "data_config": prepared["data_config"],
        "run_profile": prepared.get("run_profile", {}),
        "model_config": model_cfg,
        "experiment_config": trainer_experiment_cfg,
        "tracking_config": tracking_cfg,
        "config_path": str(config_path),
        "feature_configs": [
            {
                "name": item.name,
                "source_key": item.source_key,
                "encoding": list(item.encoding),
                "source": item.source,
                "dtype": item.dtype,
                "role": item.role,
            }
            for item in prepared["feature_configs"]
        ],
        "policy_config": config.get("policies", {}),
        "data_metrics": prepared["data_metrics"],
        "dataset_label": ds_label,
        "model_label": md_label,
        "feature_layout": {
            "num_cat_features": len(prepared["feature_layout"].cat_feature_names),
            "num_num_channels": int(prepared["feature_layout"].num_dim),
            "cat_feature_names": list(prepared["feature_layout"].cat_feature_names),
        },
        "seed": seed,
        "class_weight_cap": float(training_cfg.get("class_weight_cap", 50.0)),
        "retrain": retrain,
        "checkpoint_dir": checkpoint_dir,
        "checkpoint_path": resolved_checkpoint_path,
        "mode": mode,
        "drift_window_size": int(experiment_cfg.get("drift_window_size", 500)),
        "drift_window_sliding": int(experiment_cfg.get("drift_window_sliding", 0) or 0),
    }

    trainer = ModelTrainer(
        xes_adapter=trace_adapter,
        prefix_policy=PrefixPolicy(),
        graph_builder=prepared["graph_builder"],
        model=model,
        log_path=prepared["log_path"],
        config=trainer_config,
        tracker=tracker,
        class_weights=class_weights,
        prepared_data=prepared,
    )

    try:
        results = trainer.run()
        test_metrics = results.get("test_metrics", {})

        print("=== Final Test Metrics ===")
        for key, value in test_metrics.items():
            print(f"{key}: {value}")
    finally:
        if tracker is not None:
            tracker.close()


if __name__ == "__main__":
    main()

