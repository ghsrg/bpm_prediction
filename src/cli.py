"""Composition Root for MVP1 training pipeline."""

# Р’С–РґРїРѕРІС–РґРЅРѕ РґРѕ:
# - AGENT_GUIDE.MD -> СЂРѕР·РґС–Р» 2 (Clean Architecture, MVP1 scope) С– СЂРѕР·РґС–Р» 5 (РїРѕСЃР»С–РґРѕРІРЅС–СЃС‚СЊ РїРѕС‚РѕРєС–РІ)
# - ARCHITECTURE_RULES.MD -> СЂРѕР·РґС–Р» 2-4 (Application orchestration С‡РµСЂРµР· РїРѕСЂС‚Рё)
# - DATA_FLOWS_MVP1.MD -> СЂРѕР·РґС–Р» 3.1 (Training Pipeline)
# - EVF_MVP1.MD -> СЂРѕР·РґС–Р» 3 (Strict Temporal Split) С– СЂРѕР·РґС–Р» 5 (MLflow tracking)

from __future__ import annotations

import argparse
import gc
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import logging
import random
from time import monotonic
from typing import Any, Dict, Iterable, Iterator, List, Sequence, Tuple

import numpy as np
import psutil
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
from src.infrastructure.runtime.progress_events import emit_progress_event

GRAPH_DATASET_CACHE_SCHEMA = 2
GRAPH_DATASET_CACHE_FORMAT_LEGACY = "list_v1"
GRAPH_DATASET_CACHE_FORMAT_SHARDED = "sharded_v2"


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


def _configure_cpu_threading(training_cfg: Dict[str, Any], logger: logging.Logger) -> None:
    """Apply optional torch CPU threading overrides from training config."""
    raw_num_threads = training_cfg.get("torch_num_threads")
    if raw_num_threads is not None and str(raw_num_threads).strip() != "":
        num_threads = max(1, int(raw_num_threads))
        torch.set_num_threads(num_threads)

    raw_num_interop_threads = training_cfg.get("torch_num_interop_threads")
    if raw_num_interop_threads is not None and str(raw_num_interop_threads).strip() != "":
        num_interop_threads = max(1, int(raw_num_interop_threads))
        try:
            torch.set_num_interop_threads(num_interop_threads)
        except RuntimeError as exc:
            logger.warning(
                "Could not apply training.torch_num_interop_threads=%s: %s",
                raw_num_interop_threads,
                exc,
            )

    logger.info(
        "CPU_THREADING torch_num_threads=%d torch_num_interop_threads=%d",
        int(torch.get_num_threads()),
        int(torch.get_num_interop_threads()),
    )


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


def _safe_cache_token(raw: str) -> str:
    text = str(raw).strip()
    if not text:
        return "__auto__"
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text)
    return cleaned or "__auto__"


def _normalize_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _normalize_for_json(val) for key, val in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple, set)):
        return [_normalize_for_json(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _resolve_graph_dataset_cache_policy(experiment_cfg: Dict[str, Any]) -> str:
    raw = experiment_cfg.get("graph_dataset_cache_policy", experiment_cfg.get("graph_cache_policy", "off"))
    text = str(raw).strip().lower() or "off"
    if text in {"none", "false", "disabled"}:
        text = "off"
    if text in {"on", "true"}:
        text = "full"
    if text not in {"off", "read", "write", "full"}:
        return "off"
    return text


def _resolve_graph_dataset_cache_dir(experiment_cfg: Dict[str, Any]) -> str:
    raw = experiment_cfg.get("graph_dataset_cache_dir", experiment_cfg.get("graph_cache_dir", ".cache/graph_datasets"))
    text = str(raw).strip()
    return text or ".cache/graph_datasets"


def _trace_split_signature(traces: Sequence[RawTrace]) -> Dict[str, Any]:
    digest = hashlib.sha1()
    events_total = 0
    for trace in traces:
        case_id = str(getattr(trace, "case_id", "")).strip()
        version = str(getattr(trace, "process_version", "")).strip()
        events = list(getattr(trace, "events", []) or [])
        event_count = int(len(events))
        events_total += event_count
        first_ts = float(getattr(events[0], "timestamp", 0.0)) if event_count > 0 else 0.0
        last_ts = float(getattr(events[-1], "timestamp", 0.0)) if event_count > 0 else 0.0
        digest.update(f"{case_id}|{version}|{event_count}|{first_ts:.6f}|{last_ts:.6f};".encode("utf-8", errors="ignore"))
    return {
        "traces": int(len(traces)),
        "events": int(events_total),
        "digest": digest.hexdigest(),
    }


def _resolve_log_path_identity(log_path: str) -> str:
    text = str(log_path).strip()
    if not text or text == "__adapter_source__":
        return text or "__adapter_source__"
    path = Path(text).expanduser()
    try:
        return str(path.resolve())
    except OSError:
        return str(path)


def _graph_dataset_cache_fingerprint(
    *,
    config: Dict[str, Any],
    dataset_name: str,
    adapter_kind: str,
    mode: str,
    split_strategy: str,
    split_ratio: Tuple[float, float, float],
    train_ratio: float,
    fraction: float,
    stats_time_policy: str,
    on_missing_asof_snapshot: str,
    log_path: str,
    traces_all: Sequence[RawTrace],
    train_traces: Sequence[RawTrace],
    val_traces: Sequence[RawTrace],
    test_traces: Sequence[RawTrace],
    activity_vocab: Dict[str, int],
    resource_vocab: Dict[str, int],
    graph_feature_mapping: Dict[str, Any],
) -> str:
    mapping_cfg = config.get("mapping", {})
    features_cfg = config.get("features", None)
    if features_cfg is None:
        if isinstance(mapping_cfg, dict):
            features_cfg = mapping_cfg.get("features", [])
        else:
            features_cfg = []
    payload = {
        "schema": int(GRAPH_DATASET_CACHE_SCHEMA),
        "dataset_name": str(dataset_name),
        "adapter": str(adapter_kind),
        "mode": str(mode),
        "split_strategy": str(split_strategy),
        "split_ratio": [float(item) for item in split_ratio],
        "train_ratio": float(train_ratio),
        "fraction": float(fraction),
        "stats_time_policy": str(stats_time_policy),
        "on_missing_asof_snapshot": str(on_missing_asof_snapshot),
        "log_path": _resolve_log_path_identity(log_path),
        "traces_signature": {
            "all": _trace_split_signature(traces_all),
            "train": _trace_split_signature(train_traces),
            "validation": _trace_split_signature(val_traces),
            "test": _trace_split_signature(test_traces),
        },
        "activity_vocab": sorted((str(key), int(value)) for key, value in activity_vocab.items()),
        "resource_vocab": sorted((str(key), int(value)) for key, value in resource_vocab.items()),
        "graph_feature_mapping": graph_feature_mapping,
        "mapping": mapping_cfg if isinstance(mapping_cfg, dict) else {},
        "policies": config.get("policies", {}),
        "features": features_cfg,
        "model": config.get("model", {}),
    }
    encoded = json.dumps(_normalize_for_json(payload), ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(encoded.encode("utf-8", errors="ignore")).hexdigest()


def _graph_dataset_cache_entry_dir(cache_dir: str, dataset_name: str, fingerprint: str) -> Path:
    root = Path(cache_dir).expanduser().resolve()
    return root / _safe_cache_token(dataset_name) / str(fingerprint)


def _write_json_file(path: Path, payload: Dict[str, Any]) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(path)


def _read_json_file(path: Path) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _resolve_graph_dataset_shard_size(experiment_cfg: Dict[str, Any]) -> int:
    raw = experiment_cfg.get("graph_dataset_shard_size", 2000)
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = 2000
    return max(128, value)


def _resolve_graph_dataset_disk_spill_enabled(experiment_cfg: Dict[str, Any]) -> bool:
    raw = experiment_cfg.get("graph_dataset_disk_spill_enabled", False)
    if isinstance(raw, bool):
        return raw
    text = str(raw).strip().lower()
    return text in {"1", "true", "yes", "on"}


def _resolve_max_ram_bytes(experiment_cfg: Dict[str, Any]) -> int | None:
    raw = experiment_cfg.get("max_ram_gb", 0.0)
    try:
        max_ram_gb = float(raw)
    except (TypeError, ValueError):
        max_ram_gb = 0.0
    if max_ram_gb <= 0.0:
        return None
    return int(max_ram_gb * 1024.0 * 1024.0 * 1024.0)


def _process_rss_bytes() -> int:
    try:
        return int(psutil.Process().memory_info().rss)
    except Exception:
        return 0


def _dataset_payload_graph_count(payload: Any) -> int:
    if isinstance(payload, dict) and payload.get("kind") == "sharded_cache_split":
        return int(payload.get("graphs", 0))
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, dict)):
        return int(len(payload))
    return 0


def _iter_graphs_from_dataset_payload(payload: Any) -> Iterator[Data]:
    if isinstance(payload, dict) and payload.get("kind") == "sharded_cache_split":
        entry_dir_text = str(payload.get("entry_dir", "")).strip()
        if not entry_dir_text:
            return
        entry_dir = Path(entry_dir_text)
        shards = payload.get("shards", [])
        if not isinstance(shards, list):
            return
        for shard in shards:
            if not isinstance(shard, dict):
                continue
            rel_path = str(shard.get("path", "")).strip()
            if not rel_path:
                continue
            shard_path = entry_dir / rel_path
            if not shard_path.exists():
                continue
            try:
                loaded = torch.load(shard_path, map_location="cpu")
            except Exception:
                continue
            if not isinstance(loaded, list):
                continue
            for item in loaded:
                if isinstance(item, Data):
                    yield item
        return

    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, dict)):
        for item in payload:
            if isinstance(item, Data):
                yield item


def _load_graph_dataset_cache(
    *,
    cache_dir: str,
    dataset_name: str,
    fingerprint: str,
) -> Dict[str, Any] | None:
    entry_dir = _graph_dataset_cache_entry_dir(cache_dir, dataset_name, fingerprint)
    meta_path = entry_dir / "meta.json"
    meta = _read_json_file(meta_path)
    if meta is None:
        return None
    if int(meta.get("schema", -1)) != int(GRAPH_DATASET_CACHE_SCHEMA):
        return None
    if str(meta.get("fingerprint", "")).strip() != str(fingerprint):
        return None
    cache_format = str(meta.get("format", GRAPH_DATASET_CACHE_FORMAT_LEGACY)).strip() or GRAPH_DATASET_CACHE_FORMAT_LEGACY

    if cache_format == GRAPH_DATASET_CACHE_FORMAT_SHARDED:
        splits = meta.get("splits", {})
        if not isinstance(splits, dict):
            return None
        split_payloads: Dict[str, Dict[str, Any]] = {}
        for split_key in ("train", "validation", "test"):
            split_meta = splits.get(split_key, {})
            if not isinstance(split_meta, dict):
                return None
            shard_rows = split_meta.get("shards", [])
            if not isinstance(shard_rows, list):
                return None
            normalized_shards: List[Dict[str, Any]] = []
            graph_count = 0
            for shard in shard_rows:
                if not isinstance(shard, dict):
                    continue
                rel_path = str(shard.get("path", "")).strip()
                count = int(shard.get("count", 0))
                if not rel_path:
                    continue
                shard_path = entry_dir / rel_path
                if not shard_path.exists():
                    return None
                normalized_shards.append({"path": rel_path, "count": count})
                graph_count += max(0, count)
            split_payloads[split_key] = {
                "kind": "sharded_cache_split",
                "entry_dir": str(entry_dir),
                "split": split_key,
                "graphs": int(graph_count),
                "shards": normalized_shards,
            }
        version_to_idx_raw = meta.get("version_to_idx", {})
        snapshot_to_idx_raw = meta.get("stats_snapshot_version_to_idx", {})
        version_to_idx = {
            str(key): int(value)
            for key, value in version_to_idx_raw.items()
        } if isinstance(version_to_idx_raw, dict) else {}
        stats_snapshot_version_to_idx = {
            str(key): int(value)
            for key, value in snapshot_to_idx_raw.items()
        } if isinstance(snapshot_to_idx_raw, dict) else {}
        meta["last_access_utc"] = datetime.now(timezone.utc).isoformat()
        try:
            _write_json_file(meta_path, meta)
        except OSError:
            pass
        return {
            "meta": meta,
            "entry_dir": entry_dir,
            "train_dataset": split_payloads["train"],
            "val_dataset": split_payloads["validation"],
            "test_dataset": split_payloads["test"],
            "version_to_idx": version_to_idx,
            "stats_snapshot_version_to_idx": stats_snapshot_version_to_idx,
        }

    train_path = entry_dir / "train.pt"
    val_path = entry_dir / "validation.pt"
    test_path = entry_dir / "test.pt"
    if (not train_path.exists()) or (not val_path.exists()) or (not test_path.exists()):
        return None
    try:
        train_dataset = torch.load(train_path, map_location="cpu")
        val_dataset = torch.load(val_path, map_location="cpu")
        test_dataset = torch.load(test_path, map_location="cpu")
    except Exception:
        return None
    if not isinstance(train_dataset, list) or not isinstance(val_dataset, list) or not isinstance(test_dataset, list):
        return None
    version_to_idx_raw = meta.get("version_to_idx", {})
    snapshot_to_idx_raw = meta.get("stats_snapshot_version_to_idx", {})
    version_to_idx = {
        str(key): int(value)
        for key, value in version_to_idx_raw.items()
    } if isinstance(version_to_idx_raw, dict) else {}
    stats_snapshot_version_to_idx = {
        str(key): int(value)
        for key, value in snapshot_to_idx_raw.items()
    } if isinstance(snapshot_to_idx_raw, dict) else {}

    meta["last_access_utc"] = datetime.now(timezone.utc).isoformat()
    try:
        _write_json_file(meta_path, meta)
    except OSError:
        pass
    return {
        "meta": meta,
        "entry_dir": entry_dir,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
        "version_to_idx": version_to_idx,
        "stats_snapshot_version_to_idx": stats_snapshot_version_to_idx,
    }


def _save_graph_dataset_cache(
    *,
    cache_dir: str,
    dataset_name: str,
    fingerprint: str,
    train_dataset: Sequence[Data],
    val_dataset: Sequence[Data],
    test_dataset: Sequence[Data],
    version_to_idx: Dict[str, int],
    stats_snapshot_version_to_idx: Dict[str, int],
    payload_meta: Dict[str, Any] | None = None,
) -> Path:
    entry_dir = _graph_dataset_cache_entry_dir(cache_dir, dataset_name, fingerprint)
    entry_dir.mkdir(parents=True, exist_ok=True)

    def _save_tensor_list(path: Path, dataset: Sequence[Data]) -> None:
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        torch.save(list(dataset), tmp_path)
        tmp_path.replace(path)

    _save_tensor_list(entry_dir / "train.pt", train_dataset)
    _save_tensor_list(entry_dir / "validation.pt", val_dataset)
    _save_tensor_list(entry_dir / "test.pt", test_dataset)

    meta_payload = {
        "schema": int(GRAPH_DATASET_CACHE_SCHEMA),
        "format": GRAPH_DATASET_CACHE_FORMAT_LEGACY,
        "fingerprint": str(fingerprint),
        "dataset_name": str(dataset_name),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "last_access_utc": datetime.now(timezone.utc).isoformat(),
        "counts": {
            "train_graphs": int(len(train_dataset)),
            "validation_graphs": int(len(val_dataset)),
            "test_graphs": int(len(test_dataset)),
        },
        "version_to_idx": {str(key): int(value) for key, value in version_to_idx.items()},
        "stats_snapshot_version_to_idx": {
            str(key): int(value)
            for key, value in stats_snapshot_version_to_idx.items()
        },
        "payload_meta": payload_meta or {},
    }
    _write_json_file(entry_dir / "meta.json", meta_payload)
    return entry_dir


def _save_graph_dataset_cache_sharded(
    *,
    cache_dir: str,
    dataset_name: str,
    fingerprint: str,
    train_split: Dict[str, Any],
    val_split: Dict[str, Any],
    test_split: Dict[str, Any],
    version_to_idx: Dict[str, int],
    stats_snapshot_version_to_idx: Dict[str, int],
    payload_meta: Dict[str, Any] | None = None,
) -> Path:
    entry_dir = _graph_dataset_cache_entry_dir(cache_dir, dataset_name, fingerprint)
    entry_dir.mkdir(parents=True, exist_ok=True)
    meta_payload = {
        "schema": int(GRAPH_DATASET_CACHE_SCHEMA),
        "format": GRAPH_DATASET_CACHE_FORMAT_SHARDED,
        "fingerprint": str(fingerprint),
        "dataset_name": str(dataset_name),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "last_access_utc": datetime.now(timezone.utc).isoformat(),
        "counts": {
            "train_graphs": int(train_split.get("graphs", 0)),
            "validation_graphs": int(val_split.get("graphs", 0)),
            "test_graphs": int(test_split.get("graphs", 0)),
        },
        "splits": {
            "train": {
                "graphs": int(train_split.get("graphs", 0)),
                "shards": list(train_split.get("shards", [])),
            },
            "validation": {
                "graphs": int(val_split.get("graphs", 0)),
                "shards": list(val_split.get("shards", [])),
            },
            "test": {
                "graphs": int(test_split.get("graphs", 0)),
                "shards": list(test_split.get("shards", [])),
            },
        },
        "version_to_idx": {str(key): int(value) for key, value in version_to_idx.items()},
        "stats_snapshot_version_to_idx": {
            str(key): int(value)
            for key, value in stats_snapshot_version_to_idx.items()
        },
        "payload_meta": payload_meta or {},
    }
    _write_json_file(entry_dir / "meta.json", meta_payload)
    return entry_dir


def _build_graph_dataset_sharded(
    traces: Sequence[RawTrace],
    prefix_policy: PrefixPolicy,
    graph_builder: DynamicGraphBuilder,
    version_to_idx: Dict[str, int],
    stats_snapshot_version_to_idx: Dict[str, int],
    *,
    show_progress: bool,
    tqdm_disable: bool,
    desc: str,
    progress_stage: str,
    entry_dir: Path,
    split_key: str,
    shard_size: int,
    max_ram_bytes: int | None,
) -> Dict[str, Any]:
    """Build graph dataset and spill buffered Data objects into on-disk shards."""
    split_dir = entry_dir / f"{split_key}_shards"
    split_dir.mkdir(parents=True, exist_ok=True)
    dataset_payload = {
        "kind": "sharded_cache_split",
        "entry_dir": str(entry_dir),
        "split": split_key,
        "graphs": 0,
        "shards": [],
    }

    total_traces = int(len(traces))
    emit_progress_event(stage=progress_stage, status="start", message=desc, current=0, total=total_traces)
    iterator = tqdm(
        traces,
        desc=desc,
        unit="trace",
        leave=False,
        disable=(not show_progress) or tqdm_disable,
    )

    buffer: List[Data] = []
    shard_rows: List[Dict[str, Any]] = []
    shard_idx = 0
    total_graphs = 0
    last_progress_emit = 0.0
    last_ram_check = 0

    def _flush_buffer(force: bool = False) -> None:
        nonlocal shard_idx
        if not buffer:
            return
        if not force and len(buffer) < shard_size:
            return
        shard_idx += 1
        shard_name = f"{split_key}_{shard_idx:05d}.pt"
        shard_path = split_dir / shard_name
        tmp_path = shard_path.with_suffix(".pt.tmp")
        torch.save(buffer, tmp_path)
        tmp_path.replace(shard_path)
        shard_rows.append(
            {
                "path": str(shard_path.relative_to(entry_dir).as_posix()),
                "count": int(len(buffer)),
            }
        )
        buffer.clear()
        gc.collect()

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
            stats_missing_asof_snapshot = contract.get("stats_missing_asof_snapshot")
            if isinstance(stats_missing_asof_snapshot, bool):
                payload["stats_missing_asof_snapshot"] = torch.tensor(
                    [1 if stats_missing_asof_snapshot else 0],
                    dtype=torch.long,
                )
            snapshot_seq_raw = contract.get("stats_snapshot_version_seq")
            if isinstance(snapshot_seq_raw, (int, float)) and not isinstance(snapshot_seq_raw, bool):
                snapshot_idx = int(snapshot_seq_raw)
                snapshot_version = f"k{snapshot_idx:06d}"
                stats_snapshot_version_to_idx.setdefault(snapshot_version, snapshot_idx)
                payload["stats_snapshot_version_idx"] = torch.tensor([snapshot_idx], dtype=torch.long)
            else:
                stats_snapshot_version = contract.get("stats_snapshot_version")
                if stats_snapshot_version is not None:
                    snapshot_version = str(stats_snapshot_version).strip()
                    if snapshot_version:
                        snapshot_idx = stats_snapshot_version_to_idx.setdefault(
                            snapshot_version,
                            len(stats_snapshot_version_to_idx),
                        )
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

            buffer.append(Data(**payload))
            total_graphs += 1
            if len(buffer) >= shard_size:
                _flush_buffer(force=True)
            else:
                last_ram_check += 1
                if max_ram_bytes is not None and last_ram_check >= 128:
                    last_ram_check = 0
                    rss_bytes = _process_rss_bytes()
                    if rss_bytes >= max_ram_bytes:
                        _flush_buffer(force=True)

        now = monotonic()
        if idx == 1 or idx == total_traces or (now - last_progress_emit) >= 0.8:
            emit_progress_event(
                stage=progress_stage,
                status="update",
                message=desc,
                current=idx,
                total=total_traces,
                payload={"graphs": int(total_graphs), "shards": int(len(shard_rows))},
            )
            last_progress_emit = now
        if idx % 1000 == 0:
            iterator.set_postfix({"graphs": total_graphs, "shards": len(shard_rows)})

    _flush_buffer(force=True)
    dataset_payload["graphs"] = int(total_graphs)
    dataset_payload["shards"] = shard_rows
    emit_progress_event(
        stage=progress_stage,
        status="done",
        message=f"{desc} completed",
        current=total_traces,
        total=total_traces,
        payload={"graphs": int(total_graphs), "shards": int(len(shard_rows))},
    )
    return dataset_payload


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
    progress_stage: str | None = None,
) -> List[Data]:
    """Convert traces into a list of PyG Data graphs via prefix slicing + graph builder."""
    dataset: List[Data] = []
    total_traces = int(len(traces))
    stage = str(progress_stage or "").strip()
    if stage:
        emit_progress_event(stage=stage, status="start", message=desc, current=0, total=total_traces)
    iterator = tqdm(
        traces,
        desc=desc,
        unit="trace",
        leave=False,
        disable=(not show_progress) or tqdm_disable,
    )
    last_progress_emit = 0.0
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
            stats_missing_asof_snapshot = contract.get("stats_missing_asof_snapshot")
            if isinstance(stats_missing_asof_snapshot, bool):
                payload["stats_missing_asof_snapshot"] = torch.tensor(
                    [1 if stats_missing_asof_snapshot else 0],
                    dtype=torch.long,
                )
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
        if stage:
            now = monotonic()
            if idx == 1 or idx == total_traces or (now - last_progress_emit) >= 0.8:
                emit_progress_event(
                    stage=stage,
                    status="update",
                    message=desc,
                    current=idx,
                    total=total_traces,
                    payload={"graphs": int(len(dataset))},
                )
                last_progress_emit = now
        if idx % 1000 == 0:
            iterator.set_postfix({"graphs": len(dataset)})
    if stage:
        emit_progress_event(
            stage=stage,
            status="done",
            message=f"{desc} completed",
            current=total_traces,
            total=total_traces,
            payload={"graphs": int(len(dataset))},
        )
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
    emit_progress_event(stage="prepare_data", status="start", message="Preparing data artifacts")
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
    emit_progress_event(stage="prepare.read_events", status="start", message=f"Reading events via {adapter.__class__.__name__}")
    logger.info("Reading events for preparation via adapter=%s...", adapter.__class__.__name__)
    traces = list(adapter.read(log_path, mapping_cfg))
    logger.info("Event read finished: traces=%d", len(traces))
    emit_progress_event(
        stage="prepare.read_events",
        status="done",
        message="Event read finished",
        current=int(len(traces)),
        total=int(len(traces)),
    )
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
    emit_progress_event(
        stage="prepare.feature_encoder",
        status="start",
        message=f"Fitting feature encoder from {len(traces)} traces",
    )
    logger.info("Fitting feature encoder from %d traces...", len(traces))
    feature_encoder = FeatureEncoder(
        feature_configs=feature_configs,
        traces=traces,
        state_dict=config.get("encoder_state"),
        schema_resolver=schema_resolver,
        policy_config=policy_cfg,
    )
    logger.info("Feature encoder ready.")
    emit_progress_event(stage="prepare.feature_encoder", status="done", message="Feature encoder ready")
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
    on_missing_asof_snapshot = str(
        experiment_cfg.get("on_missing_asof_snapshot", "disable_stats")
    ).strip().lower() or "disable_stats"
    if on_missing_asof_snapshot not in {"disable_stats", "use_base", "raise"}:
        on_missing_asof_snapshot = "disable_stats"
    cache_policy = str(experiment_cfg.get("cache_policy", "full")).strip().lower() or "full"
    if cache_policy in {"none", "disabled", "false"}:
        cache_policy = "off"
    if cache_policy not in {"off", "dto", "full"}:
        cache_policy = "full"
    cache_dir = str(experiment_cfg.get("cache_dir", "")).strip()
    graph_dataset_cache_policy = _resolve_graph_dataset_cache_policy(experiment_cfg)
    graph_dataset_cache_dir = _resolve_graph_dataset_cache_dir(experiment_cfg)
    graph_dataset_disk_spill_enabled = _resolve_graph_dataset_disk_spill_enabled(experiment_cfg)
    graph_dataset_shard_size = _resolve_graph_dataset_shard_size(experiment_cfg)
    max_ram_bytes = _resolve_max_ram_bytes(experiment_cfg)
    graph_dataset_cache_read = graph_dataset_cache_policy in {"read", "full"}
    graph_dataset_cache_write = graph_dataset_cache_policy in {"write", "full"}
    if graph_dataset_disk_spill_enabled and not graph_dataset_cache_write:
        logger.warning(
            "graph_dataset_disk_spill_enabled=true requires cache write mode. Forcing graph_dataset_cache_policy=write for this run."
        )
        graph_dataset_cache_write = True
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
        "on_missing_asof_snapshot": on_missing_asof_snapshot,
        "cache_policy": cache_policy,
        "cache_dir": cache_dir or ".cache/dynamic_graph_builder",
        "graph_dataset_cache_policy": graph_dataset_cache_policy,
        "graph_dataset_cache_dir": graph_dataset_cache_dir,
        "graph_dataset_disk_spill_enabled": bool(graph_dataset_disk_spill_enabled),
        "graph_dataset_shard_size": int(graph_dataset_shard_size),
        "max_ram_gb": float(experiment_cfg.get("max_ram_gb", 0.0) or 0.0),
        "knowledge_backend": str(knowledge_cfg.get("backend", "in_memory")),
        "knowledge_strict_load": bool(knowledge_cfg.get("strict_load", False)),
        "knowledge_versions_count": int(len(available_versions)),
        "xes_use_classifier": xes_use_classifier,
        "xes_activity_key": xes_activity_key or None,
        "xes_version_key": xes_version_key or None,
        "dataloader_num_workers": max(0, int(training_cfg.get("dataloader_num_workers", 0))),
        "dataloader_pin_memory": bool(training_cfg.get("dataloader_pin_memory", False)),
    }
    logger.info("========== RUN PROFILE ==========")
    logger.info(
        "RUN_PROFILE mode=%s model=%s model_family=%s adapter=%s dataset=%s stats_time_policy=%s on_missing_asof_snapshot=%s cache_policy=%s cache_dir=%s graph_cache_policy=%s graph_cache_dir=%s spill=%s shard_size=%d max_ram_gb=%.2f",
        mode,
        model_type,
        model_family,
        adapter_kind,
        dataset_name,
        stats_time_policy,
        on_missing_asof_snapshot,
        cache_policy,
        run_profile["cache_dir"],
        graph_dataset_cache_policy,
        graph_dataset_cache_dir,
        bool(graph_dataset_disk_spill_enabled),
        int(graph_dataset_shard_size),
        float(run_profile["max_ram_gb"]),
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
        "RUN_PROFILE checks alignment_guard=%s quality_guard=%s forward_stats_summary=%s cache_policy=%s",
        "manual",
        "on" if run_profile["stats_quality_gate_enabled"] else "off",
        "on",
        cache_policy,
    )
    logger.info(
        "RUN_PROFILE runtime dataloader_workers=%d pin_memory=%s",
        run_profile["dataloader_num_workers"],
        run_profile["dataloader_pin_memory"],
    )
    logger.info("=================================")
    graph_builder = DynamicGraphBuilder(
        feature_encoder=feature_encoder,
        knowledge_port=knowledge_repo,
        process_name=dataset_name,
        graph_feature_mapping=graph_feature_mapping,
        stats_time_policy=stats_time_policy,
        on_missing_asof_snapshot=on_missing_asof_snapshot,
        cache_policy=cache_policy,
        cache_dir=cache_dir,
    )
    show_progress = bool(training_cfg.get("show_progress", True))
    tqdm_disable = bool(training_cfg.get("tqdm_disable", False))

    logger.info(
        "Building graph datasets: train_traces=%d, val_traces=%d, test_traces=%d",
        len(train_traces),
        len(val_traces),
        len(test_traces),
    )
    cache_fingerprint = _graph_dataset_cache_fingerprint(
        config=config,
        dataset_name=dataset_name,
        adapter_kind=adapter_kind,
        mode=mode,
        split_strategy=split_strategy,
        split_ratio=split_ratio,
        train_ratio=train_ratio,
        fraction=fraction,
        stats_time_policy=stats_time_policy,
        on_missing_asof_snapshot=on_missing_asof_snapshot,
        log_path=log_path,
        traces_all=traces,
        train_traces=train_traces,
        val_traces=val_traces,
        test_traces=test_traces,
        activity_vocab=activity_vocab,
        resource_vocab=resource_vocab,
        graph_feature_mapping=graph_feature_mapping,
    )
    cache_hit = False
    version_to_idx: Dict[str, int] = {}
    stats_snapshot_version_to_idx: Dict[str, int] = {}
    train_dataset: Any
    val_dataset: Any
    test_dataset: Any
    cache_bundle = None
    if graph_dataset_cache_read:
        cache_bundle = _load_graph_dataset_cache(
            cache_dir=graph_dataset_cache_dir,
            dataset_name=dataset_name,
            fingerprint=cache_fingerprint,
        )
    if cache_bundle is not None:
        cache_hit = True
        train_dataset = cache_bundle["train_dataset"]
        val_dataset = cache_bundle["val_dataset"]
        test_dataset = cache_bundle["test_dataset"]
        version_to_idx = dict(cache_bundle["version_to_idx"])
        stats_snapshot_version_to_idx = dict(cache_bundle["stats_snapshot_version_to_idx"])
        logger.info(
            "Graph dataset cache hit: policy=%s dir=%s key=%s train=%d val=%d test=%d",
            graph_dataset_cache_policy,
            graph_dataset_cache_dir,
            cache_fingerprint[:12],
            _dataset_payload_graph_count(train_dataset),
            _dataset_payload_graph_count(val_dataset),
            _dataset_payload_graph_count(test_dataset),
        )
        emit_progress_event(
            stage="build_graph.train",
            status="done",
            message="Train graphs loaded from disk cache",
            current=int(len(train_traces)),
            total=int(len(train_traces)),
            payload={"graphs": int(_dataset_payload_graph_count(train_dataset)), "from_cache": True},
        )
        emit_progress_event(
            stage="build_graph.validation",
            status="done",
            message="Validation graphs loaded from disk cache",
            current=int(len(val_traces)),
            total=int(len(val_traces)),
            payload={"graphs": int(_dataset_payload_graph_count(val_dataset)), "from_cache": True},
        )
        emit_progress_event(
            stage="build_graph.test",
            status="done",
            message="Test graphs loaded from disk cache",
            current=int(len(test_traces)),
            total=int(len(test_traces)),
            payload={"graphs": int(_dataset_payload_graph_count(test_dataset)), "from_cache": True},
        )
    else:
        version_to_idx = {}
        stats_snapshot_version_to_idx = {}
        if graph_dataset_disk_spill_enabled:
            entry_dir = _graph_dataset_cache_entry_dir(
                graph_dataset_cache_dir,
                dataset_name,
                cache_fingerprint,
            )
            train_dataset = _build_graph_dataset_sharded(
                train_traces,
                prefix_policy,
                graph_builder,
                version_to_idx,
                stats_snapshot_version_to_idx,
                show_progress=show_progress,
                tqdm_disable=tqdm_disable,
                desc="Build train graphs",
                progress_stage="build_graph.train",
                entry_dir=entry_dir,
                split_key="train",
                shard_size=graph_dataset_shard_size,
                max_ram_bytes=max_ram_bytes,
            )
            val_dataset = _build_graph_dataset_sharded(
                val_traces,
                prefix_policy,
                graph_builder,
                version_to_idx,
                stats_snapshot_version_to_idx,
                show_progress=show_progress,
                tqdm_disable=tqdm_disable,
                desc="Build val graphs",
                progress_stage="build_graph.validation",
                entry_dir=entry_dir,
                split_key="validation",
                shard_size=graph_dataset_shard_size,
                max_ram_bytes=max_ram_bytes,
            )
            test_dataset = _build_graph_dataset_sharded(
                test_traces,
                prefix_policy,
                graph_builder,
                version_to_idx,
                stats_snapshot_version_to_idx,
                show_progress=show_progress,
                tqdm_disable=tqdm_disable,
                desc="Build test graphs",
                progress_stage="build_graph.test",
                entry_dir=entry_dir,
                split_key="test",
                shard_size=graph_dataset_shard_size,
                max_ram_bytes=max_ram_bytes,
            )
            if graph_dataset_cache_write:
                try:
                    saved_dir = _save_graph_dataset_cache_sharded(
                        cache_dir=graph_dataset_cache_dir,
                        dataset_name=dataset_name,
                        fingerprint=cache_fingerprint,
                        train_split=train_dataset,
                        val_split=val_dataset,
                        test_split=test_dataset,
                        version_to_idx=version_to_idx,
                        stats_snapshot_version_to_idx=stats_snapshot_version_to_idx,
                        payload_meta={
                            "mode": mode,
                            "adapter": adapter_kind,
                            "split_strategy": split_strategy,
                            "fraction": float(fraction),
                            "train_ratio": float(train_ratio),
                            "split_ratio": [float(item) for item in split_ratio],
                            "disk_spill_enabled": True,
                            "shard_size": int(graph_dataset_shard_size),
                            "max_ram_bytes": int(max_ram_bytes or 0),
                        },
                    )
                    logger.info(
                        "Graph dataset cache saved (sharded): dir=%s key=%s",
                        saved_dir,
                        cache_fingerprint[:12],
                    )
                except Exception as exc:
                    logger.warning(
                        "Graph dataset cache save failed (sharded, dir=%s key=%s): %s",
                        graph_dataset_cache_dir,
                        cache_fingerprint[:12],
                        exc,
                    )
        else:
            train_dataset = _build_graph_dataset(
                train_traces,
                prefix_policy,
                graph_builder,
                version_to_idx,
                stats_snapshot_version_to_idx,
                show_progress=show_progress,
                tqdm_disable=tqdm_disable,
                desc="Build train graphs",
                progress_stage="build_graph.train",
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
                progress_stage="build_graph.validation",
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
                progress_stage="build_graph.test",
            )
            if graph_dataset_cache_write:
                try:
                    saved_dir = _save_graph_dataset_cache(
                        cache_dir=graph_dataset_cache_dir,
                        dataset_name=dataset_name,
                        fingerprint=cache_fingerprint,
                        train_dataset=train_dataset,
                        val_dataset=val_dataset,
                        test_dataset=test_dataset,
                        version_to_idx=version_to_idx,
                        stats_snapshot_version_to_idx=stats_snapshot_version_to_idx,
                        payload_meta={
                            "mode": mode,
                            "adapter": adapter_kind,
                            "split_strategy": split_strategy,
                            "fraction": float(fraction),
                            "train_ratio": float(train_ratio),
                            "split_ratio": [float(item) for item in split_ratio],
                        },
                    )
                    logger.info(
                        "Graph dataset cache saved: dir=%s key=%s",
                        saved_dir,
                        cache_fingerprint[:12],
                    )
                except Exception as exc:
                    logger.warning(
                        "Graph dataset cache save failed (dir=%s key=%s): %s",
                        graph_dataset_cache_dir,
                        cache_fingerprint[:12],
                        exc,
                    )
    idx_to_version = {idx: version for version, idx in version_to_idx.items()}
    idx_to_stats_snapshot_version = {
        idx: snapshot_version for snapshot_version, idx in stats_snapshot_version_to_idx.items()
    }
    run_profile["graph_dataset_cache_hit"] = bool(cache_hit)
    run_profile["graph_dataset_cache_key"] = cache_fingerprint[:12]
    logger.info(
        "Graph datasets ready: train_graphs=%d, val_graphs=%d, test_graphs=%d",
        _dataset_payload_graph_count(train_dataset),
        _dataset_payload_graph_count(val_dataset),
        _dataset_payload_graph_count(test_dataset),
    )
    emit_progress_event(
        stage="prepare_data",
        status="done",
        message="Data preparation completed",
        payload={
            "train_graphs": int(_dataset_payload_graph_count(train_dataset)),
            "validation_graphs": int(_dataset_payload_graph_count(val_dataset)),
            "test_graphs": int(_dataset_payload_graph_count(test_dataset)),
        },
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
        "graph_dataset_cache_hit": bool(cache_hit),
        "graph_dataset_cache_fingerprint": cache_fingerprint,
    }


def _compute_class_weights(train_dataset: Any, num_classes: int, device: torch.device) -> torch.Tensor:
    """Compute inverse-frequency class weights with 0.0 for absent classes."""
    counts = np.zeros(num_classes, dtype=np.float64)
    for sample in _iter_graphs_from_dataset_payload(train_dataset):
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
    _configure_cpu_threading(training_cfg, logger)
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

