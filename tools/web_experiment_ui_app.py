"""Streamlit single-page web UI for BPM experiment configuration and execution."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
import html
import json
import queue
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml
import streamlit as st
import streamlit.components.v1 as components

try:
    import psutil
except Exception:  # pragma: no cover - optional in runtime
    psutil = None

try:
    from streamlit_ace import st_ace
except Exception:  # pragma: no cover - optional in runtime
    st_ace = None

try:
    import mlflow
    from mlflow.entities import ViewType
    from mlflow.tracking import MlflowClient
except Exception:  # pragma: no cover - optional in runtime
    mlflow = None
    ViewType = None
    MlflowClient = None


ROOT_DIR = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT_DIR / "configs" / "ui" / "config_catalog.yaml"
PROGRESS_EVENT_PREFIX = "__BPM_PROGRESS__"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.infrastructure.config.yaml_loader import load_yaml_with_includes


RUN_STAGE_WEIGHTS: Dict[str, float] = {
    "run.pipeline": 0.02,
    "prepare_data": 0.10,
    "prepare.read_events": 0.08,
    "prepare.feature_encoder": 0.07,
    "build_graph.train": 0.22,
    "build_graph.validation": 0.10,
    "build_graph.test": 0.08,
    "trainer.dataloaders": 0.03,
    "trainer.dry_run": 0.05,
    "train.epochs": 0.15,
    "train.batches": 0.04,
    "validation.batches": 0.03,
    "test.eval": 0.03,
    "test.batches": 0.02,
    "eval_drift.windows": 0.04,
}

RUN_STAGE_ORDER: List[str] = [
    "run.pipeline",
    "prepare_data",
    "prepare.read_events",
    "prepare.feature_encoder",
    "build_graph.train",
    "build_graph.validation",
    "build_graph.test",
    "trainer.dataloaders",
    "trainer.dry_run",
    "train.epochs",
    "train.batches",
    "validation.batches",
    "test.eval",
    "test.batches",
    "eval_drift.windows",
]

STAGE_TITLE: Dict[str, str] = {
    "run.pipeline": "Pipeline",
    "prepare_data": "Prepare data",
    "prepare.read_events": "Read events",
    "prepare.feature_encoder": "Build feature encoder",
    "build_graph.train": "Build train graphs",
    "build_graph.validation": "Build validation graphs",
    "build_graph.test": "Build test graphs",
    "trainer.dataloaders": "Build dataloaders",
    "trainer.dry_run": "Dry run",
    "train.epochs": "Train epochs",
    "train.batches": "Train batches",
    "validation.batches": "Validation batches",
    "test.eval": "Test eval",
    "test.batches": "Test batches",
    "eval_drift.windows": "Drift windows",
}

CORE_FIELDS: List[str] = [
    "experiment.mode",
    "experiment.project",
    "experiment.name",
    "data.dataset_name",
    "data.dataset_label",
    "model.model_label",
    "model.type",
    "seed",
    "mapping.adapter",
    "experiment.fraction",
    "experiment.train_ratio",
    "experiment.split_ratio",
    "experiment.split_strategy",
    "experiment.stats_time_policy",
    "experiment.on_missing_asof_snapshot",
    "experiment.load_checkpoint",
    "training.checkpoint_path",
    "training.checkpoint_run_name",
    "training.checkpoint_dir",
    "experiment.retrain",
    "mapping.graph_feature_mapping.enabled",
]

STRICT_ENUM_PATHS = {
    "experiment.mode",
    "mapping.adapter",
    "experiment.cache_policy",
    "experiment.graph_dataset_cache_policy",
    "experiment.stats_time_policy",
    "experiment.on_missing_asof_snapshot",
    "experiment.split_strategy",
    "mapping.knowledge_graph.backend",
    "model.type",
    "model.struct_encoder_type",
    "model.fusion_mode",
    "tracking.backend",
}

EXCLUDED_PREFIXES = (
    "mapping.features",
    "features",
    "policies",
)

PATH_PICKER_GLOBS: Dict[str, List[str]] = {
    "experiment.load_checkpoint": ["checkpoints/**/*.pth"],
    "training.checkpoint_path": ["checkpoints/**/*.pth"],
    "data.log_path": ["data/**/*.xes", "Data/**/*.xes"],
}

SPECIAL_SELECT_CHOICES: Dict[str, List[tuple[str, str]]] = {
    "experiment.mode": [
        ("train", "train"),
        ("eval_drift", "eval_drift"),
        ("eval_cross_dataset", "eval_cross_dataset"),
        ("sync-topology", "sync-topology"),
        ("sync-stats", "sync-stats"),
        ("sync-stats-backfill", "sync-stats-backfill"),
    ],
    "mapping.adapter": [
        ("camunda", "camunda"),
        ("xes", "xes"),
    ],
    "experiment.stats_time_policy": [
        ("strict_asof", "strict_asof"),
        ("latest", "latest"),
    ],
    "experiment.on_missing_asof_snapshot": [
        ("raise", "raise"),
        ("disable_stats", "disable_stats"),
        ("use_base", "use_base"),
    ],
    "experiment.split_strategy": [
        ("temporal", "temporal"),
    ],
    "experiment.cache_policy": [
        ("full", "full"),
        ("dto", "dto"),
        ("off", "off"),
    ],
    "experiment.graph_dataset_cache_policy": [
        ("off", "off"),
        ("read", "read"),
        ("write", "write"),
        ("full", "full"),
    ],
    "mapping.knowledge_graph.backend": [
        ("in_memory", "in_memory"),
        ("file", "file"),
        ("neo4j", "neo4j"),
    ],
    "tracking.backend": [
        ("mlflow", "mlflow"),
    ],
    "model.type": [
        ("BaselineGATv2", "BaselineGATv2 (Baseline)"),
        ("BaselineGCN", "BaselineGCN (Baseline)"),
        ("EOPKGGATv2", "EOPKGGATv2 (EOPKG)"),
    ],
    "model.struct_encoder_type": [
        ("GATv2Conv", "GATv2Conv"),
        ("GCNConv", "GCNConv"),
    ],
    "model.fusion_mode": [
        ("Attention", "Attention"),
        ("Concat", "Concat"),
    ],
    "mapping.graph_feature_mapping.topology_projection.gateway_mode": [
        ("preserve", "preserve"),
        ("collapse_for_prediction", "collapse_for_prediction"),
    ],
}


@dataclass
class CatalogMeta:
    path: str
    label: str
    description: str
    affects: str
    default: Any
    enum: List[str]
    required_in_modes: List[str]
    required_when: Dict[str, Any]
    ui_tab: str
    ui_group: str
    ui_priority: int
    ui_order: int


def _deep_get(payload: Dict[str, Any], dotted_path: str, default: Any = None) -> Any:
    cursor: Any = payload
    for key in dotted_path.split("."):
        if not isinstance(cursor, dict):
            return default
        if key not in cursor:
            return default
        cursor = cursor[key]
    return cursor


def _deep_has(payload: Dict[str, Any], dotted_path: str) -> bool:
    cursor: Any = payload
    for key in dotted_path.split("."):
        if not isinstance(cursor, dict) or key not in cursor:
            return False
        cursor = cursor[key]
    return True


def _deep_set(payload: Dict[str, Any], dotted_path: str, value: Any) -> None:
    parts = [part for part in dotted_path.split(".") if part]
    if not parts:
        return
    cursor = payload
    for key in parts[:-1]:
        next_obj = cursor.get(key)
        if not isinstance(next_obj, dict):
            next_obj = {}
            cursor[key] = next_obj
        cursor = next_obj
    cursor[parts[-1]] = value


def _flatten(payload: Any, prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not isinstance(payload, dict):
        return out
    for key, value in payload.items():
        full = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            out.update(_flatten(value, full))
        else:
            out[full] = value
    return out


def _safe_yaml_dump(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return yaml.safe_dump(value, allow_unicode=True, sort_keys=False).strip()
    if value is None:
        return ""
    return str(value)


def _parse_yaml_block(text: str, field_name: str) -> tuple[Any, str | None]:
    raw = str(text or "").strip()
    if raw == "":
        return None, None
    try:
        return yaml.safe_load(raw), None
    except yaml.YAMLError as exc:
        return None, f"Invalid YAML in {field_name}: {exc}"


def _normalize_bool(value: Any, fallback: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return fallback


def _coerce_from_text(text: str, expected_type: str) -> Any:
    raw = str(text)
    stripped = raw.strip()
    if expected_type == "bool":
        return _normalize_bool(stripped, fallback=False)
    if stripped == "":
        return ""
    if expected_type == "int":
        try:
            return int(float(stripped))
        except ValueError:
            return raw
    if expected_type == "float":
        try:
            return float(stripped)
        except ValueError:
            return raw
    if expected_type in {"list", "dict"}:
        try:
            return yaml.safe_load(stripped)
        except yaml.YAMLError:
            return raw
    return raw


def _resolve_config_path(raw_path: str) -> Path:
    path = Path(str(raw_path).strip())
    if not path.is_absolute():
        path = (ROOT_DIR / path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    return path


def _load_config_from_path(raw_path: str) -> tuple[Path, Dict[str, Any]]:
    resolved = _resolve_config_path(raw_path)
    loaded = load_yaml_with_includes(resolved)
    if not isinstance(loaded, dict):
        raise ValueError("Loaded config must be a mapping.")
    return resolved, loaded


def _load_catalog() -> Dict[str, CatalogMeta]:
    if not CATALOG_PATH.exists():
        return {}
    payload = yaml.safe_load(CATALOG_PATH.read_text(encoding="utf-8")) or {}
    fields = payload.get("fields", {})
    out: Dict[str, CatalogMeta] = {}
    if not isinstance(fields, dict):
        return out
    for path, spec in fields.items():
        if not isinstance(spec, dict):
            continue
        ui_spec = spec.get("ui", {}) if isinstance(spec.get("ui"), dict) else {}
        out[path] = CatalogMeta(
            path=str(spec.get("path", path)),
            label=str(spec.get("label", path)),
            description=str(spec.get("description", "")),
            affects=str(spec.get("affects", "")),
            default=spec.get("default"),
            enum=[str(item) for item in (spec.get("enum") or [])],
            required_in_modes=[str(item) for item in (spec.get("required_in_modes") or [])],
            required_when=spec.get("required_when") if isinstance(spec.get("required_when"), dict) else {},
            ui_tab=str(ui_spec.get("tab", "")),
            ui_group=str(ui_spec.get("group", "")),
            ui_priority=int(ui_spec.get("priority", 100)),
            ui_order=int(ui_spec.get("order", 1000)),
        )
    return out


def _infer_type_name(value: Any) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "dict"
    return "str"


def _collect_editable_paths(config: Dict[str, Any], catalog: Dict[str, CatalogMeta]) -> List[str]:
    flat = _flatten(config)
    pool = set(flat.keys()) | set(catalog.keys())
    out: List[str] = []
    for path in pool:
        if path == "mapping.graph_feature_mapping":
            continue
        if path.startswith("mapping.graph_feature_mapping.") and path != "mapping.graph_feature_mapping.enabled":
            continue
        if path == "features":
            continue
        if any(path == prefix or path.startswith(prefix + ".") for prefix in EXCLUDED_PREFIXES):
            continue
        out.append(path)
    return out


def _paths_sort_key(path: str, catalog: Dict[str, CatalogMeta]) -> tuple[int, int, int, str]:
    if path in CORE_FIELDS:
        return (0, CORE_FIELDS.index(path), 0, path)
    meta = catalog.get(path)
    if meta is None:
        return (50, 1000, 1000, path)
    return (10, meta.ui_priority, meta.ui_order, path)


def _build_type_hints(
    paths: Iterable[str],
    config: Dict[str, Any],
    catalog: Dict[str, CatalogMeta],
) -> Dict[str, str]:
    hints: Dict[str, str] = {}
    for path in paths:
        current = _deep_get(config, path, None)
        if current is not None:
            hints[path] = _infer_type_name(current)
            continue
        meta = catalog.get(path)
        if meta is not None and meta.default not in {"", None}:
            hints[path] = _infer_type_name(meta.default)
            continue
        if path in {
            "training.retrain",
            "mapping.graph_feature_mapping.enabled",
            "experiment.retrain",
            "experiment.statistic_enabled",
            "experiment.structural_mode",
            "experiment.mask_guided_enabled",
        }:
            hints[path] = "bool"
            continue
        hints[path] = "str"
    return hints


def _serialize_for_field(value: Any, type_name: str) -> Any:
    if type_name == "bool":
        return _normalize_bool(value, fallback=False)
    if type_name in {"list", "dict"}:
        return _safe_yaml_dump(value)
    if value is None:
        return ""
    return str(value)


def _default_config_path() -> str:
    experiments_dir = ROOT_DIR / "configs" / "experiments"
    if not experiments_dir.exists():
        return ""
    ui_runs = sorted(experiments_dir.glob("ui_run_*.yaml"), key=lambda p: p.stat().st_mtime, reverse=True)
    if ui_runs:
        return str(ui_runs[0])
    all_yaml = sorted(experiments_dir.glob("*.yaml"), key=lambda p: p.stat().st_mtime, reverse=True)
    if all_yaml:
        return str(all_yaml[0])
    return ""


def _init_session() -> None:
    st.session_state.setdefault("catalog", _load_catalog())
    st.session_state.setdefault("loaded_config_path", "")
    st.session_state.setdefault("base_config", {})
    st.session_state.setdefault("editable_paths", [])
    st.session_state.setdefault("type_hints", {})
    st.session_state.setdefault("field_values", {})
    st.session_state.setdefault("yaml_features_text", "[]")
    st.session_state.setdefault("yaml_policies_text", "{}")
    st.session_state.setdefault("yaml_graph_mapping_text", "{}")
    st.session_state.setdefault("graph_mapping_enabled", False)
    st.session_state.setdefault("graph_gateway_mode", "collapse_for_prediction")
    st.session_state.setdefault("pending_run_cfg", None)
    st.session_state.setdefault("pending_run_warnings", [])
    st.session_state.setdefault("mlflow_runs", [])
    st.session_state.setdefault("mlflow_error", "")
    st.session_state.setdefault("run_process", None)
    st.session_state.setdefault("run_queue", None)
    st.session_state.setdefault("run_thread", None)
    st.session_state.setdefault("run_logs", [])
    st.session_state.setdefault("run_progress_seen", False)
    st.session_state.setdefault("run_stage_progress", {})
    st.session_state.setdefault("run_stage_runtime", {})
    st.session_state.setdefault("run_stage_detail", {})
    st.session_state.setdefault("run_current_stage", "")
    st.session_state.setdefault("run_overall_percent", 0.0)
    st.session_state.setdefault("run_last_exit_code", None)
    st.session_state.setdefault("run_temp_cfg", "")
    st.session_state.setdefault("run_started_ts", None)


def _apply_loaded_config(config: Dict[str, Any], loaded_path: str) -> None:
    catalog = st.session_state["catalog"]
    for key in list(st.session_state.keys()):
        if str(key).startswith("field::"):
            st.session_state.pop(key, None)
    for key in (
        "yaml_features_widget",
        "yaml_policies_widget",
        "yaml_graph_mapping_widget",
        "graph_mapping_enabled_widget",
        "graph_gateway_mode_widget",
        "confirm_run_warnings",
    ):
        st.session_state.pop(key, None)

    editable_paths = sorted(_collect_editable_paths(config, catalog), key=lambda p: _paths_sort_key(p, catalog))
    type_hints = _build_type_hints(editable_paths, config, catalog)

    values: Dict[str, Any] = {}
    for path in editable_paths:
        value = _deep_get(config, path, None)
        values[path] = _serialize_for_field(value, type_hints.get(path, "str"))

    features_value = _deep_get(config, "mapping.features", None)
    if features_value is None:
        features_value = _deep_get(config, "features", [])
    policies_value = _deep_get(config, "policies", {})
    graph_mapping_value = _deep_get(config, "mapping.graph_feature_mapping", {})
    graph_enabled = _normalize_bool(_deep_get(config, "mapping.graph_feature_mapping.enabled", False), fallback=False)
    graph_gateway_mode = str(
        _deep_get(
            config,
            "mapping.graph_feature_mapping.topology_projection.gateway_mode",
            "collapse_for_prediction" if str(_deep_get(config, "mapping.adapter", "")).strip().lower() == "xes" else "preserve",
        )
    ).strip()
    if graph_gateway_mode not in {"preserve", "collapse_for_prediction"}:
        graph_gateway_mode = "collapse_for_prediction" if str(_deep_get(config, "mapping.adapter", "")).strip().lower() == "xes" else "preserve"

    st.session_state["loaded_config_path"] = loaded_path
    st.session_state["base_config"] = deepcopy(config)
    st.session_state["editable_paths"] = editable_paths
    st.session_state["type_hints"] = type_hints
    st.session_state["field_values"] = values
    st.session_state["yaml_features_text"] = _safe_yaml_dump(features_value if features_value is not None else [])
    st.session_state["yaml_policies_text"] = _safe_yaml_dump(policies_value if policies_value is not None else {})
    st.session_state["yaml_graph_mapping_text"] = _safe_yaml_dump(
        graph_mapping_value if graph_mapping_value is not None else {"enabled": graph_enabled}
    )
    st.session_state["graph_mapping_enabled"] = graph_enabled
    st.session_state["graph_gateway_mode"] = graph_gateway_mode
    st.session_state["yaml_features_widget"] = st.session_state["yaml_features_text"]
    st.session_state["yaml_policies_widget"] = st.session_state["yaml_policies_text"]
    st.session_state["yaml_graph_mapping_widget"] = st.session_state["yaml_graph_mapping_text"]
    st.session_state["graph_mapping_enabled_widget"] = graph_enabled
    st.session_state["graph_gateway_mode_widget"] = graph_gateway_mode
    st.session_state["pending_run_cfg"] = None
    st.session_state["pending_run_warnings"] = []


def _compose_config() -> tuple[Dict[str, Any], List[str]]:
    base_cfg = st.session_state.get("base_config", {})
    if not isinstance(base_cfg, dict):
        base_cfg = {}
    cfg = deepcopy(base_cfg)
    parse_warnings: List[str] = []

    values = st.session_state.get("field_values", {})
    hints = st.session_state.get("type_hints", {})
    for path, raw in values.items():
        expected = hints.get(path, "str")
        if expected == "bool":
            parsed = bool(raw) if isinstance(raw, bool) else _normalize_bool(raw, fallback=False)
        elif isinstance(raw, str):
            parsed = _coerce_from_text(raw, expected)
        else:
            parsed = raw
        if parsed == "" and not _deep_has(base_cfg, path):
            continue
        _deep_set(cfg, path, parsed)

    features_payload, err = _parse_yaml_block(st.session_state.get("yaml_features_text", ""), "mapping.features")
    if err:
        parse_warnings.append(err)
    if features_payload is not None:
        _deep_set(cfg, "mapping.features", features_payload)
        if "features" in cfg:
            _deep_set(cfg, "features", features_payload)

    policies_payload, err = _parse_yaml_block(st.session_state.get("yaml_policies_text", ""), "policies")
    if err:
        parse_warnings.append(err)
    if policies_payload is not None:
        _deep_set(cfg, "policies", policies_payload)

    graph_mapping_payload, err = _parse_yaml_block(
        st.session_state.get("yaml_graph_mapping_text", ""),
        "mapping.graph_feature_mapping",
    )
    if err:
        parse_warnings.append(err)
    if not isinstance(graph_mapping_payload, dict):
        graph_mapping_payload = {}
    graph_mapping_payload["enabled"] = bool(st.session_state.get("graph_mapping_enabled", False))
    projection = graph_mapping_payload.get("topology_projection", {})
    if not isinstance(projection, dict):
        projection = {}
    gateway_mode = str(st.session_state.get("graph_gateway_mode", "")).strip()
    if gateway_mode not in {"preserve", "collapse_for_prediction"}:
        adapter = str(_deep_get(cfg, "mapping.adapter", "")).strip().lower()
        gateway_mode = "collapse_for_prediction" if adapter == "xes" else "preserve"
    projection["gateway_mode"] = gateway_mode
    graph_mapping_payload["topology_projection"] = projection
    _deep_set(cfg, "mapping.graph_feature_mapping", graph_mapping_payload)
    _deep_set(cfg, "mapping.graph_feature_mapping.enabled", bool(st.session_state.get("graph_mapping_enabled", False)))
    _deep_set(cfg, "mapping.graph_feature_mapping.topology_projection.gateway_mode", gateway_mode)

    return cfg, parse_warnings


def _is_blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) == 0
    return False


def _is_path_like(path: str) -> bool:
    last = path.split(".")[-1]
    return (
        last.endswith("_path")
        or last.endswith("_dir")
        or last in {"path", "log_path", "connections_file", "catalog_file", "bpmn_dir", "export_dir", "sql_dir"}
    )


def _is_path_active(path: str, mode: str, adapter: str) -> bool:
    sync_modes = {"sync-topology", "sync-stats", "sync-stats-backfill"}
    if mode in sync_modes and (path.startswith("model.") or path.startswith("training.") or path.startswith("tracking.")):
        return False
    if adapter == "camunda":
        if path.startswith("mapping.xes_adapter.") or path == "data.log_path":
            return False
    if adapter == "xes" and path.startswith("mapping.camunda_adapter."):
        return False
    return True


def _resolve_maybe_relative_path(raw: str) -> Path:
    path = Path(str(raw).strip()).expanduser()
    if path.is_absolute():
        return path
    return (ROOT_DIR / path).resolve()


def _to_rel_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT_DIR.resolve()))
    except Exception:
        try:
            return str(path.resolve())
        except Exception:
            return str(path)


def _discover_path_options(path_key: str) -> List[str]:
    patterns = PATH_PICKER_GLOBS.get(path_key, [])
    seen: set[str] = set()
    out: List[str] = []
    for pattern in patterns:
        for candidate in ROOT_DIR.glob(pattern):
            if not candidate.is_file():
                continue
            text = _to_rel_path(candidate)
            if text in seen:
                continue
            seen.add(text)
            out.append(text)
    out.sort()
    return out


def _format_elapsed(seconds: float | None) -> str:
    if seconds is None or seconds <= 0:
        return "00:00"
    total = int(seconds)
    mins, sec = divmod(total, 60)
    hrs, mins = divmod(mins, 60)
    if hrs > 0:
        return f"{hrs:02d}:{mins:02d}:{sec:02d}"
    return f"{mins:02d}:{sec:02d}"


def _format_eta(seconds: float | None) -> str:
    if seconds is None or seconds <= 0:
        return "--:--"
    return _format_elapsed(seconds)


def _format_bytes(value: int | float | None) -> str:
    if value is None:
        return "--"
    amount = float(value)
    units = ["B", "KB", "MB", "GB", "TB"]
    idx = 0
    while amount >= 1024.0 and idx < len(units) - 1:
        amount /= 1024.0
        idx += 1
    if idx == 0:
        return f"{int(amount)} {units[idx]}"
    return f"{amount:.1f} {units[idx]}"


def _validate_config_against_catalog(cfg: Dict[str, Any], catalog: Dict[str, CatalogMeta]) -> List[str]:
    warnings: List[str] = []
    mode = str(_deep_get(cfg, "experiment.mode", "")).strip()
    adapter = str(_deep_get(cfg, "mapping.adapter", "")).strip().lower()

    for path, meta in catalog.items():
        if not _is_path_active(path, mode, adapter):
            continue
        should_check_required = mode in meta.required_in_modes if meta.required_in_modes else False
        if not should_check_required and meta.required_when:
            required_match = True
            for cond_path, expected in meta.required_when.items():
                current = _deep_get(cfg, cond_path, None)
                if isinstance(expected, list):
                    if current not in expected:
                        required_match = False
                        break
                elif current != expected:
                    required_match = False
                    break
            should_check_required = required_match
        if should_check_required and _is_blank(_deep_get(cfg, path, None)):
            warnings.append(f"Missing required field: {path}")

        current = _deep_get(cfg, path, None)
        if meta.enum and not _is_blank(current) and path in STRICT_ENUM_PATHS and not _is_path_like(path):
            current_text = str(current).strip()
            allowed = [str(item).strip() for item in meta.enum]
            if path == "experiment.cache_policy":
                current_norm = current_text.lower()
                if current_norm in {"off", "false", "none", "disabled"}:
                    current_norm = "off"
                allowed_norm: List[str] = []
                for item in allowed:
                    token = item.lower()
                    if token in {"off", "false", "none", "disabled"}:
                        token = "off"
                    allowed_norm.append(token)
                if current_norm not in allowed_norm:
                    warnings.append(f"Invalid enum value for {path}: {current_text} (allowed: {', '.join(allowed)})")
                continue
            if current_text not in allowed:
                warnings.append(f"Invalid enum value for {path}: {current_text} (allowed: {', '.join(allowed)})")

    mode_key = str(_deep_get(cfg, "experiment.mode", "train")).strip().lower()
    if mode_key.startswith("eval_"):
        checkpoint = str(_deep_get(cfg, "experiment.load_checkpoint", "")).strip()
        if checkpoint == "":
            warnings.append("eval_* mode without experiment.load_checkpoint.")
        elif not _resolve_maybe_relative_path(checkpoint).exists():
            warnings.append(f"Checkpoint path does not exist: {checkpoint}")

    adapter_key = str(_deep_get(cfg, "mapping.adapter", "")).strip().lower()
    if adapter_key == "xes":
        log_path = str(_deep_get(cfg, "data.log_path", "")).strip()
        if log_path and not _resolve_maybe_relative_path(log_path).exists():
            warnings.append(f"XES path does not exist: {log_path}")

    return warnings


def _sanitize_token(raw: str, fallback: str = "x") -> str:
    text = str(raw or "").strip().lower()
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in text).strip("_")
    if not cleaned:
        return fallback
    return cleaned


def _build_ui_run_filename(cfg: Dict[str, Any]) -> str:
    mode = _sanitize_token(str(_deep_get(cfg, "experiment.mode", "run")), fallback="run")
    dataset = _sanitize_token(
        str(_deep_get(cfg, "data.dataset_label", "")) or str(_deep_get(cfg, "data.dataset_name", "")),
        fallback="dataset",
    )
    model = _sanitize_token(str(_deep_get(cfg, "model.model_label", "")) or str(_deep_get(cfg, "model.type", "")), fallback="model")
    seed = _sanitize_token(str(_deep_get(cfg, "seed", "0")), fallback="0")
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    return f"ui_{mode}_{dataset}_{model}_s{seed}_{stamp}.yaml"


def _compute_full_run_name(cfg: Dict[str, Any]) -> str:
    exp_name = str(_deep_get(cfg, "experiment.name", "Run")).strip() or "Run"
    ds_label = str(_deep_get(cfg, "data.dataset_label", _deep_get(cfg, "experiment.dataset", "unknown_data"))).strip() or "unknown_data"
    model_label = str(_deep_get(cfg, "model.model_label", _deep_get(cfg, "model.type", "unknown_model"))).strip() or "unknown_model"
    return f"{exp_name}_{ds_label}_{model_label}"


def _checkpoint_summary(cfg: Dict[str, Any], full_run_name: str) -> Dict[str, str]:
    mode = str(_deep_get(cfg, "experiment.mode", "train")).strip().lower()
    eval_path = str(_deep_get(cfg, "experiment.load_checkpoint", "")).strip()
    checkpoint_dir = str(_deep_get(cfg, "training.checkpoint_dir", "checkpoints")).strip() or "checkpoints"
    checkpoint_run_name = str(_deep_get(cfg, "training.checkpoint_run_name", full_run_name)).strip() or full_run_name
    training_override = str(_deep_get(cfg, "training.checkpoint_path", "")).strip()
    training_resolved = training_override or str(Path(checkpoint_dir) / f"{checkpoint_run_name}_best.pth")
    effective = eval_path if mode.startswith("eval_") else training_resolved
    return {
        "mode": mode,
        "eval_path": eval_path,
        "training_path": training_resolved,
        "effective": effective,
    }


def _structure_variant(cfg: Dict[str, Any]) -> str:
    model_type = str(_deep_get(cfg, "model.type", "")).strip()
    if model_type == "EOPKGGATv2":
        return "EOPKG"
    return "baseline"


def _current_epoch_text() -> str:
    details = st.session_state.get("run_stage_detail", {})
    epoch_info = details.get("train.epochs")
    if not isinstance(epoch_info, dict):
        return ""
    current = _safe_float(epoch_info.get("current", 0.0), default=0.0)
    total = _safe_float(epoch_info.get("total", 0.0), default=0.0)
    if total <= 0.0:
        return ""
    if abs(current - round(current)) < 1e-9:
        current_text = str(int(round(current)))
    else:
        current_text = f"{current:.1f}"
    if abs(total - round(total)) < 1e-9:
        total_text = str(int(round(total)))
    else:
        total_text = f"{total:.1f}"
    return f"{current_text}/{total_text}"


def _build_run_command(temp_cfg_path: Path, cfg: Dict[str, Any]) -> List[str]:
    mode = str(_deep_get(cfg, "experiment.mode", "train")).strip()
    cmd = [sys.executable, "main.py"]
    if mode == "sync-stats":
        cmd += ["sync-stats", "--config", str(temp_cfg_path)]
    elif mode == "sync-stats-backfill":
        step = str(_deep_get(cfg, "sync_stats_backfill.step", "weekly")).strip() or "weekly"
        cmd += ["sync-stats-backfill", "--config", str(temp_cfg_path), "--step", step]
        step_days = str(_deep_get(cfg, "sync_stats_backfill.step_days", "")).strip()
        date_from = str(_deep_get(cfg, "sync_stats_backfill.from", "")).strip()
        date_to = str(_deep_get(cfg, "sync_stats_backfill.to", "")).strip()
        if step_days:
            cmd += ["--step-days", step_days]
        if date_from:
            cmd += ["--from", date_from]
        if date_to:
            cmd += ["--to", date_to]
    elif mode == "sync-topology":
        cmd += ["sync-topology", "--config", str(temp_cfg_path)]
    else:
        cmd += ["--config", str(temp_cfg_path)]
    return cmd


def _start_run(cfg: Dict[str, Any]) -> tuple[bool, str]:
    if st.session_state.get("run_process") is not None:
        return False, "Process is already running."

    tmp_dir = ROOT_DIR / "outputs" / "ui" / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix="ui_run_", suffix=".yaml", dir=str(tmp_dir))
    Path(temp_path).write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    try:
        import os

        os.close(fd)
    except OSError:
        pass

    cmd = _build_run_command(Path(temp_path), cfg)

    import os

    env = os.environ.copy()
    env.update(
        {
            "BPM_PROGRESS_EVENTS": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )

    process = subprocess.Popen(
        cmd,
        cwd=str(ROOT_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )

    q: queue.Queue[str] = queue.Queue()

    def _reader(proc: subprocess.Popen[str], out_queue: queue.Queue[str]) -> None:
        try:
            if proc.stdout is not None:
                for line in proc.stdout:
                    out_queue.put(line)
            rc = proc.wait()
            out_queue.put(f"[exit code: {rc}]")
        except Exception as exc:  # pragma: no cover
            out_queue.put(f"[run failed] {exc}")
        finally:
            out_queue.put("__FINISHED__")

    thread = threading.Thread(target=_reader, args=(process, q), daemon=True)
    thread.start()

    st.session_state["run_process"] = process
    st.session_state["run_queue"] = q
    st.session_state["run_thread"] = thread
    st.session_state["run_logs"] = [f"$ {' '.join(cmd)}"]
    st.session_state["run_progress_seen"] = False
    st.session_state["run_stage_progress"] = {}
    st.session_state["run_stage_runtime"] = {}
    st.session_state["run_stage_detail"] = {}
    st.session_state["run_current_stage"] = "launch"
    st.session_state["run_overall_percent"] = 0.0
    st.session_state["run_last_exit_code"] = None
    st.session_state["run_temp_cfg"] = temp_path
    st.session_state["run_started_ts"] = time.time()

    return True, ""


def _stop_run() -> None:
    process = st.session_state.get("run_process")
    if process is None:
        return
    try:
        process.terminate()
    except Exception:
        pass


def _stage_title(stage: str) -> str:
    raw = str(stage or "").strip()
    if raw == "":
        return "idle"
    return STAGE_TITLE.get(raw, raw.replace("_", " "))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _compute_overall_percent() -> float:
    stage_progress: Dict[str, float] = st.session_state.get("run_stage_progress", {})
    weighted = 0.0
    total_weight = 0.0
    for stage, progress in stage_progress.items():
        weight = float(RUN_STAGE_WEIGHTS.get(stage, 0.01))
        total_weight += weight
        weighted += weight * max(0.0, min(1.0, float(progress)))
    if total_weight <= 1e-9:
        return 0.0
    return max(0.0, min(100.0, (weighted / total_weight) * 100.0))


def _apply_progress_event(event: Dict[str, Any]) -> None:
    st.session_state["run_progress_seen"] = True
    stage = str(event.get("stage", "unknown")).strip() or "unknown"
    status = str(event.get("status", "update")).strip().lower() or "update"
    current = _safe_float(event.get("current", 0.0), default=0.0)
    total = _safe_float(event.get("total", 0.0), default=0.0)
    event_percent = _safe_float(event.get("percent", -1.0), default=-1.0)
    if event_percent < 0.0:
        event_percent = (current / total * 100.0) if total > 0.0 else 0.0
    if status == "done":
        event_percent = 100.0
    event_percent = max(0.0, min(100.0, event_percent))

    stage_runtime = st.session_state.get("run_stage_runtime", {})
    now_ts = time.time()
    runtime = stage_runtime.get(stage)
    if not isinstance(runtime, dict):
        runtime = {"start_ts": now_ts, "elapsed": 0.0, "eta": None}
        stage_runtime[stage] = runtime
    if status == "start":
        runtime["start_ts"] = now_ts
    start_ts = _safe_float(runtime.get("start_ts", now_ts), default=now_ts)
    elapsed = max(0.0, now_ts - start_ts)
    eta = None
    if 0.0 < event_percent < 100.0:
        ratio = event_percent / 100.0
        eta = (elapsed / ratio) - elapsed if ratio > 1e-9 else None
    elif status == "done":
        eta = 0.0
    runtime["elapsed"] = elapsed
    runtime["eta"] = eta
    stage_runtime[stage] = runtime
    st.session_state["run_stage_runtime"] = stage_runtime

    stage_progress = st.session_state.get("run_stage_progress", {})
    if status == "done":
        stage_progress[stage] = 1.0
    else:
        stage_progress[stage] = max(0.0, min(1.0, event_percent / 100.0))
    st.session_state["run_stage_progress"] = stage_progress

    stage_detail = st.session_state.get("run_stage_detail", {})
    if not isinstance(stage_detail, dict):
        stage_detail = {}
    stage_detail[stage] = {
        "status": status,
        "message": str(event.get("message", "")).strip(),
        "level": str(event.get("level", "info")).strip().lower() or "info",
        "current": current,
        "total": total,
        "percent": event_percent,
        "payload": event.get("payload", {}) if isinstance(event.get("payload"), dict) else {},
        "ts": _safe_float(event.get("ts", now_ts), default=now_ts),
    }
    st.session_state["run_stage_detail"] = stage_detail

    st.session_state["run_current_stage"] = stage
    st.session_state["run_overall_percent"] = _compute_overall_percent()


def _append_run_log(text: str) -> None:
    logs = st.session_state.get("run_logs", [])
    logs.append(text.rstrip("\n"))
    if len(logs) > 3000:
        logs = logs[-3000:]
    st.session_state["run_logs"] = logs


def _poll_run_queue() -> None:
    q = st.session_state.get("run_queue")
    if q is None:
        return
    while True:
        try:
            item = q.get_nowait()
        except queue.Empty:
            break
        if item == "__FINISHED__":
            process = st.session_state.get("run_process")
            if process is not None:
                try:
                    process.wait(timeout=0.2)
                except Exception:
                    pass
            st.session_state["run_process"] = None
            st.session_state["run_queue"] = None
            st.session_state["run_thread"] = None
            temp_cfg = str(st.session_state.get("run_temp_cfg", "")).strip()
            if temp_cfg:
                try:
                    Path(temp_cfg).unlink(missing_ok=True)
                except OSError:
                    pass
            st.session_state["run_temp_cfg"] = ""
            if st.session_state.get("run_overall_percent", 0.0) < 100.0:
                st.session_state["run_overall_percent"] = 100.0
            continue

        line = str(item).rstrip("\n")
        if line.startswith(PROGRESS_EVENT_PREFIX):
            try:
                payload = json.loads(line[len(PROGRESS_EVENT_PREFIX) :].strip())
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                _apply_progress_event(payload)
            continue

        if line.startswith("[exit code:") and line.endswith("]"):
            try:
                st.session_state["run_last_exit_code"] = int(line[len("[exit code:") : -1].strip())
            except ValueError:
                st.session_state["run_last_exit_code"] = None
        _append_run_log(line)


def _render_sticky_summary(cfg: Dict[str, Any], warnings: List[str], parse_warnings: List[str]) -> None:
    full_run_name = _compute_full_run_name(cfg)
    cp = _checkpoint_summary(cfg, full_run_name)
    summary_warnings = len(warnings) + len(parse_warnings)
    seed = str(_deep_get(cfg, "seed", "")).strip() or "<empty>"
    structure_mode = _structure_variant(cfg)
    stats_enabled = bool(st.session_state.get("graph_mapping_enabled", False))
    epoch_text = _current_epoch_text() or "<empty>"
    html_block = f"""
    <div class="web-summary">
      <div><strong>Final MLflow run name:</strong> {html.escape(full_run_name)}</div>
      <div><strong>Mode:</strong> {html.escape(cp["mode"])} | <strong>epoch:</strong> {html.escape(epoch_text)} | <strong>seed:</strong> {html.escape(seed)} | <strong>Structure:</strong> {html.escape(structure_mode)} | <strong>Statistics:</strong> {str(stats_enabled).lower()}</div>
      <div><strong>Eval checkpoint:</strong> {html.escape(cp["eval_path"] or "<empty>")}</div>
      <div><strong>Training checkpoint:</strong> {html.escape(cp["training_path"] or "<empty>")}</div>
      <div><strong>Effective checkpoint:</strong> {html.escape(cp["effective"] or "<empty>")}</div>
      <div><strong>Validation warnings:</strong> {summary_warnings}</div>
    </div>
    """
    st.markdown(html_block, unsafe_allow_html=True)


def _group_for_path(path: str) -> str:
    if path in CORE_FIELDS:
        return "Core Variant"
    if path.startswith("experiment.") or path == "seed":
        return "Experiment Settings"
    if path.startswith("data.") or path.startswith("mapping.xes_adapter.") or path.startswith("mapping.camunda_adapter."):
        return "Data & Ingestion"
    if path.startswith("mapping.knowledge_graph.") or path.startswith("sync_stats.") or path.startswith("mapping.graph_feature_mapping"):
        return "Structure & Stats"
    if path.startswith("model."):
        return "Model"
    if path.startswith("training."):
        return "Training"
    if path.startswith("tracking."):
        return "Tracking"
    if path.startswith("mapping."):
        return "Mapping"
    return "Other"


def _render_field(path: str, meta: CatalogMeta | None) -> None:
    values = st.session_state["field_values"]
    hints = st.session_state["type_hints"]
    type_name = hints.get(path, "str")
    raw_value = values.get(path, "")

    label = meta.label if meta is not None and str(meta.label).strip() else path
    help_parts: List[str] = [path]
    if meta is not None:
        if meta.description.strip():
            help_parts.append(meta.description.strip())
        if meta.affects.strip():
            help_parts.append(f"Affects: {meta.affects.strip()}")
        if meta.required_in_modes:
            help_parts.append(f"Required in modes: {', '.join(meta.required_in_modes)}")
    help_text = "\n\n".join(help_parts)
    widget_key = f"field::{path}"
    current_cfg = st.session_state.get("base_config", {})
    model_type = str(_deep_get(current_cfg, "model.type", "")).strip()
    eopkg_only_field = path in {"model.struct_encoder_type", "model.fusion_mode"}
    struct_encoder_disabled = eopkg_only_field and model_type != "EOPKGGATv2"

    if path in SPECIAL_SELECT_CHOICES:
        value_labels = SPECIAL_SELECT_CHOICES[path]
        options = [item[0] for item in value_labels]
        labels = {item[0]: item[1] for item in value_labels}
        current = str(raw_value).strip()
        if current and current not in options:
            options = [current] + options
            labels[current] = current
        idx = options.index(current) if current in options else 0
        chosen = st.selectbox(
            label,
            options=options,
            index=idx,
            format_func=lambda token: labels.get(token, str(token)),
            help=help_text,
            key=widget_key,
            disabled=struct_encoder_disabled,
        )
        values[path] = chosen
        if struct_encoder_disabled:
            st.caption(f"`{path}` застосовується лише для `model.type=EOPKGGATv2`.")
        return

    if meta is not None and meta.enum and path in STRICT_ENUM_PATHS:
        options = [str(item).strip() for item in meta.enum if str(item).strip()]
        current = str(raw_value).strip()
        if current and current not in options:
            options = [current] + options
        if options:
            idx = options.index(current) if current in options else 0
            chosen = st.selectbox(
                label,
                options=options,
                index=idx,
                help=help_text,
                key=widget_key,
                disabled=struct_encoder_disabled,
            )
            values[path] = chosen
            if struct_encoder_disabled:
                st.caption(f"`{path}` застосовується лише для `model.type=EOPKGGATv2`.")
            return

    if type_name == "bool":
        checked = bool(raw_value) if isinstance(raw_value, bool) else _normalize_bool(raw_value, fallback=False)
        chosen = st.checkbox(label, value=checked, help=help_text, key=widget_key)
        values[path] = bool(chosen)
        return

    if _is_path_like(path):
        text = raw_value if isinstance(raw_value, str) else str(raw_value)
        p1, p2 = st.columns([2.8, 1.2])
        with p1:
            new_text = st.text_input(label, value=text, help=help_text, key=widget_key)
        with p2:
            discovered = _discover_path_options(path)
            select_options = ["<manual>"] + discovered
            pick_key = f"{widget_key}::picker"
            picked = st.selectbox("Known files", options=select_options, index=0, key=pick_key, label_visibility="visible")
            if picked != "<manual>":
                new_text = picked
                st.session_state[widget_key] = picked
        values[path] = new_text
        return

    if type_name in {"list", "dict"}:
        text = raw_value if isinstance(raw_value, str) else _safe_yaml_dump(raw_value)
        new_text = st.text_area(label, value=text, help=help_text, key=widget_key, height=110)
        values[path] = new_text
        return

    text = raw_value if isinstance(raw_value, str) else str(raw_value)
    new_text = st.text_input(label, value=text, help=help_text, key=widget_key)
    values[path] = new_text


def _render_yaml_editor(label: str, key: str, value: str, height: int = 420) -> str:
    if st_ace is not None:
        edited = st_ace(
            value=value,
            language="yaml",
            theme="tomorrow_night",
            key=key,
            height=height,
            wrap=True,
            show_gutter=True,
            show_print_margin=False,
            auto_update=True,
            font_size=13,
        )
        st.caption(f"{label} (YAML editor)")
        return str(edited if edited is not None else value)
    return st.text_area(label, value=value, height=height, key=key)


def _fetch_mlflow_runs(tracking_uri: str, experiment_id: str) -> tuple[List[Dict[str, Any]], str]:
    if mlflow is None or MlflowClient is None or ViewType is None:
        return [], "MLflow client is unavailable."
    try:
        if tracking_uri.strip():
            mlflow.set_tracking_uri(tracking_uri.strip())
            client = MlflowClient(tracking_uri=tracking_uri.strip())
        else:
            client = MlflowClient()

        exp_ids: List[str] = []
        explicit = str(experiment_id).strip()
        if explicit:
            exp_ids = [explicit]
        else:
            experiments = client.search_experiments(view_type=ViewType.ACTIVE_ONLY)
            experiments = sorted(experiments, key=lambda e: getattr(e, "last_update_time", 0), reverse=True)
            exp_ids = [str(exp.experiment_id) for exp in experiments[:25]]
        if not exp_ids:
            return [], "No experiments found in MLflow."

        df = mlflow.search_runs(experiment_ids=exp_ids, max_results=150, order_by=["attribute.start_time DESC"])
        runs: List[Dict[str, Any]] = []
        if df is None or df.empty:
            return [], "No runs found for selected experiments."

        for _, row in df.iterrows():
            run_id = str(row.get("run_id", "") or "")
            exp_id = str(row.get("experiment_id", "") or "")
            run_name = str(row.get("tags.mlflow.runName", "") or "")
            start_time = row.get("start_time", None)
            created = ""
            if start_time is not None:
                try:
                    ts = getattr(start_time, "to_pydatetime", lambda: start_time)()
                    created = ts.strftime("%Y-%m-%d %H:%M:%S")
                except Exception:
                    created = str(start_time)
            runs.append(
                {
                    "run_id": run_id,
                    "experiment_id": exp_id,
                    "run_name": run_name,
                    "created": created,
                }
            )
        return runs, ""
    except Exception as exc:  # pragma: no cover
        return [], f"Failed to fetch MLflow runs: {exc}"


def _parse_mlflow_param_value(raw: str) -> Any:
    text = str(raw or "").strip()
    if text == "":
        return ""
    lower = text.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    try:
        if "." in text:
            return float(text)
        return int(text)
    except ValueError:
        pass
    if (text.startswith("[") and text.endswith("]")) or (text.startswith("{") and text.endswith("}")):
        try:
            return yaml.safe_load(text)
        except yaml.YAMLError:
            return text
    return text


def _clone_from_mlflow_run(cfg: Dict[str, Any], target_run_id: str, tracking_uri: str) -> tuple[Dict[str, Any], str]:
    if MlflowClient is None:
        return cfg, "MLflow client is unavailable."
    try:
        client = MlflowClient(tracking_uri=tracking_uri.strip() or None)
        run = client.get_run(target_run_id)
    except Exception as exc:  # pragma: no cover
        return cfg, f"Failed to load MLflow run {target_run_id}: {exc}"

    merged = deepcopy(cfg)
    allowed_prefixes = ("experiment.", "data.", "model.", "training.", "mapping.", "tracking.", "sync_stats.")
    for key, value in run.data.params.items():
        if key == "seed" or key.startswith(allowed_prefixes):
            parsed = _parse_mlflow_param_value(str(value))
            _deep_set(merged, key, parsed)
    return merged, ""


def _render_mlflow_clone_panel(cfg: Dict[str, Any]) -> None:
    with st.expander("Clone Parameters From MLflow Run", expanded=False):
        if mlflow is None or MlflowClient is None or ViewType is None:
            st.warning("MLflow is not available in this environment. Install dependencies to enable cloning.")
            return

        default_uri = str(_deep_get(cfg, "tracking.uri", "")).strip()
        tracking_uri = st.text_input("Tracking URI (optional)", value=default_uri, key="mlflow_tracking_uri")
        experiment_id = st.text_input("Experiment ID (optional)", value="", key="mlflow_experiment_id")
        run_id_manual = st.text_input("Run ID (optional, overrides selection)", value="", key="mlflow_run_id_manual")

        c1, c2 = st.columns([1, 1])
        refresh_clicked = c1.button("Refresh run list", use_container_width=True)
        clone_clicked = c2.button("Clone params", use_container_width=True)

        if refresh_clicked:
            runs, err = _fetch_mlflow_runs(tracking_uri=tracking_uri, experiment_id=experiment_id)
            st.session_state["mlflow_runs"] = runs
            st.session_state["mlflow_error"] = err

        err_text = str(st.session_state.get("mlflow_error", "")).strip()
        if err_text:
            st.warning(err_text)

        runs = st.session_state.get("mlflow_runs", [])
        options: List[str] = []
        by_label: Dict[str, Dict[str, Any]] = {}
        for row in runs:
            created = str(row.get("created", ""))
            run_name = str(row.get("run_name", ""))
            run_id = str(row.get("run_id", ""))
            exp_id = str(row.get("experiment_id", ""))
            label = f"{created} | exp:{exp_id} | {run_name} | {run_id}"
            options.append(label)
            by_label[label] = row

        selected_label = ""
        if options:
            selected_label = st.selectbox("Recent runs (sorted by created)", options=options, index=0, key="mlflow_selected_run")
        else:
            st.caption("No runs loaded yet.")

        if clone_clicked:
            target_run_id = str(run_id_manual).strip()
            if not target_run_id and selected_label and selected_label in by_label:
                target_run_id = str(by_label[selected_label].get("run_id", "")).strip()
            if not target_run_id:
                st.warning("Select a run or provide Run ID.")
            else:
                merged, err = _clone_from_mlflow_run(cfg, target_run_id=target_run_id, tracking_uri=tracking_uri)
                if err:
                    st.warning(err)
                else:
                    _apply_loaded_config(merged, st.session_state.get("loaded_config_path", ""))
                    st.success(f"Cloned params from run: {target_run_id}")


def _save_ui_run(cfg: Dict[str, Any]) -> tuple[Path, str]:
    experiments_dir = ROOT_DIR / "configs" / "experiments"
    experiments_dir.mkdir(parents=True, exist_ok=True)
    filename = _build_ui_run_filename(cfg)
    target = experiments_dir / filename
    if target.exists():
        stem = target.stem
        suffix = target.suffix
        for idx in range(2, 1000):
            candidate = experiments_dir / f"{stem}_{idx}{suffix}"
            if not candidate.exists():
                target = candidate
                break
    target.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return target, target.name


def _rerun() -> None:
    if hasattr(st, "rerun"):
        st.rerun()
    else:  # pragma: no cover
        st.experimental_rerun()


def _bootstrap_initial_config(default_from_cli: str) -> None:
    if st.session_state.get("base_config"):
        return
    candidate = str(default_from_cli or "").strip() or _default_config_path()
    if candidate == "":
        return
    try:
        resolved, cfg = _load_config_from_path(candidate)
    except Exception:
        return
    _apply_loaded_config(cfg, str(resolved))
    st.session_state["config_path_input"] = str(resolved)


def _render_run_status() -> None:
    _poll_run_queue()
    process = st.session_state.get("run_process")
    overall = float(st.session_state.get("run_overall_percent", 0.0))
    current_stage = str(st.session_state.get("run_current_stage", "")).strip()
    stage_progress = st.session_state.get("run_stage_progress", {})
    stage_runtime = st.session_state.get("run_stage_runtime", {})
    run_started_ts = st.session_state.get("run_started_ts")
    now_ts = time.time()
    total_elapsed = max(0.0, now_ts - float(run_started_ts)) if run_started_ts else 0.0
    overall_eta: float | None = None
    if overall > 0.0 and overall < 100.0:
        ratio = overall / 100.0
        overall_eta = (total_elapsed / ratio) - total_elapsed if ratio > 1e-9 else None
    elif overall >= 100.0:
        overall_eta = 0.0

    st.subheader("Run Status")
    c1, c2, c3 = st.columns([2.2, 1.4, 1.4])
    c1.progress(min(100, max(0, int(round(overall)))) / 100.0)
    c1.caption(f"Overall: {overall:.1f}%")
    c2.metric("Elapsed", _format_elapsed(total_elapsed))
    c3.metric("ETA", _format_eta(overall_eta))

    if current_stage:
        stage_percent = max(0.0, min(100.0, float(stage_progress.get(current_stage, 0.0)) * 100.0))
        current_runtime = stage_runtime.get(current_stage, {})
        current_elapsed = _safe_float(current_runtime.get("elapsed", 0.0), default=0.0)
        current_eta = current_runtime.get("eta")
        st.caption(
            f"Current stage: {_stage_title(current_stage)} | {stage_percent:.1f}% | elapsed {_format_elapsed(current_elapsed)} | ETA {_format_eta(current_eta)}"
        )
    else:
        st.caption("Current stage: idle")

    if process is not None:
        st.info("Process is running. Output updates automatically.")
    else:
        exit_code = st.session_state.get("run_last_exit_code")
        if exit_code is None:
            st.caption("No active process.")
        elif int(exit_code) == 0:
            st.success("Last run finished successfully.")
        else:
            st.warning(f"Last run finished with exit code {exit_code}.")

    st.markdown("**Stages**")
    for stage in RUN_STAGE_ORDER:
        progress = max(0.0, min(100.0, float(stage_progress.get(stage, 0.0)) * 100.0))
        runtime = stage_runtime.get(stage, {})
        elapsed = _safe_float(runtime.get("elapsed", 0.0), default=0.0)
        eta = runtime.get("eta")
        s1, s2 = st.columns([4.5, 1.5])
        with s1:
            st.progress(progress / 100.0)
            st.caption(f"{_stage_title(stage)}: {progress:.1f}%")
        with s2:
            st.caption(f"elapsed {_format_elapsed(elapsed)}")
            st.caption(f"ETA {_format_eta(eta)}")

    system_text = "System RAM: --"
    run_ram_text = "Run RAM (tree): --"
    if psutil is not None:
        try:
            vm = psutil.virtual_memory()
            system_text = f"System RAM: {_format_bytes(vm.used)} / {_format_bytes(vm.total)} ({float(vm.percent):.1f}%)"
        except Exception:
            system_text = "System RAM: --"
        if process is not None:
            try:
                tree_rss = int(psutil.Process(int(process.pid)).memory_info().rss)
                for child in psutil.Process(int(process.pid)).children(recursive=True):
                    try:
                        tree_rss += int(child.memory_info().rss)
                    except Exception:
                        continue
                run_ram_text = f"Run RAM (tree): {_format_bytes(tree_rss)}"
            except Exception:
                run_ram_text = "Run RAM (tree): --"
    m1, m2 = st.columns([1, 1])
    m1.caption(system_text)
    m2.caption(run_ram_text)

    logs = "\n".join(st.session_state.get("run_logs", []))
    escaped = html.escape(logs)
    components.html(
        f"""
        <div id="run-log" style="height:360px; overflow:auto; border:1px solid #d0d7de; border-radius:6px; background:#0f172a; color:#e2e8f0; padding:10px;">
          <pre style="margin:0; font-size:12px; white-space:pre-wrap;">{escaped}</pre>
        </div>
        <script>
          const box = document.getElementById("run-log");
          if (box) {{
            box.scrollTop = box.scrollHeight;
          }}
        </script>
        """,
        height=390,
        scrolling=False,
    )


def _cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", default="", help="Optional config path preloaded in web UI.")
    known, _unknown = parser.parse_known_args(sys.argv[1:])
    return known


def main() -> None:
    args = _cli_args()
    st.set_page_config(page_title="BPM Experiment Web UI", layout="wide")
    st.markdown(
        """
        <style>
        div.block-container {padding-top: 1rem;}
        .web-summary {
            position: sticky;
            top: 0.25rem;
            z-index: 999;
            background: #f6f8fa;
            border: 1px solid #d0d7de;
            border-radius: 8px;
            padding: 10px 12px;
            margin-bottom: 10px;
            line-height: 1.5;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    _init_session()
    _bootstrap_initial_config(default_from_cli=str(args.config or ""))

    st.title("BPM Experiment Web UI")
    st.caption("Config + Run UI with reference dictionaries, YAML editors, and stage-by-stage runtime telemetry.")

    tab_config, tab_run = st.tabs(["Config", "Run"])

    cfg_path_default = st.session_state.get("config_path_input", st.session_state.get("loaded_config_path", ""))
    catalog = st.session_state["catalog"]
    composed_cfg: Dict[str, Any] = {}
    parse_warnings: List[str] = []
    validation_warnings: List[str] = []
    all_warnings: List[str] = []
    save_clicked = False
    run_clicked = False
    stop_clicked = False

    with tab_config:
        known_cfgs: List[str] = []
        for item in ROOT_DIR.glob("configs/experiments/**/*.yaml"):
            if item.is_file():
                known_cfgs.append(_to_rel_path(item))
        known_cfgs = sorted(set(known_cfgs))

        cpath1, cpath2, cpath3 = st.columns([2.5, 1.5, 1.0])
        with cpath1:
            st.session_state["config_path_input"] = st.text_input(
                "Config path",
                value=str(cfg_path_default),
                key="cfg_path_input_widget",
            )
        with cpath2:
            known_default = "<manual>"
            cfg_current = str(st.session_state.get("config_path_input", "")).strip()
            known_options = [known_default] + known_cfgs
            if cfg_current and cfg_current not in known_options:
                known_options.append(cfg_current)
            picked_cfg = st.selectbox("Known configs", options=known_options, index=0, key="cfg_known_select")
        with cpath3:
            use_known = st.button("Use selected", use_container_width=True)

        if use_known and picked_cfg != "<manual>":
            st.session_state["config_path_input"] = picked_cfg
            st.session_state["cfg_path_input_widget"] = picked_cfg
            _rerun()

        a1, a2 = st.columns([1.2, 1.2])
        load_clicked = a1.button("Load config", use_container_width=True)
        save_clicked = a2.button("Save as new ui_run", use_container_width=True)

        if load_clicked:
            try:
                resolved, cfg = _load_config_from_path(st.session_state["config_path_input"])
                _apply_loaded_config(cfg, str(resolved))
                st.success(f"Loaded: {resolved}")
            except Exception as exc:
                st.error(str(exc))

        if not st.session_state.get("base_config"):
            st.warning("Load an experiment config to start editing.")
        else:
            paths = list(st.session_state.get("editable_paths", []))
            if not paths:
                paths = sorted(_collect_editable_paths(st.session_state["base_config"], catalog), key=lambda p: _paths_sort_key(p, catalog))
                st.session_state["editable_paths"] = paths
                st.session_state["type_hints"] = _build_type_hints(paths, st.session_state["base_config"], catalog)

            search = st.text_input("Search parameter by path/label", value="", key="search_query")
            search_key = str(search).strip().lower()
            filtered: List[str] = []
            for path in paths:
                meta = catalog.get(path)
                label = meta.label if meta is not None else path
                hay = f"{path} {label}".lower()
                if search_key and search_key not in hay:
                    continue
                filtered.append(path)

            grouped: Dict[str, List[str]] = {}
            for path in filtered:
                group = _group_for_path(path)
                grouped.setdefault(group, []).append(path)

            for group_name in (
                "Core Variant",
                "Experiment Settings",
                "Data & Ingestion",
                "Structure & Stats",
                "Model",
                "Training",
                "Tracking",
                "Mapping",
                "Other",
            ):
                items = grouped.get(group_name, [])
                if not items:
                    continue
                with st.expander(f"{group_name} ({len(items)})", expanded=(group_name == "Core Variant")):
                    for path in items:
                        _render_field(path, catalog.get(path))

            with st.expander("YAML blocks (features / policies / graph_feature_mapping)", expanded=True):
                st.session_state["yaml_features_text"] = _render_yaml_editor(
                    "mapping.features (or root features fallback)",
                    key="yaml_features_widget",
                    value=st.session_state.get("yaml_features_text", "[]"),
                    height=460,
                )
                st.session_state["yaml_policies_text"] = _render_yaml_editor(
                    "policies",
                    key="yaml_policies_widget",
                    value=st.session_state.get("yaml_policies_text", "{}"),
                    height=420,
                )
                st.session_state["graph_mapping_enabled"] = st.checkbox(
                    "mapping.graph_feature_mapping.enabled",
                    value=bool(st.session_state.get("graph_mapping_enabled", False)),
                    key="graph_mapping_enabled_widget",
                )
                gateway_options = ["preserve", "collapse_for_prediction"]
                current_gateway = str(st.session_state.get("graph_gateway_mode", "collapse_for_prediction")).strip()
                gateway_index = gateway_options.index(current_gateway) if current_gateway in gateway_options else 1
                st.session_state["graph_gateway_mode"] = st.selectbox(
                    "gateways",
                    options=gateway_options,
                    index=gateway_index,
                    help=(
                        "mapping.graph_feature_mapping.topology_projection.gateway_mode. "
                        "Use collapse_for_prediction for XES logs without gateway events; preserve for Camunda/BPMN runtime topology."
                    ),
                    key="graph_gateway_mode_widget",
                )
                st.session_state["yaml_graph_mapping_text"] = _render_yaml_editor(
                    "mapping.graph_feature_mapping",
                    key="yaml_graph_mapping_widget",
                    value=st.session_state.get("yaml_graph_mapping_text", "{}"),
                    height=520,
                )

    if st.session_state.get("base_config"):
        composed_cfg, parse_warnings = _compose_config()
        validation_warnings = _validate_config_against_catalog(composed_cfg, catalog)
        all_warnings = parse_warnings + validation_warnings

    if save_clicked and composed_cfg:
        try:
            saved_path, saved_name = _save_ui_run(composed_cfg)
            with tab_config:
                st.success(f"Saved: {saved_name}")
                st.caption(str(saved_path))
        except Exception as exc:
            with tab_config:
                st.error(f"Failed to save config: {exc}")

    with tab_run:
        if not composed_cfg:
            st.warning("Load and compose config first (tab Config).")
        else:
            _render_sticky_summary(composed_cfg, validation_warnings, parse_warnings)
            _render_mlflow_clone_panel(composed_cfg)

            r1, r2 = st.columns([1.2, 1.2])
            run_clicked = r1.button("Run", use_container_width=True, type="primary")
            stop_clicked = r2.button("Stop", use_container_width=True)

            if stop_clicked:
                _stop_run()

            if run_clicked:
                if all_warnings:
                    st.session_state["pending_run_cfg"] = deepcopy(composed_cfg)
                    st.session_state["pending_run_warnings"] = list(all_warnings)
                else:
                    ok, err = _start_run(composed_cfg)
                    if ok:
                        st.success("Run started.")
                    else:
                        st.warning(err)

            pending_cfg = st.session_state.get("pending_run_cfg")
            pending_warnings = st.session_state.get("pending_run_warnings", [])
            if isinstance(pending_cfg, dict) and pending_warnings:
                st.warning("Validation produced warnings. You can still run after explicit confirmation.")
                with st.expander(f"Warnings ({len(pending_warnings)})", expanded=True):
                    for item in pending_warnings[:80]:
                        st.write(f"- {item}")
                confirm = st.checkbox("Run despite warnings", value=False, key="confirm_run_warnings")
                cw1, cw2 = st.columns([1, 1])
                confirm_clicked = cw1.button("Confirm run", use_container_width=True)
                cancel_clicked = cw2.button("Cancel pending run", use_container_width=True)
                if cancel_clicked:
                    st.session_state["pending_run_cfg"] = None
                    st.session_state["pending_run_warnings"] = []
                if confirm_clicked and confirm:
                    ok, err = _start_run(pending_cfg)
                    if ok:
                        st.success("Run started.")
                        st.session_state["pending_run_cfg"] = None
                        st.session_state["pending_run_warnings"] = []
                    else:
                        st.warning(err)
                elif confirm_clicked and not confirm:
                    st.warning("Enable confirmation checkbox first.")
            elif all_warnings:
                with st.expander(f"Current warnings ({len(all_warnings)})", expanded=False):
                    for item in all_warnings[:80]:
                        st.write(f"- {item}")

        _render_run_status()

    process = st.session_state.get("run_process")
    if process is not None:
        time.sleep(1.0)
        _rerun()


if __name__ == "__main__":
    main()
