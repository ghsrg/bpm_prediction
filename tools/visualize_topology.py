"""CLI utility for visual sanity-check of extracted process topology."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from src.adapters.ingestion.camunda_trace_adapter import CamundaTraceAdapter
from src.adapters.ingestion.xes_adapter import XESAdapter
from src.application.services.topology_extractor_service import TopologyExtractorService
from src.cli import load_yaml_config
from src.domain.entities.raw_trace import RawTrace
from src.infrastructure.repositories.in_memory_networkx_repository import InMemoryNetworkXRepository


def _prepare_train_traces(traces: Sequence[RawTrace], experiment_cfg: Dict[str, Any]) -> List[RawTrace]:
    """Apply train-side cascade: temporal sort -> macro train split -> fraction."""
    traces_with_events = [trace for trace in traces if trace.events]
    split_strategy = str(experiment_cfg.get("split_strategy", "temporal")).strip().lower()
    if split_strategy == "time":
        split_strategy = "temporal"
    if split_strategy not in {"temporal", "none"}:
        raise ValueError("experiment.split_strategy must be 'temporal' or 'none'.")

    if split_strategy == "temporal":
        ordered = sorted(traces_with_events, key=lambda trace: trace.events[0].timestamp)
    else:
        ordered = list(traces_with_events)

    train_ratio = float(experiment_cfg.get("train_ratio", 0.7))
    if train_ratio < 0.0 or train_ratio > 1.0:
        raise ValueError("experiment.train_ratio must be within [0.0, 1.0].")
    split_idx = int(len(ordered) * train_ratio)
    macro = ordered[:split_idx]

    fraction = float(experiment_cfg.get("fraction", 1.0))
    if fraction <= 0.0 or fraction > 1.0:
        raise ValueError("experiment.fraction must be within (0.0, 1.0].")
    if fraction >= 1.0:
        return macro
    return macro[: int(len(macro) * fraction)]


def _resolve_dataset_name(data_cfg: Dict[str, Any], fallback_path: str) -> str:
    """Resolve dataset name used as process namespace and fallback version key."""
    candidates = [
        data_cfg.get("dataset_name"),
        data_cfg.get("dataset_label"),
        data_cfg.get("process_name"),
    ]
    fallback = str(fallback_path).strip()
    if fallback:
        candidates.append(Path(fallback).stem)
    for candidate in candidates:
        value = str(candidate).strip() if candidate is not None else ""
        if value:
            return value
    return "default_dataset"


def _build_mapping_with_dataset(mapping_cfg: Dict[str, Any], dataset_name: str) -> Dict[str, Any]:
    """Inject dataset name into adapter mapping blocks."""
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
    return result


def _resolve_adapter_kind(mapping_cfg: Dict[str, Any]) -> str:
    adapter = str(mapping_cfg.get("adapter", "")).strip().lower()
    if adapter:
        return adapter
    if isinstance(mapping_cfg.get("camunda_adapter"), dict):
        return "camunda"
    return "xes"


def _build_camunda_mapping_from_legacy(cfg: Dict[str, Any], data_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Build mapping block from legacy Stage 3.1 config where `camunda` is top-level."""
    camunda_cfg = cfg.get("camunda", {})
    if not isinstance(camunda_cfg, dict):
        return {}
    process_name = str(
        data_cfg.get("dataset_name")
        or data_cfg.get("dataset_label")
        or data_cfg.get("process_name")
        or camunda_cfg.get("process_name")
        or "default_process"
    ).strip() or "default_process"
    camunda_adapter_cfg = dict(camunda_cfg)
    camunda_adapter_cfg.setdefault("process_name", process_name)
    runtime_cfg = camunda_cfg.get("runtime", {})
    if isinstance(runtime_cfg, dict):
        camunda_adapter_cfg["runtime"] = dict(runtime_cfg)
    return {
        "adapter": "camunda",
        "camunda_adapter": camunda_adapter_cfg,
    }


def _load_train_traces_from_config(config_path: str) -> Tuple[List[RawTrace], str]:
    """Load traces from experiment config and return train-side subset with process name."""
    cfg = load_yaml_config(config_path)
    data_cfg = cfg.get("data", {})
    mapping_cfg = cfg.get("mapping", {})
    if not isinstance(mapping_cfg, dict):
        mapping_cfg = {}
    if not mapping_cfg and isinstance(cfg.get("camunda"), dict):
        mapping_cfg = _build_camunda_mapping_from_legacy(cfg, data_cfg)
    experiment_cfg = cfg.get("experiment", {})

    adapter_kind = _resolve_adapter_kind(mapping_cfg)
    log_path = str(data_cfg.get("log_path", "")).strip()
    if adapter_kind == "xes" and not log_path:
        raise ValueError("Config must define data.log_path for topology visualization with XES adapter.")
    if not log_path:
        log_path = "__adapter_source__"
    dataset_name = _resolve_dataset_name(data_cfg, log_path)
    mapping_cfg = _build_mapping_with_dataset(mapping_cfg, dataset_name)
    adapter = CamundaTraceAdapter() if adapter_kind == "camunda" else XESAdapter()
    traces = list(adapter.read(log_path, mapping_cfg))
    if not traces and adapter_kind == "camunda":
        camunda_cfg_raw = mapping_cfg.get("camunda_adapter", {})
        camunda_cfg = dict(camunda_cfg_raw) if isinstance(camunda_cfg_raw, dict) else {}
        has_explicit_bounds = bool(str(camunda_cfg.get("since", "")).strip() or str(camunda_cfg.get("until", "")).strip())
        lookback_hours = int(camunda_cfg.get("lookback_hours", 0) or 0)
        if lookback_hours > 0 and not has_explicit_bounds:
            retry_mapping = dict(mapping_cfg)
            retry_camunda_cfg = dict(camunda_cfg)
            retry_camunda_cfg["lookback_hours"] = 0
            retry_mapping["camunda_adapter"] = retry_camunda_cfg
            print(
                f"[Warning] No traces found with lookback_hours={lookback_hours}. "
                "Retrying topology extraction with full history (lookback_hours=0)."
            )
            traces = list(adapter.read(log_path, retry_mapping))
    return _prepare_train_traces(traces, experiment_cfg), dataset_name


def _load_traces_from_data(data_path: str) -> Tuple[List[RawTrace], str]:
    """Load full trace list directly from XES path (no config cascade)."""
    dataset_name = Path(data_path).stem or "default_dataset"
    mapping_cfg = _build_mapping_with_dataset({}, dataset_name)
    return list(XESAdapter().read(data_path, mapping_cfg)), dataset_name


def _resolve_plot_version(requested_version: str, available_versions: List[str]) -> str:
    """Resolve requested version with smart fallback for single-version datasets."""
    if not available_versions:
        raise ValueError("No traces or transitions found in the dataset")

    if requested_version in available_versions:
        return requested_version

    if len(available_versions) == 1:
        fallback = available_versions[0]
        print(
            f"[Warning] Requested version '{requested_version}' not found. "
            f"Using available version '{fallback}'."
        )
        return fallback

    raise ValueError(f"Version '{requested_version}' not found. Available versions: {available_versions}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Visualize extracted process topology by process version.")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--config", help="Path to experiment YAML config.")
    source_group.add_argument("--data", help="Path to source XES file.")
    parser.add_argument("--version", default="1", help="Process version (kappa) to render. Default: 1")
    parser.add_argument("--out", default=None, help="Optional output PNG path. If omitted, opens interactive window.")
    parser.add_argument(
        "--min-freq",
        type=int,
        default=1,
        help="Minimum edge frequency to keep in DFG (filters rare transitions). Default: 1",
    )
    parser.add_argument(
        "--renderer",
        choices=["graphviz", "pm4py"],
        default="graphviz",
        help="Topology renderer. 'graphviz' supports typed colors/labels. Default: graphviz",
    )
    parser.add_argument(
        "--label-mode",
        choices=["id", "name", "id+name", "id+name+type"],
        default="id+name+type",
        help="Node label mode for graphviz renderer. Default: id+name+type",
    )
    parser.add_argument(
        "--typed-colors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable activity-type color mapping for graphviz renderer. Default: true",
    )

    args = parser.parse_args(argv)

    if args.config:
        train_traces, process_name = _load_train_traces_from_config(args.config)
    else:
        train_traces, process_name = _load_traces_from_data(args.data)

    if not train_traces:
        raise ValueError("No train traces found for topology extraction.")

    repository = InMemoryNetworkXRepository()
    service = TopologyExtractorService(knowledge_port=repository, process_name=process_name)
    service.fit(train_traces, process_name=process_name)
    selected_version = _resolve_plot_version(str(args.version), service.available_versions)
    try:
        service.plot_topology(
        version=selected_version,
        process_name=process_name,
        save_path=args.out,
        min_edge_freq=args.min_freq,
        renderer=args.renderer,
        label_mode=args.label_mode,
        typed_colors=bool(args.typed_colors),
    )
    except ValueError as exc:
        message = str(exc)
        if "No transitions found for process version" in message:
            print(
                f"[Error] Not enough transitions to build a graph with --min-freq={args.min_freq}. "
                "Lower --min-freq or use a larger sample."
            )
            print(f"[Details] {message}")
            return 2
        raise

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
