"""CLI utility for visual sanity-check of extracted process topology."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

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
    for candidate in (data_cfg.get("dataset_name"), data_cfg.get("dataset_label"), Path(fallback_path).stem):
        value = str(candidate).strip() if candidate is not None else ""
        if value:
            return value
    return "default_dataset"


def _build_mapping_with_dataset(mapping_cfg: Dict[str, Any], dataset_name: str) -> Dict[str, Any]:
    """Inject dataset name into xes_adapter mapping block."""
    result = dict(mapping_cfg)
    xes_cfg_raw = result.get("xes_adapter", {})
    xes_cfg = dict(xes_cfg_raw) if isinstance(xes_cfg_raw, dict) else {}
    xes_cfg.setdefault("dataset_name", dataset_name)
    result["xes_adapter"] = xes_cfg
    return result


def _load_train_traces_from_config(config_path: str) -> Tuple[List[RawTrace], str]:
    """Load traces from experiment config and return train-side subset with process name."""
    cfg = load_yaml_config(config_path)
    data_cfg = cfg.get("data", {})
    mapping_cfg = cfg.get("mapping", {})
    experiment_cfg = cfg.get("experiment", {})

    log_path = str(data_cfg.get("log_path", "")).strip()
    if not log_path:
        raise ValueError("Config must define data.log_path for topology visualization.")
    dataset_name = _resolve_dataset_name(data_cfg, log_path)
    mapping_cfg = _build_mapping_with_dataset(mapping_cfg, dataset_name)

    traces = list(XESAdapter().read(log_path, mapping_cfg))
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
    service.plot_topology(
        version=selected_version,
        process_name=process_name,
        save_path=args.out,
        min_edge_freq=args.min_freq,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
