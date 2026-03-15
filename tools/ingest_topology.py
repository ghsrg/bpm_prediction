"""Dedicated topology ingestion entrypoint (decoupled from training)."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Sequence

from src.application.services.topology_extractor_service import TopologyExtractorService
from src.cli import (
    _apply_fraction,
    _build_trace_adapter,
    _inject_dataset_name_mapping,
    _parse_split_ratio,
    _resolve_dataset_name,
    _resolve_trace_adapter_kind,
    _strict_temporal_split,
    load_yaml_config,
)
from src.infrastructure.repositories.file_based_knowledge_graph_repository import (
    FileBasedKnowledgeGraphRepository,
)
from src.infrastructure.repositories.knowledge_graph_repository_factory import (
    build_knowledge_graph_repository,
    get_knowledge_graph_settings,
)


def _resolve_ingest_split(
    *,
    cli_value: str | None,
    config_value: str,
) -> str:
    split = (cli_value or config_value or "train").strip().lower()
    if split not in {"train", "full"}:
        raise ValueError("ingest split must be either 'train' or 'full'.")
    return split


def _serialize_summary(payload: Dict[str, Any]) -> Dict[str, Any]:
    serializable: Dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, Path):
            serializable[key] = str(value)
            continue
        serializable[key] = value
    return serializable


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Ingest and persist topology artifacts (no training).")
    parser.add_argument("--config", required=True, help="YAML config path.")
    parser.add_argument(
        "--split",
        choices=["train", "full"],
        default=None,
        help="Override ingestion split source.",
    )
    parser.add_argument(
        "--out",
        default="outputs/ingest_topology_summary.json",
        help="Optional output JSON summary path.",
    )
    args = parser.parse_args(argv)

    config = load_yaml_config(args.config)
    data_cfg = config.get("data", {})
    experiment_cfg = config.get("experiment", {})
    mapping_cfg_raw = config.get("mapping", {})
    adapter = _build_trace_adapter(mapping_cfg_raw)
    adapter_kind = _resolve_trace_adapter_kind(mapping_cfg_raw)

    log_path = str(data_cfg.get("log_path", "")).strip()
    if adapter_kind == "xes" and not log_path:
        raise ValueError("Config must define data.log_path for XES adapter.")
    if not log_path:
        log_path = "__adapter_source__"

    dataset_name = _resolve_dataset_name(data_cfg, log_path)
    mapping_cfg = _inject_dataset_name_mapping(mapping_cfg_raw, dataset_name)

    fraction = float(experiment_cfg.get("fraction", 1.0))
    split_strategy = str(experiment_cfg.get("split_strategy", "temporal")).strip().lower()
    if split_strategy == "time":
        split_strategy = "temporal"
    split_ratio = _parse_split_ratio(experiment_cfg)

    logger.info("Reading traces via adapter=%s for topology ingestion...", adapter.__class__.__name__)
    traces = list(adapter.read(log_path, mapping_cfg))
    traces = _apply_fraction(traces, fraction)
    logger.info("Traces read=%d after fraction=%.4f", len(traces), fraction)

    knowledge_cfg = get_knowledge_graph_settings(config)
    ingest_split = _resolve_ingest_split(
        cli_value=args.split,
        config_value=str(knowledge_cfg.get("ingest_split", "train")),
    )
    if ingest_split == "train":
        train_traces, _, _ = _strict_temporal_split(traces, split_ratio, split_strategy)
        selected_traces = train_traces
    else:
        selected_traces = list(traces)
    logger.info(
        "Ingestion split=%s selected_traces=%d (total=%d)",
        ingest_split,
        len(selected_traces),
        len(traces),
    )

    knowledge_repo = build_knowledge_graph_repository(config)
    extractor = TopologyExtractorService(knowledge_port=knowledge_repo, process_name=dataset_name)
    extractor.extract_from_logs(selected_traces, process_name=dataset_name)
    versions = knowledge_repo.list_versions(process_name=dataset_name)

    storage_path = ""
    if isinstance(knowledge_repo, FileBasedKnowledgeGraphRepository):
        storage_path = str(knowledge_repo.base_dir)
    summary = {
        "status": "ok",
        "adapter": adapter_kind,
        "dataset_name": dataset_name,
        "ingest_split": ingest_split,
        "total_traces": len(traces),
        "selected_traces": len(selected_traces),
        "versions_saved": versions,
        "knowledge_backend": knowledge_cfg.get("backend", "in_memory"),
        "knowledge_storage_path": storage_path,
    }

    out_path = Path(str(args.out)).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(_serialize_summary(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    logger.info("Topology ingestion finished. versions=%s", versions)
    logger.info("Summary saved to %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
