"""Dedicated topology ingestion entrypoint (decoupled from training)."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Sequence

from src.adapters.ingestion.camunda_bpmn_adapter import CamundaBpmnAdapter
from src.application.services.bpmn_structure_parser_service import BpmnStructureParserService
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


def _resolve_camunda_structure_config(mapping_cfg: Dict[str, Any]) -> Dict[str, Any]:
    camunda_cfg_raw = mapping_cfg.get("camunda_adapter", {})
    camunda_cfg = dict(camunda_cfg_raw) if isinstance(camunda_cfg_raw, dict) else {}
    structure_raw = camunda_cfg.get("structure", {})
    structure_cfg = dict(structure_raw) if isinstance(structure_raw, dict) else {}
    return {
        "camunda": camunda_cfg,
        "structure": structure_cfg,
    }


def _should_ingest_camunda_bpmn(adapter_kind: str, mapping_cfg: Dict[str, Any]) -> bool:
    if adapter_kind != "camunda":
        return False
    cfg = _resolve_camunda_structure_config(mapping_cfg)
    structure_cfg = cfg["structure"]
    source = str(structure_cfg.get("source", "logs")).strip().lower() or "logs"
    structure_from_logs = bool(structure_cfg.get("structure_from_logs", False))
    return source == "bpmn" and not structure_from_logs


def _ingest_camunda_bpmn(
    *,
    mapping_cfg: Dict[str, Any],
    dataset_name: str,
    knowledge_repo: Any,
) -> Dict[str, Any]:
    cfg = _resolve_camunda_structure_config(mapping_cfg)
    camunda_cfg = cfg["camunda"]
    structure_cfg = cfg["structure"]
    process_name = str(
        camunda_cfg.get("process_name")
        or mapping_cfg.get("dataset_name")
        or dataset_name
        or "default_process"
    ).strip() or "default_process"
    process_filters_raw = camunda_cfg.get("process_filters")
    process_filters: List[str] = []
    if isinstance(process_filters_raw, list):
        process_filters = [str(item).strip() for item in process_filters_raw if str(item).strip()]
    if not process_filters:
        process_filters = [process_name]

    parser_mode = str(structure_cfg.get("parser_mode", "recover")).strip().lower() or "recover"
    subprocess_mode = str(
        structure_cfg.get("subprocess_mode", "flattened-no-subprocess-node")
    ).strip() or "flattened-no-subprocess-node"
    call_cfg_raw = structure_cfg.get("call_activity", {})
    call_cfg = dict(call_cfg_raw) if isinstance(call_cfg_raw, dict) else {}
    inference_fallback_strategy = str(
        call_cfg.get("inference_fallback_strategy", "use_aggregated_stats")
    ).strip() or "use_aggregated_stats"

    adapter = CamundaBpmnAdapter(camunda_cfg)
    parser = BpmnStructureParserService(
        subprocess_mode=subprocess_mode,
        parser_mode=parser_mode,
        inference_fallback_strategy=inference_fallback_strategy,
    )

    catalog = adapter.fetch_procdef_catalog(
        process_name=process_name,
        process_filters=process_filters,
    )
    parsed = 0
    quarantined = 0
    quarantine_records: List[Dict[str, Any]] = []
    versions_saved: set[str] = set()

    for definition in catalog:
        proc_def_id = str(definition.get("proc_def_id", "")).strip()
        if not proc_def_id:
            quarantined += 1
            quarantine_records.append(
                {
                    "proc_def_id": "",
                    "proc_def_key": str(definition.get("proc_def_key", "")).strip(),
                    "error_code": "missing_proc_def_id",
                    "error_message": "Catalog row does not contain proc_def_id.",
                }
            )
            continue
        xml_content = adapter.fetch_bpmn_xml(proc_def_id)
        enriched_definition = dict(definition)
        enriched_definition["bpmn_xml_content"] = xml_content

        result = parser.parse_definition(
            definition=enriched_definition,
            catalog=catalog,
            process_name=process_name,
            process_filters=process_filters,
        )
        if result.dto is None:
            quarantined += 1
            if result.quarantine_record:
                quarantine_records.append(result.quarantine_record)
            continue

        dto = result.dto
        knowledge_repo.save_process_structure(
            version=dto.version,
            dto=dto,
            process_name=dataset_name,
        )
        versions_saved.add(dto.version)
        parsed += 1

    return {
        "adapter": "camunda",
        "structure_source": "bpmn",
        "dataset_name": dataset_name,
        "process_name": process_name,
        "process_filters": process_filters,
        "total_procdefs": len(catalog),
        "parsed_procdefs": parsed,
        "quarantined_procdefs": quarantined,
        "quarantine_records": quarantine_records,
        "versions_saved": sorted(versions_saved),
    }


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
    adapter_kind = _resolve_trace_adapter_kind(mapping_cfg_raw)

    log_path = str(data_cfg.get("log_path", "")).strip()
    if adapter_kind == "xes" and not log_path:
        raise ValueError("Config must define data.log_path for XES adapter.")
    if not log_path:
        log_path = "__adapter_source__"

    dataset_name = _resolve_dataset_name(data_cfg, log_path)
    mapping_cfg = _inject_dataset_name_mapping(mapping_cfg_raw, dataset_name)

    knowledge_cfg = get_knowledge_graph_settings(config)
    knowledge_repo = build_knowledge_graph_repository(config)
    if _should_ingest_camunda_bpmn(adapter_kind=adapter_kind, mapping_cfg=mapping_cfg):
        logger.info("Running Camunda BPMN structure ingestion (source=bpmn)...")
        result = _ingest_camunda_bpmn(
            mapping_cfg=mapping_cfg,
            dataset_name=dataset_name,
            knowledge_repo=knowledge_repo,
        )
        versions = list(result.get("versions_saved", []))
        ingest_split = "n/a"
        traces_count = 0
        selected_count = 0
        summary = {
            "status": "ok",
            "adapter": result.get("adapter", adapter_kind),
            "structure_source": result.get("structure_source", "bpmn"),
            "dataset_name": dataset_name,
            "process_name": result.get("process_name"),
            "process_filters": result.get("process_filters", []),
            "ingest_split": ingest_split,
            "total_traces": traces_count,
            "selected_traces": selected_count,
            "total_procdefs": int(result.get("total_procdefs", 0)),
            "parsed_procdefs": int(result.get("parsed_procdefs", 0)),
            "quarantined_procdefs": int(result.get("quarantined_procdefs", 0)),
            "versions_saved": versions,
            "knowledge_backend": knowledge_cfg.get("backend", "in_memory"),
            "knowledge_storage_path": "",
            "quarantine_records": result.get("quarantine_records", []),
        }
    else:
        adapter = _build_trace_adapter(mapping_cfg_raw)
        fraction = float(experiment_cfg.get("fraction", 1.0))
        split_strategy = str(experiment_cfg.get("split_strategy", "temporal")).strip().lower()
        if split_strategy == "time":
            split_strategy = "temporal"
        split_ratio = _parse_split_ratio(experiment_cfg)

        logger.info("Reading traces via adapter=%s for topology ingestion...", adapter.__class__.__name__)
        traces = list(adapter.read(log_path, mapping_cfg))
        traces = _apply_fraction(traces, fraction)
        logger.info("Traces read=%d after fraction=%.4f", len(traces), fraction)

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

        extractor = TopologyExtractorService(knowledge_port=knowledge_repo, process_name=dataset_name)
        extractor.extract_from_logs(selected_traces, process_name=dataset_name)
        versions = knowledge_repo.list_versions(process_name=dataset_name)

        summary = {
            "status": "ok",
            "adapter": adapter_kind,
            "structure_source": "logs",
            "dataset_name": dataset_name,
            "ingest_split": ingest_split,
            "total_traces": len(traces),
            "selected_traces": len(selected_traces),
            "versions_saved": versions,
            "knowledge_backend": knowledge_cfg.get("backend", "in_memory"),
            "knowledge_storage_path": "",
        }

    storage_path = ""
    if isinstance(knowledge_repo, FileBasedKnowledgeGraphRepository):
        storage_path = str(knowledge_repo.base_dir)
    summary["knowledge_storage_path"] = storage_path

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
