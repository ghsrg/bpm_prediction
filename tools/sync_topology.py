"""Bulk topology synchronization tool (Camunda BPMN or XES directories)."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Sequence

from src.adapters.ingestion.camunda_bpmn_adapter import CamundaBpmnAdapter
from src.adapters.ingestion.xes_adapter import XESAdapter
from src.application.services.bpmn_structure_parser_service import BpmnStructureParserService
from src.application.services.topology_extractor_service import TopologyExtractorService
from src.cli import _inject_dataset_name_mapping, _resolve_trace_adapter_kind, load_yaml_config
from src.infrastructure.repositories.file_based_knowledge_graph_repository import (
    FileBasedKnowledgeGraphRepository,
)
from src.infrastructure.repositories.knowledge_graph_repository_factory import (
    build_knowledge_graph_repository,
    get_knowledge_graph_settings,
)


def _serialize_summary(payload: Dict[str, Any]) -> Dict[str, Any]:
    serializable: Dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, Path):
            serializable[key] = str(value)
            continue
        if isinstance(value, list):
            serializable[key] = [
                str(item) if isinstance(item, Path) else item for item in value
            ]
            continue
        serializable[key] = value
    return serializable


def _normalize_tenant(value: Any) -> str:
    text = str(value).strip() if value is not None else ""
    if not text:
        return ""
    if text.lower() in {"nan", "none", "null", "<na>", "nat"}:
        return ""
    return text


def _compose_namespace(process_key: str, tenant_id: str) -> str:
    return f"{process_key}@{tenant_id}" if tenant_id else process_key


def _collect_xes_files(data_cfg: Dict[str, Any]) -> List[Path]:
    log_dir_raw = str(data_cfg.get("log_dir", "")).strip()
    log_path_raw = str(data_cfg.get("log_path", "")).strip()
    candidate_raw = log_dir_raw or log_path_raw
    if not candidate_raw:
        raise ValueError("XES sync requires data.log_dir or data.log_path.")
    candidate = Path(candidate_raw)
    if candidate.is_file():
        return [candidate.resolve()]
    if not candidate.exists() or not candidate.is_dir():
        raise ValueError(f"XES sync source path does not exist or is not a directory: {candidate}")

    files = sorted(
        path.resolve()
        for path in candidate.rglob("*")
        if path.is_file() and path.suffix.lower() == ".xes"
    )
    if not files:
        raise ValueError(f"No .xes files found in directory: {candidate}")
    return files


def _sync_xes_logs(
    *,
    config: Dict[str, Any],
    knowledge_repo: Any,
) -> Dict[str, Any]:
    logger = logging.getLogger(__name__)
    data_cfg = config.get("data", {})
    mapping_raw = config.get("mapping", {})
    xes_files = _collect_xes_files(data_cfg)
    adapter = XESAdapter()

    datasets_synced = 0
    total_files = len(xes_files)
    total_traces = 0
    per_dataset: List[Dict[str, Any]] = []

    for file_path in xes_files:
        dataset_name = file_path.stem
        mapping_cfg = _inject_dataset_name_mapping(mapping_raw, dataset_name)
        traces = list(adapter.read(str(file_path), mapping_cfg))
        total_traces += len(traces)
        if not traces:
            logger.warning("No traces parsed from XES file: %s", file_path)
            per_dataset.append(
                {
                    "dataset_name": dataset_name,
                    "source_file": str(file_path),
                    "traces": 0,
                    "versions_saved": [],
                    "status": "warning_empty",
                }
            )
            continue

        extractor = TopologyExtractorService(
            knowledge_port=knowledge_repo,
            process_name=dataset_name,
        )
        extractor.extract_from_logs(traces, process_name=dataset_name)
        versions = knowledge_repo.list_versions(process_name=dataset_name)
        datasets_synced += 1
        per_dataset.append(
            {
                "dataset_name": dataset_name,
                "source_file": str(file_path),
                "traces": len(traces),
                "versions_saved": versions,
                "status": "ok",
            }
        )

    return {
        "status": "ok",
        "adapter": "xes",
        "source_mode": "directory" if total_files > 1 else "single_file",
        "files_scanned": total_files,
        "datasets_synced": datasets_synced,
        "total_traces": total_traces,
        "details": per_dataset,
    }


def _sync_camunda_bpmn(
    *,
    config: Dict[str, Any],
    knowledge_repo: Any,
) -> Dict[str, Any]:
    logger = logging.getLogger(__name__)
    mapping_cfg = config.get("mapping", {})
    if not isinstance(mapping_cfg, dict):
        mapping_cfg = {}
    camunda_cfg_raw = mapping_cfg.get("camunda_adapter", {})
    camunda_cfg = dict(camunda_cfg_raw) if isinstance(camunda_cfg_raw, dict) else {}
    structure_raw = camunda_cfg.get("structure", {})
    structure_cfg = dict(structure_raw) if isinstance(structure_raw, dict) else {}

    parser_mode = str(structure_cfg.get("parser_mode", "recover")).strip().lower() or "recover"
    subprocess_mode = str(
        structure_cfg.get("subprocess_mode", "flattened-no-subprocess-node")
    ).strip() or "flattened-no-subprocess-node"
    call_cfg_raw = structure_cfg.get("call_activity", {})
    call_cfg = dict(call_cfg_raw) if isinstance(call_cfg_raw, dict) else {}
    inference_fallback_strategy = str(
        call_cfg.get("inference_fallback_strategy", "use_aggregated_stats")
    ).strip() or "use_aggregated_stats"

    process_filters_raw = camunda_cfg.get("process_filters")
    process_filters: List[str] = []
    if isinstance(process_filters_raw, list):
        process_filters = [str(item).strip() for item in process_filters_raw if str(item).strip()]
    tenant_filters_raw = camunda_cfg.get("tenant_filters")
    tenant_filters: List[str] = []
    if isinstance(tenant_filters_raw, list):
        tenant_filters = [str(item).strip() for item in tenant_filters_raw if str(item).strip()]
    tenant_id_cfg = _normalize_tenant(camunda_cfg.get("tenant_id"))
    if tenant_id_cfg and tenant_id_cfg not in tenant_filters:
        tenant_filters.append(tenant_id_cfg)

    adapter = CamundaBpmnAdapter(camunda_cfg)
    parser = BpmnStructureParserService(
        subprocess_mode=subprocess_mode,
        parser_mode=parser_mode,
        inference_fallback_strategy=inference_fallback_strategy,
    )

    catalog = adapter.fetch_procdef_catalog(
        process_name=None,
        process_filters=process_filters or None,
    )
    if not catalog:
        return {
            "status": "ok",
            "adapter": "camunda",
            "structure_source": "bpmn",
            "total_procdefs": 0,
            "parsed_procdefs": 0,
            "quarantined_procdefs": 0,
            "missing_bpmn_count": 0,
            "details": [],
            "quarantine_error_codes": [],
            "quarantine_records": [],
        }

    parsed = 0
    quarantined = 0
    missing_bpmn_count = 0
    quarantine_records: List[Dict[str, Any]] = []
    quarantine_error_codes: set[str] = set()
    per_process_stats: Dict[str, Dict[str, Any]] = {}

    for definition in catalog:
        proc_def_id = str(definition.get("proc_def_id", "")).strip()
        proc_def_key = str(definition.get("proc_def_key", "")).strip() or "unknown_process"
        tenant_id = _normalize_tenant(definition.get("tenant_id")) or tenant_id_cfg
        process_namespace = _compose_namespace(proc_def_key, tenant_id)
        if not proc_def_id:
            quarantined += 1
            record = {
                "proc_def_id": "",
                "proc_def_key": proc_def_key,
                "tenant_id": tenant_id or None,
                "version": "v0",
                "error_code": "missing_proc_def_id",
                "error_message": "Catalog row does not contain proc_def_id.",
                "source_hint": "sync_topology.catalog_row_without_proc_def_id",
            }
            quarantine_records.append(record)
            quarantine_error_codes.add("missing_proc_def_id")
            continue

        xml_content = adapter.fetch_bpmn_xml(proc_def_id)
        if not xml_content:
            missing_bpmn_count += 1
            logger.warning(
                "No BPMN payload found for proc_def_id=%s proc_def_key=%s. Skipping as quarantine.",
                proc_def_id,
                proc_def_key,
            )

        enriched_definition = dict(definition)
        enriched_definition["bpmn_xml_content"] = xml_content
        result = parser.parse_definition(
            definition=enriched_definition,
            catalog=catalog,
            process_name=proc_def_key,
            process_filters=process_filters or None,
        )
        if result.dto is None:
            quarantined += 1
            if result.quarantine_record:
                quarantine_records.append(result.quarantine_record)
                code = str(result.quarantine_record.get("error_code", "")).strip().lower()
                if code:
                    quarantine_error_codes.add(code)
            continue

        dto = result.dto
        knowledge_repo.save_process_structure(
            version=dto.version,
            dto=dto,
            process_name=process_namespace,
        )
        parsed += 1
        process_entry = per_process_stats.setdefault(
            process_namespace,
            {
                "process_name": proc_def_key,
                "namespace": process_namespace,
                "tenant_id": tenant_id or None,
                "procdefs_total": 0,
                "procdefs_parsed": 0,
                "versions_saved": set(),
            },
        )
        process_entry["procdefs_total"] += 1
        process_entry["procdefs_parsed"] += 1
        process_entry["versions_saved"].add(dto.version)

    for definition in catalog:
        proc_def_key = str(definition.get("proc_def_key", "")).strip() or "unknown_process"
        tenant_id = _normalize_tenant(definition.get("tenant_id")) or tenant_id_cfg
        process_namespace = _compose_namespace(proc_def_key, tenant_id)
        process_entry = per_process_stats.setdefault(
            process_namespace,
            {
                "process_name": proc_def_key,
                "namespace": process_namespace,
                "tenant_id": tenant_id or None,
                "procdefs_total": 0,
                "procdefs_parsed": 0,
                "versions_saved": set(),
            },
        )
        process_entry["procdefs_total"] += 1

    details: List[Dict[str, Any]] = []
    for process_name in sorted(per_process_stats.keys()):
        item = per_process_stats[process_name]
        versions = sorted(str(v) for v in item["versions_saved"])
        details.append(
            {
                "process_name": process_name,
                "namespace": item.get("namespace"),
                "tenant_id": item.get("tenant_id"),
                "procdefs_total": int(item["procdefs_total"]),
                "procdefs_parsed": int(item["procdefs_parsed"]),
                "versions_saved": versions,
            }
        )

    return {
        "status": "ok",
        "adapter": "camunda",
        "structure_source": "bpmn",
        "bpmn_source": str(structure_cfg.get("bpmn_source", "files")).strip().lower() or "files",
        "tenant_filters": tenant_filters,
        "total_procdefs": len(catalog),
        "parsed_procdefs": parsed,
        "quarantined_procdefs": quarantined,
        "missing_bpmn_count": missing_bpmn_count,
        "details": details,
        "quarantine_error_codes": sorted(quarantine_error_codes),
        "quarantine_records": quarantine_records,
    }


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(
        description="Bulk sync topology structures into knowledge backend."
    )
    parser.add_argument("--config", required=True, help="YAML config path.")
    parser.add_argument(
        "--out",
        default="outputs/sync_topology_summary.json",
        help="Output JSON summary path.",
    )
    args = parser.parse_args(argv)

    config = load_yaml_config(args.config)
    mapping_cfg = config.get("mapping", {})
    adapter_kind = _resolve_trace_adapter_kind(mapping_cfg if isinstance(mapping_cfg, dict) else {})
    knowledge_cfg = get_knowledge_graph_settings(config)
    knowledge_repo = build_knowledge_graph_repository(config)

    if adapter_kind == "camunda":
        summary = _sync_camunda_bpmn(config=config, knowledge_repo=knowledge_repo)
    else:
        summary = _sync_xes_logs(config=config, knowledge_repo=knowledge_repo)

    storage_path = ""
    if isinstance(knowledge_repo, FileBasedKnowledgeGraphRepository):
        storage_path = str(knowledge_repo.base_dir)
    summary["knowledge_backend"] = knowledge_cfg.get("backend", "in_memory")
    summary["knowledge_storage_path"] = storage_path

    out_path = Path(str(args.out)).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(_serialize_summary(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Topology sync finished: backend=%s", summary.get("knowledge_backend"))
    logger.info("Summary saved to %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
