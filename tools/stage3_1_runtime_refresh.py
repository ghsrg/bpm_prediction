"""Stage 3.1 runner for Camunda runtime ingestion (files or MSSQL)."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import json
from pathlib import Path
import sys
from typing import Any, Dict

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.adapters.ingestion.camunda_runtime_adapter import CamundaRuntimeAdapter
from src.application.services.instance_graph_assembler_service import InstanceGraphAssemblerService
from src.domain.entities.process_structure import ProcessStructureDTO
from src.infrastructure.config.yaml_loader import load_yaml_with_includes
from src.infrastructure.repositories.in_memory_instance_graph_repository import InMemoryInstanceGraphRepository
from src.infrastructure.repositories.in_memory_networkx_repository import InMemoryNetworkXRepository


def _parse_iso(value: str | None) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return datetime.fromisoformat(text)


def _load_config(config_path: str) -> Dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    return load_yaml_with_includes(path)


def _seed_fallback_structure(knowledge_repo: InMemoryNetworkXRepository, config: Dict[str, Any], process_name: str, version_key: str) -> None:
    fallback_cfg = config.get("fallback_structure", {})
    edges = fallback_cfg.get("allowed_edges", [])
    normalized = []
    for item in edges:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            normalized.append((str(item[0]), str(item[1])))
    if not normalized:
        return
    knowledge_repo.save_process_structure(
        version=version_key,
        process_name=process_name,
        dto=ProcessStructureDTO(version=version_key, allowed_edges=normalized),
    )


def _serialize_result(payload: Dict[str, Any]) -> Dict[str, Any]:
    def _serialize_value(value: Any) -> Any:
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, dict):
            return {str(k): _serialize_value(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_serialize_value(item) for item in value]
        return value

    return _serialize_value(payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 3.1 Camunda runtime refresh")
    parser.add_argument("--config", required=True, help="Path to Stage 3.1 YAML config")
    parser.add_argument("--out", default="outputs/stage3_1_refresh_result.json", help="Result JSON path")
    args = parser.parse_args()

    config = _load_config(args.config)
    data_cfg = config.get("data", {})
    camunda_cfg = config.get("camunda", {})
    if not camunda_cfg:
        mapping_cfg = config.get("mapping", {})
        camunda_cfg = dict(mapping_cfg.get("camunda_adapter", {})) if isinstance(mapping_cfg, dict) else {}
    runtime_cfg = dict(camunda_cfg.get("runtime", {}))
    if not runtime_cfg:
        runtime_cfg = dict(camunda_cfg)

    process_name = str(
        data_cfg.get("process_name")
        or camunda_cfg.get("process_name")
        or data_cfg.get("dataset_name")
        or "default_process"
    ).strip() or "default_process"
    version_key = str(camunda_cfg.get("version_key", "")).strip()
    lookback_hours = int(camunda_cfg.get("lookback_hours", 24))
    now = datetime.utcnow()
    since = _parse_iso(camunda_cfg.get("since")) or (now - timedelta(hours=lookback_hours))
    until = _parse_iso(camunda_cfg.get("until")) or now

    adapter = CamundaRuntimeAdapter(runtime_cfg)
    knowledge_repo = InMemoryNetworkXRepository()
    _seed_fallback_structure(knowledge_repo, camunda_cfg, process_name, version_key)
    assembler = InstanceGraphAssemblerService(knowledge_port=knowledge_repo)
    instance_repo = InMemoryInstanceGraphRepository()

    events, diagnostics = adapter.fetch_historic_activity_events(process_name=process_name, version_key=version_key, since=since, until=until)
    execution_rows = adapter.fetch_runtime_execution_tree(
        process_name=process_name,
        version_key=version_key,
        depth_limit=int(camunda_cfg.get("execution_tree_depth_limit", 4)),
        since=since,
        until=until,
    )
    task_rows = adapter.fetch_historic_task_events(
        process_name=process_name,
        version_key=version_key,
        since=since,
        until=until,
    )
    variable_rows = adapter.fetch_multi_instance_variables(process_name=process_name, version_key=version_key, since=since, until=until)
    process_variable_rows = adapter.fetch_process_variables(process_name=process_name, version_key=version_key, since=since, until=until)
    process_instance_links = adapter.fetch_process_instance_links(
        process_name=process_name,
        version_key=version_key,
        since=since,
        until=until,
    )
    identity_rows = adapter.fetch_identity_links(process_name=process_name, version_key=version_key, since=since, until=until)

    result = assembler.build(
        process_name=process_name,
        version_key=version_key,
        events=events,
        execution_rows=execution_rows,
        variables_rows=variable_rows,
        identity_rows=identity_rows,
        diagnostics=diagnostics,
        config=camunda_cfg,
        task_rows=task_rows,
        process_variables_rows=process_variable_rows,
        process_instance_links=process_instance_links,
    )

    instance_repo.save_instance_events(process_name, version_key, events)
    instance_repo.save_instance_graph(process_name, version_key, result.graph)
    instance_repo.save_instance_projection(process_name, version_key, result.projection)
    instance_repo.set_last_watermark(process_name, version_key, until)

    output_payload = {
        "process_name": process_name,
        "version_key": version_key,
        "runtime_source": runtime_cfg.get("runtime_source", "files"),
        "mode": result.mode,
        "event_count": len(events),
        "execution_rows": len(execution_rows),
        "task_rows": len(task_rows),
        "variable_rows": len(variable_rows),
        "process_variable_rows": len(process_variable_rows),
        "identity_rows": len(identity_rows),
        "process_instance_links": len(process_instance_links),
        "diagnostics": result.diagnostics.model_dump(),
        "graph_meta": result.graph.get("metadata", {}),
        "projection_keys": list(result.projection.keys()),
    }
    serialized = _serialize_result(output_payload)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(serialized, ensure_ascii=True, indent=2), encoding="utf-8")

    print("Stage 3.1 refresh completed.")
    print(f"Process: {process_name} | Version: {version_key}")
    print(f"Mode: {result.mode} | Events: {len(events)}")
    print(f"Diagnostics: coverage={result.diagnostics.history_coverage_percent}, fallback={result.diagnostics.fallback_triggered}")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
