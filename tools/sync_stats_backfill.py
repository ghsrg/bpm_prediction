"""Backfill wrapper for periodic sync-stats runs over historical as-of points."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from tqdm import tqdm

from src.adapters.ingestion.camunda_runtime_adapter import CamundaRuntimeAdapter
from src.adapters.ingestion.xes_adapter import XESAdapter
from src.cli import load_yaml_config
from src.infrastructure.repositories.knowledge_graph_repository_factory import build_knowledge_graph_repository
from tools.sync_stats import (
    _derive_process_tenant_filters,
    _event_ts,
    _filter_by_tenant,
    _inject_xes_dataset_name,
    _iter_xes_paths,
    _normalize_text,
    _resolve_runtime_config,
    _select_process_names,
    _split_namespace,
    main as sync_stats_main,
)


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BackfillStep:
    mode: str
    days: int


def _parse_iso_utc(value: str) -> datetime:
    text = str(value).strip()
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _resolve_step(args: argparse.Namespace) -> BackfillStep:
    if args.step_days is not None:
        days = int(args.step_days)
        if days <= 0:
            raise ValueError("--step-days must be > 0.")
        return BackfillStep(mode="days", days=days)
    mode = str(args.step).strip().lower() or "weekly"
    if mode == "daily":
        return BackfillStep(mode="days", days=1)
    if mode == "weekly":
        return BackfillStep(mode="days", days=7)
    if mode == "monthly":
        return BackfillStep(mode="months", days=0)
    raise ValueError("--step must be one of {'daily','weekly','monthly'}.")


def _month_add_utc(value: datetime) -> datetime:
    year = int(value.year)
    month = int(value.month) + 1
    if month > 12:
        year += 1
        month = 1
    # normalize day to month end
    day = int(value.day)
    while True:
        try:
            return value.replace(year=year, month=month, day=day)
        except ValueError:
            day -= 1
            if day <= 0:
                return value.replace(year=year, month=month, day=1)


def _iter_as_of_points(start_ts: datetime, end_ts: datetime, step: BackfillStep) -> List[datetime]:
    if end_ts < start_ts:
        return []
    points: List[datetime] = []
    current = start_ts
    while current <= end_ts:
        points.append(current)
        if step.mode == "months":
            current = _month_add_utc(current)
        else:
            current = current + timedelta(days=step.days)
    if points and points[-1] != end_ts:
        points.append(end_ts)
    return points


def _discover_camunda_bounds(config: Dict[str, Any], selected_processes: Sequence[str]) -> tuple[datetime | None, datetime | None]:
    runtime = CamundaRuntimeAdapter(_resolve_runtime_config(config))
    min_ts: datetime | None = None
    max_ts: datetime | None = None
    for process_namespace in selected_processes:
        process_key, tenant_id = _split_namespace(process_namespace)
        if not process_key:
            continue
        events, _ = runtime.fetch_historic_activity_events(
            process_name=process_key,
            version_key="",
            since=None,
            until=None,
        )
        events = _filter_by_tenant(events, tenant_id)
        for event in events:
            ts = _event_ts(event)
            if ts is None:
                continue
            if min_ts is None or ts < min_ts:
                min_ts = ts
            if max_ts is None or ts > max_ts:
                max_ts = ts
    return min_ts, max_ts


def _discover_xes_bounds(config: Dict[str, Any], selected_processes: Sequence[str]) -> tuple[datetime | None, datetime | None]:
    mapping = config.get("mapping", {})
    if not isinstance(mapping, dict):
        mapping = {}
    data_cfg = config.get("data", {})
    if not isinstance(data_cfg, dict):
        data_cfg = {}
    log_path = _normalize_text(data_cfg.get("log_path"))
    if not log_path:
        raise ValueError("sync-stats-backfill for xes requires data.log_path.")
    dataset_name_cfg = _normalize_text(data_cfg.get("dataset_name"))
    process_set = {_normalize_text(item) for item in selected_processes if _normalize_text(item)}
    process_set_l = {item.lower(): item for item in process_set}

    min_ts: datetime | None = None
    max_ts: datetime | None = None
    adapter = XESAdapter()
    for file_path in _iter_xes_paths(log_path):
        fallback_name = dataset_name_cfg or file_path.stem
        mapping_for_file = _inject_xes_dataset_name(mapping, fallback_name)
        for trace in adapter.read(str(file_path), mapping_for_file):
            trace_version = _normalize_text(getattr(trace, "process_version", ""))
            resolved_namespace = ""
            candidates = [trace_version, dataset_name_cfg, file_path.stem, fallback_name]
            for candidate in candidates:
                norm = _normalize_text(candidate)
                if not norm:
                    continue
                if process_set:
                    matched = process_set_l.get(norm.lower())
                    if matched:
                        resolved_namespace = matched
                        break
                if not resolved_namespace:
                    resolved_namespace = norm
            process_namespace = resolved_namespace
            if process_set and process_namespace not in process_set:
                continue
            for event in getattr(trace, "events", []) or []:
                try:
                    ts = float(getattr(event, "timestamp", 0.0))
                except (TypeError, ValueError):
                    continue
                dt = datetime.fromtimestamp(ts, tz=timezone.utc)
                if min_ts is None or dt < min_ts:
                    min_ts = dt
                if max_ts is None or dt > max_ts:
                    max_ts = dt
    return min_ts, max_ts


def _discover_bounds(config: Dict[str, Any]) -> tuple[List[str], datetime | None, datetime | None, str]:
    mapping = config.get("mapping", {})
    if not isinstance(mapping, dict):
        mapping = {}
    adapter_kind = _normalize_text(mapping.get("adapter", "camunda")).lower() or "camunda"
    if adapter_kind not in {"camunda", "xes"}:
        raise ValueError("sync-stats-backfill supports mapping.adapter in {'camunda','xes'}.")

    sync_cfg_raw = config.get("sync_stats", {})
    sync_cfg = dict(sync_cfg_raw) if isinstance(sync_cfg_raw, dict) else {}
    knowledge_repo = build_knowledge_graph_repository(config)
    all_processes = knowledge_repo.list_process_names()
    process_filters, tenant_filters = _derive_process_tenant_filters(config, sync_cfg)
    selected = _select_process_names(
        all_processes=all_processes,
        process_filters=process_filters,
        tenant_filters=tenant_filters,
    )

    if adapter_kind == "camunda":
        min_ts, max_ts = _discover_camunda_bounds(config, selected)
    else:
        min_ts, max_ts = _discover_xes_bounds(config, selected)
    return list(selected), min_ts, max_ts, adapter_kind


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser(description="Backfill sync-stats snapshots from first to last event timestamp.")
    parser.add_argument("--config", required=True, help="Experiment config path.")
    parser.add_argument("--out-dir", default="outputs/sync_stats_backfill", help="Directory for per-run summaries.")
    parser.add_argument("--summary-out", default="", help="Optional path for backfill summary JSON.")
    parser.add_argument("--step", default="weekly", help="One of: daily, weekly, monthly.")
    parser.add_argument("--step-days", dest="step_days", type=int, default=None, help="Custom step in days (overrides --step).")
    parser.add_argument("--from", dest="from_ts", default="", help="Optional ISO lower bound override (UTC).")
    parser.add_argument("--to", dest="to_ts", default="", help="Optional ISO upper bound override (UTC).")
    parser.add_argument("--dry-run", action="store_true", help="Only print planned as-of points, no sync runs.")
    args = parser.parse_args(argv)

    config = load_yaml_config(args.config)
    selected_processes, min_ts, max_ts, adapter_kind = _discover_bounds(config)
    if min_ts is None or max_ts is None:
        raise ValueError("Unable to discover first/last event timestamp for selected processes.")

    start_ts = _parse_iso_utc(args.from_ts) if _normalize_text(args.from_ts) else min_ts
    end_ts = _parse_iso_utc(args.to_ts) if _normalize_text(args.to_ts) else max_ts
    if end_ts < start_ts:
        raise ValueError("Backfill bounds are invalid: --to is earlier than --from.")

    step = _resolve_step(args)
    points = _iter_as_of_points(start_ts, end_ts, step)
    if not points:
        raise ValueError("No as-of points resolved for backfill.")

    out_dir = Path(str(args.out_dir)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("========== SYNC-STATS-BACKFILL PROFILE ==========")
    logger.info(
        "BACKFILL_PROFILE adapter=%s processes=%d start=%s end=%s points=%d step=%s",
        adapter_kind,
        len(selected_processes),
        start_ts.isoformat(),
        end_ts.isoformat(),
        len(points),
        f"{step.mode}:{step.days}" if step.mode == "days" else "months:1",
    )
    logger.info("==================================================")

    runs: List[Dict[str, Any]] = []
    iterator: Iterable[datetime] = tqdm(points, desc="sync-stats-backfill runs", unit="run")
    for idx, as_of_ts in enumerate(iterator, start=1):
        as_of_iso = as_of_ts.isoformat().replace("+00:00", "Z")
        out_path = out_dir / f"sync_stats_{idx:04d}_{as_of_ts.strftime('%Y%m%dT%H%M%SZ')}.json"
        run_payload = {
            "index": idx,
            "as_of_ts": as_of_iso,
            "out": str(out_path),
            "status": "planned" if args.dry_run else "pending",
        }
        if not args.dry_run:
            rc = sync_stats_main(
                [
                    "--config",
                    str(args.config),
                    "--out",
                    str(out_path),
                    "--as-of",
                    as_of_iso,
                ]
            )
            run_payload["rc"] = int(rc)
            run_payload["status"] = "ok" if int(rc) == 0 else "failed"
            if int(rc) != 0:
                runs.append(run_payload)
                break
        runs.append(run_payload)

    summary = {
        "status": "ok" if all(item.get("status") != "failed" for item in runs) else "failed",
        "mode": "sync-stats-backfill",
        "config": str(args.config),
        "adapter": adapter_kind,
        "selected_processes": selected_processes,
        "time_bounds_discovered": {
            "first_event_ts": min_ts.isoformat(),
            "last_event_ts": max_ts.isoformat(),
        },
        "time_bounds_effective": {
            "from_ts": start_ts.isoformat(),
            "to_ts": end_ts.isoformat(),
        },
        "step": {"mode": step.mode, "days": step.days},
        "runs_total": len(points),
        "runs_completed": len([item for item in runs if item.get("status") in {"ok", "planned"}]),
        "dry_run": bool(args.dry_run),
        "runs": runs,
    }
    summary_out = Path(str(args.summary_out)).resolve() if _normalize_text(args.summary_out) else out_dir / "backfill_summary.json"
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Backfill summary saved to %s", summary_out)
    return 0 if summary["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
