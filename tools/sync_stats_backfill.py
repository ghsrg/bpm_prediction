"""Backfill wrapper for periodic sync-stats runs over historical as-of points."""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
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


def _counter_dict(counter: Counter[str]) -> Dict[str, int]:
    return {key: int(value) for key, value in sorted(counter.items())}


def _safe_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed


def _min_metric(current: float | None, value: Any) -> float | None:
    parsed = _safe_float(value)
    if parsed is None:
        return current
    if current is None or parsed < current:
        return parsed
    return current


def _load_run_summary(path: Path) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _new_backfill_aggregate(runs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    status_counts = Counter(str(item.get("status", "unknown")) for item in runs)
    return {
        "runs": {
            "total": int(len(runs)),
            "ok": int(status_counts.get("ok", 0)),
            "failed": int(status_counts.get("failed", 0)),
            "planned": int(status_counts.get("planned", 0)),
        },
        "versions": {
            "processed": 0,
            "skipped": 0,
            "usable_for_training": 0,
            "not_usable_for_training": 0,
        },
        "quality": {"reasons": {}},
        "alignment": {
            "ok": 0,
            "failed": 0,
            "min_event_match_ratio": None,
            "min_unique_activity_coverage": None,
            "min_node_coverage": None,
            "failed_reasons": {},
        },
        "skips": {"reasons": {}},
        "by_process_version": {},
        "summary_files": {"read": 0, "missing_or_invalid": 0},
    }


def _build_backfill_aggregate(runs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    aggregate = _new_backfill_aggregate(runs)
    quality_reasons: Counter[str] = Counter()
    alignment_failed_reasons: Counter[str] = Counter()
    skip_reasons: Counter[str] = Counter()
    by_process_version: Dict[str, Dict[str, Any]] = {}

    for run in runs:
        if str(run.get("status")) != "ok":
            continue
        run_summary = _load_run_summary(Path(str(run.get("out", ""))))
        if run_summary is None:
            aggregate["summary_files"]["missing_or_invalid"] += 1
            continue
        aggregate["summary_files"]["read"] += 1

        details = run_summary.get("details", [])
        if not isinstance(details, list):
            details = []
        skipped_details = run_summary.get("skipped_details", [])
        if not isinstance(skipped_details, list):
            skipped_details = []

        processed_count = len(details)
        skipped_count = len(skipped_details)
        if processed_count <= 0:
            processed_count = int(run_summary.get("processed_versions", 0) or 0)
        if skipped_count <= 0:
            skipped_count = int(run_summary.get("skipped_versions", 0) or 0)
        aggregate["versions"]["processed"] += int(processed_count)
        aggregate["versions"]["skipped"] += int(skipped_count)

        for item in details:
            if not isinstance(item, dict):
                continue
            process_name = str(
                item.get("process_namespace") or item.get("process_name") or ""
            ).strip()
            version = str(item.get("version") or "").strip()
            key = f"{process_name}::{version}" if process_name or version else "unknown::unknown"
            process_bucket = by_process_version.setdefault(
                key,
                {
                    "runs_seen": 0,
                    "usable_for_training": 0,
                    "not_usable_for_training": 0,
                    "alignment_failed": 0,
                    "quality_reasons": Counter(),
                    "alignment_failed_reasons": Counter(),
                    "min_event_match_ratio": None,
                    "min_unique_activity_coverage": None,
                    "min_node_coverage": None,
                },
            )
            process_bucket["runs_seen"] += 1

            if bool(item.get("is_usable_for_training", False)):
                aggregate["versions"]["usable_for_training"] += 1
                process_bucket["usable_for_training"] += 1
                quality_reason = str(item.get("quality_reason") or "ok")
            else:
                aggregate["versions"]["not_usable_for_training"] += 1
                process_bucket["not_usable_for_training"] += 1
                quality_reason = str(item.get("quality_reason") or "not_usable")
            quality_reasons[quality_reason] += 1
            process_bucket["quality_reasons"][quality_reason] += 1

            aggregate["alignment"]["min_event_match_ratio"] = _min_metric(
                aggregate["alignment"]["min_event_match_ratio"],
                item.get("alignment_event_match_ratio"),
            )
            aggregate["alignment"]["min_unique_activity_coverage"] = _min_metric(
                aggregate["alignment"]["min_unique_activity_coverage"],
                item.get("alignment_unique_activity_coverage"),
            )
            aggregate["alignment"]["min_node_coverage"] = _min_metric(
                aggregate["alignment"]["min_node_coverage"],
                item.get("alignment_node_coverage"),
            )
            process_bucket["min_event_match_ratio"] = _min_metric(
                process_bucket["min_event_match_ratio"],
                item.get("alignment_event_match_ratio"),
            )
            process_bucket["min_unique_activity_coverage"] = _min_metric(
                process_bucket["min_unique_activity_coverage"],
                item.get("alignment_unique_activity_coverage"),
            )
            process_bucket["min_node_coverage"] = _min_metric(
                process_bucket["min_node_coverage"],
                item.get("alignment_node_coverage"),
            )

            if bool(item.get("alignment_is_ok", True)):
                aggregate["alignment"]["ok"] += 1
            else:
                aggregate["alignment"]["failed"] += 1
                process_bucket["alignment_failed"] += 1
                failures = item.get("alignment_failures", [])
                if not isinstance(failures, list) or not failures:
                    failures = [str(item.get("alignment_reason") or "alignment_failed")]
                for failure in failures:
                    failure_key = str(failure or "alignment_failed")
                    alignment_failed_reasons[failure_key] += 1
                    process_bucket["alignment_failed_reasons"][failure_key] += 1

        for item in skipped_details:
            if not isinstance(item, dict):
                continue
            reason = str(item.get("reason") or "unknown")
            skip_reasons[reason] += 1

    aggregate["quality"]["reasons"] = _counter_dict(quality_reasons)
    aggregate["alignment"]["failed_reasons"] = _counter_dict(alignment_failed_reasons)
    aggregate["skips"]["reasons"] = _counter_dict(skip_reasons)
    aggregate["by_process_version"] = {
        key: {
            "runs_seen": int(value["runs_seen"]),
            "usable_for_training": int(value["usable_for_training"]),
            "not_usable_for_training": int(value["not_usable_for_training"]),
            "alignment_failed": int(value["alignment_failed"]),
            "quality_reasons": _counter_dict(value["quality_reasons"]),
            "alignment_failed_reasons": _counter_dict(value["alignment_failed_reasons"]),
            "min_event_match_ratio": value["min_event_match_ratio"],
            "min_unique_activity_coverage": value["min_unique_activity_coverage"],
            "min_node_coverage": value["min_node_coverage"],
        }
        for key, value in sorted(by_process_version.items())
    }
    return aggregate


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
        "aggregate": _build_backfill_aggregate(runs),
        "runs": runs,
    }
    summary_out = Path(str(args.summary_out)).resolve() if _normalize_text(args.summary_out) else out_dir / "backfill_summary.json"
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Backfill summary saved to %s", summary_out)
    return 0 if summary["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
