"""Periodic statistics synchronization tool (Stage 3.4 Tier A)."""

from __future__ import annotations

import argparse
import json
import logging
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

from src.adapters.ingestion.xes_adapter import XESAdapter
from src.adapters.ingestion.camunda_runtime_adapter import CamundaRuntimeAdapter
from src.cli import load_yaml_config
from src.domain.entities.process_event import ProcessEventDTO
from src.domain.entities.process_structure import ProcessStructureDTO
from src.domain.services.activity_topology_alignment_service import (
    ActivityTopologyAlignmentService,
    AlignmentGateConfig,
)
from src.infrastructure.repositories.knowledge_graph_repository_factory import (
    build_knowledge_graph_repository,
    get_knowledge_graph_settings,
)


logger = logging.getLogger(__name__)


_STATS_CONTRACT_VERSION = "1.0"
_NON_ZERO_EPSILON = 1e-12
_ZERO_DOMINANT_THRESHOLD = 0.95
_DEFAULT_QUALITY_GATE = {
    "enabled": True,
    "zero_dominant_threshold": 0.95,
    "min_non_zero_ratio_overall": 0.0,
    "min_history_coverage_percent": 0.0,
    "on_fail": "write_with_flag",
    "warn_on_fail": True,
}
_DEFAULT_ALIGNMENT_GATE = {
    "enabled": True,
    "profile": "legacy_exact",
    "min_event_match_ratio": 0.6,
    "min_unique_activity_coverage": 0.6,
    "min_node_coverage": 0.0,
    "candidate_node_fields": ["id"],
    "ignore_structural_only_nodes": False,
    "strip_classifier_suffix": False,
    "normalize_case": False,
    "collapse_separators": False,
    "fail_on_ambiguity": True,
    "on_fail": "write_with_flag",
    "warn_on_fail": True,
}


@dataclass
class ScopeStatsResult:
    node_index: Dict[str, Dict[str, float]]
    edge_index: Dict[str, Dict[str, float]]
    global_index: Dict[str, float]


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"", "none", "null", "nan", "nat", "<na>"}:
        return ""
    return text


def _split_namespace(process_namespace: str) -> tuple[str, str]:
    text = _normalize_text(process_namespace)
    if "@" not in text:
        return text, ""
    process_key, tenant_id = text.rsplit("@", 1)
    return _normalize_text(process_key), _normalize_text(tenant_id)


def _normalize_version(value: str) -> str:
    text = _normalize_text(value)
    if not text:
        return ""
    if text.lower().startswith("v") and text[1:].isdigit():
        return f"v{int(text[1:])}"
    if text.isdigit():
        return f"v{int(text)}"
    return text


def _version_rank(value: str) -> Optional[int]:
    normalized = _normalize_version(value)
    if normalized.lower().startswith("v") and normalized[1:].isdigit():
        return int(normalized[1:])
    return None


def _parse_as_of(value: str | None) -> datetime:
    if value is None or not str(value).strip():
        return datetime.now(timezone.utc)
    text = str(value).strip()
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"Invalid --as-of timestamp: {value}") from exc
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _event_ts(event: ProcessEventDTO) -> Optional[datetime]:
    ts = event.end_time or event.start_time
    if ts is None:
        return None
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def _event_duration_ms(event: ProcessEventDTO) -> Optional[float]:
    if event.duration_ms is not None:
        try:
            return float(event.duration_ms)
        except (TypeError, ValueError):
            return None
    if event.start_time is not None and event.end_time is not None:
        delta = event.end_time - event.start_time
        return float(delta.total_seconds() * 1000.0)
    return None


def _event_actor(event: ProcessEventDTO) -> str:
    for candidate in (event.executed_by, event.assigned_executor, event.assignee):
        text = _normalize_text(candidate)
        if text:
            return text
    return "UNKNOWN"


def _event_tenant(event: ProcessEventDTO) -> str:
    if not isinstance(event.extra, dict):
        return ""
    for key in ("tenant_id", "tenant", "tenant_id_", "TENANT_ID_"):
        value = _normalize_text(event.extra.get(key))
        if value:
            return value
    return ""


def _iter_windows(as_of_ts: datetime, windows_days: Sequence[int]) -> Dict[str, datetime]:
    windows: Dict[str, datetime] = {}
    for day in sorted({int(item) for item in windows_days if int(item) > 0}):
        windows[f"last_{day}d"] = as_of_ts - timedelta(days=day)
    return windows


def _filter_events_by_time(events: Sequence[ProcessEventDTO], *, start_ts: Optional[datetime], end_ts: datetime) -> List[ProcessEventDTO]:
    out: List[ProcessEventDTO] = []
    for event in events:
        ts = _event_ts(event)
        if ts is None:
            continue
        if ts > end_ts:
            continue
        if start_ts is not None and ts < start_ts:
            continue
        out.append(event)
    return out


def _group_by_case(events: Sequence[ProcessEventDTO]) -> Dict[str, List[ProcessEventDTO]]:
    grouped: Dict[str, List[ProcessEventDTO]] = defaultdict(list)
    for event in events:
        case_id = _normalize_text(event.case_id) or "unknown_case"
        grouped[case_id].append(event)
    for case_id in list(grouped.keys()):
        grouped[case_id] = sorted(
            grouped[case_id],
            key=lambda item: (
                _event_ts(item) or datetime.min.replace(tzinfo=timezone.utc),
                _normalize_text(item.act_inst_id),
                _normalize_text(item.activity_def_id),
            ),
        )
    return grouped


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.quantile(np.asarray(values, dtype=np.float64), q))


def _safe_entropy(counter: Counter[str]) -> float:
    total = float(sum(counter.values()))
    if total <= 0.0:
        return 0.0
    probs = [float(count) / total for count in counter.values() if count > 0]
    if not probs:
        return 0.0
    return float(-sum(p * math.log(p + 1e-12, 2) for p in probs))


def _freshness_score(last_seen: Optional[datetime], as_of_ts: datetime, half_life_days: float) -> float:
    if last_seen is None:
        return 0.0
    age_days = max(0.0, float((as_of_ts - last_seen).total_seconds() / 86400.0))
    if half_life_days <= 0.0:
        return 1.0
    decay_lambda = math.log(2.0) / half_life_days
    return float(math.exp(-decay_lambda * age_days))


def _build_transition_maps(case_events: Dict[str, List[ProcessEventDTO]]) -> tuple[Counter[tuple[str, str]], Dict[tuple[str, str], List[float]], Counter[str]]:
    transition_counts: Counter[tuple[str, str]] = Counter()
    transition_latencies: Dict[tuple[str, str], List[float]] = defaultdict(list)
    outgoing: Counter[str] = Counter()

    for events in case_events.values():
        for left, right in zip(events, events[1:]):
            src = _normalize_text(left.activity_def_id)
            dst = _normalize_text(right.activity_def_id)
            if not src or not dst:
                continue
            edge = (src, dst)
            transition_counts[edge] += 1
            outgoing[src] += 1

            left_ts = _event_ts(left)
            right_ts = _event_ts(right)
            if left_ts is not None and right_ts is not None:
                delta_ms = max(0.0, (right_ts - left_ts).total_seconds() * 1000.0)
                transition_latencies[edge].append(float(delta_ms))

    return transition_counts, transition_latencies, outgoing


def _compute_scope_stats(
    *,
    events: Sequence[ProcessEventDTO],
    activity_ids: Sequence[str],
    allowed_edges: Sequence[tuple[str, str]],
    as_of_ts: datetime,
    coverage_percent: float,
    confidence_weights: Dict[str, float],
    freshness_half_life_days: float,
) -> ScopeStatsResult:
    grouped = _group_by_case(events)

    duration_by_activity: Dict[str, List[float]] = defaultdict(list)
    exec_count: Counter[str] = Counter()
    last_seen: Dict[str, datetime] = {}
    repeats_per_case: Dict[str, List[float]] = defaultdict(list)
    concurrent_per_case: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    actor_handover = Counter()

    for case_id, case_events in grouped.items():
        per_case_activity = Counter()
        per_case_actors: List[str] = []
        for event in case_events:
            activity = _normalize_text(event.activity_def_id)
            if not activity:
                continue
            per_case_activity[activity] += 1
            exec_count[activity] += 1

            duration_ms = _event_duration_ms(event)
            if duration_ms is not None:
                duration_by_activity[activity].append(float(duration_ms))

            ts = _event_ts(event)
            if ts is not None:
                previous = last_seen.get(activity)
                if previous is None or ts > previous:
                    last_seen[activity] = ts

            if bool(event.is_concurrent):
                concurrent_per_case[case_id][activity] += 1

            per_case_actors.append(_event_actor(event))

        for activity in activity_ids:
            repeats_per_case[activity].append(float(per_case_activity.get(activity, 0)))

        for left_actor, right_actor in zip(per_case_actors, per_case_actors[1:]):
            actor_handover[(left_actor, right_actor)] += 1

    max_exec_count = max(exec_count.values()) if exec_count else 1
    weight_sample = float(confidence_weights.get("sample_size", 0.4))
    weight_fresh = float(confidence_weights.get("freshness", 0.3))
    weight_cov = float(confidence_weights.get("coverage", 0.3))
    coverage_score = max(0.0, min(1.0, coverage_percent / 100.0))

    node_index: Dict[str, Dict[str, float]] = {
        "exec_count": {},
        "duration_median": {},
        "duration_p95": {},
        "loop_density": {},
        "parallel_token_variance": {},
        "freshness_score": {},
        "confidence_score": {},
    }

    for activity in activity_ids:
        count = float(exec_count.get(activity, 0))
        durations = duration_by_activity.get(activity, [])
        repeats = repeats_per_case.get(activity, [0.0])
        concurrent_counts = [float(case_map.get(activity, 0)) for case_map in concurrent_per_case.values()]
        concurrent_variance = float(np.var(np.asarray(concurrent_counts), ddof=0)) if concurrent_counts else 0.0

        freshness = _freshness_score(last_seen.get(activity), as_of_ts, freshness_half_life_days)
        sample_score = 0.0
        if max_exec_count > 0 and count > 0:
            sample_score = float(math.log1p(count) / math.log1p(float(max_exec_count)))

        confidence = (
            weight_sample * sample_score
            + weight_fresh * freshness
            + weight_cov * coverage_score
        )

        node_index["exec_count"][activity] = count
        node_index["duration_median"][activity] = _quantile(durations, 0.5)
        node_index["duration_p95"][activity] = _quantile(durations, 0.95)
        node_index["loop_density"][activity] = float(np.mean(np.asarray(repeats, dtype=np.float64))) if repeats else 0.0
        node_index["parallel_token_variance"][activity] = concurrent_variance
        node_index["freshness_score"][activity] = freshness
        node_index["confidence_score"][activity] = float(max(0.0, min(1.0, confidence)))

    transition_counts, transition_latencies, outgoing = _build_transition_maps(grouped)
    edge_index: Dict[str, Dict[str, float]] = {
        "transition_probability": {},
        "latency_median": {},
        "latency_p95": {},
    }
    activity_id_set = {_normalize_text(activity) for activity in activity_ids if _normalize_text(activity)}
    edge_candidates = {
        (_normalize_text(src), _normalize_text(dst))
        for src, dst in allowed_edges
        if _normalize_text(src) and _normalize_text(dst)
    }
    edge_candidates.update(
        (src, dst)
        for src, dst in transition_counts.keys()
        if src in activity_id_set and dst in activity_id_set
    )
    for src, dst in sorted(edge_candidates):
        edge_key = f"{src}|||{dst}"
        count = float(transition_counts.get((src, dst), 0))
        out_count = float(outgoing.get(src, 0))
        probability = (count / out_count) if out_count > 0 else 0.0
        latencies = transition_latencies.get((src, dst), [])
        edge_index["transition_probability"][edge_key] = float(max(0.0, min(1.0, probability)))
        edge_index["latency_median"][edge_key] = _quantile(latencies, 0.5)
        edge_index["latency_p95"][edge_key] = _quantile(latencies, 0.95)

    global_index = {
        "resource_handover_entropy": _safe_entropy(Counter({f"{a}->{b}": c for (a, b), c in actor_handover.items()})),
        "coverage_percent": float(coverage_percent),
        "num_cases": float(len(grouped)),
    }
    return ScopeStatsResult(node_index=node_index, edge_index=edge_index, global_index=global_index)


def _filter_by_tenant(events: Sequence[ProcessEventDTO], tenant_id: str) -> List[ProcessEventDTO]:
    tenant = _normalize_text(tenant_id)
    if not tenant:
        return list(events)
    return [event for event in events if _normalize_text(_event_tenant(event)) == tenant]


def _filter_by_scope_policy(
    events: Sequence[ProcessEventDTO],
    *,
    target_version: str,
    scope: str,
    process_scope_policy: str,
) -> List[ProcessEventDTO]:
    scope_key = str(scope).strip().lower()
    target_norm = _normalize_version(target_version)
    target_rank = _version_rank(target_norm)

    filtered: List[ProcessEventDTO] = []
    for event in events:
        event_version = _normalize_version(event.proc_def_version or "")
        if not event_version:
            event_version = _normalize_version(str((event.extra or {}).get("version_key", "")))
        event_rank = _version_rank(event_version)

        if scope_key == "version":
            if target_rank is not None and event_rank is not None:
                if event_rank == target_rank:
                    filtered.append(event)
                continue
            if event_version == target_norm:
                filtered.append(event)
            continue

        # scope == process
        if process_scope_policy == "up_to_target_version":
            if target_rank is not None and event_rank is not None:
                if event_rank <= target_rank:
                    filtered.append(event)
                continue
            # For non-ranked versions (not vN), and mixed rank/non-rank cases,
            # we keep exact-match behavior only.
            if event_version == target_norm:
                filtered.append(event)
            continue

        filtered.append(event)
    return filtered


def _normalize_split_strategy(raw: Any) -> str:
    text = _normalize_text(raw).lower() or "temporal"
    if text == "time":
        text = "temporal"
    if text not in {"temporal", "none"}:
        raise ValueError("experiment.split_strategy must be 'temporal' or 'none'.")
    return text


def _resolve_train_cut_config(config: Dict[str, Any]) -> tuple[str, float, float]:
    experiment_cfg = config.get("experiment", {})
    if not isinstance(experiment_cfg, dict):
        experiment_cfg = {}
    split_strategy = _normalize_split_strategy(experiment_cfg.get("split_strategy", "temporal"))

    train_ratio = float(experiment_cfg.get("train_ratio", 1.0) or 1.0)
    if train_ratio < 0.0 or train_ratio > 1.0:
        raise ValueError("experiment.train_ratio must be within [0.0, 1.0].")

    fraction = float(experiment_cfg.get("fraction", 1.0) or 1.0)
    if fraction <= 0.0 or fraction > 1.0:
        raise ValueError("experiment.fraction must be within (0.0, 1.0].")
    return split_strategy, train_ratio, fraction


def _resolve_quality_gate_config(sync_cfg: Dict[str, Any]) -> Dict[str, Any]:
    raw = sync_cfg.get("quality_gate", {})
    cfg = dict(raw) if isinstance(raw, dict) else {}

    enabled = bool(cfg.get("enabled", _DEFAULT_QUALITY_GATE["enabled"]))
    warn_on_fail = bool(cfg.get("warn_on_fail", _DEFAULT_QUALITY_GATE["warn_on_fail"]))

    zero_dominant_threshold = float(cfg.get("zero_dominant_threshold", _DEFAULT_QUALITY_GATE["zero_dominant_threshold"]) or 0.0)
    if zero_dominant_threshold < 0.0 or zero_dominant_threshold > 1.0:
        raise ValueError("sync_stats.quality_gate.zero_dominant_threshold must be within [0.0, 1.0].")

    min_non_zero_ratio_overall = float(
        cfg.get("min_non_zero_ratio_overall", _DEFAULT_QUALITY_GATE["min_non_zero_ratio_overall"]) or 0.0
    )
    if min_non_zero_ratio_overall < 0.0 or min_non_zero_ratio_overall > 1.0:
        raise ValueError("sync_stats.quality_gate.min_non_zero_ratio_overall must be within [0.0, 1.0].")

    min_history_coverage_percent = float(
        cfg.get("min_history_coverage_percent", _DEFAULT_QUALITY_GATE["min_history_coverage_percent"]) or 0.0
    )
    if min_history_coverage_percent < 0.0 or min_history_coverage_percent > 100.0:
        raise ValueError("sync_stats.quality_gate.min_history_coverage_percent must be within [0.0, 100.0].")

    on_fail = _normalize_text(cfg.get("on_fail", _DEFAULT_QUALITY_GATE["on_fail"])).lower() or "write_with_flag"
    if on_fail not in {"write_with_flag", "skip_snapshot"}:
        raise ValueError("sync_stats.quality_gate.on_fail must be one of {'write_with_flag', 'skip_snapshot'}.")

    return {
        "enabled": enabled,
        "zero_dominant_threshold": float(zero_dominant_threshold),
        "min_non_zero_ratio_overall": float(min_non_zero_ratio_overall),
        "min_history_coverage_percent": float(min_history_coverage_percent),
        "on_fail": on_fail,
        "warn_on_fail": warn_on_fail,
    }


def _resolve_alignment_gate_config(sync_cfg: Dict[str, Any]) -> Dict[str, Any]:
    raw = sync_cfg.get("alignment_gate", {})
    cfg = dict(raw) if isinstance(raw, dict) else {}

    enabled = bool(cfg.get("enabled", _DEFAULT_ALIGNMENT_GATE["enabled"]))
    warn_on_fail = bool(cfg.get("warn_on_fail", _DEFAULT_ALIGNMENT_GATE["warn_on_fail"]))
    profile = _normalize_text(cfg.get("profile", _DEFAULT_ALIGNMENT_GATE["profile"])) or "legacy_exact"
    base_profile = AlignmentGateConfig.for_profile(profile)

    min_event_match_ratio = float(cfg.get("min_event_match_ratio", base_profile.min_event_match_ratio) or 0.0)
    if min_event_match_ratio < 0.0 or min_event_match_ratio > 1.0:
        raise ValueError("sync_stats.alignment_gate.min_event_match_ratio must be within [0.0, 1.0].")

    min_unique_activity_coverage = float(
        cfg.get("min_unique_activity_coverage", base_profile.min_unique_activity_coverage) or 0.0
    )
    if min_unique_activity_coverage < 0.0 or min_unique_activity_coverage > 1.0:
        raise ValueError("sync_stats.alignment_gate.min_unique_activity_coverage must be within [0.0, 1.0].")

    min_node_coverage = float(cfg.get("min_node_coverage", base_profile.min_node_coverage) or 0.0)
    if min_node_coverage < 0.0 or min_node_coverage > 1.0:
        raise ValueError("sync_stats.alignment_gate.min_node_coverage must be within [0.0, 1.0].")

    on_fail = _normalize_text(cfg.get("on_fail", _DEFAULT_ALIGNMENT_GATE["on_fail"])).lower() or "write_with_flag"
    if on_fail not in {"write_with_flag", "skip_snapshot", "raise"}:
        raise ValueError(
            "sync_stats.alignment_gate.on_fail must be one of {'write_with_flag', 'skip_snapshot', 'raise'}."
        )

    candidate_node_fields = cfg.get("candidate_node_fields", base_profile.candidate_node_fields)
    if isinstance(candidate_node_fields, str):
        candidate_node_fields = [
            item.strip() for item in candidate_node_fields.split(",") if item.strip()
        ]
    elif isinstance(candidate_node_fields, (list, tuple, set)):
        candidate_node_fields = [
            str(item).strip() for item in candidate_node_fields if str(item).strip()
        ]
    else:
        candidate_node_fields = list(base_profile.candidate_node_fields)
    if not candidate_node_fields:
        candidate_node_fields = list(base_profile.candidate_node_fields)

    return {
        "enabled": enabled,
        "profile": profile,
        "min_event_match_ratio": float(min_event_match_ratio),
        "min_unique_activity_coverage": float(min_unique_activity_coverage),
        "min_node_coverage": float(min_node_coverage),
        "candidate_node_fields": list(candidate_node_fields),
        "ignore_structural_only_nodes": bool(
            cfg.get(
                "ignore_structural_only_nodes",
                base_profile.ignore_structural_only_nodes,
            )
        ),
        "strip_classifier_suffix": bool(
            cfg.get("strip_classifier_suffix", base_profile.strip_classifier_suffix)
        ),
        "normalize_case": bool(cfg.get("normalize_case", base_profile.normalize_case)),
        "collapse_separators": bool(
            cfg.get("collapse_separators", base_profile.collapse_separators)
        ),
        "fail_on_ambiguity": bool(
            cfg.get("fail_on_ambiguity", base_profile.fail_on_ambiguity)
        ),
        "on_fail": on_fail,
        "warn_on_fail": warn_on_fail,
    }


def _trace_start_ts(trace: Any) -> datetime | None:
    events = getattr(trace, "events", None) or []
    if not events:
        return None
    try:
        ts = float(getattr(events[0], "timestamp", 0.0))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(ts):
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def _trace_end_ts(trace: Any) -> datetime | None:
    events = getattr(trace, "events", None) or []
    if not events:
        return None
    try:
        ts = float(getattr(events[-1], "timestamp", 0.0))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(ts):
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def _max_event_ts(events: Sequence[ProcessEventDTO]) -> datetime | None:
    latest: datetime | None = None
    for event in events:
        ts = _event_ts(event)
        if ts is None:
            continue
        if latest is None or ts > latest:
            latest = ts
    return latest


def _select_train_traces(
    traces: Sequence[Any],
    *,
    split_strategy: str,
    train_ratio: float,
    fraction: float,
) -> tuple[List[Any], datetime | None, int, int]:
    traces_with_events = [trace for trace in traces if getattr(trace, "events", None)]
    if split_strategy == "temporal":
        ordered = sorted(
            traces_with_events,
            key=lambda tr: _trace_start_ts(tr) or datetime.min.replace(tzinfo=timezone.utc),
        )
    else:
        ordered = list(traces_with_events)

    split_idx = int(len(ordered) * train_ratio)
    macro = ordered[:split_idx]
    selected = macro if fraction >= 1.0 else macro[: int(len(macro) * fraction)]

    max_ts: datetime | None = None
    for trace in selected:
        trace_ts = _trace_end_ts(trace)
        if trace_ts is None:
            continue
        if max_ts is None or trace_ts > max_ts:
            max_ts = trace_ts
    return selected, max_ts, len(ordered), len(selected)


def _select_train_events(
    events: Sequence[ProcessEventDTO],
    *,
    split_strategy: str,
    train_ratio: float,
    fraction: float,
) -> tuple[List[ProcessEventDTO], datetime | None, int, int]:
    grouped = _group_by_case(events)
    case_items: List[tuple[str, List[ProcessEventDTO], datetime]] = []
    for case_id, case_events in grouped.items():
        first_ts = _event_ts(case_events[0]) if case_events else None
        case_items.append((case_id, case_events, first_ts or datetime.min.replace(tzinfo=timezone.utc)))

    if split_strategy == "temporal":
        case_items.sort(key=lambda item: item[2])

    split_idx = int(len(case_items) * train_ratio)
    macro = case_items[:split_idx]
    selected_cases = macro if fraction >= 1.0 else macro[: int(len(macro) * fraction)]
    selected_case_ids = {case_id for case_id, _, _ in selected_cases}

    selected_events: List[ProcessEventDTO] = []
    for case_id, case_events in grouped.items():
        if case_id in selected_case_ids:
            selected_events.extend(case_events)
    selected_events = sorted(
        selected_events,
        key=lambda item: (
            _event_ts(item) or datetime.min.replace(tzinfo=timezone.utc),
            _normalize_text(item.case_id),
            _normalize_text(item.act_inst_id),
            _normalize_text(item.activity_def_id),
        ),
    )
    return selected_events, _max_event_ts(selected_events), len(case_items), len(selected_case_ids)


def _dto_activity_ids(dto: ProcessStructureDTO) -> List[str]:
    activity_ids = sorted(
        {
            str(item.get("id", "")).strip()
            for item in (dto.nodes or [])
            if isinstance(item, dict) and str(item.get("id", "")).strip()
        }
    )
    if activity_ids:
        return activity_ids
    return sorted({src for src, _ in dto.allowed_edges} | {dst for _, dst in dto.allowed_edges})


def _collect_scope_events(
    events: Sequence[ProcessEventDTO],
    *,
    target_version: str,
    process_scope_policy: str,
    as_of_ts: datetime,
) -> tuple[List[ProcessEventDTO], List[ProcessEventDTO]]:
    version_events = _filter_by_scope_policy(
        events,
        target_version=target_version,
        scope="version",
        process_scope_policy=process_scope_policy,
    )
    process_events = _filter_by_scope_policy(
        events,
        target_version=target_version,
        scope="process",
        process_scope_policy=process_scope_policy,
    )
    version_events = _filter_events_by_time(version_events, start_ts=None, end_ts=as_of_ts)
    process_events = _filter_events_by_time(process_events, start_ts=None, end_ts=as_of_ts)
    return version_events, process_events


def _scope_event_counts(
    events: Sequence[ProcessEventDTO],
    *,
    target_version: str,
    process_scope_policy: str,
    as_of_ts: datetime,
) -> tuple[int, int]:
    version_events, process_events = _collect_scope_events(
        events,
        target_version=target_version,
        process_scope_policy=process_scope_policy,
        as_of_ts=as_of_ts,
    )
    return len(version_events), len(process_events)


def _count_non_zero_nested(index_map: Dict[str, Dict[str, float]]) -> tuple[int, int]:
    total = 0
    non_zero = 0
    for metric_values in index_map.values():
        if not isinstance(metric_values, dict):
            continue
        for value in metric_values.values():
            total += 1
            try:
                as_float = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(as_float) and abs(as_float) > _NON_ZERO_EPSILON:
                non_zero += 1
    return non_zero, total


def _count_non_zero_flat(index_map: Dict[str, float]) -> tuple[int, int]:
    total = 0
    non_zero = 0
    for value in index_map.values():
        total += 1
        try:
            as_float = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(as_float) and abs(as_float) > _NON_ZERO_EPSILON:
            non_zero += 1
    return non_zero, total


def _safe_ratio(non_zero: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return float(non_zero / total)


def _build_stats_quality(
    *,
    coverage_percent: float,
    stats_index_node: Dict[str, Dict[str, float]],
    stats_index_edge: Dict[str, Dict[str, float]],
    stats_index_global: Dict[str, float],
    quality_gate_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    node_non_zero, node_total = _count_non_zero_nested(stats_index_node)
    edge_non_zero, edge_total = _count_non_zero_nested(stats_index_edge)
    global_non_zero, global_total = _count_non_zero_flat(stats_index_global)

    total_non_zero = int(node_non_zero + edge_non_zero + global_non_zero)
    total_values = int(node_total + edge_total + global_total)
    overall_non_zero_ratio = _safe_ratio(total_non_zero, total_values)
    zero_ratio = 1.0 - overall_non_zero_ratio
    zero_dominant_threshold = float(quality_gate_cfg.get("zero_dominant_threshold", _ZERO_DOMINANT_THRESHOLD))
    min_non_zero_ratio_overall = float(quality_gate_cfg.get("min_non_zero_ratio_overall", 0.0))
    min_history_coverage_percent = float(quality_gate_cfg.get("min_history_coverage_percent", 0.0))
    zero_dominant = bool(total_values > 0 and zero_ratio >= zero_dominant_threshold)

    quality_failures: List[str] = []
    is_usable_for_training = True
    if float(coverage_percent) <= 0.0:
        quality_failures.append("no_coverage")
    if float(coverage_percent) < min_history_coverage_percent:
        quality_failures.append("below_min_coverage_threshold")
    if total_values <= 0:
        quality_failures.append("empty_stats")
    if zero_dominant:
        quality_failures.append("zero_dominant")
    if overall_non_zero_ratio < min_non_zero_ratio_overall:
        quality_failures.append("below_min_non_zero_ratio_threshold")

    quality_reason = "ok"
    if quality_failures:
        quality_reason = quality_failures[0]
        is_usable_for_training = False

    return {
        "history_coverage_percent": float(coverage_percent),
        "non_zero_ratio_node": _safe_ratio(node_non_zero, node_total),
        "non_zero_ratio_edge": _safe_ratio(edge_non_zero, edge_total),
        "non_zero_ratio_global": _safe_ratio(global_non_zero, global_total),
        "non_zero_ratio_overall": float(overall_non_zero_ratio),
        "zero_dominant_threshold": float(zero_dominant_threshold),
        "min_non_zero_ratio_overall": float(min_non_zero_ratio_overall),
        "min_history_coverage_percent": float(min_history_coverage_percent),
        "zero_dominant": bool(zero_dominant),
        "is_usable_for_training": bool(is_usable_for_training),
        "quality_reason": quality_reason,
        "quality_failures": list(dict.fromkeys(quality_failures)),
    }


def _build_stats_for_dto(
    *,
    dto: ProcessStructureDTO,
    process_namespace: str,
    all_events: Sequence[ProcessEventDTO],
    as_of_ts: datetime,
    windows_days: Sequence[int],
    stats_time_policy: str,
    process_scope_policy: str,
    coverage_percent: float,
    confidence_weights: Dict[str, float],
    freshness_half_life_days: float,
    split_strategy: str,
    train_ratio: float,
    fraction: float,
    quality_gate_cfg: Dict[str, Any],
    alignment_summary: Dict[str, Any] | None = None,
) -> tuple[ProcessStructureDTO, Dict[str, Any]]:
    activity_ids = _dto_activity_ids(dto)
    allowed_edges = list(dto.allowed_edges)

    windows = _iter_windows(as_of_ts, windows_days)
    windows["all_time"] = None  # type: ignore[assignment]

    node_stats: Dict[str, Any] = {"windows": {}}
    edge_stats: Dict[str, Any] = {"windows": {}}
    gnn_features: Dict[str, Any] = {"windows": {}}

    stats_index_node: Dict[str, Dict[str, float]] = {}
    stats_index_edge: Dict[str, Dict[str, float]] = {}
    stats_index_global: Dict[str, float] = {}

    for window_name, start_ts in windows.items():
        version_events = _filter_by_scope_policy(
            all_events,
            target_version=dto.version,
            scope="version",
            process_scope_policy=process_scope_policy,
        )
        process_events = _filter_by_scope_policy(
            all_events,
            target_version=dto.version,
            scope="process",
            process_scope_policy=process_scope_policy,
        )

        version_events = _filter_events_by_time(version_events, start_ts=start_ts, end_ts=as_of_ts)
        process_events = _filter_events_by_time(process_events, start_ts=start_ts, end_ts=as_of_ts)

        version_scope = _compute_scope_stats(
            events=version_events,
            activity_ids=activity_ids,
            allowed_edges=allowed_edges,
            as_of_ts=as_of_ts,
            coverage_percent=coverage_percent,
            confidence_weights=confidence_weights,
            freshness_half_life_days=freshness_half_life_days,
        )
        process_scope = _compute_scope_stats(
            events=process_events,
            activity_ids=activity_ids,
            allowed_edges=allowed_edges,
            as_of_ts=as_of_ts,
            coverage_percent=coverage_percent,
            confidence_weights=confidence_weights,
            freshness_half_life_days=freshness_half_life_days,
        )

        node_stats["windows"][window_name] = {
            "version": version_scope.node_index,
            "process": process_scope.node_index,
        }
        edge_stats["windows"][window_name] = {
            "version": version_scope.edge_index,
            "process": process_scope.edge_index,
        }
        gnn_features["windows"][window_name] = {
            "version": {
                "transition_probability": version_scope.edge_index.get("transition_probability", {}),
                "resource_handover_entropy": version_scope.global_index.get("resource_handover_entropy", 0.0),
            },
            "process": {
                "transition_probability": process_scope.edge_index.get("transition_probability", {}),
                "resource_handover_entropy": process_scope.global_index.get("resource_handover_entropy", 0.0),
            },
        }

        for scope_name, scope_data in (("version", version_scope), ("process", process_scope)):
            for metric_name, metric_values in scope_data.node_index.items():
                stats_index_node[f"{window_name}.{scope_name}.{metric_name}"] = metric_values
            for metric_name, metric_values in scope_data.edge_index.items():
                stats_index_edge[f"{window_name}.{scope_name}.{metric_name}"] = metric_values
            for metric_name, metric_value in scope_data.global_index.items():
                stats_index_global[f"{window_name}.{scope_name}.{metric_name}"] = float(metric_value)

    stats_diagnostics = {
        "stats_contract_version": _STATS_CONTRACT_VERSION,
        "stats_time_policy": stats_time_policy,
        "process_scope_policy": process_scope_policy,
        "as_of_ts": as_of_ts.isoformat(),
        "history_coverage_percent": float(coverage_percent),
        "process_namespace": process_namespace,
        "version": dto.version,
        "windows": list(windows.keys()),
        "tier": "A",
        "alignment": dict(alignment_summary or {}),
    }

    payload = dto.model_copy(deep=True)
    payload.node_stats = node_stats
    payload.edge_stats = edge_stats
    payload.gnn_features = gnn_features
    payload.stats_diagnostics = stats_diagnostics

    process_key, tenant_id = _split_namespace(process_namespace)
    stats_quality = _build_stats_quality(
        coverage_percent=coverage_percent,
        stats_index_node=stats_index_node,
        stats_index_edge=stats_index_edge,
        stats_index_global=stats_index_global,
        quality_gate_cfg=quality_gate_cfg,
    )
    alignment_payload = dict(alignment_summary or {})
    stats_policy = {
        "stats_time_policy": stats_time_policy,
        "process_scope_policy": process_scope_policy,
        "split_strategy": split_strategy,
        "train_ratio": float(train_ratio),
        "fraction": float(fraction),
    }
    stats_identity = {
        "tenant_id": tenant_id or None,
        "process_name": process_namespace,
        "process_key": process_key,
        "version_key": dto.version,
        "proc_def_id": _normalize_text(dto.proc_def_id) or None,
        "knowledge_version": None,
        "as_of_ts": as_of_ts.isoformat(),
    }

    metadata = dict(payload.metadata or {})
    metadata["stats_index"] = {
        "node": stats_index_node,
        "edge": stats_index_edge,
        "global": stats_index_global,
    }
    metadata["stats_policy"] = stats_policy
    metadata["stats_contract"] = {
        "version": _STATS_CONTRACT_VERSION,
        "key_format": "{window}.{scope}.{metric}",
        "identity": stats_identity,
        "quality": stats_quality,
        "alignment": alignment_payload,
        "policy": stats_policy,
    }
    payload.metadata = metadata

    summary = {
        "stats_contract_version": _STATS_CONTRACT_VERSION,
        "process_namespace": process_namespace,
        "version": dto.version,
        "history_coverage_percent": float(coverage_percent),
        "num_node_metrics": int(sum(len(values) for values in stats_index_node.values())),
        "num_edge_metrics": int(sum(len(values) for values in stats_index_edge.values())),
        "non_zero_ratio_node": float(stats_quality["non_zero_ratio_node"]),
        "non_zero_ratio_edge": float(stats_quality["non_zero_ratio_edge"]),
        "non_zero_ratio_global": float(stats_quality["non_zero_ratio_global"]),
        "non_zero_ratio_overall": float(stats_quality["non_zero_ratio_overall"]),
        "is_usable_for_training": bool(stats_quality["is_usable_for_training"]),
        "quality_reason": str(stats_quality["quality_reason"]),
        "quality_failures": list(stats_quality.get("quality_failures", [])),
        "alignment_is_ok": bool(alignment_payload.get("is_aligned", True)),
        "alignment_reason": str(alignment_payload.get("alignment_reason", "ok")),
        "alignment_failures": list(alignment_payload.get("alignment_failures", [])),
        "alignment_event_match_ratio": float(alignment_payload.get("event_match_ratio", 1.0)),
        "alignment_unique_activity_coverage": float(alignment_payload.get("unique_activity_coverage", 1.0)),
        "alignment_node_coverage": float(alignment_payload.get("node_coverage", 1.0)),
        "alignment_scope_used": str(alignment_payload.get("scope_used", "none")),
    }
    return payload, summary


def _select_process_names(
    *,
    all_processes: Sequence[str],
    process_filters: Sequence[str],
    tenant_filters: Sequence[str],
) -> List[str]:
    filters = {_normalize_text(item).lower() for item in process_filters if _normalize_text(item)}
    tenants = {_normalize_text(item).lower() for item in tenant_filters if _normalize_text(item)}

    selected: List[str] = []
    for process_namespace in all_processes:
        namespace = _normalize_text(process_namespace)
        process_key, tenant_id = _split_namespace(namespace)
        namespace_l = namespace.lower()
        process_key_l = process_key.lower()
        tenant_l = tenant_id.lower()

        if filters and namespace_l not in filters and process_key_l not in filters:
            continue
        if tenants and tenant_l not in tenants:
            continue
        selected.append(namespace)
    return sorted(selected)


def _resolve_runtime_config(config: Dict[str, Any]) -> Dict[str, Any]:
    mapping = config.get("mapping", {})
    if not isinstance(mapping, dict):
        mapping = {}
    camunda_cfg = mapping.get("camunda_adapter", {})
    if not isinstance(camunda_cfg, dict):
        camunda_cfg = {}
    runtime_cfg = camunda_cfg.get("runtime", {})
    runtime = dict(runtime_cfg) if isinstance(runtime_cfg, dict) else {}

    if not runtime:
        for key in (
            "runtime_source",
            "export_dir",
            "sql_dir",
            "history_cleanup_aware",
            "legacy_removal_time_policy",
            "on_missing_removal_time",
            "mssql",
            "connections_file",
            "connection_profile",
            "profile",
        ):
            if key in camunda_cfg:
                runtime[key] = camunda_cfg[key]

    runtime.setdefault("runtime_source", "files")
    runtime.setdefault("history_cleanup_aware", True)
    runtime.setdefault("legacy_removal_time_policy", "treat_as_eternal")
    runtime.setdefault("on_missing_removal_time", "auto_fallback")
    return runtime


def _derive_process_tenant_filters(config: Dict[str, Any], sync_cfg: Dict[str, Any]) -> tuple[List[str], List[str]]:
    mapping = config.get("mapping", {})
    if not isinstance(mapping, dict):
        mapping = {}
    camunda_cfg = mapping.get("camunda_adapter", {})
    if not isinstance(camunda_cfg, dict):
        camunda_cfg = {}

    raw_proc = sync_cfg.get("process_filters")
    process_filters: List[str] = []
    if isinstance(raw_proc, list):
        process_filters = [_normalize_text(item) for item in raw_proc if _normalize_text(item)]
    elif isinstance(camunda_cfg.get("process_filters"), list):
        process_filters = [
            _normalize_text(item)
            for item in camunda_cfg.get("process_filters", [])
            if _normalize_text(item)
        ]

    tenant_filters: List[str] = []
    raw_tenant = sync_cfg.get("tenant_filters")
    if isinstance(raw_tenant, list):
        tenant_filters = [_normalize_text(item) for item in raw_tenant if _normalize_text(item)]
    elif isinstance(camunda_cfg.get("tenant_filters"), list):
        tenant_filters = [
            _normalize_text(item)
            for item in camunda_cfg.get("tenant_filters", [])
            if _normalize_text(item)
        ]
    tenant_single = _normalize_text(sync_cfg.get("tenant_id") or camunda_cfg.get("tenant_id"))
    if tenant_single and tenant_single not in tenant_filters:
        tenant_filters.append(tenant_single)

    return process_filters, tenant_filters


def _inject_xes_dataset_name(mapping: Dict[str, Any], dataset_name: str) -> Dict[str, Any]:
    result = dict(mapping)
    xes_raw = result.get("xes_adapter", {})
    xes_cfg = dict(xes_raw) if isinstance(xes_raw, dict) else {}
    xes_cfg.setdefault("dataset_name", dataset_name)
    result["xes_adapter"] = xes_cfg
    return result


def _iter_xes_paths(log_path: str) -> List[Path]:
    path = Path(str(log_path)).resolve()
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise ValueError(f"XES log_path does not exist: {log_path}")
    candidates = list(path.glob("*.xes")) + list(path.glob("*.mxml"))
    return sorted(candidates)


def _trace_to_process_events(
    trace: Any,
    *,
    fallback_process: str,
) -> tuple[List[ProcessEventDTO], int, int]:
    events_out: List[ProcessEventDTO] = []
    total = 0
    valid = 0
    # Version policy for XES stats:
    # 1) use trace.process_version (concept:version or dataset fallback from XES adapter)
    # 2) fallback to resolved process namespace only when version is absent.
    trace_version = _normalize_text(getattr(trace, "process_version", "")) or _normalize_text(fallback_process) or fallback_process
    case_id = _normalize_text(getattr(trace, "case_id", "")) or "unknown_case"
    for idx, event in enumerate(getattr(trace, "events", []) or []):
        total += 1
        try:
            ts = float(getattr(event, "timestamp", 0.0))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(ts):
            continue
        valid += 1
        end_dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        duration_sec = float(getattr(event, "duration", 0.0) or 0.0)
        duration_sec = max(0.0, duration_sec)
        start_dt = end_dt - timedelta(seconds=duration_sec)
        activity = _normalize_text(getattr(event, "activity_id", "")) or "UNKNOWN_ACTIVITY"
        resource = _normalize_text(getattr(event, "resource_id", "")) or "UNKNOWN"
        act_inst_id = _normalize_text(getattr(event, "activity_instance_id", "")) or f"{case_id}:{idx}"
        extra_raw = getattr(event, "extra", {})
        extra = dict(extra_raw) if isinstance(extra_raw, dict) else {}
        extra.setdefault("source_adapter", "xes")
        events_out.append(
            ProcessEventDTO(
                case_id=case_id,
                activity_def_id=activity,
                activity_name=activity,
                activity_type=_normalize_text(extra.get("activity_type")) or None,
                proc_def_key=trace_version,
                proc_def_version=trace_version,
                act_inst_id=act_inst_id,
                sequence_counter=idx,
                start_time=start_dt,
                end_time=end_dt,
                duration_ms=float(duration_sec * 1000.0),
                assignee=resource,
                assigned_executor=resource,
                executed_by=resource,
                is_concurrent=False,
                extra=extra,
            )
        )
    return events_out, total, valid


def _load_xes_events_by_process(
    *,
    config: Dict[str, Any],
    selected_processes: Sequence[str],
    show_progress: bool,
    split_strategy: str,
    train_ratio: float,
    fraction: float,
) -> tuple[Dict[str, List[ProcessEventDTO]], Dict[str, float], Dict[str, datetime | None], Dict[str, Dict[str, int]]]:
    mapping = config.get("mapping", {})
    if not isinstance(mapping, dict):
        mapping = {}
    data_cfg = config.get("data", {})
    if not isinstance(data_cfg, dict):
        data_cfg = {}
    log_path = _normalize_text(data_cfg.get("log_path"))
    if not log_path:
        raise ValueError("sync-stats for xes requires data.log_path.")
    dataset_name_cfg = _normalize_text(data_cfg.get("dataset_name"))

    process_set = {_normalize_text(item) for item in selected_processes if _normalize_text(item)}
    process_set_l = {item.lower(): item for item in process_set}
    adapter = XESAdapter()
    traces_by_process: Dict[str, List[Any]] = defaultdict(list)

    paths = _iter_xes_paths(log_path)
    iterator: Iterable[Path] = paths
    if show_progress:
        iterator = tqdm(paths, desc="sync-stats xes files", unit="file")

    for file_path in iterator:
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
            traces_by_process[process_namespace].append(trace)

    process_events: Dict[str, List[ProcessEventDTO]] = defaultdict(list)
    coverage: Dict[str, float] = {}
    effective_as_of: Dict[str, datetime | None] = {}
    selection_meta: Dict[str, Dict[str, int]] = {}
    target_processes = sorted(process_set or set(traces_by_process.keys()))
    for process_namespace in target_processes:
        process_traces = traces_by_process.get(process_namespace, [])
        selected_traces, max_ts, total_traces, selected_traces_count = _select_train_traces(
            process_traces,
            split_strategy=split_strategy,
            train_ratio=train_ratio,
            fraction=fraction,
        )
        total_count = 0
        valid_count = 0
        for trace in selected_traces:
            events, total, valid = _trace_to_process_events(trace, fallback_process=process_namespace)
            if events:
                process_events[process_namespace].extend(events)
            total_count += total
            valid_count += valid
        coverage[process_namespace] = float((valid_count / total_count) * 100.0) if total_count > 0 else 0.0
        effective_as_of[process_namespace] = max_ts
        selection_meta[process_namespace] = {
            "total_traces": int(total_traces),
            "selected_traces": int(selected_traces_count),
            "total_events": int(total_count),
            "selected_events": int(valid_count),
        }

    return dict(process_events), coverage, effective_as_of, selection_meta


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser(description="Refresh process statistics snapshots (Stage 3.4 Tier A).")
    parser.add_argument("--config", required=True, help="Experiment config path.")
    parser.add_argument("--out", default="outputs/sync_stats_summary.json", help="Summary output JSON path.")
    parser.add_argument("--as-of", dest="as_of", default="", help="Optional ISO timestamp cutoff (UTC).")
    args = parser.parse_args(argv)

    config = load_yaml_config(args.config)
    mapping = config.get("mapping", {})
    if not isinstance(mapping, dict):
        mapping = {}
    adapter_kind = _normalize_text(mapping.get("adapter", "camunda")).lower() or "camunda"
    if adapter_kind not in {"camunda", "xes"}:
        raise ValueError("sync-stats supports mapping.adapter in {'camunda','xes'}.")

    sync_cfg_raw = config.get("sync_stats", {})
    sync_cfg = dict(sync_cfg_raw) if isinstance(sync_cfg_raw, dict) else {}
    if not bool(sync_cfg.get("enabled", True)):
        logger.info("sync_stats.enabled=false -> nothing to do.")
        return 0

    as_of_input = _normalize_text(args.as_of or sync_cfg.get("as_of"))
    as_of_ts_explicit = _parse_as_of(as_of_input) if as_of_input else None
    stats_time_policy = _normalize_text(sync_cfg.get("stats_time_policy", "strict_asof")).lower() or "strict_asof"
    process_scope_policy = _normalize_text(sync_cfg.get("process_scope_policy", "up_to_target_version")).lower() or "up_to_target_version"
    split_strategy, train_ratio, fraction = _resolve_train_cut_config(config)
    windows_days = sync_cfg.get("windows_days", [7, 30, 90])
    if not isinstance(windows_days, list):
        windows_days = [7, 30, 90]
    windows_days = [int(item) for item in windows_days if int(item) > 0]
    if not windows_days:
        windows_days = [7, 30, 90]

    confidence_weights_raw = sync_cfg.get("confidence_weights", {})
    confidence_weights = dict(confidence_weights_raw) if isinstance(confidence_weights_raw, dict) else {}
    if not confidence_weights:
        confidence_weights = {"sample_size": 0.4, "freshness": 0.3, "coverage": 0.3}

    freshness_half_life_days = float(sync_cfg.get("freshness_half_life_days", 14.0) or 14.0)
    quality_gate_cfg = _resolve_quality_gate_config(sync_cfg)
    alignment_gate_cfg = _resolve_alignment_gate_config(sync_cfg)
    show_progress = bool(sync_cfg.get("show_progress", True))

    knowledge_repo = build_knowledge_graph_repository(config)
    knowledge_settings = get_knowledge_graph_settings(config)

    all_processes = knowledge_repo.list_process_names()
    process_filters, tenant_filters = _derive_process_tenant_filters(config, sync_cfg)
    selected_processes = _select_process_names(
        all_processes=all_processes,
        process_filters=process_filters,
        tenant_filters=tenant_filters,
    )

    logger.info("========== SYNC-STATS PROFILE ==========")
    logger.info(
        "SYNC_STATS_PROFILE adapter=%s stats_time_policy=%s process_scope_policy=%s split=%s train_ratio=%.4f fraction=%.4f as_of=%s",
        adapter_kind,
        stats_time_policy,
        process_scope_policy,
        split_strategy,
        train_ratio,
        fraction,
        as_of_ts_explicit.isoformat() if as_of_ts_explicit is not None else "auto:max_event_ts_per_process",
    )
    logger.info(
        "SYNC_STATS_GATES quality=%s(on_fail=%s) alignment=%s(on_fail=%s) selected_processes=%d/%d",
        "on" if bool(quality_gate_cfg.get("enabled", True)) else "off",
        quality_gate_cfg.get("on_fail", "write_with_flag"),
        "on" if bool(alignment_gate_cfg.get("enabled", True)) else "off",
        alignment_gate_cfg.get("on_fail", "write_with_flag"),
        len(selected_processes),
        len(all_processes),
    )
    logger.info("========================================")

    runtime: CamundaRuntimeAdapter | None = None
    xes_events: Dict[str, List[ProcessEventDTO]] = {}
    xes_coverage: Dict[str, float] = {}
    xes_as_of: Dict[str, datetime | None] = {}
    xes_selection: Dict[str, Dict[str, int]] = {}
    if adapter_kind == "camunda":
        runtime_cfg = _resolve_runtime_config(config)
        runtime = CamundaRuntimeAdapter(runtime_cfg)
    else:
        xes_events, xes_coverage, xes_as_of, xes_selection = _load_xes_events_by_process(
            config=config,
            selected_processes=selected_processes,
            show_progress=show_progress,
            split_strategy=split_strategy,
            train_ratio=train_ratio,
            fraction=fraction,
        )

    details: List[Dict[str, Any]] = []
    skipped_details: List[Dict[str, Any]] = []
    processed = 0
    skipped = 0

    process_iter: Iterable[str]
    process_iter = selected_processes
    if show_progress:
        process_iter = tqdm(selected_processes, desc="sync-stats processes", unit="process")

    for process_namespace in process_iter:
        versions = knowledge_repo.list_versions(process_name=process_namespace)
        if not versions:
            skipped += 1
            skipped_details.append(
                {
                    "process_name": process_namespace,
                    "version": None,
                    "reason": "no_versions_for_process",
                    "effective_as_of_ts": None,
                }
            )
            continue

        process_key, tenant_id = _split_namespace(process_namespace)
        if not process_key:
            skipped += 1
            skipped_details.append(
                {
                    "process_name": process_namespace,
                    "version": None,
                    "reason": "invalid_process_namespace",
                    "effective_as_of_ts": None,
                }
            )
            continue

        if adapter_kind == "camunda":
            if runtime is None:
                raise RuntimeError("Camunda runtime adapter is not initialized.")
            until_arg = (
                as_of_ts_explicit.replace(tzinfo=None)
                if (stats_time_policy == "strict_asof" and as_of_ts_explicit is not None)
                else None
            )
            source_events, diagnostics = runtime.fetch_historic_activity_events(
                process_name=process_key,
                version_key="",
                since=None,
                until=until_arg,
            )
            source_events = _filter_by_tenant(source_events, tenant_id)
            all_events, derived_as_of, total_cases, selected_cases = _select_train_events(
                source_events,
                split_strategy=split_strategy,
                train_ratio=train_ratio,
                fraction=fraction,
            )
            coverage_percent = float(diagnostics.history_coverage_percent or 0.0)
            selection_info = {
                "total_cases": int(total_cases),
                "selected_cases": int(selected_cases),
                "total_events": int(len(source_events)),
                "selected_events": int(len(all_events)),
            }
        else:
            all_events = list(xes_events.get(process_namespace, []))
            coverage_percent = float(xes_coverage.get(process_namespace, 0.0))
            derived_as_of = xes_as_of.get(process_namespace)
            selection_info = dict(xes_selection.get(process_namespace, {}))

        effective_as_of = as_of_ts_explicit or derived_as_of
        logger.info(
            "sync-stats process=%s events=%d coverage=%.2f%% split=%s train_ratio=%.4f fraction=%.4f effective_as_of=%s",
            process_namespace,
            len(all_events),
            coverage_percent,
            split_strategy,
            train_ratio,
            fraction,
            effective_as_of.isoformat() if effective_as_of is not None else "none",
        )
        if effective_as_of is None:
            logger.warning(
                "No events available for process=%s after train-cut. Skip all versions.",
                process_namespace,
            )
            for version in versions:
                skipped_details.append(
                    {
                        "process_name": process_namespace,
                        "version": version,
                        "reason": "no_events_after_train_cut",
                        "effective_as_of_ts": None,
                        **selection_info,
                    }
                )
            skipped += len(versions)
            continue

        version_iter: Iterable[str]
        version_iter = sorted(versions, key=lambda item: (_version_rank(item) is None, _version_rank(item) or 10**9, item))
        if show_progress:
            version_iter = tqdm(list(version_iter), desc=f"{process_namespace} versions", unit="version", leave=False)

        for version in version_iter:
            dto = knowledge_repo.get_process_structure(version=version, process_name=process_namespace)
            if dto is None:
                skipped += 1
                skipped_details.append(
                    {
                        "process_name": process_namespace,
                        "version": version,
                        "reason": "process_structure_not_found",
                        "effective_as_of_ts": effective_as_of.isoformat(),
                    }
                )
                continue
            version_scope_events, process_scope_events = _collect_scope_events(
                all_events,
                target_version=version,
                process_scope_policy=process_scope_policy,
                as_of_ts=effective_as_of,
            )
            version_event_count = len(version_scope_events)
            process_event_count = len(process_scope_events)
            if version_event_count == 0 and process_event_count == 0:
                logger.warning(
                    "Skip stats snapshot for process=%s version=%s: no events matched scope policy up to as_of=%s.",
                    process_namespace,
                    version,
                    effective_as_of.isoformat(),
                )
                skipped += 1
                skipped_details.append(
                    {
                        "process_name": process_namespace,
                        "version": version,
                        "reason": "no_scope_events_up_to_as_of",
                        "effective_as_of_ts": effective_as_of.isoformat(),
                        "scope_events_version": int(version_event_count),
                        "scope_events_process": int(process_event_count),
                        **selection_info,
                    }
                )
                continue

            alignment_scope_used = "version" if version_event_count > 0 else "process"
            alignment_events = version_scope_events if version_event_count > 0 else process_scope_events
            alignment_summary = ActivityTopologyAlignmentService().evaluate(
                events=alignment_events,
                dto=dto,
                config=AlignmentGateConfig.from_mapping(alignment_gate_cfg),
                scope_used=alignment_scope_used,
            ).to_dict()
            alignment_rejected = bool(alignment_gate_cfg.get("enabled", True)) and (not bool(alignment_summary.get("is_aligned", True)))
            if alignment_rejected and alignment_gate_cfg.get("warn_on_fail", True):
                logger.warning(
                    "Alignment gate rejected stats snapshot for process=%s version=%s: reason=%s failures=%s event_match_ratio=%.4f unique_activity_coverage=%.4f node_coverage=%.4f.",
                    process_namespace,
                    version,
                    alignment_summary.get("alignment_reason"),
                    ",".join(str(item) for item in alignment_summary.get("alignment_failures", [])),
                    float(alignment_summary.get("event_match_ratio", 0.0)),
                    float(alignment_summary.get("unique_activity_coverage", 0.0)),
                    float(alignment_summary.get("node_coverage", 0.0)),
                )
            if alignment_rejected:
                on_fail = str(alignment_gate_cfg.get("on_fail", "write_with_flag"))
                if on_fail == "raise":
                    raise ValueError(
                        f"Alignment gate rejected process='{process_namespace}' version='{version}': "
                        f"{alignment_summary.get('alignment_reason')}"
                    )
                if on_fail == "skip_snapshot":
                    skipped += 1
                    skipped_details.append(
                        {
                            "process_name": process_namespace,
                            "version": version,
                            "reason": "alignment_gate_rejected",
                            "effective_as_of_ts": effective_as_of.isoformat(),
                            "alignment_is_ok": bool(alignment_summary.get("is_aligned", False)),
                            "alignment_reason": str(alignment_summary.get("alignment_reason", "rejected")),
                            "alignment_failures": list(alignment_summary.get("alignment_failures", [])),
                            "alignment_event_match_ratio": float(alignment_summary.get("event_match_ratio", 0.0)),
                            "alignment_unique_activity_coverage": float(
                                alignment_summary.get("unique_activity_coverage", 0.0)
                            ),
                            "alignment_node_coverage": float(alignment_summary.get("node_coverage", 0.0)),
                            "alignment_scope_used": str(alignment_summary.get("scope_used", "none")),
                            "scope_events_version": int(version_event_count),
                            "scope_events_process": int(process_event_count),
                            **selection_info,
                        }
                    )
                    continue

            enriched_dto, stat_summary = _build_stats_for_dto(
                dto=dto,
                process_namespace=process_namespace,
                all_events=all_events,
                as_of_ts=effective_as_of,
                windows_days=windows_days,
                stats_time_policy=stats_time_policy,
                process_scope_policy=process_scope_policy,
                coverage_percent=coverage_percent,
                confidence_weights=confidence_weights,
                freshness_half_life_days=freshness_half_life_days,
                split_strategy=split_strategy,
                train_ratio=train_ratio,
                fraction=fraction,
                quality_gate_cfg=quality_gate_cfg,
                alignment_summary=alignment_summary,
            )

            logger.info(
                "SYNC_STATS_CHECK process=%s version=%s alignment=%s(reason=%s) quality=%s(reason=%s)",
                process_namespace,
                version,
                "pass" if bool(stat_summary.get("alignment_is_ok", True)) else "fail",
                stat_summary.get("alignment_reason"),
                "pass" if bool(stat_summary.get("is_usable_for_training", True)) else "fail",
                stat_summary.get("quality_reason"),
            )

            quality_rejected = quality_gate_cfg.get("enabled", True) and (not bool(stat_summary.get("is_usable_for_training", True)))
            if quality_rejected and quality_gate_cfg.get("warn_on_fail", True):
                logger.warning(
                    "Quality gate rejected stats snapshot for process=%s version=%s: reason=%s failures=%s.",
                    process_namespace,
                    version,
                    stat_summary.get("quality_reason"),
                    ",".join(str(item) for item in stat_summary.get("quality_failures", [])),
                )

            if quality_rejected and str(quality_gate_cfg.get("on_fail", "write_with_flag")) == "skip_snapshot":
                skipped += 1
                skipped_details.append(
                    {
                        "process_name": process_namespace,
                        "version": version,
                        "reason": "quality_gate_rejected",
                        "effective_as_of_ts": effective_as_of.isoformat(),
                        "quality_reason": stat_summary.get("quality_reason"),
                        "quality_failures": list(stat_summary.get("quality_failures", [])),
                        **selection_info,
                    }
                )
                continue

            knowledge_version = knowledge_repo.save_process_structure_snapshot(
                version=version,
                dto=enriched_dto,
                process_name=process_namespace,
                as_of_ts=effective_as_of,
            )

            details.append(
                {
                    **stat_summary,
                    "knowledge_version": knowledge_version,
                    "tenant_id": tenant_id or None,
                    "effective_as_of_ts": effective_as_of.isoformat(),
                    "scope_events_version": int(version_event_count),
                    "scope_events_process": int(process_event_count),
                    **selection_info,
                }
            )
            processed += 1

    summary = {
        "status": "ok",
        "mode": "sync-stats",
        "adapter": adapter_kind,
        "tier": "A",
        "stats_contract_version": _STATS_CONTRACT_VERSION,
        "stats_time_policy": stats_time_policy,
        "process_scope_policy": process_scope_policy,
        "as_of_ts": as_of_ts_explicit.isoformat() if as_of_ts_explicit is not None else "auto:max_event_ts_per_process",
        "quality_gate": dict(quality_gate_cfg),
        "alignment_gate": dict(alignment_gate_cfg),
        "split_strategy": split_strategy,
        "train_ratio": float(train_ratio),
        "fraction": float(fraction),
        "knowledge_backend": knowledge_settings.get("backend", "in_memory"),
        "processed_versions": processed,
        "skipped_versions": skipped,
        "processes_total": len(all_processes),
        "processes_selected": len(selected_processes),
        "details": details,
        "skipped_details": skipped_details,
    }

    out_path = Path(str(args.out)).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("sync-stats finished: processed=%d skipped=%d", processed, skipped)
    logger.info("Summary saved to %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
