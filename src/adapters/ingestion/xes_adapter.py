"""Streaming XES adapter implementation."""

# Відповідно до:
# - AGENT_GUIDE.MD -> розділ 2 (No DOM parsing), розділ 4 (DTO contract), розділ 6 (doc links)
# - ADAPTER_XES.MD -> розділ 4 (парсинг/дельти/lifecycle/pairing), розділ 7 (streaming Iterator[RawTrace])
# - DATA_FLOWS_MVP1.MD -> розділ 2.1 (IXESAdapter ingestion contract)
# - DATA_MODEL_MVP1.MD -> розділ 3 (External Boundary Objects)

from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, Iterator, List, Optional, Sequence, Set, Tuple
import logging

from lxml import etree

from src.application.ports.xes_adapter_port import IXESAdapter
from src.domain.entities.feature_config import FeatureConfig, parse_feature_configs
from src.domain.entities.event_record import EventRecord
from src.domain.entities.raw_trace import RawTrace

logger = logging.getLogger(__name__)


_XES_ATTR_TAGS = {"string", "date", "int", "float", "boolean", "id", "list"}
_DEFAULT_COMPLETE_TRANSITIONS = {"complete", "ate_abort", "pi_abort", "manualskip", "autoskip"}


class XESAdapter(IXESAdapter):
    """Streaming parser for XES logs that emits canonical RawTrace DTOs."""

    def read(self, file_path: str, mapping_config: dict) -> Iterator[RawTrace]:
        """Stream traces from XES without loading the full XML into memory."""
        config = mapping_config.get("xes_adapter", mapping_config)
        feature_configs = parse_feature_configs(mapping_config)

        role_keys = _resolve_role_keys(feature_configs)

        case_id_key = config.get("case_id_key", "concept:name")
        activity_key = role_keys.get("activity") or config.get("activity_key", "concept:name")
        timestamp_key = role_keys.get("timestamp") or config.get("timestamp_key", "time:timestamp")
        resource_key = role_keys.get("resource") or config.get("resource_key", "org:resource")
        lifecycle_key = role_keys.get("lifecycle") or config.get("lifecycle_key", "lifecycle:transition")
        version_key = role_keys.get("version") or config.get("version_key", "concept:version")
        complete_transitions = {
            str(v).strip().lower() for v in config.get("complete_transitions", list(_DEFAULT_COMPLETE_TRANSITIONS))
        }
        pairing_strategy = str(config.get("pairing_strategy", "lifo")).strip().lower()
        use_classifier = bool(config.get("use_classifier", True))
        extra_trace_keys = _normalize_key_set(config.get("extra_trace_keys"))
        extra_event_keys = _normalize_key_set(config.get("extra_event_keys"))

        log_attributes: Dict[str, Any] = {}
        classifiers: Dict[str, List[str]] = {}
        processed_traces = 0
        produced_events = 0
        skipped_events = 0

        try:
            context = etree.iterparse(file_path, events=("end",), recover=False)
            for _, elem in context:
                tag = _local_name(elem.tag)
                parent = elem.getparent()
                parent_tag = _local_name(parent.tag) if parent is not None else None

                if parent_tag == "log" and tag in _XES_ATTR_TAGS:
                    log_attributes.update(_extract_xes_attributes(elem))

                elif parent_tag == "log" and tag == "classifier":
                    name = (elem.get("name") or "").strip()
                    keys_raw = (elem.get("keys") or "").strip()
                    if name and keys_raw:
                        classifiers[name] = [key for key in keys_raw.split() if key]

                elif tag == "trace":
                    raw_trace, trace_skips = self._parse_trace(
                        trace_elem=elem,
                        file_path=file_path,
                        case_id_key=case_id_key,
                        activity_key=activity_key,
                        timestamp_key=timestamp_key,
                        resource_key=resource_key,
                        lifecycle_key=lifecycle_key,
                        version_key=version_key,
                        complete_transitions=complete_transitions,
                        pairing_strategy=pairing_strategy,
                        use_classifier=use_classifier,
                        classifiers=classifiers,
                        log_attributes=log_attributes,
                        extra_trace_keys=extra_trace_keys,
                        extra_event_keys=extra_event_keys,
                        feature_configs=feature_configs,
                    )
                    skipped_events += trace_skips
                    processed_traces += 1
                    produced_events += len(raw_trace.events)
                    yield raw_trace

                    # Критично для стрімінгу: очищення обробленого піддерева trace з пам'яті.
                    elem.clear()
                    while elem.getprevious() is not None:
                        del elem.getparent()[0]

            logger.info(
                "Processed %s traces, %s events, %s skipped",
                processed_traces,
                produced_events,
                skipped_events,
            )
        except etree.XMLSyntaxError:
            logger.error("Invalid XML/XES structure in file: %s", file_path)
            raise

    def _parse_trace(
        self,
        *,
        trace_elem: etree._Element,
        file_path: str,
        case_id_key: str,
        activity_key: Any,
        timestamp_key: str,
        resource_key: str,
        lifecycle_key: str,
        version_key: str,
        complete_transitions: set[str],
        pairing_strategy: str,
        use_classifier: bool,
        classifiers: Dict[str, List[str]],
        log_attributes: Dict[str, Any],
        extra_trace_keys: Set[str],
        extra_event_keys: Set[str],
        feature_configs: Sequence[FeatureConfig],
    ) -> Tuple[RawTrace, int]:
        trace_attributes: Dict[str, Any] = {}
        event_payloads: List[Dict[str, Any]] = []
        skipped_events = 0

        for child in trace_elem:
            tag = _local_name(child.tag)
            if tag in _XES_ATTR_TAGS:
                trace_attributes.update(_extract_xes_attributes(child))
            elif tag == "event":
                event_attrs: Dict[str, Any] = {}
                for event_child in child:
                    event_attrs.update(_extract_xes_attributes(event_child))
                event_payloads.append(event_attrs)

        case_id = str(trace_attributes.get(case_id_key) or "UNKNOWN_CASE")
        classifier_keys = _select_classifier_keys(classifiers) if use_classifier else []
        selected_trace_attrs = {
            k: v for k, v in trace_attributes.items() if k not in {case_id_key, version_key}
        }
        selected_trace_attrs.update(_extract_typed_trace_features(trace_attributes, feature_configs))
        if extra_trace_keys:
            selected_trace_attrs = {k: v for k, v in selected_trace_attrs.items() if k in extra_trace_keys}

        starts_by_key: Dict[Tuple[str, ...], Deque[float]] = defaultdict(deque)
        completed_events_raw: List[Dict[str, Any]] = []
        has_lifecycle_data = any(payload.get(lifecycle_key) is not None for payload in event_payloads)

        for original_index, payload in enumerate(event_payloads):
            timestamp_epoch = _parse_timestamp(payload.get(timestamp_key))
            if timestamp_epoch is None:
                skipped_events += 1
                logger.warning("Skipping event with invalid timestamp in case_id=%s", case_id)
                continue

            lifecycle_value = _normalize_optional_str(payload.get(lifecycle_key))
            lifecycle_norm = lifecycle_value.lower() if lifecycle_value is not None else None

            activity_id = _resolve_activity_id(
                payload=payload,
                activity_key=activity_key,
                classifier_keys=classifier_keys,
            )
            resource_id = _normalize_optional_str(payload.get(resource_key)) or "UNKNOWN"
            instance_id = _normalize_optional_str(payload.get("lifecycle:instance")) or _normalize_optional_str(
                payload.get("concept:instance")
            )

            pairing_key = _build_pairing_key(
                activity_id=activity_id,
                resource_id=resource_id,
                instance_id=instance_id,
                strategy=pairing_strategy,
            )

            if lifecycle_norm == "start":
                starts_by_key[pairing_key].append(timestamp_epoch)
                continue

            if has_lifecycle_data and lifecycle_norm is not None and lifecycle_norm not in complete_transitions:
                continue

            duration = 0.0
            if has_lifecycle_data:
                duration = _match_duration(
                    starts_by_key=starts_by_key,
                    pairing_key=pairing_key,
                    end_timestamp=timestamp_epoch,
                    strategy=pairing_strategy,
                )

            mapped_keys = {
                timestamp_key,
                resource_key,
                lifecycle_key,
                version_key,
                "lifecycle:instance",
                "concept:instance",
                *classifier_keys,
            }
            if isinstance(activity_key, str):
                mapped_keys.add(activity_key)
            elif isinstance(activity_key, Sequence):
                mapped_keys.update(str(k) for k in activity_key)

            # Зберігаємо лише явно дозволені event-level extra ключі з mapping config.
            extra = {k: v for k, v in payload.items() if k not in mapped_keys}
            extra.update(_extract_typed_event_features(payload, feature_configs))
            if extra_event_keys:
                extra = {k: v for k, v in extra.items() if k in extra_event_keys}
            # Дублюємо trace-level extra у кожну подію, щоб вузол мав доступ до контексту кейсу.
            for trace_key, trace_value in selected_trace_attrs.items():
                extra.setdefault(trace_key, trace_value)

            completed_events_raw.append(
                {
                    "activity_id": activity_id,
                    "timestamp": timestamp_epoch,
                    "resource_id": resource_id,
                    "lifecycle": lifecycle_value,
                    "duration": duration,
                    "activity_instance_id": instance_id,
                    "extra": extra,
                    "original_index": original_index,
                    "event_version": _normalize_optional_str(payload.get(version_key)),
                }
            )

        completed_events_raw.sort(key=lambda item: (item["timestamp"], item["original_index"]))

        normalized_events: List[EventRecord] = []
        case_start_time = completed_events_raw[0]["timestamp"] if completed_events_raw else 0.0
        prev_time = case_start_time

        for pos, event in enumerate(completed_events_raw):
            current_time = float(event["timestamp"])
            time_since_case_start = 0.0 if pos == 0 else max(0.0, current_time - case_start_time)
            time_since_previous = 0.0 if pos == 0 else max(0.0, current_time - prev_time)
            prev_time = current_time

            # Додаємо обчислені базові часові фічі у event.extra для конфігурованого FeatureEncoder.
            enriched_extra = dict(event["extra"])
            enriched_extra.setdefault("duration", max(0.0, float(event["duration"])))
            enriched_extra.setdefault("time_since_case_start", time_since_case_start)
            enriched_extra.setdefault("time_since_previous_event", time_since_previous)
            enriched_extra.setdefault(timestamp_key, current_time)
            enriched_extra.setdefault(activity_key if isinstance(activity_key, str) else "concept:name", event["activity_id"])
            enriched_extra.setdefault(resource_key, event["resource_id"])

            normalized_events.append(
                EventRecord(
                    activity_id=event["activity_id"],
                    timestamp=current_time,
                    resource_id=event["resource_id"],
                    lifecycle=event["lifecycle"],
                    position_in_trace=pos,
                    duration=max(0.0, float(event["duration"])),
                    time_since_case_start=time_since_case_start,
                    time_since_previous_event=time_since_previous,
                    extra=enriched_extra,
                    activity_instance_id=event["activity_instance_id"],
                )
            )

        process_version = _resolve_process_version(
            event_candidates=[item.get("event_version") for item in completed_events_raw],
            trace_version=_normalize_optional_str(trace_attributes.get(version_key)),
            log_version=_normalize_optional_str(log_attributes.get(version_key)),
            file_path=file_path,
        )

        cleaned_trace_attributes = selected_trace_attrs

        return (
            RawTrace(
                case_id=case_id,
                process_version=process_version,
                events=normalized_events,
                trace_attributes=cleaned_trace_attributes,
            ),
            skipped_events,
        )


def _resolve_activity_id(payload: Dict[str, Any], activity_key: Any, classifier_keys: List[str]) -> str:
    """Resolve activity_id using classifier first, then mapping activity_key fallback."""
    if classifier_keys:
        parts = [str(payload.get(key)).strip() for key in classifier_keys if payload.get(key) is not None]
        if parts:
            return "+".join(parts)

    if isinstance(activity_key, str):
        return str(payload.get(activity_key) or "UNKNOWN_ACTIVITY")
    if isinstance(activity_key, Sequence):
        parts = [str(payload.get(str(key))).strip() for key in activity_key if payload.get(str(key)) is not None]
        return "+".join(parts) if parts else "UNKNOWN_ACTIVITY"
    return "UNKNOWN_ACTIVITY"


def _select_classifier_keys(classifiers: Dict[str, List[str]]) -> List[str]:
    """Select classifier keys with Activity classifier priority."""
    if not classifiers:
        return []
    if "Activity" in classifiers:
        return classifiers["Activity"]
    return next(iter(classifiers.values()))


def _build_pairing_key(
    *,
    activity_id: str,
    resource_id: str,
    instance_id: Optional[str],
    strategy: str,
) -> Tuple[str, ...]:
    """Construct pairing key according to ADAPTER_XES pairing priority rules."""
    if strategy == "by_instance" and instance_id:
        return (activity_id, instance_id)
    if resource_id and resource_id != "UNKNOWN":
        return (activity_id, resource_id)
    return (activity_id,)


def _match_duration(
    *,
    starts_by_key: Dict[Tuple[str, ...], Deque[float]],
    pairing_key: Tuple[str, ...],
    end_timestamp: float,
    strategy: str,
) -> float:
    """Match start/end events using selected LIFO/FIFO strategy."""
    starts = starts_by_key.get(pairing_key)
    if not starts:
        return 0.0

    if strategy == "fifo":
        start_time = starts.popleft()
    else:
        start_time = starts.pop()

    return max(0.0, end_timestamp - start_time)


def _resolve_process_version(
    *,
    event_candidates: List[Optional[str]],
    trace_version: Optional[str],
    log_version: Optional[str],
    file_path: str,
) -> str:
    """Resolve κ version with priority: event -> trace -> log -> filename -> default."""
    for candidate in event_candidates:
        if candidate:
            return candidate
    if trace_version:
        return trace_version
    if log_version:
        return log_version

    filename = Path(file_path).stem.strip()
    if filename:
        return filename
    return "default"


def _normalize_optional_str(value: Any) -> Optional[str]:
    """Convert a value to stripped string or return None for empty values."""
    if value is None:
        return None
    as_str = str(value).strip()
    return as_str if as_str else None


def _parse_timestamp(value: Any) -> Optional[float]:
    """Parse timestamp to UTC unix epoch float seconds."""
    if isinstance(value, (int, float)):
        return float(value)

    text = _normalize_optional_str(value)
    if text is None:
        return None

    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.timestamp()


def _extract_xes_attributes(elem: etree._Element, prefix: str = "") -> Dict[str, Any]:
    """Flatten XES nested attributes into dotted keys."""
    result: Dict[str, Any] = {}
    elem_key = elem.get("key")
    tag = _local_name(elem.tag)

    current_key: Optional[str]
    if elem_key:
        current_key = f"{prefix}.{elem_key}" if prefix else elem_key
    else:
        current_key = prefix if prefix else None

    children = list(elem)
    if children:
        next_prefix = current_key or prefix
        for child in children:
            result.update(_extract_xes_attributes(child, next_prefix))
        if tag == "list" and current_key and current_key not in result:
            result[current_key] = "{}"
        return result

    value = elem.get("value")
    if current_key is None:
        return result

    result[current_key] = _cast_xes_typed_value(value=value, tag=tag)
    return result


def _cast_xes_typed_value(value: Any, tag: str) -> Any:
    """Cast XES attribute by XML tag type (without value-based auto-cast)."""
    text = "" if value is None else str(value)
    if tag == "float":
        try:
            return float(text)
        except ValueError:
            return text
    if tag == "int":
        try:
            return int(text)
        except ValueError:
            return text
    if tag == "boolean":
        return text.strip().lower() == "true"
    return text


def _normalize_key_set(value: Any) -> Set[str]:
    """Normalize optional config list into a set of non-empty string keys."""
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return set()
    return {str(item).strip() for item in value if str(item).strip()}


def _resolve_role_keys(feature_configs: Sequence[FeatureConfig]) -> Dict[str, str]:
    """Resolve optional role->field mapping from feature configuration."""
    role_keys: Dict[str, str] = {}
    for cfg in feature_configs:
        if cfg.role in {"activity", "timestamp", "resource", "lifecycle", "version"}:
            role_keys[cfg.role] = cfg.name
    return role_keys


def _extract_typed_trace_features(
    trace_attributes: Dict[str, Any],
    feature_configs: Sequence[FeatureConfig],
) -> Dict[str, Any]:
    """Build typed+filled trace-level features according to FeatureConfig."""
    typed: Dict[str, Any] = {}
    for cfg in feature_configs:
        if cfg.source != "trace":
            continue
        raw = trace_attributes.get(cfg.name)
        typed[cfg.name] = _coerce_to_dtype(raw, cfg)
    return typed


def _extract_typed_event_features(payload: Dict[str, Any], feature_configs: Sequence[FeatureConfig]) -> Dict[str, Any]:
    """Build typed+filled event-level features according to FeatureConfig."""
    typed: Dict[str, Any] = {}
    for cfg in feature_configs:
        if cfg.source != "event":
            continue
        raw = payload.get(cfg.name)
        typed[cfg.name] = _coerce_to_dtype(raw, cfg)
    return typed


def _coerce_to_dtype(raw: Any, feature_cfg: FeatureConfig) -> Any:
    """Force-cast raw value to dtype from FeatureConfig with fill_na fallback."""
    if raw is None:
        return feature_cfg.fill_na

    dtype = feature_cfg.dtype
    try:
        if dtype == "string":
            text = str(raw).strip()
            return text if text else feature_cfg.fill_na
        if dtype == "float":
            return float(raw)
        if dtype == "int":
            return int(raw)
        if dtype == "boolean":
            if isinstance(raw, bool):
                return raw
            return str(raw).strip().lower() == "true"
        if dtype == "timestamp":
            ts = _parse_timestamp(raw)
            return ts if ts is not None else feature_cfg.fill_na
    except (TypeError, ValueError):
        return feature_cfg.fill_na

    return raw


def _local_name(tag: Any) -> str:
    """Extract local XML tag name from namespaced or plain tags."""
    if not isinstance(tag, str):
        return ""
    if "}" in tag:
        return tag.rsplit("}", 1)[-1]
    return tag
