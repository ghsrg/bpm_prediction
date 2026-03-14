"""In-memory implementation of Stage 3.1 instance-graph persistence port."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from src.domain.entities.process_event import ProcessEventDTO
from src.domain.ports.instance_graph_port import IInstanceGraphPort


class InMemoryInstanceGraphRepository(IInstanceGraphPort):
    """Store events, graphs, projections, and watermarks in-memory."""

    def __init__(self) -> None:
        self._events: Dict[Tuple[str, str], List[ProcessEventDTO]] = {}
        self._graphs: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._projections: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._watermarks: Dict[Tuple[str, str], datetime] = {}

    def save_instance_events(
        self,
        process_name: str,
        version_key: str,
        events: List[ProcessEventDTO],
    ) -> None:
        key = self._norm_key(process_name, version_key)
        existing = self._events.get(key, [])
        merged = existing + list(events)
        self._events[key] = self._deduplicate_events(merged)

    def save_instance_graph(
        self,
        process_name: str,
        version_key: str,
        graph: Dict[str, Any],
    ) -> None:
        self._graphs[self._norm_key(process_name, version_key)] = dict(graph)

    def save_instance_projection(
        self,
        process_name: str,
        version_key: str,
        projection: Dict[str, Any],
    ) -> None:
        self._projections[self._norm_key(process_name, version_key)] = dict(projection)

    def get_instance_events(
        self,
        process_name: str,
        version_key: str,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
    ) -> List[ProcessEventDTO]:
        key = self._norm_key(process_name, version_key)
        events = self._events.get(key, [])
        if since is None and until is None:
            return list(events)
        return [event for event in events if self._in_interval(event, since, until)]

    def get_instance_graph(self, process_name: str, version_key: str) -> Optional[Dict[str, Any]]:
        payload = self._graphs.get(self._norm_key(process_name, version_key))
        return None if payload is None else dict(payload)

    def get_instance_projection(self, process_name: str, version_key: str) -> Optional[Dict[str, Any]]:
        payload = self._projections.get(self._norm_key(process_name, version_key))
        return None if payload is None else dict(payload)

    def get_last_watermark(self, process_name: str, version_key: str) -> Optional[datetime]:
        return self._watermarks.get(self._norm_key(process_name, version_key))

    def set_last_watermark(self, process_name: str, version_key: str, watermark: datetime) -> None:
        self._watermarks[self._norm_key(process_name, version_key)] = watermark

    @staticmethod
    def _norm_key(process_name: str, version_key: str) -> Tuple[str, str]:
        process = str(process_name).strip() or "default_process"
        version = str(version_key).strip() or "default_version"
        return process, version

    @staticmethod
    def _event_unique_key(event: ProcessEventDTO) -> str:
        return "|".join(
            [
                event.case_id,
                str(event.act_inst_id or event.task_id or ""),
                str(event.activity_def_id),
                str(event.start_time.isoformat() if event.start_time else ""),
                str(event.end_time.isoformat() if event.end_time else ""),
            ]
        )

    @classmethod
    def _deduplicate_events(cls, events: List[ProcessEventDTO]) -> List[ProcessEventDTO]:
        unique: Dict[str, ProcessEventDTO] = {}
        for event in events:
            unique[cls._event_unique_key(event)] = event
        return sorted(
            unique.values(),
            key=lambda item: (
                item.case_id,
                item.start_time or datetime.min,
                item.end_time or datetime.min,
                item.activity_def_id,
            ),
        )

    @staticmethod
    def _in_interval(
        event: ProcessEventDTO,
        since: Optional[datetime],
        until: Optional[datetime],
    ) -> bool:
        event_ts = event.end_time or event.start_time
        if event_ts is None:
            return since is None and until is None
        if since is not None and event_ts < since:
            return False
        if until is not None and event_ts > until:
            return False
        return True

