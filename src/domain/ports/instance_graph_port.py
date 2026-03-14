"""Domain port for Stage 3.1 instance graph persistence."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol

from src.domain.entities.process_event import ProcessEventDTO


class IInstanceGraphPort(Protocol):
    """Port for saving and reading canonical instance-graph artifacts."""

    def save_instance_events(
        self,
        process_name: str,
        version_key: str,
        events: List[ProcessEventDTO],
    ) -> None:
        """Persist deduplicated execution-level events."""
        ...

    def save_instance_graph(
        self,
        process_name: str,
        version_key: str,
        graph: Dict[str, Any],
    ) -> None:
        """Persist canonical instance graph payload."""
        ...

    def save_instance_projection(
        self,
        process_name: str,
        version_key: str,
        projection: Dict[str, Any],
    ) -> None:
        """Persist collapsed/aggregated projection payload."""
        ...

    def get_instance_events(
        self,
        process_name: str,
        version_key: str,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
    ) -> List[ProcessEventDTO]:
        """Return events filtered by optional time interval."""
        ...

    def get_instance_graph(self, process_name: str, version_key: str) -> Optional[Dict[str, Any]]:
        """Return saved canonical graph payload if present."""
        ...

    def get_instance_projection(self, process_name: str, version_key: str) -> Optional[Dict[str, Any]]:
        """Return saved projection payload if present."""
        ...

    def get_last_watermark(self, process_name: str, version_key: str) -> Optional[datetime]:
        """Return last persisted watermark for incremental reads."""
        ...

    def set_last_watermark(self, process_name: str, version_key: str, watermark: datetime) -> None:
        """Persist new watermark after successful processing."""
        ...

