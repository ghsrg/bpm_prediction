"""Domain port for Stage 3.1 Camunda runtime ingestion."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol, Tuple

from src.domain.entities.process_event import ProcessEventDTO
from src.domain.entities.runtime_fetch_diagnostics import RuntimeFetchDiagnosticsDTO


class ICamundaRuntimePort(Protocol):
    """Port contract for fetching Camunda runtime/history data in resilient way."""

    def fetch_historic_activity_events(
        self,
        process_name: str,
        version_key: str,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
    ) -> Tuple[List[ProcessEventDTO], RuntimeFetchDiagnosticsDTO]:
        """Return activity events and diagnostics without raising on missing history."""
        ...

    def fetch_historic_task_events(
        self,
        process_name: str,
        version_key: str,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """Return task-level rows (nullable fields allowed)."""
        ...

    def fetch_identity_links(
        self,
        process_name: str,
        version_key: str,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """Return identity links for assignee/candidate enrichments."""
        ...

    def fetch_runtime_execution_tree(
        self,
        process_name: str,
        version_key: str,
        depth_limit: int,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """Return bounded execution-tree rows used for parallel/scope refinement."""
        ...

    def fetch_multi_instance_variables(
        self,
        process_name: str,
        version_key: str,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """Return loopCounter and nrOf* variables from runtime/history variable tables."""
        ...

