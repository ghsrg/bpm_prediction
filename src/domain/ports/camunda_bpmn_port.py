"""Port contract for Camunda BPMN structure source."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol


class ICamundaBpmnPort(Protocol):
    """Read Camunda process-definition catalog and BPMN payloads."""

    def fetch_procdef_catalog(
        self,
        process_name: Optional[str] = None,
        process_filters: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Return normalized process-definition rows."""
        ...

    def fetch_bpmn_xml(self, proc_def_id: str) -> Optional[str]:
        """Return BPMN XML content for process definition id."""
        ...
