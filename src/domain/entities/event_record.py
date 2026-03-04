"""Domain DTO for normalized event records."""

# Відповідно до:
# - AGENT_GUIDE.MD -> розділ 4 (Зафіксовані контракти даних DTO)
# - DATA_MODEL_MVP1.MD -> розділ 3.1 EventRecord
# - ADAPTER_XES.MD -> розділ 5 (Структура DTO Контрактів)

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel


class EventRecord(BaseModel):
    """Canonical normalized event used after ingestion."""

    activity_id: str
    timestamp: float
    resource_id: str
    lifecycle: Optional[str]
    position_in_trace: int
    duration: float
    time_since_case_start: float
    time_since_previous_event: float
    extra: Dict[str, Any]
    activity_instance_id: Optional[str]
