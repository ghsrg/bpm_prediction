"""Domain DTO for a normalized process trace."""

# Відповідно до:
# - AGENT_GUIDE.MD -> розділ 4 (Зафіксовані контракти даних DTO)
# - DATA_MODEL_MVP1.MD -> розділ 3.2 RawTrace
# - ADAPTER_XES.MD -> розділ 5 (Структура DTO Контрактів)

from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel

from .event_record import EventRecord


class RawTrace(BaseModel):
    """Canonical trace with ordered normalized events."""

    case_id: str
    process_version: str = "1"
    events: List[EventRecord]
    trace_attributes: Dict[str, Any]
