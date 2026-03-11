"""Domain DTO for prefix-target training slices."""

# Відповідно до:
# - AGENT_GUIDE.MD -> розділ 2 (потік RawTrace -> PrefixSlice)
# - DATA_MODEL_MVP1.MD -> розділ 5.1 (PrefixSlice)
# - LLD_MVP1.MD -> розділ 3.1 (математика префіксів) та 3.2 (edge cases)

from __future__ import annotations

from typing import List

from pydantic import BaseModel

from src.domain.entities.event_record import EventRecord


class PrefixSlice(BaseModel):
    """Observed prefix σ_[1:k] with target event e_(k+1)."""

    case_id: str
    process_version: str = "1"
    prefix_events: List[EventRecord]
    target_event: EventRecord
