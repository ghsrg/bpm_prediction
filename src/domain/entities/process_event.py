"""Domain DTO for Camunda runtime event rows used in Stage 3.1."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class ProcessEventDTO(BaseModel):
    """Execution-level event extracted from Camunda runtime/history tables."""

    case_id: str
    activity_def_id: str
    activity_name: Optional[str] = None
    activity_type: Optional[str] = None
    proc_def_id: Optional[str] = None
    proc_def_key: Optional[str] = None
    proc_def_version: Optional[str] = None
    task_id: Optional[str] = None
    act_inst_id: Optional[str] = None
    execution_id: Optional[str] = None
    parent_execution_id: Optional[str] = None
    call_proc_inst_id: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    assignee: Optional[str] = None
    candidate_groups: Optional[list[str]] = None
    removal_time: Optional[datetime] = None
    extra: Dict[str, Any] = Field(default_factory=dict)
