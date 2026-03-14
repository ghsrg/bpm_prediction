"""Diagnostics DTO for Stage 3.1 runtime fetch and assembly quality."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class RuntimeFetchDiagnosticsDTO(BaseModel):
    """Quality and resilience indicators emitted by Camunda runtime adapter."""

    rows_raw: int = 0
    rows_after_cleanup_filter: int = 0
    rows_after_dedup: int = 0
    cleaned_instances_percent: float = 0.0
    history_coverage_percent: Optional[float] = None
    fallback_triggered: bool = False
    fallback_reason: Optional[str] = None
    removal_time_supported_tables: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    meta: Dict[str, Any] = Field(default_factory=dict)
