"""Application port contract for XES ingestion."""

# Відповідно до:
# - AGENT_GUIDE.MD -> розділ 2 (Dependency inversion) і розділ 4 (RawTrace DTO)
# - DATA_FLOWS_MVP1.MD -> розділ 2.1 (IXESAdapter контракт)
# - ADAPTER_XES.MD -> розділ 7 (Output: Iterator[RawTrace])

from __future__ import annotations

from typing import Iterator, Protocol

from src.domain.entities.raw_trace import RawTrace


class IXESAdapter(Protocol):
    """Port for streaming conversion from event logs into canonical traces."""

    def read(self, file_path: str, mapping_config: dict) -> Iterator[RawTrace]:
        """Stream normalized traces from an input event log."""
        ...
