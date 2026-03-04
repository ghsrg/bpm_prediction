"""Application port contract for trace prefix slicing."""

# Відповідно до:
# - AGENT_GUIDE.MD -> розділ 2 (Application залежить від портів)
# - DATA_FLOWS_MVP1.MD -> розділ 2.2 (IPrefixPolicy контракт)
# - LLD_MVP1.MD -> розділ 3 (All Prefixes)

from __future__ import annotations

from typing import List, Protocol

from src.domain.entities.prefix_slice import PrefixSlice
from src.domain.entities.raw_trace import RawTrace


class IPrefixPolicy(Protocol):
    """Port for converting a normalized trace into training prefixes."""

    def generate_slices(self, trace: RawTrace) -> List[PrefixSlice]:
        """Generate all valid prefix slices for one trace."""
        ...
