"""Domain service that generates All Prefixes slices."""

# Відповідно до:
# - AGENT_GUIDE.MD -> розділ 2 (MVP1 data flow RawTrace -> PrefixSlice)
# - DATA_FLOWS_MVP1.MD -> розділ 2.2 (IPrefixPolicy.generate_slices)
# - LLD_MVP1.MD -> розділ 3.1 (k=1..n-1; y=e_{k+1}) та 3.2 (n<2 skip, no padding)

from __future__ import annotations

from typing import List

from src.domain.entities.prefix_slice import PrefixSlice
from src.domain.entities.raw_trace import RawTrace


class PrefixPolicy:
    """Stateless All-Prefixes slicing policy for a single trace."""

    def generate_slices(self, trace: RawTrace) -> List[PrefixSlice]:
        """Generate σ_[1:k] -> e_(k+1) slices for all valid k."""
        events = trace.events
        if len(events) < 2:
            return []

        slices: List[PrefixSlice] = []
        for k in range(1, len(events)):
            slices.append(
                PrefixSlice(
                    case_id=trace.case_id,
                    process_version=trace.process_version,
                    prefix_events=events[:k],
                    target_event=events[k],
                )
            )
        return slices
