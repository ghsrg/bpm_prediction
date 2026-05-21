"""Application port for selective structural prediction traces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class StructuralTraceEvent:
    """Plain JSON-serializable trace event passed to infrastructure recorders."""

    name: str
    inputs: dict[str, Any]
    outputs: dict[str, Any]
    attributes: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "attributes": self.attributes,
        }


class ITraceRecorder(Protocol):
    """Port for recording selective prediction traces."""

    def record(self, event: StructuralTraceEvent) -> None:
        """Record a selected structural trace event."""
        ...
