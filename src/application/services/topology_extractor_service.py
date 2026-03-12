"""Topology extractor service for MVP2 knowledge injection."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from graphviz.backend import ExecutableNotFound
from pm4py.visualization.dfg import visualizer as dfg_visualizer

from src.application.ports.knowledge_graph_port import IKnowledgeGraphPort
from src.domain.entities.process_structure import ProcessStructureDTO
from src.domain.entities.raw_trace import RawTrace


class TopologyExtractorService(IKnowledgeGraphPort):
    """In-memory multi-source topology extractor (logs now, BPMN in future)."""

    def __init__(self) -> None:
        self._topology_registry: Dict[str, ProcessStructureDTO] = {}
        self._start_activities_registry: Dict[str, Dict[str, int]] = {}
        self._end_activities_registry: Dict[str, Dict[str, int]] = {}

    @property
    def available_versions(self) -> List[str]:
        """Return currently extracted process-version keys."""
        return list(self._topology_registry.keys())

    def fit(self, train_traces: List[RawTrace]) -> None:
        """Alias for logs-based extraction to keep trainer/tooling API concise."""
        self.extract_from_logs(train_traces)

    def _event_label(self, event: Any) -> str:
        """Resolve stable activity label for topology edges (prefer semantic name)."""
        extra = getattr(event, "extra", {}) or {}
        if isinstance(extra, dict):
            for key in ("concept:name", "activity"):
                value = extra.get(key)
                if value is not None and str(value).strip():
                    return str(value).strip()
        activity_id = getattr(event, "activity_id", "")
        return str(activity_id).strip()

    def extract_from_logs(self, train_traces: List[RawTrace]) -> None:
        """Extract version-scoped unique transitions from train traces."""
        by_version: Dict[str, Set[Tuple[str, str]]] = {}
        freq_by_version: Dict[str, Dict[Tuple[str, str], int]] = {}
        start_by_version: Dict[str, Dict[str, int]] = {}
        end_by_version: Dict[str, Dict[str, int]] = {}

        for trace in train_traces:
            if not trace.events:
                continue
            version_key = trace.process_version if trace.process_version else "1"
            version_key = str(version_key).strip() or "1"

            first_activity = self._event_label(trace.events[0])
            last_activity = self._event_label(trace.events[-1])
            version_starts = start_by_version.setdefault(version_key, {})
            version_ends = end_by_version.setdefault(version_key, {})
            version_starts[first_activity] = version_starts.get(first_activity, 0) + 1
            version_ends[last_activity] = version_ends.get(last_activity, 0) + 1

            if len(trace.events) < 2:
                continue
            edges = by_version.setdefault(version_key, set())
            edge_freq = freq_by_version.setdefault(version_key, {})
            for idx in range(len(trace.events) - 1):
                src = self._event_label(trace.events[idx])
                dst = self._event_label(trace.events[idx + 1])
                edge = (src, dst)
                edges.add(edge)
                edge_freq[edge] = edge_freq.get(edge, 0) + 1

        self._topology_registry = {
            version: ProcessStructureDTO(
                version=version,
                allowed_edges=sorted(list(edges)),
                edge_statistics={
                    edge: {"count": float(freq)}
                    for edge, freq in sorted(freq_by_version.get(version, {}).items())
                },
            )
            for version, edges in by_version.items()
        }
        self._start_activities_registry = start_by_version
        self._end_activities_registry = end_by_version

    def extract_from_bpmn(self, bpmn_data: Any) -> None:
        """Reserved entrypoint for Enterprise PoC BPMN/Camunda integration.

        Future visualization path for BPMN will use pm4py.visualization.bpmn.visualizer.
        """
        _ = bpmn_data
        raise NotImplementedError("Will be implemented in Enterprise PoC for Camunda integration")

    def get_process_structure(self, version: str) -> Optional[ProcessStructureDTO]:
        """Return extracted process structure by version."""
        return self._topology_registry.get(version)

    def _build_dfg_payload(
        self, version: str
    ) -> tuple[Dict[Tuple[str, str], int], Dict[str, int], Dict[str, int]]:
        """Build PM4Py DFG payload: edge frequencies + start/end activity maps."""
        dto = self.get_process_structure(version)
        if dto is None:
            raise ValueError(f"No topology found for process version '{version}'.")

        edge_stats = dto.edge_statistics or {}
        dfg: Dict[Tuple[str, str], int] = {}
        incoming: Dict[str, int] = {}
        outgoing: Dict[str, int] = {}

        for src, dst in dto.allowed_edges:
            stats = edge_stats.get((src, dst), {})
            frequency = int(stats.get("count", 1))
            dfg[(src, dst)] = max(frequency, 1)
            outgoing[src] = outgoing.get(src, 0) + 1
            incoming[dst] = incoming.get(dst, 0) + 1
            incoming.setdefault(src, incoming.get(src, 0))
            outgoing.setdefault(dst, outgoing.get(dst, 0))

        start_activities = dict(self._start_activities_registry.get(version, {}))
        end_activities = dict(self._end_activities_registry.get(version, {}))

        if not start_activities:
            start_activities = {node: 1 for node, in_count in incoming.items() if in_count == 0}
        if not end_activities:
            end_activities = {node: 1 for node, out_count in outgoing.items() if out_count == 0}

        return dfg, start_activities, end_activities

    def plot_topology(self, version: str, save_path: str | None = None, min_edge_freq: int = 1) -> None:
        """Render topology as PM4Py DFG with directed edges and start/end coloring."""
        if min_edge_freq < 1:
            raise ValueError("min_edge_freq must be >= 1.")

        dfg, start_activities, end_activities = self._build_dfg_payload(version)
        filtered_dfg = {edge: freq for edge, freq in dfg.items() if freq >= min_edge_freq}
        if not filtered_dfg:
            raise ValueError(
                f"No transitions found for process version '{version}' after min_edge_freq={min_edge_freq} filter."
            )

        nodes_in_filtered = {src for src, _ in filtered_dfg}.union({dst for _, dst in filtered_dfg})
        filtered_start = {act: cnt for act, cnt in start_activities.items() if act in nodes_in_filtered}
        filtered_end = {act: cnt for act, cnt in end_activities.items() if act in nodes_in_filtered}

        if not filtered_start:
            incoming: Dict[str, int] = {}
            for src, dst in filtered_dfg:
                incoming[dst] = incoming.get(dst, 0) + 1
                incoming.setdefault(src, incoming.get(src, 0))
            filtered_start = {node: 1 for node, in_count in incoming.items() if in_count == 0}
        if not filtered_end:
            outgoing: Dict[str, int] = {}
            for src, dst in filtered_dfg:
                outgoing[src] = outgoing.get(src, 0) + 1
                outgoing.setdefault(dst, outgoing.get(dst, 0))
            filtered_end = {node: 1 for node, out_count in outgoing.items() if out_count == 0}

        parameters = {
            "format": "png",
            dfg_visualizer.Variants.FREQUENCY.value.Parameters.START_ACTIVITIES: filtered_start,
            dfg_visualizer.Variants.FREQUENCY.value.Parameters.END_ACTIVITIES: filtered_end,
        }

        try:
            gviz = dfg_visualizer.apply(
                dfg0=filtered_dfg, parameters=parameters, variant=dfg_visualizer.Variants.FREQUENCY
            )
            if save_path:
                target = Path(save_path)
                target.parent.mkdir(parents=True, exist_ok=True)
                dfg_visualizer.save(gviz, str(target))
                return
            dfg_visualizer.view(gviz)
        except ExecutableNotFound as exc:
            raise RuntimeError(
                "Graphviz executable 'dot' was not found. Install Graphviz and add it to PATH "
                "to render PM4Py DFG visuals."
            ) from exc
