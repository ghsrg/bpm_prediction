"""Topology extractor service for MVP2 knowledge injection."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from graphviz.backend import ExecutableNotFound
from pm4py.visualization.dfg import visualizer as dfg_visualizer

from src.domain.entities.process_structure import ProcessStructureDTO
from src.domain.entities.raw_trace import RawTrace
from src.domain.ports.knowledge_graph_port import IKnowledgeGraphPort


class TopologyExtractorService:
    """Extract topology from logs and persist it via the knowledge-graph port."""

    def __init__(self, knowledge_port: IKnowledgeGraphPort, process_name: str | None = None) -> None:
        self.knowledge_port = knowledge_port
        self._default_process_name = self._normalize_process_name(process_name)

    @property
    def available_versions(self) -> List[str]:
        """Return extracted process-version keys for default process scope."""
        return self.knowledge_port.list_versions(process_name=self._default_process_name)

    def fit(self, train_traces: List[RawTrace], process_name: str | None = None) -> None:
        """Alias for logs-based extraction to keep trainer/tooling API concise."""
        self.extract_from_logs(train_traces=train_traces, process_name=process_name)

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

    def extract_from_logs(self, train_traces: List[RawTrace], process_name: str | None = None) -> None:
        """Extract version-scoped transitions and persist DTOs via port."""
        process_key = self._normalize_process_name(process_name) or self._default_process_name or "default_process"
        self._default_process_name = process_key

        by_version: Dict[str, Set[Tuple[str, str]]] = {}
        freq_by_version: Dict[str, Dict[Tuple[str, str], int]] = {}

        for trace in train_traces:
            if not trace.events:
                continue

            raw_version = str(trace.process_version).strip()
            version_key = raw_version or process_key

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

        for version, edges in by_version.items():
            dto = ProcessStructureDTO(
                version=version,
                allowed_edges=sorted(list(edges)),
                edge_statistics={
                    edge: {"count": float(freq)}
                    for edge, freq in sorted(freq_by_version.get(version, {}).items())
                },
            )
            self.knowledge_port.save_process_structure(
                version=version,
                dto=dto,
                process_name=process_key,
            )

    def extract_from_bpmn(self, bpmn_data: Any) -> None:
        """Reserved entrypoint for Enterprise PoC BPMN/Camunda integration.

        Future visualization path for BPMN will use pm4py.visualization.bpmn.visualizer.
        """
        _ = bpmn_data
        raise NotImplementedError("Will be implemented in Enterprise PoC for Camunda integration")

    def get_process_structure(
        self,
        version: str,
        process_name: str | None = None,
    ) -> Optional[ProcessStructureDTO]:
        """Return extracted process structure by version."""
        return self.knowledge_port.get_process_structure(
            version=version,
            process_name=self._resolve_process_name(process_name),
        )

    def _build_dfg_payload(
        self,
        version: str,
        process_name: str | None = None,
        min_edge_frequency: int = 1,
    ) -> tuple[Dict[Tuple[str, str], int], Dict[str, int], Dict[str, int]]:
        """Build PM4Py DFG payload from graph returned by the knowledge port."""
        process_key = self._resolve_process_name(process_name)
        try:
            graph = self.knowledge_port.get_graph_for_visualization(
                process_name=process_key,
                version_key=version,
                min_edge_frequency=max(0, int(min_edge_frequency)),
            )
        except ValueError as exc:
            raise ValueError(
                f"No topology found for process '{process_key}' and process version '{version}'."
            ) from exc

        dfg: Dict[Tuple[str, str], int] = {}
        for src, dst, attrs in graph.edges(data=True):
            freq = int(attrs.get("weight", 1))
            dfg[(str(src), str(dst))] = max(freq, 1)

        start_activities: Dict[str, int] = {}
        end_activities: Dict[str, int] = {}
        for node in graph.nodes():
            in_degree = int(graph.in_degree(node))
            out_degree = int(graph.out_degree(node))
            node_name = str(node)
            if in_degree == 0 and out_degree > 0:
                out_weight_sum = sum(int(attrs.get("weight", 1)) for _, _, attrs in graph.out_edges(node, data=True))
                start_activities[node_name] = max(out_weight_sum, 1)
            if out_degree == 0 and in_degree > 0:
                in_weight_sum = sum(int(attrs.get("weight", 1)) for _, _, attrs in graph.in_edges(node, data=True))
                end_activities[node_name] = max(in_weight_sum, 1)

        return dfg, start_activities, end_activities

    def plot_topology(
        self,
        version: str,
        process_name: str | None = None,
        save_path: str | None = None,
        min_edge_freq: int = 1,
    ) -> None:
        """Render topology as PM4Py DFG using graph returned by the knowledge port."""
        if min_edge_freq < 1:
            raise ValueError("min_edge_freq must be >= 1.")

        dfg, start_activities, end_activities = self._build_dfg_payload(
            version=version,
            process_name=process_name,
            min_edge_frequency=min_edge_freq,
        )
        if not dfg:
            raise ValueError(
                f"No transitions found for process version '{version}' after min_edge_freq={min_edge_freq} filter."
            )

        parameters = {
            "format": "png",
            dfg_visualizer.Variants.FREQUENCY.value.Parameters.START_ACTIVITIES: start_activities,
            dfg_visualizer.Variants.FREQUENCY.value.Parameters.END_ACTIVITIES: end_activities,
        }

        try:
            gviz = dfg_visualizer.apply(
                dfg0=dfg,
                parameters=parameters,
                variant=dfg_visualizer.Variants.FREQUENCY,
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

    def _resolve_process_name(self, process_name: str | None) -> str:
        return self._normalize_process_name(process_name) or self._default_process_name or "default_process"

    @staticmethod
    def _normalize_process_name(process_name: str | None) -> str | None:
        if process_name is None:
            return None
        cleaned = str(process_name).strip()
        return cleaned or None
