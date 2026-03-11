"""Topology extractor service for MVP2 knowledge injection."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import networkx as nx

from src.application.ports.knowledge_graph_port import IKnowledgeGraphPort
from src.domain.entities.process_structure import ProcessStructureDTO
from src.domain.entities.raw_trace import RawTrace


class TopologyExtractorService(IKnowledgeGraphPort):
    """In-memory multi-source topology extractor (logs now, BPMN in future)."""

    def __init__(self) -> None:
        self._topology_registry: Dict[str, ProcessStructureDTO] = {}

    @property
    def available_versions(self) -> List[str]:
        """Return currently extracted process-version keys."""
        return list(self._topology_registry.keys())

    def fit(self, train_traces: List[RawTrace]) -> None:
        """Alias for logs-based extraction to keep trainer/tooling API concise."""
        self.extract_from_logs(train_traces)

    def extract_from_logs(self, train_traces: List[RawTrace]) -> None:
        """Extract version-scoped unique transitions from train traces."""
        by_version: Dict[str, Set[Tuple[str, str]]] = {}

        for trace in train_traces:
            if len(trace.events) < 2:
                continue
            version_key = trace.process_version if trace.process_version else "1"
            version_key = str(version_key).strip() or "1"
            edges = by_version.setdefault(version_key, set())
            for idx in range(len(trace.events) - 1):
                src = trace.events[idx].activity_id
                dst = trace.events[idx + 1].activity_id
                edges.add((src, dst))

        self._topology_registry = {
            version: ProcessStructureDTO(version=version, allowed_edges=sorted(list(edges)))
            for version, edges in by_version.items()
        }

    def extract_from_bpmn(self, bpmn_data: Any) -> None:
        """Reserved entrypoint for Enterprise PoC BPMN/Camunda integration."""
        _ = bpmn_data
        raise NotImplementedError("Will be implemented in Enterprise PoC for Camunda integration")

    def get_process_structure(self, version: str) -> Optional[ProcessStructureDTO]:
        """Return extracted process structure by version."""
        return self._topology_registry.get(version)

    def export_to_networkx(self, version: str) -> nx.DiGraph:
        """Export stored version topology to a directed NetworkX graph."""
        dto = self.get_process_structure(version)
        if dto is None:
            raise ValueError(f"No topology found for process version '{version}'.")

        graph = nx.DiGraph()
        graph.add_edges_from(dto.allowed_edges)
        return graph

    def plot_topology(self, version: str, save_path: str | None = None) -> None:
        """Plot topology graph; save to file when save_path is provided."""
        graph = self.export_to_networkx(version)
        plt.figure(figsize=(10, 7))

        if graph.number_of_nodes() > 0:
            pos = nx.spring_layout(graph, seed=42)
            nx.draw_networkx_nodes(
                graph,
                pos,
                node_size=1800,
                node_color="#cfe8ff",
                edgecolors="#2b4c7e",
                linewidths=1.2,
            )
            nx.draw_networkx_edges(
                graph,
                pos,
                edge_color="#4b6584",
                width=1.8,
                arrows=True,
                arrowstyle="-|>",
                arrowsize=18,
                connectionstyle="arc3,rad=0.08",
            )
            nx.draw_networkx_labels(graph, pos, font_size=10, font_weight="bold")

        plt.title(f"Extracted Topology for process_version={version}", fontsize=12, pad=12)
        plt.axis("off")
        plt.tight_layout()

        if save_path:
            target = Path(save_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(target, dpi=180, bbox_inches="tight")
            plt.close()
            return

        plt.show()
        plt.close()
