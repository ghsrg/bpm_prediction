"""Topology extractor service for MVP2 knowledge injection."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from graphviz.backend import ExecutableNotFound
from graphviz import Digraph
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
        node_meta_by_version: Dict[str, Dict[str, Dict[str, str]]] = {}

        for trace in train_traces:
            if not trace.events:
                continue

            raw_version = str(trace.process_version).strip()
            version_key = raw_version or process_key
            node_meta = node_meta_by_version.setdefault(version_key, {})

            for event in trace.events:
                node_id = self._event_label(event)
                if not node_id:
                    continue
                meta = self._extract_event_node_metadata(event=event, node_id=node_id)
                existing = node_meta.get(node_id, {})
                node_meta[node_id] = {
                    "activity_name": existing.get("activity_name") or meta.get("activity_name", ""),
                    "activity_type": existing.get("activity_type") or meta.get("activity_type", ""),
                }

            if len(trace.events) < 2:
                continue
            edges = by_version.setdefault(version_key, set())
            edge_freq = freq_by_version.setdefault(version_key, {})
            for idx in range(len(trace.events) - 1):
                src_event = trace.events[idx]
                dst_event = trace.events[idx + 1]
                src = self._event_label(src_event)
                dst = self._event_label(dst_event)
                edge = (src, dst)
                src_meta = self._extract_event_node_metadata(event=src_event, node_id=src)
                dst_meta = self._extract_event_node_metadata(event=dst_event, node_id=dst)
                src_type = src_meta.get("activity_type", "")
                dst_type = dst_meta.get("activity_type", "")
                if self._is_end_event(src_type):
                    continue
                if self._is_start_event(dst_type):
                    continue
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
                node_metadata=node_meta_by_version.get(version, {}),
            )
            self.knowledge_port.save_process_structure(
                version=version,
                dto=dto,
                process_name=process_key,
            )

    @staticmethod
    def _extract_event_node_metadata(event: Any, node_id: str) -> Dict[str, str]:
        extra = getattr(event, "extra", {}) or {}
        activity_name = ""
        activity_type = ""
        if isinstance(extra, dict):
            raw_name = extra.get("activity_name")
            if raw_name is None:
                raw_name = extra.get("concept:name")
            if raw_name is not None:
                activity_name = str(raw_name).strip()
            raw_type = extra.get("activity_type")
            if raw_type is not None:
                activity_type = str(raw_type).strip()

        if not activity_name:
            activity_name = str(node_id)
        return {
            "activity_name": activity_name,
            "activity_type": activity_type,
        }

    @staticmethod
    def _is_start_event(activity_type: str) -> bool:
        raw = str(activity_type).strip().lower()
        if not raw:
            return False
        normalized = raw.replace(" ", "")
        return "startevent" in normalized

    @staticmethod
    def _is_end_event(activity_type: str) -> bool:
        raw = str(activity_type).strip().lower()
        if not raw:
            return False
        normalized = raw.replace(" ", "")
        return "endevent" in normalized

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
        renderer: str = "pm4py",
        label_mode: str = "id",
        typed_colors: bool = False,
    ) -> None:
        """Render topology as PM4Py DFG using graph returned by the knowledge port."""
        if min_edge_freq < 1:
            raise ValueError("min_edge_freq must be >= 1.")
        renderer_norm = str(renderer).strip().lower() or "pm4py"
        if renderer_norm not in {"pm4py", "graphviz"}:
            raise ValueError("renderer must be either 'pm4py' or 'graphviz'.")

        if renderer_norm == "graphviz":
            self._plot_topology_graphviz(
                version=version,
                process_name=process_name,
                save_path=save_path,
                min_edge_freq=min_edge_freq,
                label_mode=label_mode,
                typed_colors=typed_colors,
            )
            return

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

    def _plot_topology_graphviz(
        self,
        *,
        version: str,
        process_name: str | None,
        save_path: str | None,
        min_edge_freq: int,
        label_mode: str,
        typed_colors: bool,
    ) -> None:
        process_key = self._resolve_process_name(process_name)
        graph = self.knowledge_port.get_graph_for_visualization(
            process_name=process_key,
            version_key=version,
            min_edge_frequency=max(0, int(min_edge_freq)),
        )
        if graph.number_of_edges() == 0:
            raise ValueError(
                f"No transitions found for process version '{version}' after min_edge_freq={min_edge_freq} filter."
            )

        dot = Digraph(comment=f"Process topology {process_key}:{version}")
        dot.attr(rankdir="LR", fontname="Helvetica")

        for node_id, attrs in graph.nodes(data=True):
            category = self._classify_node_category(graph=graph, node_id=str(node_id), attrs=attrs)
            color, shape, peripheries = self._node_style(category=category, typed_colors=typed_colors)
            label = self._format_node_label(node_id=str(node_id), attrs=attrs, label_mode=label_mode)
            node_kwargs = {
                "label": label,
                "shape": shape,
                "style": "filled",
                "fillcolor": color,
                "color": "#263238",
                "fontname": "Helvetica",
                "fontsize": "10",
            }
            if peripheries > 1:
                node_kwargs["peripheries"] = str(peripheries)
            dot.node(str(node_id), **node_kwargs)

        for src, dst, attrs in graph.edges(data=True):
            weight = int(attrs.get("weight", 1))
            penwidth = 1.0 + min(5.0, math.log2(max(weight, 1) + 1.0))
            dot.edge(
                str(src),
                str(dst),
                label=str(weight),
                color="#546E7A",
                fontname="Helvetica",
                fontsize="9",
                penwidth=f"{penwidth:.2f}",
            )

        try:
            if save_path:
                target = Path(save_path)
                target.parent.mkdir(parents=True, exist_ok=True)
                output_format = target.suffix.lstrip(".").lower() if target.suffix else "png"
                dot.format = output_format or "png"
                dot.render(filename=target.stem, directory=str(target.parent), cleanup=True)
                return
            dot.view(cleanup=True)
        except ExecutableNotFound as exc:
            raise RuntimeError(
                "Graphviz executable 'dot' was not found. Install Graphviz and add it to PATH "
                "to render topology visuals."
            ) from exc

    @staticmethod
    def _format_node_label(node_id: str, attrs: Dict[str, Any], label_mode: str) -> str:
        mode = str(label_mode).strip().lower() or "id"
        activity_name = str(attrs.get("activity_name", "")).strip()
        activity_type = str(attrs.get("activity_type", "")).strip()
        if mode == "name":
            return activity_name or node_id
        if mode == "id+name":
            if activity_name and activity_name != node_id:
                return f"{node_id}\\n{activity_name}"
            return node_id
        if mode == "id+name+type":
            parts = [node_id]
            if activity_name and activity_name != node_id:
                parts.append(activity_name)
            if activity_type:
                parts.append(f"[{activity_type}]")
            return "\\n".join(parts)
        return node_id

    @staticmethod
    def _classify_node_category(graph: Any, node_id: str, attrs: Dict[str, Any]) -> str:
        node_type = str(attrs.get("activity_type", "")).strip().lower()
        if node_type:
            if "start" in node_type:
                return "start"
            if "end" in node_type:
                return "end"
            if "gateway" in node_type:
                return "gateway"
            if "usertask" in node_type or ("user" in node_type and "task" in node_type):
                return "user_task"
            if "servicetask" in node_type or ("service" in node_type and "task" in node_type):
                return "service_task"
            if any(token in node_type for token in ("timer", "escalation", "message", "signal", "event", "boundary")):
                return "event_other"

        if int(graph.in_degree(node_id)) == 0:
            return "start"
        if int(graph.out_degree(node_id)) == 0:
            return "end"
        return "other"

    @staticmethod
    def _node_style(category: str, typed_colors: bool) -> Tuple[str, str, int]:
        if not typed_colors:
            if category == "start":
                return "#C8E6C9", "circle", 1
            if category == "end":
                return "#FFE0B2", "doublecircle", 2
            return "#ECEFF1", "box", 1

        style_map = {
            "start": ("#A5D6A7", "circle", 1),
            "end": ("#FFCC80", "doublecircle", 2),
            "gateway": ("#90CAF9", "diamond", 1),
            "user_task": ("#B3E5FC", "box", 1),
            "service_task": ("#80CBC4", "box", 1),
            "event_other": ("#FFF59D", "ellipse", 1),
            "other": ("#CFD8DC", "box", 1),
        }
        return style_map.get(category, style_map["other"])

    def _resolve_process_name(self, process_name: str | None) -> str:
        return self._normalize_process_name(process_name) or self._default_process_name or "default_process"

    @staticmethod
    def _normalize_process_name(process_name: str | None) -> str | None:
        if process_name is None:
            return None
        cleaned = str(process_name).strip()
        return cleaned or None
