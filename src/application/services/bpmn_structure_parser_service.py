"""BPMN structure parser service for Stage 3.2."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple
import xml.etree.ElementTree as ET

import networkx as nx

from src.domain.entities.process_structure import ProcessStructureDTO


logger = logging.getLogger(__name__)


@dataclass
class BpmnParseResult:
    """Result of BPMN parse attempt."""

    dto: Optional[ProcessStructureDTO]
    quarantine_record: Optional[Dict[str, Any]]


class BpmnStructureParserService:
    """Parse BPMN XML into ProcessStructureDTO with Camunda-aware metadata."""

    ERROR_MISSING_BPMN_PAYLOAD = "missing_bpmn_payload"
    ERROR_EMPTY_BPMN_PAYLOAD = "empty_bpmn_payload"
    ERROR_XML_PARSE = "xml_parse_error"
    ERROR_NO_PROCESS_ELEMENT = "no_process_element"

    QUARANTINE_ERROR_CODES = {
        ERROR_MISSING_BPMN_PAYLOAD,
        ERROR_EMPTY_BPMN_PAYLOAD,
        ERROR_XML_PARSE,
        ERROR_NO_PROCESS_ELEMENT,
        "missing_proc_def_id",
    }

    def __init__(
        self,
        *,
        subprocess_mode: str = "flattened-no-subprocess-node",
        parser_mode: str = "recover",
        inference_fallback_strategy: str = "use_aggregated_stats",
    ) -> None:
        self.subprocess_mode = str(subprocess_mode).strip() or "flattened-no-subprocess-node"
        self.parser_mode = str(parser_mode).strip().lower() or "recover"
        self.inference_fallback_strategy = (
            str(inference_fallback_strategy).strip() or "use_aggregated_stats"
        )

    def parse_definition(
        self,
        *,
        definition: Dict[str, Any],
        catalog: List[Dict[str, Any]],
        process_name: str,
        process_filters: Optional[List[str]] = None,
    ) -> BpmnParseResult:
        proc_def_id = str(definition.get("proc_def_id", "")).strip()
        proc_def_key = str(definition.get("proc_def_key", "")).strip()
        deployment_id = str(definition.get("deployment_id", "")).strip() or None
        version_raw = str(definition.get("version", "")).strip()
        version_key = self._to_version_key(version_raw)

        xml_payload = definition.get("bpmn_xml_content")
        if xml_payload is None:
            return BpmnParseResult(
                dto=None,
                quarantine_record=self._build_quarantine(
                    proc_def_id=proc_def_id,
                    proc_def_key=proc_def_key,
                    deployment_id=deployment_id,
                    version=version_key,
                    error_code=self.ERROR_MISSING_BPMN_PAYLOAD,
                    error_message="BPMN payload is missing for process definition.",
                    xml_snippet=None,
                    source_hint=f"process_name={process_name};proc_def_id={proc_def_id}",
                ),
            )

        sanitized = self._sanitize_xml(xml_payload)
        if not sanitized:
            return BpmnParseResult(
                dto=None,
                quarantine_record=self._build_quarantine(
                    proc_def_id=proc_def_id,
                    proc_def_key=proc_def_key,
                    deployment_id=deployment_id,
                    version=version_key,
                    error_code=self.ERROR_EMPTY_BPMN_PAYLOAD,
                    error_message="BPMN payload is empty after sanitation.",
                    xml_snippet=None,
                    source_hint=f"process_name={process_name};proc_def_id={proc_def_id}",
                ),
            )

        try:
            root = ET.fromstring(sanitized)
        except ET.ParseError as exc:
            return BpmnParseResult(
                dto=None,
                quarantine_record=self._build_quarantine(
                    proc_def_id=proc_def_id,
                    proc_def_key=proc_def_key,
                    deployment_id=deployment_id,
                    version=version_key,
                    error_code=self.ERROR_XML_PARSE,
                    error_message=str(exc),
                    xml_snippet=sanitized[:3000],
                    source_hint=f"process_name={process_name};proc_def_id={proc_def_id}",
                ),
            )

        process_elem = self._select_process_element(root=root, proc_def_key=proc_def_key)
        if process_elem is None:
            return BpmnParseResult(
                dto=None,
                quarantine_record=self._build_quarantine(
                    proc_def_id=proc_def_id,
                    proc_def_key=proc_def_key,
                    deployment_id=deployment_id,
                    version=version_key,
                    error_code=self.ERROR_NO_PROCESS_ELEMENT,
                    error_message="No <process> element found in BPMN payload.",
                    xml_snippet=sanitized[:3000],
                    source_hint=f"process_name={process_name};proc_def_id={proc_def_id}",
                ),
            )

        warnings: List[str] = []
        nodes: Dict[str, Dict[str, Any]] = {}
        edges: List[Dict[str, Any]] = []
        default_flow_ids: set[str] = set()

        self._collect_scope(
            container=process_elem,
            nodes=nodes,
            edges=edges,
            warnings=warnings,
            parent_scope=None,
            scope_level=0,
            default_flow_ids=default_flow_ids,
        )
        self._apply_boundary_semantics(nodes=nodes, edges=edges)
        if self.subprocess_mode == "flattened-no-subprocess-node":
            self._flatten_subprocess_nodes(nodes=nodes, edges=edges, warnings=warnings)
        self._remove_inlined_scope_terminal_events(nodes=nodes, edges=edges)
        ordered_nodes = self._ordered_nodes(nodes)
        ordered_edges = self._ordered_edges(edges)

        call_bindings = self._build_call_bindings(
            nodes=nodes,
            catalog=catalog,
            process_filters=process_filters,
            current_deployment_id=deployment_id,
        )
        for call_node_id, payload in call_bindings.items():
            node = nodes.get(call_node_id)
            if node is not None:
                node["call_binding_status"] = payload.get("status", "unresolved")
        ordered_nodes = self._ordered_nodes(nodes)

        allowed_edges = sorted({(str(edge["source"]), str(edge["target"])) for edge in ordered_edges})
        cycle_count = self._count_cycles(allowed_edges)
        cycles_detected = cycle_count > 0
        node_metadata = self._build_node_metadata(nodes=ordered_nodes)
        graph_topology = self._build_graph_topology(
            nodes=ordered_nodes,
            edges=ordered_edges,
            allowed_edges=allowed_edges,
            cycle_count=cycle_count,
        )

        dto = ProcessStructureDTO(
            version=version_key,
            proc_def_id=proc_def_id or None,
            proc_def_key=proc_def_key or None,
            deployment_id=deployment_id,
            allowed_edges=allowed_edges,
            edge_statistics=None,
            node_metadata=node_metadata or None,
            nodes=ordered_nodes,
            edges=ordered_edges,
            graph_topology=graph_topology,
            call_bindings=call_bindings or None,
            metadata={
                "process_name": process_name,
                "parser_mode": self.parser_mode,
                "warnings": sorted(set(warnings)),
                "warning_count": len(set(warnings)),
                "subprocess_mode": self.subprocess_mode,
                "parsed_at_utc": datetime.now(timezone.utc).isoformat(),
                "quarantine_error_codes": sorted(self.QUARANTINE_ERROR_CODES),
                "cycles_detected": cycles_detected,
            },
        )
        return BpmnParseResult(dto=dto, quarantine_record=None)

    @staticmethod
    def _sanitize_xml(payload: Any) -> str:
        if isinstance(payload, (bytes, bytearray)):
            try:
                text = bytes(payload).decode("utf-8")
            except UnicodeDecodeError:
                text = bytes(payload).decode("utf-8-sig", errors="ignore")
        else:
            text = str(payload)
        text = text.lstrip("\ufeff").strip()
        if not text:
            return ""
        start_idx = text.find("<")
        end_idx = text.rfind(">")
        if start_idx >= 0 and end_idx >= start_idx:
            text = text[start_idx : end_idx + 1]
        return text.strip()

    @staticmethod
    def _local(tag: str) -> str:
        if "}" in tag:
            return tag.split("}", 1)[1]
        return tag

    def _select_process_element(self, *, root: ET.Element, proc_def_key: str) -> Optional[ET.Element]:
        process_elems = [elem for elem in root.iter() if self._local(elem.tag) == "process"]
        if not process_elems:
            return None
        if proc_def_key:
            for elem in process_elems:
                if str(elem.attrib.get("id", "")).strip() == proc_def_key:
                    return elem
        for elem in process_elems:
            if str(elem.attrib.get("isExecutable", "true")).strip().lower() != "false":
                return elem
        return process_elems[0]

    def _collect_scope(
        self,
        *,
        container: ET.Element,
        nodes: Dict[str, Dict[str, Any]],
        edges: List[Dict[str, Any]],
        warnings: List[str],
        parent_scope: Optional[str],
        scope_level: int,
        default_flow_ids: set[str],
    ) -> None:
        for child in list(container):
            tag = self._local(child.tag)
            node_id = str(child.attrib.get("id", "")).strip()
            if "default" in child.attrib:
                default_flow_ids.add(str(child.attrib.get("default", "")).strip())
            if tag == "sequenceFlow":
                source = str(child.attrib.get("sourceRef", "")).strip()
                target = str(child.attrib.get("targetRef", "")).strip()
                if not source or not target:
                    warnings.append("sequence_flow_missing_source_or_target")
                    continue
                edge_id = node_id or f"{source}->{target}"
                edges.append(
                    {
                        "id": edge_id,
                        "source": source,
                        "target": target,
                        "edge_type": "sequence",
                        "condition_expr": self._extract_condition_expression(child),
                        "condition_complexity": self._condition_complexity(child),
                        "is_default": edge_id in default_flow_ids,
                        "scope_level": scope_level,
                    }
                )
                continue

            if not self._is_flow_node(tag):
                continue
            if not node_id:
                warnings.append(f"flow_node_without_id:{tag}")
                continue

            node = self._build_node_payload(
                element=child,
                node_id=node_id,
                bpmn_tag=tag,
                parent_scope=parent_scope,
                scope_level=scope_level,
            )
            nodes[node_id] = node

            attached_to = str(child.attrib.get("attachedToRef", "")).strip()
            if attached_to:
                edges.append(
                    {
                        "id": f"boundary_attachment::{attached_to}->{node_id}",
                        "source": attached_to,
                        "target": node_id,
                        "edge_type": "boundary_attachment",
                        "condition_expr": None,
                        "condition_complexity": 0.0,
                        "is_default": False,
                        "scope_level": scope_level,
                        "is_interrupting": self._is_interrupting_boundary(node),
                    }
                )

            if tag == "subProcess":
                self._collect_scope(
                    container=child,
                    nodes=nodes,
                    edges=edges,
                    warnings=warnings,
                    parent_scope=node_id,
                    scope_level=scope_level + 1,
                    default_flow_ids=default_flow_ids,
                )

    @staticmethod
    def _is_flow_node(tag: str) -> bool:
        return tag in {
            "startEvent",
            "endEvent",
            "intermediateCatchEvent",
            "intermediateThrowEvent",
            "boundaryEvent",
            "exclusiveGateway",
            "parallelGateway",
            "inclusiveGateway",
            "eventBasedGateway",
            "subProcess",
            "callActivity",
            "userTask",
            "serviceTask",
            "scriptTask",
            "businessRuleTask",
            "sendTask",
            "receiveTask",
            "manualTask",
            "task",
        }

    def _build_node_payload(
        self,
        *,
        element: ET.Element,
        node_id: str,
        bpmn_tag: str,
        parent_scope: Optional[str],
        scope_level: int,
    ) -> Dict[str, Any]:
        name = str(element.attrib.get("name", "")).strip() or node_id
        camunda_type = self._pick_camunda_attr(element.attrib, "type")
        node_type = f"{bpmn_tag}:{camunda_type}" if camunda_type else bpmn_tag
        is_event_subprocess = (
            bpmn_tag == "subProcess"
            and str(element.attrib.get("triggeredByEvent", "false")).strip().lower() == "true"
        )
        loop_meta = self._extract_loop_metadata(element)
        extensions = self._extract_extensions(element)
        cancel_activity = str(element.attrib.get("cancelActivity", "")).strip()
        if cancel_activity:
            extensions["cancelActivity"] = cancel_activity
        return {
            "id": node_id,
            "name": name,
            "type": node_type,
            "bpmn_tag": bpmn_tag,
            "activity_type": bpmn_tag,
            "camunda_type": camunda_type or "",
            "attached_to": str(element.attrib.get("attachedToRef", "")).strip() or None,
            "parent_scope_id": parent_scope,
            "scope_level": scope_level,
            "is_event_subprocess": is_event_subprocess,
            "is_multi_instance": loop_meta["is_multi_instance"],
            "is_sequential_mi": loop_meta["is_sequential_mi"],
            "loop_cardinality_expr": loop_meta["loop_cardinality_expr"],
            "extensions": extensions,
            "called_element": self._pick_camunda_attr(element.attrib, "calledElement") or element.attrib.get("calledElement"),
            "binding_type": self._pick_camunda_attr(element.attrib, "calledElementBinding"),
            "version_tag": self._pick_camunda_attr(element.attrib, "calledElementVersionTag"),
            "version_number": self._pick_camunda_attr(element.attrib, "calledElementVersion"),
        }

    @staticmethod
    def _pick_camunda_attr(attrs: Dict[str, Any], suffix: str) -> Optional[str]:
        suffix_l = suffix.lower()
        for key, value in attrs.items():
            k = str(key)
            if k.lower().endswith(suffix_l):
                text = str(value).strip()
                if text:
                    return text
        return None

    def _extract_loop_metadata(self, element: ET.Element) -> Dict[str, Any]:
        loop_cardinality_expr = None
        is_multi_instance = False
        is_sequential_mi = None
        for child in list(element):
            tag = self._local(child.tag)
            if tag == "multiInstanceLoopCharacteristics":
                is_multi_instance = True
                raw_seq = str(child.attrib.get("isSequential", "")).strip().lower()
                if raw_seq in {"true", "false"}:
                    is_sequential_mi = raw_seq == "true"
                for nested in child.iter():
                    if self._local(nested.tag) == "loopCardinality":
                        text = "".join(nested.itertext()).strip()
                        if text:
                            loop_cardinality_expr = text
                        break
            if tag == "standardLoopCharacteristics":
                is_multi_instance = True
        return {
            "is_multi_instance": is_multi_instance,
            "is_sequential_mi": is_sequential_mi,
            "loop_cardinality_expr": loop_cardinality_expr,
        }

    def _extract_extensions(self, element: ET.Element) -> Dict[str, Any]:
        extensions: Dict[str, Any] = {}
        ext_elem = None
        for child in list(element):
            if self._local(child.tag) == "extensionElements":
                ext_elem = child
                break
        if ext_elem is None:
            return extensions

        properties: Dict[str, str] = {}
        input_output: Dict[str, Dict[str, str]] = {"input": {}, "output": {}}
        listeners: List[Dict[str, str]] = []
        fields: List[Dict[str, str]] = []
        unknown: List[str] = []

        for nested in ext_elem.iter():
            tag = self._local(nested.tag)
            if tag == "property":
                key = str(nested.attrib.get("name", "")).strip()
                value = str(nested.attrib.get("value", "")).strip()
                if key:
                    properties[key] = value
            elif tag in {"inputParameter", "outputParameter"}:
                key = str(nested.attrib.get("name", "")).strip()
                value = "".join(nested.itertext()).strip()
                if key:
                    bucket = "input" if tag == "inputParameter" else "output"
                    input_output[bucket][key] = value
            elif tag in {"executionListener", "taskListener"}:
                listener: Dict[str, str] = {"type": tag}
                for attr_key in ("event", "class", "delegateExpression", "expression"):
                    value = self._pick_camunda_attr(nested.attrib, attr_key) or nested.attrib.get(attr_key)
                    if value is not None and str(value).strip():
                        listener[attr_key] = str(value).strip()
                listeners.append(listener)
            elif tag == "field":
                field_payload: Dict[str, str] = {}
                name = str(nested.attrib.get("name", "")).strip()
                if name:
                    field_payload["name"] = name
                string_value = self._pick_camunda_attr(nested.attrib, "stringValue") or nested.attrib.get("stringValue")
                expression = self._pick_camunda_attr(nested.attrib, "expression") or nested.attrib.get("expression")
                if string_value is not None and str(string_value).strip():
                    field_payload["stringValue"] = str(string_value).strip()
                if expression is not None and str(expression).strip():
                    field_payload["expression"] = str(expression).strip()
                if field_payload:
                    fields.append(field_payload)
            elif tag not in {
                "extensionElements",
                "properties",
                "inputOutput",
            } and tag not in {"camunda:properties", "camunda:inputOutput"}:
                if tag not in {"property", "inputParameter", "outputParameter", "executionListener", "taskListener", "field"}:
                    unknown.append(tag)

        if properties:
            extensions["properties"] = properties
        if input_output["input"] or input_output["output"]:
            extensions["input_output"] = input_output
        if listeners:
            extensions["listeners"] = listeners
        if fields:
            extensions["fields"] = fields
        if unknown:
            extensions["unknown_elements"] = sorted(set(unknown))
        return extensions

    def _extract_condition_expression(self, sequence_flow: ET.Element) -> Optional[str]:
        for child in sequence_flow.iter():
            if self._local(child.tag) == "conditionExpression":
                text = "".join(child.itertext()).strip()
                if text:
                    return text
        return None

    def _condition_complexity(self, sequence_flow: ET.Element) -> float:
        expr = self._extract_condition_expression(sequence_flow)
        if not expr:
            return 0.0
        score = 0.0
        if "${" in expr or "#{" in expr:
            score += 0.35
        score += min(0.35, len(expr) / 200.0)
        for token in (" and ", " or ", " not ", ">", "<", "==", "!=", "contains", "matches"):
            if token in expr:
                score += 0.05
        return float(min(1.0, score))

    @staticmethod
    def _is_interrupting_boundary(node: Dict[str, Any]) -> bool:
        extensions = node.get("extensions", {})
        if isinstance(extensions, dict):
            cancel_activity = extensions.get("cancelActivity")
            if isinstance(cancel_activity, str):
                return cancel_activity.strip().lower() != "false"
        return True

    def _apply_boundary_semantics(self, *, nodes: Dict[str, Dict[str, Any]], edges: List[Dict[str, Any]]) -> None:
        boundary_flags: Dict[str, bool] = {}
        for node_id, node in nodes.items():
            if node.get("bpmn_tag") != "boundaryEvent":
                continue
            raw = node.get("extensions", {}).get("cancelActivity")
            is_interrupting = True
            if isinstance(raw, str):
                is_interrupting = raw.strip().lower() != "false"
            boundary_flags[node_id] = is_interrupting
        for edge in edges:
            source = str(edge.get("source", "")).strip()
            if source not in boundary_flags:
                continue
            if edge.get("edge_type") != "sequence":
                continue
            edge["edge_type"] = "cancellation" if boundary_flags[source] else "fork"
            edge["is_interrupting"] = boundary_flags[source]

    def _flatten_subprocess_nodes(
        self,
        *,
        nodes: Dict[str, Dict[str, Any]],
        edges: List[Dict[str, Any]],
        warnings: List[str],
    ) -> None:
        subprocess_ids = [
            node_id
            for node_id, node in nodes.items()
            if str(node.get("bpmn_tag", "")).strip() == "subProcess"
        ]
        subprocess_ids.sort(key=lambda node_id: int(nodes[node_id].get("scope_level", 0)), reverse=True)

        for subprocess_id in subprocess_ids:
            if subprocess_id not in nodes:
                continue
            parent_scope = nodes.get(subprocess_id, {}).get("parent_scope_id")
            children = [
                node_id
                for node_id, node in nodes.items()
                if str(node.get("parent_scope_id", "")).strip() == subprocess_id
            ]
            incoming_external = [edge for edge in edges if edge["target"] == subprocess_id]
            outgoing_external = [edge for edge in edges if edge["source"] == subprocess_id]

            if not children:
                warnings.append(f"subprocess_without_children:{subprocess_id}")
                edges[:] = [edge for edge in edges if edge["source"] != subprocess_id and edge["target"] != subprocess_id]
                nodes.pop(subprocess_id, None)
                continue

            start_children = self._scope_start_nodes(children=children, edges=edges)
            end_children = self._scope_end_nodes(children=children, edges=edges)
            if not start_children:
                start_children = children
            if not end_children:
                end_children = children

            new_edges: List[Dict[str, Any]] = []
            for edge in incoming_external:
                for child in start_children:
                    new_edges.append(
                        {
                            "id": f"subprocess_rewire_in::{edge['source']}->{child}",
                            "source": edge["source"],
                            "target": child,
                            "edge_type": "subprocess_rewire",
                            "condition_expr": None,
                            "condition_complexity": 0.0,
                            "is_default": False,
                            "scope_level": int(nodes.get(child, {}).get("scope_level", 0)),
                        }
                    )
            for edge in outgoing_external:
                for child in end_children:
                    new_edges.append(
                        {
                            "id": f"subprocess_rewire_out::{child}->{edge['target']}",
                            "source": child,
                            "target": edge["target"],
                            "edge_type": "subprocess_rewire",
                            "condition_expr": None,
                            "condition_complexity": 0.0,
                            "is_default": False,
                            "scope_level": int(nodes.get(child, {}).get("scope_level", 0)),
                        }
                    )

            edges[:] = [
                edge
                for edge in edges
                if edge["source"] != subprocess_id and edge["target"] != subprocess_id
            ]
            edges.extend(new_edges)
            nodes.pop(subprocess_id, None)

            for child_id in children:
                child = nodes.get(child_id)
                if child is None:
                    continue
                child["parent_scope_id"] = parent_scope
                child["scope_level"] = max(0, int(child.get("scope_level", 1)) - 1)
                child["_inlined_scope_node"] = True

    @staticmethod
    def _scope_start_nodes(*, children: List[str], edges: List[Dict[str, Any]]) -> List[str]:
        children_set = set(children)
        incoming_by_child = {child: 0 for child in children}
        for edge in edges:
            source = str(edge.get("source", "")).strip()
            target = str(edge.get("target", "")).strip()
            if source in children_set and target in children_set:
                incoming_by_child[target] += 1
        return [child for child, count in incoming_by_child.items() if count == 0]

    @staticmethod
    def _scope_end_nodes(*, children: List[str], edges: List[Dict[str, Any]]) -> List[str]:
        children_set = set(children)
        outgoing_by_child = {child: 0 for child in children}
        for edge in edges:
            source = str(edge.get("source", "")).strip()
            target = str(edge.get("target", "")).strip()
            if source in children_set and target in children_set:
                outgoing_by_child[source] += 1
        return [child for child, count in outgoing_by_child.items() if count == 0]

    def _remove_inlined_scope_terminal_events(
        self,
        *,
        nodes: Dict[str, Dict[str, Any]],
        edges: List[Dict[str, Any]],
    ) -> None:
        removable = [
            node_id
            for node_id, node in nodes.items()
            if bool(node.get("_inlined_scope_node", False))
            and str(node.get("bpmn_tag", "")) in {"startEvent", "endEvent"}
        ]
        for node_id in removable:
            incoming = [edge for edge in edges if edge["target"] == node_id]
            outgoing = [edge for edge in edges if edge["source"] == node_id]
            bridge_edges: List[Dict[str, Any]] = []
            for in_edge in incoming:
                for out_edge in outgoing:
                    bridge_edges.append(
                        {
                            "id": f"inlined_event_bridge::{in_edge['source']}->{out_edge['target']}",
                            "source": in_edge["source"],
                            "target": out_edge["target"],
                            "edge_type": "inlined_event_bridge",
                            "condition_expr": None,
                            "condition_complexity": 0.0,
                            "is_default": False,
                            "scope_level": int(nodes.get(out_edge["target"], {}).get("scope_level", 0)),
                        }
                    )
            edges[:] = [edge for edge in edges if edge["source"] != node_id and edge["target"] != node_id]
            edges.extend(bridge_edges)
            nodes.pop(node_id, None)

    def _build_call_bindings(
        self,
        *,
        nodes: Dict[str, Dict[str, Any]],
        catalog: List[Dict[str, Any]],
        process_filters: Optional[List[str]],
        current_deployment_id: Optional[str],
    ) -> Dict[str, Dict[str, Any]]:
        by_key: Dict[str, List[Dict[str, Any]]] = {}
        for row in catalog:
            key = str(row.get("proc_def_key", "")).strip()
            if not key:
                continue
            by_key.setdefault(key, []).append(row)
        for key_rows in by_key.values():
            key_rows.sort(key=lambda item: self._safe_version_int(item.get("version")), reverse=True)

        filters_lower = {str(item).strip().lower() for item in (process_filters or []) if str(item).strip()}
        call_bindings: Dict[str, Dict[str, Any]] = {}
        for node_id, node in nodes.items():
            if str(node.get("bpmn_tag", "")) != "callActivity":
                continue
            called_element = str(node.get("called_element", "") or "").strip()
            binding_type = str(node.get("binding_type", "") or "latest").strip() or "latest"
            version_tag = str(node.get("version_tag", "") or "").strip() or None
            version_number = str(node.get("version_number", "") or "").strip() or None
            resolved_proc_def_id: Optional[str] = None
            reason: Optional[str] = None
            status = "unresolved"

            candidates = by_key.get(called_element, [])
            if not candidates:
                if filters_lower and called_element and called_element.lower() not in filters_lower:
                    reason = "child_process_out_of_current_filter"
                else:
                    reason = "child_not_found_in_catalog"
            else:
                resolved = self._resolve_binding_candidate(
                    candidates=candidates,
                    binding_type=binding_type,
                    version_tag=version_tag,
                    version_number=version_number,
                    deployment_id=current_deployment_id,
                )
                if resolved is None:
                    reason = "child_not_found_in_catalog"
                else:
                    resolved_proc_def_id = str(resolved.get("proc_def_id", "")).strip() or None
                    if resolved_proc_def_id:
                        status = "resolved"
                        reason = None
                    else:
                        reason = "child_not_found_in_catalog"

            call_bindings[node_id] = {
                "called_element": called_element or None,
                "binding_type": binding_type,
                "version_tag": version_tag,
                "version_number": version_number,
                "resolved_child_proc_def_id": resolved_proc_def_id,
                "status": status,
                "reason": reason,
                "requires_separate_inference": True,
                "inference_trigger": "child_process_root",
                "child_process_key": called_element or None,
                "child_version_tag": version_tag,
                "inference_fallback_strategy": self.inference_fallback_strategy,
            }
        return call_bindings

    @staticmethod
    def _resolve_binding_candidate(
        *,
        candidates: List[Dict[str, Any]],
        binding_type: str,
        version_tag: Optional[str],
        version_number: Optional[str],
        deployment_id: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        mode = binding_type.strip().lower()
        if mode == "latest":
            return candidates[0] if candidates else None
        if mode == "versiontag" and version_tag:
            for row in candidates:
                if str(row.get("version_tag", "")).strip() == version_tag:
                    return row
            return None
        if mode == "version" and version_number:
            for row in candidates:
                if str(row.get("version", "")).strip() == str(version_number):
                    return row
            return None
        if mode == "deployment" and deployment_id:
            for row in candidates:
                if str(row.get("deployment_id", "")).strip() == deployment_id:
                    return row
            return None
        return candidates[0] if candidates else None

    @staticmethod
    def _safe_version_int(value: Any) -> int:
        if value is None:
            return -1
        text = str(value).strip().lower()
        if text.startswith("v"):
            text = text[1:]
        try:
            return int(text)
        except ValueError:
            return -1

    @staticmethod
    def _ordered_nodes(nodes: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        ordered = [dict(node) for node in nodes.values()]
        ordered.sort(
            key=lambda item: (
                int(item.get("scope_level", 0)),
                str(item.get("parent_scope_id", "") or ""),
                str(item.get("id", "")),
            )
        )
        return ordered

    @staticmethod
    def _ordered_edges(edges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        dedup: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
        for edge in edges:
            source = str(edge.get("source", "")).strip()
            target = str(edge.get("target", "")).strip()
            if not source or not target:
                continue
            edge_type = str(edge.get("edge_type", "sequence")).strip() or "sequence"
            edge_id = str(edge.get("id", "")).strip() or f"{edge_type}::{source}->{target}"
            item = dict(edge)
            item["source"] = source
            item["target"] = target
            item["edge_type"] = edge_type
            item["id"] = edge_id
            dedup[(source, target, edge_type, edge_id)] = item
        ordered = list(dedup.values())
        ordered.sort(
            key=lambda item: (
                str(item.get("source", "")),
                str(item.get("target", "")),
                str(item.get("edge_type", "")),
                str(item.get("id", "")),
            )
        )
        return ordered

    @staticmethod
    def _build_node_metadata(nodes: List[Dict[str, Any]]) -> Dict[str, Dict[str, str]]:
        metadata: Dict[str, Dict[str, str]] = {}
        for node in nodes:
            node_id = str(node.get("id", "")).strip()
            if not node_id:
                continue
            metadata[str(node_id)] = {
                "activity_name": str(node.get("name", "")).strip() or str(node_id),
                "activity_type": str(node.get("activity_type", "")).strip() or str(node.get("bpmn_tag", "")),
            }
        return metadata

    @staticmethod
    def _build_graph_topology(
        *,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        allowed_edges: List[Tuple[str, str]],
        cycle_count: int,
    ) -> Dict[str, Any]:
        gateway_node_ids = sorted(
            str(node.get("id", "")).strip()
            for node in nodes
            if "gateway" in str(node.get("bpmn_tag", "")).lower()
        )
        event_subprocess_node_ids = sorted(
            str(node.get("id", "")).strip()
            for node in nodes
            if bool(node.get("is_event_subprocess", False))
        )
        return {
            "allowed_edges": [[src, dst] for src, dst in allowed_edges],
            "cycles_detected": cycle_count > 0,
            "cycle_count": cycle_count,
            "node_count": len(nodes),
            "edge_count": len(edges),
            "gateway_node_ids": gateway_node_ids,
            "event_subprocess_node_ids": event_subprocess_node_ids,
        }

    @staticmethod
    def _count_cycles(allowed_edges: Iterable[Tuple[str, str]]) -> int:
        graph = nx.DiGraph()
        graph.add_edges_from(list(allowed_edges))
        if graph.number_of_edges() == 0:
            return 0
        try:
            cycles = list(nx.simple_cycles(graph))
        except Exception:
            return 0
        return len(cycles)

    @staticmethod
    def _to_version_key(raw_version: str) -> str:
        text = str(raw_version).strip()
        if not text:
            return "v0"
        if text.lower().startswith("v"):
            suffix = text[1:]
            if suffix.isdigit():
                return f"v{int(suffix)}"
            return text
        if text.isdigit():
            return f"v{int(text)}"
        return text

    @staticmethod
    def _build_quarantine(
        *,
        proc_def_id: str,
        proc_def_key: str,
        deployment_id: Optional[str],
        version: str,
        error_code: str,
        error_message: str,
        xml_snippet: Optional[str],
        source_hint: str,
    ) -> Dict[str, Any]:
        normalized_error_code = str(error_code).strip().lower()
        return {
            "proc_def_id": proc_def_id,
            "proc_def_key": proc_def_key,
            "deployment_id": deployment_id,
            "version": version,
            "severity": "error",
            "error_code": normalized_error_code,
            "error_message": str(error_message),
            "source_hint": str(source_hint).strip(),
            "xml_snippet": xml_snippet,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
