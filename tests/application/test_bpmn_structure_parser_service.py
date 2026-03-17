from __future__ import annotations

from pathlib import Path

import pytest

from src.application.services.bpmn_structure_parser_service import BpmnStructureParserService


def _parent_bpmn() -> str:
    return """<?xml version="1.0" encoding="UTF-8"?>
<definitions xmlns="http://www.omg.org/spec/BPMN/20100524/MODEL"
             xmlns:camunda="http://camunda.org/schema/1.0/bpmn"
             targetNamespace="Examples">
  <process id="parent_proc" isExecutable="true">
    <startEvent id="start" />
    <subProcess id="sub1" name="Sub Flow">
      <startEvent id="sub_start" />
      <userTask id="sub_task" name="Sub Task" />
      <endEvent id="sub_end" />
      <sequenceFlow id="sf1" sourceRef="sub_start" targetRef="sub_task" />
      <sequenceFlow id="sf2" sourceRef="sub_task" targetRef="sub_end" />
    </subProcess>
    <callActivity id="call1"
                  name="Call Child"
                  calledElement="child_proc"
                  camunda:calledElementBinding="latest" />
    <endEvent id="end" />
    <sequenceFlow id="f1" sourceRef="start" targetRef="sub1" />
    <sequenceFlow id="f2" sourceRef="sub1" targetRef="call1" />
    <sequenceFlow id="f3" sourceRef="call1" targetRef="end" />
  </process>
</definitions>
"""


def _fixture_bpmn(name: str) -> str:
    root = Path(__file__).resolve().parents[1] / "fixtures" / "bpmn_dirty"
    return (root / name).read_text(encoding="utf-8")


def test_bpmn_parser_flattens_embedded_subprocess_and_builds_call_bindings():
    parser = BpmnStructureParserService()
    catalog = [
        {
            "proc_def_id": "parent_def",
            "proc_def_key": "parent_proc",
            "version": "22",
            "deployment_id": "dep_parent",
        },
        {
            "proc_def_id": "child_def",
            "proc_def_key": "child_proc",
            "version": "5",
            "deployment_id": "dep_child",
        },
    ]
    result = parser.parse_definition(
        definition={
            "proc_def_id": "parent_def",
            "proc_def_key": "parent_proc",
            "version": "22",
            "deployment_id": "dep_parent",
            "bpmn_xml_content": _parent_bpmn(),
        },
        catalog=catalog,
        process_name="camunda_dataset",
        process_filters=["parent_proc", "child_proc"],
    )
    assert result.quarantine_record is None
    assert result.dto is not None
    dto = result.dto

    assert dto.version == "v22"
    node_ids = {node["id"] for node in (dto.nodes or [])}
    assert "sub1" not in node_ids
    assert "sub_task" in node_ids
    assert ("start", "sub_task") in set(dto.allowed_edges)
    assert ("sub_task", "call1") in set(dto.allowed_edges)

    assert dto.call_bindings is not None
    call_binding = dto.call_bindings["call1"]
    assert call_binding["status"] == "resolved"
    assert call_binding["resolved_child_proc_def_id"] == "child_def"
    assert call_binding["requires_separate_inference"] is True
    assert call_binding["inference_fallback_strategy"] == "use_aggregated_stats"


def test_bpmn_parser_marks_unresolved_call_when_child_not_in_filter():
    parser = BpmnStructureParserService(inference_fallback_strategy="skip")
    catalog = [
        {
            "proc_def_id": "parent_def",
            "proc_def_key": "parent_proc",
            "version": "22",
            "deployment_id": "dep_parent",
        }
    ]
    result = parser.parse_definition(
        definition={
            "proc_def_id": "parent_def",
            "proc_def_key": "parent_proc",
            "version": "22",
            "deployment_id": "dep_parent",
            "bpmn_xml_content": _parent_bpmn(),
        },
        catalog=catalog,
        process_name="camunda_dataset",
        process_filters=["parent_proc"],
    )
    assert result.dto is not None
    binding = result.dto.call_bindings["call1"]
    assert binding["status"] == "unresolved"
    assert binding["reason"] == "child_process_out_of_current_filter"
    assert binding["inference_fallback_strategy"] == "skip"


def test_bpmn_parser_quarantines_invalid_xml():
    parser = BpmnStructureParserService()
    result = parser.parse_definition(
        definition={
            "proc_def_id": "broken",
            "proc_def_key": "broken_proc",
            "version": "1",
            "deployment_id": "dep",
            "bpmn_xml_content": "<definitions><process></definitions>",
        },
        catalog=[],
        process_name="camunda_dataset",
        process_filters=["broken_proc"],
    )
    assert result.dto is None
    assert result.quarantine_record is not None
    assert result.quarantine_record["error_code"] == "xml_parse_error"
    assert "source_hint" in result.quarantine_record


@pytest.mark.parametrize(
    "fixture_name, expected_warning",
    [
        ("missing_ids.bpmn", "flow_node_without_id:userTask"),
        ("unknown_extensions.bpmn", None),
    ],
)
def test_bpmn_parser_dirty_xml_is_resilient(fixture_name: str, expected_warning: str | None):
    parser = BpmnStructureParserService()
    result = parser.parse_definition(
        definition={
            "proc_def_id": "dirty_def",
            "proc_def_key": "proc_missing_ids",
            "version": "7",
            "deployment_id": "dep_dirty",
            "bpmn_xml_content": _fixture_bpmn(fixture_name),
        },
        catalog=[],
        process_name="camunda_dataset",
        process_filters=["proc_missing_ids"],
    )
    assert result.quarantine_record is None
    assert result.dto is not None
    warnings = (result.dto.metadata or {}).get("warnings", [])
    if expected_warning is not None:
        assert expected_warning in warnings


def test_bpmn_parser_invalid_fixture_goes_to_quarantine():
    parser = BpmnStructureParserService()
    result = parser.parse_definition(
        definition={
            "proc_def_id": "broken_fixture",
            "proc_def_key": "broken_proc",
            "version": "1",
            "deployment_id": "dep",
            "bpmn_xml_content": _fixture_bpmn("invalid_xml.bpmn"),
        },
        catalog=[],
        process_name="camunda_dataset",
        process_filters=["broken_proc"],
    )
    assert result.dto is None
    assert result.quarantine_record is not None
    assert result.quarantine_record["error_code"] == "xml_parse_error"
    assert result.quarantine_record["source_hint"].startswith("process_name=camunda_dataset")
