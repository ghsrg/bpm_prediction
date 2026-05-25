from __future__ import annotations

from src.application.services.topology_drift_audit import (
    PredictionRecord,
    VersionTopologyDiff,
    VersionTopologySnapshot,
    attribute_prediction_error,
    parse_bpmn_topology_snapshot,
    diff_topology_versions,
    parse_xes_prefix_last_activity_lookup,
)


def test_diff_topology_versions_reports_added_and_removed_activities() -> None:
    source = VersionTopologySnapshot(
        version="v1",
        activities={"A", "B", "D"},
        transitions={("A", "B"), ("B", "D")},
    )
    target = VersionTopologySnapshot(
        version="v2",
        activities={"A", "C", "D"},
        transitions={("A", "C"), ("C", "D")},
    )

    diff = diff_topology_versions(source, target)

    assert diff.added_activities == {"C"}
    assert diff.removed_activities == {"B"}
    assert diff.common_activities == {"A", "D"}


def test_diff_topology_versions_reports_changed_successors() -> None:
    source = VersionTopologySnapshot(
        version="v1",
        activities={"A", "B", "C"},
        transitions={("A", "B"), ("B", "C")},
    )
    target = VersionTopologySnapshot(
        version="v2",
        activities={"A", "B", "C"},
        transitions={("A", "C"), ("B", "C")},
    )

    diff = diff_topology_versions(source, target)

    assert diff.changed_successors == {
        "A": {
            "source_successors": {"B"},
            "target_successors": {"C"},
            "removed_successors": {"B"},
            "added_successors": {"C"},
        }
    }


def test_attribute_prediction_marks_unseen_target_class() -> None:
    record = PredictionRecord(
        run_id="r1",
        model_label="struct",
        trace_idx=1,
        step=10,
        process_version="v2",
        prefix_last_activity="A",
        target_label="C",
        pred_label="B",
        strict_correct=False,
        pred_in_mask=True,
        target_in_mask=True,
        strict_error_but_allowed=True,
        mask_cardinality=2,
    )
    diff = VersionTopologyDiff(
        source_version="v1",
        target_version="v2",
        added_activities={"C"},
        removed_activities=set(),
        common_activities={"A", "B"},
        added_transitions={("A", "C")},
        removed_transitions={("A", "B")},
        changed_successors={
            "A": {
                "source_successors": {"B"},
                "target_successors": {"C"},
                "removed_successors": {"B"},
                "added_successors": {"C"},
            }
        },
    )

    attr = attribute_prediction_error(
        record,
        train_activity_counts={"A": 100, "B": 100},
        topology_diff=diff,
    )

    assert attr.target_seen_in_train is False
    assert attr.target_is_new_in_eval_version is True
    assert attr.prefix_transition_changed is True
    assert attr.error_bucket == "unseen_target_class"


def test_attribute_prediction_marks_removed_prediction_class() -> None:
    record = PredictionRecord(
        run_id="r1",
        model_label="struct",
        trace_idx=1,
        step=10,
        process_version="v2",
        prefix_last_activity="A",
        target_label="C",
        pred_label="B",
        strict_correct=False,
        pred_in_mask=False,
        target_in_mask=True,
        strict_error_but_allowed=False,
        mask_cardinality=1,
    )
    diff = VersionTopologyDiff(
        source_version="v1",
        target_version="v2",
        added_activities={"C"},
        removed_activities={"B"},
        common_activities={"A"},
        added_transitions={("A", "C")},
        removed_transitions={("A", "B")},
        changed_successors={
            "A": {
                "source_successors": {"B"},
                "target_successors": {"C"},
                "removed_successors": {"B"},
                "added_successors": {"C"},
            }
        },
    )

    attr = attribute_prediction_error(
        record,
        train_activity_counts={"A": 100, "B": 100, "C": 1},
        topology_diff=diff,
    )

    assert attr.pred_removed_in_eval_version is True
    assert attr.error_bucket == "removed_prediction_class"


def test_attribute_prediction_marks_known_class_changed_transition_zone() -> None:
    record = PredictionRecord(
        run_id="r1",
        model_label="struct",
        trace_idx=1,
        step=10,
        process_version="v2",
        prefix_last_activity="A",
        target_label="C",
        pred_label="B",
        strict_correct=False,
        pred_in_mask=True,
        target_in_mask=True,
        strict_error_but_allowed=True,
        mask_cardinality=2,
    )
    diff = VersionTopologyDiff(
        source_version="v1",
        target_version="v2",
        added_activities=set(),
        removed_activities=set(),
        common_activities={"A", "B", "C"},
        added_transitions={("A", "C")},
        removed_transitions={("A", "B")},
        changed_successors={
            "A": {
                "source_successors": {"B"},
                "target_successors": {"C"},
                "removed_successors": {"B"},
                "added_successors": {"C"},
            }
        },
    )

    attr = attribute_prediction_error(
        record,
        train_activity_counts={"A": 100, "B": 100, "C": 100},
        topology_diff=diff,
    )

    assert attr.target_seen_in_train is True
    assert attr.prefix_transition_changed is True
    assert attr.old_allowed_targets == {"B"}
    assert attr.new_allowed_targets == {"C"}
    assert attr.error_bucket == "changed_transition_zone"


def test_parse_bpmn_topology_snapshot_collapses_gateways(tmp_path) -> None:
    bpmn = tmp_path / "sample.bpmn"
    bpmn.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<definitions xmlns="http://www.omg.org/spec/BPMN/20100524/MODEL">
  <process id="p1">
    <task id="a" name="A" />
    <exclusiveGateway id="g" name="Choice" />
    <task id="b" name="B" />
    <task id="c" name="C" />
    <sequenceFlow id="f1" sourceRef="a" targetRef="g" />
    <sequenceFlow id="f2" sourceRef="g" targetRef="b" />
    <sequenceFlow id="f3" sourceRef="g" targetRef="c" />
  </process>
</definitions>
""",
        encoding="utf-8",
    )

    snapshot = parse_bpmn_topology_snapshot(bpmn, version="v1")

    assert snapshot.activities == {"A", "B", "C"}
    assert snapshot.transitions == {("A", "B"), ("A", "C")}


def test_parse_bpmn_topology_snapshot_can_use_node_ids(tmp_path) -> None:
    bpmn = tmp_path / "sample.bpmn"
    bpmn.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<definitions xmlns="http://www.omg.org/spec/BPMN/20100524/MODEL">
  <process id="p1">
    <task id="task_a" name="A" />
    <exclusiveGateway id="gateway" name="Choice" />
    <task id="task_b" name="B" />
    <sequenceFlow id="f1" sourceRef="task_a" targetRef="gateway" />
    <sequenceFlow id="f2" sourceRef="gateway" targetRef="task_b" />
  </process>
</definitions>
""",
        encoding="utf-8",
    )

    snapshot = parse_bpmn_topology_snapshot(bpmn, version="v1", label_attr="id")

    assert snapshot.activities == {"task_a", "task_b"}
    assert snapshot.transitions == {("task_a", "task_b")}


def test_parse_xes_prefix_last_activity_lookup_uses_trace_idx_and_prefix_len(tmp_path) -> None:
    xes = tmp_path / "sample.xes"
    xes.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<log xmlns="http://www.xes-standard.org/">
  <trace>
    <string key="concept:name" value="case-1" />
    <event><string key="concept:name" value="A" /></event>
    <event><string key="concept:name" value="B" /></event>
  </trace>
  <trace>
    <string key="concept:name" value="case-2" />
    <event><string key="concept:name" value="C" /></event>
  </trace>
</log>
""",
        encoding="utf-8",
    )

    lookup = parse_xes_prefix_last_activity_lookup(xes)

    assert lookup[(0, 1)] == "A"
    assert lookup[(0, 2)] == "B"
    assert lookup[(1, 1)] == "C"
