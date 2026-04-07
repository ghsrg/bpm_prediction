from __future__ import annotations

import math

from src.adapters.ingestion.xes_adapter import XESAdapter


def test_xes_adapter_populates_start_ts_and_complete_ts_for_completed_events(tmp_path):
    xes_payload = """<?xml version="1.0" encoding="UTF-8" ?>
<log xes.version="1.0" xes.features="nested-attributes">
  <trace>
    <string key="concept:name" value="case_1"/>
    <event>
      <string key="concept:name" value="A"/>
      <string key="lifecycle:transition" value="start"/>
      <date key="time:timestamp" value="2020-01-01T00:00:00+00:00"/>
    </event>
    <event>
      <string key="concept:name" value="A"/>
      <string key="lifecycle:transition" value="complete"/>
      <date key="time:timestamp" value="2020-01-01T00:00:10+00:00"/>
    </event>
    <event>
      <string key="concept:name" value="B"/>
      <string key="lifecycle:transition" value="complete"/>
      <date key="time:timestamp" value="2020-01-01T00:00:20+00:00"/>
    </event>
  </trace>
</log>
"""
    log_path = tmp_path / "sample.xes"
    log_path.write_text(xes_payload, encoding="utf-8")

    adapter = XESAdapter()
    traces = list(adapter.read(str(log_path), mapping_config={"xes_adapter": {}}))
    assert len(traces) == 1
    assert len(traces[0].events) == 2

    event_a = traces[0].events[0]
    event_b = traces[0].events[1]

    assert math.isclose(float(event_a.extra["start_ts"]), 1577836800.0, rel_tol=0.0, abs_tol=1e-9)
    assert math.isclose(float(event_a.extra["complete_ts"]), 1577836810.0, rel_tol=0.0, abs_tol=1e-9)
    assert math.isclose(float(event_b.extra["start_ts"]), float(event_b.timestamp), rel_tol=0.0, abs_tol=1e-9)
    assert math.isclose(float(event_b.extra["complete_ts"]), float(event_b.timestamp), rel_tol=0.0, abs_tol=1e-9)


def test_xes_adapter_uses_configured_start_transitions_for_pairing(tmp_path):
    xes_payload = """<?xml version="1.0" encoding="UTF-8" ?>
<log xes.version="1.0" xes.features="nested-attributes">
  <trace>
    <string key="concept:name" value="case_1"/>
    <event>
      <string key="concept:name" value="A"/>
      <string key="lifecycle:transition" value="assign"/>
      <date key="time:timestamp" value="2020-01-01T00:00:00+00:00"/>
    </event>
    <event>
      <string key="concept:name" value="A"/>
      <string key="lifecycle:transition" value="complete"/>
      <date key="time:timestamp" value="2020-01-01T00:00:10+00:00"/>
    </event>
  </trace>
</log>
"""
    log_path = tmp_path / "sample_assign_start.xes"
    log_path.write_text(xes_payload, encoding="utf-8")

    adapter = XESAdapter()
    traces = list(
        adapter.read(
            str(log_path),
            mapping_config={"xes_adapter": {"start_transitions": ["start", "assign"]}},
        )
    )
    assert len(traces) == 1
    assert len(traces[0].events) == 1
    event_a = traces[0].events[0]
    assert math.isclose(float(event_a.extra["start_ts"]), 1577836800.0, rel_tol=0.0, abs_tol=1e-9)
    assert math.isclose(float(event_a.extra["complete_ts"]), 1577836810.0, rel_tol=0.0, abs_tol=1e-9)


def test_xes_adapter_pairs_assign_and_complete_when_complete_resource_is_missing(tmp_path):
    xes_payload = """<?xml version="1.0" encoding="UTF-8" ?>
<log xes.version="1.0" xes.features="nested-attributes">
  <trace>
    <string key="concept:name" value="case_1"/>
    <event>
      <string key="concept:name" value="A"/>
      <string key="lifecycle:transition" value="assign"/>
      <string key="org:resource" value="operator"/>
      <date key="time:timestamp" value="2020-01-01T00:00:00+00:00"/>
    </event>
    <event>
      <string key="concept:name" value="A"/>
      <string key="lifecycle:transition" value="complete"/>
      <date key="time:timestamp" value="2020-01-01T00:00:10+00:00"/>
    </event>
  </trace>
</log>
"""
    log_path = tmp_path / "sample_assign_missing_complete_resource.xes"
    log_path.write_text(xes_payload, encoding="utf-8")

    adapter = XESAdapter()
    traces = list(
        adapter.read(
            str(log_path),
            mapping_config={"xes_adapter": {"start_transitions": ["start", "assign"]}},
        )
    )
    assert len(traces) == 1
    assert len(traces[0].events) == 1
    event_a = traces[0].events[0]
    assert math.isclose(float(event_a.extra["start_ts"]), 1577836800.0, rel_tol=0.0, abs_tol=1e-9)
    assert math.isclose(float(event_a.extra["complete_ts"]), 1577836810.0, rel_tol=0.0, abs_tol=1e-9)


def test_xes_adapter_populates_active_set_snapshots_after_complete(tmp_path):
    xes_payload = """<?xml version="1.0" encoding="UTF-8" ?>
<log xes.version="1.0" xes.features="nested-attributes">
  <trace>
    <string key="concept:name" value="case_1"/>
    <event>
      <string key="concept:name" value="A"/>
      <string key="lifecycle:transition" value="assign"/>
      <date key="time:timestamp" value="2020-01-01T00:00:00+00:00"/>
    </event>
    <event>
      <string key="concept:name" value="A"/>
      <string key="lifecycle:transition" value="complete"/>
      <date key="time:timestamp" value="2020-01-01T00:00:10+00:00"/>
    </event>
    <event>
      <string key="concept:name" value="B"/>
      <string key="lifecycle:transition" value="assign"/>
      <date key="time:timestamp" value="2020-01-01T00:00:10+00:00"/>
    </event>
    <event>
      <string key="concept:name" value="C"/>
      <string key="lifecycle:transition" value="assign"/>
      <date key="time:timestamp" value="2020-01-01T00:00:10+00:00"/>
    </event>
    <event>
      <string key="concept:name" value="B"/>
      <string key="lifecycle:transition" value="complete"/>
      <date key="time:timestamp" value="2020-01-01T00:00:20+00:00"/>
    </event>
    <event>
      <string key="concept:name" value="C"/>
      <string key="lifecycle:transition" value="complete"/>
      <date key="time:timestamp" value="2020-01-01T00:00:30+00:00"/>
    </event>
  </trace>
</log>
"""
    log_path = tmp_path / "sample_active_snapshots.xes"
    log_path.write_text(xes_payload, encoding="utf-8")

    adapter = XESAdapter()
    traces = list(
        adapter.read(
            str(log_path),
            mapping_config={"xes_adapter": {"start_transitions": ["assign", "start"]}},
        )
    )
    assert len(traces) == 1
    assert len(traces[0].events) == 3

    event_a = traces[0].events[0]
    event_b = traces[0].events[1]
    event_c = traces[0].events[2]

    assert event_a.activity_id == "A"
    assert event_b.activity_id == "B"
    assert event_c.activity_id == "C"

    assert event_a.extra["active_activities_after_complete"] == []
    assert event_a.extra["active_activity_counts_after_complete"] == {}

    assert event_b.extra["active_activities_after_complete"] == ["C"]
    assert event_b.extra["active_activity_counts_after_complete"] == {"C": 1}

    assert event_c.extra["active_activities_after_complete"] == []
    assert event_c.extra["active_activity_counts_after_complete"] == {}
