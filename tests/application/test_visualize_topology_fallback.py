from __future__ import annotations

import pytest

from src.domain.entities.event_record import EventRecord
from src.domain.entities.raw_trace import RawTrace
from tools import visualize_topology


def _event(idx: int, activity: str) -> EventRecord:
    return EventRecord(
        activity_id=activity,
        timestamp=float(1700001000 + idx),
        resource_id="R",
        lifecycle="complete",
        position_in_trace=idx,
        duration=1.0,
        time_since_case_start=float(idx),
        time_since_previous_event=1.0 if idx > 0 else 0.0,
        extra={"concept:name": activity},
        activity_instance_id=f"ai_{idx}",
    )


def _trace(case_id: str, version: str, activities: list[str]) -> RawTrace:
    events = [_event(i, act) for i, act in enumerate(activities)]
    return RawTrace(case_id=case_id, process_version=version, events=events, trace_attributes={})


def test_visualize_topology_raises_when_no_transitions_found(monkeypatch, tmp_path):
    monkeypatch.setattr(
        visualize_topology,
        "_load_train_traces_from_config",
        lambda _cfg: [_trace("c1", "1", ["A"])],  # one event only => no transitions
    )

    with pytest.raises(ValueError, match="No traces or transitions found in the dataset"):
        visualize_topology.main(["--config", "dummy.yaml", "--version", "1", "--out", str(tmp_path / "x.png")])


def test_visualize_topology_auto_falls_back_to_single_available_version(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(
        visualize_topology,
        "_load_train_traces_from_config",
        lambda _cfg: [_trace("c1", "default", ["A", "B"])],
    )

    selected: dict[str, str] = {}

    def _fake_plot(self, version: str, save_path: str | None = None) -> None:
        _ = self
        _ = save_path
        selected["version"] = version

    monkeypatch.setattr("src.application.services.topology_extractor_service.TopologyExtractorService.plot_topology", _fake_plot)

    rc = visualize_topology.main(["--config", "dummy.yaml", "--version", "1", "--out", str(tmp_path / "x.png")])
    out = capsys.readouterr().out

    assert rc == 0
    assert selected["version"] == "default"
    assert "Requested version '1' not found" in out


def test_visualize_topology_raises_with_available_versions_for_multi_version_dataset(monkeypatch, tmp_path):
    monkeypatch.setattr(
        visualize_topology,
        "_load_train_traces_from_config",
        lambda _cfg: [
            _trace("c1", "A", ["X", "Y"]),
            _trace("c2", "B", ["P", "Q"]),
        ],
    )

    with pytest.raises(ValueError, match=r"Version 'Z' not found\. Available versions: \['A', 'B'\]"):
        visualize_topology.main(["--config", "dummy.yaml", "--version", "Z", "--out", str(tmp_path / "x.png")])

