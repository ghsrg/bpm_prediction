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
        lambda _cfg: ([_trace("c1", "dataset_a", ["A"])], "dataset_a"),  # one event only => no transitions
    )

    with pytest.raises(ValueError, match="No traces or transitions found in the dataset"):
        visualize_topology.main(["--config", "dummy.yaml", "--version", "1", "--out", str(tmp_path / "x.png")])


def test_visualize_topology_auto_falls_back_to_single_available_version(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(
        visualize_topology,
        "_load_train_traces_from_config",
        lambda _cfg: ([_trace("c1", "default", ["A", "B"])], "dataset_a"),
    )

    selected: dict[str, object] = {}

    def _fake_plot(
        self,
        version: str,
        process_name: str | None = None,
        save_path: str | None = None,
        min_edge_freq: int = 1,
    ) -> None:
        _ = self
        selected["save_path"] = save_path
        selected["version"] = version
        selected["process_name"] = process_name
        selected["min_edge_freq"] = min_edge_freq

    monkeypatch.setattr("src.application.services.topology_extractor_service.TopologyExtractorService.plot_topology", _fake_plot)

    rc = visualize_topology.main(
        ["--config", "dummy.yaml", "--version", "1", "--out", str(tmp_path / "x.png"), "--min-freq", "7"]
    )
    out = capsys.readouterr().out

    assert rc == 0
    assert selected["version"] == "default"
    assert selected["process_name"] == "dataset_a"
    assert selected["min_edge_freq"] == 7
    assert "Requested version '1' not found" in out


def test_visualize_topology_raises_with_available_versions_for_multi_version_dataset(monkeypatch, tmp_path):
    monkeypatch.setattr(
        visualize_topology,
        "_load_train_traces_from_config",
        lambda _cfg: (
            [
                _trace("c1", "A", ["X", "Y"]),
                _trace("c2", "B", ["P", "Q"]),
            ],
            "dataset_a",
        ),
    )

    with pytest.raises(ValueError, match=r"Version 'Z' not found\. Available versions: \['A', 'B'\]"):
        visualize_topology.main(["--config", "dummy.yaml", "--version", "Z", "--out", str(tmp_path / "x.png")])


def test_visualize_topology_legacy_camunda_config_without_log_path(monkeypatch, tmp_path, capsys):
    cfg = {
        "data": {
            "process_name": "procurement",
        },
        "camunda": {
            "version_key": "v1",
            "runtime": {
                "runtime_source": "files",
                "export_dir": "data/camunda_exports",
            },
        },
        "experiment": {
            "train_ratio": 1.0,
            "fraction": 1.0,
            "split_strategy": "temporal",
        },
    }

    monkeypatch.setattr(visualize_topology, "load_yaml_config", lambda _path: cfg)

    selected: dict[str, object] = {}

    def _fake_plot(
        self,
        version: str,
        process_name: str | None = None,
        save_path: str | None = None,
        min_edge_freq: int = 1,
    ) -> None:
        _ = self
        selected["version"] = version
        selected["process_name"] = process_name
        selected["save_path"] = save_path
        selected["min_edge_freq"] = min_edge_freq

    monkeypatch.setattr("src.application.services.topology_extractor_service.TopologyExtractorService.plot_topology", _fake_plot)

    rc = visualize_topology.main(
        [
            "--config",
            "dummy.yaml",
            "--version",
            "bpi2012",
            "--min-freq",
            "1",
            "--out",
            str(tmp_path / "topology.png"),
        ]
    )
    out = capsys.readouterr().out

    assert rc == 0
    assert selected["process_name"] == "procurement"
    assert selected["version"] == "v1"
    assert selected["min_edge_freq"] == 1
    assert "Requested version 'bpi2012' not found" in out


def test_visualize_topology_retries_camunda_without_lookback_when_empty(monkeypatch):
    cfg = {
        "data": {
            "process_name": "procurement",
        },
        "camunda": {
            "version_key": "v1",
            "lookback_hours": 48,
            "runtime": {
                "runtime_source": "files",
                "export_dir": "data/camunda_exports",
            },
        },
        "experiment": {
            "train_ratio": 1.0,
            "fraction": 1.0,
            "split_strategy": "temporal",
        },
    }
    monkeypatch.setattr(visualize_topology, "load_yaml_config", lambda _path: cfg)

    calls: list[int] = []

    def _fake_read(self, file_path: str, mapping_config: dict):
        _ = self
        _ = file_path
        lookback = int(mapping_config.get("camunda_adapter", {}).get("lookback_hours", 0) or 0)
        calls.append(lookback)
        if lookback > 0:
            return iter([])
        return iter([_trace("c1", "v1", ["A", "B"])])

    monkeypatch.setattr("src.adapters.ingestion.camunda_trace_adapter.CamundaTraceAdapter.read", _fake_read)

    traces, process_name = visualize_topology._load_train_traces_from_config("dummy.yaml")

    assert process_name == "procurement"
    assert len(traces) == 1
    assert calls == [48, 0]


def test_visualize_topology_handles_not_enough_transitions_without_traceback(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(
        visualize_topology,
        "_load_train_traces_from_config",
        lambda _cfg: ([_trace("c1", "v1", ["A", "B"])], "dataset_a"),
    )

    def _fake_plot(
        self,
        version: str,
        process_name: str | None = None,
        save_path: str | None = None,
        min_edge_freq: int = 1,
    ) -> None:
        _ = (self, version, process_name, save_path, min_edge_freq)
        raise ValueError("No transitions found for process version 'v1' after min_edge_freq=100 filter.")

    monkeypatch.setattr("src.application.services.topology_extractor_service.TopologyExtractorService.plot_topology", _fake_plot)

    rc = visualize_topology.main(
        ["--config", "dummy.yaml", "--version", "v1", "--min-freq", "100", "--out", str(tmp_path / "x.png")]
    )
    out = capsys.readouterr().out

    assert rc == 2
    assert "Not enough transitions to build a graph" in out
    assert "min_edge_freq=100" in out
