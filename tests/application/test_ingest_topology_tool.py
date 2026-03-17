from __future__ import annotations

import json
from pathlib import Path

from src.application.services.topology_extractor_service import TopologyExtractorService
from src.cli import load_yaml_config, prepare_data
from src.infrastructure.repositories.file_based_knowledge_graph_repository import (
    FileBasedKnowledgeGraphRepository,
)
from tools.ingest_topology import main as ingest_topology_main


def _write_minimal_xes(path: Path) -> None:
    content = """<?xml version=\"1.0\" encoding=\"UTF-8\" ?>
<log xes.version=\"1.0\" xes.features=\"nested-attributes\" xmlns=\"http://www.xes-standard.org/\">
  <trace>
    <string key=\"concept:name\" value=\"case_1\"/>
    <event>
      <string key=\"concept:name\" value=\"A\"/>
      <string key=\"org:resource\" value=\"r1\"/>
      <date key=\"time:timestamp\" value=\"2026-03-10T10:00:00.000+00:00\"/>
    </event>
    <event>
      <string key=\"concept:name\" value=\"B\"/>
      <string key=\"org:resource\" value=\"r2\"/>
      <date key=\"time:timestamp\" value=\"2026-03-10T10:01:00.000+00:00\"/>
    </event>
    <event>
      <string key=\"concept:name\" value=\"C\"/>
      <string key=\"org:resource\" value=\"r3\"/>
      <date key=\"time:timestamp\" value=\"2026-03-10T10:02:00.000+00:00\"/>
    </event>
  </trace>
</log>
"""
    path.write_text(content, encoding="utf-8")


def _write_config(path: Path, xes_path: Path, kg_path: Path) -> None:
    content = f"""
data:
  dataset_name: \"test_xes\"
  dataset_label: \"test_xes\"
  log_path: \"{xes_path.as_posix()}\"

mapping:
  adapter: \"xes\"
  knowledge_graph:
    backend: \"file\"
    path: \"{kg_path.as_posix()}\"
    strict_load: false
    ingest_split: \"train\"
  features:
    - name: \"concept:name\"
      role: \"activity\"
      source: \"event\"
      dtype: \"string\"
      fill_na: \"<UNK>\"
      encoding: [\"embedding\"]
    - name: \"org:resource\"
      role: \"resource\"
      source: \"event\"
      dtype: \"string\"
      fill_na: \"UNKNOWN\"
      encoding: [\"embedding\"]
    - name: \"duration\"
      source: \"event\"
      dtype: \"float\"
      fill_na: 0.0
      encoding: [\"z-score\"]

experiment:
  mode: \"train\"
  fraction: 1.0
  split_strategy: \"temporal\"
  train_ratio: 1.0
  split_ratio: [1.0, 0.0, 0.0]

training:
  show_progress: false
  tqdm_disable: true
"""
    path.write_text(content.strip() + "\n", encoding="utf-8")


def test_ingest_topology_creates_file_artifact_for_xes(tmp_path: Path):
    xes_path = tmp_path / "tiny.xes"
    cfg_path = tmp_path / "cfg.yaml"
    kg_path = tmp_path / "knowledge"
    out_path = tmp_path / "summary.json"
    _write_minimal_xes(xes_path)
    _write_config(cfg_path, xes_path, kg_path)

    rc = ingest_topology_main(["--config", str(cfg_path), "--out", str(out_path)])
    assert rc == 0
    assert out_path.exists()

    summary = json.loads(out_path.read_text(encoding="utf-8"))
    assert summary["status"] == "ok"
    assert summary["knowledge_backend"] == "file"
    assert summary["versions_saved"] == ["test_xes"]

    artifact_path = kg_path / "test_xes" / "test_xes" / "process_structure.json"
    assert artifact_path.exists()
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact["schema_version"] == FileBasedKnowledgeGraphRepository.SCHEMA_VERSION


def test_prepare_data_uses_prebuilt_topology_without_sync_extraction(tmp_path: Path, monkeypatch):
    xes_path = tmp_path / "tiny.xes"
    cfg_path = tmp_path / "cfg.yaml"
    kg_path = tmp_path / "knowledge"
    out_path = tmp_path / "summary.json"
    _write_minimal_xes(xes_path)
    _write_config(cfg_path, xes_path, kg_path)
    ingest_topology_main(["--config", str(cfg_path), "--out", str(out_path)])

    def _fail_extract(*args, **kwargs):
        _ = (args, kwargs)
        raise AssertionError("Synchronous topology extraction should not run during prepare_data().")

    monkeypatch.setattr(TopologyExtractorService, "extract_from_logs", _fail_extract)

    config = load_yaml_config(str(cfg_path))
    prepared = prepare_data(config)
    assert len(prepared["train_dataset"]) > 0

    dto = prepared["graph_builder"].knowledge_port.get_process_structure(
        "test_xes",
        process_name="test_xes",
    )
    assert dto is not None
