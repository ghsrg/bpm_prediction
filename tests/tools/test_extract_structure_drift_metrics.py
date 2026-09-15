from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


TOOL = Path(__file__).resolve().parents[2] / "tools" / "extract_structure_drift_metrics.py"


def _node(node_id: str, name: str, tag: str) -> dict[str, str]:
    return {
        "id": node_id,
        "name": name,
        "bpmn_tag": tag,
        "type": tag,
        "activity_type": "task" if tag == "task" else tag,
    }


def _write_structure(root: Path, version: str, nodes: list[dict[str, str]], pairs: list[tuple[str, str]]) -> None:
    path = root / "loan_fixture" / version / "process_structure.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    edges = [
        {"id": f"flow_{index:03d}", "edge_type": "sequence", "source": source, "target": target}
        for index, (source, target) in enumerate(pairs, start=1)
    ]
    payload = {
        "schema_version": "1.0",
        "repository_backend": "file",
        "process_name": "loan_fixture",
        "version": version,
        "dto": {"nodes": nodes, "edges": edges, "allowed_edges": [list(pair) for pair in pairs]},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_projected_structure(
    root: Path,
    version: str,
    pairs: list[tuple[str, str]],
    *,
    metadata_ids: set[str] | None = None,
) -> None:
    path = root / "loan_fixture" / version / "process_structure.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    endpoint_ids = {item for pair in pairs for item in pair}
    metadata_ids = endpoint_ids if metadata_ids is None else metadata_ids
    payload = {
        "schema_version": "1.0",
        "repository_backend": "file",
        "process_name": "loan_fixture",
        "version": version,
        "dto": {
            "nodes": [],
            "edges": [],
            "allowed_edges": [list(pair) for pair in pairs],
            "graph_topology": {"allowed_edges": [list(pair) for pair in pairs]},
            "node_metadata": {
                node_id: {"activity_name": f"Human {node_id}", "activity_type": ""}
                for node_id in metadata_ids
            },
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_cli_extracts_canonical_relations_cycles_and_adjacent_deltas(tmp_path: Path):
    structure_root = tmp_path / "knowledge_graph"
    output_dir = tmp_path / "metrics"
    v1_nodes = [
        _node("start", "", "startEvent"),
        _node("a", "A", "task"),
        _node("gateway", "", "parallelGateway"),
        _node("b", "B", "task"),
        _node("end", "", "endEvent"),
    ]
    v2_nodes = [
        *v1_nodes,
        _node("c", "C", "task"),
    ]
    _write_structure(
        structure_root,
        "v1",
        v1_nodes,
        [("start", "a"), ("a", "gateway"), ("gateway", "b"), ("b", "a")],
    )
    v2_pairs = [("start", "a"), ("a", "gateway"), ("gateway", "b"), ("gateway", "c"), ("b", "end"), ("c", "end")]
    for version in ("v2", "v3", "v4", "v5"):
        _write_structure(structure_root, version, v2_nodes, v2_pairs)

    result = subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--dataset",
            "loan",
            "--structure-root",
            str(structure_root),
            "--output-dir",
            str(output_dir),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    for name in (
        "structure_version_metrics.csv",
        "structure_transition_metrics.csv",
        "structure_relation_deltas.csv",
        "structure_source_inventory.csv",
        "structure_validation_report.csv",
        "structure_metric_dictionary.md",
    ):
        assert (output_dir / name).is_file()

    versions = _rows(output_dir / "structure_version_metrics.csv")
    v1 = next(row for row in versions if row["version"] == "v1")
    assert v1["canonical_task_relation_count"] == "2"
    assert v1["canonical_task_relation_cycle_present"] == "true"
    assert v1["parallel_gateway_count"] == "1"
    assert len(v1["structure_source_sha256"]) == 64

    transition = _rows(output_dir / "structure_transition_metrics.csv")[0]
    assert transition["source_version"] == "v1"
    assert transition["target_version"] == "v2"
    assert transition["added_task_count"] == "1"
    assert transition["removed_relation_count"] == "1"
    assert transition["added_relation_count"] == "1"
    assert transition["cycle_presence_changed"] == "true"
    assert transition["net_task_count_delta"] == "1"
    assert transition["net_relation_count_delta"] == "0"
    assert 0.0 <= float(transition["activity_set_delta"]) <= 1.0
    assert 0.0 <= float(transition["relation_set_delta"]) <= 1.0

    deltas = _rows(output_dir / "structure_relation_deltas.csv")
    assert {tuple((row["change_direction"], row["source_task"], row["target_task"])) for row in deltas} == {
        ("added", "A", "C"),
        ("removed", "B", "A"),
    }


def test_cli_rejects_duplicate_task_labels_instead_of_merging_them(tmp_path: Path):
    structure_root = tmp_path / "knowledge_graph"
    nodes = [_node("a", "Repeated", "task"), _node("b", "Repeated", "task")]
    for version in ("v1", "v2", "v3", "v4", "v5"):
        _write_structure(structure_root, version, nodes, [("a", "b")])

    result = subprocess.run(
        [sys.executable, str(TOOL), "--dataset", "loan", "--structure-root", str(structure_root)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "duplicate task label" in result.stderr.lower()


def test_cli_extracts_projected_topology_when_explicit_lists_are_empty(tmp_path: Path):
    structure_root = tmp_path / "knowledge_graph"
    output_dir = tmp_path / "metrics"
    version_edges = {
        "v1": [("task_a", "task_b"), ("task_b", "task_c")],
        "v2": [("task_a", "task_b"), ("task_a", "task_c")],
        "v3": [("task_a", "task_b"), ("task_a", "task_c")],
        "v4": [("task_a", "task_b"), ("task_a", "task_c")],
        "v5": [("task_a", "task_b"), ("task_a", "task_c")],
    }
    for version, pairs in version_edges.items():
        _write_projected_structure(structure_root, version, pairs)

    result = subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--dataset",
            "loan",
            "--structure-root",
            str(structure_root),
            "--output-dir",
            str(output_dir),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    versions = _rows(output_dir / "structure_version_metrics.csv")
    v1 = next(row for row in versions if row["version"] == "v1")
    assert v1["topology_layout"] == "projected_allowed_edges"
    assert v1["raw_node_count"] == ""
    assert v1["raw_sequence_flow_count"] == ""
    assert v1["projected_node_metadata_count"] == "3"
    assert v1["task_count"] == "3"
    assert v1["task_keys_json"] == '["task_a", "task_b", "task_c"]'
    assert v1["projected_allowed_edge_count"] == "2"

    transitions = _rows(output_dir / "structure_transition_metrics.csv")
    first = transitions[0]
    assert first["activity_jaccard"] == "1.0"
    assert first["activity_set_delta"] == "0.0"
    assert first["relation_jaccard"] == "0.3333333333333333"
    assert first["relation_set_delta"] == "0.6666666666666667"
    assert first["net_task_count_delta"] == "0"
    assert first["net_relation_count_delta"] == "0"
    assert first["unchanged_transition"] == "false"


def test_cli_rejects_projected_edges_without_metadata_for_every_endpoint(tmp_path: Path):
    structure_root = tmp_path / "knowledge_graph"
    for version in ("v1", "v2", "v3", "v4", "v5"):
        _write_projected_structure(
            structure_root,
            version,
            [("task_a", "task_b")],
            metadata_ids={"task_a"},
        )

    result = subprocess.run(
        [sys.executable, str(TOOL), "--dataset", "loan", "--structure-root", str(structure_root)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "node metadata" in result.stderr.lower()


def test_cli_rejects_zero_task_projected_artifacts_with_topology_source_signal(tmp_path: Path):
    structure_root = tmp_path / "knowledge_graph"
    for version in ("v1", "v2", "v3", "v4", "v5"):
        _write_projected_structure(structure_root, version, [], metadata_ids={"task_a"})

    result = subprocess.run(
        [sys.executable, str(TOOL), "--dataset", "loan", "--structure-root", str(structure_root)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "zero task" in result.stderr.lower()
