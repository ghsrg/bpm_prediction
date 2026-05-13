from __future__ import annotations

from pathlib import Path

import torch
from torch_geometric.data import Data

from src.cli import (
    _build_graph_dataset_sharded,
    _graph_dataset_cache_fingerprint,
    _iter_graphs_from_dataset_payload,
    _load_graph_dataset_cache,
    _save_graph_dataset_cache,
    _save_graph_dataset_cache_sharded,
    _strip_structural_payload,
    _structural_payload_key_from_data,
)
from src.domain.services.baseline_graph_builder import BaselineGraphBuilder
from src.domain.services.feature_encoder import FeatureEncoder
from src.domain.services.prefix_policy import PrefixPolicy
from src.domain.entities.event_record import EventRecord
from src.domain.entities.raw_trace import RawTrace


def _event(idx: int, activity: str) -> EventRecord:
    return EventRecord(
        activity_id=activity,
        timestamp=float(1700000000 + idx),
        resource_id="r1",
        lifecycle="complete",
        position_in_trace=idx,
        duration=1.0,
        time_since_case_start=float(idx),
        time_since_previous_event=1.0 if idx > 0 else 0.0,
        extra={"concept:name": activity, "org:resource": "r1"},
        activity_instance_id=f"ai_{idx}_{activity}",
    )


def _trace(case_id: str, version: str, activities: list[str]) -> RawTrace:
    return RawTrace(
        case_id=case_id,
        process_version=version,
        events=[_event(i, act) for i, act in enumerate(activities)],
        trace_attributes={},
    )


def test_graph_dataset_fingerprint_changes_when_graph_feature_mapping_changes():
    traces = [_trace("c1", "v1", ["A", "B", "C"])]
    config = {
        "mapping": {"adapter": "xes", "xes_adapter": {"activity_key": "concept:name"}},
        "model": {"type": "EOPKGGATv2"},
        "policies": {},
        "features": [],
    }
    base_mapping = {
        "enabled": True,
        "node_numeric": [{"metric": "exec_count", "window": "all_time", "scope": "version", "encoding": ["identity"]}],
    }
    changed_mapping = {
        "enabled": True,
        "node_numeric": [{"metric": "exec_count", "window": "all_time", "scope": "version", "encoding": ["zscore"]}],
    }
    common_kwargs = {
        "config": config,
        "dataset_name": "bpi2012",
        "adapter_kind": "xes",
        "mode": "train",
        "split_strategy": "temporal",
        "split_ratio": (0.7, 0.2, 0.1),
        "train_ratio": 0.7,
        "fraction": 1.0,
        "stats_time_policy": "strict_asof",
        "on_missing_asof_snapshot": "disable_stats",
        "log_path": "Data/sample.xes",
        "traces_all": traces,
        "train_traces": traces,
        "val_traces": [],
        "test_traces": [],
        "activity_vocab": {"<UNK>": 0, "A": 1, "B": 2, "C": 3},
        "resource_vocab": {"<UNK>": 0, "r1": 1},
    }
    fp_base = _graph_dataset_cache_fingerprint(graph_feature_mapping=base_mapping, **common_kwargs)
    fp_changed = _graph_dataset_cache_fingerprint(graph_feature_mapping=changed_mapping, **common_kwargs)
    assert fp_base != fp_changed


def test_graph_dataset_cache_roundtrip(tmp_path: Path):
    train_dataset = [Data(y=torch.tensor([1], dtype=torch.long))]
    val_dataset = [Data(y=torch.tensor([2], dtype=torch.long))]
    test_dataset = [Data(y=torch.tensor([3], dtype=torch.long))]
    version_to_idx = {"v1": 0, "v2": 1}
    snapshot_to_idx = {"k000001": 1}

    cache_dir = tmp_path / "graph_cache"
    fingerprint = "abc123"
    _save_graph_dataset_cache(
        cache_dir=str(cache_dir),
        dataset_name="bpi2012",
        fingerprint=fingerprint,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        version_to_idx=version_to_idx,
        stats_snapshot_version_to_idx=snapshot_to_idx,
        payload_meta={"mode": "train"},
    )

    loaded = _load_graph_dataset_cache(
        cache_dir=str(cache_dir),
        dataset_name="bpi2012",
        fingerprint=fingerprint,
    )
    assert loaded is not None
    assert len(loaded["train_dataset"]) == 1
    assert len(loaded["val_dataset"]) == 1
    assert len(loaded["test_dataset"]) == 1
    assert loaded["version_to_idx"] == version_to_idx
    assert loaded["stats_snapshot_version_to_idx"] == snapshot_to_idx


def test_graph_dataset_cache_sharded_roundtrip(tmp_path: Path):
    cache_dir = tmp_path / "graph_cache"
    dataset_name = "bpi2012"
    fingerprint = "sharded123"
    entry_dir = cache_dir / dataset_name / fingerprint
    train_shards = entry_dir / "train_shards"
    val_shards = entry_dir / "validation_shards"
    test_shards = entry_dir / "test_shards"
    train_shards.mkdir(parents=True, exist_ok=True)
    val_shards.mkdir(parents=True, exist_ok=True)
    test_shards.mkdir(parents=True, exist_ok=True)

    train_data = [Data(y=torch.tensor([1], dtype=torch.long))]
    val_data = [Data(y=torch.tensor([2], dtype=torch.long))]
    test_data = [Data(y=torch.tensor([3], dtype=torch.long))]
    torch.save(train_data, train_shards / "train_00001.pt")
    torch.save(val_data, val_shards / "validation_00001.pt")
    torch.save(test_data, test_shards / "test_00001.pt")

    _save_graph_dataset_cache_sharded(
        cache_dir=str(cache_dir),
        dataset_name=dataset_name,
        fingerprint=fingerprint,
        train_split={
            "kind": "sharded_cache_split",
            "entry_dir": str(entry_dir),
            "split": "train",
            "graphs": 1,
            "shards": [{"path": "train_shards/train_00001.pt", "count": 1}],
        },
        val_split={
            "kind": "sharded_cache_split",
            "entry_dir": str(entry_dir),
            "split": "validation",
            "graphs": 1,
            "shards": [{"path": "validation_shards/validation_00001.pt", "count": 1}],
        },
        test_split={
            "kind": "sharded_cache_split",
            "entry_dir": str(entry_dir),
            "split": "test",
            "graphs": 1,
            "shards": [{"path": "test_shards/test_00001.pt", "count": 1}],
        },
        version_to_idx={"v1": 0},
        stats_snapshot_version_to_idx={"k000001": 1},
        payload_meta={"mode": "train"},
    )

    loaded = _load_graph_dataset_cache(
        cache_dir=str(cache_dir),
        dataset_name=dataset_name,
        fingerprint=fingerprint,
    )
    assert loaded is not None
    assert loaded["train_dataset"]["kind"] == "sharded_cache_split"
    assert loaded["train_dataset"]["graphs"] == 1
    train_items = list(_iter_graphs_from_dataset_payload(loaded["train_dataset"]))
    assert len(train_items) == 1
    assert int(train_items[0].y.view(-1)[0].item()) == 1


def test_sharded_cache_deduplicates_structural_payloads(tmp_path: Path):
    _ = tmp_path
    shared_struct_x = torch.tensor([[1.0], [2.0]], dtype=torch.float32)
    shared_edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    shared_edge_weight = torch.tensor([0.5], dtype=torch.float32)
    shared_node_to_class = torch.tensor([0, 1], dtype=torch.long)

    graph_a = Data(y=torch.tensor([1]), struct_x=shared_struct_x, structural_edge_index=shared_edge_index)
    graph_a.structural_edge_weight = shared_edge_weight
    graph_a.struct_node_to_class_index = shared_node_to_class
    graph_b = Data(y=torch.tensor([1]), struct_x=shared_struct_x, structural_edge_index=shared_edge_index)
    graph_b.structural_edge_weight = shared_edge_weight
    graph_b.struct_node_to_class_index = shared_node_to_class

    payload_key = _structural_payload_key_from_data(graph_a)
    stripped_a, payloads = _strip_structural_payload(graph_a, {})
    stripped_b, payloads = _strip_structural_payload(graph_b, payloads)

    assert payload_key in payloads
    assert len(payloads) == 1
    assert not hasattr(stripped_a, "struct_x")
    assert not hasattr(stripped_b, "struct_x")
    assert getattr(stripped_a, "structural_payload_key") == payload_key


def test_structural_payload_key_is_identity_based_not_tensor_content(monkeypatch):
    shared_struct_x = torch.tensor([[1.0], [2.0]], dtype=torch.float32)
    graph = Data(y=torch.tensor([1]), struct_x=shared_struct_x)

    def _raise_if_content_digest_is_used(tensor):
        _ = tensor
        raise AssertionError("content tensor digest should not be used for sharded structural payload key")

    monkeypatch.setattr("src.cli._tensor_digest", _raise_if_content_digest_is_used)

    key = _structural_payload_key_from_data(graph)

    assert str(id(shared_struct_x)) in key


def test_iter_graphs_rehydrates_deduplicated_structural_payload(tmp_path: Path):
    entry_dir = tmp_path / "entry"
    shard_dir = entry_dir / "test_shards"
    shard_dir.mkdir(parents=True)
    graph = Data(y=torch.tensor([1], dtype=torch.long))
    graph.structural_payload_key = "payload-a"
    torch.save(
        {
            "schema": 2,
            "format": "dedup_structural_payloads",
            "graphs": [graph],
            "structural_payloads": {
                "payload-a": {
                    "struct_x": torch.tensor([[1.0], [2.0]], dtype=torch.float32),
                    "structural_edge_index": torch.tensor([[0], [1]], dtype=torch.long),
                    "structural_edge_weight": torch.tensor([0.5], dtype=torch.float32),
                    "struct_node_to_class_index": torch.tensor([0, 1], dtype=torch.long),
                }
            },
        },
        shard_dir / "test_00001.pt",
    )

    items = list(
        _iter_graphs_from_dataset_payload(
            {
                "kind": "sharded_cache_split",
                "entry_dir": str(entry_dir),
                "split": "test",
                "graphs": 1,
                "shards": [{"path": "test_shards/test_00001.pt", "count": 1}],
            }
        )
    )

    assert len(items) == 1
    assert torch.equal(items[0].struct_x, torch.tensor([[1.0], [2.0]], dtype=torch.float32))
    assert torch.equal(items[0].structural_edge_index, torch.tensor([[0], [1]], dtype=torch.long))
    assert not hasattr(items[0], "structural_payload_key")


def test_build_graph_dataset_sharded_logs_without_name_error(mock_feature_configs, tmp_path: Path):
    traces = [_trace("c1", "v1", ["A", "B"])]
    encoder = FeatureEncoder(feature_configs=mock_feature_configs, traces=traces)
    graph_builder = BaselineGraphBuilder(feature_encoder=encoder)

    payload = _build_graph_dataset_sharded(
        traces=traces,
        prefix_policy=PrefixPolicy(),
        graph_builder=graph_builder,  # type: ignore[arg-type]
        version_to_idx={},
        stats_snapshot_version_to_idx={},
        show_progress=False,
        tqdm_disable=True,
        desc="Build test graphs",
        progress_stage="test.build_graph",
        entry_dir=tmp_path / "entry",
        split_key="test",
        shard_size=1,
        max_ram_bytes=None,
    )

    assert payload["graphs"] == 1
    assert payload["shards"]
