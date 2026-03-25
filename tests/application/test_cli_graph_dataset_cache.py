from __future__ import annotations

from pathlib import Path

import torch
from torch_geometric.data import Data

from src.cli import (
    _graph_dataset_cache_fingerprint,
    _iter_graphs_from_dataset_payload,
    _load_graph_dataset_cache,
    _save_graph_dataset_cache,
    _save_graph_dataset_cache_sharded,
)
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
