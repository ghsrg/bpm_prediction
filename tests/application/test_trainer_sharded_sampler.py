from __future__ import annotations

from pathlib import Path

import torch
from torch_geometric.data import Data

from src.application.use_cases.trainer import ModelTrainer, ShardedGraphDataset
from src.domain.models.baseline_gcn import BaselineGCN
from src.domain.services.baseline_graph_builder import BaselineGraphBuilder
from src.domain.services.feature_encoder import FeatureEncoder
from src.domain.services.prefix_policy import PrefixPolicy


class _DummyAdapter:
    def read(self, file_path: str, mapping_config: dict):
        _ = file_path
        _ = mapping_config
        return iter(())


def _dummy_graph(label: int) -> Data:
    return Data(
        x_cat=torch.zeros((2, 1), dtype=torch.long),
        x_num=torch.zeros((2, 1), dtype=torch.float32),
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
        edge_type=torch.tensor([0], dtype=torch.long),
        y=torch.tensor([int(label)], dtype=torch.long),
        num_nodes=2,
    )


def _build_trainer(mock_feature_configs, mock_raw_trace, tmp_path: Path) -> ModelTrainer:
    traces = [mock_raw_trace, mock_raw_trace]
    prefix_policy = PrefixPolicy()
    encoder = FeatureEncoder(feature_configs=mock_feature_configs, traces=traces)
    graph_builder = BaselineGraphBuilder(feature_encoder=encoder)
    model = BaselineGCN(
        feature_layout=encoder.feature_layout,
        hidden_dim=8,
        output_dim=len(encoder.categorical_vocabs[encoder.activity_feature_name]),
        dropout=0.0,
    )
    return ModelTrainer(
        xes_adapter=_DummyAdapter(),
        prefix_policy=prefix_policy,
        graph_builder=graph_builder,
        model=model,
        log_path="dummy.xes",
        config={
            "epochs": 1,
            "batch_size": 2,
            "learning_rate": 0.001,
            "device": "cpu",
            "show_progress": False,
            "tqdm_disable": True,
            "checkpoint_dir": str(tmp_path),
            "seed": 42,
            "experiment_config": {"name": "pytest_sampler", "mode": "train"},
        },
        prepared_data={},
    )


def test_shard_sampler_keeps_shard_locality(mock_feature_configs, mock_raw_trace, tmp_path: Path):
    entry_dir = tmp_path / "cache_entry"
    shard_dir = entry_dir / "train_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard1 = [_dummy_graph(0), _dummy_graph(1), _dummy_graph(2)]
    shard2 = [_dummy_graph(3), _dummy_graph(4), _dummy_graph(5)]
    torch.save(shard1, shard_dir / "train_00001.pt")
    torch.save(shard2, shard_dir / "train_00002.pt")

    ds = ShardedGraphDataset(
        entry_dir=str(entry_dir),
        shards=[
            {"path": "train_shards/train_00001.pt", "count": len(shard1)},
            {"path": "train_shards/train_00002.pt", "count": len(shard2)},
        ],
        max_cached_shards=2,
    )
    trainer = _build_trainer(mock_feature_configs, mock_raw_trace, tmp_path)
    loader = trainer._create_data_loader_from_source(ds, shuffle=True)

    sampler = loader.sampler
    indices = list(iter(sampler))
    assert sorted(indices) == list(range(len(ds)))

    ranges = ds.shard_index_ranges()
    shard_of = {}
    for shard_idx, (start, end) in enumerate(ranges):
        for idx in range(start, end):
            shard_of[idx] = shard_idx

    shard_seq = [shard_of[idx] for idx in indices]
    transitions = sum(1 for i in range(1, len(shard_seq)) if shard_seq[i] != shard_seq[i - 1])
    assert transitions <= 1


def test_sharded_loader_forces_single_worker(mock_feature_configs, mock_raw_trace, tmp_path: Path):
    entry_dir = tmp_path / "cache_entry"
    shard_dir = entry_dir / "train_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_data = [_dummy_graph(0), _dummy_graph(1)]
    torch.save(shard_data, shard_dir / "train_00001.pt")

    ds = ShardedGraphDataset(
        entry_dir=str(entry_dir),
        shards=[{"path": "train_shards/train_00001.pt", "count": len(shard_data)}],
        max_cached_shards=2,
    )
    trainer = _build_trainer(mock_feature_configs, mock_raw_trace, tmp_path)
    trainer.dataloader_num_workers = 2
    loader = trainer._create_data_loader_from_source(ds, shuffle=True)
    assert int(loader.num_workers) == 0
