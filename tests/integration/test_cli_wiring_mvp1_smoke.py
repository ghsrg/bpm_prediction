from __future__ import annotations

from typing import Iterator

import pytest

from src.adapters.ingestion.xes_adapter import XESAdapter
from src.cli import prepare_data
from src.domain.entities.raw_trace import RawTrace


@pytest.mark.mvp1_regression
def test_cli_prepare_data_wiring_mvp1_with_in_memory_adapter(monkeypatch, mock_raw_trace):
    def _mock_read(self, file_path: str, mapping_config: dict) -> Iterator[RawTrace]:
        _ = self
        _ = file_path
        _ = mapping_config
        return iter([mock_raw_trace, mock_raw_trace])

    monkeypatch.setattr(XESAdapter, "read", _mock_read)

    cfg = {
        "data": {
            "log_path": "in_memory.xes",
            "dataset_label": "mvp1_smoke",
        },
        "experiment": {
            "mode": "train",
            "fraction": 1.0,
            "split_strategy": "temporal",
            "train_ratio": 1.0,
            "split_ratio": [0.6, 0.2, 0.2],
        },
        "mapping": {
            "features": [
                {
                    "name": "concept:name",
                    "source_key": "activity",
                    "source": "event",
                    "dtype": "string",
                    "fill_na": "<UNK>",
                    "encoding": ["embedding"],
                    "role": "activity",
                },
                {
                    "name": "org:resource",
                    "source": "event",
                    "dtype": "string",
                    "fill_na": "<UNK>",
                    "encoding": ["embedding"],
                },
                {
                    "name": "cost",
                    "source_key": "amount",
                    "source": "event",
                    "dtype": "float",
                    "fill_na": 0.0,
                    "encoding": ["z-score"],
                },
            ]
        },
        "policies": {"activity_fallback_feature": "concept:name"},
        "model": {"type": "BaselineGCN", "hidden_dim": 8, "dropout": 0.0},
        "training": {"show_progress": False, "tqdm_disable": True},
    }

    prepared = prepare_data(cfg)

    assert isinstance(prepared, dict)
    assert "train_dataset" in prepared and "val_dataset" in prepared and "test_dataset" in prepared
    assert prepared["feature_layout"].num_dim >= 1
    assert len(prepared["activity_vocab"]) >= 2  # includes <UNK> + at least one observed activity

