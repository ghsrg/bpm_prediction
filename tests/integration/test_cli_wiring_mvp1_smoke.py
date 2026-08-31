from __future__ import annotations

import sys
from typing import Iterator

import pytest
import torch
import yaml
from torch_geometric.data import Data

from src.adapters.ingestion.xes_adapter import XESAdapter
from src.cli import main, prepare_data
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


def test_cli_mou_mode_skips_model_and_trainer(monkeypatch, tmp_path):
    checkpoint = tmp_path / "encoder-reference.pth"
    torch.save({"encoder_state": {}}, checkpoint)
    monkeypatch.setattr("src.cli.create_model", lambda **_: pytest.fail("model must not be created"))
    monkeypatch.setattr("src.cli.ModelTrainer", lambda **_: pytest.fail("trainer must not be created"))
    monkeypatch.setattr("src.cli.TopologyMaskUniformEvaluator", _RecordingMouEvaluator)
    monkeypatch.setattr("src.cli.prepare_data", lambda *_args, **_kwargs: _prepared_mou_payload())
    monkeypatch.setattr(sys, "argv", ["cli.py", "--config", str(_write_mou_config(tmp_path, checkpoint))])

    main()

    assert _RecordingMouEvaluator.last_kwargs["empty_mask_policy"] == "raise"
    assert _RecordingMouEvaluator.last_kwargs["evaluation_seed"] == 41
    assert _RecordingMouEvaluator.last_kwargs["mc_draws"] == 100
    assert _RecordingMouEvaluator.last_kwargs["drift_window_sliding"] == 10
    assert _RecordingMouEvaluator.last_samples == 1


def test_cli_mou_mode_iterates_sharded_dataset_payload(monkeypatch, tmp_path):
    checkpoint = tmp_path / "encoder-reference.pth"
    torch.save({"encoder_state": {}}, checkpoint)
    sample = _mou_sample()
    payload = _prepared_mou_payload()
    payload["test_dataset"] = {"kind": "sharded_cache_split", "graphs": 1, "shards": [{"path": "test_00001.pt"}]}
    calls = {"used": False}

    def _fake_iter_graphs(dataset_payload):
        assert dataset_payload is payload["test_dataset"]
        calls["used"] = True
        return iter([sample])

    monkeypatch.setattr("src.cli.create_model", lambda **_: pytest.fail("model must not be created"))
    monkeypatch.setattr("src.cli.ModelTrainer", lambda **_: pytest.fail("trainer must not be created"))
    monkeypatch.setattr("src.cli.TopologyMaskUniformEvaluator", _RecordingMouEvaluator)
    monkeypatch.setattr("src.cli._iter_graphs_from_dataset_payload", _fake_iter_graphs)
    monkeypatch.setattr("src.cli.prepare_data", lambda *_args, **_kwargs: payload)
    monkeypatch.setattr(sys, "argv", ["cli.py", "--config", str(_write_mou_config(tmp_path, checkpoint))])

    main()

    assert calls["used"] is True
    assert _RecordingMouEvaluator.last_samples == 1


def test_cli_mou_mode_logs_as_eval_drift_to_mlflow(monkeypatch, tmp_path):
    checkpoint = tmp_path / "encoder-reference.pth"
    torch.save({"encoder_state": {}}, checkpoint)
    tracker = _RecordingTracker()

    monkeypatch.setattr("src.cli.create_model", lambda **_: pytest.fail("model must not be created"))
    monkeypatch.setattr("src.cli.ModelTrainer", lambda **_: pytest.fail("trainer must not be created"))
    monkeypatch.setattr("src.cli.TopologyMaskUniformEvaluator", _RecordingMouEvaluator)
    monkeypatch.setattr("src.cli.prepare_data", lambda *_args, **_kwargs: _prepared_mou_payload())
    monkeypatch.setattr("src.cli.MLflowTracker", lambda **kwargs: tracker.configure(**kwargs))
    monkeypatch.setattr(sys, "argv", ["cli.py", "--config", str(_write_mou_config(tmp_path, checkpoint, tracking=True))])

    main()

    assert tracker.tags["mode"] == "eval_drift"
    assert tracker.tags["evaluation_mode"] == "eval_topology_mask_uniform"
    assert tracker.params["experiment.mode"] == "eval_drift"
    assert tracker.params["experiment.evaluation_mode"] == "eval_topology_mask_uniform"
    assert tracker.params["model.type"] == "MOU"
    assert tracker.params["model_type"] == "MOU"
    assert tracker.metrics["uniform_mask_expected_accuracy"] == 1.0
    assert tracker.metrics["drift_window_strict_macro_f1"] == 0.5
    assert tracker.metrics["drift_window_macro_f1"] == 0.6
    assert tracker.closed is True


class _RecordingMouEvaluator:
    last_kwargs: dict | None = None
    last_samples: int | None = None

    def __init__(self, **kwargs):
        type(self).last_kwargs = kwargs

    def evaluate(self, samples):
        rows = list(samples)
        type(self).last_samples = len(rows)
        return {
            "test_metrics": {"uniform_mask_expected_accuracy": 1.0},
            "monte_carlo": {"evaluation_seed": self.last_kwargs["evaluation_seed"], "draws": self.last_kwargs["mc_draws"]},
            "drift_metrics": [
                {
                    "window_index": 0,
                    "window_strict_test_macro_f1_mc_mean": 0.5,
                    "window_test_macro_f1_mc_mean": 0.6,
                }
            ],
        }


class _RecordingTracker:
    def __init__(self):
        self.tags = {}
        self.metrics = {}
        self.closed = False

    def configure(self, **kwargs):
        self.kwargs = kwargs
        return self

    def log_tag(self, key, value):
        self.tags[key] = value

    def log_params(self, params):
        self.params = params

    def log_metric(self, key, value, step=None):
        self.metrics[key] = value

    def close(self):
        self.closed = True


def _prepared_mou_payload() -> dict:
    return {
        "activity_vocab": {"A": 0, "B": 1},
        "resource_vocab": {},
        "test_dataset": [_mou_sample()],
    }


def _mou_sample() -> Data:
    return Data(
        y=torch.tensor([1]),
        target_label="B",
        candidate_allowed_target_mask=torch.tensor([[False, True]], dtype=torch.bool),
        candidate_ids=("node_A", "node_B"),
        candidate_labels=("A", "B"),
    )


def _write_mou_config(tmp_path, checkpoint, tracking: bool = False) -> str:
    path = tmp_path / "mou.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "data": {"log_path": "unused.xes", "dataset_label": "mou_smoke"},
                "experiment": {
                    "mode": "eval_topology_mask_uniform",
                    "uniform_mask_empty_mask_policy": "raise",
                    "uniform_mask_encoder_checkpoint": str(checkpoint),
                    "uniform_mask_evaluation_seed": 41,
                    "uniform_mask_mc_draws": 100,
                    "drift_window_size": 100,
                    "drift_window_sliding": 10,
                },
                "mapping": {"features": []},
                "model": {"type": "BaselineGCN", "hidden_dim": 8},
                "training": {"show_progress": False, "tqdm_disable": True},
                "tracking": {"enabled": tracking},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return str(path)
