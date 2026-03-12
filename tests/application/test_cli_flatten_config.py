from __future__ import annotations

from src.cli import _build_mlflow_params, _flatten_config_dict


def test_flatten_config_dict_produces_dotted_keys():
    config = {
        "experiment": {"fraction": 1.0, "split_ratio": [0.7, 0.2, 0.1]},
        "model": {"type": "EOPKGGATv2", "hidden_dim": 64},
        "data": {"log_path": "sample.xes"},
    }

    flat = _flatten_config_dict(config)
    assert flat["experiment.fraction"] == 1.0
    assert flat["experiment.split_ratio"] == "0.7,0.2,0.1"
    assert flat["model.type"] == "EOPKGGATv2"
    assert flat["data.log_path"] == "sample.xes"


def test_build_mlflow_params_filters_sections_and_truncates_long_values():
    config = {
        "seed": 42,
        "experiment": {"fraction": 1.0},
        "data": {"log_path": "sample.xes"},
        "model": {"type": "EOPKGGATv2"},
        "training": {"device": "cpu", "very_long": "x" * 600},
        "mapping": {"activity": "concept:name"},  # must be excluded
        "feature_configs": [{"k": "v"}],  # must be excluded
    }
    params = _build_mlflow_params(config, max_value_len=480)

    assert params["seed"] == 42
    assert params["experiment.fraction"] == 1.0
    assert params["model.type"] == "EOPKGGATv2"
    assert params["data.log_path"] == "sample.xes"
    assert "mapping.activity" not in params
    assert "feature_configs" not in params
    assert len(params["training.very_long"]) <= 495
    assert str(params["training.very_long"]).endswith("[truncated]")
