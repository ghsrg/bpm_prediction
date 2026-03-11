from __future__ import annotations

from pathlib import Path

import pytest

from src.cli import load_yaml_config


MVP1_CONFIGS = [
    "configs/experiments/01_train_bpi2012.yaml",
    "configs/experiments/01_eval_drift_bpi2012.yaml",
    "configs/experiments/01_eval_cross_dataset_bpi2012.yaml",
]


@pytest.mark.mvp1_regression
@pytest.mark.parametrize("config_path", MVP1_CONFIGS)
def test_mvp1_experiment_config_contract_is_valid(config_path: str):
    assert Path(config_path).exists(), f"Missing MVP1 config: {config_path}"

    cfg = load_yaml_config(config_path)
    assert isinstance(cfg, dict)

    for section in ("experiment", "data", "mapping", "model", "training"):
        assert section in cfg, f"Expected top-level section '{section}' in {config_path}"

    experiment = cfg["experiment"]
    assert experiment["mode"] in {"train", "eval_drift", "eval_cross_dataset"}
    assert experiment["split_strategy"] in {"temporal", "none", "time"}
    assert 0.0 < float(experiment["fraction"]) <= 1.0
    assert 0.0 <= float(experiment["train_ratio"]) <= 1.0
    assert len(experiment["split_ratio"]) == 3

    # Sprint-1 policy: legacy drift params removed from configs.
    assert "drift_window_stride" not in experiment
    assert "drift_window_overlap" not in experiment

    if experiment["mode"] == "eval_drift":
        assert int(experiment["drift_window_size"]) > 0
        assert int(experiment.get("drift_window_sliding", 0) or 0) >= 0

