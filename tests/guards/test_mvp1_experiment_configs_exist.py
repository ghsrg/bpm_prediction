from __future__ import annotations

from pathlib import Path

import pytest


MVP1_EXPERIMENT_CONFIGS = [
    Path("configs/experiments/01_train_bpi2012.yaml"),
    Path("configs/experiments/01_eval_drift_bpi2012.yaml"),
    Path("configs/experiments/01_eval_cross_dataset_bpi2012.yaml"),
]


@pytest.mark.mvp1_regression
def test_mvp1_experiment_configs_are_present():
    missing = [str(path) for path in MVP1_EXPERIMENT_CONFIGS if not path.exists()]
    assert not missing, f"Missing MVP1 experiment config files: {missing}"

