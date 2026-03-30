from __future__ import annotations

from src.cli import (
    _build_mlflow_params,
    _derive_last_checkpoint_path,
    _flatten_config_dict,
    _resolve_resume_mlflow_run_id,
    _resolve_tracking_experiment_name,
)


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


def test_resolve_resume_mlflow_run_id_uses_checkpoint_embedded_run_id_for_train_resume():
    run_id = _resolve_resume_mlflow_run_id(
        mode="train",
        retrain=False,
        checkpoint_payload={"epoch": 5, "mlflow_run_id": "abc123"},
        checkpoint_epoch=5,
        target_epochs=20,
        experiment_name="bpm_prediction_mvp2_5",
        run_name="ResumeRun",
        tracking_uri="file:./mlruns",
    )
    assert run_id == "abc123"


def test_resolve_resume_mlflow_run_id_returns_none_for_eval_modes():
    run_id = _resolve_resume_mlflow_run_id(
        mode="eval_drift",
        retrain=False,
        checkpoint_payload={"epoch": 5, "mlflow_run_id": "abc123"},
        checkpoint_epoch=5,
        target_epochs=20,
        experiment_name="bpm_prediction_mvp2_5",
        run_name="EvalRun",
        tracking_uri="file:./mlruns",
    )
    assert run_id is None


def test_resolve_tracking_experiment_name_for_eval_mode_keeps_base_project():
    resolved = _resolve_tracking_experiment_name("bpm_prediction_mvp2_5", "eval_drift")
    assert resolved == "bpm_prediction_mvp2_5"


def test_resolve_tracking_experiment_name_for_train_keeps_base_project():
    resolved = _resolve_tracking_experiment_name("bpm_prediction_mvp2_5", "train")
    assert resolved == "bpm_prediction_mvp2_5"


def test_derive_last_checkpoint_path_from_best_checkpoint_name():
    resolved = _derive_last_checkpoint_path("checkpoints/run_abc_best.pth")
    assert resolved.endswith("run_abc_last.pth")
