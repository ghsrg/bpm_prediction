from __future__ import annotations

from pathlib import Path
import shutil

from src.cli import prepare_data


def _copy_mock_exports(target_dir: Path) -> None:
    source_dir = Path("data/camunda_exports")
    target_dir.mkdir(parents=True, exist_ok=True)
    for file_name in (
        "mock_historic_activity_events.csv",
        "mock_historic_tasks.csv",
        "mock_identity_links.csv",
        "mock_execution_tree.csv",
        "mock_multi_instance_variables.csv",
        "mock_process_variables.csv",
        "mock_process_instance_links.csv",
    ):
        shutil.copy2(source_dir / file_name, target_dir / file_name)


def test_cli_prepare_data_camunda_adapter_smoke(tmp_path: Path):
    export_dir = tmp_path / "exports"
    _copy_mock_exports(export_dir)
    cfg = {
        "data": {
            "dataset_label": "camunda_procurement",
            "dataset_name": "procurement",
        },
        "experiment": {
            "mode": "train",
            "fraction": 1.0,
            "split_strategy": "temporal",
            "train_ratio": 1.0,
            "split_ratio": [1.0, 0.0, 0.0],
        },
        "mapping": {
            "adapter": "camunda",
            "camunda_adapter": {
                "process_name": "procurement",
                "version_key": "v1",
                "runtime": {
                    "runtime_source": "files",
                    "export_dir": str(export_dir),
                    "history_cleanup_aware": True,
                    "legacy_removal_time_policy": "treat_as_eternal",
                    "on_missing_removal_time": "auto_fallback",
                },
            },
            "features": [
                {
                    "name": "concept:name",
                    "source_key": "activity_def_id",
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
                    "fill_na": "UNKNOWN",
                    "encoding": ["embedding"],
                    "role": "resource",
                },
                {
                    "name": "duration",
                    "source": "event",
                    "dtype": "float",
                    "fill_na": 0.0,
                    "encoding": ["z-score"],
                },
            ],
        },
        "policies": {"activity_fallback_feature": "concept:name"},
        "training": {"show_progress": False, "tqdm_disable": True},
    }

    prepared = prepare_data(cfg)
    assert prepared["data_config"]["dataset_name"] == "procurement"
    assert len(prepared["train_dataset"]) > 0
    assert len(prepared["activity_vocab"]) >= 2
