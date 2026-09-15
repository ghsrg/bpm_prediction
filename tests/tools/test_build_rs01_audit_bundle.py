from __future__ import annotations

import csv
from pathlib import Path

import pytest

from tools import build_rs01_audit_bundle as bundle


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _metric(path: Path, metric: str, value: float) -> None:
    _write_csv(
        path / f"{metric}.csv",
        [
            "run_set",
            "paper_model",
            "model_type",
            "run_id",
            "experiment_id",
            "run_name",
            "preset_name",
            "dataset_complexity",
            "seed",
            "metric",
            "step",
            "timestamp",
            "value",
        ],
        [
            {
                "run_set": "drift",
                "paper_model": "MOU",
                "model_type": "MOU",
                "run_id": "audit-run",
                "experiment_id": "355",
                "run_name": "audit",
                "preset_name": "preset",
                "dataset_complexity": "complex",
                "seed": "42",
                "metric": metric,
                "step": 0,
                "timestamp": 100,
                "value": value,
            }
        ],
    )


def test_build_rs01_audit_bundle_writes_partition_profile_manifest_and_report(tmp_path: Path):
    audit_root = tmp_path / "article_audits" / "rs01_demo"
    raw_dir = audit_root / "article_run_metrics" / "loan" / "drift"
    registry = audit_root / "audit_run_registry.csv"
    source_file = tmp_path / "config.yaml"
    source_file.write_text("experiment: {}\n", encoding="utf-8")
    _write_csv(
        registry,
        [
            "historical_run_id",
            "audit_run_id",
            "experiment_id",
            "paper_model",
            "seed",
            "checkpoint_sha256",
            "config_sha256",
            "data_sha256",
            "code_revision",
            "contract_id",
            "status",
            "config_path",
        ],
        [
            {
                "historical_run_id": "old-run",
                "audit_run_id": "audit-run",
                "experiment_id": "355",
                "paper_model": "MOU",
                "seed": "42",
                "checkpoint_sha256": "",
                "config_sha256": "",
                "data_sha256": "",
                "code_revision": "abc",
                "contract_id": "mou_native_candidate_label_mask.v1",
                "status": "EXPORTED",
                "config_path": str(source_file),
            }
        ],
    )
    for metric, value in {
        "strict_test_macro_f1": 0.5,
        "parallelism_admissible_error_rate": 0.25,
        "oos_error_rate": 0.10,
        "strict_correct_rate": 0.65,
        "partition_sum": 1.0,
        "valid_prediction_count": 100.0,
        "audited_prefix_count": 100.0,
        "unresolved_mapping_count": 0.0,
        "excluded_count": 0.0,
    }.items():
        _metric(raw_dir, metric, value)

    rc = bundle.main(
        [
            "--audit-root",
            str(audit_root),
            "--raw-metrics-dir",
            str(raw_dir),
            "--registry",
            str(registry),
        ]
    )

    assert rc == 0
    assert (audit_root / "outcome_partition.csv").exists()
    outcome_rows = list(csv.DictReader((audit_root / "outcome_partition.csv").open("r", encoding="utf-8")))
    assert outcome_rows[0]["unresolved_mapping_count"] == "0.0"
    assert outcome_rows[0]["excluded_count"] == "0.0"
    assert (audit_root / "endpoint_error_profile.csv").exists()
    assert (audit_root / "source_manifest.csv").exists()
    assert (audit_root / "metric_contract_audit.md").exists()


def test_build_rs01_audit_bundle_fails_on_invalid_partition_sum(tmp_path: Path):
    audit_root = tmp_path / "article_audits" / "rs01_bad"
    raw_dir = audit_root / "article_run_metrics" / "loan" / "drift"
    registry = audit_root / "audit_run_registry.csv"
    _write_csv(
        registry,
        ["audit_run_id", "paper_model", "contract_id", "status"],
        [{"audit_run_id": "audit-run", "paper_model": "MOU", "contract_id": "contract", "status": "EXPORTED"}],
    )
    _metric(raw_dir, "partition_sum", 0.9)

    assert bundle.main(["--audit-root", str(audit_root), "--raw-metrics-dir", str(raw_dir), "--registry", str(registry)]) == 1


def test_build_common_mask_bundle_exports_common_safety_identity(tmp_path: Path):
    audit_root = tmp_path / "article_audits" / "rs01_common"
    raw_dir = audit_root / "raw" / "drift"
    registry = audit_root / "audit_run_registry.csv"
    _write_csv(registry, ["audit_run_id", "paper_model", "contract_id", "status"], [{
        "audit_run_id": "audit-run", "paper_model": "GATv2+Mask",
        "contract_id": "state_aware_activity_label_mask.v2", "status": "EXPORTED",
    }])
    values = {
        "partition_sum": 1.0, "strict_correct_rate": 0.7,
        "parallelism_admissible_error_rate": 0.2, "oos_error_rate": 0.1,
        "exact_outside_mask_rate": 0.05, "common_oos_rate": 0.15,
        "common_oos_count": 15, "common_pred_in_mask_count": 85,
        "common_pred_in_mask_rate": 0.85, "common_target_in_mask_count": 90,
        "common_target_in_mask_rate": 0.9, "common_safety_denominator_count": 100,
        "valid_prediction_count": 100, "audited_prefix_count": 101,
        "excluded_count": 1, "unresolved_mapping_count": 1,
    }
    for metric, value in values.items():
        _metric(raw_dir, metric, value)
    assert bundle.main(["--audit-root", str(audit_root), "--raw-metrics-dir", str(raw_dir),
                        "--registry", str(registry)]) == 0
    row = next(csv.DictReader((audit_root / "outcome_partition.csv").open(encoding="utf-8")))
    assert row["common_oos_rate"] == "0.15"
    assert row["common_pred_in_mask_rate"] == "0.85"
    assert row["common_target_in_mask_rate"] == "0.9"
