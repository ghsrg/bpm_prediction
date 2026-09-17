from __future__ import annotations

import csv
import pytest
from pathlib import Path

from tools import aggregate_article_run_metrics as aggregate


def test_common_safety_aggregation_rejects_mixed_reference_policies(tmp_path):
    with (tmp_path / "run_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["run_id", "rs01_metric_contract_id", "rs01_mask_policy_id"])
        writer.writeheader()
        for idx, policy in enumerate(["policy-a", "policy-b"]):
            writer.writerow(dict(run_id=str(idx), rs01_metric_contract_id="state_aware_activity_label_mask.v2",
                                 rs01_mask_policy_id=policy))
    with pytest.raises(ValueError, match="reference"):
        aggregate._aggregate_run_set(tmp_path, 3, "last", audit_mode="rs01")


def _write_metric(path: Path, metric: str, rows: list[dict[str, object]]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    fieldnames = ["paper_model", "run_id", "metric", "step", "timestamp", "value"]
    with (path / f"{metric}.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_rs01_aggregation_uses_runs_not_windows_for_endpoint_details(tmp_path: Path):
    drift_dir = tmp_path / "raw" / "drift"
    _write_metric(
        drift_dir,
        "parallelism_admissible_error_rate",
        [
            {"paper_model": "GATv2", "run_id": "audit-gat-1", "metric": "parallelism_admissible_error_rate", "step": 0, "timestamp": 1, "value": 0.1},
            {"paper_model": "GATv2", "run_id": "audit-gat-1", "metric": "parallelism_admissible_error_rate", "step": 1, "timestamp": 2, "value": 0.2},
            {"paper_model": "GATv2", "run_id": "audit-gat-2", "metric": "parallelism_admissible_error_rate", "step": 1, "timestamp": 2, "value": 0.4},
        ],
    )
    _write_metric(
        drift_dir,
        "partition_sum",
        [
            {"paper_model": "GATv2", "run_id": "audit-gat-1", "metric": "partition_sum", "step": 1, "timestamp": 2, "value": 1.0},
            {"paper_model": "GATv2", "run_id": "audit-gat-2", "metric": "partition_sum", "step": 1, "timestamp": 2, "value": 1.0},
        ],
    )

    summary, details = aggregate._aggregate_run_set(drift_dir, decimals=3, strategy="last", audit_mode="rs01")
    row = next(
        item
        for item in details
        if item["metric"] == "parallelism_admissible_error_rate" and item["paper_model"] == "GATv2"
    )

    assert summary
    assert row["n"] == 2
    assert row["run_ids"] == "audit-gat-1;audit-gat-2"


def test_version_scope_aggregates_only_the_combined_unseen_endpoint(tmp_path: Path):
    drift_dir = tmp_path / "raw" / "drift"
    _write_metric(
        drift_dir,
        "endpoint_v3_v4_v5_strict_correct_rate",
        [
            {
                "paper_model": "GATv2",
                "run_id": "audit-gat-1",
                "metric": "endpoint_v3_v4_v5_strict_correct_rate",
                "step": 0,
                "timestamp": 1,
                "value": 0.7,
            },
        ],
    )
    _write_metric(
        drift_dir,
        "drift_window_strict_correct_rate",
        [
            {
                "paper_model": "GATv2",
                "run_id": "audit-gat-1",
                "metric": "drift_window_strict_correct_rate",
                "step": 10,
                "timestamp": 2,
                "value": 0.2,
            },
        ],
    )

    summary, details = aggregate._aggregate_run_set(
        drift_dir,
        decimals=3,
        strategy="last",
        audit_mode="rs01",
        versions=("v3", "v4", "v5"),
    )

    assert [row["metric"] for row in summary] == ["strict_correct_rate"]
    assert next(row for row in details if row["paper_model"] == "GATv2")["mean"] == "0.700"


def test_version_scope_requires_an_exact_combined_endpoint(tmp_path: Path):
    drift_dir = tmp_path / "raw" / "drift"
    _write_metric(
        drift_dir,
        "endpoint_v3_strict_correct_rate",
        [
            {
                "paper_model": "GATv2",
                "run_id": "audit-gat-1",
                "metric": "endpoint_v3_strict_correct_rate",
                "step": 0,
                "timestamp": 1,
                "value": 0.7,
            },
        ],
    )

    with pytest.raises(ValueError, match="combined endpoint scope"):
        aggregate._aggregate_run_set(
            drift_dir,
            decimals=3,
            strategy="last",
            audit_mode="rs01",
            versions=("v3", "v4"),
        )


def _write_future_endpoint_metric(
    drift_dir: Path,
    endpoint: str,
    metric: str,
    rows: list[dict[str, object]],
) -> None:
    drift_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "paper_model", "model_type", "run_id", "run_name", "preset_name",
        "dataset_complexity", "seed", "metric", "step", "timestamp", "value",
    ]
    with (drift_dir / f"endpoint_{endpoint}_{metric}.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _future_row(model_type: str, seed: int, endpoint: str, metric: str, value: float) -> dict[str, object]:
    return {
        "paper_model": "incorrect-fallback",
        "model_type": model_type,
        "run_id": f"{model_type}-{seed}",
        "seed": seed,
        "metric": f"endpoint_{endpoint}_{metric}",
        "step": 0,
        "timestamp": 1,
        "value": value,
    }


def _write_future_manifest(drift_dir: Path, rows: list[dict[str, object]]) -> None:
    fields = ["run_id", "rs01_metric_contract_id", "rs01_mask_policy_id"]
    with (drift_dir / "run_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_valid_future_fixture(drift_dir: Path, *, omit: tuple[int, str, str] | None = None) -> None:
    values = {
        13: {"v3": 0.3, "v4": 0.6, "v5": 0.9},
        42: {"v3": 0.6, "v4": 0.9, "v5": 0.3},
    }
    metrics = {
        "strict_test_macro_f1": values,
        "strict_correct_rate": values,
        "parallelism_admissible_error_rate": {seed: {version: 1.0 - value for version, value in per_version.items()} for seed, per_version in values.items()},
        "oos_error_rate": {seed: {version: 0.0 for version in per_version} for seed, per_version in values.items()},
    }
    for endpoint in ("v3", "v4", "v5"):
        for metric, per_seed in metrics.items():
            rows = [
                _future_row("BaselineGATv2", seed, endpoint, metric, value)
                for seed, per_version in per_seed.items()
                for version, value in per_version.items()
                if version == endpoint and omit != (seed, endpoint, metric)
            ]
            _write_future_endpoint_metric(drift_dir, endpoint, metric, rows)
        for metric, value in {
            "audit_coverage": 1.0,
            "excluded_count": 0.0,
            "unresolved_mapping_count": 0.0,
            "audit_contract_discrepancy_count": 0.0,
        }.items():
            _write_future_endpoint_metric(
                drift_dir,
                endpoint,
                metric,
                [_future_row("BaselineGATv2", seed, endpoint, metric, value) for seed in values],
            )
    _write_future_manifest(
        drift_dir,
        [
            {
                "run_id": f"BaselineGATv2-{seed}",
                "rs01_metric_contract_id": "state_aware_activity_label_mask.v2",
                "rs01_mask_policy_id": "policy-1",
            }
            for seed in values
        ],
    )


def test_versions_builds_equal_per_seed_endpoint_means_and_required_reports(tmp_path: Path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    _write_valid_future_fixture(input_dir / "drift")

    exit_code = aggregate.main([
        "--input-dir", str(input_dir),
        "--output-dir", str(output_dir),
        "--run-set", "drift",
        "--audit-mode", "rs01",
        "--versions", "v3,v4,v5",
    ])

    assert exit_code == 0
    per_seed = list(csv.DictReader((output_dir / "drift" / "summary_rs01_future_v3_v5_per_seed.csv").open()))
    f1_rows = [row for row in per_seed if row["metric"] == "strict_test_macro_f1"]
    assert {row["paper_model"] for row in f1_rows} == {"GATv2"}
    assert [row["per_seed_future_mean"] for row in f1_rows] == ["0.6", "0.6"]
    details = list(csv.DictReader((output_dir / "drift" / "summary_rs01_future_v3_v5_mean_std_details.csv").open()))
    assert next(row for row in details if row["metric"] == "strict_test_macro_f1")["aggregation_scope"] == (
        "per_seed_equal_mean_over_endpoints_v3_v4_v5_then_across_seeds"
    )
    assert (output_dir / "drift" / "summary_rs01_endpoint_v3_mean_std_details.csv").exists()
    assert (output_dir / "drift" / "summary_rs01_endpoint_v4_mean_std_details.csv").exists()
    assert (output_dir / "drift" / "summary_rs01_endpoint_v5_mean_std_details.csv").exists()


def test_versions_fails_and_reports_a_seed_with_a_missing_endpoint_metric(tmp_path: Path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    _write_valid_future_fixture(input_dir / "drift", omit=(42, "v5", "strict_correct_rate"))

    exit_code = aggregate.main([
        "--input-dir", str(input_dir),
        "--output-dir", str(output_dir),
        "--run-set", "drift",
        "--audit-mode", "rs01",
        "--versions", "v3,v4,v5",
    ])

    assert exit_code == 1
    exclusions = list(csv.DictReader((output_dir / "drift" / "summary_rs01_future_v3_v5_excluded_seeds.csv").open()))
    assert any(row["seed"] == "42" and "missing endpoint metric" in row["reason"] for row in exclusions)


def test_versions_duplicate_report_keeps_both_source_run_ids(tmp_path: Path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    drift_dir = input_dir / "drift"
    _write_valid_future_fixture(drift_dir)
    duplicate_path = drift_dir / "endpoint_v3_strict_correct_rate.csv"
    rows = list(csv.DictReader(duplicate_path.open()))
    duplicate = dict(rows[0])
    duplicate["run_id"] = "duplicate-run-id"
    rows.append(duplicate)
    with duplicate_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    exit_code = aggregate.main([
        "--input-dir", str(input_dir),
        "--output-dir", str(output_dir),
        "--run-set", "drift",
        "--audit-mode", "rs01",
        "--versions", "v3,v4,v5",
    ])

    assert exit_code == 1
    exclusions = list(csv.DictReader((output_dir / "drift" / "summary_rs01_future_v3_v5_excluded_seeds.csv").open()))
    row = next(row for row in exclusions if row["seed"] == "13")
    assert row["run_ids"] == "BaselineGATv2-13;duplicate-run-id"


def test_versions_fails_when_an_endpoint_qc_value_is_invalid(tmp_path: Path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    drift_dir = input_dir / "drift"
    _write_valid_future_fixture(drift_dir)
    qc_path = drift_dir / "endpoint_v4_audit_coverage.csv"
    rows = list(csv.DictReader(qc_path.open()))
    rows[0]["value"] = "0.99"
    with qc_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    exit_code = aggregate.main([
        "--input-dir", str(input_dir),
        "--output-dir", str(output_dir),
        "--run-set", "drift",
        "--audit-mode", "rs01",
        "--versions", "v3,v4,v5",
    ])

    assert exit_code == 1
    exclusions = list(csv.DictReader((output_dir / "drift" / "summary_rs01_future_v3_v5_excluded_seeds.csv").open()))
    assert any("audit_coverage != 1.0 for v4" in row["reason"] for row in exclusions)


def test_versions_groups_cdlg_processes_by_derived_dataset_complexity(tmp_path: Path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    drift_dir = input_dir / "drift"
    processes = ("simple1", "simple2")
    values = {"simple1": 0.4, "simple2": 0.8}

    for endpoint in ("v3", "v4", "v5"):
        for metric in (*aggregate.RS01_FUTURE_METRICS, *aggregate.RS01_FUTURE_QC_METRICS):
            rows = []
            for process in processes:
                row = _future_row("BaselineGATv2", 42, endpoint, metric, 0.0)
                row["run_id"] = f"{process}-{endpoint}-run"
                row["run_name"] = f"_CDLG-{process}_GATv2-drift-42-hash_cdlg-{process}_GATv2"
                row["preset_name"] = f"_CDLG-{process}_GATv2-drift"
                if metric == "strict_test_macro_f1" or metric == "strict_correct_rate":
                    row["value"] = values[process]
                elif metric == "parallelism_admissible_error_rate":
                    row["value"] = 1.0 - values[process]
                elif metric == "audit_coverage":
                    row["value"] = 1.0
                rows.append(row)
            _write_future_endpoint_metric(drift_dir, endpoint, metric, rows)

    _write_future_manifest(
        drift_dir,
        [
            {
                "run_id": f"{process}-{endpoint}-run",
                "rs01_metric_contract_id": "state_aware_activity_label_mask.v2",
                "rs01_mask_policy_id": "policy-1",
            }
            for process in processes
            for endpoint in ("v3", "v4", "v5")
        ],
    )

    exit_code = aggregate.main([
        "--input-dir", str(input_dir),
        "--output-dir", str(output_dir),
        "--run-set", "drift",
        "--audit-mode", "rs01",
        "--versions", "v3,v4,v5",
    ])

    assert exit_code == 0
    per_seed = list(csv.DictReader((output_dir / "drift" / "summary_rs01_future_v3_v5_per_seed.csv").open()))
    f1_rows = [row for row in per_seed if row["metric"] == "strict_test_macro_f1"]
    assert {row["dataset_complexity"] for row in f1_rows} == {"simple"}
    assert {row["process_id"] for row in f1_rows} == {"simple1", "simple2"}
    details = list(csv.DictReader((output_dir / "drift" / "summary_rs01_future_v3_v5_mean_std_details.csv").open()))
    detail = next(row for row in details if row["metric"] == "strict_test_macro_f1")
    assert detail["dataset_complexity"] == "simple"
    assert detail["n"] == "2"
