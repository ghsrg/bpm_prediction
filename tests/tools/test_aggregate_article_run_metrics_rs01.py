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
