from __future__ import annotations

import csv
from pathlib import Path

from tools import aggregate_article_run_metrics
from tools import export_article_figures


def test_mou_is_registered_in_aggregation_order():
    assert "MOU" in aggregate_article_run_metrics.MODEL_ORDER


def test_mou_is_registered_in_figure_order_and_style():
    assert "MOU" in export_article_figures.MODEL_ORDER
    assert "MOU" in export_article_figures.MODEL_COLORS
    assert "MOU" in export_article_figures.MODEL_LINESTYLES
    assert "MOU" in export_article_figures.MODEL_BAND_ALPHA


def test_aggregation_retains_drift_only_mou_rows(tmp_path: Path):
    drift_dir = tmp_path / "drift"
    drift_dir.mkdir(parents=True)
    metric_path = drift_dir / "drift_window_strict_macro_f1.csv"
    with metric_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "run_set",
                "paper_model",
                "model_type",
                "run_id",
                "experiment_id",
                "run_name",
                "preset_name",
                "seed",
                "metric",
                "step",
                "timestamp",
                "value",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "run_set": "drift",
                "paper_model": "MOU",
                "model_type": "MOU",
                "run_id": "mou-run",
                "experiment_id": "355",
                "run_name": "_CDLG-complex1_MOU-drift",
                "preset_name": "_CDLG-complex1_MOU-drift",
                "seed": "42",
                "metric": "drift_window_strict_macro_f1",
                "step": "0",
                "timestamp": "100",
                "value": "0.5",
            }
        )

    summary_rows, detail_rows = aggregate_article_run_metrics._aggregate_run_set(
        drift_dir,
        decimals=3,
        strategy="last",
    )

    assert summary_rows == [
        {
            "metric": "drift_window_strict_macro_f1",
            "GATv2": "",
            "GATv2+Mask": "",
            "LSTM": "",
            "EOPKG-WI": "",
            "EOPKG": "",
            "MOU": "0.500 ± 0.000",
        }
    ]
    mou_details = [
        row for row in detail_rows if row["metric"] == "drift_window_strict_macro_f1" and row["paper_model"] == "MOU"
    ]
    assert mou_details == [
        {
            "metric": "drift_window_strict_macro_f1",
            "paper_model": "MOU",
            "mean": "0.500",
            "std": "0.000",
            "n": 1,
            "aggregation_scope": "last",
            "run_ids": "mou-run",
        }
    ]


def test_aggregation_writes_summaries_to_output_dir_without_mutating_raw_dir(tmp_path: Path):
    input_dir = tmp_path / "raw" / "CDLG"
    output_dir = tmp_path / "aggregated" / "CDLG"
    learn_dir = input_dir / "learn"
    drift_dir = input_dir / "drift"
    learn_dir.mkdir(parents=True)
    drift_dir.mkdir(parents=True)

    for directory, metric, value in [
        (learn_dir, "strict_val_macro_f1", "0.7"),
        (drift_dir, "drift_window_strict_macro_f1", "0.4"),
    ]:
        with (directory / f"{metric}.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["paper_model", "run_id", "metric", "step", "timestamp", "value"],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "paper_model": "MOU",
                    "run_id": "run-1",
                    "metric": metric,
                    "step": "0",
                    "timestamp": "100",
                    "value": value,
                }
            )

    assert aggregate_article_run_metrics.main(
        ["--input-dir", str(input_dir), "--output-dir", str(output_dir), "--run-set", "all"]
    ) == 0

    assert (output_dir / "learn" / "summary_mean_std.csv").exists()
    assert (output_dir / "drift" / "summary_mean_std.csv").exists()
    assert not (input_dir / "learn" / "summary_mean_std.csv").exists()
    assert not (input_dir / "drift" / "summary_mean_std.csv").exists()
