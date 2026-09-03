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
