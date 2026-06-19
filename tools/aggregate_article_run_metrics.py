"""Create article-ready mean ± std summaries from raw MLflow metric exports.

Aggregation strategy:
- learn: prefer the metric point at each run's best_epoch; fall back to the
  last point when the metric is logged only once after best-checkpoint eval.
- drift: use the last point per run, matching final drift-window reporting.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable


MODEL_ORDER = ["GATv2", "GATv2+Mask", "LSTM", "EOPKG-WI", "EOPKG"]
SKIP_FILES = {"run_manifest.csv", "missing_runs.csv"}


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, math.sqrt(variance)


def _format_number(value: float, decimals: int) -> str:
    if math.isnan(value):
        return ""
    return f"{value:.{decimals}f}"


def _format_mean_std(values: list[float], decimals: int) -> str:
    if not values:
        return ""
    mean, std = _mean_std(values)
    return f"{_format_number(mean, decimals)} ± {_format_number(std, decimals)}"


def _last_points_by_run(
    rows: list[dict[str, str]],
    strategy: str,
    best_epoch_by_run: dict[str, int],
) -> dict[tuple[str, str], dict[str, str]]:
    if strategy == "best_epoch":
        return _best_or_last_points_by_run(rows, best_epoch_by_run)

    latest: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        paper_model = str(row.get("paper_model", "")).strip()
        run_id = str(row.get("run_id", "")).strip()
        if not paper_model or not run_id:
            continue
        row = dict(row)
        row["_aggregation_scope"] = "last"
        key = (paper_model, run_id)
        old = latest.get(key)
        if old is None or _row_order(row) >= _row_order(old):
            latest[key] = row
    return latest


def _best_or_last_points_by_run(
    rows: list[dict[str, str]],
    best_epoch_by_run: dict[str, int],
) -> dict[tuple[str, str], dict[str, str]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        paper_model = str(row.get("paper_model", "")).strip()
        run_id = str(row.get("run_id", "")).strip()
        if paper_model and run_id:
            grouped[(paper_model, run_id)].append(row)

    selected: dict[tuple[str, str], dict[str, str]] = {}
    for key, run_rows in grouped.items():
        _paper_model, run_id = key
        best_epoch = best_epoch_by_run.get(run_id)
        best_rows = [
            row for row in run_rows if best_epoch is not None and _safe_int(row.get("step", "")) == best_epoch
        ]
        if best_rows:
            chosen = max(best_rows, key=_row_order)
            scope = "best_epoch"
        else:
            chosen = max(run_rows, key=_row_order)
            scope = "fallback_last_no_best_step"
        chosen = dict(chosen)
        chosen["_aggregation_scope"] = scope
        selected[key] = chosen
    return selected


def _row_order(row: dict[str, str]) -> tuple[int, int]:
    return (_safe_int(row.get("step", "")), _safe_int(row.get("timestamp", "")))


def _safe_int(value: str | None) -> int:
    try:
        return int(str(value or "").strip())
    except ValueError:
        return -1


def _safe_float(value: str | None) -> float | None:
    try:
        return float(str(value or "").strip())
    except ValueError:
        return None


def _best_epoch_by_run(run_dir: Path) -> dict[str, int]:
    path = run_dir / "best_epoch.csv"
    if not path.exists():
        return {}
    latest = _last_points_by_run(_read_rows(path), strategy="last", best_epoch_by_run={})
    best: dict[str, int] = {}
    for (_paper_model, run_id), row in latest.items():
        value = _safe_float(row.get("value", ""))
        if value is not None:
            best[run_id] = int(round(value))
    return best


def _aggregate_run_set(
    run_dir: Path,
    decimals: int,
    strategy: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    summary_rows: list[dict[str, object]] = []
    detail_rows: list[dict[str, object]] = []
    best_epoch_by_run = _best_epoch_by_run(run_dir) if strategy == "best_epoch" else {}

    for metric_file in sorted(run_dir.glob("*.csv")):
        if (
            metric_file.name in SKIP_FILES
            or metric_file.name.startswith("summary_")
            or metric_file.name.startswith("significance_")
        ):
            continue
        rows = _read_rows(metric_file)
        if not rows:
            continue
        metric_name = str(rows[0].get("metric", "")).strip() or metric_file.stem
        latest = _last_points_by_run(rows, strategy=strategy, best_epoch_by_run=best_epoch_by_run)
        values_by_model: dict[str, list[float]] = defaultdict(list)
        run_ids_by_model: dict[str, list[str]] = defaultdict(list)
        scopes_by_model: dict[str, set[str]] = defaultdict(set)

        for (paper_model, run_id), row in latest.items():
            value = _safe_float(row.get("value", ""))
            if value is None:
                continue
            values_by_model[paper_model].append(value)
            run_ids_by_model[paper_model].append(run_id)
            scopes_by_model[paper_model].add(str(row.get("_aggregation_scope", "")))

        summary_row: dict[str, object] = {"metric": metric_name}
        for model in MODEL_ORDER:
            summary_row[model] = _format_mean_std(values_by_model.get(model, []), decimals)
        summary_rows.append(summary_row)

        for model in MODEL_ORDER:
            values = values_by_model.get(model, [])
            mean, std = _mean_std(values)
            detail_rows.append(
                {
                    "metric": metric_name,
                    "paper_model": model,
                    "mean": _format_number(mean, decimals),
                    "std": _format_number(std, decimals),
                    "n": len(values),
                    "aggregation_scope": ";".join(sorted(scopes_by_model.get(model, []))),
                    "run_ids": ";".join(sorted(run_ids_by_model.get(model, []))),
                }
            )

    summary_rows.sort(key=lambda row: str(row["metric"]))
    detail_rows.sort(key=lambda row: (str(row["metric"]), MODEL_ORDER.index(str(row["paper_model"]))))
    return summary_rows, detail_rows


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate article raw MLflow metric exports as mean ± std by model."
    )
    parser.add_argument("--input-dir", default="outputs/Export_metrics/article_run_metrics")
    parser.add_argument("--run-set", choices=["learn", "drift", "all"], default="all")
    parser.add_argument("--decimals", type=int, default=3)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(argv or sys.argv[1:]))
    input_dir = Path(args.input_dir)
    run_sets = ["learn", "drift"] if args.run_set == "all" else [args.run_set]

    for run_set in run_sets:
        run_dir = input_dir / run_set
        if not run_dir.exists():
            print(f"Missing run-set directory: {run_dir}", file=sys.stderr)
            return 1
        strategy = "best_epoch" if run_set == "learn" else "last"
        summary_name = "summary_best_mean_std.csv" if run_set == "learn" else "summary_last_mean_std.csv"
        details_name = (
            "summary_best_mean_std_details.csv"
            if run_set == "learn"
            else "summary_last_mean_std_details.csv"
        )
        summary_rows, detail_rows = _aggregate_run_set(run_dir, args.decimals, strategy)
        _write_csv(run_dir / "summary_mean_std.csv", ["metric", *MODEL_ORDER], summary_rows)
        _write_csv(
            run_dir / "summary_mean_std_details.csv",
            ["metric", "paper_model", "mean", "std", "n", "aggregation_scope", "run_ids"],
            detail_rows,
        )
        _write_csv(run_dir / summary_name, ["metric", *MODEL_ORDER], summary_rows)
        _write_csv(
            run_dir / details_name,
            ["metric", "paper_model", "mean", "std", "n", "aggregation_scope", "run_ids"],
            detail_rows,
        )
        print(
            f"{run_set}: metrics={len(summary_rows)} "
            f"strategy={strategy} summary={run_dir / summary_name}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
