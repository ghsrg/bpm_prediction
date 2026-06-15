"""Compute paired significance tests for article drift-resilience claims.

The script consumes raw metric histories exported by
``export_mlflow_run_metrics_for_article.py`` and evaluates Table 6.3 metrics.
For each metric it selects the last point per (model, seed), aligns models by
seed, and compares EOPKG against each baseline with an exact paired sign-flip
permutation test.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable


TARGET_MODEL = "EOPKG"
BASELINES = ["GATv2", "GATv2+Mask", "LSTM", "EOPKG-WI"]

METRIC_SPECS = {
    "drift_window_macro_f1": {
        "label": "Macro F1",
        "family": "primary",
        "better": "higher",
    },
    "drift_window_test_oos": {
        "label": "OOS rate",
        "family": "primary",
        "better": "lower",
    },
    "drift_window_test_ece": {
        "label": "Test ECE",
        "family": "primary",
        "better": "lower",
    },
    "drift_window_test_set_nll": {
        "label": "Test Set NLL",
        "family": "primary",
        "better": "lower",
    },
    "drift_window_strict_macro_f1": {
        "label": "Strict Macro F1",
        "family": "secondary",
        "better": "higher",
    },
    "drift_window_pred_in_mask_rate": {
        "label": "Prediction in Mask rate",
        "family": "secondary",
        "better": "higher",
    },
    "drift_window_strict_error_but_allowed_rate": {
        "label": "Strict Error but Allowed rate",
        "family": "error_profile",
        "better": "higher",
    },
}


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


def _row_order(row: dict[str, str]) -> tuple[int, int]:
    return (_safe_int(row.get("step")), _safe_int(row.get("timestamp")))


def _fmt(value: float | None, decimals: int = 6) -> str:
    if value is None or math.isnan(value):
        return ""
    return f"{value:.{decimals}f}"


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0 if values else math.nan
    mean = _mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def _median(values: list[float]) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _cohen_dz(diffs: list[float]) -> float:
    std = _std(diffs)
    if not diffs or std == 0:
        return math.inf if diffs and _mean(diffs) > 0 else math.nan
    return _mean(diffs) / std


def _exact_sign_flip_p_one_sided(diffs: list[float]) -> float:
    """One-sided exact paired permutation p-value for mean(diff) > 0."""
    if not diffs:
        return math.nan
    observed = _mean(diffs)
    total = 0
    extreme = 0
    eps = 1e-12
    for signs in itertools.product((-1.0, 1.0), repeat=len(diffs)):
        total += 1
        statistic = _mean([sign * diff for sign, diff in zip(signs, diffs)])
        if statistic >= observed - eps:
            extreme += 1
    return extreme / total


def _holm_adjust(rows: list[dict[str, object]], family: str) -> None:
    family_rows = [row for row in rows if row["family"] == family and row.get("p_exact")]
    ordered = sorted(family_rows, key=lambda row: float(row["p_exact"]))
    m = len(ordered)
    running = 0.0
    for idx, row in enumerate(ordered, start=1):
        adjusted = min(1.0, float(row["p_exact"]) * (m - idx + 1))
        running = max(running, adjusted)
        row["p_holm_family"] = _fmt(min(1.0, running), 6)


def _last_values_by_seed(metric_path: Path) -> dict[str, dict[str, dict[str, object]]]:
    latest: dict[tuple[str, str], dict[str, str]] = {}
    for row in _read_rows(metric_path):
        paper_model = str(row.get("paper_model", "")).strip()
        seed = str(row.get("seed", "")).strip()
        if not paper_model or not seed:
            continue
        key = (paper_model, seed)
        old = latest.get(key)
        if old is None or _row_order(row) >= _row_order(old):
            latest[key] = row

    values: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
    for (paper_model, seed), row in latest.items():
        value = _safe_float(row.get("value"))
        if value is None:
            continue
        values[seed][paper_model] = {
            "value": value,
            "run_id": str(row.get("run_id", "")).strip(),
            "step": str(row.get("step", "")).strip(),
        }
    return dict(values)


def _compute_metric(
    metric_name: str,
    metric_path: Path,
    decimals: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    spec = METRIC_SPECS[metric_name]
    by_seed = _last_values_by_seed(metric_path)
    paired_rows: list[dict[str, object]] = []
    result_rows: list[dict[str, object]] = []

    for baseline in BASELINES:
        seeds = sorted(
            seed
            for seed, values in by_seed.items()
            if TARGET_MODEL in values and baseline in values
        )
        diffs: list[float] = []
        target_values: list[float] = []
        baseline_values: list[float] = []
        for seed in seeds:
            target = float(by_seed[seed][TARGET_MODEL]["value"])
            base = float(by_seed[seed][baseline]["value"])
            diff = target - base if spec["better"] == "higher" else base - target
            diffs.append(diff)
            target_values.append(target)
            baseline_values.append(base)
            paired_rows.append(
                {
                    "metric": metric_name,
                    "metric_label": spec["label"],
                    "family": spec["family"],
                    "comparison": f"{TARGET_MODEL} vs {baseline}",
                    "seed": seed,
                    "baseline_model": baseline,
                    "baseline_value": _fmt(base, decimals),
                    "eopkg_value": _fmt(target, decimals),
                    "advantage_direction": spec["better"],
                    "paired_advantage": _fmt(diff, decimals),
                    "baseline_run_id": by_seed[seed][baseline]["run_id"],
                    "eopkg_run_id": by_seed[seed][TARGET_MODEL]["run_id"],
                }
            )

        p_exact = _exact_sign_flip_p_one_sided(diffs)
        result_rows.append(
            {
                "metric": metric_name,
                "metric_label": spec["label"],
                "family": spec["family"],
                "comparison": f"{TARGET_MODEL} vs {baseline}",
                "baseline_model": baseline,
                "advantage_direction": spec["better"],
                "n": len(diffs),
                "baseline_mean": _fmt(_mean(baseline_values), decimals),
                "eopkg_mean": _fmt(_mean(target_values), decimals),
                "mean_paired_advantage": _fmt(_mean(diffs), decimals),
                "median_paired_advantage": _fmt(_median(diffs), decimals),
                "std_paired_advantage": _fmt(_std(diffs), decimals),
                "cohen_dz": _fmt(_cohen_dz(diffs), decimals),
                "p_exact": _fmt(p_exact, 6),
                "p_holm_family": "",
            }
        )
    return result_rows, paired_rows


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute paired exact permutation tests for article Table 6.3 metrics."
    )
    parser.add_argument("--input-dir", default="Export_metrics/article_run_metrics/drift")
    parser.add_argument("--output-prefix", default="significance_table_6_3")
    parser.add_argument("--decimals", type=int, default=6)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(argv or sys.argv[1:]))
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Missing input directory: {input_dir}", file=sys.stderr)
        return 1

    results: list[dict[str, object]] = []
    paired: list[dict[str, object]] = []
    for metric_name in METRIC_SPECS:
        metric_path = input_dir / f"{metric_name}.csv"
        if not metric_path.exists():
            print(f"Missing metric file: {metric_path}", file=sys.stderr)
            return 1
        metric_results, metric_paired = _compute_metric(metric_name, metric_path, args.decimals)
        results.extend(metric_results)
        paired.extend(metric_paired)

    for family in sorted({str(row["family"]) for row in results}):
        _holm_adjust(results, family)

    result_path = input_dir / f"{args.output_prefix}.csv"
    paired_path = input_dir / f"{args.output_prefix}_paired_values.csv"
    _write_csv(
        result_path,
        [
            "metric",
            "metric_label",
            "family",
            "comparison",
            "baseline_model",
            "advantage_direction",
            "n",
            "baseline_mean",
            "eopkg_mean",
            "mean_paired_advantage",
            "median_paired_advantage",
            "std_paired_advantage",
            "cohen_dz",
            "p_exact",
            "p_holm_family",
        ],
        results,
    )
    _write_csv(
        paired_path,
        [
            "metric",
            "metric_label",
            "family",
            "comparison",
            "seed",
            "baseline_model",
            "baseline_value",
            "eopkg_value",
            "advantage_direction",
            "paired_advantage",
            "baseline_run_id",
            "eopkg_run_id",
        ],
        paired,
    )
    print(f"wrote {result_path}")
    print(f"wrote {paired_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
