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
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


MODEL_ORDER = ["GATv2", "GATv2+Mask", "LSTM", "EOPKG-WI", "EOPKG", "MOU"]
SKIP_FILES = {"run_manifest.csv", "missing_runs.csv"}
RS01_FUTURE_METRICS = (
    "strict_test_macro_f1",
    "strict_correct_rate",
    "parallelism_admissible_error_rate",
    "oos_error_rate",
)
RS01_FUTURE_QC_METRICS = (
    "audit_coverage",
    "excluded_count",
    "unresolved_mapping_count",
    "audit_contract_discrepancy_count",
)
RS01_COMMON_CONTRACT = "state_aware_activity_label_mask.v2"
RS01_FUTURE_MODEL_ORDER = ["GATv2", "GATv2+Mask", "LSTM", "EOPKG-WI", "EOPKG"]
MODEL_TYPE_TO_PAPER = {
    "BaselineGATv2": "GATv2",
    "BaselineGATv2Mask": "GATv2+Mask",
    "LSTM_Baseline": "LSTM",
    "EOPKGTopologyConditioned": "EOPKG",
    "EOPKGTopologyConditionedWI": "EOPKG-WI",
    "MOU": "MOU",
}
CDLG_PROCESS_PATTERN = re.compile(r"_CDLG-(simple|medium|complex)(\d+)_", re.IGNORECASE)


@dataclass
class Rs01FutureAggregation:
    per_seed_rows: list[dict[str, object]]
    details_rows: list[dict[str, object]]
    summary_rows: list[dict[str, object]]
    endpoint_details: dict[str, list[dict[str, object]]]
    exclusion_rows: list[dict[str, object]]
    validation_errors: list[str]


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


def _parse_versions(raw: str) -> tuple[str, ...]:
    values = [item.strip().lower() for item in str(raw or "").split(",") if item.strip()]
    if not values:
        return ()
    if any(not value.startswith("v") or not value[1:].isdigit() or int(value[1:]) < 1 for value in values):
        raise ValueError("--versions must be a comma-separated list such as v3,v4,v5")
    if len(set(values)) != len(values):
        raise ValueError("--versions must not contain duplicates")
    return tuple(sorted(values, key=lambda value: int(value[1:])))


def _version_scope_prefix(versions: tuple[str, ...]) -> str:
    return f"endpoint_{'_'.join(versions)}_"


def _future_version_label(versions: tuple[str, ...]) -> str:
    numbers = [int(version[1:]) for version in versions]
    if numbers == list(range(numbers[0], numbers[-1] + 1)):
        return f"{versions[0]}_{versions[-1]}"
    return "_".join(versions)


def _future_metric_path(run_dir: Path, version: str, metric: str) -> Path:
    return run_dir / f"endpoint_{version}_{metric}.csv"


def _future_dataset_identity(row: dict[str, str]) -> tuple[str, str]:
    configured_complexity = str(row.get("dataset_complexity", "")).strip().lower()
    normalized_complexity = {"medium": "middle"}.get(configured_complexity, configured_complexity)
    for field in ("preset_name", "run_name"):
        match = CDLG_PROCESS_PATTERN.search(str(row.get(field, "")))
        if match:
            family, number = match.groups()
            return {"medium": "middle"}.get(family.lower(), family.lower()), f"{family.lower()}{number}"
    return normalized_complexity, ""


def _future_row_key(row: dict[str, str]) -> tuple[str, str, str, str, str, str] | None:
    model_type = str(row.get("model_type", "")).strip()
    paper_model = MODEL_TYPE_TO_PAPER.get(model_type)
    seed = str(row.get("seed", "")).strip()
    run_id = str(row.get("run_id", "")).strip()
    if not paper_model or not seed or not run_id:
        return None
    dataset_complexity, process_id = _future_dataset_identity(row)
    return paper_model, model_type, dataset_complexity, process_id, seed, run_id


def _future_sort_key(row: dict[str, object]) -> tuple[int, str, int, str]:
    model = str(row["paper_model"])
    try:
        model_order = RS01_FUTURE_MODEL_ORDER.index(model)
    except ValueError:
        model_order = len(RS01_FUTURE_MODEL_ORDER)
    return model_order, str(row.get("dataset_complexity", "")), _safe_int(str(row["seed"])), str(row["metric"])


def _rs01_future_exclusion(
    paper_model: str,
    model_type: str,
    dataset_complexity: str,
    process_id: str,
    seed: str,
    reason: str,
    run_ids: Iterable[str],
    contract: str = "",
) -> dict[str, object]:
    return {
        "paper_model": paper_model,
        "model_type": model_type,
        "dataset_complexity": dataset_complexity,
        "process_id": process_id,
        "seed": seed,
        "reason": reason,
        "run_ids": ";".join(sorted({run_id for run_id in run_ids if run_id})),
        "metric_contract_id": contract,
    }


def _aggregate_rs01_future_versions(
    run_dir: Path,
    versions: tuple[str, ...],
    decimals: int,
) -> Rs01FutureAggregation:
    required_files = [
        _future_metric_path(run_dir, version, metric)
        for version in versions
        for metric in (*RS01_FUTURE_METRICS, *RS01_FUTURE_QC_METRICS)
    ]
    missing_files = [path.name for path in required_files if not path.exists()]
    if missing_files:
        return Rs01FutureAggregation(
            per_seed_rows=[], details_rows=[], summary_rows=[], endpoint_details={},
            exclusion_rows=[_rs01_future_exclusion("", "", "", "", "", "missing endpoint input files: " + ", ".join(missing_files), [])],
            validation_errors=["missing endpoint input files"],
        )

    manifest_path = run_dir / "run_manifest.csv"
    manifest_by_run: dict[str, list[dict[str, str]]] = defaultdict(list)
    if manifest_path.exists():
        for row in _read_rows(manifest_path):
            run_id = str(row.get("run_id", "")).strip()
            if run_id:
                manifest_by_run[run_id].append(row)

    observations: dict[tuple[str, str, str, str, str, str, str], tuple[float, str]] = {}
    run_ids_by_seed_endpoint: dict[tuple[str, str, str, str, str, str], set[str]] = defaultdict(set)
    reasons: dict[tuple[str, str, str, str, str], set[str]] = defaultdict(set)
    candidate_model_types: dict[tuple[str, str, str, str, str], str] = {}
    policy_exclusions: list[dict[str, object]] = []

    for version in versions:
        for metric in (*RS01_FUTURE_METRICS, *RS01_FUTURE_QC_METRICS):
            for row in _read_rows(_future_metric_path(run_dir, version, metric)):
                key = _future_row_key(row)
                if key is None:
                    reasons[("", "", "", "", "")].add(
                        f"invalid required identity fields in {version}:{metric}"
                    )
                    continue
                paper_model, model_type, dataset_complexity, process_id, seed, run_id = key
                seed_key = (paper_model, model_type, dataset_complexity, process_id, seed)
                if paper_model == "MOU":
                    policy_exclusions.append(_rs01_future_exclusion(
                        paper_model, model_type, dataset_complexity, process_id, seed, "excluded_model_policy", [run_id],
                    ))
                    continue
                candidate_model_types[seed_key] = model_type
                value = _safe_float(row.get("value", ""))
                if value is None:
                    reasons[seed_key].add(f"invalid numeric value for {version}:{metric}")
                    continue
                observation_key = (*seed_key, version, metric)
                run_ids_by_seed_endpoint[(*seed_key, version)].add(run_id)
                if observation_key in observations:
                    reasons[seed_key].add(f"duplicate value for {version}:{metric}")
                    continue
                observations[observation_key] = (value, run_id)

    for seed_key, model_type in candidate_model_types.items():
        paper_model, _model_type, _dataset_complexity, _process_id, _seed = seed_key
        for version in versions:
            endpoint_key = (*seed_key, version)
            run_ids = run_ids_by_seed_endpoint.get(endpoint_key, set())
            if len(run_ids) != 1:
                reasons[seed_key].add(f"missing or inconsistent run_id for endpoint {version}")
            for metric in RS01_FUTURE_METRICS:
                if (*seed_key, version, metric) not in observations:
                    reasons[seed_key].add(f"missing endpoint metric {version}:{metric}")
            for metric in RS01_FUTURE_QC_METRICS:
                observation = observations.get((*seed_key, version, metric))
                if observation is None:
                    reasons[seed_key].add(f"missing endpoint QC metric {version}:{metric}")
                    continue
                value, _run_id = observation
                if metric == "audit_coverage" and abs(value - 1.0) > 1.0e-9:
                    reasons[seed_key].add(f"audit_coverage != 1.0 for {version}")
                elif metric != "audit_coverage" and abs(value) > 1.0e-9:
                    reasons[seed_key].add(f"{metric} != 0 for {version}")

            rate_values = [
                observations.get((*seed_key, version, metric), (math.nan, ""))[0]
                for metric in (
                    "strict_correct_rate",
                    "parallelism_admissible_error_rate",
                    "oos_error_rate",
                )
            ]
            if not any(math.isnan(value) for value in rate_values) and abs(sum(rate_values) - 1.0) > 1.0e-6:
                reasons[seed_key].add(f"RS-01 partition does not sum to 1.0 for {version}")

    policy_ids: set[str] = set()
    contracts_by_seed: dict[tuple[str, str, str, str, str], str] = {}
    for seed_key, model_type in candidate_model_types.items():
        paper_model, _model_type, _dataset_complexity, _process_id, _seed = seed_key
        endpoint_run_ids = {
            run_id
            for version in versions
            for run_id in run_ids_by_seed_endpoint.get((*seed_key, version), set())
        }
        contracts: set[str] = set()
        for run_id in endpoint_run_ids:
            manifest_rows = manifest_by_run.get(run_id, [])
            if len(manifest_rows) != 1:
                reasons[seed_key].add(f"manifest entry missing or ambiguous for run_id={run_id}")
                continue
            manifest_row = manifest_rows[0]
            contract = str(manifest_row.get("rs01_metric_contract_id", "")).strip()
            policy_id = str(manifest_row.get("rs01_mask_policy_id", "")).strip()
            contracts.add(contract)
            if contract != RS01_COMMON_CONTRACT:
                reasons[seed_key].add(f"invalid metric contract for run_id={run_id}")
            if not policy_id:
                reasons[seed_key].add(f"missing reference policy for run_id={run_id}")
            else:
                policy_ids.add(policy_id)
        if len(contracts) == 1:
            contracts_by_seed[seed_key] = next(iter(contracts))
        elif contracts:
            reasons[seed_key].add("mixed metric contracts across endpoint runs")

    if len(policy_ids) > 1:
        for seed_key in candidate_model_types:
            reasons[seed_key].add("mixed reference policies across selected runs")

    valid_seed_keys = [seed_key for seed_key in candidate_model_types if not reasons[seed_key]]
    runs_by_model: dict[str, set[str]] = defaultdict(set)
    for paper_model, model_type, dataset_complexity, process_id, seed in valid_seed_keys:
        for version in versions:
            runs_by_model[paper_model].update(
                run_ids_by_seed_endpoint[(paper_model, model_type, dataset_complexity, process_id, seed, version)]
            )
    overlap = runs_by_model["EOPKG"] & runs_by_model["EOPKG-WI"]
    if overlap:
        for seed_key in valid_seed_keys:
            if seed_key[0] in {"EOPKG", "EOPKG-WI"}:
                reasons[seed_key].add("EOPKG and EOPKG-WI run_id overlap")
        valid_seed_keys = [seed_key for seed_key in candidate_model_types if not reasons[seed_key]]

    contract_id = RS01_COMMON_CONTRACT
    per_seed_rows: list[dict[str, object]] = []
    for paper_model, model_type, dataset_complexity, process_id, seed in valid_seed_keys:
        endpoint_run_ids = {
            version: next(iter(run_ids_by_seed_endpoint[(paper_model, model_type, dataset_complexity, process_id, seed, version)]))
            for version in versions
        }
        for metric in RS01_FUTURE_METRICS:
            endpoint_values = {
                version: observations[(paper_model, model_type, dataset_complexity, process_id, seed, version, metric)][0]
                for version in versions
            }
            per_seed_rows.append({
                "paper_model": paper_model,
                "model_type": model_type,
                "dataset_complexity": dataset_complexity,
                "process_id": process_id,
                "seed": seed,
                "metric": metric,
                **{f"endpoint_{version}_value": str(endpoint_values[version]) for version in versions},
                "per_seed_future_mean": str(sum(endpoint_values.values()) / len(versions)),
                **{f"run_id_{version}": endpoint_run_ids[version] for version in versions},
                "metric_contract_id": contract_id,
            })

    version_label = _future_version_label(versions)
    loan_aggregation_scope = f"per_seed_equal_mean_over_endpoints_{'_'.join(versions)}_then_across_seeds"
    cdlg_aggregation_scope = f"per_process_seed_equal_mean_over_endpoints_{'_'.join(versions)}_then_across_processes_and_seeds"
    endpoint_versions = ";".join(versions)
    details_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    endpoint_details: dict[str, list[dict[str, object]]] = {version: [] for version in versions}
    for metric in RS01_FUTURE_METRICS:
        complexities = sorted({str(row["dataset_complexity"]) for row in per_seed_rows}, key=lambda value: (value != "", value))
        for dataset_complexity in complexities:
            aggregation_scope = cdlg_aggregation_scope if dataset_complexity else loan_aggregation_scope
            summary_row: dict[str, object] = {"dataset_complexity": dataset_complexity, "metric": metric}
            for model in RS01_FUTURE_MODEL_ORDER:
                metric_rows = [
                    row for row in per_seed_rows
                    if row["paper_model"] == model and row["metric"] == metric
                    and row["dataset_complexity"] == dataset_complexity
                ]
                values = [_safe_float(str(row["per_seed_future_mean"])) for row in metric_rows]
                numeric_values = [value for value in values if value is not None]
                summary_row[model] = _format_mean_std(numeric_values, decimals)
                details_rows.append({
                    "metric": metric,
                    "paper_model": model,
                    "dataset_complexity": dataset_complexity,
                    "mean": _format_number(_mean_std(numeric_values)[0], decimals),
                    "std": _format_number(_mean_std(numeric_values)[1], decimals),
                    "n": len(numeric_values),
                    "aggregation_scope": aggregation_scope,
                    "endpoint_versions": endpoint_versions,
                    "included_seeds": ";".join(sorted({str(row["seed"]) for row in metric_rows})),
                    "included_process_ids": ";".join(sorted({str(row["process_id"]) for row in metric_rows if row["process_id"]})),
                    "run_ids": ";".join(sorted({str(row[f"run_id_{version}"]) for row in metric_rows for version in versions})),
                    "metric_contract_id": contract_id,
                })
                for version in versions:
                    endpoint_values = [_safe_float(str(row[f"endpoint_{version}_value"])) for row in metric_rows]
                    numeric_endpoint_values = [value for value in endpoint_values if value is not None]
                    endpoint_details[version].append({
                        "metric": metric,
                        "paper_model": model,
                        "dataset_complexity": dataset_complexity,
                        "mean": _format_number(_mean_std(numeric_endpoint_values)[0], decimals),
                        "std": _format_number(_mean_std(numeric_endpoint_values)[1], decimals),
                        "n": len(numeric_endpoint_values),
                        "aggregation_scope": f"endpoint_{version}_then_across_processes_and_seeds",
                        "endpoint_versions": version,
                        "included_seeds": ";".join(sorted({str(row["seed"]) for row in metric_rows})),
                        "included_process_ids": ";".join(sorted({str(row["process_id"]) for row in metric_rows if row["process_id"]})),
                        "run_ids": ";".join(sorted({str(row[f"run_id_{version}"]) for row in metric_rows})),
                        "metric_contract_id": contract_id,
                    })
            summary_rows.append(summary_row)

    exclusion_rows = list(policy_exclusions)
    for seed_key, seed_reasons in sorted(reasons.items()):
        if not seed_reasons:
            continue
        paper_model, model_type, dataset_complexity, process_id, seed = seed_key
        run_ids = {
            run_id
            for version in versions
            for run_id in run_ids_by_seed_endpoint.get((paper_model, model_type, dataset_complexity, process_id, seed, version), set())
        }
        exclusion_rows.append(_rs01_future_exclusion(
            paper_model, model_type, dataset_complexity, process_id, seed, "; ".join(sorted(seed_reasons)), run_ids,
            contracts_by_seed.get(seed_key, ""),
        ))

    validation_errors = [row["reason"] for row in exclusion_rows if row["reason"] != "excluded_model_policy"]
    return Rs01FutureAggregation(
        per_seed_rows=sorted(per_seed_rows, key=_future_sort_key),
        details_rows=sorted(details_rows, key=lambda row: (str(row["dataset_complexity"]), str(row["metric"]), RS01_FUTURE_MODEL_ORDER.index(str(row["paper_model"])))),
        summary_rows=sorted(summary_rows, key=lambda row: (str(row["dataset_complexity"]), str(row["metric"]))),
        endpoint_details={
            version: sorted(rows, key=lambda row: (str(row["dataset_complexity"]), str(row["metric"]), RS01_FUTURE_MODEL_ORDER.index(str(row["paper_model"]))))
            for version, rows in endpoint_details.items()
        },
        exclusion_rows=sorted(exclusion_rows, key=lambda row: (str(row["paper_model"]), str(row["dataset_complexity"]), str(row["process_id"]), _safe_int(str(row["seed"])), str(row["reason"]))),
        validation_errors=validation_errors,
    )


def _aggregate_run_set(
    run_dir: Path,
    decimals: int,
    strategy: str,
    audit_mode: str = "",
    versions: tuple[str, ...] = (),
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    summary_rows: list[dict[str, object]] = []
    metric_files = [
        path for path in sorted(run_dir.glob("*.csv"))
        if (
            path.name not in SKIP_FILES
            and not path.name.startswith("summary_")
            and not path.name.startswith("significance_")
        )
    ]
    version_scope_prefix = _version_scope_prefix(versions) if versions else ""
    if versions:
        metric_files = [path for path in metric_files if path.stem.startswith(version_scope_prefix)]
        if not metric_files:
            raise ValueError(
                f"No combined endpoint scope found for --versions {','.join(versions)}"
            )
    manifest = run_dir / "run_manifest.csv"
    if manifest.exists():
        manifest_rows = _read_rows(manifest)
        if versions:
            scoped_run_ids = {
                str(row.get("run_id", "")).strip()
                for metric_file in metric_files
                for row in _read_rows(metric_file)
                if str(row.get("run_id", "")).strip()
            }
            manifest_rows = [
                row for row in manifest_rows if str(row.get("run_id", "")).strip() in scoped_run_ids
            ]
        contracts = {row.get("rs01_metric_contract_id", "") for row in manifest_rows}
        if "state_aware_activity_label_mask.v2" in contracts:
            policies = {row.get("rs01_mask_policy_id", "") for row in manifest_rows}
            if len(contracts) != 1 or len(policies) != 1 or "" in policies:
                raise ValueError("Common safety aggregation requires one reference policy and contract")
    detail_rows: list[dict[str, object]] = []
    best_epoch_by_run = _best_epoch_by_run(run_dir) if strategy == "best_epoch" else {}

    for metric_file in metric_files:
        rows = _read_rows(metric_file)
        if not rows:
            continue
        metric_name = str(rows[0].get("metric", "")).strip() or metric_file.stem
        if version_scope_prefix:
            metric_name = metric_name.removeprefix(version_scope_prefix)
        latest = _last_points_by_run(rows, strategy=strategy, best_epoch_by_run=best_epoch_by_run)
        if audit_mode == "rs01" and metric_name == "partition_sum":
            for (_paper_model, run_id), row in latest.items():
                value = _safe_float(row.get("value", ""))
                if value is not None and abs(value - 1.0) > 1.0e-9:
                    raise ValueError(f"RS-01 partition_sum invalid for run_id={run_id}: {value}")
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
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for aggregated CSV files. Defaults to --input-dir for compatibility.",
    )
    parser.add_argument("--run-set", choices=["learn", "drift", "all"], default="all")
    parser.add_argument("--decimals", type=int, default=3)
    parser.add_argument("--audit-mode", choices=["", "rs01"], default="")
    parser.add_argument(
        "--versions",
        default="",
        help="Comma-separated unseen endpoint versions, for example v3,v4,v5.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(argv or sys.argv[1:]))
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir is not None else input_dir
    run_sets = ["learn", "drift"] if args.run_set == "all" else [args.run_set]
    try:
        versions = _parse_versions(args.versions)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if versions and args.run_set != "drift":
        print("--versions accepts only --run-set drift", file=sys.stderr)
        return 2
    if versions and args.audit_mode != "rs01":
        print("--versions requires --audit-mode rs01", file=sys.stderr)
        return 2

    for run_set in run_sets:
        input_run_dir = input_dir / run_set
        if args.audit_mode == "rs01" and run_set != "drift":
            print("--audit-mode rs01 accepts only drift input", file=sys.stderr)
            return 2
        if not input_run_dir.exists():
            print(f"Missing run-set directory: {input_run_dir}", file=sys.stderr)
            return 1
        output_run_dir = output_dir / run_set
        if versions:
            future = _aggregate_rs01_future_versions(input_run_dir, versions, args.decimals)
            version_label = _future_version_label(versions)
            exclusion_fields = [
                "paper_model", "model_type", "dataset_complexity", "process_id", "seed",
                "reason", "run_ids", "metric_contract_id",
            ]
            _write_csv(
                output_run_dir / f"summary_rs01_future_{version_label}_excluded_seeds.csv",
                exclusion_fields,
                future.exclusion_rows,
            )
            if future.validation_errors:
                print(
                    "RS-01 future-version aggregation rejected invalid or incomplete seeds; "
                    f"see {output_run_dir / f'summary_rs01_future_{version_label}_excluded_seeds.csv'}",
                    file=sys.stderr,
                )
                return 1
            per_seed_fields = [
                "paper_model", "model_type", "dataset_complexity", "process_id", "seed", "metric",
                *[f"endpoint_{version}_value" for version in versions],
                "per_seed_future_mean",
                *[f"run_id_{version}" for version in versions],
                "metric_contract_id",
            ]
            detail_fields = [
                "metric", "paper_model", "dataset_complexity", "mean", "std", "n", "aggregation_scope",
                "endpoint_versions", "included_seeds", "included_process_ids", "run_ids", "metric_contract_id",
            ]
            _write_csv(
                output_run_dir / f"summary_rs01_future_{version_label}_per_seed.csv",
                per_seed_fields,
                future.per_seed_rows,
            )
            _write_csv(
                output_run_dir / f"summary_rs01_future_{version_label}_mean_std_details.csv",
                detail_fields,
                future.details_rows,
            )
            _write_csv(
                output_run_dir / f"summary_rs01_future_{version_label}_mean_std.csv",
                ["dataset_complexity", "metric", *RS01_FUTURE_MODEL_ORDER],
                future.summary_rows,
            )
            for version, rows in future.endpoint_details.items():
                _write_csv(
                    output_run_dir / f"summary_rs01_endpoint_{version}_mean_std_details.csv",
                    detail_fields,
                    rows,
                )
            print(
                f"drift: metrics={len(future.summary_rows)} endpoint_versions={';'.join(versions)} "
                f"summary={output_run_dir / f'summary_rs01_future_{version_label}_mean_std.csv'}"
            )
            continue
        strategy = "best_epoch" if run_set == "learn" else "last"
        summary_name = "summary_best_mean_std.csv" if run_set == "learn" else "summary_last_mean_std.csv"
        details_name = (
            "summary_best_mean_std_details.csv"
            if run_set == "learn"
            else "summary_last_mean_std_details.csv"
        )
        try:
            summary_rows, detail_rows = _aggregate_run_set(
                input_run_dir,
                args.decimals,
                strategy,
                audit_mode=args.audit_mode,
                versions=versions,
            )
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        _write_csv(output_run_dir / "summary_mean_std.csv", ["metric", *MODEL_ORDER], summary_rows)
        _write_csv(
            output_run_dir / "summary_mean_std_details.csv",
            ["metric", "paper_model", "mean", "std", "n", "aggregation_scope", "run_ids"],
            detail_rows,
        )
        _write_csv(output_run_dir / summary_name, ["metric", *MODEL_ORDER], summary_rows)
        _write_csv(
            output_run_dir / details_name,
            ["metric", "paper_model", "mean", "std", "n", "aggregation_scope", "run_ids"],
            detail_rows,
        )
        if args.audit_mode == "rs01":
            _write_csv(
                output_run_dir / "summary_rs01_endpoint_details.csv",
                ["metric", "paper_model", "mean", "std", "n", "aggregation_scope", "run_ids"],
                detail_rows,
            )
        print(
            f"{run_set}: metrics={len(summary_rows)} "
            f"strategy={strategy} summary={output_run_dir / summary_name}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
