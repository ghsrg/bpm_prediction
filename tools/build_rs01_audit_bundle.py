"""Build an isolated RS-01 audit evidence bundle from exported metric CSVs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path
from typing import Iterable


ENDPOINT_PROFILE_METRICS = [
    "strict_test_macro_f1",
    "parallelism_admissible_error_rate",
    "oos_error_rate",
]
OUTCOME_METRICS = [
    "strict_correct_count",
    "parallelism_admissible_error_count",
    "oos_error_count",
    "exact_outside_mask_rate",
    "common_oos_count",
    "common_oos_rate",
    "common_pred_in_mask_count",
    "common_pred_in_mask_rate",
    "common_target_in_mask_count",
    "common_target_in_mask_rate",
    "common_safety_denominator_count",
    "abstention_count",
    "abstention_rate",
    "audit_coverage",
    "prediction_coverage",
    "strict_correct_rate",
    "parallelism_admissible_error_rate",
    "oos_error_rate",
    "partition_sum",
    "valid_prediction_count",
    "audited_prefix_count",
    "unresolved_mapping_count",
    "exact_outside_mask_count",
    "excluded_count",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _safe_float(value: str | None) -> float | None:
    try:
        return float(str(value or "").strip())
    except ValueError:
        return None


def _row_order(row: dict[str, str]) -> tuple[int, int]:
    def _as_int(raw: str | None) -> int:
        try:
            return int(str(raw or "").strip())
        except ValueError:
            return -1

    return (_as_int(row.get("step")), _as_int(row.get("timestamp")))


def _endpoint_values(raw_metrics_dir: Path) -> dict[tuple[str, str], dict[str, str]]:
    latest: dict[tuple[str, str], dict[str, str]] = {}
    for path in sorted(raw_metrics_dir.glob("*.csv")):
        if path.name in {"run_manifest.csv", "missing_runs.csv"}:
            continue
        rows = _read_csv(path)
        for row in rows:
            run_id = str(row.get("run_id", "")).strip()
            metric = str(row.get("metric", "")).strip() or path.stem
            if not run_id or not metric:
                continue
            key = (run_id, metric)
            old = latest.get(key)
            if old is None or _row_order(row) >= _row_order(old):
                latest[key] = row
    return latest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _source_manifest(registry_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in registry_rows:
        run_id = row.get("audit_run_id", "")
        for field in ("checkpoint_path", "config_path", "data_path"):
            raw = str(row.get(field, "")).strip()
            if not raw:
                continue
            path = Path(raw)
            if not path.exists():
                raise ValueError(f"Missing source file for {field}: {path}")
            rows.append(
                {
                    "audit_run_id": run_id,
                    "source_field": field,
                    "path": str(path),
                    "sha256": _sha256(path),
                }
            )
    return rows


def build_bundle(*, audit_root: Path, raw_metrics_dir: Path, registry: Path) -> None:
    registry_rows = _read_csv(registry)
    registry_by_run = {str(row.get("audit_run_id", "")).strip(): row for row in registry_rows}
    endpoint = _endpoint_values(raw_metrics_dir)
    run_ids = sorted({run_id for run_id, _metric in endpoint})

    outcome_rows: list[dict[str, object]] = []
    profile_rows: list[dict[str, object]] = []
    errors: list[str] = []
    for run_id in run_ids:
        reg = registry_by_run.get(run_id, {})
        base = {
            "run_id": run_id,
            "historical_run_id": reg.get("historical_run_id", ""),
            "paper_model": reg.get("paper_model", ""),
            "metric_contract_id": reg.get("contract_id", ""),
            "mask_policy_id": reg.get("mask_policy_id", ""),
            "seed": reg.get("seed", ""),
        }
        partition_row = dict(base)
        for metric in OUTCOME_METRICS:
            value = _safe_float(endpoint.get((run_id, metric), {}).get("value"))
            partition_row[metric] = "" if value is None else value
        partition_sum = _safe_float(str(partition_row.get("partition_sum", "")))
        if partition_sum is not None and abs(partition_sum - 1.0) > 1.0e-9:
            errors.append(f"{run_id}: partition_sum={partition_sum}")
        outcome_rows.append(partition_row)
        if base["metric_contract_id"] == "state_aware_activity_label_mask.v2":
            common = _safe_float(str(partition_row.get("common_oos_rate", "")))
            outside = _safe_float(str(partition_row.get("exact_outside_mask_rate", "")))
            error = _safe_float(str(partition_row.get("oos_error_rate", "")))
            if any(value is None for value in (common, outside, error)):
                errors.append(f"{run_id}: missing common safety metrics")
            elif abs(common - outside - error) > 1e-6:
                errors.append(f"{run_id}: common OOS identity failed")
        for metric in ENDPOINT_PROFILE_METRICS:
            source = endpoint.get((run_id, metric), {})
            value = _safe_float(source.get("value"))
            profile_rows.append({**base, "metric": metric, "value": "" if value is None else value})
    if errors:
        raise ValueError("Invalid RS-01 partition: " + "; ".join(errors))

    identity_columns = ["run_id", "historical_run_id", "paper_model", "metric_contract_id", "mask_policy_id", "seed"]
    _write_csv(audit_root / "outcome_partition.csv", [*identity_columns, *OUTCOME_METRICS], outcome_rows)
    _write_csv(audit_root / "endpoint_error_profile.csv", [*identity_columns, "metric", "value"], profile_rows)
    _write_csv(audit_root / "source_manifest.csv", ["audit_run_id", "source_field", "path", "sha256"], _source_manifest(registry_rows))
    report = [
        "# RS-01 Metric Contract Audit",
        "",
        f"- audit_runs: {len(run_ids)}",
        f"- partition_gate: {'passed' if not errors else 'failed'}",
        "- limitation: mask-admissible alternative errors are not a causal BPMN parallelism label.",
        "",
    ]
    (audit_root / "metric_contract_audit.md").write_text("\n".join(report), encoding="utf-8")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an isolated RS-01 audit bundle.")
    parser.add_argument("--audit-root", required=True)
    parser.add_argument("--raw-metrics-dir", required=True)
    parser.add_argument("--aggregates-dir", default="")
    parser.add_argument("--registry", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(argv or sys.argv[1:]))
    try:
        build_bundle(
            audit_root=Path(args.audit_root),
            raw_metrics_dir=Path(args.raw_metrics_dir),
            registry=Path(args.registry),
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
