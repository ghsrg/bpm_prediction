"""Export article-source MLflow metric histories by explicit run ids.

The script intentionally exports raw metric histories only. It does not compute
paper summaries such as mean/std, so the generated CSV files can be used as
reproducibility source data for later article-table aggregation.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable


ARTICLE_LEARN_RUN_IDS = [
    "bbb9ea80d692474a87731fdf952ebe1b",
    "4de188c1615448e39860458609d7ba5b",
    "0827dcd6f1a34d1a8330115b73a5f288",
    "19d9078ee52942b08295cf257ee764d0",
    "6ede1f8547d64f9ca43d7c45efbd429f",
    "5e56c5c7531d4675b4c740f509ff678f",
    "ebdfb988ea654392a4a4fc270f9e7c71",
    "a9ec65c003b14176b0d853a15471bbbb",
    "b8f6130db97f4f28b27d4528497aa1f3",
    "5fcc89889b1d45c489329f5e3df9a5a2",
    "e76233f053ef480991b58d327df48a9c",
    "91bf625c79f04aeeb993f58ff30e43bb",
    "97a69231f63141a7bb7f6780b16c17fe",
    "de92aac60c8343dc8e495144fb1f50fb",
    "cbbdeac775de461c865ab13391a906dd",
    "a29ca3bbdf9a497d9fccaf8f2e3a6269",
    "65820d654ea74559b1b19d0faa56202d",
    "8d2a911a18bb4d61a536433690b744f0",
    "d0cd559af6bc4bcd986fb11b0ed56877",
    "88e10c5c5b45467e935548e0decdf824",
]


ARTICLE_DRIFT_RUN_IDS = [
    "210a1a668a5649d1b5ff4f4adc5fb93b",
    "e53ccb9930de4f7bb9b8aac4b4105174",
    "b03b8880277a467287a08bb2d2153b28",
    "e61540633e6547ed8c59bb28154d4917",
    "b739aca822a14a1cbd642cd4ac692d17",
    "bf6da114c8534a8eb85cf5a9d296cb1c",
    "2ab927c10aa14ff6b86c7fca2ad45124",
    "e03f160a63b34907b648e6b29b051f93",
    "d1400b7f5cee485485c75935ba0856fe",
    "6da86c9f8a8745ddb59c64a7e23ad129",
    "f6c4ddc23599407f882f9d237f19c6e0",
    "e35ad47b0b2c402aad5b87be01d9b6c8",
    "947216b4af334db88539b25462b18624",
    "c0522b44b1764d5086add47291774d05",
    "27d845ee73b648a3a49992c21b7d59c7",
    "431a79f4e8044d2987d5f97c90a221c6",
    "5f855ce1446645a4a4d52a5b3c12507f",
    "090ed1f9329c470dacabe8e87cad00e7",
    "edd97efded1f4ee08e676233e55461b9",
    "52976b78cd13459d8aeef2428f0736f8",
]


METRIC_COLUMNS = [
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
]

MANIFEST_COLUMNS = [
    "run_set",
    "paper_model",
    "model_type",
    "model_label",
    "topology_conditioning_mode",
    "impulse_activation_enabled",
    "run_id",
    "experiment_id",
    "run_name",
    "preset_name",
    "seed",
    "status",
    "start_time",
    "end_time",
    "metric_count",
]


def _read_runs_file(path: Path) -> list[str]:
    values: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        values.extend(part.strip() for part in line.split(",") if part.strip())
    return values


def _param(params: dict[str, str], *names: str) -> str:
    for name in names:
        if name in params and str(params[name]).strip():
            return str(params[name]).strip()
    for key, value in params.items():
        suffix = key.rsplit(".", 1)[-1]
        if suffix in names and str(value).strip():
            return str(value).strip()
    return ""


def _tag(tags: dict[str, str], *names: str) -> str:
    for name in names:
        if name in tags and str(tags[name]).strip():
            return str(tags[name]).strip()
    return ""


def _normalize_paper_model(params: dict[str, str], tags: dict[str, str]) -> str:
    model_type = _param(params, "model.type", "type", "model_type") or _tag(tags, "model_type")
    model_label = _param(params, "model.model_label", "model_label")
    preset_name = _param(params, "preset_name", "vars.preset_name")
    run_name = _tag(tags, "mlflow.runName")
    topology_mode = _param(params, "model.topology_conditioning_mode", "topology_conditioning_mode")
    impulse_enabled = _param(params, "model.impulse_activation_enabled", "impulse_activation_enabled")

    blob = " ".join([model_type, model_label, preset_name, run_name]).lower()
    if "lstm" in blob:
        return "LSTM"
    if "baselinegatv2" in blob or (model_type == "BaselineGATv2"):
        return "GATv2"
    if "eopkgtopologyconditionedwi" in blob or "_wi-" in blob or "without impulse" in blob:
        return "EOPKG-WI"
    if model_type == "EOPKGTopologyConditioned":
        if topology_mode == "static_candidates" or impulse_enabled.lower() == "false":
            return "EOPKG-WI"
        return "EOPKG"
    if "eopkg" in blob:
        return "EOPKG"
    if "gatv2" in blob:
        return "GATv2"
    return model_type or model_label or "UNKNOWN"


def _safe_metric_filename(metric_name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", metric_name.strip())
    safe = safe.strip("._")
    return safe or "metric"


def _derive_preset_name(run_name: str, seed: str) -> str:
    value = str(run_name or "").strip()
    if not value:
        return ""
    dataset_marker = "_loan_"
    if dataset_marker in value:
        value = value.split(dataset_marker, 1)[0]
    if seed:
        value = re.sub(rf"-{re.escape(str(seed))}$", "", value)
    return value


def _sort_metric_rows(rows: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    return sorted(
        rows,
        key=lambda row: (
            str(row["paper_model"]),
            str(row["run_id"]),
            int(row["step"]),
            int(row["timestamp"]),
        ),
    )


def _write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _export_run_set(
    *,
    client: object,
    run_set: str,
    run_ids: list[str],
    output_dir: Path,
) -> tuple[int, int, int]:
    metric_rows_by_name: dict[str, list[dict[str, object]]] = defaultdict(list)
    manifest_rows: list[dict[str, object]] = []
    missing_rows: list[dict[str, object]] = []

    for run_id in run_ids:
        try:
            run = client.get_run(run_id)  # type: ignore[attr-defined]
        except Exception as exc:  # pragma: no cover - MLflow exception type varies by version.
            missing_rows.append({"run_set": run_set, "run_id": run_id, "error": str(exc)})
            continue

        params = dict(run.data.params)
        tags = dict(run.data.tags)
        model_type = _param(params, "model.type", "type", "model_type") or _tag(tags, "model_type")
        paper_model = _normalize_paper_model(params, tags)
        run_name = _tag(tags, "mlflow.runName")
        seed = _param(params, "seed", "vars.seed")
        preset_name = _param(params, "preset_name", "vars.preset_name")
        if not preset_name:
            preset_name = _derive_preset_name(run_name, seed)

        metric_names = sorted(run.data.metrics.keys())
        manifest_rows.append(
            {
                "run_set": run_set,
                "paper_model": paper_model,
                "model_type": model_type,
                "model_label": _param(params, "model.model_label", "model_label"),
                "topology_conditioning_mode": _param(
                    params,
                    "model.topology_conditioning_mode",
                    "topology_conditioning_mode",
                ),
                "impulse_activation_enabled": _param(
                    params,
                    "model.impulse_activation_enabled",
                    "impulse_activation_enabled",
                ),
                "run_id": run_id,
                "experiment_id": run.info.experiment_id,
                "run_name": run_name,
                "preset_name": preset_name,
                "seed": seed,
                "status": str(run.info.status),
                "start_time": run.info.start_time,
                "end_time": run.info.end_time,
                "metric_count": len(metric_names),
            }
        )

        for metric_name in metric_names:
            history = client.get_metric_history(run_id, metric_name)  # type: ignore[attr-defined]
            for point in history:
                metric_rows_by_name[metric_name].append(
                    {
                        "run_set": run_set,
                        "paper_model": paper_model,
                        "model_type": model_type,
                        "run_id": run_id,
                        "experiment_id": run.info.experiment_id,
                        "run_name": run_name,
                        "preset_name": preset_name,
                        "seed": seed,
                        "metric": metric_name,
                        "step": int(point.step),
                        "timestamp": int(point.timestamp),
                        "value": repr(float(point.value)),
                    }
                )

    run_output_dir = output_dir / run_set
    _write_csv(run_output_dir / "run_manifest.csv", MANIFEST_COLUMNS, manifest_rows)
    _write_csv(run_output_dir / "missing_runs.csv", ["run_set", "run_id", "error"], missing_rows)

    for metric_name, rows in sorted(metric_rows_by_name.items()):
        file_name = _safe_metric_filename(metric_name) + ".csv"
        _write_csv(run_output_dir / file_name, METRIC_COLUMNS, _sort_metric_rows(rows))

    return len(manifest_rows), len(missing_rows), len(metric_rows_by_name)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export raw MLflow metric histories for the EOPKG structural-drift article."
    )
    parser.add_argument("--tracking-uri", default="file:./mlruns")
    parser.add_argument("--output-dir", default="Export_metrics/article_run_metrics")
    parser.add_argument("--run-set", choices=["learn", "drift", "all"], default="all")
    parser.add_argument(
        "--runs-file",
        type=Path,
        default=None,
        help="Optional newline/comma separated run ids. Requires --run-set learn or drift.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(argv or sys.argv[1:]))
    if args.runs_file is not None and args.run_set == "all":
        print("--runs-file requires --run-set learn or --run-set drift", file=sys.stderr)
        return 2

    try:
        import mlflow
        from mlflow.tracking import MlflowClient
    except Exception as exc:  # pragma: no cover - depends on runtime environment.
        print(f"Failed to import mlflow: {exc}", file=sys.stderr)
        return 2

    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient(tracking_uri=args.tracking_uri)
    output_dir = Path(args.output_dir)

    run_sets: list[tuple[str, list[str]]] = []
    if args.run_set in {"learn", "all"}:
        run_sets.append(("learn", ARTICLE_LEARN_RUN_IDS))
    if args.run_set in {"drift", "all"}:
        run_sets.append(("drift", ARTICLE_DRIFT_RUN_IDS))
    if args.runs_file is not None:
        run_sets = [(args.run_set, _read_runs_file(args.runs_file))]

    total_runs = 0
    total_missing = 0
    for run_set, run_ids in run_sets:
        exported, missing, metric_files = _export_run_set(
            client=client,
            run_set=run_set,
            run_ids=run_ids,
            output_dir=output_dir,
        )
        total_runs += exported
        total_missing += missing
        print(
            f"{run_set}: exported_runs={exported} missing_runs={missing} "
            f"metric_csv_files={metric_files}"
        )

    if total_missing:
        print(f"Completed with missing runs: {total_missing}", file=sys.stderr)
        return 1
    print(f"Completed: exported_runs={total_runs} output_dir={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
