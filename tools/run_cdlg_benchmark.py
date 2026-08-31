"""Run an explicitly ordered CDLG Experiment UI preset queue."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from copy import deepcopy
from pathlib import Path
import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from typing import Any, Literal, Mapping

import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.infrastructure.config.yaml_loader import load_yaml_with_includes

PRESETS_PATH = ROOT_DIR / "configs" / "ui" / "experiment_ui_presets.json"
PLAN_PATH = ROOT_DIR / "configs" / "ui" / "cdlg_benchmark_plan.yaml"
OUTPUT_DIR = ROOT_DIR / "outputs" / "cdlg_benchmark"
PROGRESS_EVENT_PREFIX = "__BPM_PROGRESS__"


CDLG_PRESET_PATTERN = re.compile(
    r"^_CDLG-(?P<complexity>simple|medium|complex)(?P<case_index>\d+)_(?P<model_label>GATv2|EOPKG)(?P<drift>-drift)?$"
)

FORM_NAMES = (
    "input_data_form",
    "input_xes_form",
    "input_camunda_runtime_form",
    "input_camunda_mapping_form",
    "eopkg_backend_form",
    "eopkg_structure_form",
    "sync_stats_form",
    "model_form",
    "general_experiment_form",
    "general_training_form",
    "general_tracking_form",
)

RUN_STAGE_WEIGHTS = {
    "run.pipeline": 0.02,
    "prepare_data": 0.10,
    "prepare.read_events": 0.08,
    "prepare.feature_encoder": 0.07,
    "build_graph.train": 0.22,
    "build_graph.validation": 0.10,
    "build_graph.test": 0.08,
    "trainer.dataloaders": 0.03,
    "trainer.dry_run": 0.05,
    "train.epochs": 0.15,
    "train.batches": 0.04,
    "validation.batches": 0.03,
    "test.eval": 0.03,
    "test.batches": 0.02,
    "eval_drift.one_pass_inference": 0.04,
    "eval_drift.windows": 0.04,
}

STAGE_TITLES_UA = {
    "run.pipeline": "Пайплайн",
    "prepare_data": "Підготовка даних",
    "prepare.read_events": "Читання подій",
    "prepare.feature_encoder": "Побудова енкодера ознак",
    "build_graph.train": "Побудова train-графів",
    "build_graph.validation": "Побудова validation-графів",
    "build_graph.test": "Побудова test-графів",
    "trainer.dataloaders": "Підготовка DataLoader",
    "trainer.dry_run": "Dry run моделі",
    "train.epochs": "Епохи тренування",
    "train.batches": "Train batch-и",
    "validation.batches": "Validation batch-и",
    "test.eval": "Підсумкове тестування",
    "test.batches": "Test batch-и",
    "eval_drift.one_pass_inference": "One-pass inference для drift",
    "eval_drift.windows": "Оцінка drift-вікон",
}

VAR_PATHS = {
    "project": "experiment.project",
    "experiment_name": "experiment.name",
    "mode": "experiment.mode",
    "fraction": "experiment.fraction",
    "fraction_strategy": "experiment.fraction_strategy",
    "split_strategy": "experiment.split_strategy",
    "train_ratio": "experiment.train_ratio",
    "split_ratio": "experiment.split_ratio",
    "version_scope_policy": "experiment.version_scope_policy",
    "stats_time_policy": "experiment.stats_time_policy",
    "on_missing_asof_snapshot": "experiment.on_missing_asof_snapshot",
    "cache_policy": "experiment.cache_policy",
    "graph_dataset_cache_policy": "experiment.graph_dataset_cache_policy",
    "graph_dataset_cache_dir": "experiment.graph_dataset_cache_dir",
    "adapter": "mapping.adapter",
}


@dataclass(frozen=True)
class PlannedRun:
    preset_name: str
    complexity: str
    case_index: int
    model_label: str
    phase: Literal["train", "drift"]
    payload: dict[str, Any]


@dataclass(frozen=True)
class QueueResult:
    exit_code: int


def load_run_plan(plan_path: Path, presets: Mapping[str, Any], *, validate_cdlg_names: bool = True) -> list[PlannedRun]:
    raw = yaml.safe_load(plan_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Benchmark plan must be a YAML mapping: {plan_path}")
    entries = raw.get("runs")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Benchmark plan must contain a non-empty 'runs' list")
    return [_resolve_planned_run(entry, presets, validate_cdlg_names=validate_cdlg_names) for entry in entries]


def load_plan_preset_names(plan_path: Path) -> list[str]:
    raw = yaml.safe_load(plan_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Benchmark plan must be a YAML mapping: {plan_path}")
    entries = raw.get("runs")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Benchmark plan must contain a non-empty 'runs' list")
    names: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Each benchmark plan entry must be a mapping with a 'preset' key")
        preset_name = str(entry.get("preset", "")).strip()
        if not preset_name:
            raise ValueError("Each benchmark plan entry must define a non-empty 'preset'")
        names.append(preset_name)
    return names


def _resolve_planned_run(entry: Any, presets: Mapping[str, Any], *, validate_cdlg_names: bool) -> PlannedRun:
    if not isinstance(entry, dict):
        raise ValueError("Each benchmark plan entry must be a mapping with a 'preset' key")
    preset_name = str(entry.get("preset", "")).strip()
    if not preset_name:
        raise ValueError("Each benchmark plan entry must define a non-empty 'preset'")
    preset_entry = presets.get(preset_name)
    if not isinstance(preset_entry, dict):
        raise ValueError(f"Preset not found: {preset_name}")
    payload = preset_entry.get("payload", preset_entry)
    if not isinstance(payload, dict):
        raise ValueError(f"Preset payload must be a mapping: {preset_name}")

    match = CDLG_PRESET_PATTERN.fullmatch(preset_name)
    if validate_cdlg_names:
        if match is None:
            raise ValueError(f"Preset does not use the canonical CDLG name format: {preset_name}")
        groups = match.groupdict()
        return PlannedRun(
            preset_name=preset_name,
            complexity=groups["complexity"],
            case_index=int(groups["case_index"]),
            model_label=groups["model_label"],
            phase="drift" if groups["drift"] else "train",
            payload=dict(payload),
        )

    variables = _mapping(payload, "vars")
    mode = str(variables.get("mode", "")).strip().lower()
    model_label = _infer_model_label(preset_name, payload)
    return PlannedRun(
        preset_name=preset_name,
        complexity=match.group("complexity") if match else "",
        case_index=int(match.group("case_index")) if match else 0,
        model_label=model_label,
        phase="drift" if "drift" in mode or "drift" in preset_name.lower() else "train",
        payload=dict(payload),
    )


def _infer_model_label(preset_name: str, payload: Mapping[str, Any]) -> str:
    variables = _mapping(payload, "vars")
    for key in ("model_type", "model_label"):
        value = str(variables.get(key, "")).strip()
        if value:
            return value
    model_form = _mapping(payload, "model_form")
    for key in ("model.type", "model.model_label"):
        value = str(model_form.get(key, "")).strip()
        if value:
            return value
    for marker in ("MOU", "EOPKG", "GATv2", "GCN", "LSTM"):
        if marker.lower() in preset_name.lower():
            return marker
    return ""


def compose_preset_config(payload: Mapping[str, Any]) -> dict[str, Any]:
    variables = _mapping(payload, "vars")
    config_path = _required_text(variables, "config_path")
    cfg = deepcopy(load_yaml_with_includes(Path(config_path)))
    _apply_vars(cfg, variables)
    for form_name in FORM_NAMES:
        _apply_flat_form(cfg, _mapping(payload, form_name), base=cfg)
    _apply_yaml_blocks(cfg, payload, variables)
    return cfg


def build_run_command(
    mode: str,
    config_path: Path,
    *,
    sync_as_of: str = "",
    backfill_step: str = "",
    backfill_step_days: str = "",
    backfill_from: str = "",
    backfill_to: str = "",
    extra_args: str = "",
) -> list[str]:
    normalized_mode = str(mode).strip()
    command = [sys.executable, "main.py"]
    if normalized_mode == "sync-stats":
        command += ["sync-stats", "--config", str(config_path)]
        if sync_as_of.strip():
            command += ["--as-of", sync_as_of.strip()]
    elif normalized_mode == "sync-stats-backfill":
        command += ["sync-stats-backfill", "--config", str(config_path), "--step", backfill_step.strip()]
        if backfill_step_days.strip():
            command += ["--step-days", backfill_step_days.strip()]
        if backfill_from.strip():
            command += ["--from", backfill_from.strip()]
        if backfill_to.strip():
            command += ["--to", backfill_to.strip()]
    elif normalized_mode == "sync-topology":
        command += ["sync-topology", "--config", str(config_path)]
    else:
        command += ["--config", str(config_path)]
    if extra_args.strip():
        command.extend(shlex.split(extra_args))
    return command


def _mapping(payload: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key, {})
    return dict(value) if isinstance(value, Mapping) else {}


def _required_text(payload: Mapping[str, Any], key: str) -> str:
    value = str(payload.get(key, "")).strip()
    if not value:
        raise ValueError(f"Preset variable '{key}' must be non-empty")
    return value


def _apply_vars(cfg: dict[str, Any], variables: Mapping[str, Any]) -> None:
    for key, path in VAR_PATHS.items():
        if key not in variables:
            continue
        value = _parse_preset_value(variables[key])
        if key in {"fraction", "train_ratio"} and value != "":
            value = float(value)
        elif key == "split_ratio" and isinstance(value, str) and value.strip():
            value = _parse_preset_value(value)
        _deep_set(cfg, path, value)
    if "seed" in variables:
        _deep_set(cfg, "seed", int(float(str(variables["seed"]).strip() or "42")))


def _apply_flat_form(cfg: dict[str, Any], values: Mapping[str, Any], *, base: Mapping[str, Any]) -> None:
    for path, raw_value in values.items():
        if not isinstance(path, str) or not path.strip():
            continue
        value = _parse_preset_value(raw_value)
        if _is_blank(value) and not _deep_has(base, path):
            continue
        _deep_set(cfg, path, value)


def _apply_yaml_blocks(cfg: dict[str, Any], payload: Mapping[str, Any], variables: Mapping[str, Any]) -> None:
    features = _parse_preset_value(payload.get("features_text", ""))
    _deep_set(cfg, "mapping.features", features)
    if "features" in cfg:
        _deep_set(cfg, "features", features)
    _deep_set(cfg, "policies", _parse_preset_value(payload.get("policies_text", "")))
    graph_mapping = _parse_preset_value(payload.get("graph_mapping_text", ""))
    if not isinstance(graph_mapping, dict):
        graph_mapping = {}
    projection = graph_mapping.get("topology_projection")
    if not isinstance(projection, dict):
        projection = {}
    gateway_mode = str(variables.get("gateway_mode", "")).strip()
    if gateway_mode in {"preserve", "collapse_for_prediction"}:
        projection["gateway_mode"] = gateway_mode
    graph_mapping["topology_projection"] = projection
    _deep_set(cfg, "mapping.graph_feature_mapping", graph_mapping)


def _parse_preset_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return ""
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return text


def _deep_has(payload: Mapping[str, Any], dotted_path: str) -> bool:
    cursor: Any = payload
    for key in dotted_path.split("."):
        if not isinstance(cursor, Mapping) or key not in cursor:
            return False
        cursor = cursor[key]
    return True


def _deep_set(payload: dict[str, Any], dotted_path: str, value: Any) -> None:
    keys = [key for key in dotted_path.split(".") if key]
    if not keys:
        return
    cursor = payload
    for key in keys[:-1]:
        nested = cursor.get(key)
        if not isinstance(nested, dict):
            nested = {}
            cursor[key] = nested
        cursor = nested
    cursor[keys[-1]] = value


def _is_blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, tuple, set, dict)):
        return not value
    return False


@dataclass
class QueueEtaEstimator:
    durations_by_phase: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))

    def record(self, phase: str, seconds: float) -> None:
        if seconds > 0:
            self.durations_by_phase[str(phase)].append(float(seconds))

    def estimate(self, remaining_phases: list[str]) -> float | None:
        known = [duration for values in self.durations_by_phase.values() for duration in values]
        if not known:
            return None
        overall_average = sum(known) / len(known)
        total = 0.0
        for phase in remaining_phases:
            phase_durations = self.durations_by_phase.get(phase, [])
            total += (sum(phase_durations) / len(phase_durations)) if phase_durations else overall_average
        return total


@dataclass
class ProgressTracker:
    started_at: float
    stage_progress: dict[str, float] = field(default_factory=dict)
    stage_started_at: dict[str, float] = field(default_factory=dict)

    def consume(
        self,
        *,
        run: PlannedRun,
        queue_total: int,
        completed_count: int,
        event: Mapping[str, Any],
        now: float,
        queue_eta_seconds: float | None,
    ) -> str:
        stage = str(event.get("stage", "unknown")).strip() or "unknown"
        status = str(event.get("status", "update")).strip().lower() or "update"
        current = _as_float(event.get("current"))
        total = _as_float(event.get("total"))
        percent = _as_float(event.get("percent"))
        if percent is None:
            percent = (current / total * 100.0) if current is not None and total and total > 0 else 0.0
        if status == "done":
            percent = 100.0
        percent = max(0.0, min(100.0, percent))
        if status == "start" or stage not in self.stage_started_at:
            self.stage_started_at[stage] = now
        self.stage_progress[stage] = percent / 100.0

        stage_elapsed = max(0.0, now - self.stage_started_at[stage])
        stage_eta = _remaining_eta(stage_elapsed, percent)
        run_percent = self._overall_percent()
        run_eta = _remaining_eta(max(0.0, now - self.started_at), run_percent)
        position = completed_count + 1
        remaining = max(0, queue_total - position)
        progress = _progress_text(current, total, percent)
        message = str(event.get("message", "")).strip()
        stage_label = STAGE_TITLES_UA.get(stage, stage.replace("_", " "))
        suffix = f" | {message}" if message else ""
        run_line = (
            f"Run {position:02d}/{queue_total:02d} | {run.preset_name} | completed {completed_count} | remaining {remaining} | "
            f"{stage_label}{suffix} | {progress} | stage ETA {_format_duration(stage_eta)} | run ETA {_format_duration(run_eta)}"
        )
        queue_line = (
            f"Queue | completed {completed_count}/{queue_total} | remaining {remaining} | "
            f"estimated remaining {_format_duration(queue_eta_seconds)}"
        )
        return f"{run_line}\n{queue_line}"

    def _overall_percent(self) -> float:
        weighted = 0.0
        total_weight = 0.0
        for stage, progress in self.stage_progress.items():
            weight = RUN_STAGE_WEIGHTS.get(stage, 0.01)
            weighted += weight * max(0.0, min(1.0, progress))
            total_weight += weight
        return (weighted / total_weight * 100.0) if total_weight else 0.0


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _remaining_eta(elapsed: float, percent: float) -> float | None:
    if percent <= 0.0 or percent >= 100.0:
        return 0.0 if percent >= 100.0 else None
    return max(0.0, elapsed / (percent / 100.0) - elapsed)


def _progress_text(current: float | None, total: float | None, percent: float) -> str:
    if current is not None and total is not None and total > 0:
        return f"{_format_number(current)}/{_format_number(total)} ({percent:.1f}%)"
    return f"{percent:.1f}%"


def _format_number(value: float) -> str:
    return str(int(value)) if value.is_integer() else f"{value:.2f}"


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "--:--"
    total = max(0, int(seconds))
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def parse_progress_event(line: str) -> dict[str, Any] | None:
    if not line.startswith(PROGRESS_EVENT_PREFIX):
        return None
    try:
        payload = json.loads(line[len(PROGRESS_EVENT_PREFIX) :].strip())
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def execute_queue(runs: list[PlannedRun], *, output_dir: Path, process_factory=subprocess.Popen) -> QueueResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    estimator = QueueEtaEstimator()
    for index, run in enumerate(runs, start=1):
        print(f"Run {index:02d}/{len(runs):02d} | {run.preset_name} | completed {index - 1} | remaining {len(runs) - index}")
        started_at = time.time()
        config_path = _write_generated_config(output_dir, index, run)
        variables = _mapping(run.payload, "vars")
        command = build_run_command(
            str(variables.get("mode", "")),
            config_path,
            sync_as_of=str(variables.get("sync_as_of", "")),
            backfill_step=str(variables.get("backfill_step", "")),
            backfill_step_days=str(variables.get("backfill_step_days", "")),
            backfill_from=str(variables.get("backfill_from", "")),
            backfill_to=str(variables.get("backfill_to", "")),
            extra_args=str(variables.get("extra_args", "")),
        )
        log_path = output_dir / "logs" / f"{index:03d}_{_safe_filename(run.preset_name)}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        environment = os.environ.copy()
        environment["BPM_PROGRESS_EVENTS"] = "1"
        environment["PYTHONUNBUFFERED"] = "1"
        tracker = ProgressTracker(started_at=started_at)
        with log_path.open("w", encoding="utf-8") as log_file:
            process = process_factory(
                command,
                cwd=str(ROOT_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=environment,
            )
            for line in process.stdout or ():
                log_file.write(line)
                event = parse_progress_event(line)
                if event is not None:
                    print(
                        tracker.consume(
                            run=run,
                            queue_total=len(runs),
                            completed_count=index - 1,
                            event=event,
                            now=time.time(),
                            queue_eta_seconds=estimator.estimate([item.phase for item in runs[index:]]),
                        )
                    )
                elif "WARNING" in line.upper() or "ERROR" in line.upper() or line.startswith("Traceback"):
                    print(line.rstrip())
        returncode = int(process.wait())
        elapsed = time.time() - started_at
        status = "completed" if returncode == 0 else "failed"
        _append_manifest(output_dir, run, index, status, config_path, log_path, returncode, elapsed)
        if returncode:
            paired_name = f"{run.preset_name}-drift" if run.phase == "train" else ""
            for blocked_index, candidate in enumerate(runs[index:], start=index + 1):
                if candidate.preset_name == paired_name:
                    _append_manifest(output_dir, candidate, blocked_index, "blocked", None, None, None, None)
                    break
            return QueueResult(exit_code=returncode)
        estimator.record(run.phase, elapsed)
    return QueueResult(exit_code=0)


def _write_generated_config(output_dir: Path, index: int, run: PlannedRun) -> Path:
    path = output_dir / "configs" / f"{index:03d}_{_safe_filename(run.preset_name)}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(compose_preset_config(run.payload), allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


def _append_manifest(output_dir: Path, run: PlannedRun, index: int, status: str, config_path: Path | None, log_path: Path | None, returncode: int | None, elapsed: float | None) -> None:
    row = {"index": index, "preset_name": run.preset_name, "complexity": run.complexity, "case_index": run.case_index, "model_label": run.model_label, "phase": run.phase, "status": status, "config_path": str(config_path) if config_path else None, "log_path": str(log_path) if log_path else None, "returncode": returncode, "elapsed_seconds": elapsed}
    with (output_dir / "manifest.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=PLAN_PATH)
    parser.add_argument("--presets-path", type=Path, default=PRESETS_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    presets = json.loads(args.presets_path.read_text(encoding="utf-8"))
    runs = load_run_plan(args.plan, presets)
    for index, run in enumerate(runs, start=1):
        print(f"Run {index:02d}/{len(runs):02d} | {run.preset_name} | {run.complexity}{run.case_index} | {run.model_label} | {run.phase}")
    return 0 if args.dry_run else execute_queue(runs, output_dir=args.output_dir).exit_code


if __name__ == "__main__":
    raise SystemExit(main())
