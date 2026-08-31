"""Run an explicitly ordered Experiment UI preset queue without name validation."""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys
from typing import Any, Mapping

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tools import run_cdlg_benchmark as runner


def _load_presets(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Preset store must be a JSON mapping: {path}")
    return payload


def _dry_run(plan_path: Path, presets: Mapping[str, Any]) -> int:
    preset_names = runner.load_plan_preset_names(plan_path)
    missing = False
    for index, preset_name in enumerate(preset_names, start=1):
        if preset_name in presets:
            print(f"Run {index:02d}/{len(preset_names):02d} | {preset_name} | available")
        else:
            missing = True
            print(f"Run {index:02d}/{len(preset_names):02d} | Missing preset: {preset_name}")
    return 2 if missing else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=runner.PLAN_PATH)
    parser.add_argument("--presets-path", type=Path, default=runner.PRESETS_PATH)
    parser.add_argument("--output-dir", type=Path, default=runner.OUTPUT_DIR)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    presets = _load_presets(args.presets_path)
    if args.dry_run:
        return _dry_run(args.plan, presets)

    runs = runner.load_run_plan(args.plan, presets, validate_cdlg_names=False)
    for index, run in enumerate(runs, start=1):
        print(f"Run {index:02d}/{len(runs):02d} | {run.preset_name} | {run.model_label} | {run.phase}")
    return runner.execute_queue(runs, output_dir=args.output_dir).exit_code


if __name__ == "__main__":
    raise SystemExit(main())
