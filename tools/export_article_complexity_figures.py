"""Export article figures split into dataset-complexity panels.

The tool mirrors the article figure definitions and visual styles from
``export_article_figures.py``. It performs hierarchical aggregation: seeds are
averaged within each generated dataset first, then generated datasets are
aggregated within ``simple``, ``middle``, and ``complex`` panels.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

try:
    from tools import export_article_figures as base
except ModuleNotFoundError:  # Direct ``python tools/<script>.py`` execution.
    import export_article_figures as base  # type: ignore[no-redef]


COMPLEXITIES = ("simple", "middle", "complex")
MODEL_ORDER = ["GATv2", "GATv2+Mask", "EOPKG-WI", "EOPKG", "MOU"]
MODEL_COLORS = {model: base.MODEL_COLORS[model] for model in MODEL_ORDER}
MODEL_LINESTYLES = {model: base.MODEL_LINESTYLES[model] for model in MODEL_ORDER}
MODEL_BAND_ALPHA = {model: base.MODEL_BAND_ALPHA[model] for model in MODEL_ORDER}


def _complexity_specs() -> dict[str, base.FigureSpec]:
    specs: dict[str, base.FigureSpec] = {}
    number = 7
    for source_name in ("Fig3", "Fig4", "Fig5", "Fig6"):
        source = base.FIGURES[source_name]
        layout = (len(source.panels), 3)
        specs[f"Fig{number}"] = replace(
            source,
            name=f"Fig{number}",
            layout=layout,
            shared_y=False,
            y_bounds=(0.0, 1.01),
        )
        number += 1
    return specs


FIGURES = _complexity_specs()


def _panel_y_bounds(metric: str) -> tuple[float, float] | None:
    if metric == "drift_window_test_set_nll":
        return None
    return 0.0, 1.01


def _read_metric_rows(path: Path, max_step: int | None) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Metric CSV not found: {path}")
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"paper_model", "step", "value", "dataset_complexity"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} missing required columns: {sorted(missing)}")

        for row in reader:
            model = row.get("paper_model", "").strip()
            complexity = row.get("dataset_complexity", "").strip().lower()
            if model not in MODEL_ORDER or complexity not in COMPLEXITIES:
                continue
            try:
                step = int(float(row.get("step", "")))
                float(row.get("value", ""))
            except (TypeError, ValueError):
                continue
            if max_step is not None and step > max_step:
                continue
            normalized = dict(row)
            normalized["step"] = str(step)
            normalized["dataset_complexity"] = complexity
            rows.append(normalized)
    return rows


def _dataset_key(row: dict[str, str]) -> str:
    for field in ("dataset_id", "generated_dataset", "preset_name"):
        value = row.get(field, "").strip()
        if value:
            return value
    return row.get("run_id", "").strip()


def _hierarchical_aggregate(rows: list[dict[str, str]]) -> dict[str, dict[str, list[tuple[int, float, float, float]]]]:
    # First collapse duplicate metric points within one dataset and seed.
    dataset_seed_values: dict[tuple[str, str, str, str, int], list[float]] = defaultdict(list)
    for row in rows:
        key = (
            row["dataset_complexity"],
            row["paper_model"],
            _dataset_key(row),
            row.get("seed", "").strip(),
            int(row["step"]),
        )
        dataset_seed_values[key].append(float(row["value"]))

    # Average seeds inside each generated dataset before comparing datasets.
    dataset_values: dict[tuple[str, str, str, int], list[float]] = defaultdict(list)
    for (complexity, model, dataset, _seed, step), values in dataset_seed_values.items():
        dataset_values[(complexity, model, dataset, step)].append(sum(values) / len(values))

    complexity_values: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for (complexity, model, _dataset, step), values in dataset_values.items():
        complexity_values[(complexity, model, step)].append(sum(values) / len(values))

    result: dict[str, dict[str, list[tuple[int, float, float, float]]]] = defaultdict(dict)
    for (complexity, model, step), values in sorted(complexity_values.items()):
        result[complexity].setdefault(model, []).append(
            (step, sum(values) / len(values), min(values), max(values))
        )
    return result


def _metric_series(input_dir: Path, spec: base.FigureSpec, panel: base.PanelSpec) -> dict[str, dict[str, list[tuple[int, float, float, float]]]]:
    path = input_dir / spec.source / f"{panel.metric}.csv"
    return _hierarchical_aggregate(_read_metric_rows(path, spec.max_step))


def _plot_panel(
    ax: plt.Axes,
    series: dict[str, list[tuple[int, float, float, float]]],
    panel: base.PanelSpec,
    complexity: str,
    xlabel: str,
    line_width_scale: float,
) -> None:
    for model in MODEL_ORDER:
        rows = series.get(model)
        if not rows:
            continue
        xs = [row[0] for row in rows]
        means = [row[1] for row in rows]
        lows = [row[2] for row in rows]
        highs = [row[3] for row in rows]
        color = MODEL_COLORS[model]
        ax.fill_between(xs, lows, highs, color=color, alpha=MODEL_BAND_ALPHA[model], linewidth=0)
        ax.plot(
            xs,
            means,
            label=model,
            color=color,
            linestyle=MODEL_LINESTYLES[model],
            linewidth=(1.75 if model == "EOPKG" else 1.45) * line_width_scale,
        )

    ax.set_title(f"{complexity.title()} - {panel.title}", fontsize=11, pad=16)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(panel.ylabel, fontsize=10)
    bounds = _panel_y_bounds(panel.metric)
    if bounds is None:
        values = [bound for row in series.values() for _, _, low, high in row for bound in (low, high)]
        if values:
            lower = min(values)
            upper = max(values)
            padding = max((upper - lower) * 0.05, 0.05)
            ax.set_ylim(lower - padding, upper + padding)
    else:
        ax.set_ylim(*bounds)
    ax.grid(True, color="#D9D9D9", linewidth=0.7, alpha=0.75)
    ax.tick_params(axis="both", labelsize=9)


def _figure_size(spec: base.FigureSpec) -> tuple[float, float]:
    rows, _cols = spec.layout
    return (13.5, 4.8) if rows == 1 else (13.5, 3.4 * rows)


def _render_figure(input_dir: Path, output_dir: Path, spec: base.FigureSpec, formats: list[str], dpi: int) -> list[Path]:
    panel_series = [_metric_series(input_dir, spec, panel) for panel in spec.panels]
    rows, cols = spec.layout
    fig, axes = plt.subplots(rows, cols, figsize=_figure_size(spec), squeeze=False)
    flat_axes = [axis for row in axes for axis in row]
    drift_x_values = [
        row[0]
        for panel_series_item in panel_series
        for complexity_series in panel_series_item.values()
        for model_rows in complexity_series.values()
        for row in model_rows
    ]
    drift_x_bounds = (min(drift_x_values), max(drift_x_values)) if drift_x_values else None

    for panel_index, panel in enumerate(spec.panels):
        for complexity_index, complexity in enumerate(COMPLEXITIES):
            axis = axes[panel_index][complexity_index]
            _plot_panel(
                axis,
                panel_series[panel_index].get(complexity, {}),
                panel,
                complexity,
                spec.xlabel,
                spec.line_width_scale,
            )
            if spec.source == "drift" and drift_x_bounds is not None:
                base._draw_version_boundaries(axis, *drift_x_bounds)

    handles, labels = flat_axes[0].get_legend_handles_labels()
    label_to_handle = dict(zip(labels, handles))
    ordered_labels = [model for model in MODEL_ORDER if model in label_to_handle]
    fig.legend(
        [label_to_handle[label] for label in ordered_labels],
        ordered_labels,
        loc="lower center",
        ncol=len(ordered_labels),
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.5, 0.01),
    )
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 0.92) if spec.source == "drift" else (0.0, 0.06, 1.0, 1.0))

    saved: list[Path] = []
    for fmt in formats:
        target_dir = output_dir / fmt
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"{spec.name}.{fmt}"
        save_kwargs = {"bbox_inches": "tight"}
        if fmt == "png":
            save_kwargs["dpi"] = dpi
        fig.savefig(target, format=fmt, **save_kwargs)
        saved.append(target)
    plt.close(fig)
    return saved


def _parse_formats(value: str) -> list[str]:
    return base._parse_formats(value)


def _select_figures(names: str) -> list[base.FigureSpec]:
    requested = [item.strip() for item in names.split(",") if item.strip()]
    if not requested or requested == ["all"]:
        return [FIGURES[name] for name in sorted(FIGURES)]
    unknown = [name for name in requested if name not in FIGURES]
    if unknown:
        raise argparse.ArgumentTypeError(f"Unknown figure(s): {', '.join(unknown)}")
    return [FIGURES[name] for name in requested]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export article figures split by dataset complexity.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("outputs/Export_metrics/article_run_metrics/CDLG"),
        help="Directory containing raw learn/ and drift/ metric CSV folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/article_complexity_figures"),
        help="Directory where complexity figure format folders are written.",
    )
    parser.add_argument("--figures", default="all", help="Comma-separated complexity figure ids, e.g. Fig7,Fig8.")
    parser.add_argument("--formats", type=_parse_formats, default=["svg", "png"])
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected = _select_figures(args.figures)
    saved: list[Path] = []
    for spec in selected:
        saved.extend(_render_figure(args.input_dir, args.output_dir, spec, args.formats, args.dpi))
    print("Exported article complexity figures:")
    for path in saved:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
