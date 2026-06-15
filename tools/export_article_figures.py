"""Export publication-ready article figures from aggregated run metric CSVs.

The script reads per-metric CSV files produced under
``Export_metrics/article_run_metrics`` and renders the figures referenced in
the EOPKG structural drift article draft. Lines show the mean over seeds, while
the translucent band shows the min/max seed envelope at each step.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


MODEL_ORDER = ["LSTM", "GATv2", "GATv2+Mask", "EOPKG-WI", "EOPKG"]

MODEL_COLORS = {
    "LSTM": "#4C78A8",
    "GATv2": "#F58518",
    "GATv2+Mask": "#F58518",
    "EOPKG-WI": "#2CA02C",
    "EOPKG": "#2CA02C",
}

MODEL_LINESTYLES = {
    "LSTM": "-",
    "GATv2": "-",
    "GATv2+Mask": (0, (4, 2)),
    "EOPKG-WI": (0, (4, 2)),
    "EOPKG": "-",
}

MODEL_BAND_ALPHA = {
    "LSTM": 0.15,
    "GATv2": 0.13,
    "GATv2+Mask": 0.20,
    "EOPKG-WI": 0.16,
    "EOPKG": 0.20,
}


@dataclass(frozen=True)
class PanelSpec:
    metric: str
    title: str
    ylabel: str


@dataclass(frozen=True)
class FigureSpec:
    name: str
    source: str
    panels: tuple[PanelSpec, ...]
    layout: tuple[int, int]
    max_step: int | None = None
    shared_y: bool = False
    y_bounds: tuple[float | None, float | None] | None = None
    xlabel: str = "Step"
    line_width_scale: float = 1.0


FIGURES = {
    "Fig3": FigureSpec(
        name="Fig3",
        source="learn",
        panels=(
            PanelSpec("train_loss", "A. Training loss", "Loss"),
            PanelSpec("val_loss", "B. Validation loss", "Loss"),
        ),
        layout=(1, 2),
        max_step=50,
        shared_y=True,
        xlabel="Epoch",
    ),
    "Fig4": FigureSpec(
        name="Fig4",
        source="learn",
        panels=(
            PanelSpec("strict_val_macro_f1", "Strict validation macro-F1", "Macro-F1"),
        ),
        layout=(1, 1),
        max_step=50,
        xlabel="Epoch",
    ),
    "Fig5": FigureSpec(
        name="Fig5",
        source="drift",
        panels=(PanelSpec("drift_window_strict_macro_f1", "Strict macro-F1 under structural drift", "Strict macro-F1"),),
        layout=(1, 1),
        xlabel="Drift step",
        line_width_scale=0.8,
    ),
    "Fig6": FigureSpec(
        name="Fig6",
        source="drift",
        panels=(PanelSpec("drift_window_pred_in_mask_rate", "Prediction in current topology mask", "Prediction in mask rate"),),
        layout=(1, 1),
        y_bounds=(0.0, 1.02),
        xlabel="Drift step",
        line_width_scale=0.8,
    ),
    "Fig7": FigureSpec(
        name="Fig7",
        source="drift",
        panels=(
            PanelSpec("drift_window_test_oos", "A. Out-of-structure rate", "OOS rate"),
            PanelSpec(
                "drift_window_strict_error_but_allowed_rate",
                "B. Strict error but allowed",
                "Allowed strict errors",
            ),
            PanelSpec("drift_window_test_set_nll", "C. Negative log-likelihood", "NLL"),
            PanelSpec("drift_window_test_ece", "D. Expected calibration error", "ECE"),
        ),
        layout=(2, 2),
        xlabel="Drift step",
        line_width_scale=0.8,
    ),
}


def _parse_formats(value: str) -> list[str]:
    formats = [item.strip().lower() for item in value.split(",") if item.strip()]
    allowed = {"svg", "png", "pdf"}
    unsupported = sorted(set(formats) - allowed)
    if unsupported:
        raise argparse.ArgumentTypeError(f"Unsupported format(s): {', '.join(unsupported)}")
    return formats or ["svg", "png"]


def _read_metric_csv(path: Path, max_step: int | None) -> dict[str, dict[int, list[float]]]:
    if not path.exists():
        raise FileNotFoundError(f"Metric CSV not found: {path}")

    grouped: dict[str, dict[int, list[float]]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"paper_model", "step", "value"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} missing required columns: {sorted(missing)}")

        for row in reader:
            model = row["paper_model"].strip()
            if model not in MODEL_ORDER:
                continue
            try:
                step = int(float(row["step"]))
                value = float(row["value"])
            except (TypeError, ValueError):
                continue
            if max_step is not None and step > max_step:
                continue
            grouped.setdefault(model, {}).setdefault(step, []).append(value)
    return grouped


def _aggregate(grouped: dict[str, dict[int, list[float]]]) -> dict[str, list[tuple[int, float, float, float]]]:
    aggregated: dict[str, list[tuple[int, float, float, float]]] = {}
    for model in MODEL_ORDER:
        by_step = grouped.get(model, {})
        rows: list[tuple[int, float, float, float]] = []
        for step in sorted(by_step):
            values = by_step[step]
            if not values:
                continue
            mean = sum(values) / len(values)
            rows.append((step, mean, min(values), max(values)))
        if rows:
            aggregated[model] = rows
    return aggregated


def _metric_series(input_dir: Path, spec: FigureSpec, panel: PanelSpec) -> dict[str, list[tuple[int, float, float, float]]]:
    path = input_dir / spec.source / f"{panel.metric}.csv"
    return _aggregate(_read_metric_csv(path, spec.max_step))


def _values_for_ylim(series_list: Iterable[dict[str, list[tuple[int, float, float, float]]]]) -> list[float]:
    values: list[float] = []
    for series in series_list:
        for rows in series.values():
            for _, _, ymin, ymax in rows:
                values.extend([ymin, ymax])
    return values


def _shared_ylim(
    series_list: Iterable[dict[str, list[tuple[int, float, float, float]]]],
    y_bounds: tuple[float | None, float | None] | None,
    padding: float = 0.05,
) -> tuple[float, float] | None:
    values = _values_for_ylim(series_list)
    if not values:
        return None

    ymin = min(values)
    ymax = max(values)
    span = ymax - ymin
    if span == 0:
        span = max(abs(ymax), 1.0) * 0.05
    ymin -= span * padding
    ymax += span * padding

    if y_bounds is not None:
        lower, upper = y_bounds
        if lower is not None:
            ymin = max(lower, ymin)
        if upper is not None:
            ymax = min(upper, ymax)
    return ymin, ymax


def _plot_panel(
    ax: plt.Axes,
    series: dict[str, list[tuple[int, float, float, float]]],
    panel: PanelSpec,
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

    ax.set_title(panel.title, fontsize=11, pad=8)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(panel.ylabel, fontsize=10)
    ax.grid(True, color="#D9D9D9", linewidth=0.7, alpha=0.75)
    ax.tick_params(axis="both", labelsize=9)


def _figure_size(layout: tuple[int, int]) -> tuple[float, float]:
    rows, cols = layout
    if rows == 1 and cols == 1:
        return 7.0, 4.4
    if rows == 1 and cols == 2:
        return 11.5, 4.4
    return 11.5, 7.2


def _render_figure(input_dir: Path, output_dir: Path, spec: FigureSpec, formats: list[str], dpi: int) -> list[Path]:
    panel_series = [_metric_series(input_dir, spec, panel) for panel in spec.panels]
    rows, cols = spec.layout
    fig, axes = plt.subplots(rows, cols, figsize=_figure_size(spec.layout), squeeze=False)
    flat_axes = [axis for row in axes for axis in row]

    shared_ylim = _shared_ylim(panel_series, spec.y_bounds) if spec.shared_y else None

    for ax, panel, series in zip(flat_axes, spec.panels, panel_series):
        _plot_panel(ax, series, panel, spec.xlabel, spec.line_width_scale)
        if shared_ylim is not None:
            ax.set_ylim(*shared_ylim)
        elif spec.y_bounds is not None:
            lower, upper = spec.y_bounds
            current_lower, current_upper = ax.get_ylim()
            ax.set_ylim(lower if lower is not None else current_lower, upper if upper is not None else current_upper)

    for ax in flat_axes[len(spec.panels) :]:
        ax.axis("off")

    handles, labels = flat_axes[0].get_legend_handles_labels()
    label_to_handle = dict(zip(labels, handles))
    ordered_labels = [model for model in MODEL_ORDER if model in label_to_handle]
    ordered_handles = [label_to_handle[label] for label in ordered_labels]
    fig.legend(
        ordered_handles,
        ordered_labels,
        loc="lower center",
        ncol=len(ordered_labels),
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.5, 0.01),
    )
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))

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


def _select_figures(names: str) -> list[FigureSpec]:
    requested = [item.strip() for item in names.split(",") if item.strip()]
    if not requested or requested == ["all"]:
        return [FIGURES[name] for name in sorted(FIGURES)]
    unknown = [name for name in requested if name not in FIGURES]
    if unknown:
        raise argparse.ArgumentTypeError(f"Unknown figure(s): {', '.join(unknown)}")
    return [FIGURES[name] for name in requested]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export EOPKG article figures from article_run_metrics CSV files.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("Export_metrics/article_run_metrics"),
        help="Directory containing learn/ and drift/ metric CSV folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/article_figures"),
        help="Directory where format-specific figure folders will be written.",
    )
    parser.add_argument(
        "--figures",
        default="all",
        help="Comma-separated figure ids to export, e.g. Fig3,Fig4. Default: all.",
    )
    parser.add_argument(
        "--formats",
        type=_parse_formats,
        default=["svg", "png"],
        help="Comma-separated output formats: svg,png,pdf. Default: svg,png.",
    )
    parser.add_argument("--dpi", type=int, default=600, help="PNG export DPI. Default: 600.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected = _select_figures(args.figures)
    saved: list[Path] = []

    for spec in selected:
        saved.extend(_render_figure(args.input_dir, args.output_dir, spec, args.formats, args.dpi))

    print("Exported article figures:")
    for path in saved:
        print(f"  {path}")
    print("Learn figures are clipped to epochs 0-50; drift figures use the full chronological trajectory.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
