from pathlib import Path

from tools import export_article_figures as exporter


def test_boundary_on_flag_is_false_by_default(monkeypatch):
    monkeypatch.setattr("sys.argv", ["export_article_figures.py"])

    args = exporter.parse_args()

    assert args.boundary_on is False


def test_boundary_on_flag_enables_drift_boundaries(monkeypatch, tmp_path: Path):
    metric_dir = tmp_path / "drift"
    metric_dir.mkdir()
    (metric_dir / "drift_window_strict_macro_f1.csv").write_text(
        "paper_model,step,value\nGATv2,0,0.7\nGATv2,1,0.6\n",
        encoding="utf-8",
    )
    calls = []
    monkeypatch.setattr(
        exporter,
        "_draw_version_boundaries",
        lambda ax, x_min, x_max: calls.append((x_min, x_max)),
    )

    exporter._render_figure(
        tmp_path,
        tmp_path / "output-off",
        exporter.FIGURES["Fig4"],
        ["png"],
        40,
        boundary_on=False,
    )
    assert calls == []

    exporter._render_figure(
        tmp_path,
        tmp_path / "output-on",
        exporter.FIGURES["Fig4"],
        ["png"],
        40,
        boundary_on=True,
    )
    assert calls == [(0, 1)]


def test_version_boundaries_include_plus_minus_ten_step_shaded_zones():
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots()
    exporter._draw_version_boundaries(axis, 0, 100)

    assert len(axis.patches) == 4
    assert [(patch.get_x(), patch.get_width()) for patch in axis.patches] == [
        (10.0, 20.0),
        (30.0, 20.0),
        (50.0, 20.0),
        (70.0, 20.0),
    ]
    plt.close(figure)
