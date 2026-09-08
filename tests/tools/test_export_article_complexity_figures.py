from __future__ import annotations

import pytest

from tools import export_article_complexity_figures as exporter


def test_hierarchical_aggregation_weights_datasets_equally_over_seeds():
    rows = [
        {"paper_model": "GATv2", "dataset_complexity": "simple", "preset_name": "dataset-a", "seed": "1", "step": "0", "value": "0.2"},
        {"paper_model": "GATv2", "dataset_complexity": "simple", "preset_name": "dataset-a", "seed": "2", "step": "0", "value": "0.4"},
        {"paper_model": "GATv2", "dataset_complexity": "simple", "preset_name": "dataset-b", "seed": "1", "step": "0", "value": "0.8"},
    ]

    result = exporter._hierarchical_aggregate(rows)

    assert result["simple"]["GATv2"][0] == pytest.approx((0, 0.55, 0.3, 0.8))


def test_complexity_figures_have_three_panels_and_fixed_y_axis():
    assert exporter.FIGURES["Fig7"].layout == (1, 3)
    assert exporter.FIGURES["Fig8"].layout == (1, 3)
    assert exporter.FIGURES["Fig9"].layout == (1, 3)
    assert exporter.FIGURES["Fig10"].layout == (4, 3)
    assert all(spec.y_bounds == (0.0, 1.01) for spec in exporter.FIGURES.values())
    assert exporter._panel_y_bounds("drift_window_strict_macro_f1") == (0.0, 1.01)
    assert exporter._panel_y_bounds("drift_window_test_set_nll") is None


def test_complexity_tool_reuses_article_model_styles():
    from tools import export_article_figures

    expected_models = ["GATv2", "GATv2+Mask", "EOPKG-WI", "EOPKG", "MOU"]
    assert exporter.MODEL_ORDER == expected_models
    assert exporter.MODEL_COLORS == {model: export_article_figures.MODEL_COLORS[model] for model in expected_models}
    assert exporter.MODEL_LINESTYLES == {model: export_article_figures.MODEL_LINESTYLES[model] for model in expected_models}
    assert exporter.MODEL_BAND_ALPHA == {model: export_article_figures.MODEL_BAND_ALPHA[model] for model in expected_models}


def test_version_boundaries_are_equal_step_transitions_with_zone_labels():
    assert exporter.base._version_boundaries(0, 190, 5) == [38.0, 76.0, 114.0, 152.0]
    assert exporter.base._version_labels(0, 190, 5) == [
        (19.0, "v1"),
        (57.0, "v2"),
        (95.0, "v3"),
        (133.0, "v4"),
        (171.0, "v5"),
    ]
