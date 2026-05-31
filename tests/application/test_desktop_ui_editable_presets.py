from __future__ import annotations

import json
import os
from pathlib import Path
import pytest

from PySide6.QtWidgets import QApplication, QComboBox
from tools.desktop_ui.field_registry import DesktopFieldMeta
from tools.desktop_ui.field_widgets import create_field_widget
from tools.desktop_ui.preset_manager import (
    VARS_MAPPING,
    flatten_preset_payload,
    build_legacy_payload,
    PresetDrawer,
)


def test_field_widget_combobox_is_editable(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    _app = QApplication.instance() or QApplication([])

    field = DesktopFieldMeta(
        path="data.dataset_label",
        section="data",
        label="Dataset Label",
        description="",
        affects="",
        default="bpi2012",
        enum=("bpi2012", "synthetic_drift_4tu"),
        ui_level="project_setup",
        ui_tab="input",
        ui_group="General",
        ui_order=100,
        required_in_modes=(),
        required_when={},
        active_when={},
        emit_when={},
        runtime_consumers=(),
    )

    changed_val = []
    widget = create_field_widget(
        field,
        "custom_value",
        lambda p, val: changed_val.append((p, val)),
    )

    assert isinstance(widget, QComboBox)
    assert widget.isEditable()
    assert widget.currentText() == "custom_value"

    # Simulate user typing custom text
    widget.setCurrentText("my_custom_dataset")
    assert len(changed_val) == 1
    assert changed_val[0] == ("data.dataset_label", "my_custom_dataset")


def test_flatten_preset_payload():
    payload = {
        "input_data_form": {
            "data.dataset_label": "bpi2012_stage4_2",
            "data.dataset_name": "bpi2012",
        },
        "vars": {
            "adapter": "xes",
            "mode": "eval_drift",
            "fraction": "0.5",
        },
        "features_text": "features yaml content",
        "policies_text": "policies yaml content",
    }

    flat = flatten_preset_payload(payload)

    # Check parsed forms
    assert flat["data.dataset_label"] == "bpi2012_stage4_2"
    assert flat["data.dataset_name"] == "bpi2012"

    # Check mapped vars
    assert flat["mapping.adapter"] == "xes"
    assert flat["experiment.mode"] == "eval_drift"
    assert flat["experiment.fraction"] == "0.5"

    # Check text fields mapping
    assert flat["mapping.features"] == "features yaml content"
    assert flat["features"] == "features yaml content"
    assert flat["policies"] == "policies yaml content"


def test_build_legacy_payload():
    values = {
        "data.dataset_label": "camunda_procurement",
        "mapping.adapter": "camunda",
        "experiment.mode": "train",
        "mapping.knowledge_graph.backend": "neo4j",
        "mapping.features": "- name: concept:name",
        "policies": "profile: default",
    }

    payload = build_legacy_payload(values)

    # Verify text blocks
    assert payload["features_text"] == "- name: concept:name"
    assert payload["policies_text"] == "profile: default"

    # Verify forms
    assert payload["input_data_form"]["data.dataset_label"] == "camunda_procurement"
    assert payload["eopkg_backend_form"]["mapping.knowledge_graph.backend"] == "neo4j"

    # Verify mapped vars
    assert payload["vars"]["adapter"] == "camunda"
    assert payload["vars"]["mode"] == "train"


def test_preset_drawer_loads_legacy_presets(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    _app = QApplication.instance() or QApplication([])

    presets_file = tmp_path / "presets.json"
    preset_data = {
        "LegacyPreset": {
            "last_saved_at": "2026-04-09T17:27:01.460315+00:00",
            "payload": {
                "input_data_form": {
                    "data.dataset_label": "bpi2012",
                },
                "vars": {
                    "adapter": "xes",
                }
            }
        }
    }
    presets_file.write_text(json.dumps(preset_data), encoding="utf-8")

    drawer = PresetDrawer(
        presets_path=presets_file,
        on_save_callback=lambda: {},
    )

    loaded_values = []
    drawer.preset_loaded.connect(loaded_values.append)

    # Trigger tree selection on legacy preset
    drawer.selected_preset_name = "LegacyPreset"
    drawer._load_selected()

    assert len(loaded_values) == 1
    flat_values = loaded_values[0]
    assert flat_values["data.dataset_label"] == "bpi2012"
    assert flat_values["mapping.adapter"] == "xes"
