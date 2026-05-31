# tools/desktop_ui/preset_manager.py
from __future__ import annotations

import json
import yaml
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import QEasingCurve, QPropertyAnimation, QSize, Qt, Signal
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)


VARS_MAPPING = {
    "adapter": "mapping.adapter",
    "mode": "experiment.mode",
    "project": "experiment.project",
    "experiment_name": "experiment.name",
    "fraction": "experiment.fraction",
    "fraction_strategy": "experiment.fraction_strategy",
    "split_strategy": "experiment.split_strategy",
    "train_ratio": "experiment.train_ratio",
    "split_ratio": "experiment.split_ratio",
    "version_scope_policy": "experiment.version_scope_policy",
    "seed": "seed",
    "stats_time_policy": "experiment.stats_time_policy",
    "on_missing_asof_snapshot": "experiment.on_missing_asof_snapshot",
    "cache_policy": "experiment.cache_policy",
    "graph_dataset_cache_policy": "experiment.graph_dataset_cache_policy",
    "graph_dataset_cache_dir": "experiment.graph_dataset_cache_dir",
    "gateway_mode": "mapping.graph_feature_mapping.topology_projection.gateway_mode",
    "sync_as_of": "sync_stats.sync_as_of",
    "backfill_step": "sync_stats.backfill_step",
    "backfill_step_days": "sync_stats.backfill_step_days",
    "backfill_from": "sync_stats.backfill_from",
    "backfill_to": "sync_stats.backfill_to",
}


def flatten_preset_payload(payload: dict[str, Any]) -> dict[str, Any]:
    flat = {}
    for form_key, form_values in payload.items():
        if isinstance(form_values, dict):
            if form_key == "vars":
                for k, v in form_values.items():
                    mapped_key = VARS_MAPPING.get(k)
                    if mapped_key:
                        flat[mapped_key] = v
            else:
                for k, v in form_values.items():
                    flat[k] = v
        else:
            if form_key == "features_text":
                flat["mapping.features"] = form_values
                flat["features"] = form_values
            elif form_key == "policies_text":
                flat["policies"] = form_values
            elif form_key == "graph_mapping_text":
                flat["mapping.graph_feature_mapping"] = form_values
            else:
                flat[form_key] = form_values
    return flat


def build_legacy_payload(values: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "eopkg_backend_form": {},
        "eopkg_structure_form": {},
        "general_experiment_form": {},
        "general_tracking_form": {},
        "general_training_form": {},
        "input_camunda_mapping_form": {},
        "input_camunda_runtime_form": {},
        "input_data_form": {},
        "input_xes_form": {},
        "model_form": {},
        "sync_stats_form": {},
        "vars": {},
    }
    VARS_INVERSE = {v: k for k, v in VARS_MAPPING.items()}

    def to_yaml_str(val: Any) -> str:
        if val is None:
            return ""
        if isinstance(val, (dict, list)):
            return yaml.safe_dump(val, default_flow_style=False)
        return str(val)

    payload["features_text"] = to_yaml_str(values.get("mapping.features", values.get("features", "")))
    payload["policies_text"] = to_yaml_str(values.get("policies", ""))
    payload["graph_mapping_text"] = to_yaml_str(values.get("mapping.graph_feature_mapping", ""))

    for path, val in values.items():
        if path in {"mapping.features", "features", "policies", "mapping.graph_feature_mapping"}:
            continue
        if path in VARS_INVERSE:
            short_key = VARS_INVERSE[path]
            payload["vars"][short_key] = str(val)
        
        if path.startswith("mapping.knowledge_graph."):
            payload["eopkg_backend_form"][path] = str(val)
        elif path.startswith("mapping.camunda_adapter.structure.") or path == "mapping.camunda_adapter.subprocess_graph_mode":
            payload["eopkg_structure_form"][path] = str(val)
        elif path.startswith("experiment."):
            if path not in VARS_INVERSE:
                payload["general_experiment_form"][path] = str(val)
        elif path.startswith("tracking."):
            payload["general_tracking_form"][path] = str(val)
        elif path.startswith("training."):
            if path not in VARS_INVERSE:
                payload["general_training_form"][path] = str(val)
        elif path.startswith("mapping.camunda_adapter.runtime."):
            payload["input_camunda_runtime_form"][path] = str(val)
        elif path.startswith("mapping.camunda_adapter."):
            camunda_mapping_keys = {
                "mapping.camunda_adapter.lookback_hours",
                "mapping.camunda_adapter.process_filters",
                "mapping.camunda_adapter.process_name",
                "mapping.camunda_adapter.since",
                "mapping.camunda_adapter.tenant_filters",
                "mapping.camunda_adapter.tenant_id",
                "mapping.camunda_adapter.until",
                "mapping.camunda_adapter.version_key"
            }
            if path in camunda_mapping_keys:
                payload["input_camunda_mapping_form"][path] = str(val)
        elif path.startswith("data."):
            payload["input_data_form"][path] = str(val)
        elif path.startswith("mapping.xes_adapter."):
            payload["input_xes_form"][path] = str(val)
        elif path.startswith("model."):
            payload["model_form"][path] = str(val)
        elif path.startswith("sync_stats."):
            if path not in VARS_INVERSE:
                payload["sync_stats_form"][path] = str(val)
    return payload


class PresetDrawer(QWidget):
    preset_loaded = Signal(dict)  # Emits configuration dict when Load clicked

    def __init__(self, *, presets_path: Path, on_save_callback: Callable[[], dict[str, Any]], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.presets_path = presets_path
        self.on_save_callback = on_save_callback
        self.presets: dict[str, Any] = {}
        self.selected_preset_name: str | None = None
        self.drawer_visible = False

        self._init_ui()
        self._load_presets_file()
        self._populate_tree()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        # Header
        header_layout = QHBoxLayout()
        title = QLabel("Presets Manager")
        title.setStyleSheet("font-weight: bold; font-size: 15px;")
        self.close_btn = QPushButton("✕")
        self.close_btn.setFixedSize(24, 24)
        self.close_btn.setStyleSheet("border: none; font-weight: bold;")
        self.close_btn.clicked.connect(lambda: self.set_drawer_visible(False))
        header_layout.addWidget(title)
        header_layout.addStretch()
        header_layout.addWidget(self.close_btn)
        layout.addLayout(header_layout)

        # Search bar
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Search presets...")
        self.search_box.textChanged.connect(self._filter_presets)
        layout.addWidget(self.search_box)

        # Tree View
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Presets"])
        self.tree.header().setSectionResizeMode(QHeaderView.Stretch)
        self.tree.itemSelectionChanged.connect(self._on_preset_selected)
        layout.addWidget(self.tree, 1)

        # Details Panel
        self.details_widget = QWidget()
        details_layout = QVBoxLayout(self.details_widget)
        details_layout.setContentsMargins(0, 0, 0, 0)
        details_layout.setSpacing(6)

        details_layout.addWidget(QLabel("Preset Name:"))
        self.preset_name_input = QLineEdit()
        details_layout.addWidget(self.preset_name_input)

        self.mode_label = QLabel("Mode: -")
        self.date_label = QLabel("Saved: -")
        details_layout.addWidget(self.mode_label)
        details_layout.addWidget(self.date_label)

        details_layout.addWidget(QLabel("Comment:"))
        self.comment_input = QTextEdit()
        self.comment_input.setMaximumHeight(80)
        details_layout.addWidget(self.comment_input)

        layout.addWidget(self.details_widget)

        # Action Buttons
        btn_layout = QHBoxLayout()
        self.load_btn = QPushButton("Load Preset")
        self.load_btn.setEnabled(False)
        self.load_btn.clicked.connect(self._load_selected)
        self.save_btn = QPushButton("Save")
        self.save_btn.clicked.connect(self._save_current)
        self.delete_btn = QPushButton("Delete")
        self.delete_btn.setEnabled(False)
        self.delete_btn.clicked.connect(self._delete_selected)

        btn_layout.addWidget(self.load_btn)
        btn_layout.addWidget(self.save_btn)
        btn_layout.addWidget(self.delete_btn)
        layout.addLayout(btn_layout)

        # Animation Settings
        self.anim = QPropertyAnimation(self, b"minimumWidth")
        self.anim.setDuration(250)
        self.anim.setEasingCurve(QEasingCurve.OutQuad)

    def set_drawer_visible(self, visible: bool) -> None:
        self.drawer_visible = visible
        self.anim.stop()
        try:
            self.anim.finished.disconnect()
        except (RuntimeError, TypeError):
            pass
        if visible:
            self.show()
            self.anim.setStartValue(self.width())
            self.anim.setEndValue(320)
        else:
            self.anim.setStartValue(self.width())
            self.anim.setEndValue(0)
            self.anim.finished.connect(self.hide)
        self.anim.start()

    def _load_presets_file(self) -> None:
        if self.presets_path.exists():
            try:
                with open(self.presets_path, "r", encoding="utf-8") as f:
                    self.presets = json.load(f) or {}
            except Exception:
                self.presets = {}
        else:
            self.presets = {}

    def _save_presets_file(self) -> None:
        self.presets_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.presets_path, "w", encoding="utf-8") as f:
                json.dump(self.presets, f, indent=2, ensure_ascii=False, sort_keys=True)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save presets: {e}")

    def _populate_tree(self) -> None:
        self.tree.clear()
        projects: dict[str, list[str]] = {}
        for name, data in self.presets.items():
            project = data.get("project") or data.get("values", {}).get("experiment.project") or "Other"
            projects.setdefault(project, []).append(name)

        for project, names in sorted(projects.items()):
            parent_item = QTreeWidgetItem(self.tree, [project])
            parent_item.setFlags(parent_item.flags() & ~Qt.ItemIsSelectable)
            for name in sorted(names):
                QTreeWidgetItem(parent_item, [name])
        self.tree.expandAll()

    def _filter_presets(self, text: str) -> None:
        needle = text.strip().lower()
        for i in range(self.tree.topLevelItemCount()):
            parent = self.tree.topLevelItem(i)
            parent_match = False
            for j in range(parent.childCount()):
                child = parent.child(j)
                name = child.text(0)
                preset_data = self.presets.get(name, {})
                comment = preset_data.get("comment", "")
                visible = not needle or needle in name.lower() or needle in comment.lower()
                child.setHidden(not visible)
                if visible:
                    parent_match = True
            parent.setHidden(not parent_match and needle != "")

    def _on_preset_selected(self) -> None:
        items = self.tree.selectedItems()
        if not items or items[0].parent() is None:
            self.selected_preset_name = None
            self.load_btn.setEnabled(False)
            self.delete_btn.setEnabled(False)
            return

        name = items[0].text(0)
        self.selected_preset_name = name
        data = self.presets[name]

        self.preset_name_input.setText(name)
        self.mode_label.setText(f"Mode: {data.get('mode', '-')}")
        self.date_label.setText(f"Saved: {data.get('saved_at', '-')[:19]}")
        self.comment_input.setPlainText(data.get("comment", ""))

        self.load_btn.setEnabled(True)
        self.delete_btn.setEnabled(True)

    def _load_selected(self) -> None:
        if not self.selected_preset_name:
            return
        data = self.presets[self.selected_preset_name]
        flat_values = {}
        if "values" in data and data["values"]:
            flat_values.update(data["values"])
        if "payload" in data and data["payload"]:
            legacy_flat = flatten_preset_payload(data["payload"])
            for k, v in legacy_flat.items():
                if k not in flat_values:
                    flat_values[k] = v
        self.preset_loaded.emit(flat_values)

    def _save_current(self) -> None:
        name = self.preset_name_input.text().strip()
        if not name:
            QMessageBox.warning(self, "Warning", "Please enter a preset name.")
            return

        values = self.on_save_callback()
        mode = values.get("experiment.mode", "train")
        project = values.get("experiment.project", "Other")
        legacy_payload = build_legacy_payload(values)

        self.presets[name] = {
            "values": values,
            "payload": legacy_payload,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "last_saved_at": datetime.now(timezone.utc).isoformat(),
            "mode": mode,
            "project": project,
            "comment": self.comment_input.toPlainText().strip()
        }
        self._save_presets_file()
        self._populate_tree()
        QMessageBox.information(self, "Success", f"Preset '{name}' saved successfully.")

    def _delete_selected(self) -> None:
        if not self.selected_preset_name:
            return
        reply = QMessageBox.question(
            self, "Confirm Delete", f"Are you sure you want to delete preset '{self.selected_preset_name}'?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.presets.pop(self.selected_preset_name, None)
            self._save_presets_file()
            self._populate_tree()
            self.selected_preset_name = None
            self.preset_name_input.clear()
            self.comment_input.clear()
            self.load_btn.setEnabled(False)
            self.delete_btn.setEnabled(False)
