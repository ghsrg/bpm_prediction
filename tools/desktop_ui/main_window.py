# tools/desktop_ui/main_window.py
from __future__ import annotations

import os
import yaml
from datetime import datetime
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QStatusBar,
    QTextEdit,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from .checkpoint_resolver import DEFAULT_CHECKPOINT_VALUE, resolve_checkpoint_candidates
from .field_registry import DesktopFieldMeta, DesktopFieldRegistry
from .field_widgets import create_field_widget
from .preset_manager import PresetDrawer
from .run_monitor import RunMonitorWidget
from .styles import DARK_SLATE_THEME

START_MAXIMIZED = True
DEFAULT_NAV_MIN_WIDTH = 50
DEFAULT_NAV_INITIAL_WIDTH = 260
DEFAULT_INSPECTOR_WIDTH = 420
DEFAULT_SPLITTER_SIZES = [DEFAULT_NAV_INITIAL_WIDTH, 1180, DEFAULT_INSPECTOR_WIDTH]
EXPERIMENT_RUN_COLUMNS = 1
FIELD_WIDGET_MAX_WIDTH = 720


EXPERIMENT_RUN_GROUP_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "Run Identity",
        (
            "experiment.mode",
            "experiment.project",
            "experiment.name",
            "seed",
        ),
    ),
    (
        "Data Slice",
        (
            "experiment.fraction",
            "experiment.fraction_strategy",
            "experiment.train_ratio",
            "experiment.split_strategy",
            "experiment.split_ratio",
            "experiment.version_scope_policy",
            "experiment.stats_time_policy",
            "experiment.on_missing_asof_snapshot",
        ),
    ),
    (
        "Structure Signal",
        (
            "experiment.structural_mode",
            "experiment.statistic_enabled",
            "experiment.mask_guided_enabled",
        ),
    ),
    (
        "Model / Fusion",
        (
            "model.type",
            "model.model_label",
            "model.fusion_mode",
            "model.pooling_strategy",
            "model.graph_strategy",
            "model.structural_score_mode",
        ),
    ),
    (
        "Learning",
        (
            "training.learning_strategy",
            "training.retrain",
            "training.epochs",
            "training.batch_size",
            "training.learning_rate",
            "training.device",
            "training.delta",
            "training.patience",
            "training.backend",
        ),
    ),
    (
        "Checkpoint / Tracking",
        (
            "experiment.load_checkpoint",
            "tracking.backend",
            "tracking.uri",
            "tracking.experiment_name",
        ),
    ),
)


PROJECT_SETUP_GROUP_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "Dataset / Adapter",
        (
            "data.dataset_name",
            "data.dataset_label",
            "mapping.adapter",
        ),
    ),
    (
        "XES Source",
        (
            "data.log_path",
            "mapping.xes_adapter.activity_key",
            "mapping.xes_adapter.timestamp_key",
            "mapping.xes_adapter.case_id_key",
            "mapping.xes_adapter.version_key",
            "mapping.xes_adapter.lifecycle_key",
            "mapping.xes_adapter.start_transitions",
            "mapping.xes_adapter.complete_transitions",
            "mapping.xes_adapter.use_classifier",
        ),
    ),
    (
        "Camunda Runtime",
        (
            "mapping.camunda_adapter.runtime.runtime_source",
            "mapping.camunda_adapter.runtime.export_dir",
            "mapping.camunda_adapter.runtime.sql_dir",
            "mapping.camunda_adapter.runtime.mssql_profile",
            "mapping.camunda_adapter.runtime.proc_def_key",
            "mapping.camunda_adapter.runtime.tenant_id",
        ),
    ),
    (
        "Camunda Structure",
        (
            "mapping.camunda_adapter.structure.source",
            "mapping.camunda_adapter.structure.bpmn_source",
            "mapping.camunda_adapter.structure.files.bpmn_dir",
            "mapping.camunda_adapter.structure.files.catalog_file",
            "mapping.camunda_adapter.structure.sql_dir",
            "mapping.camunda_adapter.structure.mssql_profile",
        ),
    ),
    (
        "Knowledge Graph",
        (
            "mapping.knowledge_graph.backend",
            "mapping.knowledge_graph.path",
            "mapping.knowledge_graph.strict_load",
            "mapping.knowledge_graph.uri",
            "mapping.knowledge_graph.user",
            "mapping.knowledge_graph.password",
            "mapping.knowledge_graph.database",
        ),
    ),
)


PROJECT_SETUP_TABS = {
    "Data & Adapters": [
        ("Dataset / Adapter", "Dataset & Adapter Configuration"),
        ("XES Source", "XES Log Source Settings"),
        ("Camunda Runtime", "Camunda Process Engine Runtime Configuration"),
        ("Camunda Structure", "Camunda Process Model Structure Configuration"),
    ],
    "Knowledge Graph & Stats": [
        ("Knowledge Graph", "Knowledge Graph Connection Settings"),
        ("Stats / Mapping", "Historical Statistics & Feature Mapping"),
        ("Other", "Other Connection Settings"),
    ]
}

EXPERIMENT_RUN_TABS = {
    "Experiment Parameters": [
        ("Run Identity", "Experiment & Run Identity"),
        ("Data Slice", "Data Split & Cascade Sampling Strategy"),
        ("Checkpoint / Tracking", "Model Checkpoints & MLflow Tracking"),
    ],
    "Model & Training": [
        ("Structure Signal", "Knowledge Graph Structural Signals"),
        ("Model / Fusion", "Model Architecture & Structural Fusion Modes"),
        ("Learning", "Learning Strategy & Training Hyperparameters"),
        ("Other", "Other Run Settings"),
    ]
}

ADVANCED_TABS = {
    "System & Performance": [
        ("General", "Performance, Cache & Memory Limits"),
        ("Other", "Other Advanced Settings"),
    ]
}


def group_fields_for_page(registry: DesktopFieldRegistry, ui_level: str) -> dict[str, list[DesktopFieldMeta]]:
    if ui_level == "project_setup":
        return _group_project_setup_fields(registry)
    if ui_level != "experiment_run":
        return registry.grouped_fields(ui_level)

    fields_by_path = {field.path: field for field in registry.fields_for_level(ui_level)}
    grouped: dict[str, list[DesktopFieldMeta]] = {}
    used: set[str] = set()
    for group_name, paths in EXPERIMENT_RUN_GROUP_RULES:
        fields = [fields_by_path[path] for path in paths if path in fields_by_path]
        if fields:
            grouped[group_name] = fields
            used.update(field.path for field in fields)
    remaining = [field for field in fields_by_path.values() if field.path not in used]
    if remaining:
        grouped["Other"] = remaining
    return grouped


def _group_project_setup_fields(registry: DesktopFieldRegistry) -> dict[str, list[DesktopFieldMeta]]:
    fields_by_path = {field.path: field for field in registry.fields_for_level("project_setup")}
    grouped: dict[str, list[DesktopFieldMeta]] = {}
    used: set[str] = set()
    for group_name, paths in PROJECT_SETUP_GROUP_RULES:
        fields = [fields_by_path[path] for path in paths if path in fields_by_path]
        if fields:
            grouped[group_name] = fields
            used.update(field.path for field in fields)

    stats_fields = [
        field
        for field in fields_by_path.values()
        if field.path not in used
        and (
            field.path.startswith("sync_stats.")
            or field.path.startswith("mapping.graph_feature_mapping.")
            or "stats" in field.path
            or "graph_feature_mapping" in field.path
        )
    ]
    if stats_fields:
        grouped["Stats / Mapping"] = stats_fields
        used.update(field.path for field in stats_fields)

    remaining = [field for field in fields_by_path.values() if field.path not in used]
    if remaining:
        grouped["Other"] = remaining
    return grouped


class DesktopPrototypeWindow:
    def __init__(self, *, registry: DesktopFieldRegistry, root_dir: Path) -> None:
        self.registry = registry
        self.root_dir = root_dir
        self.values: dict[str, Any] = {field.path: field.default for field in registry}
        self.field_widgets: dict[str, Any] = {}
        self.field_rows: list[tuple[DesktopFieldMeta, Any, Any, Any]] = []
        self.selected_field: DesktopFieldMeta | None = None
        self.values.setdefault("experiment.load_checkpoint", DEFAULT_CHECKPOINT_VALUE)
        if not self.values.get("experiment.load_checkpoint"):
            self.values["experiment.load_checkpoint"] = DEFAULT_CHECKPOINT_VALUE

        self.window = QMainWindow()
        self.window.setWindowTitle("BPM Experiment UI (PySide6)")
        self.window.resize(1720, 960)
        self.window.setStyleSheet(DARK_SLATE_THEME)

        self._build_ui()
        self._update_field_visibilities()

    def _build_ui(self) -> None:
        # Toolbar
        toolbar = QToolBar("Main")
        self.window.addToolBar(toolbar)

        # Show Hidden fields checkbox
        self.show_hidden_cb = QCheckBox("Show Hidden (RO)")
        self.show_hidden_cb.stateChanged.connect(self._update_field_visibilities)
        toolbar.addWidget(self.show_hidden_cb)

        # Preset Toggle Button
        preset_toggle_btn = QPushButton("Presets 📁")
        preset_toggle_btn.clicked.connect(self._toggle_presets_drawer)
        toolbar.addWidget(preset_toggle_btn)

        toolbar.addSeparator()

        # Save/Load UI Buttons
        self.run_btn = QPushButton("Run")
        self.run_btn.setObjectName("runBtn")
        self.run_btn.clicked.connect(self._on_run_clicked)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setObjectName("stopBtn")
        self.stop_btn.clicked.connect(self._on_stop_clicked)
        self.validate_btn = QPushButton("Validate")
        self.validate_btn.clicked.connect(self._on_validate_clicked)

        toolbar.addWidget(self.run_btn)
        toolbar.addWidget(self.stop_btn)
        toolbar.addWidget(self.validate_btn)

        self.search = QLineEdit()
        self.search.setPlaceholderText("Search fields...")
        self.search.textChanged.connect(self._apply_search_filter)
        toolbar.addWidget(self.search)

        focus_search = QAction("Search", self.window)
        focus_search.setShortcut(QKeySequence.Find)
        focus_search.triggered.connect(self.search.setFocus)
        self.window.addAction(focus_search)

        # Main splitter layout
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)

        self.splitter = QSplitter(Qt.Horizontal)

        # Navigation List
        self.nav = QListWidget()
        self.nav.addItems(["Project Setup", "Experiment Run", "Run Status / Logs", "Advanced"])
        self.nav.setMinimumWidth(DEFAULT_NAV_MIN_WIDTH)
        self.nav.setMaximumWidth(16777215)
        self.splitter.addWidget(self.nav)

        # Preset sliding drawer
        self.preset_drawer = PresetDrawer(
            presets_path=self.root_dir / "outputs" / "ui" / "experiment_ui_presets.json",
            on_save_callback=self._get_flat_values,
        )
        self.preset_drawer.preset_loaded.connect(self._apply_preset_values)
        self.preset_drawer.setMinimumWidth(0)
        self.preset_drawer.setMaximumWidth(320)
        self.preset_drawer.hide()
        self.splitter.addWidget(self.preset_drawer)

        # Stacked Pages
        self.stack = QStackedWidget()
        self.splitter.addWidget(self.stack)

        # Inspector Panel
        self.inspector = self._build_inspector()
        self.splitter.addWidget(self.inspector)

        # Ensure elements are not collapsible via splitter drag
        self.splitter.setChildrenCollapsible(False)
        self.splitter.setCollapsible(0, False)
        self.splitter.setCollapsible(1, False)
        self.splitter.setCollapsible(2, False)
        self.splitter.setCollapsible(3, False)
        
        self.splitter.setSizes([DEFAULT_NAV_INITIAL_WIDTH, 0, 1180, DEFAULT_INSPECTOR_WIDTH])
        main_layout.addWidget(self.splitter)
        self.window.setCentralWidget(main_widget)
        self.window.setStatusBar(QStatusBar())

        # Load Dynamic Form Pages
        self.stack.addWidget(self._build_registry_page("project_setup", "Project Setup"))
        self.stack.addWidget(self._build_registry_page("experiment_run", "Experiment Run"))
        
        # Run Monitor Integration (Page 2)
        self.run_monitor = RunMonitorWidget(self.root_dir)
        self.stack.addWidget(self.run_monitor)
        
        self.stack.addWidget(self._build_registry_page("advanced", "Advanced"))

        self.nav.currentRowChanged.connect(self.stack.setCurrentIndex)
        self.nav.setCurrentRow(1)

    def show(self) -> None:
        if START_MAXIMIZED:
            self.window.showMaximized()
        else:
            self.window.show()

    def _toggle_presets_drawer(self) -> None:
        self.preset_drawer.set_drawer_visible(not self.preset_drawer.drawer_visible)

    def _build_registry_page(self, ui_level: str, title: str) -> QWidget:
        from PySide6.QtWidgets import QScrollArea, QTabWidget, QGroupBox

        tab = QTabWidget()
        grouped = group_fields_for_page(self.registry, ui_level)
        
        if ui_level == "project_setup":
            spec = PROJECT_SETUP_TABS
        elif ui_level == "experiment_run":
            spec = EXPERIMENT_RUN_TABS
        else:
            spec = ADVANCED_TABS

        # Map grouped keys to the specified tabs
        tab_groups = {}
        for tab_name, groups_spec in spec.items():
            tab_groups[tab_name] = []
            for group_key, gb_title in groups_spec:
                if group_key in grouped:
                    tab_groups[tab_name].append((group_key, gb_title, grouped.pop(group_key)))

        # Fail-safe: add any remaining groups to the last tab as fallback
        if grouped:
            last_tab_name = list(spec.keys())[-1]
            if last_tab_name not in tab_groups:
                tab_groups[last_tab_name] = []
            for group_key, fields in list(grouped.items()):
                tab_groups[last_tab_name].append((group_key, f"Other: {group_key}", fields))

        for tab_title, gb_list in tab_groups.items():
            if not gb_list:
                continue
            page = QWidget()
            page_layout = QVBoxLayout(page)
            page_layout.setContentsMargins(8, 8, 8, 8)
            page_layout.setSpacing(10)

            for group_key, gb_title, fields in gb_list:
                group_box = QGroupBox(gb_title)
                group_box.setStyleSheet(
                    "QGroupBox { font-weight: bold; border: 1px solid #2d2d35; border-radius: 6px; margin-top: 10px; padding-top: 15px; } "
                    "QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; left: 10px; padding: 0 3px; }"
                )
                gb_layout = QVBoxLayout(group_box)
                gb_layout.setContentsMargins(10, 10, 10, 10)
                gb_layout.setSpacing(6)

                for field in fields:
                    row = QWidget()
                    row_layout = QHBoxLayout(row)
                    row_layout.setContentsMargins(0, 0, 0, 0)
                    label = self._make_field_label(field)
                    widget = create_field_widget(
                        field,
                        self.values.get(field.path, field.default),
                        self._set_value,
                        max_width=FIELD_WIDGET_MAX_WIDTH,
                    )
                    self.field_widgets[field.path] = widget
                    # Store 4-tuple: (field, row, widget, group_box)
                    self.field_rows.append((field, row, widget, group_box))
                    row_layout.addWidget(label)
                    row_layout.addWidget(widget)
                    if field.path == "experiment.load_checkpoint":
                        fill = QPushButton("Fill from experiment.name")
                        fill.clicked.connect(self._fill_checkpoint_from_experiment_name)
                        fill.setMinimumWidth(180)
                        row_layout.addWidget(fill)
                    row_layout.addStretch(1)
                    gb_layout.addWidget(row)
                page_layout.addWidget(group_box)

            page_layout.addStretch(1)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(page)
            tab.addTab(scroll, tab_title)

        if tab.count() == 0:
            empty = QWidget()
            layout = QVBoxLayout(empty)
            layout.addWidget(QLabel(f"No fields classified for {title}."))
            tab.addTab(empty, title)
        return tab

    def _build_inspector(self) -> QWidget:
        panel = QWidget()
        panel.setMinimumWidth(DEFAULT_INSPECTOR_WIDTH)
        layout = QVBoxLayout(panel)
        title = QLabel("Field Inspector")
        title.setStyleSheet("font-weight: 600;")
        layout.addWidget(title)
        layout.addWidget(QLabel("Path"))
        self.inspector_path = QLineEdit()
        self.inspector_path.setReadOnly(True)
        layout.addWidget(self.inspector_path)
        layout.addWidget(QLabel("Current value"))
        self.inspector_value = QLineEdit()
        self.inspector_value.setReadOnly(True)
        layout.addWidget(self.inspector_value)
        layout.addWidget(QLabel("Description / impact"))
        self.inspector_text = QTextEdit()
        self.inspector_text.setReadOnly(True)
        self.inspector_text.setAcceptRichText(False)
        self.inspector_text.setMinimumHeight(260)
        layout.addWidget(self.inspector_text, 1)
        copy_path = QPushButton("Copy path")
        copy_path.clicked.connect(self._copy_selected_path)
        copy_value = QPushButton("Copy value")
        copy_value.clicked.connect(self._copy_selected_value)
        layout.addWidget(copy_path)
        layout.addWidget(copy_value)
        layout.addStretch(1)
        return panel

    def _make_field_label(self, field: DesktopFieldMeta) -> QLabel:
        label = QLabel(field.path)
        label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        label.setContextMenuPolicy(Qt.CustomContextMenu)
        label.setMinimumWidth(240)
        label.setMaximumWidth(420)
        label.setToolTip("Select text or right-click to copy field path/value.")
        label.customContextMenuRequested.connect(
            lambda point, item=field, source=label: self._show_field_menu(item, source, point)
        )
        label.mousePressEvent = lambda event, item=field, original=label.mousePressEvent: (
            self._inspect_field(item),
            original(event),
        )
        return label

    def _inspect_field(self, field: DesktopFieldMeta) -> None:
        self.selected_field = field
        current_value = self.values.get(field.path, field.default)
        self.inspector_path.setText(field.path)
        self.inspector_value.setText("" if current_value is None else str(current_value))
        self.inspector_text.setPlainText(
            "\n".join(
                [
                    f"Label: {field.label}",
                    f"Level: {field.ui_level}",
                    f"Group: {field.ui_group}",
                    f"Default: {field.default}",
                    "",
                    "Description:",
                    field.description or "No description in catalog.",
                    "",
                    "Impact:",
                    field.affects or "No impact text in catalog.",
                    "",
                    f"Active when: {field.active_when or 'always'}",
                    f"Emit when: {field.emit_when or 'always'}",
                    f"Consumers: {', '.join(field.runtime_consumers) if field.runtime_consumers else 'unknown'}",
                ]
            )
        )

        # Trigger checkpoint autocomplete candidates
        if field.path == "experiment.load_checkpoint":
            exp_name = self.values.get("experiment.name", "")
            model_type = self.values.get("model.type", "")
            candidates = resolve_checkpoint_candidates(
                checkpoint_root=self.root_dir / "checkpoints",
                experiment_name=exp_name,
                model_type=model_type,
            )
            if candidates:
                cand_info = "\n\nResolved Checkpoint Candidates (Newest first):\n"
                for c in candidates[:5]:
                    cand_info += f"- {c['filename']} ({c['size_mb']:.2f} MB)\n"
                self.inspector_text.append(cand_info)

    def _set_value(self, path: str, value: Any) -> None:
        self.values[path] = value
        self._update_field_visibilities()

    def _fill_checkpoint_from_experiment_name(self) -> None:
        experiment_name = str(self.values.get("experiment.name") or "")
        model_type = str(self.values.get("model.type") or "")
        candidates = resolve_checkpoint_candidates(
            checkpoint_root=self.root_dir / "checkpoints",
            experiment_name=experiment_name,
            model_type=model_type,
        )
        if candidates:
            first_path = str(candidates[0]["path"])
            self.values["experiment.load_checkpoint"] = first_path
            self._set_widget_value("experiment.load_checkpoint", first_path)
            self.window.statusBar().showMessage(f"Checkpoint selected: {candidates[0]['filename']}")
        else:
            self.values["experiment.load_checkpoint"] = DEFAULT_CHECKPOINT_VALUE
            self._set_widget_value("experiment.load_checkpoint", DEFAULT_CHECKPOINT_VALUE)
            self.window.statusBar().showMessage("No checkpoint candidate found; keeping checkpoints/.")

    def _set_widget_value(self, path: str, value: Any) -> None:
        widget = self.field_widgets.get(path)
        if widget is None:
            return
        if hasattr(widget, "setChecked"):
            widget.setChecked(str(value).lower() == "true" or value is True)
        elif hasattr(widget, "setText"):
            widget.setText(str(value))
        elif hasattr(widget, "setCurrentText"):
            widget.setCurrentText(str(value))

    def _show_field_menu(self, field: DesktopFieldMeta, source: Any, point: Any) -> None:
        from PySide6.QtWidgets import QMenu

        menu = QMenu(source)
        copy_path = menu.addAction("Copy field path")
        copy_value = menu.addAction("Copy field value")
        action = menu.exec(source.mapToGlobal(point))
        clipboard = QApplication.clipboard()
        if action == copy_path:
            clipboard.setText(field.path)
        elif action == copy_value:
            clipboard.setText(str(self.values.get(field.path, field.default)))

    def _copy_selected_path(self) -> None:
        if self.selected_field is not None:
            QApplication.clipboard().setText(self.selected_field.path)
            self.window.statusBar().showMessage(f"Copied path: {self.selected_field.path}")

    def _copy_selected_value(self) -> None:
        if self.selected_field is not None:
            value = str(self.values.get(self.selected_field.path, self.selected_field.default))
            QApplication.clipboard().setText(value)
            self.window.statusBar().showMessage(f"Copied value for {self.selected_field.path}")

    def _update_field_visibilities(self) -> None:
        show_hidden = self.show_hidden_cb.isChecked()
        flat_values = self._get_flat_values()

        group_box_visibility = {}

        for field, row, widget, group_box in self.field_rows:
            active = self.registry.is_active(field.path, flat_values)
            visible = False
            if active:
                visible = True
                row.setVisible(True)
                widget.setEnabled(True)
            else:
                if show_hidden:
                    visible = True
                    row.setVisible(True)
                    widget.setEnabled(False)  # Read-only
                else:
                    row.setVisible(False)

            if visible:
                group_box_visibility[group_box] = True

        unique_group_boxes = {item[3] for item in self.field_rows if item[3] is not None}
        for gb in unique_group_boxes:
            gb.setVisible(gb in group_box_visibility)

    def _get_flat_values(self) -> dict[str, Any]:
        return {k: v for k, v in self.values.items()}

    def _apply_preset_values(self, values: dict[str, Any]) -> None:
        for k, v in values.items():
            if k in self.field_widgets:
                self.values[k] = v
                self._set_widget_value(k, v)
        self._update_field_visibilities()

    def _apply_search_filter(self, text: str) -> None:
        needle = text.strip().lower()
        group_box_visibility = {}
        for field, row, widget, group_box in self.field_rows:
            visible = not needle or needle in field.path.lower() or needle in field.label.lower() or needle in field.description.lower()
            row.setVisible(visible)
            if visible:
                group_box_visibility[group_box] = True

        unique_group_boxes = {item[3] for item in self.field_rows if item[3] is not None}
        for gb in unique_group_boxes:
            gb.setVisible(gb in group_box_visibility)

    def _on_validate_clicked(self) -> None:
        mode = self.values.get("experiment.mode", "")
        stats_policy = self.values.get("experiment.stats_time_policy", "")
        lr_strategy = self.values.get("training.learning_strategy", "")
        model_type = self.values.get("model.type", "")

        # 1. Temporal Drift Data Leakage Warning
        if mode in ["eval_drift", "eval_cross_dataset"] and stats_policy == "latest":
            QMessageBox.warning(
                self.window,
                "Validation Warning",
                "Using stats_time_policy='latest' with drift eval causes data leakage. Please use 'strict_asof' for scientific runs.",
            )
            return

        # 2. Topology Conditioned model constraint checks
        if lr_strategy == "topology_conditioned" and model_type != "EOPKGTopologyConditioned":
            QMessageBox.critical(
                self.window,
                "Validation Failed",
                f"training.learning_strategy='topology_conditioned' requires model.type='EOPKGTopologyConditioned', got '{model_type}'.",
            )
            return

        QMessageBox.information(self.window, "Validation Successful", "Configuration constraints validation passed.")

    def _on_run_clicked(self) -> None:
        # Write active fields to formatted YAML config
        flat_values = self._get_flat_values()
        active_config = {}
        for path, val in flat_values.items():
            if self.registry.is_active(path, flat_values):
                parts = path.split(".")
                curr = active_config
                for part in parts[:-1]:
                    curr = curr.setdefault(part, {})
                curr[parts[-1]] = val

        project_name = flat_values.get("experiment.project", "default_project")
        run_name = flat_values.get("experiment.name", "run")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        out_dir = self.root_dir / "outputs" / "ui" / "generated_configs" / project_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{run_name}_{timestamp}.yaml"

        with open(out_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(active_config, f, default_flow_style=False)

        # Switch to Logs tab
        self.nav.setCurrentRow(2)

        # Trigger QProcess run monitor
        self.run_monitor.start_run(flat_values.get("experiment.mode", "train"), out_path)

    def _on_stop_clicked(self) -> None:
        self.run_monitor.stop_run()
