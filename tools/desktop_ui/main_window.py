from __future__ import annotations

from pathlib import Path
from typing import Any

from .checkpoint_resolver import DEFAULT_CHECKPOINT_VALUE, resolve_checkpoint_candidates
from .field_registry import DesktopFieldMeta, DesktopFieldRegistry
from .field_widgets import create_field_widget


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
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QAction, QKeySequence
        from PySide6.QtWidgets import (
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QListWidget,
            QMainWindow,
            QPushButton,
            QSplitter,
            QStackedWidget,
            QStatusBar,
            QToolBar,
            QVBoxLayout,
            QWidget,
        )

        self.registry = registry
        self.root_dir = root_dir
        self.values: dict[str, Any] = {field.path: field.default for field in registry}
        self.field_widgets: dict[str, Any] = {}
        self.field_rows: list[tuple[DesktopFieldMeta, Any, Any]] = []
        self.selected_field: DesktopFieldMeta | None = None
        self.values.setdefault("experiment.load_checkpoint", DEFAULT_CHECKPOINT_VALUE)
        if not self.values.get("experiment.load_checkpoint"):
            self.values["experiment.load_checkpoint"] = DEFAULT_CHECKPOINT_VALUE

        self.window = QMainWindow()
        self.window.setWindowTitle("BPM Experiment UI Prototype")
        self.window.resize(1720, 960)

        toolbar = QToolBar("Main")
        self.window.addToolBar(toolbar)
        for text in ["Save", "Load", "Validate", "Build YAML", "Run", "Stop"]:
            action = QAction(text, self.window)
            toolbar.addAction(action)

        self.search = QLineEdit()
        self.search.setPlaceholderText("Search fields")
        self.search.textChanged.connect(self._apply_search_filter)
        toolbar.addWidget(self.search)

        focus_search = QAction("Search", self.window)
        focus_search.setShortcut(QKeySequence.Find)
        focus_search.triggered.connect(self.search.setFocus)
        self.window.addAction(focus_search)

        main = QWidget()
        main_layout = QHBoxLayout(main)
        self.nav = QListWidget()
        self.nav.addItems(["Project Setup", "Experiment Run", "Run Status / Logs", "Advanced"])
        self.nav.setMinimumWidth(DEFAULT_NAV_MIN_WIDTH)
        self.nav.setMaximumWidth(16777215)
        self.stack = QStackedWidget()
        self.inspector = self._build_inspector()

        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.addWidget(self.nav)
        self.splitter.addWidget(self.stack)
        self.splitter.addWidget(self.inspector)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.setCollapsible(0, False)
        self.splitter.setCollapsible(1, False)
        self.splitter.setCollapsible(2, False)
        self.splitter.setSizes(DEFAULT_SPLITTER_SIZES)
        main_layout.addWidget(self.splitter)
        self.window.setCentralWidget(main)
        self.window.setStatusBar(QStatusBar())
        self.window.statusBar().showMessage("Prototype mode: layout and field registry validation only.")

        self.stack.addWidget(self._build_registry_page("project_setup", "Project Setup"))
        self.stack.addWidget(self._build_registry_page("experiment_run", "Experiment Run", compact=True))
        self.stack.addWidget(self._build_status_page())
        self.stack.addWidget(self._build_registry_page("advanced", "Advanced"))
        self.nav.currentRowChanged.connect(self.stack.setCurrentIndex)
        self.nav.setCurrentRow(1)

    def show(self) -> None:
        if START_MAXIMIZED:
            self.window.showMaximized()
        else:
            self.window.show()

    def _build_registry_page(self, ui_level: str, title: str, compact: bool = False):
        from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QScrollArea, QTabWidget, QVBoxLayout, QWidget

        tab = QTabWidget()
        grouped = group_fields_for_page(self.registry, ui_level)
        for group_name, fields in grouped.items():
            page = QWidget()
            page_layout = QVBoxLayout(page)
            page_layout.setContentsMargins(8, 8, 8, 8)
            page_layout.setSpacing(6)
            for field in fields:
                row = QWidget()
                row_layout = QHBoxLayout(row)
                row_layout.setContentsMargins(0, 0, 0, 0)
                row_layout.setSpacing(8)
                label = self._make_field_label(field)
                widget = create_field_widget(
                    field,
                    self.values.get(field.path, field.default),
                    self._set_value,
                    max_width=FIELD_WIDGET_MAX_WIDTH,
                )
                self.field_widgets[field.path] = widget
                self.field_rows.append((field, row, widget))
                row_layout.addWidget(label)
                row_layout.addWidget(widget)
                if field.path == "experiment.load_checkpoint":
                    fill = QPushButton("Fill from experiment.name")
                    fill.clicked.connect(self._fill_checkpoint_from_experiment_name)
                    fill.setMinimumWidth(180)
                    row_layout.addWidget(fill)
                row_layout.addStretch(1)
                page_layout.addWidget(row)
            page_layout.addStretch(1)
            wrapper = QWidget()
            wrapper_layout = QVBoxLayout(wrapper)
            wrapper_layout.addWidget(page)
            wrapper_layout.addStretch(1)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(wrapper)
            tab.addTab(scroll, group_name)
        if tab.count() == 0:
            empty = QWidget()
            layout = QVBoxLayout(empty)
            layout.addWidget(QLabel(f"No fields classified for {title}."))
            tab.addTab(empty, title)
        return tab

    def _build_inspector(self):
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QLabel, QLineEdit, QPushButton, QTextEdit, QVBoxLayout, QWidget

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

    def _make_field_label(self, field: DesktopFieldMeta):
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QLabel

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

    def _build_status_page(self):
        from PySide6.QtWidgets import QCheckBox, QComboBox, QLabel, QPushButton, QTextEdit, QVBoxLayout, QWidget

        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(QLabel("Run Status / Logs prototype"))
        layout.addWidget(QLabel("Mode-aware progress bars and process control are planned for Phase 2."))
        pause = QCheckBox("Pause autoscroll")
        pause.setChecked(False)
        layout.addWidget(pause)
        filters = QComboBox()
        filters.addItems(["All", "Progress", "Warnings", "Errors"])
        layout.addWidget(filters)
        log = QTextEdit()
        log.setPlainText("Execution log preview. Full run integration is intentionally not wired in Phase 1.")
        layout.addWidget(log)
        layout.addWidget(QPushButton("Copy All"))
        return page

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

    def _set_value(self, path: str, value: Any) -> None:
        self.values[path] = value

    def _fill_checkpoint_from_experiment_name(self) -> None:
        experiment_name = str(self.values.get("experiment.name") or "")
        candidates = resolve_checkpoint_candidates(
            checkpoint_root=self.root_dir / "checkpoints",
            experiment_name=experiment_name,
        )
        if candidates:
            self.values["experiment.load_checkpoint"] = str(candidates[0])
            self._set_widget_value("experiment.load_checkpoint", str(candidates[0]))
            self.window.statusBar().showMessage(f"Checkpoint selected: {candidates[0]}")
        else:
            self.values["experiment.load_checkpoint"] = DEFAULT_CHECKPOINT_VALUE
            self._set_widget_value("experiment.load_checkpoint", DEFAULT_CHECKPOINT_VALUE)
            self.window.statusBar().showMessage("No checkpoint candidate found; keeping checkpoints/.")

    def _set_widget_value(self, path: str, value: Any) -> None:
        widget = self.field_widgets.get(path)
        if widget is None:
            return
        if hasattr(widget, "setText"):
            widget.setText(str(value))
        elif hasattr(widget, "setCurrentText"):
            widget.setCurrentText(str(value))

    def _show_field_menu(self, field: DesktopFieldMeta, source: Any, point: Any) -> None:
        from PySide6.QtWidgets import QApplication, QMenu

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
        from PySide6.QtWidgets import QApplication

        if self.selected_field is not None:
            QApplication.clipboard().setText(self.selected_field.path)
            self.window.statusBar().showMessage(f"Copied path: {self.selected_field.path}")

    def _copy_selected_value(self) -> None:
        from PySide6.QtWidgets import QApplication

        if self.selected_field is not None:
            value = str(self.values.get(self.selected_field.path, self.selected_field.default))
            QApplication.clipboard().setText(value)
            self.window.statusBar().showMessage(f"Copied value for {self.selected_field.path}")

    def _apply_search_filter(self, text: str) -> None:
        needle = text.strip().lower()
        for field, row, widget in self.field_rows:
            visible = not needle or needle in field.path.lower() or needle in field.label.lower() or needle in field.description.lower()
            row.setVisible(visible)
