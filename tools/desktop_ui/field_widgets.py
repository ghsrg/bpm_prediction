from __future__ import annotations

from typing import Any, Callable

from .field_registry import DesktopFieldMeta


def create_field_widget(
    field: DesktopFieldMeta,
    value: Any,
    on_change: Callable[[str, Any], None],
    *,
    max_width: int | None = None,
):
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QCheckBox, QComboBox, QLineEdit, QTextEdit

    if field.enum:
        widget = QComboBox()
        widget.setEditable(True)
        widget.addItems(list(field.enum))
        widget.setCurrentText(str(value) if value is not None else "")
        if max_width:
            widget.setMaximumWidth(max_width)
        widget.currentTextChanged.connect(lambda text: on_change(field.path, text))
        return widget

    if isinstance(field.default, bool) or str(field.default).lower() in {"true", "false"}:
        widget = QCheckBox()
        widget.setChecked(str(value).lower() == "true")
        widget.stateChanged.connect(lambda state: on_change(field.path, state == Qt.Checked))
        return widget

    if "yaml" in field.ui_group.lower() or field.path.endswith(("features", "policies")):
        widget = QTextEdit()
        widget.setPlainText("" if value is None else str(value))
        widget.setMaximumHeight(140)
        if max_width:
            widget.setMaximumWidth(max_width)
        widget.textChanged.connect(lambda: on_change(field.path, widget.toPlainText()))
        return widget

    widget = QLineEdit()
    widget.setText("" if value is None else str(value))
    widget.setMinimumWidth(260)
    if max_width:
        widget.setMaximumWidth(max_width)
    widget.textChanged.connect(lambda text: on_change(field.path, text))
    return widget
