# tools/desktop_ui/styles.py
from __future__ import annotations

DARK_SLATE_THEME = """
QMainWindow {
    background-color: #1e1e24;
}
QWidget {
    background-color: #1e1e24;
    color: #e3e3e6;
    font-family: 'Segoe UI', system-ui, sans-serif;
    font-size: 13px;
}
QToolBar {
    background-color: #121216;
    border-bottom: 1px solid #2d2d35;
    spacing: 10px;
    padding: 4px;
}
QListWidget {
    background-color: #121216;
    border-right: 1px solid #2d2d35;
    outline: 0;
}
QListWidget::item {
    padding: 10px;
    border-bottom: 1px solid #25252d;
}
QListWidget::item:selected {
    background-color: #00adb5;
    color: #ffffff;
}
QTabWidget::pane {
    border: 1px solid #2d2d35;
    background-color: #1a1a20;
}
QTabBar::tab {
    background-color: #121216;
    padding: 8px 16px;
    border: 1px solid #2d2d35;
    border-bottom: none;
    margin-right: 2px;
}
QTabBar::tab:selected {
    background-color: #1a1a20;
    border-bottom: 2px solid #00adb5;
}
QLineEdit, QTextEdit, QComboBox, QSpinBox {
    background-color: #25252d;
    border: 1px solid #3a3a42;
    border-radius: 4px;
    padding: 6px;
    color: #ffffff;
}
QLineEdit:focus, QTextEdit:focus, QComboBox:focus {
    border: 1px solid #00adb5;
}
QPushButton {
    background-color: #2a2a35;
    border: 1px solid #3a3a42;
    border-radius: 4px;
    padding: 6px 12px;
    font-weight: 600;
}
QPushButton:hover {
    background-color: #353542;
}
QPushButton#runBtn {
    background-color: #2e7d32;
    color: white;
}
QPushButton#runBtn:hover {
    background-color: #388e3c;
}
QPushButton#stopBtn {
    background-color: #c62828;
    color: white;
}
QPushButton#stopBtn:hover {
    background-color: #d32f2f;
}
QTreeWidget {
    background-color: #121216;
    border: 1px solid #2d2d35;
}
QTreeWidget::item {
    padding: 6px;
}
QTreeWidget::item:selected {
    background-color: #00adb5;
    color: white;
}
QScrollBar:vertical {
    border: none;
    background: #121216;
    width: 10px;
}
QScrollBar::handle:vertical {
    background: #3a3a42;
    border-radius: 5px;
}
"""
