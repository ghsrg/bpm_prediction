from __future__ import annotations

import sys
from pathlib import Path

from .field_registry import DesktopFieldRegistry


ROOT_DIR = Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError as exc:
        print("PySide6 is required for `main.py ui`. Install with: .\\.venv-modern\\Scripts\\python.exe -m pip install \"PySide6>=6.7\"", file=sys.stderr)
        return 2

    from .main_window import DesktopPrototypeWindow

    app = QApplication(argv or sys.argv[1:])
    registry = DesktopFieldRegistry.load(
        catalog_path=ROOT_DIR / "configs" / "ui" / "config_catalog.yaml",
        audit_matrix_path=ROOT_DIR / "configs" / "ui" / "desktop_ui_field_dependency_matrix.csv",
    )
    window = DesktopPrototypeWindow(registry=registry, root_dir=ROOT_DIR)
    window.show()
    return app.exec()

