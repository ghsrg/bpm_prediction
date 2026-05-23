from __future__ import annotations

from pathlib import Path


def test_main_exposes_parallel_ui_command_without_replacing_legacy_experiment_ui():
    source = Path("main.py").read_text(encoding="utf-8")

    assert 'sys.argv[1] == "ui"' in source
    assert 'sys.argv[1] == "experiment-ui"' in source
    assert "desktop_ui_main" in source
    assert "experiment_ui_main" in source

