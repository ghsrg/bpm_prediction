"""Launcher for Streamlit-based web UI experiment configurator."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
APP_PATH = ROOT_DIR / "tools" / "web_experiment_ui_app.py"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Streamlit web UI for BPM experiment configuration.")
    parser.add_argument("--config", default="", help="Optional config path loaded on first UI render.")
    args = parser.parse_args(argv)

    try:
        import streamlit  # noqa: F401
    except Exception:
        print("streamlit is not installed. Install dependencies and retry: pip install -r requirements.txt")
        return 2

    cmd = [sys.executable, "-m", "streamlit", "run", str(APP_PATH)]
    if str(args.config).strip():
        cmd.extend(["--", "--config", str(args.config).strip()])
    return subprocess.call(cmd, cwd=str(ROOT_DIR))


if __name__ == "__main__":
    raise SystemExit(main())
