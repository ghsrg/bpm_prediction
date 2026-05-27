# tools/desktop_ui/run_monitor.py
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from PySide6.QtCore import QProcess, QTimer, Qt
from PySide6.QtGui import QColor, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

RUN_STAGE_ORDER = [
    "run.pipeline",
    "prepare_data",
    "prepare.read_events",
    "prepare.feature_encoder",
    "build_graph.train",
    "build_graph.validation",
    "build_graph.test",
    "trainer.dataloaders",
    "trainer.dry_run",
    "train.epochs",
    "train.batches",
    "validation.batches",
    "test.eval",
    "test.batches",
    "eval_drift.one_pass_inference",
    "eval_drift.windows",
]

MODE_STAGES: dict[str, list[str]] = {
    "train": ["run.pipeline", "prepare_data", "build_graph.train", "build_graph.validation", "trainer.dataloaders", "trainer.dry_run", "train.epochs", "test.eval"],
    "eval_drift": ["run.pipeline", "prepare_data", "build_graph.test", "eval_drift.one_pass_inference", "eval_drift.windows"],
    "sync-stats": ["run.pipeline", "prepare_data"],
    "sync-topology": ["run.pipeline", "prepare_data"],
    "sync-stats-backfill": ["run.pipeline", "prepare_data"],
}


class RunMonitorWidget(QWidget):
    def __init__(self, root_dir: Path) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.process: QProcess | None = None
        self.start_time: float = 0.0
        self.active_stages: list[str] = []
        self.current_stage: str | None = None
        self.stage_progress: dict[str, float] = {}

        self._init_ui()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._update_timer_labels)

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        # Header Info
        info_layout = QHBoxLayout()
        self.elapsed_label = QLabel("Elapsed: 0s")
        self.eta_label = QLabel("ETA: -")
        self.status_label = QLabel("Status: Idle")
        self.status_label.setStyleSheet("font-weight: bold;")
        info_layout.addWidget(self.status_label)
        info_layout.addStretch()
        info_layout.addWidget(self.elapsed_label)
        info_layout.addWidget(self.eta_label)
        layout.addLayout(info_layout)

        # Dynamic Stage Progress Bars Container
        self.progress_container = QWidget()
        self.progress_layout = QVBoxLayout(self.progress_container)
        self.progress_layout.setContentsMargins(0, 0, 0, 0)
        self.progress_layout.setSpacing(4)
        layout.addWidget(self.progress_container)

        self.progress_bars: dict[str, QProgressBar] = {}
        self.progress_labels: dict[str, QLabel] = {}

        # Log Controls
        log_ctrl_layout = QHBoxLayout()
        self.freeze_checkbox = QCheckBox("Freeze Logs")
        self.clear_btn = QPushButton("Clear Console")
        self.clear_btn.clicked.connect(self.clear_logs)
        log_ctrl_layout.addWidget(self.freeze_checkbox)
        log_ctrl_layout.addStretch()
        log_ctrl_layout.addWidget(self.clear_btn)
        layout.addLayout(log_ctrl_layout)

        # Scrolled Log Console
        self.console = QPlainTextEdit()
        self.console.setReadOnly(True)
        self.console.setStyleSheet("background-color: #121216; font-family: Consolas, monospace; font-size: 12px;")
        layout.addWidget(self.console, 1)

    def clear_logs(self) -> None:
        self.console.clear()

    def start_run(self, mode: str, config_path: Path) -> None:
        self.clear_logs()
        self.stage_progress.clear()
        self.current_stage = None

        # Build dynamic progress bars for active mode stages
        for child in self.progress_container.findChildren(QWidget):
            child.deleteLater()
        self.progress_bars.clear()
        self.progress_labels.clear()

        self.active_stages = MODE_STAGES.get(mode, ["run.pipeline"])
        for stage in self.active_stages:
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            lbl = QLabel(stage)
            lbl.setMinimumWidth(180)
            pbar = QProgressBar()
            pbar.setRange(0, 100)
            pbar.setValue(0)
            row_layout.addWidget(lbl)
            row_layout.addWidget(pbar)
            self.progress_layout.addWidget(row)
            self.progress_bars[stage] = pbar
            self.progress_labels[stage] = lbl

        self.process = QProcess(self)
        self.process.setWorkingDirectory(str(self.root_dir))
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._handle_output)
        self.process.finished.connect(self._handle_finished)

        # Set progress events env
        env = self.process.processEnvironment()
        env.insert("BPM_PROGRESS_EVENTS", "1")
        self.process.setProcessEnvironment(env)

        venv_python = self.root_dir / ".venv-modern" / "Scripts" / "python.exe"
        args = ["main.py", "--config", str(config_path)]

        self.status_label.setText("Status: Running")
        self.status_label.setStyleSheet("color: #00adb5; font-weight: bold;")
        self.start_time = time.time()
        self.timer.start(1000)

        self.process.start(str(venv_python), args)

    def stop_run(self) -> None:
        if self.process and self.process.state() == QProcess.Running:
            self.process.terminate()
            self.status_label.setText("Status: Stopping")
            self.status_label.setStyleSheet("color: #ff9800; font-weight: bold;")

    def _handle_output(self) -> None:
        if not self.process:
            return
        data = self.process.readAllStandardOutput().data().decode("utf-8", errors="ignore")
        for line in data.splitlines():
            line_str = line.strip()
            if line_str.startswith("__BPM_PROGRESS__"):
                self._parse_progress_event(line_str[len("__BPM_PROGRESS__"):])
            else:
                self._append_log(line)

    def _parse_progress_event(self, json_str: str) -> None:
        try:
            event = json.loads(json_str)
            stage = event.get("stage")
            status = event.get("status")
            percent = event.get("percent")

            if stage in self.progress_bars:
                pbar = self.progress_bars[stage]
                if status == "done":
                    pbar.setValue(100)
                elif percent is not None:
                    pbar.setValue(int(percent))
                self.current_stage = stage
        except Exception:
            pass

    def _append_log(self, text: str) -> None:
        cursor = self.console.textCursor()
        cursor.movePosition(QTextCursor.End)

        # Style line color based on tag
        fmt = QTextCharFormat()
        if "ERROR" in text or "Fail" in text:
            fmt.setForeground(QColor("#ef5350"))
        elif "WARNING" in text or "Warning" in text:
            fmt.setForeground(QColor("#ffb74d"))
        else:
            fmt.setForeground(QColor("#e3e3e6"))

        cursor.setCharFormat(fmt)
        cursor.insertText(text + "\n")

        if not self.freeze_checkbox.isChecked():
            self.console.moveCursor(QTextCursor.End)

    def _handle_finished(self, exit_code: int) -> None:
        self.timer.stop()
        self.elapsed_label.setText(f"Elapsed: {self._format_duration(time.time() - self.start_time)}")
        self.eta_label.setText("ETA: -")

        if exit_code == 0:
            self.status_label.setText("Status: Success")
            self.status_label.setStyleSheet("color: #4caf50; font-weight: bold;")
            for pbar in self.progress_bars.values():
                pbar.setValue(100)
        else:
            self.status_label.setText(f"Status: Failed ({exit_code})")
            self.status_label.setStyleSheet("color: #f44336; font-weight: bold;")

    def _update_timer_labels(self) -> None:
        elapsed = time.time() - self.start_time
        self.elapsed_label.setText(f"Elapsed: {self._format_duration(elapsed)}")

        # Estimate ETA
        completed_bars = 0
        total_p = 0.0
        for pbar in self.progress_bars.values():
            total_p += pbar.value()
        avg_percent = total_p / len(self.progress_bars) if self.progress_bars else 0

        if avg_percent > 5:
            total_est = elapsed / (avg_percent / 100.0)
            eta = max(0.0, total_est - elapsed)
            self.eta_label.setText(f"ETA: {self._format_duration(eta)}")
        else:
            self.eta_label.setText("ETA: -")

    def _format_duration(self, seconds: float) -> str:
        s = int(seconds)
        if s < 60:
            return f"{s}s"
        m = s // 60
        s = s % 60
        if m < 60:
            return f"{m:02d}:{s:02d}"
        h = m // 60
        m = m % 60
        return f"{h:02d}:{m:02d}:{s:02d}"
