"""Desktop UI wrapper for configuring and running BPM experiments.

Goals:
1. Use base config as source of truth.
2. Expose frequently used experiment/data/mapping params as form fields.
3. Keep `features`, `policies`, `graph_feature_mapping` as YAML text blocks.
4. Run CLI with generated temporary config override.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
import queue
import shlex
import subprocess
import sys
import tempfile
import threading
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

import yaml
import psutil

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.infrastructure.config.yaml_loader import load_yaml_with_includes

OUTPUT_DIR = ROOT_DIR / "outputs" / "ui"
STATE_PATH = OUTPUT_DIR / "experiment_ui_state.json"
PRESETS_PATH = OUTPUT_DIR / "experiment_ui_presets.json"
CATALOG_PATH = ROOT_DIR / "configs" / "ui" / "config_catalog.yaml"
PROGRESS_EVENT_PREFIX = "__BPM_PROGRESS__"

RUN_STAGE_WEIGHTS: Dict[str, float] = {
    "run.pipeline": 0.02,
    "prepare_data": 0.10,
    "prepare.read_events": 0.08,
    "prepare.feature_encoder": 0.07,
    "build_graph.train": 0.22,
    "build_graph.validation": 0.10,
    "build_graph.test": 0.08,
    "trainer.dataloaders": 0.03,
    "trainer.dry_run": 0.05,
    "train.epochs": 0.15,
    "train.batches": 0.04,
    "validation.batches": 0.03,
    "test.eval": 0.03,
    "test.batches": 0.02,
    "eval_drift.windows": 0.04,
}

RUN_STAGE_ORDER: List[str] = [
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
    "eval_drift.windows",
]

STAGE_TITLE_UA: Dict[str, str] = {
    "run.pipeline": "Пайплайн",
    "prepare_data": "Підготовка даних",
    "prepare.read_events": "Читання подій",
    "prepare.feature_encoder": "Побудова енкодера ознак",
    "build_graph.train": "Побудова train-графів",
    "build_graph.validation": "Побудова validation-графів",
    "build_graph.test": "Побудова test-графів",
    "trainer.dataloaders": "Підготовка DataLoader",
    "trainer.dry_run": "Dry run моделі",
    "train.epochs": "Епохи тренування",
    "train.batches": "Train batch-и",
    "validation.batches": "Validation batch-и",
    "test.eval": "Підсумкове тестування",
    "test.batches": "Test batch-и",
    "eval_drift.windows": "Оцінка drift-вікон",
}

FIELD_HINTS: Dict[str, str] = {
    "data.dataset_name": "Стабільна назва процесу/датасету для репозиторію знань і run-profile.",
    "data.dataset_label": "Лейбл датасету для трекінгу/логів.",
    "data.log_path": "Шлях до XES (обов'язково для adapter=xes).",
    "mapping.adapter": "Вибір інжест-адаптера: xes або camunda.",
    "mapping.knowledge_graph.backend": "Сховище структури/статистики: in_memory/file/neo4j.",
    "mapping.knowledge_graph.strict_load": "Fail-fast якщо структура відсутня.",
    "mapping.camunda_adapter.runtime.runtime_source": "Джерело runtime: files або sql.",
    "mapping.camunda_adapter.runtime.export_dir": "Каталог експортів Camunda (files runtime).",
    "mapping.camunda_adapter.runtime.sql_dir": "Каталог SQL-шаблонів Camunda runtime.",
    "mapping.camunda_adapter.structure.source": "Джерело структури: bpmn або logs fallback.",
    "mapping.camunda_adapter.structure.bpmn_source": "Звідки читати BPMN: files або sql.",
    "mapping.camunda_adapter.structure.files.bpmn_dir": "Каталог BPMN XML.",
    "mapping.camunda_adapter.structure.files.catalog_file": "CSV каталог process definitions.",
    "mapping.xes_adapter.activity_key": "Ключ активності в XES.",
    "mapping.xes_adapter.start_transitions": "Lifecycle значення, які трактуються як старт активності (наприклад start, assign).",
    "mapping.xes_adapter.version_key": "Ключ версії процесу в XES.",
    "mapping.xes_adapter.use_classifier": "Читати classifier XES перед activity_key fallback.",
    "experiment.cache_policy": "Політика кешування побудови графів: off/dto/full.",
    "experiment.graph_dataset_cache_policy": "Кеш готових train/val/test граф-датасетів: off/read/write/full.",
    "experiment.graph_dataset_cache_dir": "Директорія disk-cache для готових граф-датасетів.",
    "experiment.graph_dataset_disk_spill_enabled": "Вмикає побудову графів з поетапним скиданням шард на диск (менше RAM).",
    "experiment.graph_dataset_shard_size": "Кількість графів у одному disk-shard (баланс IO vs RAM).",
    "experiment.max_ram_gb": "Soft-ліміт RAM для build_graph; при перевищенні буфер графів примусово flush-иться на диск.",
}

DETAILED_HINTS_UA: Dict[str, str] = {
    "experiment.mode": "Режим запуску пайплайна. Від нього залежить, яку команду виконає UI: train/eval/sync-topology/sync-stats/sync-stats-backfill.",
    "experiment.project": "Група експериментів у трекінгу (MLflow/інший backend). Допомагає тримати серії запусків разом.",
    "experiment.name": "Назва конкретного запуску в межах `experiment.project`.",
    "training.delta": "Мінімальне покращення `val_loss`, яке вважається значущим. Якщо покращення менше за `delta` протягом `training.patience` епох, спрацьовує early stopping.",
    "training.patience": "Кількість епох без значущого покращення `val_loss`, після яких тренування зупиняється (early stopping).",
    "training.learning_rate": "Крок оптимізатора. Завеликий `learning_rate` може робити навчання нестабільним, замалий - дуже повільним.",
    "training.batch_size": "Розмір мінібатчу. Впливає на швидкість, пам'ять та стабільність градієнтів.",
    "training.epochs": "Максимальна кількість епох навчання.",
    "training.retrain": "Якщо `true`, навчання стартує з нуля; якщо `false`, система може намагатися продовжити з checkpoint.",
    "training.device": "Пристрій виконання: `cpu` або `cuda` (якщо доступно). Впливає на швидкість навчання.",
    "training.class_weight_cap": "Верхня межа ваг класів у loss, щоб рідкісні класи не давали нестабільні градієнти.",
    "training.loss_function": "Функція втрат для next-activity prediction (наприклад cross_entropy).",
    "training.ece_bins": "Кількість бінів для метрики калібрування ECE.",
    "experiment.split_ratio": "Трійка часток `[train, val, test]` для розбиття даних. Рекомендовано, щоб сума дорівнювала 1.0.",
    "experiment.train_ratio": "Частка train для каскадної temporal-підготовки (до внутрішнього split train/val/test).",
    "experiment.fraction": "Яку частину доступних трас взяти в поточний запуск (0..1). Зручно для швидких прогонів.",
    "experiment.stats_time_policy": "Політика вибору snapshot статистики: `latest` (останній) або `strict_asof` (на момент події, без витоку майбутнього).",
    "experiment.on_missing_asof_snapshot": "Що робити, якщо для `strict_asof` немає snapshot на потрібний час: `disable_stats`, `use_base` або `raise`.",
    "experiment.cache_policy": "Керує кешуванням під час build_graph: `off` без кешу, `dto` кешує DTO-читання, `full` кешує DTO і скомпільовані структурні тензори (рекомендовано для великих прогонів).",
    "experiment.graph_dataset_cache_policy": "Керує disk-cache для вже побудованих train/validation/test графів: `off` - вимкнено, `read` - лише читати, `write` - лише записувати новий кеш, `full` - і читати, і записувати.",
    "experiment.graph_dataset_cache_dir": "Шлях до директорії cache готових граф-датасетів. Використовуйте швидкий SSD і періодично очищайте старі записи командою `python main.py cache-clean`.",
    "experiment.graph_dataset_disk_spill_enabled": "Якщо `true`, build_graph пише дані в shard-файли на диск під час побудови, щоб не накопичувати весь train/val/test в RAM.",
    "experiment.graph_dataset_shard_size": "Розмір shard (у кількості графів). Менше значення знижує RAM-пік, але збільшує кількість файлів і disk I/O.",
    "experiment.max_ram_gb": "Soft-ліміт RSS (ГБ) для build_graph. Коли процес підходить до ліміту, поточний буфер графів форсовано зберігається на диск.",
    "mapping.adapter": "Джерело подій: `camunda` (runtime/BPMN) або `xes` (лог-файл XES).",
    "mapping.knowledge_graph.backend": "Де зберігається структура/статистика EOPKG: `neo4j`, `file` або `in_memory`.",
    "mapping.knowledge_graph.strict_load": "Якщо `true`, запуск завершується помилкою, коли структура відсутня/неконсистентна.",
    "mapping.camunda_adapter.runtime.runtime_source": "Звідки брати runtime-події Camunda: з експорт-файлів (`files`) або з БД (`mssql`).",
    "mapping.camunda_adapter.structure.bpmn_source": "Звідки брати BPMN-структуру: з файлів (`files`) або з БД (`mssql`).",
    "mapping.camunda_adapter.structure.files.bpmn_dir": "Каталог з BPMN XML для побудови структури процесу.",
    "mapping.camunda_adapter.structure.files.catalog_file": "CSV-каталог process definitions (version/proc_def_id/інше), який використовує structure sync.",
    "mapping.xes_adapter.use_classifier": "Якщо `true`, adapter спочатку пробує XES classifier; якщо `false`, читає активність напряму з `activity_key`.",
    "mapping.xes_adapter.activity_key": "Назва XES-атрибуту, з якого брати activity id/label, якщо classifier не використовується.",
    "mapping.xes_adapter.start_transitions": "Список lifecycle-значень, які вважати початком активності для pairing start->complete (наприклад `start`, `assign`).",
    "mapping.xes_adapter.version_key": "Назва XES-атрибуту версії процесу (для сценаріїв дрейфу по версіях).",
    "sync_stats.stats_time_policy": "Політика часу під час sync-stats. Для коректного temporal-аналізу зазвичай використовують `strict_asof`.",
    "sync_stats.process_scope_policy": "Обмеження історії подій при розрахунку статистики (наприклад до target version).",
}


@dataclass
class PoolFieldMeta:
    path: str
    section: str
    values: set[str]


@dataclass
class CatalogFieldMeta:
    path: str
    section: str
    label: str
    description: str
    affects: str
    default: Any
    enum: List[str]
    required_in_modes: List[str]
    required_when: Dict[str, Any]
    ui_tab: str
    ui_group: str
    ui_priority: int
    ui_order: int


def _merge_catalog_section_fields(
    *,
    base_fields: Dict[str, Any],
    pool_meta: Dict[str, PoolFieldMeta],
    catalog: Dict[str, CatalogFieldMeta],
    section: str,
) -> Dict[str, Any]:
    """Merge config, scanned pool, and catalog-only fields for one UI section."""
    merged: Dict[str, Any] = dict(base_fields)
    for path, meta in pool_meta.items():
        if meta.section == section and path not in merged:
            merged[path] = ""
    for path, meta in catalog.items():
        if meta.section == section and path not in merged:
            merged[path] = meta.default if meta.default not in ("", None) else ""
    return merged


class _ToolTip:
    def __init__(self, widget: tk.Widget, text: str) -> None:
        self.widget = widget
        self.text = text
        self.tip: tk.Toplevel | None = None
        widget.bind("<Enter>", self._show)
        widget.bind("<Leave>", self._hide)

    def _show(self, _event: tk.Event) -> None:
        if self.tip is not None or not str(self.text).strip():
            return
        x = self.widget.winfo_rootx() + 12
        y = self.widget.winfo_rooty() + 18
        self.tip = tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        label = ttk.Label(
            self.tip,
            text=self.text,
            justify="left",
            background="#ffffe0",
            relief="solid",
            borderwidth=1,
            wraplength=560,
        )
        label.pack(ipadx=6, ipady=4)
        self.tip.update_idletasks()
        screen_w = self.widget.winfo_screenwidth()
        screen_h = self.widget.winfo_screenheight()
        tip_w = self.tip.winfo_reqwidth()
        tip_h = self.tip.winfo_reqheight()
        x = max(8, min(x, screen_w - tip_w - 8))
        y = max(8, min(y, screen_h - tip_h - 8))
        self.tip.wm_geometry(f"+{x}+{y}")

    def _hide(self, _event: tk.Event) -> None:
        if self.tip is not None:
            self.tip.destroy()
            self.tip = None


def _read_json(path: Path, default: Any) -> Any:
    try:
        if not path.exists():
            return default
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _deep_get(payload: Dict[str, Any], dotted_path: str, default: Any = None) -> Any:
    cursor: Any = payload
    for key in dotted_path.split("."):
        if not isinstance(cursor, dict):
            return default
        if key not in cursor:
            return default
        cursor = cursor[key]
    return cursor


def _deep_has(payload: Dict[str, Any], dotted_path: str) -> bool:
    cursor: Any = payload
    for key in dotted_path.split("."):
        if not isinstance(cursor, dict) or key not in cursor:
            return False
        cursor = cursor[key]
    return True


def _unique_paths(paths: List[Path]) -> List[Path]:
    seen: set[str] = set()
    out: List[Path] = []
    for p in paths:
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def _deep_set(payload: Dict[str, Any], dotted_path: str, value: Any) -> None:
    parts = [item for item in dotted_path.split(".") if item]
    if not parts:
        return
    cursor = payload
    for key in parts[:-1]:
        next_obj = cursor.get(key)
        if not isinstance(next_obj, dict):
            next_obj = {}
            cursor[key] = next_obj
        cursor = next_obj
    cursor[parts[-1]] = value


def _flatten(payload: Any, prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not isinstance(payload, dict):
        return out
    for key, value in payload.items():
        full = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            out.update(_flatten(value, full))
        else:
            out[full] = value
    return out


def _to_text(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return yaml.safe_dump(value, allow_unicode=True, sort_keys=False).strip()
    if value is None:
        return ""
    return str(value)


def _parse_text(value: str) -> Any:
    text = str(value).strip()
    if text == "":
        return ""
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return text


class _Hint:
    def __init__(self, parent: tk.Widget, text: str, row: int, col_span: int = 3) -> None:
        label = ttk.Label(parent, text=text, foreground="#5f6368", wraplength=900, justify="left")
        label.grid(row=row, column=0, columnspan=col_span, sticky="w", padx=4, pady=(0, 6))


class _DynamicForm(ttk.Frame):
    def __init__(
        self,
        parent: tk.Widget,
        title: str,
        label_provider: Callable[[str], str],
        hint_provider: Callable[[str], str],
        choice_provider: Callable[[str], List[str]],
        on_change: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(parent)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        ttk.Label(self, text=title).grid(row=0, column=0, sticky="w", padx=4, pady=(0, 4))

        self.canvas = tk.Canvas(self, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas)
        self.inner.bind("<Configure>", lambda _e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.window_id = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.bind("<Configure>", self._on_canvas_resize)

        self.canvas.grid(row=1, column=0, sticky="nsew")
        self.scrollbar.grid(row=1, column=1, sticky="ns")
        self.entries: Dict[str, tk.StringVar] = {}
        self.widgets: Dict[str, tk.Widget] = {}
        self._label_provider = label_provider
        self._hint_provider = hint_provider
        self._choice_provider = choice_provider
        self._on_change = on_change

    @staticmethod
    def _is_bool_choice_set(choices: List[str]) -> bool:
        if not choices:
            return False
        normalized = {str(item).strip().lower() for item in choices if str(item).strip() != ""}
        return normalized.issubset({"true", "false"}) and len(normalized) >= 1

    @staticmethod
    def _to_bool_text(value: Any) -> str:
        text = str(value).strip().lower()
        if text in {"true", "1", "yes", "on"}:
            return "true"
        return "false"

    def _on_canvas_resize(self, event: tk.Event) -> None:
        self.canvas.itemconfigure(self.window_id, width=event.width)

    def set_fields(self, values: Dict[str, Any]) -> None:
        for child in self.inner.winfo_children():
            child.destroy()
        self.entries.clear()
        self.widgets.clear()

        row = 0
        for key in values.keys():
            hint_text = str(self._hint_provider(key) or "").strip()
            if hint_text:
                mark = ttk.Label(self.inner, text="?", foreground="#1a73e8")
                mark.grid(row=row, column=0, sticky="w", padx=(4, 4), pady=2)
                _ToolTip(mark, hint_text)
            else:
                ttk.Label(self.inner, text=" ").grid(row=row, column=0, sticky="w", padx=(4, 4), pady=2)
            ttk.Label(self.inner, text=self._label_provider(key)).grid(row=row, column=1, sticky="w", padx=(0, 8), pady=2)
            choices = self._choice_provider(key)
            if self._is_bool_choice_set(choices):
                var = tk.StringVar(value=self._to_bool_text(values[key]))
                entry = ttk.Checkbutton(self.inner, variable=var, onvalue="true", offvalue="false")
            else:
                var = tk.StringVar(value=_to_text(values[key]))
            if choices and not self._is_bool_choice_set(choices):
                entry = ttk.Combobox(self.inner, textvariable=var, values=choices, state="normal")
            elif not self._is_bool_choice_set(choices):
                entry = ttk.Entry(self.inner, textvariable=var)
            entry.grid(row=row, column=2, sticky="ew", padx=4, pady=2)
            self.inner.columnconfigure(2, weight=1)
            self.entries[key] = var
            self.widgets[key] = entry
            if self._on_change is not None:
                var.trace_add("write", lambda *_args: self._on_change and self._on_change())
            row += 1

    def get_values(self) -> Dict[str, Any]:
        return {key: _parse_text(var.get()) for key, var in self.entries.items()}

    def set_enabled_by_prefix(self, enabled_prefixes: Tuple[str, ...], disabled_prefixes: Tuple[str, ...]) -> None:
        for key, widget in self.widgets.items():
            enabled = True
            if enabled_prefixes:
                enabled = any(key.startswith(prefix) for prefix in enabled_prefixes)
            if any(key.startswith(prefix) for prefix in disabled_prefixes):
                enabled = False
            widget.configure(state="normal" if enabled else "disabled")


class ExperimentUI:
    _STRICT_ENUM_PATHS: set[str] = {
        "experiment.mode",
        "experiment.cache_policy",
        "experiment.graph_dataset_cache_policy",
        "experiment.stats_time_policy",
        "experiment.on_missing_asof_snapshot",
        "mapping.adapter",
        "mapping.knowledge_graph.backend",
        "mapping.camunda_adapter.runtime.runtime_source",
        "mapping.camunda_adapter.structure.bpmn_source",
        "training.retrain",
    }

    def __init__(self, default_config: str | None = None) -> None:
        self.root = tk.Tk()
        self.root.title("BPM Experiment UI")
        self.root.geometry("1200x850")

        self._process: subprocess.Popen[str] | None = None
        self._queue: queue.Queue[str] = queue.Queue()
        self._temp_config_path: Path | None = None
        self._base_config: Dict[str, Any] = {}
        self._base_config_path: Path | None = None
        self._progress_started_ts: float | None = None
        self._stage_progress: Dict[str, float] = {}
        self._stage_runtime: Dict[str, Dict[str, Any]] = {}
        self._stage_widgets: Dict[str, Dict[str, Any]] = {}
        self._current_stage: str = ""
        self._last_status_text: str = ""
        self._last_exit_code: int | None = None
        self._progress_event_seen: bool = False
        self._telemetry_last_update_ts: float = 0.0
        self._telemetry_last_disk_update_ts: float = 0.0

        self._state = _read_json(STATE_PATH, {})
        self._presets = _read_json(PRESETS_PATH, {})
        if not isinstance(self._presets, dict):
            self._presets = {}
        self._pool_meta = self._scan_pool_meta()
        self._catalog = self._load_or_create_catalog()
        self._preset_highlight_after_id: str | None = None

        self.vars: Dict[str, tk.Variable] = {}
        self._build_vars(default_config)
        self._build_ui()
        self._bind_input_shortcuts()
        self._reset_progress_ui()
        self._refresh_preset_choices()
        self._load_state()
        self._load_base_config_into_form()
        self._apply_payload_to_forms(self._state)
        self._sync_preset_saved_at_label()
        self._refresh_state_controls()
        self._refresh_preview()
        self._poll_queue()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _scan_pool_meta(self) -> Dict[str, PoolFieldMeta]:
        pool: Dict[str, PoolFieldMeta] = {}
        roots = [ROOT_DIR / "configs"]
        for root in roots:
            if not root.exists():
                continue
            for file_path in sorted(root.rglob("*.y*ml")):
                try:
                    payload = load_yaml_with_includes(file_path)
                except Exception:
                    continue
                if not isinstance(payload, dict):
                    continue
                flat = _flatten(payload)
                for path, value in flat.items():
                    if any(
                        path.startswith(block)
                        for block in (
                            "features",
                            "policies",
                            "mapping.features",
                            "mapping.policies",
                            "mapping.graph_feature_mapping",
                        )
                    ):
                        continue
                    section = path.split(".", 1)[0]
                    if section not in {"experiment", "training", "data", "mapping", "model", "seed", "tracking", "sync_stats"}:
                        continue
                    meta = pool.get(path)
                    if meta is None:
                        meta = PoolFieldMeta(path=path, section=section, values=set())
                        pool[path] = meta
                    if isinstance(value, (str, int, float, bool)) and not isinstance(value, bool):
                        meta.values.add(str(value))
                    elif isinstance(value, bool):
                        meta.values.add("true" if value else "false")
        if "seed" not in pool:
            pool["seed"] = PoolFieldMeta(path="seed", section="seed", values={"42"})
        return pool

    @staticmethod
    def _default_tab_for_section(section: str) -> str:
        if section in {"experiment", "training", "tracking", "seed"}:
            return "general"
        if section in {"data", "mapping", "sync_stats"}:
            return "input"
        if section == "model":
            return "model"
        return "input"

    @staticmethod
    def _contains_cyrillic(text: str) -> bool:
        return any("\u0400" <= ch <= "\u04FF" for ch in str(text))

    @staticmethod
    def _humanize_path_ua(path: str) -> str:
        explicit = {
            "experiment.mode": "Режим запуску",
            "experiment.project": "Проєкт експерименту",
            "experiment.name": "Назва запуску",
            "experiment.fraction": "Частка даних (fraction)",
            "experiment.split_strategy": "Стратегія розбиття",
            "experiment.train_ratio": "Частка train",
            "experiment.split_ratio": "Співвідношення train/val/test",
            "experiment.stats_time_policy": "Політика часу статистики",
            "experiment.on_missing_asof_snapshot": "Що робити без as-of snapshot",
            "experiment.cache_policy": "Політика кешування графів",
            "experiment.graph_dataset_cache_policy": "Політика кешу граф-датасетів",
            "experiment.graph_dataset_cache_dir": "Каталог кешу граф-датасетів",
            "training.retrain": "Перенавчати з нуля",
            "seed": "Фіксоване зерно (seed)",
            "mapping.adapter": "Адаптер даних",
            "mapping.knowledge_graph.backend": "Бекенд репозиторію знань",
            "mapping.knowledge_graph.strict_load": "Суворе завантаження структури",
            "data.dataset_name": "Канонічна назва датасету",
            "data.dataset_label": "Мітка датасету",
            "data.log_path": "Шлях до XES логу",
        }
        if path in explicit:
            return explicit[path]
        parts = path.split(".")
        token_map = {
            "dataset": "датасет",
            "label": "мітка",
            "name": "назва",
            "path": "шлях",
            "backend": "бекенд",
            "strict": "суворий",
            "load": "завантаження",
            "connection": "підключення",
            "connections": "підключення",
            "profile": "профіль",
            "file": "файл",
            "files": "файли",
            "source": "джерело",
            "strategy": "стратегія",
            "runtime": "runtime",
            "structure": "структура",
            "model": "модель",
            "training": "тренування",
            "sync": "синк",
            "stats": "статистика",
            "process": "процес",
            "filters": "фільтри",
            "tenant": "тенант",
            "version": "версія",
            "key": "ключ",
            "mode": "режим",
            "ratio": "частка",
            "split": "розбиття",
            "policy": "політика",
            "dropout": "dropout",
            "hidden": "прихований",
            "dim": "розмірність",
            "heads": "голови",
            "pooling": "пулінг",
            "enabled": "увімкнено",
            "verify": "перевірка",
        }
        raw_words = parts[-1].split("_")
        translated = [token_map.get(w.lower(), w.lower()) for w in raw_words]
        tail = " ".join(translated)
        tail = tail[0].upper() + tail[1:] if tail else path
        prefix = parts[0] if parts else ""
        prefix_ua = {
            "data": "Дані",
            "mapping": "Мапінг",
            "model": "Модель",
            "training": "Тренування",
            "experiment": "Експеримент",
            "sync_stats": "Синхронізація статистики",
            "tracking": "Трекінг",
        }.get(prefix, "Параметр")
        return f"{prefix_ua}: {tail}"

    @staticmethod
    def _auto_description_ua(path: str) -> str:
        if path.startswith("model."):
            return f"Налаштування архітектури/поведінки моделі. Впливає на якість і стабільність навчання. ({path})"
        if path.startswith("training."):
            return f"Параметр процесу навчання (епохи, батч, LR, patience тощо). ({path})"
        if path.startswith("experiment."):
            return f"Керує сценарієм запуску експерименту та розбиттям даних. ({path})"
        if path.startswith("data."):
            return f"Визначає джерело/ідентифікацію вхідних даних. ({path})"
        if path.startswith("mapping.knowledge_graph."):
            return f"Керує EOPKG-сховищем (neo4j/file/in_memory) та правилами завантаження структури. ({path})"
        if path.startswith("mapping.camunda_adapter.structure."):
            return f"Керує отриманням BPMN/структури процесу з Camunda (files/mssql). ({path})"
        if path.startswith("mapping.camunda_adapter.runtime."):
            return f"Керує отриманням runtime-подій з Camunda (files/mssql). ({path})"
        if path.startswith("mapping.xes_adapter."):
            return f"Керує читанням полів і класифікаторів із XES. ({path})"
        if path.startswith("mapping."):
            return f"Параметр мапінгу даних/структури процесу. ({path})"
        if path.startswith("sync_stats."):
            return f"Параметр синхронізації статистики графа. ({path})"
        if path == "seed":
            return "Фіксує генератори випадковості для відтворюваності запусків."
        return f"Службовий параметр конфігурації. ({path})"

    @staticmethod
    def _default_catalog_entry(meta: PoolFieldMeta) -> Dict[str, Any]:
        enum_values = sorted(meta.values)
        return {
            "path": meta.path,
            "section": meta.section,
            "label": meta.path,
            "description": FIELD_HINTS.get(meta.path, ""),
            "affects": "",
            "default": "",
            "enum": enum_values if 1 < len(enum_values) <= 20 else [],
            "required_in_modes": [],
            "required_when": {},
            "ui": {
                "tab": ExperimentUI._default_tab_for_section(meta.section),
                "group": "advanced",
                "priority": 100,
                "order": 1000,
            },
        }

    def _load_or_create_catalog(self) -> Dict[str, CatalogFieldMeta]:
        try:
            raw = yaml.safe_load(CATALOG_PATH.read_text(encoding="utf-8")) if CATALOG_PATH.exists() else {}
        except Exception:
            raw = {}
        if not isinstance(raw, dict):
            raw = {}
        fields_raw = raw.get("fields", {})
        if not isinstance(fields_raw, dict):
            fields_raw = {}
        updated = False
        for path, meta in self._pool_meta.items():
            if path not in fields_raw or not isinstance(fields_raw.get(path), dict):
                fields_raw[path] = self._default_catalog_entry(meta)
                updated = True
        # Built-in defaults for most important controls.
        defaults: Dict[str, Dict[str, Any]] = {
            "experiment.mode": {
                "description": "Main run mode that selects CLI command path.",
                "enum": ["train", "eval_drift", "eval_cross_dataset", "sync-topology", "sync-stats", "sync-stats-backfill"],
                "required_in_modes": ["train", "eval_drift", "eval_cross_dataset", "sync-topology", "sync-stats", "sync-stats-backfill"],
                "ui": {"tab": "general", "group": "core", "priority": 1, "order": 1},
            },
            "experiment.project": {
                "description": "Experiment project name for tracking.",
                "required_in_modes": ["train", "eval_drift", "eval_cross_dataset"],
                "ui": {"tab": "general", "group": "core", "priority": 1, "order": 2},
            },
            "experiment.name": {
                "description": "Experiment run name.",
                "required_in_modes": ["train", "eval_drift", "eval_cross_dataset"],
                "ui": {"tab": "general", "group": "core", "priority": 1, "order": 3},
            },
            "data.dataset_name": {
                "description": "Canonical process/dataset name used by repository and logs.",
                "required_in_modes": ["train", "eval_drift", "eval_cross_dataset", "sync-topology", "sync-stats", "sync-stats-backfill"],
                "ui": {"tab": "input", "group": "source", "priority": 1, "order": 1},
            },
            "data.dataset_label": {
                "description": "Human-readable dataset label for run profile and tracking.",
                "ui": {"tab": "input", "group": "source", "priority": 1, "order": 2},
            },
            "mapping.adapter": {
                "description": "Ingestion adapter: xes or camunda.",
                "enum": ["xes", "camunda"],
                "required_in_modes": ["train", "eval_drift", "eval_cross_dataset", "sync-topology", "sync-stats", "sync-stats-backfill"],
                "ui": {"tab": "input", "group": "source", "priority": 1, "order": 3},
            },
            "data.log_path": {
                "description": "Path to XES file when adapter=xes.",
                "required_when": {"mapping.adapter": "xes"},
                "ui": {"tab": "input", "group": "source", "priority": 1, "order": 4},
            },
            "mapping.camunda_adapter.runtime.runtime_source": {
                "description": "Camunda runtime source mode: files or mssql.",
                "enum": ["files", "mssql"],
                "required_when": {"mapping.adapter": "camunda"},
                "ui": {"tab": "input", "group": "source", "priority": 2, "order": 10},
            },
            "mapping.camunda_adapter.structure.bpmn_source": {
                "description": "BPMN structure source for Camunda: files or mssql.",
                "enum": ["files", "mssql"],
                "required_when": {"mapping.adapter": "camunda"},
                "ui": {"tab": "input", "group": "source", "priority": 2, "order": 11},
            },
            "mapping.knowledge_graph.backend": {
                "description": "Knowledge graph backend storage.",
                "enum": ["in_memory", "file", "neo4j"],
                "ui": {"tab": "input", "group": "mapping", "priority": 2, "order": 12},
            },
            "mapping.knowledge_graph.strict_load": {
                "description": "Fail-fast when structure is missing.",
                "ui": {"tab": "input", "group": "mapping", "priority": 2, "order": 13},
            },
        }
        for path, override in defaults.items():
            if path not in fields_raw or not isinstance(fields_raw[path], dict):
                continue
            payload = fields_raw[path]
            for k, v in override.items():
                if k == "ui":
                    if not isinstance(payload.get("ui"), dict):
                        payload["ui"] = {}
                    for uk, uv in v.items():
                        if payload["ui"].get(uk) in (None, "", 0, []):
                            payload["ui"][uk] = uv
                            updated = True
                else:
                    if payload.get(k) in (None, "", [], {}):
                        payload[k] = v
                        updated = True
        for path, payload in fields_raw.items():
            if not isinstance(payload, dict):
                continue
            label_text = str(payload.get("label", "")).strip()
            if label_text == "":
                payload["label"] = path
                updated = True
            desc = str(payload.get("description", "")).strip()
            if desc == "":
                payload["description"] = self._auto_description_ua(path)
                updated = True
            if self._is_path_like_field(path) and payload.get("enum"):
                payload["enum"] = []
                updated = True
            if not isinstance(payload.get("ui"), dict):
                payload["ui"] = {"tab": self._default_tab_for_section(str(payload.get("section", path.split(".", 1)[0])))}
                updated = True
        # Preserve existing catalog entries that may point to non-pooled pseudo fields.
        catalog_dump = {
            "version": int(raw.get("version", 1) or 1),
            "fields": dict(sorted(fields_raw.items(), key=lambda item: item[0])),
        }
        if updated or not CATALOG_PATH.exists():
            CATALOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            CATALOG_PATH.write_text(yaml.safe_dump(catalog_dump, allow_unicode=True, sort_keys=False), encoding="utf-8")

        catalog: Dict[str, CatalogFieldMeta] = {}
        for path, payload in fields_raw.items():
            if not isinstance(payload, dict):
                continue
            ui_payload = payload.get("ui", {}) if isinstance(payload.get("ui"), dict) else {}
            pool_meta = self._pool_meta.get(path)
            section = str(payload.get("section", (pool_meta.section if pool_meta else path.split(".", 1)[0]))).strip() or (
                pool_meta.section if pool_meta else path.split(".", 1)[0]
            )
            enum_raw = payload.get("enum", [])
            required_modes_raw = payload.get("required_in_modes", [])
            required_when_raw = payload.get("required_when", {})
            catalog[path] = CatalogFieldMeta(
                path=path,
                section=section,
                label=str(payload.get("label", path)).strip() or path,
                description=str(payload.get("description", "")).strip(),
                affects=str(payload.get("affects", "")).strip(),
                default=payload.get("default", ""),
                enum=[str(v) for v in enum_raw] if isinstance(enum_raw, list) else [],
                required_in_modes=[str(v) for v in required_modes_raw] if isinstance(required_modes_raw, list) else [],
                required_when=dict(required_when_raw) if isinstance(required_when_raw, dict) else {},
                ui_tab=str(ui_payload.get("tab", self._default_tab_for_section(section))).strip() or self._default_tab_for_section(section),
                ui_group=str(ui_payload.get("group", "advanced")).strip() or "advanced",
                ui_priority=int(ui_payload.get("priority", 100) or 100),
                ui_order=int(ui_payload.get("order", 1000) or 1000),
            )
        return catalog

    def _catalog_meta(self, field_path: str) -> CatalogFieldMeta | None:
        return self._catalog.get(field_path)

    def _ordered_field_dict(self, values: Dict[str, Any]) -> Dict[str, Any]:
        ordered_keys = sorted(
            values.keys(),
            key=lambda key: (
                self._catalog_meta(key).ui_priority if self._catalog_meta(key) is not None else 1000,
                self._catalog_meta(key).ui_order if self._catalog_meta(key) is not None else 1000,
                key,
            ),
        )
        return {key: values[key] for key in ordered_keys}

    def _label_for(self, field_path: str) -> str:
        # Keep technical config keys visible in UI.
        return field_path

    @staticmethod
    def _is_eopkg_mapping_path(path: str) -> bool:
        return (
            path.startswith("mapping.camunda_adapter.structure.")
            or path.startswith("mapping.camunda_adapter.subprocess_graph_mode")
        )


    @staticmethod
    def _hint_text_for_path(field_path: str) -> str:
        exact: Dict[str, str] = {
            "experiment.mode": "Режим запуску: train/eval/sync. Визначає, яку команду фактично виконає UI.",
            "experiment.project": "Назва групи запусків у трекінгу.",
            "experiment.name": "Назва конкретного запуску.",
            "experiment.fraction": "Частка трас (0..1] після каскадного відбору. Менше значення = швидше, але шумніші метрики.",
            "experiment.split_strategy": "Стратегія розбиття: temporal (рекомендовано) або none.",
            "experiment.train_ratio": "Частка історії у train-cut. Типово 0.6-0.8.",
            "experiment.split_ratio": "Трійка [train,val,test], що має сумуватися до 1.0.",
            "experiment.load_checkpoint": "Шлях до checkpoint для eval або продовження навчання.",
            "experiment.stats_time_policy": "Часова політика snapshot статистик: latest або strict_asof.",
            "experiment.on_missing_asof_snapshot": "Що робити, якщо strict_asof snapshot відсутній: disable_stats/use_base/raise.",
            "experiment.cache_policy": "Політика кешу побудови графів: off (без кешу), dto (кеш DTO), full (DTO + готові структурні тензори).",
            "experiment.graph_dataset_cache_policy": "Кеш уже побудованих graph datasets: off/read/write/full.",
            "experiment.graph_dataset_cache_dir": "Каталог disk-cache для train/validation/test графів.",
            "experiment.drift_window_size": "Розмір drift-вікна у трасах для eval_drift.",
            "experiment.drift_window_sliding": "Крок зсуву drift-вікна у трасах. 0 = без перекриття.",
            "data.dataset_name": "Канонічний ідентифікатор процесу/датасету, який використовують структура і run-profile.",
            "data.dataset_label": "Людяний ярлик для логів, артефактів і трекінгу.",
            "data.log_path": "Шлях до XES-файлу (або каталогу XES для sync XES).",
            "mapping.adapter": "Джерело подій: xes або camunda.",
            "model.type": "Тип моделі з реєстру (наприклад BaselineGATv2, BaselineGCN, EOPKGGATv2).",
            "model.model_label": "Коротка мітка моделі для назв запуску та checkpoint.",
            "model.hidden_dim": "Розмір прихованого представлення основного енкодера.",
            "model.struct_hidden_dim": "Розмір прихованого представлення структурної гілки EOPKG.",
            "model.struct_encoder_type": "Тип структурного GNN шару (GATv2Conv або GCNConv).",
            "model.fusion_mode": "Режим злиття observed + structural контексту в EOPKGGATv2 (Attention або Concat).",
            "model.cross_attention_heads": "Кількість голів cross-attention у EOPKG.",
            "model.dropout": "Ймовірність dropout для регуляризації.",
            "model.graph_strategy": "Стратегія побудови графових прикладів для моделі.",
            "model.pooling_strategy": "Спосіб агрегації вузлів у вектор графа.",
            "training.retrain": "true = старт з нуля; false = спроба продовжити з checkpoint.",
            "training.batch_size": "Розмір міні-батча. Більше значення прискорює, але вимагає більше пам'яті.",
            "training.epochs": "Максимальна кількість епох навчання.",
            "training.learning_rate": "Крок оптимізатора. Типовий старт для Adam: 1e-3.",
            "training.patience": "Скільки епох чекати покращення до early stopping.",
            "training.delta": "Мінімальне покращення val_loss для early stopping.",
            "training.device": "Пристрій виконання: cpu або cuda.",
            "training.class_weight_cap": "Верхня межа класових ваг у loss для стабільності.",
            "training.loss_function": "Назва loss-функції для next-activity prediction.",
            "training.ece_bins": "Кількість бінів для розрахунку ECE калібрування.",
            "training.dataloader_num_workers": "Кількість процесів DataLoader для підготовки batch. 0 = без мультипроцесності.",
            "training.dataloader_pin_memory": "Фіксує batch у pinned RAM для швидшого копіювання на GPU (актуально для CUDA).",
            "training.dataloader_persistent_workers": "Не перезапускати DataLoader-воркери між епохами (працює, коли workers > 0).",
            "training.dataloader_prefetch_factor": "Скільки batch наперед готує кожен DataLoader-воркер (коли workers > 0).",
            "training.torch_num_threads": "Обмеження intra-op потоків PyTorch на CPU. Порожньо = системний дефолт.",
            "training.torch_num_interop_threads": "Обмеження inter-op потоків PyTorch на CPU. Порожньо = системний дефолт.",
            "training.show_progress": "Показувати прогрес-бар у тренуванні.",
            "training.tqdm_disable": "Примусово вимкнути tqdm прогрес-бар.",
            "training.tqdm_leave": "Залишати tqdm-бар у консолі після завершення.",
            "tracking.enabled": "Увімкнути логування параметрів, метрик і артефактів.",
            "tracking.backend": "Бекенд трекінгу (у проєкті використовується MLflow).",
            "tracking.uri": "URI сховища трекінгу (file або server URL).",
            "tracking.tags.stage": "Тег етапу для фільтрації запусків у трекінгу.",
            "tracking.tags.model_family": "Тег сімейства моделі для порівнянь запусків.",
            "sync_stats.enabled": "Увімкнути або вимкнути синхронізацію статистик.",
            "sync_stats.stats_time_policy": "Часова політика snapshot статистик (strict_asof/latest).",
            "sync_stats.process_scope_policy": "Політика відбору історії для process-scope статистики.",
            "sync_stats.windows_days": "Список вікон у днях для rolling-метрик, наприклад [7, 30, 90].",
            "sync_stats.freshness_half_life_days": "Період напіврозпаду freshness score у днях.",
            "sync_stats.confidence_weights.sample_size": "Вага компоненти sample_size у confidence score.",
            "sync_stats.confidence_weights.freshness": "Вага компоненти freshness у confidence score.",
            "sync_stats.confidence_weights.coverage": "Вага компоненти coverage у confidence score.",
            "sync_stats.process_filters": "Фільтр процесів для sync-stats (порожньо = всі процеси).",
            "sync_stats.tenant_filters": "Фільтр tenant-ів для sync-stats.",
            "sync_stats.show_progress": "Показувати прогрес-бар під час sync-stats.",
            "seed": "Фіксує випадковість для відтворюваності запусків.",
        }
        if field_path in exact:
            return exact[field_path]
        if field_path.startswith("mapping.xes_adapter."):
            key = field_path.split("mapping.xes_adapter.", 1)[1]
            xes = {
                "case_id_key": "Ключ XES для case id. Визначає, як події групуються у траси.",
                "activity_key": "Ключ XES активності, якщо classifier вимкнений.",
                "timestamp_key": "Ключ XES timestamp для хронології, split і as-of логіки.",
                "resource_key": "Ключ XES ресурсу/виконавця.",
                "lifecycle_key": "Ключ lifecycle атрибуту для парування start/complete.",
                "version_key": "Ключ XES версії процесу для версійних і drift сценаріїв.",
                "complete_transitions": "Які lifecycle стани трактуються як завершення активності.",
                "pairing_strategy": "Стратегія матчінгу start/complete (lifo/fifo/by_instance).",
                "use_classifier": "true: activity_id з classifier; false: з activity_key.",
            }
            return xes.get(key, "Додатковий XES параметр читання/мапінгу.")
        if field_path.startswith("mapping.camunda_adapter.runtime."):
            key = field_path.split("mapping.camunda_adapter.runtime.", 1)[1]
            runtime = {
                "runtime_source": "Джерело runtime-подій Camunda: files (експорти) або mssql.",
                "export_dir": "Каталог з runtime експортами Camunda (CSV/XLSX).",
                "sql_dir": "Каталог SQL-шаблонів runtime запитів.",
                "connection_profile": "Профіль MSSQL підключення для runtime.",
                "connections_file": "YAML файл з MSSQL профілями підключення для runtime.",
                "mssql.connection_profile": "Локальне перевизначення MSSQL профілю в runtime.mssql.",
                "history_cleanup_aware": "Чи враховувати removal_time при очищенні історичних рядків.",
                "legacy_removal_time_policy": "Як поводитись з рядками без removal_time (legacy history).",
                "on_missing_removal_time": "Дія, якщо removal_time відсутній (auto_fallback/strict).",
            }
            return runtime.get(key, "Додаткове runtime налаштування Camunda.")
        if field_path.startswith("mapping.camunda_adapter.structure."):
            key = field_path.split("mapping.camunda_adapter.structure.", 1)[1]
            structure = {
                "source": "Джерело структури: bpmn або logs fallback.",
                "structure_from_logs": "Примусово будувати структуру з логів замість BPMN.",
                "bpmn_source": "Звідки читати BPMN: files або mssql.",
                "parser_mode": "Режим BPMN парсера (recover/strict).",
                "subprocess_mode": "Режим представлення підпроцесів у структурі.",
                "version_key_format": "Формат нормалізації ключів версій (наприклад vNN).",
                "connection_profile": "Профіль MSSQL підключення для structure блоку.",
                "connections_file": "YAML файл профілів MSSQL для structure блоку.",
                "sql_dir": "Каталог SQL-шаблонів для structure mssql.",
                "files.export_dir": "Базовий каталог BPMN експортів.",
                "files.catalog_file": "Файл каталогу process definitions (CSV/XLSX).",
                "files.bpmn_dir": "Підкаталог з BPMN XML файлами.",
                "mssql.connection_profile": "Локальне перевизначення MSSQL profile для structure.mssql.",
                "call_activity.inference_fallback_strategy": "Fallback політика для callActivity, якщо дочірній процес не резолвиться.",
            }
            return structure.get(key, "Додаткове налаштування BPMN структури Camunda.")
        if field_path.startswith("mapping.camunda_adapter."):
            key = field_path.split("mapping.camunda_adapter.", 1)[1]
            camunda = {
                "process_name": "Ключ процесу Camunda (proc_def_key) для відбору даних.",
                "process_filters": "Список process key для фільтрації Camunda даних.",
                "tenant_id": "Single-tenant фільтр Camunda даних.",
                "tenant_filters": "Multi-tenant фільтр (список tenant_id).",
                "version_key": "Фільтр конкретної версії процесу.",
                "lookback_hours": "Альтернативний фільтр «останні N годин» для runtime.",
                "since": "Початок часового інтервалу (ISO) для вибірки подій.",
                "until": "Кінець часового інтервалу (ISO) для вибірки подій.",
                "subprocess_graph_mode": "Режим представлення підпроцесів у instance graph.",
            }
            return camunda.get(key, "Додаткове top-level налаштування Camunda адаптера.")
        if field_path.startswith("mapping.knowledge_graph."):
            key = field_path.split("mapping.knowledge_graph.", 1)[1]
            kg = {
                "backend": "Бекенд сховища структури/статистики: neo4j, file або in_memory.",
                "path": "Локальний шлях сховища для file/in_memory backend.",
                "strict_load": "Fail-fast, якщо структура відсутня або неузгоджена.",
                "ingest_split": "Яку частину даних брати у sync-topology: train або full.",
                "neo4j.connection_profile": "Профіль підключення до Neo4j.",
                "neo4j.connections_file": "YAML файл профілів підключення до Neo4j.",
                "neo4j.verify_connectivity": "Перевіряти доступність Neo4j при старті.",
            }
            return kg.get(key, "Додаткове налаштування knowledge graph сховища.")
        if field_path.startswith("model."):
            return "Параметр архітектури моделі. Впливає на якість, стабільність та обчислювальну вартість."
        if field_path.startswith("training."):
            return "Параметр процесу навчання (батч, епохи, LR, early stopping, device)."
        if field_path.startswith("tracking."):
            return "Параметр трекінгу запусків і метрик."
        if field_path.startswith("sync_stats."):
            return "Параметр синхронізації snapshot статистик для структурних фіч і temporal оцінки."
        return ""

    def _hint_for(self, field_path: str) -> str:
        cat = self._catalog_meta(field_path)
        base = self._hint_text_for_path(field_path)
        if not base and cat is not None:
            candidate = str(cat.description).strip()
            question_ratio = candidate.count("?") / max(len(candidate), 1)
            if candidate and question_ratio < 0.2:
                base = candidate
        if not base:
            base = DETAILED_HINTS_UA.get(field_path, "")
        if not base:
            base = self._auto_description_ua(field_path)

        impact = ""
        if field_path.startswith("data."):
            impact = "Вплив: завантаження і ідентифікація вхідних даних."
        elif field_path.startswith("experiment."):
            impact = "Вплив: маршрут запуску, split і режими оцінки."
        elif field_path.startswith("mapping."):
            impact = "Вплив: коректність інжесту, alignment зі структурою та доступність структурних фіч."
        elif field_path.startswith("model."):
            impact = "Вплив: ємність моделі, швидкість та узагальнення."
        elif field_path.startswith("training."):
            impact = "Вплив: стабільність навчання і фінальні метрики."
        elif field_path.startswith("tracking."):
            impact = "Вплив: збереження параметрів/метрик і аналіз запусків."
        elif field_path.startswith("sync_stats."):
            impact = "Вплив: якість snapshot статистик, які потрапляють у forward."
        elif field_path == "seed":
            impact = "Вплив: відтворюваність результатів."

        lines: List[str] = [base or f"Параметр конфігурації: {field_path}", f"Шлях: {field_path}"]
        if impact:
            lines.append(impact)
        if cat is not None and cat.required_in_modes:
            lines.append(f"Обов'язково в режимах: {', '.join(cat.required_in_modes)}")
        if cat is not None and cat.required_when:
            conditions = ", ".join(f"{k}={v}" for k, v in cat.required_when.items())
            if conditions:
                lines.append(f"Обов'язково коли: {conditions}")
        if cat is not None and cat.default not in ("", None):
            lines.append(f"За замовчуванням: {cat.default}")
        return "\n".join(line for line in lines if str(line).strip())

    def _choices_for(self, field_path: str) -> List[str]:
        cat = self._catalog_meta(field_path)
        if cat is not None and cat.enum:
            return list(cat.enum)
        meta = self._pool_meta.get(field_path)
        if meta is None:
            return []
        values = sorted(v for v in meta.values if v != "")
        if 1 < len(values) <= 12:
            return values
        return []

    def _build_vars(self, default_config: str | None) -> None:
        default = default_config or str(ROOT_DIR / "configs" / "experiments" / "mvp2_5_stage4_2_eopkg_files_stat.yaml")
        self.preset_saved_at_var = tk.StringVar(value="Last saved: -")
        self.vars["config_path"] = tk.StringVar(value=default)
        self.vars["data_config_path"] = tk.StringVar(value="")
        self.vars["preset_name"] = tk.StringVar(value="")
        self.vars["mode"] = tk.StringVar(value="train")
        self.vars["project"] = tk.StringVar(value="")
        self.vars["experiment_name"] = tk.StringVar(value="")
        self.vars["fraction"] = tk.StringVar(value="1.0")
        self.vars["split_strategy"] = tk.StringVar(value="temporal")
        self.vars["train_ratio"] = tk.StringVar(value="0.7")
        self.vars["split_ratio"] = tk.StringVar(value="[0.7, 0.2, 0.1]")
        self.vars["split_ratio_train"] = tk.StringVar(value="0.7")
        self.vars["split_ratio_val"] = tk.StringVar(value="0.2")
        self.vars["split_ratio_test"] = tk.StringVar(value="0.1")
        self.vars["seed"] = tk.StringVar(value="42")
        self.vars["stats_time_policy"] = tk.StringVar(value="strict_asof")
        self.vars["on_missing_asof_snapshot"] = tk.StringVar(value="disable_stats")
        self.vars["cache_policy"] = tk.StringVar(value="full")
        self.vars["graph_dataset_cache_policy"] = tk.StringVar(value="off")
        self.vars["graph_dataset_cache_dir"] = tk.StringVar(value=".cache/graph_datasets")
        self.vars["gateway_mode"] = tk.StringVar(value="collapse_for_prediction")
        self.vars["adapter"] = tk.StringVar(value="camunda")
        self.vars["sync_as_of"] = tk.StringVar(value="")
        self.vars["backfill_step"] = tk.StringVar(value="weekly")
        self.vars["backfill_step_days"] = tk.StringVar(value="")
        self.vars["backfill_from"] = tk.StringVar(value="")
        self.vars["backfill_to"] = tk.StringVar(value="")
        self.vars["extra_args"] = tk.StringVar(value="")

    @staticmethod
    def _default_gateway_mode_for_adapter(adapter: str) -> str:
        return "collapse_for_prediction" if str(adapter).strip().lower() == "xes" else "preserve"

    def _resolve_gateway_mode_from_mapping(self, mapping_payload: Any) -> str:
        if isinstance(mapping_payload, dict):
            projection = mapping_payload.get("topology_projection", {})
            if isinstance(projection, dict):
                mode = str(projection.get("gateway_mode", "")).strip()
                if mode in {"preserve", "collapse_for_prediction"}:
                    return mode
            mode = str(mapping_payload.get("gateway_mode", "")).strip()
            if mode in {"preserve", "collapse_for_prediction"}:
                return mode
        return self._default_gateway_mode_for_adapter(self.vars["adapter"].get())

    @staticmethod
    def _normalize_graph_dataset_cache_policy(raw: Any) -> str:
        text = str(raw).strip().lower()
        if text in {"", "none", "disabled"}:
            return "off"
        if text in {"true", "on", "yes"}:
            return "full"
        if text in {"false", "off", "no"}:
            return "off"
        if text in {"read", "write", "full"}:
            return text
        return "off"

    def _set_split_ratio_vars(self, raw: Any) -> None:
        default_values = [0.7, 0.2, 0.1]
        parsed = raw
        if isinstance(raw, str):
            parsed = _parse_text(raw)
        values = list(default_values)
        if isinstance(parsed, (list, tuple)) and len(parsed) == 3:
            try:
                values = [float(parsed[0]), float(parsed[1]), float(parsed[2])]
            except (TypeError, ValueError):
                values = list(default_values)
        self.vars["split_ratio"].set(_to_text(values))
        self.vars["split_ratio_train"].set(str(values[0]))
        self.vars["split_ratio_val"].set(str(values[1]))
        self.vars["split_ratio_test"].set(str(values[2]))

    def _resolve_split_ratio(self) -> Any:
        train_raw = str(self.vars.get("split_ratio_train").get()).strip()
        val_raw = str(self.vars.get("split_ratio_val").get()).strip()
        test_raw = str(self.vars.get("split_ratio_test").get()).strip()
        if train_raw or val_raw or test_raw:
            if not train_raw or not val_raw or not test_raw:
                raise ValueError("experiment.split_ratio: fill all three values (train/val/test).")
            try:
                values = [float(train_raw), float(val_raw), float(test_raw)]
            except ValueError as exc:
                raise ValueError("experiment.split_ratio values must be numeric.") from exc
            self.vars["split_ratio"].set(_to_text(values))
            return values
        return _parse_text(str(self.vars["split_ratio"].get()))

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        top = ttk.Frame(self.root, padding=10)
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(1, weight=1)
        ttk.Label(top, text="Base config").grid(row=0, column=0, sticky="w", padx=(0, 8))
        cfg_entry = ttk.Entry(top, textvariable=self.vars["config_path"])
        cfg_entry.grid(row=0, column=1, sticky="ew")
        ttk.Button(top, text="Browse", command=self._browse_config).grid(row=0, column=2, padx=(8, 0))
        ttk.Button(top, text="Reload", command=self._load_base_config_into_form).grid(row=0, column=3, padx=(8, 0))
        ttk.Button(top, text="Fill Defaults", command=self._fill_defaults).grid(row=0, column=4, padx=(8, 0))

        notebook = ttk.Notebook(self.root)
        notebook.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))

        self.tab_general = ttk.Frame(notebook, padding=10)
        self.tab_input = ttk.Frame(notebook, padding=10)
        self.tab_eopkg = ttk.Frame(notebook, padding=10)
        self.tab_model = ttk.Frame(notebook, padding=10)
        self.tab_run = ttk.Frame(notebook, padding=10)
        notebook.add(self.tab_general, text="General")
        notebook.add(self.tab_input, text="Input Data")
        notebook.add(self.tab_eopkg, text="EOPKG / Structure")
        notebook.add(self.tab_model, text="Model")
        notebook.add(self.tab_run, text="Run")

        self._build_general_tab()
        self._build_input_tab()
        self._build_eopkg_tab()
        self._build_model_tab()
        self._build_run_tab()

        for key in ("mode", "adapter", "config_path"):
            self.vars[key].trace_add("write", lambda *_args: self._refresh_state_controls())
        for key, var in self.vars.items():
            if isinstance(var, (tk.StringVar, tk.BooleanVar, tk.IntVar, tk.DoubleVar)):
                var.trace_add("write", lambda *_args: self._refresh_preview())
        self.vars["preset_name"].trace_add("write", lambda *_args: self._sync_preset_saved_at_label())

    def _bind_input_shortcuts(self) -> None:
        def _event(name: str):
            def _handler(event: tk.Event) -> str:
                try:
                    event.widget.event_generate(name)
                except tk.TclError:
                    return "break"
                return "break"

            return _handler

        def _select_all(event: tk.Event) -> str:
            widget = event.widget
            try:
                if hasattr(widget, "selection_range"):
                    widget.selection_range(0, tk.END)
                if hasattr(widget, "icursor"):
                    widget.icursor(tk.END)
            except tk.TclError:
                return "break"
            return "break"

        for class_name in ("TEntry", "Entry", "TCombobox", "Combobox"):
            self.root.bind_class(class_name, "<Control-c>", _event("<<Copy>>"), add="+")
            self.root.bind_class(class_name, "<Control-C>", _event("<<Copy>>"), add="+")
            self.root.bind_class(class_name, "<Control-v>", _event("<<Paste>>"), add="+")
            self.root.bind_class(class_name, "<Control-V>", _event("<<Paste>>"), add="+")
            self.root.bind_class(class_name, "<Control-x>", _event("<<Cut>>"), add="+")
            self.root.bind_class(class_name, "<Control-X>", _event("<<Cut>>"), add="+")
            self.root.bind_class(class_name, "<Control-a>", _select_all, add="+")
            self.root.bind_class(class_name, "<Control-A>", _select_all, add="+")

    def _add_help_mark(self, parent: tk.Widget, row: int, text: str, col: int = 2) -> None:
        mark = ttk.Label(parent, text="?", foreground="#1a73e8")
        mark.grid(row=row, column=col, sticky="w", padx=(4, 0), pady=2)
        _ToolTip(mark, text)

    def _build_general_tab(self) -> None:
        f = self.tab_general
        f.columnconfigure(0, weight=1)
        f.rowconfigure(0, weight=1)
        self._general_field_widgets: Dict[str, tk.Widget] = {}

        sections = ttk.Notebook(f)
        sections.grid(row=0, column=0, sticky="nsew")
        core = ttk.Frame(sections, padding=8)
        advanced = ttk.Frame(sections, padding=8)
        sections.add(core, text="Core")
        sections.add(advanced, text="Advanced")
        core.columnconfigure(1, weight=1)
        advanced.columnconfigure(0, weight=1)
        advanced.rowconfigure(0, weight=1)

        ttk.Label(core, text="preset_name").grid(row=0, column=0, sticky="w")
        self.preset_name_box = ttk.Combobox(core, textvariable=self.vars["preset_name"], state="normal")
        self.preset_name_box.grid(row=0, column=1, sticky="ew")
        bar = ttk.Frame(core)
        bar.grid(row=0, column=2, sticky="e")
        ttk.Button(bar, text="Save", command=self._save_preset).grid(row=0, column=0, padx=(0, 4))
        ttk.Button(bar, text="Load", command=self._load_preset).grid(row=0, column=1, padx=(0, 4))
        ttk.Button(bar, text="Delete", command=self._delete_preset).grid(row=0, column=2)
        self._add_help_mark(core, 0, "Назва preset для збереження/відновлення поточного стану UI.", col=3)
        self.preset_saved_at_label = tk.Label(core, textvariable=self.preset_saved_at_var, fg="#2e7d32")
        self.preset_saved_at_label.grid(row=0, column=4, sticky="w", padx=(8, 0))
        _Hint(core, "Preset = збережений стан UI (це не окремий YAML-файл).", row=1)

        ttk.Label(core, text="experiment.mode").grid(row=2, column=0, sticky="w")
        self._add_help_mark(core, 2, self._hint_for("experiment.mode"))
        self.mode_box = ttk.Combobox(
            core,
            textvariable=self.vars["mode"],
            values=["train", "eval_drift", "eval_cross_dataset", "sync-topology", "sync-stats", "sync-stats-backfill"],
            state="readonly",
            width=28,
        )
        self.mode_box.grid(row=2, column=1, sticky="w")

        ttk.Label(core, text="experiment.project").grid(row=4, column=0, sticky="w")
        self._add_help_mark(core, 4, self._hint_for("experiment.project"))
        w = ttk.Entry(core, textvariable=self.vars["project"]); w.grid(row=4, column=1, sticky="ew"); self._general_field_widgets["project"] = w

        ttk.Label(core, text="experiment.name").grid(row=5, column=0, sticky="w")
        self._add_help_mark(core, 5, self._hint_for("experiment.name"))
        w = ttk.Entry(core, textvariable=self.vars["experiment_name"]); w.grid(row=5, column=1, sticky="ew"); self._general_field_widgets["experiment_name"] = w

        ttk.Label(core, text="experiment.fraction").grid(row=6, column=0, sticky="w")
        self._add_help_mark(core, 6, self._hint_for("experiment.fraction"))
        w = ttk.Entry(core, textvariable=self.vars["fraction"]); w.grid(row=6, column=1, sticky="ew"); self._general_field_widgets["fraction"] = w

        ttk.Label(core, text="experiment.split_strategy").grid(row=7, column=0, sticky="w")
        self._add_help_mark(core, 7, self._hint_for("experiment.split_strategy"))
        w = ttk.Combobox(core, textvariable=self.vars["split_strategy"], values=["temporal", "none"], state="readonly", width=18); w.grid(row=7, column=1, sticky="w"); self._general_field_widgets["split_strategy"] = w

        ttk.Label(core, text="experiment.train_ratio").grid(row=8, column=0, sticky="w")
        self._add_help_mark(core, 8, self._hint_for("experiment.train_ratio"))
        w = ttk.Entry(core, textvariable=self.vars["train_ratio"]); w.grid(row=8, column=1, sticky="ew"); self._general_field_widgets["train_ratio"] = w

        ttk.Label(core, text="experiment.split_ratio").grid(row=9, column=0, sticky="w")
        self._add_help_mark(core, 9, self._hint_for("experiment.split_ratio"))
        ratio_frame = ttk.Frame(core)
        ratio_frame.grid(row=9, column=1, sticky="ew")
        ttk.Label(ratio_frame, text="train").grid(row=0, column=0, sticky="w")
        split_train = ttk.Entry(ratio_frame, textvariable=self.vars["split_ratio_train"], width=8)
        split_train.grid(row=0, column=1, sticky="w", padx=(4, 8))
        ttk.Label(ratio_frame, text="val").grid(row=0, column=2, sticky="w")
        split_val = ttk.Entry(ratio_frame, textvariable=self.vars["split_ratio_val"], width=8)
        split_val.grid(row=0, column=3, sticky="w", padx=(4, 8))
        ttk.Label(ratio_frame, text="test").grid(row=0, column=4, sticky="w")
        split_test = ttk.Entry(ratio_frame, textvariable=self.vars["split_ratio_test"], width=8)
        split_test.grid(row=0, column=5, sticky="w", padx=(4, 0))
        self._split_ratio_widgets = [split_train, split_val, split_test]
        self._general_field_widgets["split_ratio"] = split_train

        ttk.Label(core, text="seed").grid(row=10, column=0, sticky="w")
        self._add_help_mark(core, 10, self._hint_for("seed"))
        w = ttk.Entry(core, textvariable=self.vars["seed"]); w.grid(row=10, column=1, sticky="ew"); self._general_field_widgets["seed"] = w

        ttk.Label(core, text="experiment.stats_time_policy").grid(row=11, column=0, sticky="w")
        self._add_help_mark(core, 11, self._hint_for("experiment.stats_time_policy"))
        w = ttk.Combobox(core, textvariable=self.vars["stats_time_policy"], values=["latest", "strict_asof"], state="readonly", width=18); w.grid(row=11, column=1, sticky="w"); self._general_field_widgets["stats_time_policy"] = w

        ttk.Label(core, text="experiment.on_missing_asof_snapshot").grid(row=12, column=0, sticky="w")
        self._add_help_mark(core, 12, self._hint_for("experiment.on_missing_asof_snapshot"))
        w = ttk.Combobox(
            core,
            textvariable=self.vars["on_missing_asof_snapshot"],
            values=["disable_stats", "use_base", "raise"],
            state="readonly",
            width=18,
        ); w.grid(row=12, column=1, sticky="w"); self._general_field_widgets["on_missing_asof_snapshot"] = w

        ttk.Label(core, text="experiment.cache_policy").grid(row=13, column=0, sticky="w")
        self._add_help_mark(core, 13, self._hint_for("experiment.cache_policy"))
        w = ttk.Combobox(
            core,
            textvariable=self.vars["cache_policy"],
            values=["off", "dto", "full"],
            state="readonly",
            width=18,
        ); w.grid(row=13, column=1, sticky="w"); self._general_field_widgets["cache_policy"] = w

        ttk.Label(core, text="experiment.graph_dataset_cache_policy").grid(row=14, column=0, sticky="w")
        self._add_help_mark(core, 14, self._hint_for("experiment.graph_dataset_cache_policy"))
        w = ttk.Combobox(
            core,
            textvariable=self.vars["graph_dataset_cache_policy"],
            values=["off", "read", "write", "full"],
            state="readonly",
            width=18,
        ); w.grid(row=14, column=1, sticky="w"); self._general_field_widgets["graph_dataset_cache_policy"] = w

        ttk.Label(core, text="experiment.graph_dataset_cache_dir").grid(row=15, column=0, sticky="w")
        self._add_help_mark(core, 15, self._hint_for("experiment.graph_dataset_cache_dir"))
        w = ttk.Entry(core, textvariable=self.vars["graph_dataset_cache_dir"]); w.grid(row=15, column=1, sticky="ew"); self._general_field_widgets["graph_dataset_cache_dir"] = w

        ttk.Label(core, text="sync --as-of").grid(row=16, column=0, sticky="w")
        self.sync_asof_entry = ttk.Entry(core, textvariable=self.vars["sync_as_of"])
        self.sync_asof_entry.grid(row=16, column=1, sticky="ew")
        self._add_help_mark(core, 16, "Явний as-of timestamp для `sync-stats` (ISO). Якщо порожньо, буде авто-режим на основі подій.", col=2)

        ttk.Label(core, text="backfill step").grid(row=17, column=0, sticky="w")
        self.backfill_step_box = ttk.Combobox(
            core, textvariable=self.vars["backfill_step"], values=["daily", "weekly", "monthly"], state="readonly", width=18
        )
        self.backfill_step_box.grid(row=17, column=1, sticky="w")
        self._add_help_mark(core, 17, "Крок бектесту для `sync-stats-backfill`: daily/weekly/monthly.", col=2)

        ttk.Label(core, text="backfill step-days").grid(row=18, column=0, sticky="w")
        self.backfill_step_days_entry = ttk.Entry(core, textvariable=self.vars["backfill_step_days"])
        self.backfill_step_days_entry.grid(row=18, column=1, sticky="ew")
        self._add_help_mark(core, 18, "Кастомний крок у днях (перевизначає стандартний step).", col=2)

        ttk.Label(core, text="backfill from").grid(row=19, column=0, sticky="w")
        self.backfill_from_entry = ttk.Entry(core, textvariable=self.vars["backfill_from"])
        self.backfill_from_entry.grid(row=19, column=1, sticky="ew")
        self._add_help_mark(core, 19, "Початкова дата backfill (ISO). Якщо порожньо, береться перша подія.", col=2)

        ttk.Label(core, text="backfill to").grid(row=20, column=0, sticky="w")
        self.backfill_to_entry = ttk.Entry(core, textvariable=self.vars["backfill_to"])
        self.backfill_to_entry.grid(row=20, column=1, sticky="ew")
        self._add_help_mark(core, 20, "Кінцева дата backfill (ISO). Якщо порожньо, береться остання подія.", col=2)

        ttk.Label(core, text="extra_args").grid(row=21, column=0, sticky="w")
        self._add_help_mark(core, 21, "Додаткові CLI-аргументи, які будуть додані в кінець команди запуску.")
        ttk.Entry(core, textvariable=self.vars["extra_args"]).grid(row=21, column=1, sticky="ew")

        adv_sections = ttk.Notebook(advanced)
        adv_sections.grid(row=0, column=0, sticky="nsew")
        adv_exp = ttk.Frame(adv_sections, padding=6)
        adv_training = ttk.Frame(adv_sections, padding=6)
        adv_tracking = ttk.Frame(adv_sections, padding=6)
        adv_sections.add(adv_exp, text="Experiment")
        adv_sections.add(adv_training, text="Training")
        adv_sections.add(adv_tracking, text="Tracking")
        for tab in (adv_exp, adv_training, adv_tracking):
            tab.columnconfigure(0, weight=1)
            tab.rowconfigure(0, weight=1)

        self.general_experiment_form = _DynamicForm(
            adv_exp,
            "advanced experiment.* fields",
            self._label_for,
            self._hint_for,
            self._choices_for,
        )
        self.general_experiment_form.grid(row=0, column=0, sticky="nsew")

        self.general_training_form = _DynamicForm(
            adv_training,
            "advanced training.* fields",
            self._label_for,
            self._hint_for,
            self._choices_for,
        )
        self.general_training_form.grid(row=0, column=0, sticky="nsew")

        self.general_tracking_form = _DynamicForm(
            adv_tracking,
            "tracking.* fields",
            self._label_for,
            self._hint_for,
            self._choices_for,
        )
        self.general_tracking_form.grid(row=0, column=0, sticky="nsew")

    def _build_input_tab(self) -> None:
        frame = self.tab_input
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)

        sections = ttk.Notebook(frame)
        sections.grid(row=0, column=0, sticky="nsew")

        source_mapping = ttk.Frame(sections, padding=8)
        yaml_tab = ttk.Frame(sections, padding=8)
        sections.add(source_mapping, text="Source + Runtime")
        sections.add(yaml_tab, text="Features / Policies")

        source_mapping.columnconfigure(0, weight=1)
        source_mapping.rowconfigure(2, weight=1)
        top = ttk.Frame(source_mapping)
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(1, weight=1)
        ttk.Label(top, text="Data config").grid(row=0, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.vars["data_config_path"]).grid(row=0, column=1, sticky="ew")
        ttk.Button(top, text="Browse", command=self._browse_data_config).grid(row=0, column=2, padx=(8, 0))
        ttk.Button(top, text="Apply Data Config", command=self._apply_data_config_clicked).grid(row=0, column=3, padx=(8, 0))
        _Hint(top, "Apply Data Config оновлює лише поля, присутні у data-config (без скидання інших).", row=1, col_span=4)
        ttk.Label(top, text="mapping.adapter").grid(row=2, column=0, sticky="w")
        self.adapter_box = ttk.Combobox(top, textvariable=self.vars["adapter"], values=["xes", "camunda"], state="readonly", width=18)
        self.adapter_box.grid(row=2, column=1, sticky="w")
        _Hint(top, "camunda: runtime-події через camunda_adapter.runtime.*; xes: data.log_path + xes_adapter.*", row=3, col_span=4)

        src_sections = ttk.Notebook(source_mapping)
        src_sections.grid(row=2, column=0, sticky="nsew")

        src_common = ttk.Frame(src_sections, padding=6)
        src_xes = ttk.Frame(src_sections, padding=6)
        src_cam_runtime = ttk.Frame(src_sections, padding=6)
        src_cam_mapping = ttk.Frame(src_sections, padding=6)
        src_sections.add(src_common, text="Common")
        src_sections.add(src_xes, text="XES Mapping")
        src_sections.add(src_cam_runtime, text="Camunda Runtime")
        src_sections.add(src_cam_mapping, text="Camunda Mapping")
        for tab in (src_common, src_xes, src_cam_runtime, src_cam_mapping):
            tab.columnconfigure(0, weight=1)
            tab.rowconfigure(0, weight=1)

        self.input_data_form = _DynamicForm(
            src_common,
            "data.* fields",
            self._label_for,
            self._hint_for,
            self._choices_for,
            on_change=self._refresh_state_controls,
        )
        self.input_data_form.grid(row=0, column=0, sticky="nsew")

        self.input_xes_form = _DynamicForm(
            src_xes,
            "mapping.xes_adapter.* fields",
            self._label_for,
            self._hint_for,
            self._choices_for,
            on_change=self._refresh_state_controls,
        )
        self.input_xes_form.grid(row=0, column=0, sticky="nsew")

        self.input_camunda_runtime_form = _DynamicForm(
            src_cam_runtime,
            "mapping.camunda_adapter.runtime.* fields",
            self._label_for,
            self._hint_for,
            self._choices_for,
            on_change=self._refresh_state_controls,
        )
        self.input_camunda_runtime_form.grid(row=0, column=0, sticky="nsew")

        self.input_camunda_mapping_form = _DynamicForm(
            src_cam_mapping,
            "mapping.camunda_adapter.* (advanced) fields",
            self._label_for,
            self._hint_for,
            self._choices_for,
            on_change=self._refresh_state_controls,
        )
        self.input_camunda_mapping_form.grid(row=0, column=0, sticky="nsew")

        yaml_tab.columnconfigure(0, weight=1)
        yaml_tab.columnconfigure(1, weight=1)
        yaml_tab.rowconfigure(1, weight=1)
        ttk.Label(yaml_tab, text="features (YAML)").grid(row=0, column=0, sticky="w")
        ttk.Label(yaml_tab, text="policies (YAML)").grid(row=0, column=1, sticky="w")
        self.features_text = ScrolledText(yaml_tab, height=14, wrap="word", undo=True, autoseparators=True, maxundo=500)
        self.policies_text = ScrolledText(yaml_tab, height=14, wrap="word", undo=True, autoseparators=True, maxundo=500)
        self.features_text.grid(row=1, column=0, sticky="nsew", padx=(0, 6))
        self.policies_text.grid(row=1, column=1, sticky="nsew")
        self._bind_text_shortcuts(self.features_text)
        self._bind_text_shortcuts(self.policies_text)

    def _build_eopkg_tab(self) -> None:
        frame = self.tab_eopkg
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)

        sections = ttk.Notebook(frame)
        sections.grid(row=0, column=0, sticky="nsew")

        backend_tab = ttk.Frame(sections, padding=8)
        structure_tab = ttk.Frame(sections, padding=8)
        sync_stats_tab = ttk.Frame(sections, padding=8)
        mapping_tab = ttk.Frame(sections, padding=8)
        sections.add(backend_tab, text="Backend")
        sections.add(structure_tab, text="Structure")
        sections.add(sync_stats_tab, text="Sync Stats")
        sections.add(mapping_tab, text="Graph Mapping")

        backend_tab.columnconfigure(0, weight=1)
        backend_tab.rowconfigure(0, weight=1)
        self.eopkg_backend_form = _DynamicForm(
            backend_tab,
            "knowledge_graph / storage fields",
            self._label_for,
            self._hint_for,
            self._choices_for,
            on_change=self._refresh_state_controls,
        )
        self.eopkg_backend_form.grid(row=0, column=0, sticky="nsew")

        structure_tab.columnconfigure(0, weight=1)
        structure_tab.rowconfigure(0, weight=1)
        self.eopkg_structure_form = _DynamicForm(
            structure_tab,
            "camunda structure fields",
            self._label_for,
            self._hint_for,
            self._choices_for,
            on_change=self._refresh_state_controls,
        )
        self.eopkg_structure_form.grid(row=0, column=0, sticky="nsew")

        sync_stats_tab.columnconfigure(0, weight=1)
        sync_stats_tab.rowconfigure(0, weight=1)
        self.sync_stats_form = _DynamicForm(
            sync_stats_tab,
            "sync_stats (flattened fields)",
            self._label_for,
            self._hint_for,
            self._choices_for,
            on_change=self._refresh_preview,
        )
        self.sync_stats_form.grid(row=0, column=0, sticky="nsew")

        mapping_tab.columnconfigure(0, weight=1)
        mapping_tab.rowconfigure(2, weight=1)
        gateway_bar = ttk.Frame(mapping_tab)
        gateway_bar.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        ttk.Label(gateway_bar, text="gateways").grid(row=0, column=0, sticky="w", padx=(0, 8))
        self.gateway_mode_box = ttk.Combobox(
            gateway_bar,
            textvariable=self.vars["gateway_mode"],
            values=["preserve", "collapse_for_prediction"],
            state="readonly",
            width=28,
        )
        self.gateway_mode_box.grid(row=0, column=1, sticky="w")
        self._add_help_mark(
            gateway_bar,
            0,
            "Projection for GNN/mask: preserve keeps BPMN gateway nodes; collapse_for_prediction links task-to-task through gateways for XES logs.",
            col=2,
        )
        ttk.Label(mapping_tab, text="mapping.graph_feature_mapping (YAML)").grid(row=1, column=0, sticky="w")
        self.graph_mapping_text = ScrolledText(mapping_tab, height=16, wrap="word", undo=True, autoseparators=True, maxundo=500)
        self.graph_mapping_text.grid(row=2, column=0, sticky="nsew")
        self._bind_text_shortcuts(self.graph_mapping_text)

    def _build_model_tab(self) -> None:
        frame = self.tab_model
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)
        self.model_form = _DynamicForm(
            frame,
            "model.* parameters",
            self._label_for,
            self._hint_for,
            self._choices_for,
            on_change=self._refresh_preview,
        )
        self.model_form.grid(row=0, column=0, sticky="nsew")

    def _build_run_tab(self) -> None:
        frame = self.tab_run
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(5, weight=1)

        ttk.Label(frame, text="Command preview").grid(row=0, column=0, sticky="w", pady=(2, 0))
        self.preview = tk.Text(frame, height=3, wrap="word")
        self.preview.grid(row=1, column=0, sticky="ew")

        controls = ttk.Frame(frame)
        controls.grid(row=2, column=0, sticky="w", pady=(8, 6))
        self.run_btn = ttk.Button(controls, text="Run", command=self._run_clicked)
        self.stop_btn = ttk.Button(controls, text="Stop", command=self._stop_clicked, state="disabled")
        ttk.Button(controls, text="Clear Log", command=self._clear_log).grid(row=0, column=0, padx=(0, 6))
        ttk.Button(controls, text="Copy Sel", command=self._copy_log_selection).grid(row=0, column=1, padx=(0, 6))
        ttk.Button(controls, text="Copy All", command=self._copy_log_all).grid(row=0, column=2, padx=(0, 6))
        self.run_btn.grid(row=0, column=3, padx=(0, 6))
        self.stop_btn.grid(row=0, column=4)

        status = ttk.LabelFrame(frame, text="Run Status", padding=8)
        status.grid(row=3, column=0, sticky="ew", pady=(0, 6))
        status.columnconfigure(0, weight=1)
        status.rowconfigure(0, weight=1)
        status_canvas = tk.Canvas(status, height=240, highlightthickness=0)
        status_scroll = ttk.Scrollbar(status, orient="vertical", command=status_canvas.yview)
        status_inner = ttk.Frame(status_canvas)
        status_window_id = status_canvas.create_window((0, 0), window=status_inner, anchor="nw")
        status_canvas.configure(yscrollcommand=status_scroll.set)
        status_canvas.grid(row=0, column=0, sticky="ew")
        status_scroll.grid(row=0, column=1, sticky="ns")
        status_inner.columnconfigure(1, weight=1)

        def _sync_status_scrollregion(_event: tk.Event) -> None:
            status_canvas.configure(scrollregion=status_canvas.bbox("all"))

        def _resize_status_inner(event: tk.Event) -> None:
            status_canvas.itemconfigure(status_window_id, width=event.width)

        def _on_status_mousewheel(event: tk.Event) -> str:
            delta = 0
            if hasattr(event, "delta") and event.delta:
                delta = -1 * int(event.delta / 120)
            elif getattr(event, "num", None) == 4:
                delta = -1
            elif getattr(event, "num", None) == 5:
                delta = 1
            if delta != 0:
                status_canvas.yview_scroll(delta, "units")
            return "break"

        status_inner.bind("<Configure>", _sync_status_scrollregion)
        status_canvas.bind("<Configure>", _resize_status_inner)
        for widget in (status_canvas, status_inner):
            widget.bind("<MouseWheel>", _on_status_mousewheel)
            widget.bind("<Button-4>", _on_status_mousewheel)
            widget.bind("<Button-5>", _on_status_mousewheel)

        self.run_stage_var = tk.StringVar(value="Stage: idle")
        self.run_percent_var = tk.StringVar(value="Overall: 0.0%")
        self.run_eta_var = tk.StringVar(value="ETA: --:--")
        ttk.Label(status_inner, textvariable=self.run_stage_var).grid(row=0, column=0, sticky="w")
        ttk.Label(status_inner, textvariable=self.run_percent_var).grid(row=0, column=1, sticky="e")
        self.stage_progress_bar = ttk.Progressbar(status_inner, orient="horizontal", mode="determinate", maximum=100.0)
        self.stage_progress_bar.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(4, 4))
        self.overall_progress_bar = ttk.Progressbar(status_inner, orient="horizontal", mode="determinate", maximum=100.0)
        self.overall_progress_bar.grid(row=2, column=0, columnspan=2, sticky="ew")
        ttk.Label(status_inner, textvariable=self.run_eta_var).grid(row=3, column=0, columnspan=2, sticky="w", pady=(4, 0))
        self.run_child_mem_var = tk.StringVar(value="Run RAM: --")
        self.run_system_mem_var = tk.StringVar(value="System RAM: --")
        self.run_disk_cache_var = tk.StringVar(value="Disk cache: --")
        ttk.Label(status_inner, textvariable=self.run_child_mem_var).grid(row=4, column=0, sticky="w", pady=(4, 0))
        ttk.Label(status_inner, textvariable=self.run_system_mem_var).grid(row=4, column=1, sticky="e", pady=(4, 0))
        ttk.Label(status_inner, textvariable=self.run_disk_cache_var).grid(row=5, column=0, columnspan=2, sticky="w", pady=(2, 0))

        self.stage_table = ttk.LabelFrame(status_inner, text="Stages", padding=6)
        self.stage_table.grid(row=6, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        self.stage_table.columnconfigure(1, weight=1)
        self._stage_row_next = 0
        for stage in RUN_STAGE_ORDER:
            self._ensure_stage_row(stage)

        ttk.Label(frame, text="Execution log").grid(row=4, column=0, sticky="w")
        self.log_text = ScrolledText(frame, height=14, wrap="word")
        self.log_text.grid(row=5, column=0, sticky="nsew")
        self.log_text.bind("<Control-c>", lambda _e: self._copy_log_selection())
        self.log_text.bind("<Control-a>", self._select_all_log)

    def _browse_config(self) -> None:
        path = filedialog.askopenfilename(
            title="Select base experiment config",
            initialdir=str((ROOT_DIR / "configs" / "experiments").resolve()),
            filetypes=[("YAML", "*.yaml *.yml"), ("All files", "*.*")],
        )
        if path:
            self.vars["config_path"].set(path)
            self._load_base_config_into_form()

    def _browse_data_config(self) -> None:
        path = filedialog.askopenfilename(
            title="Select data config",
            initialdir=str((ROOT_DIR / "configs" / "data").resolve()),
            filetypes=[("YAML", "*.yaml *.yml"), ("All files", "*.*")],
        )
        if path:
            self.vars["data_config_path"].set(path)

    def _apply_data_config_clicked(self) -> None:
        raw = str(self.vars.get("data_config_path").get()).strip()
        if not raw:
            messagebox.showwarning("Data config", "Select data config path first.")
            return
        path = Path(raw)
        if not path.is_absolute():
            path = (ROOT_DIR / path).resolve()
        if not path.exists():
            messagebox.showerror("Data config", f"Config not found: {path}")
            return
        try:
            loaded = load_yaml_with_includes(path)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Data config", str(exc))
            return
        if not isinstance(loaded, dict):
            messagebox.showerror("Data config", "Loaded data config must be a mapping.")
            return
        self._apply_data_config_values(loaded)
        self._refresh_state_controls()
        self._refresh_preview()
        messagebox.showinfo("Data config", f"Applied values from: {path}")

    def _apply_data_config_values(self, loaded: Dict[str, Any]) -> None:
        data_flat = _flatten(_deep_get(loaded, "data", {}), "data")
        mapping_flat = _flatten(_deep_get(loaded, "mapping", {}), "mapping")

        for key, value in data_flat.items():
            target = self.input_data_form.entries.get(key)
            if target is not None:
                target.set(_to_text(value))

        if "mapping.adapter" in mapping_flat:
            self.vars["adapter"].set(str(mapping_flat["mapping.adapter"]))

        for key, value in mapping_flat.items():
            if key.startswith("mapping.xes_adapter."):
                target = self.input_xes_form.entries.get(key)
                if target is not None:
                    target.set(_to_text(value))
                continue
            if key.startswith("mapping.camunda_adapter.runtime."):
                target = self.input_camunda_runtime_form.entries.get(key)
                if target is not None:
                    target.set(_to_text(value))
                continue
            if key.startswith("mapping.camunda_adapter.") and not key.startswith("mapping.camunda_adapter.structure."):
                target = self.input_camunda_mapping_form.entries.get(key)
                if target is not None:
                    target.set(_to_text(value))

        if _deep_has(loaded, "mapping.features"):
            self._set_text_block(self.features_text, _deep_get(loaded, "mapping.features", []))
        if _deep_has(loaded, "policies"):
            self._set_text_block(self.policies_text, _deep_get(loaded, "policies", {}))

    def _load_base_config(self) -> Dict[str, Any]:
        raw = str(self.vars["config_path"].get()).strip()
        if not raw:
            raise ValueError("Base config path is empty.")
        path = Path(raw)
        if not path.is_absolute():
            path = (ROOT_DIR / path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {path}")
        self._base_config_path = path
        loaded = load_yaml_with_includes(path)
        if not isinstance(loaded, dict):
            raise ValueError("Loaded config must be a mapping.")
        return loaded

    @staticmethod
    def _set_text_block(widget: ScrolledText, payload: Any) -> None:
        widget.delete("1.0", tk.END)
        widget.insert(tk.END, _to_text(payload))

    @staticmethod
    def _get_text_block(widget: ScrolledText) -> Any:
        return _parse_text(widget.get("1.0", tk.END))

    @staticmethod
    def _bind_text_shortcuts(widget: tk.Text) -> None:
        def _event(name: str) -> str:
            try:
                widget.event_generate(name)
            except tk.TclError:
                return "break"
            return "break"

        def _select_all() -> str:
            widget.tag_add(tk.SEL, "1.0", tk.END)
            widget.mark_set(tk.INSERT, "1.0")
            widget.see(tk.INSERT)
            return "break"

        widget.bind("<Control-a>", lambda _e: _select_all())
        widget.bind("<Control-A>", lambda _e: _select_all())
        widget.bind("<Control-c>", lambda _e: _event("<<Copy>>"))
        widget.bind("<Control-C>", lambda _e: _event("<<Copy>>"))
        widget.bind("<Control-v>", lambda _e: _event("<<Paste>>"))
        widget.bind("<Control-V>", lambda _e: _event("<<Paste>>"))
        widget.bind("<Control-x>", lambda _e: _event("<<Cut>>"))
        widget.bind("<Control-X>", lambda _e: _event("<<Cut>>"))
        widget.bind("<Control-z>", lambda _e: _event("<<Undo>>"))
        widget.bind("<Control-Z>", lambda _e: _event("<<Undo>>"))
        widget.bind("<Control-y>", lambda _e: _event("<<Redo>>"))
        widget.bind("<Control-Y>", lambda _e: _event("<<Redo>>"))

    def _load_base_config_into_form(self) -> None:
        try:
            self._base_config = self._load_base_config()
        except Exception as exc:
            messagebox.showerror("Config", str(exc))
            return

        exp = dict(_deep_get(self._base_config, "experiment", {}) or {})
        train = dict(_deep_get(self._base_config, "training", {}) or {})
        mapping = dict(_deep_get(self._base_config, "mapping", {}) or {})
        self.vars["project"].set(str(exp.get("project", "")))
        self.vars["experiment_name"].set(str(exp.get("name", "")))
        self.vars["mode"].set(str(exp.get("mode", "train")))
        self.vars["fraction"].set(str(exp.get("fraction", 1.0)))
        self.vars["split_strategy"].set(str(exp.get("split_strategy", "temporal")))
        self.vars["train_ratio"].set(str(exp.get("train_ratio", 0.7)))
        self._set_split_ratio_vars(exp.get("split_ratio", [0.7, 0.2, 0.1]))
        self.vars["stats_time_policy"].set(str(exp.get("stats_time_policy", "strict_asof")))
        self.vars["on_missing_asof_snapshot"].set(str(exp.get("on_missing_asof_snapshot", "disable_stats")))
        self.vars["cache_policy"].set(str(exp.get("cache_policy", "full")))
        self.vars["graph_dataset_cache_policy"].set(
            self._normalize_graph_dataset_cache_policy(
                exp.get("graph_dataset_cache_policy", exp.get("graph_cache_policy", "off"))
            )
        )
        self.vars["graph_dataset_cache_dir"].set(str(exp.get("graph_dataset_cache_dir", exp.get("graph_cache_dir", ".cache/graph_datasets"))))
        self.vars["seed"].set(str(self._base_config.get("seed", 42)))
        self.vars["adapter"].set(str(mapping.get("adapter", "camunda")))
        self.vars["gateway_mode"].set(
            self._resolve_gateway_mode_from_mapping(mapping.get("graph_feature_mapping", {}))
        )

        data_flat = _flatten(_deep_get(self._base_config, "data", {}), "data")
        mapping_flat = _flatten(mapping, "mapping")
        model_flat = _flatten(_deep_get(self._base_config, "model", {}), "model")
        sync_stats_flat = _flatten(_deep_get(self._base_config, "sync_stats", {}), "sync_stats")
        input_common_flat: Dict[str, Any] = {}
        input_xes_flat: Dict[str, Any] = {}
        input_cam_runtime_flat: Dict[str, Any] = {}
        input_cam_mapping_flat: Dict[str, Any] = {}
        eopkg_backend_flat: Dict[str, Any] = {}
        eopkg_structure_flat: Dict[str, Any] = {}
        input_common_flat.update(data_flat)
        for key, value in mapping_flat.items():
            if key.startswith("mapping.graph_feature_mapping") or key.startswith("mapping.features"):
                continue
            if key == "mapping.adapter":
                continue
            if key.startswith("mapping.knowledge_graph."):
                eopkg_backend_flat[key] = value
            elif key.startswith("mapping.xes_adapter."):
                input_xes_flat[key] = value
            elif key.startswith("mapping.camunda_adapter.runtime."):
                input_cam_runtime_flat[key] = value
            elif self._is_eopkg_mapping_path(key):
                eopkg_structure_flat[key] = value
            elif key.startswith("mapping.camunda_adapter."):
                input_cam_mapping_flat[key] = value
            else:
                input_common_flat[key] = value
        # include pooled params so UI exposes full parameter set, even if current base config misses some keys
        for path, meta in self._pool_meta.items():
            if meta.section == "data" and path not in input_common_flat:
                input_common_flat[path] = ""
            if meta.section == "mapping":
                if path.startswith("mapping.graph_feature_mapping") or path.startswith("mapping.features"):
                    continue
                if path == "mapping.adapter":
                    continue
                if path.startswith("mapping.knowledge_graph."):
                    if path not in eopkg_backend_flat:
                        eopkg_backend_flat[path] = ""
                elif path.startswith("mapping.xes_adapter."):
                    if path not in input_xes_flat:
                        input_xes_flat[path] = ""
                elif path.startswith("mapping.camunda_adapter.runtime."):
                    if path not in input_cam_runtime_flat:
                        input_cam_runtime_flat[path] = ""
                elif self._is_eopkg_mapping_path(path):
                    if path not in eopkg_structure_flat:
                        eopkg_structure_flat[path] = ""
                elif path.startswith("mapping.camunda_adapter."):
                    if path not in input_cam_mapping_flat:
                        input_cam_mapping_flat[path] = ""
                else:
                    if path not in input_common_flat:
                        input_common_flat[path] = ""
        self.input_data_form.set_fields(self._ordered_field_dict(input_common_flat))
        self.input_xes_form.set_fields(self._ordered_field_dict(input_xes_flat))
        self.input_camunda_runtime_form.set_fields(self._ordered_field_dict(input_cam_runtime_flat))
        self.input_camunda_mapping_form.set_fields(self._ordered_field_dict(input_cam_mapping_flat))
        self.eopkg_backend_form.set_fields(self._ordered_field_dict(eopkg_backend_flat))
        self.eopkg_structure_form.set_fields(self._ordered_field_dict(eopkg_structure_flat))
        sync_all = _merge_catalog_section_fields(
            base_fields=sync_stats_flat,
            pool_meta=self._pool_meta,
            catalog=self._catalog,
            section="sync_stats",
        )
        self.sync_stats_form.set_fields(self._ordered_field_dict(sync_all))

        model_all: Dict[str, Any] = dict(model_flat)
        # Build model form from catalog (not only from scanned pool) so newly added
        # model parameters appear even before they exist in any base config.
        for path, meta in self._catalog.items():
            if meta.section == "model" and path not in model_all:
                model_all[path] = _deep_get(self._base_config, path, self._catalog_default(path, ""))
        self.model_form.set_fields(self._ordered_field_dict(model_all))

        extra_experiment: Dict[str, Any] = {}
        extra_training: Dict[str, Any] = {}
        extra_tracking: Dict[str, Any] = {}
        # Build advanced General fields from catalog (not only from scanned pool),
        # so newly added parameters appear in UI even before they exist in base configs.
        for path, meta in self._catalog.items():
            if meta.section not in {"experiment", "training", "seed", "tracking"}:
                continue
            if path in {
                "experiment.project",
                "experiment.name",
                "experiment.mode",
                "experiment.fraction",
                "experiment.split_strategy",
                "experiment.train_ratio",
                "experiment.split_ratio",
                "experiment.stats_time_policy",
                "experiment.on_missing_asof_snapshot",
                "experiment.cache_policy",
                "experiment.graph_dataset_cache_policy",
                "experiment.graph_dataset_cache_dir",
                "training.retrain",
                "seed",
            }:
                continue
            value = _deep_get(self._base_config, path, self._catalog_default(path, ""))
            if path.startswith("experiment."):
                extra_experiment[path] = value
            elif path.startswith("training."):
                extra_training[path] = value
            elif path.startswith("tracking."):
                extra_tracking[path] = value
        self.general_experiment_form.set_fields(self._ordered_field_dict(extra_experiment))
        self.general_training_form.set_fields(self._ordered_field_dict(extra_training))
        self.general_tracking_form.set_fields(self._ordered_field_dict(extra_tracking))

        features_payload = _deep_get(self._base_config, "mapping.features", None)
        if features_payload is None:
            features_payload = _deep_get(self._base_config, "features", [])
        self._set_text_block(self.features_text, features_payload)
        self._set_text_block(self.policies_text, _deep_get(self._base_config, "policies", {}))
        self._set_text_block(self.graph_mapping_text, _deep_get(self._base_config, "mapping.graph_feature_mapping", {}))

        self._refresh_state_controls()
        self._refresh_preview()

    def _refresh_state_controls(self) -> None:
        mode = str(self.vars["mode"].get()).strip()
        adapter = str(self.vars["adapter"].get()).strip().lower()
        if str(self.vars["gateway_mode"].get()).strip() not in {"preserve", "collapse_for_prediction"}:
            self.vars["gateway_mode"].set(self._default_gateway_mode_for_adapter(adapter))

        is_sync_stats = mode == "sync-stats"
        is_backfill = mode == "sync-stats-backfill"
        self.sync_asof_entry.configure(state="normal" if is_sync_stats else "disabled")
        backfill_state = "normal" if is_backfill else "disabled"
        self.backfill_step_box.configure(state="readonly" if is_backfill else "disabled")
        self.backfill_step_days_entry.configure(state=backfill_state)
        self.backfill_from_entry.configure(state=backfill_state)
        self.backfill_to_entry.configure(state=backfill_state)

        is_sync_mode = mode in {"sync-topology", "sync-stats", "sync-stats-backfill"}
        for name, widget in self._general_field_widgets.items():
            if isinstance(widget, ttk.Combobox):
                widget.configure(state="disabled" if is_sync_mode else "readonly")
            else:
                widget.configure(state="disabled" if is_sync_mode else "normal")
        if mode in {"sync-topology", "sync-stats", "sync-stats-backfill"}:
            train_ratio_widget = self._general_field_widgets.get("train_ratio")
            if train_ratio_widget is not None:
                train_ratio_widget.configure(state="normal")
        for widget in getattr(self, "_split_ratio_widgets", []):
            widget.configure(state="disabled" if is_sync_mode else "normal")
        self.model_form.set_enabled_by_prefix(enabled_prefixes=("model.",) if not is_sync_mode else tuple(), disabled_prefixes=tuple())
        for form in self._iter_general_advanced_forms():
            form.set_enabled_by_prefix(
                enabled_prefixes=("experiment.", "training.", "tracking.", "seed") if not is_sync_mode else tuple(),
                disabled_prefixes=("experiment.mode",),
            )
        self.sync_stats_form.set_enabled_by_prefix(
            enabled_prefixes=("sync_stats.",) if mode in {"sync-stats", "sync-stats-backfill", "train", "eval_drift", "eval_cross_dataset"} else tuple(),
            disabled_prefixes=tuple(),
        )
        text_state = "normal" if not is_sync_mode else "disabled"
        for widget in (self.features_text, self.policies_text, self.graph_mapping_text):
            widget.configure(state=text_state)
        self.gateway_mode_box.configure(state="readonly" if not is_sync_mode else "disabled")

        if adapter == "camunda":
            self.input_data_form.set_enabled_by_prefix(enabled_prefixes=("data.",), disabled_prefixes=tuple())
            self.input_xes_form.set_enabled_by_prefix(enabled_prefixes=tuple(), disabled_prefixes=tuple())
            self.input_camunda_runtime_form.set_enabled_by_prefix(enabled_prefixes=("mapping.camunda_adapter.runtime.",), disabled_prefixes=tuple())
            self.input_camunda_mapping_form.set_enabled_by_prefix(
                enabled_prefixes=("mapping.camunda_adapter.",),
                disabled_prefixes=("mapping.camunda_adapter.runtime.", "mapping.camunda_adapter.structure.", "mapping.camunda_adapter.subprocess_graph_mode"),
            )
            if "data.log_path" in self.input_data_form.widgets:
                self.input_data_form.widgets["data.log_path"].configure(state="disabled")
            eopkg_backend_enabled = ("mapping.knowledge_graph.",)
            eopkg_backend_disabled = tuple()
            eopkg_structure_enabled = ("mapping.camunda_adapter.structure.", "mapping.camunda_adapter.subprocess_graph_mode")
            eopkg_structure_disabled = tuple()
        else:
            self.input_data_form.set_enabled_by_prefix(enabled_prefixes=("data.",), disabled_prefixes=tuple())
            self.input_xes_form.set_enabled_by_prefix(enabled_prefixes=("mapping.xes_adapter.",), disabled_prefixes=tuple())
            self.input_camunda_runtime_form.set_enabled_by_prefix(enabled_prefixes=tuple(), disabled_prefixes=tuple())
            self.input_camunda_mapping_form.set_enabled_by_prefix(enabled_prefixes=tuple(), disabled_prefixes=tuple())
            eopkg_backend_enabled = ("mapping.knowledge_graph.",)
            eopkg_backend_disabled = tuple()
            eopkg_structure_enabled = tuple()
            eopkg_structure_disabled = ("mapping.camunda_adapter.",)
        self.eopkg_backend_form.set_enabled_by_prefix(enabled_prefixes=eopkg_backend_enabled, disabled_prefixes=eopkg_backend_disabled)
        self.eopkg_structure_form.set_enabled_by_prefix(enabled_prefixes=eopkg_structure_enabled, disabled_prefixes=eopkg_structure_disabled)

        # runtime-source refinement for Camunda
        if adapter == "camunda":
            runtime_source = ""
            runtime_key = "mapping.camunda_adapter.runtime.runtime_source"
            if runtime_key in self.input_camunda_runtime_form.entries:
                runtime_source = str(self.input_camunda_runtime_form.entries[runtime_key].get()).strip().lower()
            bpmn_source = ""
            bpmn_source_key = "mapping.camunda_adapter.structure.bpmn_source"
            if bpmn_source_key in self.eopkg_structure_form.entries:
                bpmn_source = str(self.eopkg_structure_form.entries[bpmn_source_key].get()).strip().lower()
            for key, widget in self.input_camunda_runtime_form.widgets.items():
                if ".runtime.sql_dir" in key and runtime_source in {"files"}:
                    widget.configure(state="disabled")
                if ".runtime.export_dir" in key and runtime_source not in {"", "files"}:
                    widget.configure(state="disabled")
                if ".runtime.connections_file" in key and runtime_source in {"files"}:
                    widget.configure(state="disabled")
                if ".runtime.connection_profile" in key and runtime_source in {"files"}:
                    widget.configure(state="disabled")
                if ".runtime.mssql." in key and runtime_source in {"files"}:
                    widget.configure(state="disabled")
            for key, widget in self.eopkg_structure_form.widgets.items():
                if ".structure.files." in key and bpmn_source not in {"", "files"}:
                    widget.configure(state="disabled")
                if ".structure.sql_dir" in key and bpmn_source in {"files"}:
                    widget.configure(state="disabled")
                if ".structure.connections_file" in key and bpmn_source in {"files"}:
                    widget.configure(state="disabled")
                if ".structure.connection_profile" in key and bpmn_source in {"files"}:
                    widget.configure(state="disabled")
                if ".structure.mssql." in key and bpmn_source in {"files"}:
                    widget.configure(state="disabled")

    def _build_preview_command(self) -> str:
        mode = str(self.vars["mode"].get()).strip()
        cmd = [sys.executable, "main.py"]
        if mode == "sync-stats":
            cmd += ["sync-stats", "--config", "<temp_ui_config.yaml>"]
            as_of = str(self.vars["sync_as_of"].get()).strip()
            if as_of:
                cmd += ["--as-of", as_of]
        elif mode == "sync-stats-backfill":
            cmd += ["sync-stats-backfill", "--config", "<temp_ui_config.yaml>", "--step", str(self.vars["backfill_step"].get()).strip()]
            if str(self.vars["backfill_step_days"].get()).strip():
                cmd += ["--step-days", str(self.vars["backfill_step_days"].get()).strip()]
            if str(self.vars["backfill_from"].get()).strip():
                cmd += ["--from", str(self.vars["backfill_from"].get()).strip()]
            if str(self.vars["backfill_to"].get()).strip():
                cmd += ["--to", str(self.vars["backfill_to"].get()).strip()]
        elif mode == "sync-topology":
            cmd += ["sync-topology", "--config", "<temp_ui_config.yaml>"]
        else:
            cmd += ["--config", "<temp_ui_config.yaml>"]

        extra = str(self.vars["extra_args"].get()).strip()
        if extra:
            try:
                cmd.extend(shlex.split(extra))
            except ValueError:
                cmd.append(extra)
        return " ".join(cmd)

    def _catalog_default(self, path: str, fallback: Any = "") -> Any:
        meta = self._catalog_meta(path)
        if meta is not None and meta.default not in ("", None):
            return meta.default
        return fallback

    @staticmethod
    def _to_bool(value: Any, fallback: bool = False) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            txt = value.strip().lower()
            if txt in {"true", "1", "yes", "on"}:
                return True
            if txt in {"false", "0", "no", "off"}:
                return False
        return fallback

    def _default_for_entry(self, path: str) -> Any:
        value = self._catalog_default(path, "")
        if value in ("", None):
            meta = self._catalog_meta(path)
            if meta is not None and meta.enum:
                return meta.enum[0]
        return value

    def _iter_input_forms(self) -> List[_DynamicForm]:
        return [
            self.input_data_form,
            self.input_xes_form,
            self.input_camunda_runtime_form,
            self.input_camunda_mapping_form,
        ]

    def _iter_general_advanced_forms(self) -> List[_DynamicForm]:
        return [
            self.general_experiment_form,
            self.general_training_form,
            self.general_tracking_form,
        ]

    def _fill_defaults(self) -> None:
        # Core vars
        self.vars["mode"].set(str(self._catalog_default("experiment.mode", "train")))
        self.vars["project"].set(str(self._catalog_default("experiment.project", "")))
        self.vars["experiment_name"].set(str(self._catalog_default("experiment.name", "")))
        self.vars["fraction"].set(str(self._catalog_default("experiment.fraction", "1.0")))
        self.vars["split_strategy"].set(str(self._catalog_default("experiment.split_strategy", "temporal")))
        self.vars["train_ratio"].set(str(self._catalog_default("experiment.train_ratio", "0.7")))
        self._set_split_ratio_vars(self._catalog_default("experiment.split_ratio", [0.7, 0.2, 0.1]))
        self.vars["stats_time_policy"].set(str(self._catalog_default("experiment.stats_time_policy", "strict_asof")))
        self.vars["on_missing_asof_snapshot"].set(str(self._catalog_default("experiment.on_missing_asof_snapshot", "disable_stats")))
        self.vars["cache_policy"].set(str(self._catalog_default("experiment.cache_policy", "full")))
        self.vars["graph_dataset_cache_policy"].set(str(self._catalog_default("experiment.graph_dataset_cache_policy", "off")))
        self.vars["graph_dataset_cache_dir"].set(str(self._catalog_default("experiment.graph_dataset_cache_dir", ".cache/graph_datasets")))
        self.vars["seed"].set(str(self._catalog_default("seed", "42")))
        self.vars["adapter"].set(str(self._catalog_default("mapping.adapter", "camunda")))
        self.vars["gateway_mode"].set(self._default_gateway_mode_for_adapter(self.vars["adapter"].get()))
        self.vars["sync_as_of"].set("")
        self.vars["backfill_step"].set("weekly")
        self.vars["backfill_step_days"].set("")
        self.vars["backfill_from"].set("")
        self.vars["backfill_to"].set("")
        self.vars["extra_args"].set("")

        for form in self._iter_input_forms():
            for key, var in form.entries.items():
                var.set(_to_text(self._default_for_entry(key)))
        for key, var in self.eopkg_backend_form.entries.items():
            var.set(_to_text(self._default_for_entry(key)))
        for key, var in self.eopkg_structure_form.entries.items():
            var.set(_to_text(self._default_for_entry(key)))
        for key, var in self.sync_stats_form.entries.items():
            var.set(_to_text(self._default_for_entry(key)))
        for key, var in self.model_form.entries.items():
            var.set(_to_text(self._default_for_entry(key)))
        for form in self._iter_general_advanced_forms():
            for key, var in form.entries.items():
                var.set(_to_text(self._default_for_entry(key)))

        self._set_text_block(self.features_text, [])
        self._set_text_block(self.policies_text, {})
        self._set_text_block(self.graph_mapping_text, {})
        self._refresh_state_controls()
        self._refresh_preview()

    def _refresh_preview(self) -> None:
        self.preview.delete("1.0", tk.END)
        self.preview.insert(tk.END, self._build_preview_command())

    def _compose_config(self) -> Dict[str, Any]:
        cfg = deepcopy(self._base_config)
        _deep_set(cfg, "experiment.project", str(self.vars["project"].get()).strip())
        _deep_set(cfg, "experiment.name", str(self.vars["experiment_name"].get()).strip())
        _deep_set(cfg, "experiment.mode", str(self.vars["mode"].get()).strip())
        _deep_set(cfg, "experiment.fraction", float(str(self.vars["fraction"].get()).strip() or "1.0"))
        _deep_set(cfg, "experiment.split_strategy", str(self.vars["split_strategy"].get()).strip())
        _deep_set(cfg, "experiment.train_ratio", float(str(self.vars["train_ratio"].get()).strip() or "0.7"))
        _deep_set(cfg, "experiment.split_ratio", self._resolve_split_ratio())
        _deep_set(cfg, "experiment.stats_time_policy", str(self.vars["stats_time_policy"].get()).strip())
        _deep_set(cfg, "experiment.on_missing_asof_snapshot", str(self.vars["on_missing_asof_snapshot"].get()).strip())
        _deep_set(cfg, "experiment.cache_policy", str(self.vars["cache_policy"].get()).strip())
        _deep_set(
            cfg,
            "experiment.graph_dataset_cache_policy",
            self._normalize_graph_dataset_cache_policy(self.vars["graph_dataset_cache_policy"].get()),
        )
        _deep_set(cfg, "experiment.graph_dataset_cache_dir", str(self.vars["graph_dataset_cache_dir"].get()).strip())
        _deep_set(cfg, "seed", int(float(str(self.vars["seed"].get()).strip() or "42")))
        _deep_set(cfg, "mapping.adapter", str(self.vars["adapter"].get()).strip())

        for form in self._iter_input_forms():
            for key, value in form.get_values().items():
                if self._is_blank(value) and not _deep_has(self._base_config, key):
                    continue
                _deep_set(cfg, key, value)
        for key, value in self.eopkg_backend_form.get_values().items():
            if self._is_blank(value) and not _deep_has(self._base_config, key):
                continue
            _deep_set(cfg, key, value)
        for key, value in self.eopkg_structure_form.get_values().items():
            if self._is_blank(value) and not _deep_has(self._base_config, key):
                continue
            _deep_set(cfg, key, value)
        for key, value in self.sync_stats_form.get_values().items():
            if self._is_blank(value) and not _deep_has(self._base_config, key):
                continue
            _deep_set(cfg, key, value)
        for key, value in self.model_form.get_values().items():
            if self._is_blank(value) and not _deep_has(self._base_config, key):
                continue
            _deep_set(cfg, key, value)
        for form in self._iter_general_advanced_forms():
            for key, value in form.get_values().items():
                if self._is_blank(value) and not _deep_has(self._base_config, key):
                    continue
                _deep_set(cfg, key, value)

        features_payload = self._get_text_block(self.features_text)
        _deep_set(cfg, "mapping.features", features_payload)
        if "features" in cfg:
            _deep_set(cfg, "features", features_payload)
        _deep_set(cfg, "policies", self._get_text_block(self.policies_text))
        graph_mapping_payload = self._get_text_block(self.graph_mapping_text)
        if not isinstance(graph_mapping_payload, dict):
            graph_mapping_payload = {}
        projection = graph_mapping_payload.get("topology_projection", {})
        if not isinstance(projection, dict):
            projection = {}
        gateway_mode = str(self.vars["gateway_mode"].get()).strip()
        if gateway_mode not in {"preserve", "collapse_for_prediction"}:
            gateway_mode = self._default_gateway_mode_for_adapter(self.vars["adapter"].get())
        projection["gateway_mode"] = gateway_mode
        graph_mapping_payload["topology_projection"] = projection
        _deep_set(cfg, "mapping.graph_feature_mapping", graph_mapping_payload)
        return cfg

    @staticmethod
    def _now_iso_utc() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _format_saved_at(raw: Any) -> str:
        text = str(raw).strip()
        if not text:
            return "-"
        normalized = text.replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError:
            return text
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone().strftime("%Y-%m-%d %H:%M:%S %z")

    @staticmethod
    def _extract_preset_payload(entry: Any) -> Dict[str, Any] | None:
        if not isinstance(entry, dict):
            return None
        payload = entry.get("payload")
        if isinstance(payload, dict):
            return payload
        # Backward compatibility: old format stored payload directly.
        return entry

    @staticmethod
    def _extract_preset_last_saved_at(entry: Any) -> str:
        if not isinstance(entry, dict):
            return ""
        value = entry.get("last_saved_at")
        return str(value).strip() if value is not None else ""

    def _refresh_preset_choices(self) -> None:
        if not hasattr(self, "preset_name_box"):
            return
        names = sorted(str(key) for key in self._presets.keys())
        self.preset_name_box.configure(values=names)

    def _sync_preset_saved_at_label(self, *, highlight: bool = False) -> None:
        name = str(self.vars["preset_name"].get()).strip()
        entry = self._presets.get(name)
        saved_at = self._extract_preset_last_saved_at(entry)
        text = self._format_saved_at(saved_at) if saved_at else "-"
        self.preset_saved_at_var.set(f"Last saved: {text}")
        if not hasattr(self, "preset_saved_at_label"):
            return
        base_color = "#2e7d32" if saved_at else "#616161"
        self.preset_saved_at_label.configure(fg=base_color)
        if not highlight:
            return
        if self._preset_highlight_after_id is not None:
            try:
                self.root.after_cancel(self._preset_highlight_after_id)
            except Exception:
                pass
        self.preset_saved_at_label.configure(fg="#ef6c00")
        self._preset_highlight_after_id = self.root.after(
            1800,
            lambda: self.preset_saved_at_label.configure(fg=base_color),
        )

    def _save_state(self) -> None:
        payload = {
            "vars": {key: var.get() for key, var in self.vars.items()},
            "input_data_form": {key: var.get() for key, var in self.input_data_form.entries.items()},
            "input_xes_form": {key: var.get() for key, var in self.input_xes_form.entries.items()},
            "input_camunda_runtime_form": {key: var.get() for key, var in self.input_camunda_runtime_form.entries.items()},
            "input_camunda_mapping_form": {key: var.get() for key, var in self.input_camunda_mapping_form.entries.items()},
            "eopkg_backend_form": {key: var.get() for key, var in self.eopkg_backend_form.entries.items()},
            "eopkg_structure_form": {key: var.get() for key, var in self.eopkg_structure_form.entries.items()},
            "sync_stats_form": {key: var.get() for key, var in self.sync_stats_form.entries.items()},
            "model_form": {key: var.get() for key, var in self.model_form.entries.items()},
            "general_experiment_form": {key: var.get() for key, var in self.general_experiment_form.entries.items()},
            "general_training_form": {key: var.get() for key, var in self.general_training_form.entries.items()},
            "general_tracking_form": {key: var.get() for key, var in self.general_tracking_form.entries.items()},
            "features_text": self.features_text.get("1.0", tk.END),
            "policies_text": self.policies_text.get("1.0", tk.END),
            "graph_mapping_text": self.graph_mapping_text.get("1.0", tk.END),
        }
        _write_json(STATE_PATH, payload)

    def _load_state(self) -> None:
        if not isinstance(self._state, dict):
            return
        vars_payload = self._state.get("vars", {})
        if isinstance(vars_payload, dict):
            for key, value in vars_payload.items():
                if key in self.vars:
                    self.vars[key].set(value)
            if "split_ratio" in vars_payload and not all(
                key in vars_payload for key in ("split_ratio_train", "split_ratio_val", "split_ratio_test")
            ):
                self._set_split_ratio_vars(vars_payload.get("split_ratio"))

    def _apply_payload_to_forms(self, payload: Any) -> None:
        if not isinstance(payload, dict):
            return
        vars_payload = payload.get("vars", {})
        if isinstance(vars_payload, dict):
            for key, value in vars_payload.items():
                if key in self.vars:
                    self.vars[key].set(value)
            if "split_ratio" in vars_payload and not all(
                key in vars_payload for key in ("split_ratio_train", "split_ratio_val", "split_ratio_test")
            ):
                self._set_split_ratio_vars(vars_payload.get("split_ratio"))
        for payload_key, form in (
            ("input_data_form", self.input_data_form),
            ("input_xes_form", self.input_xes_form),
            ("input_camunda_runtime_form", self.input_camunda_runtime_form),
            ("input_camunda_mapping_form", self.input_camunda_mapping_form),
        ):
            block = payload.get(payload_key, {})
            if isinstance(block, dict):
                for key, value in block.items():
                    if key in form.entries:
                        form.entries[key].set(value)
        # Legacy single input block compatibility
        data_payload = payload.get("data_form", {})
        if isinstance(data_payload, dict):
            for key, value in data_payload.items():
                for form in self._iter_input_forms():
                    if key in form.entries:
                        form.entries[key].set(value)
        eopkg_backend_payload = payload.get("eopkg_backend_form", {})
        if isinstance(eopkg_backend_payload, dict):
            for key, value in eopkg_backend_payload.items():
                if key in self.eopkg_backend_form.entries:
                    self.eopkg_backend_form.entries[key].set(value)
        eopkg_structure_payload = payload.get("eopkg_structure_form", {})
        if isinstance(eopkg_structure_payload, dict):
            for key, value in eopkg_structure_payload.items():
                if key in self.eopkg_structure_form.entries:
                    self.eopkg_structure_form.entries[key].set(value)
        legacy_eopkg_payload = payload.get("eopkg_form", {})
        if isinstance(legacy_eopkg_payload, dict):
            for key, value in legacy_eopkg_payload.items():
                if key in self.eopkg_backend_form.entries:
                    self.eopkg_backend_form.entries[key].set(value)
                if key in self.eopkg_structure_form.entries:
                    self.eopkg_structure_form.entries[key].set(value)
        model_payload = payload.get("model_form", {})
        if isinstance(model_payload, dict):
            for key, value in model_payload.items():
                if key in self.model_form.entries:
                    self.model_form.entries[key].set(value)
        sync_stats_payload = payload.get("sync_stats_form", {})
        if isinstance(sync_stats_payload, dict):
            for key, value in sync_stats_payload.items():
                if key in self.sync_stats_form.entries:
                    self.sync_stats_form.entries[key].set(value)
        for payload_key, form in (
            ("general_experiment_form", self.general_experiment_form),
            ("general_training_form", self.general_training_form),
            ("general_tracking_form", self.general_tracking_form),
        ):
            block = payload.get(payload_key, {})
            if isinstance(block, dict):
                for key, value in block.items():
                    if key in form.entries:
                        form.entries[key].set(value)
        # Legacy advanced block compatibility
        extra_payload = payload.get("general_extra_form", {})
        if isinstance(extra_payload, dict):
            for key, value in extra_payload.items():
                for form in self._iter_general_advanced_forms():
                    if key in form.entries:
                        form.entries[key].set(value)
        if isinstance(payload.get("features_text"), str):
            self._set_text_block(self.features_text, payload.get("features_text"))
        if isinstance(payload.get("policies_text"), str):
            self._set_text_block(self.policies_text, payload.get("policies_text"))
        if isinstance(payload.get("graph_mapping_text"), str):
            self._set_text_block(self.graph_mapping_text, payload.get("graph_mapping_text"))
        if isinstance(vars_payload, dict) and "gateway_mode" not in vars_payload:
            self.vars["gateway_mode"].set(self._resolve_gateway_mode_from_mapping(self._get_text_block(self.graph_mapping_text)))

    def _save_preset(self) -> None:
        name = str(self.vars["preset_name"].get()).strip()
        if not name:
            messagebox.showwarning("Preset", "Enter preset name.")
            return
        payload = {
            "vars": {key: var.get() for key, var in self.vars.items()},
            "input_data_form": {key: var.get() for key, var in self.input_data_form.entries.items()},
            "input_xes_form": {key: var.get() for key, var in self.input_xes_form.entries.items()},
            "input_camunda_runtime_form": {key: var.get() for key, var in self.input_camunda_runtime_form.entries.items()},
            "input_camunda_mapping_form": {key: var.get() for key, var in self.input_camunda_mapping_form.entries.items()},
            "eopkg_backend_form": {key: var.get() for key, var in self.eopkg_backend_form.entries.items()},
            "eopkg_structure_form": {key: var.get() for key, var in self.eopkg_structure_form.entries.items()},
            "sync_stats_form": {key: var.get() for key, var in self.sync_stats_form.entries.items()},
            "model_form": {key: var.get() for key, var in self.model_form.entries.items()},
            "general_experiment_form": {key: var.get() for key, var in self.general_experiment_form.entries.items()},
            "general_training_form": {key: var.get() for key, var in self.general_training_form.entries.items()},
            "general_tracking_form": {key: var.get() for key, var in self.general_tracking_form.entries.items()},
            "features_text": self.features_text.get("1.0", tk.END),
            "policies_text": self.policies_text.get("1.0", tk.END),
            "graph_mapping_text": self.graph_mapping_text.get("1.0", tk.END),
        }
        self._presets[name] = {
            "payload": payload,
            "last_saved_at": self._now_iso_utc(),
        }
        _write_json(PRESETS_PATH, self._presets)
        self._refresh_preset_choices()
        self.vars["preset_name"].set(name)
        self._sync_preset_saved_at_label(highlight=True)
        messagebox.showinfo("Preset", f"Saved preset: {name}")

    def _load_preset(self) -> None:
        name = str(self.vars["preset_name"].get()).strip()
        entry = self._presets.get(name)
        payload = self._extract_preset_payload(entry)
        if not isinstance(payload, dict):
            messagebox.showwarning("Preset", f"Preset not found: {name}")
            return
        vars_payload = payload.get("vars", {})
        if isinstance(vars_payload, dict):
            cfg_path = vars_payload.get("config_path")
            current_cfg_path = str(self.vars.get("config_path").get()).strip() if "config_path" in self.vars else ""
            if cfg_path is not None and "config_path" in self.vars and not current_cfg_path:
                self.vars["config_path"].set(cfg_path)

        self._load_base_config_into_form()
        self._apply_payload_to_forms(payload)
        if isinstance(vars_payload, dict):
            for key, value in vars_payload.items():
                if key == "config_path":
                    continue
                target_var = self.vars.get(str(key))
                if target_var is None:
                    continue
                try:
                    target_var.set(value)
                except Exception:  # noqa: BLE001
                    continue
        self._refresh_state_controls()
        self._refresh_preview()
        self._sync_preset_saved_at_label(highlight=True)

    def _delete_preset(self) -> None:
        name = str(self.vars["preset_name"].get()).strip()
        if name in self._presets:
            del self._presets[name]
            _write_json(PRESETS_PATH, self._presets)
            self._refresh_preset_choices()
            self._sync_preset_saved_at_label()
            messagebox.showinfo("Preset", f"Deleted preset: {name}")
        else:
            messagebox.showwarning("Preset", f"Preset not found: {name}")

    def _build_run_command(self, temp_cfg: Path) -> list[str]:
        mode = str(self.vars["mode"].get()).strip()
        cmd = [sys.executable, "main.py"]
        if mode == "sync-stats":
            cmd += ["sync-stats", "--config", str(temp_cfg)]
            as_of = str(self.vars["sync_as_of"].get()).strip()
            if as_of:
                cmd += ["--as-of", as_of]
        elif mode == "sync-stats-backfill":
            cmd += ["sync-stats-backfill", "--config", str(temp_cfg), "--step", str(self.vars["backfill_step"].get()).strip()]
            if str(self.vars["backfill_step_days"].get()).strip():
                cmd += ["--step-days", str(self.vars["backfill_step_days"].get()).strip()]
            if str(self.vars["backfill_from"].get()).strip():
                cmd += ["--from", str(self.vars["backfill_from"].get()).strip()]
            if str(self.vars["backfill_to"].get()).strip():
                cmd += ["--to", str(self.vars["backfill_to"].get()).strip()]
        elif mode == "sync-topology":
            cmd += ["sync-topology", "--config", str(temp_cfg)]
        else:
            cmd += ["--config", str(temp_cfg)]

        extra = str(self.vars["extra_args"].get()).strip()
        if extra:
            cmd.extend(shlex.split(extra))
        return cmd

    @staticmethod
    def _is_blank(value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, str):
            return value.strip() == ""
        if isinstance(value, (list, tuple, set, dict)):
            return len(value) == 0
        return False

    def _is_path_active(self, path: str, mode: str, adapter: str) -> bool:
        is_sync_mode = mode in {"sync-topology", "sync-stats", "sync-stats-backfill"}
        if is_sync_mode and (path.startswith("model.") or path.startswith("training.") or path.startswith("tracking.")):
            return False
        if adapter == "camunda":
            if path.startswith("mapping.xes_adapter.") or path == "data.log_path":
                return False
        if adapter == "xes":
            if path.startswith("mapping.camunda_adapter."):
                return False
        return True

    @staticmethod
    def _is_path_like_field(path: str) -> bool:
        tokens = path.split(".")
        last = tokens[-1] if tokens else path
        return (
            last.endswith("_path")
            or last.endswith("_dir")
            or last in {"path", "log_path", "connections_file", "catalog_file", "bpmn_dir", "export_dir", "sql_dir"}
        )

    def _resolve_path_candidates(self, raw_path: str) -> List[Path]:
        p = Path(str(raw_path).strip()).expanduser()
        candidates: List[Path] = [p]
        if not p.is_absolute():
            candidates.append(ROOT_DIR / p)
            if self._base_config_path is not None:
                candidates.append(self._base_config_path.parent / p)
        return _unique_paths(candidates)

    def _resolve_existing_path(self, raw_path: str) -> Path | None:
        for candidate in self._resolve_path_candidates(raw_path):
            if candidate.exists():
                try:
                    return candidate.resolve()
                except OSError:
                    return candidate
        return None

    def _discover_xes_examples(self, limit: int = 8) -> List[str]:
        roots = [ROOT_DIR / "Data", ROOT_DIR.parent / "Data"]
        out: List[str] = []
        for root in roots:
            if not root.exists():
                continue
            for item in root.rglob("*.xes"):
                out.append(str(item))
                if len(out) >= limit:
                    return out
        return out

    def _normalize_and_validate_data_paths(self, cfg: Dict[str, Any]) -> List[str]:
        errors: List[str] = []
        adapter = str(_deep_get(cfg, "mapping.adapter", "")).strip().lower()
        if adapter != "xes":
            return errors
        raw_log_path = str(_deep_get(cfg, "data.log_path", "")).strip()
        if raw_log_path == "":
            return errors
        resolved = self._resolve_existing_path(raw_log_path)
        if resolved is not None:
            _deep_set(cfg, "data.log_path", str(resolved))
            return errors
        tried = [str(path) for path in self._resolve_path_candidates(raw_log_path)]
        examples = self._discover_xes_examples(limit=6)
        msg = [f"XES файл не знайдено: {raw_log_path}", "Перевірені шляхи:"] + tried
        if examples:
            msg.extend(["Знайдені XES у системі (приклади):"] + examples)
        errors.append("\n".join(msg))
        return errors

    def _validate_config_against_catalog(self, cfg: Dict[str, Any]) -> List[str]:
        errors: List[str] = []
        mode = str(_deep_get(cfg, "experiment.mode", "")).strip()
        adapter = str(_deep_get(cfg, "mapping.adapter", "")).strip().lower()
        for path, meta in self._catalog.items():
            if not self._is_path_active(path, mode, adapter):
                continue

            should_check_required = mode in meta.required_in_modes if meta.required_in_modes else False
            if not should_check_required and meta.required_when:
                required_match = True
                for cond_path, expected in meta.required_when.items():
                    current = _deep_get(cfg, cond_path, None)
                    if isinstance(expected, list):
                        if current not in expected:
                            required_match = False
                            break
                    else:
                        if current != expected:
                            required_match = False
                            break
                should_check_required = required_match
            if should_check_required:
                current = _deep_get(cfg, path, None)
                if self._is_blank(current):
                    errors.append(f"Missing required field: {path}")

            if meta.enum:
                current = _deep_get(cfg, path, None)
                if not self._is_blank(current):
                    if self._is_path_like_field(path):
                        continue
                    if path not in self._STRICT_ENUM_PATHS:
                        # For most fields enum is only a UI suggestion pool,
                        # not a strict constraint (e.g. experiment.fraction).
                        continue
                    current_text = str(current).strip()
                    if isinstance(current, bool):
                        current_text = "true" if current else "false"
                    elif current_text.lower() in {"true", "false"}:
                        current_text = current_text.lower()

                    allowed = [str(v).strip() for v in meta.enum]

                    if path == "experiment.cache_policy":
                        def _normalize_cache_policy_token(token: Any) -> str:
                            text = str(token).strip().lower()
                            if text in {"off", "false", "none", "disabled"}:
                                return "off"
                            return text

                        current_norm = _normalize_cache_policy_token(current_text)
                        allowed_norm = [_normalize_cache_policy_token(token) for token in allowed]
                        if current_norm not in allowed_norm:
                            errors.append(f"Invalid enum value for {path}: {current} (allowed: {', '.join(meta.enum)})")
                        continue

                    allowed_norm = [item.lower() if item.lower() in {"true", "false"} else item for item in allowed]
                    current_norm = current_text.lower() if current_text.lower() in {"true", "false"} else current_text
                    if current_norm not in allowed_norm:
                        errors.append(f"Invalid enum value for {path}: {current} (allowed: {', '.join(meta.enum)})")
        return errors

    def _run_clicked(self) -> None:
        if self._process is not None:
            messagebox.showwarning("Run", "Process already running.")
            return
        try:
            cfg = self._compose_config()
        except Exception as exc:
            messagebox.showerror("Config", f"Failed to compose config: {exc}")
            return
        path_errors = self._normalize_and_validate_data_paths(cfg)
        if path_errors:
            messagebox.showerror("Validation", "\n\n".join(path_errors[:5]))
            return
        errors = self._validate_config_against_catalog(cfg)
        if errors:
            messagebox.showerror("Validation", "\n".join(errors[:20]))
            return

        temp_dir = self._base_config_path.parent if self._base_config_path is not None else ROOT_DIR
        temp_dir.mkdir(parents=True, exist_ok=True)
        fd, temp_path = tempfile.mkstemp(prefix="ui_run_", suffix=".yaml", dir=str(temp_dir))
        os.close(fd)
        self._temp_config_path = Path(temp_path)
        self._temp_config_path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")

        cmd = self._build_run_command(self._temp_config_path)
        self._append_log(f"$ {' '.join(cmd)}\n")
        self._save_state()
        self._reset_progress_ui()
        self._progress_started_ts = time.time()
        if hasattr(self, "run_stage_var"):
            self.run_stage_var.set("Stage: launching run")
        self.run_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")

        def _worker() -> None:
            try:
                child_env = os.environ.copy()
                child_env["BPM_PROGRESS_EVENTS"] = "1"
                child_env["PYTHONUNBUFFERED"] = "1"
                self._process = subprocess.Popen(
                    cmd,
                    cwd=str(ROOT_DIR),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    env=child_env,
                )
                assert self._process.stdout is not None
                for line in self._process.stdout:
                    self._queue.put(line)
                rc = self._process.wait()
                self._queue.put(f"\n[exit code: {rc}]\n")
            except Exception as exc:
                self._queue.put(f"\n[run failed] {exc}\n")
            finally:
                self._process = None
                self._queue.put("__FINISHED__")

        threading.Thread(target=_worker, daemon=True).start()

    def _stop_clicked(self) -> None:
        if self._process is None:
            return
        try:
            self._process.terminate()
        except Exception:
            pass

    def _clear_log(self) -> None:
        self.log_text.delete("1.0", tk.END)

    def _copy_log_selection(self) -> str:
        try:
            selected = self.log_text.get(tk.SEL_FIRST, tk.SEL_LAST)
        except tk.TclError:
            selected = self.log_text.get("1.0", tk.END).strip()
        if selected:
            self.root.clipboard_clear()
            self.root.clipboard_append(selected)
        return "break"

    def _copy_log_all(self) -> None:
        text = self.log_text.get("1.0", tk.END).strip()
        if not text:
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(text)

    def _select_all_log(self, _event: tk.Event) -> str:
        self.log_text.tag_add(tk.SEL, "1.0", tk.END)
        self.log_text.mark_set(tk.INSERT, "1.0")
        self.log_text.see(tk.INSERT)
        return "break"

    def _append_log(self, text: str) -> None:
        self.log_text.insert(tk.END, text)
        self.log_text.see(tk.END)

    @staticmethod
    def _format_bytes(num_bytes: int) -> str:
        value = float(max(0, int(num_bytes)))
        units = ["B", "KB", "MB", "GB", "TB"]
        idx = 0
        while value >= 1024.0 and idx < len(units) - 1:
            value /= 1024.0
            idx += 1
        return f"{value:.1f} {units[idx]}"

    @staticmethod
    def _dir_size_bytes(path: Path) -> int:
        total = 0
        if not path.exists():
            return 0
        for item in path.rglob("*"):
            if item.is_file():
                try:
                    total += int(item.stat().st_size)
                except OSError:
                    continue
        return total

    def _update_resource_telemetry(self) -> None:
        now = time.time()
        if (now - self._telemetry_last_update_ts) < 1.0:
            return
        self._telemetry_last_update_ts = now

        child_rss = 0
        if self._process is not None:
            try:
                child_proc = psutil.Process(self._process.pid)
                child_rss = int(child_proc.memory_info().rss)
                for sub_proc in child_proc.children(recursive=True):
                    try:
                        child_rss += int(sub_proc.memory_info().rss)
                    except Exception:
                        continue
            except Exception:
                child_rss = 0
        if hasattr(self, "run_child_mem_var"):
            self.run_child_mem_var.set(f"Run RAM (tree): {self._format_bytes(child_rss)}")

        try:
            vm = psutil.virtual_memory()
            system_used = int(vm.used)
            system_total = int(vm.total)
            percent = float(vm.percent)
            system_text = f"System RAM: {self._format_bytes(system_used)} / {self._format_bytes(system_total)} ({percent:.1f}%)"
        except Exception:
            system_text = "System RAM: --"
        if hasattr(self, "run_system_mem_var"):
            self.run_system_mem_var.set(system_text)

        if (now - self._telemetry_last_disk_update_ts) >= 4.0:
            self._telemetry_last_disk_update_ts = now
            cache_var = self.vars.get("graph_dataset_cache_dir")
            cache_raw = cache_var.get().strip() if isinstance(cache_var, tk.Variable) else ".cache/graph_datasets"
            cache_dir = Path(str(cache_raw) or ".cache/graph_datasets")
            if not cache_dir.is_absolute():
                cache_dir = (ROOT_DIR / cache_dir).resolve()
            disk_text = f"Disk cache: {self._format_bytes(self._dir_size_bytes(cache_dir))} ({cache_dir})"
            if hasattr(self, "run_disk_cache_var"):
                self.run_disk_cache_var.set(disk_text)

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    @staticmethod
    def _format_eta(seconds: float | None) -> str:
        if seconds is None or seconds <= 0:
            return "--:--"
        total = int(seconds)
        mins, sec = divmod(total, 60)
        hrs, mins = divmod(mins, 60)
        if hrs > 0:
            return f"{hrs:02d}:{mins:02d}:{sec:02d}"
        return f"{mins:02d}:{sec:02d}"

    @staticmethod
    def _format_elapsed(seconds: float | None) -> str:
        if seconds is None or seconds <= 0:
            return "00:00"
        total = int(seconds)
        mins, sec = divmod(total, 60)
        hrs, mins = divmod(mins, 60)
        if hrs > 0:
            return f"{hrs:02d}:{mins:02d}:{sec:02d}"
        return f"{mins:02d}:{sec:02d}"

    @staticmethod
    def _stage_title(stage: str) -> str:
        raw = str(stage).strip()
        if not raw:
            return "idle"
        return STAGE_TITLE_UA.get(raw, raw.replace("_", " "))

    def _ensure_stage_row(self, stage: str) -> None:
        stage_key = str(stage).strip()
        if not stage_key:
            return
        if stage_key in self._stage_widgets:
            return
        if not hasattr(self, "stage_table"):
            return
        row = int(getattr(self, "_stage_row_next", 0))
        title_var = tk.StringVar(value=self._stage_title(stage_key))
        meta_var = tk.StringVar(value="0.0% | elapsed 00:00 | ETA --:--")
        ttk.Label(self.stage_table, textvariable=title_var, width=30).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=2)
        bar = ttk.Progressbar(self.stage_table, orient="horizontal", mode="determinate", maximum=100.0)
        bar.grid(row=row, column=1, sticky="ew", pady=2)
        ttk.Label(self.stage_table, textvariable=meta_var, width=32).grid(row=row, column=2, sticky="e", padx=(8, 0), pady=2)
        self._stage_widgets[stage_key] = {
            "title_var": title_var,
            "meta_var": meta_var,
            "bar": bar,
        }
        self._stage_row_next = row + 1

    def _update_stage_row(self, stage: str, *, percent: float, elapsed: float | None, eta: float | None) -> None:
        stage_key = str(stage).strip()
        if not stage_key:
            return
        self._ensure_stage_row(stage_key)
        widgets = self._stage_widgets.get(stage_key)
        if not isinstance(widgets, dict):
            return
        bar = widgets.get("bar")
        meta_var = widgets.get("meta_var")
        title_var = widgets.get("title_var")
        if isinstance(bar, ttk.Progressbar):
            bar.configure(value=max(0.0, min(100.0, float(percent))))
        if isinstance(title_var, tk.StringVar):
            title_var.set(self._stage_title(stage_key))
        if isinstance(meta_var, tk.StringVar):
            meta_var.set(
                f"{max(0.0, min(100.0, float(percent))):.1f}% | elapsed {self._format_elapsed(elapsed)} | ETA {self._format_eta(eta)}"
            )

    def _reset_progress_ui(self) -> None:
        self._progress_started_ts = None
        self._stage_progress = {}
        self._stage_runtime = {}
        self._current_stage = ""
        self._last_status_text = ""
        self._last_exit_code = None
        self._progress_event_seen = False
        self._telemetry_last_update_ts = 0.0
        self._telemetry_last_disk_update_ts = 0.0
        if hasattr(self, "run_stage_var"):
            self.run_stage_var.set("Stage: idle")
        if hasattr(self, "run_percent_var"):
            self.run_percent_var.set("Overall: 0.0%")
        if hasattr(self, "run_eta_var"):
            self.run_eta_var.set("ETA: --:--")
        if hasattr(self, "run_child_mem_var"):
            self.run_child_mem_var.set("Run RAM: --")
        if hasattr(self, "run_system_mem_var"):
            self.run_system_mem_var.set("System RAM: --")
        if hasattr(self, "run_disk_cache_var"):
            self.run_disk_cache_var.set("Disk cache: --")
        if hasattr(self, "stage_progress_bar"):
            self.stage_progress_bar.configure(value=0.0)
        if hasattr(self, "overall_progress_bar"):
            self.overall_progress_bar.configure(value=0.0)
        for stage in RUN_STAGE_ORDER:
            self._ensure_stage_row(stage)
        for stage, widgets in self._stage_widgets.items():
            bar = widgets.get("bar")
            meta_var = widgets.get("meta_var")
            title_var = widgets.get("title_var")
            if isinstance(bar, ttk.Progressbar):
                bar.configure(value=0.0)
            if isinstance(meta_var, tk.StringVar):
                meta_var.set("0.0% | elapsed 00:00 | ETA --:--")
            if isinstance(title_var, tk.StringVar):
                title_var.set(self._stage_title(stage))

    def _compute_overall_percent(self) -> float:
        weighted = 0.0
        total_weight = 0.0
        for stage, progress in self._stage_progress.items():
            weight = float(RUN_STAGE_WEIGHTS.get(stage, 0.01))
            total_weight += weight
            weighted += weight * max(0.0, min(1.0, float(progress)))
        if total_weight <= 1e-9:
            return 0.0
        return max(0.0, min(100.0, (weighted / total_weight) * 100.0))

    def _apply_progress_event(self, event: Dict[str, Any]) -> None:
        self._progress_event_seen = True
        stage = str(event.get("stage", "unknown")).strip() or "unknown"
        status = str(event.get("status", "update")).strip().lower() or "update"
        message = str(event.get("message", "")).strip()
        level = str(event.get("level", "info")).strip().lower() or "info"
        current = self._safe_float(event.get("current", 0.0), default=0.0)
        total = self._safe_float(event.get("total", 0.0), default=0.0)
        event_ts = self._safe_float(event.get("ts", time.time()), default=time.time())
        event_percent = self._safe_float(event.get("percent", -1.0), default=-1.0)
        if event_percent < 0.0:
            event_percent = (current / total * 100.0) if total > 0.0 else 0.0
        if status == "done":
            event_percent = 100.0
        event_percent = max(0.0, min(100.0, event_percent))

        self._current_stage = stage
        now_ts = time.time()
        if self._progress_started_ts is None:
            self._progress_started_ts = now_ts

        runtime = self._stage_runtime.get(stage)
        if not isinstance(runtime, dict):
            runtime = {
                "start_ts": now_ts,
                "percent": 0.0,
                "elapsed": 0.0,
                "eta": None,
                "status": "idle",
            }
            self._stage_runtime[stage] = runtime
        if status == "start":
            runtime["start_ts"] = now_ts
        start_ts = float(runtime.get("start_ts", now_ts))
        elapsed = max(0.0, now_ts - start_ts)
        eta_seconds: float | None = None
        if event_percent > 0.0 and event_percent < 100.0:
            ratio = event_percent / 100.0
            eta_seconds = (elapsed / ratio) - elapsed if ratio > 1e-9 else None
        elif status == "done":
            eta_seconds = 0.0
        runtime["percent"] = event_percent
        runtime["elapsed"] = elapsed
        runtime["eta"] = eta_seconds
        runtime["status"] = status

        if status in {"start"}:
            self._stage_progress[stage] = 0.0
        elif status in {"done"}:
            self._stage_progress[stage] = 1.0
        elif status in {"error"}:
            self._stage_progress[stage] = max(self._stage_progress.get(stage, 0.0), event_percent / 100.0)
        else:
            self._stage_progress[stage] = max(0.0, min(1.0, event_percent / 100.0))

        stage_title = self._stage_title(stage)
        if hasattr(self, "run_stage_var"):
            if message:
                self.run_stage_var.set(f"Stage: {stage_title} ({event_percent:.1f}%) | {message}")
            else:
                self.run_stage_var.set(f"Stage: {stage_title} ({event_percent:.1f}%)")
        if hasattr(self, "stage_progress_bar"):
            self.stage_progress_bar.configure(value=event_percent)
        self._update_stage_row(stage, percent=event_percent, elapsed=elapsed, eta=eta_seconds)

        overall_percent = self._compute_overall_percent()
        if hasattr(self, "overall_progress_bar"):
            self.overall_progress_bar.configure(value=overall_percent)
        if hasattr(self, "run_percent_var"):
            self.run_percent_var.set(f"Overall: {overall_percent:.1f}%")

        eta_text = "--:--"
        if self._progress_started_ts is not None and overall_percent > 0.0:
            elapsed = max(0.0, time.time() - self._progress_started_ts)
            ratio = overall_percent / 100.0
            remaining = (elapsed / ratio) - elapsed if ratio > 1e-9 else None
            eta_text = self._format_eta(remaining)
        if hasattr(self, "run_eta_var"):
            self.run_eta_var.set(f"ETA: {eta_text}")

        # Keep execution log concise in UI mode: show status and warnings/errors only.
        if message:
            ts_text = time.strftime("%H:%M:%S", time.localtime(event_ts))
            progress_part = ""
            if total > 0:
                if abs(current - round(current)) < 1e-9:
                    current_text = str(int(round(current)))
                else:
                    current_text = f"{current:.2f}"
                if abs(total - round(total)) < 1e-9:
                    total_text = str(int(round(total)))
                else:
                    total_text = f"{total:.2f}"
                progress_part = f" | {current_text}/{total_text} ({event_percent:.1f}%)"
            elif event_percent > 0.0:
                progress_part = f" | {event_percent:.1f}%"
            text = f"[{ts_text}] [{level}] {stage}: {message}{progress_part}\n"
            if text != self._last_status_text:
                self._append_log(text)
                self._last_status_text = text

    def _should_append_log_line(self, line: str) -> bool:
        text = str(line).strip()
        if text == "":
            return False
        if text.startswith(PROGRESS_EVENT_PREFIX):
            return False
        if not self._progress_event_seen:
            # Fallback mode for commands that do not emit structured progress yet.
            if "it/s" in text and "%" in text:
                return False
            return True
        upper = text.upper()
        if "ERROR" in upper or "WARNING" in upper:
            return True
        if text.startswith("Traceback") or text.startswith("[exit code:") or text.startswith("[run failed]"):
            return True
        if text.startswith("=== Final Test Metrics ===") or text.startswith("test_"):
            return True
        # Suppress noisy tqdm redraw lines in UI.
        if "it/s" in text and "%" in text:
            return False
        # Keep only concise info statuses in UI log; detailed INFO stays in CLI terminal.
        return False

    def _poll_queue(self) -> None:
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            if item == "__FINISHED__":
                self.run_btn.configure(state="normal")
                self.stop_btn.configure(state="disabled")
                if (self._last_exit_code is None or self._last_exit_code == 0) and self._compute_overall_percent() < 100.0 and self._current_stage:
                    runtime = self._stage_runtime.get(self._current_stage, {})
                    elapsed = None
                    if isinstance(runtime, dict):
                        elapsed = self._safe_float(runtime.get("elapsed", 0.0), default=0.0)
                    self._stage_progress[self._current_stage] = 1.0
                    self._update_stage_row(self._current_stage, percent=100.0, elapsed=elapsed, eta=0.0)
                    self.overall_progress_bar.configure(value=100.0)
                    self.run_percent_var.set("Overall: 100.0%")
                    self.run_eta_var.set("ETA: 00:00")
                if self._temp_config_path is not None:
                    try:
                        self._temp_config_path.unlink(missing_ok=True)
                    except OSError:
                        pass
                    self._temp_config_path = None
                continue
            text_item = str(item)
            if text_item.startswith(PROGRESS_EVENT_PREFIX):
                try:
                    payload = json.loads(text_item[len(PROGRESS_EVENT_PREFIX):].strip())
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    self._apply_progress_event(payload)
                continue
            stripped = text_item.strip()
            if stripped.startswith("[exit code:") and stripped.endswith("]"):
                try:
                    self._last_exit_code = int(stripped[len("[exit code:") : -1].strip())
                except ValueError:
                    self._last_exit_code = None
            if self._should_append_log_line(text_item):
                self._append_log(text_item)
        self._update_resource_telemetry()
        self.root.after(150, self._poll_queue)

    def _on_close(self) -> None:
        self._save_state()
        if self._process is not None:
            if not messagebox.askyesno("Exit", "Process is running. Stop and exit?"):
                return
            try:
                self._process.terminate()
            except Exception:
                pass
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Desktop UI for BPM experiment configuration.")
    parser.add_argument("--config", default="", help="Optional default base config path.")
    args = parser.parse_args(argv)
    ui = ExperimentUI(default_config=args.config or None)
    ui.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
