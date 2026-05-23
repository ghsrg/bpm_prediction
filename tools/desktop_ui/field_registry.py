from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


UI_LEVELS = {"project_setup", "experiment_run", "advanced"}


@dataclass(frozen=True)
class DesktopFieldMeta:
    path: str
    section: str
    label: str
    description: str
    affects: str
    default: Any
    enum: tuple[str, ...]
    ui_level: str
    ui_tab: str
    ui_group: str
    ui_order: int
    required_in_modes: tuple[str, ...]
    required_when: dict[str, Any]
    active_when: dict[str, Any]
    emit_when: dict[str, Any]
    runtime_consumers: tuple[str, ...]


class DesktopFieldRegistry:
    def __init__(self, fields: dict[str, DesktopFieldMeta]) -> None:
        self._fields = dict(sorted(fields.items(), key=lambda item: (item[1].ui_level, item[1].ui_tab, item[1].ui_group, item[1].ui_order, item[0])))

    @classmethod
    def load(cls, *, catalog_path: Path, audit_matrix_path: Path | None = None) -> "DesktopFieldRegistry":
        catalog_payload = yaml.safe_load(catalog_path.read_text(encoding="utf-8")) or {}
        catalog_fields = catalog_payload.get("fields", {})
        audit_rows = _load_audit_rows(audit_matrix_path) if audit_matrix_path else {}

        fields: dict[str, DesktopFieldMeta] = {}
        for path, payload in catalog_fields.items():
            payload = payload or {}
            audit = audit_rows.get(path, {})
            ui = payload.get("ui") or {}
            required_when = _normalize_condition_map(payload.get("required_when") or {})
            active_when = _normalize_condition_map(payload.get("active_when") or required_when)
            emit_when = _normalize_condition_map(payload.get("emit_when") or active_when)
            ui_level = str(payload.get("ui_level") or audit.get("proposed_level") or _fallback_level(path, payload)).strip()
            if ui_level not in UI_LEVELS:
                ui_level = _fallback_level(path, payload)

            fields[path] = DesktopFieldMeta(
                path=str(payload.get("path") or path),
                section=str(payload.get("section") or path.split(".", 1)[0]),
                label=str(payload.get("label") or path),
                description=str(payload.get("description") or ""),
                affects=str(payload.get("affects") or ""),
                default=payload.get("default", ""),
                enum=tuple(str(item) for item in (payload.get("enum") or [])),
                ui_level=ui_level,
                ui_tab=str(payload.get("ui_tab") or audit.get("proposed_tab") or _title_from_level(ui_level)),
                ui_group=str(payload.get("ui_group") or audit.get("proposed_group") or ui.get("group") or "General"),
                ui_order=int(ui.get("order") or audit.get("current_ui_order") or 1000),
                required_in_modes=tuple(str(item) for item in (payload.get("required_in_modes") or [])),
                required_when=required_when,
                active_when=active_when,
                emit_when=emit_when,
                runtime_consumers=tuple(_split_consumers(audit.get("runtime_consumers", ""))),
            )
        return cls(fields)

    def __iter__(self):
        return iter(self._fields.values())

    def field(self, path: str) -> DesktopFieldMeta:
        return self._fields[path]

    def fields_for_level(self, ui_level: str) -> list[DesktopFieldMeta]:
        return [field for field in self._fields.values() if field.ui_level == ui_level]

    def grouped_fields(self, ui_level: str) -> dict[str, list[DesktopFieldMeta]]:
        groups: dict[str, list[DesktopFieldMeta]] = {}
        for field in self.fields_for_level(ui_level):
            groups.setdefault(field.ui_group, []).append(field)
        return groups

    def is_active(self, path: str, values: dict[str, Any]) -> bool:
        return _matches_conditions(self.field(path).active_when, values)

    def is_emitted(self, path: str, values: dict[str, Any]) -> bool:
        return _matches_conditions(self.field(path).emit_when, values)


def _load_audit_rows(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row["path"]: row for row in csv.DictReader(handle)}


def _normalize_condition_map(raw: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in raw.items():
        if isinstance(value, list):
            result[str(key)] = [str(item) for item in value]
        else:
            result[str(key)] = str(value)
    return result


def _matches_conditions(conditions: dict[str, Any], values: dict[str, Any]) -> bool:
    for key, expected in conditions.items():
        if key not in values:
            continue
        actual = str(values.get(key))
        if isinstance(expected, list):
            if actual not in {str(item) for item in expected}:
                return False
        elif actual != str(expected):
            return False
    return True


def _fallback_level(path: str, payload: dict[str, Any]) -> str:
    if path.startswith(("data.", "mapping.", "sync_stats.")):
        return "project_setup"
    if path in {
        "experiment.mode",
        "experiment.project",
        "experiment.name",
        "experiment.fraction",
        "experiment.fraction_strategy",
        "experiment.train_ratio",
        "experiment.split_strategy",
        "experiment.split_ratio",
        "experiment.version_scope_policy",
        "experiment.load_checkpoint",
        "experiment.structural_mode",
        "experiment.statistic_enabled",
        "experiment.mask_guided_enabled",
        "model.type",
        "model.fusion_mode",
        "model.pooling_strategy",
        "model.graph_strategy",
        "training.learning_strategy",
        "training.retrain",
        "training.epochs",
        "training.batch_size",
        "training.learning_rate",
        "training.device",
        "seed",
    }:
        return "experiment_run"
    ui = payload.get("ui") or {}
    if ui.get("group") == "core":
        return "experiment_run"
    return "advanced"


def _title_from_level(ui_level: str) -> str:
    return {
        "project_setup": "Project Setup",
        "experiment_run": "Experiment Run",
        "advanced": "Advanced",
    }.get(ui_level, "Advanced")


def _split_consumers(value: str) -> list[str]:
    return [item.strip() for item in value.split(";") if item.strip()]
