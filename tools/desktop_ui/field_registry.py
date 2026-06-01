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
            ui_level = str(payload.get("ui_level") or _get_field_ui_level(path, payload)).strip()

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
        registry_active = _matches_conditions(self.field(path).active_when, values)
        return is_field_active(path, values, registry_active)

    def is_emitted(self, path: str, values: dict[str, Any]) -> bool:
        registry_emitted = _matches_conditions(self.field(path).emit_when, values)
        return is_field_active(path, values, registry_emitted)


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
        actual = str(values.get(key)).lower()
        if isinstance(expected, list):
            if actual not in {str(item).lower() for item in expected}:
                return False
        elif actual != str(expected).lower():
            return False
    return True


def is_field_active(path: str, values: dict[str, Any], registry_active: bool) -> bool:
    if not registry_active:
        return False

    def get_val(key: str, default: Any = "") -> str:
        return str(values.get(key, default)).strip().lower()

    # 1. Adapter dependencies
    adapter = get_val("mapping.adapter")
    if adapter == "xes" and path.startswith("mapping.camunda_adapter."):
        return False
    if adapter == "camunda" and (path.startswith("mapping.xes_adapter.") or path == "data.log_path"):
        return False

    # 2. Knowledge Graph backend dependencies
    kg_backend = get_val("mapping.knowledge_graph.backend")
    if kg_backend != "neo4j" and ("neo4j" in path or path in {
        "mapping.knowledge_graph.uri",
        "mapping.knowledge_graph.user",
        "mapping.knowledge_graph.password",
        "mapping.knowledge_graph.database",
    }):
        return False
    if kg_backend != "file" and path == "mapping.knowledge_graph.path":
        return False

    # 3. Structural Mode dependencies
    structural_mode = get_val("experiment.structural_mode") == "true"
    if not structural_mode:
        if path.startswith(("model.struct_xattn_", "model.structural_prior_", "model.topology_state_", "training.struct_xattn_")):
            return False
        if path in {
            "model.fusion_mode",
            "model.struct_encoder",
            "model.struct_encoder_type",
            "model.struct_hidden_dim",
            "model.structural_stats_beta",
            "model.structural_logit_scale_init",
            "model.structural_logit_scale_max",
            "model.structural_observed_scale_max",
            "model.structural_observed_scale_min",
            "model.structural_prior_fusion",
            "model.structural_prior_gate_init_bias",
            "model.structural_prior_pooling",
            "model.structural_score_mode",
            "model.topology_graph_pooling",
            "model.topology_state_beta",
            "model.topology_state_beta_max",
            "model.topology_state_class_pooling",
            "model.topology_state_dropout",
            "model.topology_state_gate_init_bias",
            "training.structural_aux_loss_enabled",
            "training.structural_aux_loss_weight",
            "training.structural_aux_exact_loss_weight",
        }:
            return False

    # 4. Model Type dependencies
    model_type = get_val("model.type")
    if model_type != "eopkgtopologyconditioned":
        if path.startswith((
            "model.impulse_",
            "model.candidate_",
            "training.topology_conditioning_",
            "training.topology_flow_",
            "training.candidate_",
        )) or path in {
            "model.topology_conditioning_mode",
            "training.dynamic_candidate_contract_enabled",
        }:
            return False
    if model_type in {"baselinegatv2", "baselinegcn"}:
        if path.startswith(("model.struct_xattn_", "model.structural_prior_", "model.topology_state_", "training.struct_xattn_")) or path in {
            "model.fusion_mode",
            "model.struct_encoder",
            "model.struct_encoder_type",
            "model.struct_hidden_dim",
            "model.structural_stats_beta",
            "model.structural_prior_fusion",
            "model.structural_prior_gate_init_bias",
            "model.structural_prior_pooling",
            "model.structural_score_mode",
            "model.topology_graph_pooling",
            "model.topology_state_beta",
            "model.topology_state_beta_max",
            "model.topology_state_class_pooling",
            "model.topology_state_dropout",
            "model.topology_state_gate_init_bias",
        }:
            return False

    # 5. Fusion Mode dependencies
    fusion_mode = get_val("model.fusion_mode")
    if fusion_mode != "structxattn" and (path.startswith("model.struct_xattn_") or path.startswith("training.struct_xattn_")):
        return False
    if fusion_mode != "structuralpriorencoder" and path.startswith("model.structural_prior_"):
        return False
    if fusion_mode != "topologystateencoder" and path.startswith("model.topology_state_"):
        return False

    # 6. Learning Strategy dependencies
    learning_strategy = get_val("training.learning_strategy")
    if learning_strategy != "topology_conditioned" and (path.startswith("training.topology_conditioning_") or path.startswith("training.topology_flow_")):
        return False

    return True


def _get_field_ui_level(path: str, payload: dict[str, Any]) -> str:
    # 1. Project Setup
    if path.startswith(("data.", "mapping.", "sync_stats.")):
        return "project_setup"

    # 2. Advanced
    # Dataloader / torch knobs
    if path in {
        "training.dataloader_num_workers",
        "training.dataloader_persistent_workers",
        "training.dataloader_pin_memory",
        "training.dataloader_prefetch_factor",
        "training.torch_num_threads",
        "training.torch_num_interop_threads",
        "training.tqdm_disable",
        "training.tqdm_leave",
    }:
        return "advanced"
    # Dataset cache/performance knobs
    if path.startswith("experiment.") and ("cache" in path or "spill" in path or "shard_size" in path or "max_ram" in path or path == "experiment.cache_policy"):
        return "advanced"
    # Detailed tracing metrics
    if path.startswith(("tracking.tracing.", "tracking.tags.")):
        return "advanced"

    # 3. Experiment Run
    if path.startswith(("model.", "training.", "tracking.", "experiment.")) or path == "seed":
        return "experiment_run"

    # Fallback to advanced
    return "advanced"


def _title_from_level(ui_level: str) -> str:
    return {
        "project_setup": "Project Setup",
        "experiment_run": "Experiment Run",
        "advanced": "Advanced",
    }.get(ui_level, "Advanced")


def _split_consumers(value: str) -> list[str]:
    return [item.strip() for item in value.split(";") if item.strip()]
