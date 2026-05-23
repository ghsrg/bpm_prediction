from __future__ import annotations

from tools.experiment_ui import (
    CatalogFieldMeta,
    ExperimentUI,
    PoolFieldMeta,
    _merge_catalog_section_fields,
)
import yaml
from pathlib import Path


def _catalog(path: str, default: str = "") -> CatalogFieldMeta:
    return CatalogFieldMeta(
        path=path,
        section="sync_stats",
        label=path,
        description="",
        affects="",
        default=default,
        enum=[],
        required_in_modes=[],
        required_when={},
        ui_tab="eopkg",
        ui_group="sync_stats",
        ui_priority=2,
        ui_order=1,
    )


def test_desktop_sync_stats_fields_include_catalog_only_entries():
    merged = _merge_catalog_section_fields(
        base_fields={"sync_stats.alignment_gate.on_fail": "skip_snapshot"},
        pool_meta={
            "sync_stats.enabled": PoolFieldMeta(
                path="sync_stats.enabled",
                section="sync_stats",
                values={"true"},
            )
        },
        catalog={
            "sync_stats.alignment_gate.profile": _catalog(
                "sync_stats.alignment_gate.profile",
                default="legacy_exact",
            )
        },
        section="sync_stats",
    )

    assert merged["sync_stats.alignment_gate.on_fail"] == "skip_snapshot"
    assert merged["sync_stats.enabled"] == ""
    assert merged["sync_stats.alignment_gate.profile"] == "legacy_exact"


def test_preset_names_sort_underscore_prefixed_first():
    names = ExperimentUI._sort_preset_names(["Base-UN", "_Top-MN", "S_Att-UN", "_Base-UN"])

    assert names == ["_Base-UN", "_Top-MN", "Base-UN", "S_Att-UN"]


def test_catalog_includes_topology_conditioned_learning_strategy_fields():
    catalog_path = Path("configs/ui/config_catalog.yaml")
    catalog = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))["fields"]

    required = {
        "training.learning_strategy",
        "training.topology_conditioning_wrong_version_negative_enabled",
        "training.topology_conditioning_drop_edges_negative_enabled",
        "training.topology_conditioning_allowed_set_loss_enabled",
        "training.topology_conditioning_retention_enabled",
    }

    assert required.issubset(set(catalog))
    assert catalog["training.learning_strategy"]["enum"] == ["standard", "topology_conditioned"]


def test_catalog_includes_versioned_fraction_split_fields():
    catalog_path = Path("configs/ui/config_catalog.yaml")
    catalog = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))["fields"]

    required = {
        "experiment.fraction",
        "experiment.fraction_strategy",
        "experiment.split_ratio",
        "experiment.split_strategy",
        "experiment.train_ratio",
        "experiment.version_scope_policy",
    }

    assert required.issubset(set(catalog))
    assert "versioned" in catalog["experiment.split_strategy"]["enum"]
    for field_name in required:
        assert catalog[field_name]["ui"]["group"] == "core"
    assert {
        field_name: catalog[field_name]["ui"]["order"]
        for field_name in required
    } == {
        "experiment.fraction": 4,
        "experiment.fraction_strategy": 5,
        "experiment.train_ratio": 6,
        "experiment.split_strategy": 7,
        "experiment.split_ratio": 8,
        "experiment.version_scope_policy": 9,
    }


def test_experiment_uis_surface_versioned_fraction_split_core_fields():
    desktop_source = Path("tools/experiment_ui.py").read_text(encoding="utf-8")

    for field_name in ("fraction_strategy", "version_scope_policy"):
        assert f'self.vars["{field_name}"]' in desktop_source
        assert f"experiment.{field_name}" in desktop_source


def test_web_experiment_ui_is_removed_from_active_project_routing():
    assert not Path("tools/web_experiment_ui_app.py").exists()
    assert not Path("tools/web_ui.py").exists()

    for path in [
        Path("AGENTS.MD"),
        Path("docs/current/project-state.md"),
        Path("docs/ARCHITECTURE_MVP2_5.MD"),
        Path("docs/UI_SPECA.MD"),
    ]:
        text = path.read_text(encoding="utf-8")
        assert "web_experiment_ui_app.py" not in text
        assert "web UI" not in text
        assert "web prototype" not in text
