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
