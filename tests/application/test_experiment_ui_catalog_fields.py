from __future__ import annotations

from tools.experiment_ui import (
    CatalogFieldMeta,
    PoolFieldMeta,
    _merge_catalog_section_fields,
)


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
