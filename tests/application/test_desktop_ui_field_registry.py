from __future__ import annotations

from pathlib import Path

from tools.desktop_ui.field_registry import DesktopFieldRegistry
from tools.desktop_ui.main_window import (
    DEFAULT_NAV_INITIAL_WIDTH,
    DEFAULT_NAV_MIN_WIDTH,
    DEFAULT_SPLITTER_SIZES,
    EXPERIMENT_RUN_COLUMNS,
    FIELD_WIDGET_MAX_WIDTH,
    START_MAXIMIZED,
    group_fields_for_page,
)

AUDIT_MATRIX_PATH = Path("configs/ui/desktop_ui_field_dependency_matrix.csv")


def test_registry_classifies_fields_from_audit_matrix():
    registry = DesktopFieldRegistry.load(
        catalog_path=Path("configs/ui/config_catalog.yaml"),
        audit_matrix_path=AUDIT_MATRIX_PATH,
    )

    assert registry.field("data.log_path").ui_level == "project_setup"
    assert registry.field("experiment.mode").ui_level == "experiment_run"
    assert registry.field("model.struct_xattn_heads").ui_level == "experiment_run"
    assert registry.field("training.dataloader_num_workers").ui_level == "advanced"


def test_registry_defaults_emit_when_to_active_when():
    registry = DesktopFieldRegistry.load(
        catalog_path=Path("configs/ui/config_catalog.yaml"),
        audit_matrix_path=AUDIT_MATRIX_PATH,
    )

    meta = registry.field("model.struct_xattn_heads")

    assert meta.active_when
    assert meta.emit_when == meta.active_when


def test_registry_computes_active_state_from_current_values():
    registry = DesktopFieldRegistry.load(
        catalog_path=Path("configs/ui/config_catalog.yaml"),
        audit_matrix_path=AUDIT_MATRIX_PATH,
    )

    inactive_values = {
        "experiment.structural_mode": "true",
        "model.type": "EOPKGGATv2",
        "model.fusion_mode": "ClassMeanConcat",
    }
    active_values = {
        "experiment.structural_mode": "true",
        "model.type": "EOPKGGATv2",
        "model.fusion_mode": "StructXAttn",
    }

    assert not registry.is_active("model.struct_xattn_heads", inactive_values)
    assert registry.is_active("model.struct_xattn_heads", active_values)


def test_registry_keeps_description_and_affects_for_inspector():
    registry = DesktopFieldRegistry.load(
        catalog_path=Path("configs/ui/config_catalog.yaml"),
        audit_matrix_path=AUDIT_MATRIX_PATH,
    )

    meta = registry.field("experiment.mode")

    assert meta.description
    assert meta.affects


def test_prototype_layout_defaults_allow_narrow_navigation_and_safe_run_form():
    assert START_MAXIMIZED is True
    assert DEFAULT_NAV_MIN_WIDTH <= 50
    assert DEFAULT_NAV_INITIAL_WIDTH >= 220
    assert DEFAULT_SPLITTER_SIZES[0] == DEFAULT_NAV_INITIAL_WIDTH
    assert EXPERIMENT_RUN_COLUMNS == 1
    assert FIELD_WIDGET_MAX_WIDTH <= 760


def test_experiment_run_uses_workflow_tabs_instead_of_one_large_frequently_changed_group():
    registry = DesktopFieldRegistry.load(
        catalog_path=Path("configs/ui/config_catalog.yaml"),
        audit_matrix_path=AUDIT_MATRIX_PATH,
    )

    groups = group_fields_for_page(registry, "experiment_run")

    assert "Frequently Changed" not in groups
    assert "Run Identity" in groups
    assert "Data Slice" in groups
    assert "Structure Signal" in groups
    assert "Model / Fusion" in groups
    assert "Learning" in groups
    assert "Checkpoint / Tracking" in groups
    assert "experiment.load_checkpoint" in {field.path for field in groups["Checkpoint / Tracking"]}


def test_project_setup_uses_workflow_tabs_instead_of_one_large_source_group():
    registry = DesktopFieldRegistry.load(
        catalog_path=Path("configs/ui/config_catalog.yaml"),
        audit_matrix_path=AUDIT_MATRIX_PATH,
    )

    groups = group_fields_for_page(registry, "project_setup")

    assert "Connection / Dataset / Adapter" not in groups
    assert "Dataset / Adapter" in groups
    assert "XES Source" in groups
    assert "Camunda Runtime" in groups
    assert "Camunda Structure" in groups
    assert "Knowledge Graph" in groups
    assert "Stats / Mapping" in groups
    assert "data.log_path" in {field.path for field in groups["XES Source"]}


def test_prototype_window_allows_narrow_navigation_without_collapsing_content(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")

    from PySide6.QtWidgets import QApplication
    from tools.desktop_ui.main_window import DesktopPrototypeWindow

    app = QApplication.instance() or QApplication([])
    registry = DesktopFieldRegistry.load(
        catalog_path=Path("configs/ui/config_catalog.yaml"),
        audit_matrix_path=AUDIT_MATRIX_PATH,
    )

    window = DesktopPrototypeWindow(registry=registry, root_dir=Path(".").resolve())

    assert window.nav.minimumWidth() == DEFAULT_NAV_MIN_WIDTH
    assert window.nav.maximumWidth() > 1000
    assert not window.splitter.childrenCollapsible()
    assert not window.splitter.isCollapsible(0)
    assert not window.splitter.isCollapsible(1)
    assert not window.splitter.isCollapsible(2)
    app.processEvents()


def test_dynamic_field_dependencies():
    registry = DesktopFieldRegistry.load(
        catalog_path=Path("configs/ui/config_catalog.yaml"),
        audit_matrix_path=AUDIT_MATRIX_PATH,
    )

    # 1. Adapter dependency
    xes_values = {"mapping.adapter": "xes"}
    camunda_values = {"mapping.adapter": "camunda"}
    assert not registry.is_active("mapping.camunda_adapter.runtime.runtime_source", xes_values)
    assert registry.is_active("mapping.camunda_adapter.runtime.runtime_source", camunda_values)

    # 2. Knowledge Graph backend dependency
    file_kg = {"mapping.knowledge_graph.backend": "file"}
    neo4j_kg = {"mapping.knowledge_graph.backend": "neo4j"}
    assert not registry.is_active("mapping.knowledge_graph.neo4j.connection_profile", file_kg)
    assert registry.is_active("mapping.knowledge_graph.neo4j.connection_profile", neo4j_kg)

    # 3. Model Type dependencies
    baseline_gatv2 = {"model.type": "BaselineGATv2"}
    eopkg_tc = {"model.type": "EOPKGTopologyConditioned"}
    assert not registry.is_active("model.topology_conditioning_mode", baseline_gatv2)
    assert registry.is_active("model.topology_conditioning_mode", eopkg_tc)
