# UI_SPECA.MD

Updated: 2026-03-21
Status: Implemented (catalog-driven UI)

## 1. Objective
Desktop UI is the primary way to configure experiments. CLI remains for automation/CI only.

## 2. Single Source of Config Metadata
Metadata catalog file:
1. [config_catalog.yaml](/C:/Users/korsr/PycharmProjects/bpm_prediction/configs/ui/config_catalog.yaml)

Catalog stores per field:
1. Attribute path (`path`)
2. Section (`section`)
3. Description (`description`)
4. Impact (`affects`)
5. Required modes (`required_in_modes`)
6. Conditional required rules (`required_when`)
7. Default value (`default`)
8. Enum values (`enum`)
9. UI placement and priority (`ui.tab`, `ui.group`, `ui.priority`, `ui.order`)

At startup, UI syncs pool-discovered config keys into this catalog (missing keys are auto-added).

## 3. UI Structure
Main tabs:
1. `General`
2. `Input Data`
3. `EOPKG / Structure`
4. `Model`
5. `Run`

Subtabs:
1. `General`: `Core`, `Advanced`
2. `General -> Advanced`: `Experiment`, `Training`, `Tracking`
3. `Input Data`: `Source + Runtime`, `Features / Policies`
4. `Input Data -> Source + Runtime`: `Common`, `XES Mapping`, `Camunda Runtime`, `Camunda Mapping`
5. `EOPKG / Structure`: `Backend`, `Structure`, `Sync Stats`, `Graph Mapping`

## 4. Dynamic Forms
Dynamic forms expose full pooled parameter set from merged configs under `configs/**` for sections:
1. `experiment`
2. `training`
3. `tracking`
4. `data`
5. `mapping`
6. `model`
7. `sync_stats`
8. `seed`

Memo/YAML blocks (not flattened):
1. `mapping.features` (fallback from root `features` if present)
2. `policies`
3. `mapping.graph_feature_mapping` (in `EOPKG / Structure -> Graph Mapping`)

## 5. Hints and UX
1. `?` marker is rendered before field name.
2. Tooltip is clamped to screen bounds.
3. Hints are catalog-driven and include examples from observed configs.
4. Dropdowns are auto-used for compact enum/discrete fields.
5. Field order uses catalog priority/order to place frequent controls first.

## 6. Mode/Adapter Activation Rules
By mode:
1. Sync modes disable training/model-focused controls.
2. `sync-stats` enables `--as-of`.
3. `sync-stats-backfill` enables `--step`, `--step-days`, `--from`, `--to`.

By adapter:
1. `camunda` disables `mapping.xes_adapter.*` and `data.log_path`.
2. `xes` disables `mapping.camunda_adapter.*`.

Camunda source refinements:
1. `runtime.runtime_source=files` disables SQL/MSSQL runtime fields.
2. `runtime.runtime_source!=files` disables runtime export-dir fields.
3. `structure.bpmn_source=files` enables `structure.files.*` and disables SQL/MSSQL structure fields.
4. `structure.bpmn_source!=files` does inverse.

## 7. Validation Before Run
UI validates composed config against catalog:
1. Required fields by `required_in_modes`
2. Conditional required fields by `required_when`
3. Enum constraints for catalog fields

Execution is blocked on validation errors.

## 8. State and Presets
1. Autosave: `outputs/ui/experiment_ui_state.json`
2. Presets: `outputs/ui/experiment_ui_presets.json`
3. Stored data includes scalar vars, dynamic forms, and YAML memo blocks.

## 9. Runtime Command Mapping
1. Train/eval: `python main.py --config <temp.yaml>`
2. Sync topology: `python main.py sync-topology --config <temp.yaml>`
3. Sync stats: `python main.py sync-stats --config <temp.yaml> [--as-of ...]`
4. Backfill: `python main.py sync-stats-backfill --config <temp.yaml> --step ... [--step-days ... --from ... --to ...]`
