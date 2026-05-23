# MVP2.5 New Desktop UI PySide6 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a new parallel desktop UI launched by `main.py ui`, keeping the current `main.py experiment-ui` stable until the new UI is validated.

**Architecture:** Introduce a new PySide6 desktop shell driven by a field registry, not by manually duplicated widgets. Phase 1 builds a visual/UX prototype for review; Phase 2 adds full preset/config/run/status/log functionality.

**Tech Stack:** Python 3.11, PySide6, existing YAML config catalog, existing CLI runner, pytest.

---

## Scope

The current Tkinter `tools/experiment_ui.py` remains unchanged as the stable legacy UI.

The new UI must be launched independently:

```powershell
.\.venv\Scripts\python.exe main.py ui
```

The existing UI remains:

```powershell
.\.venv\Scripts\python.exe main.py experiment-ui
```

Web UI is not part of this work.

---

## UX Model

### Primary Workflow

1. User fills `Project Setup` once and rarely changes it during a hypothesis cycle.
2. User works mainly in `Experiment Run`, changing key experiment parameters.
3. User opens `Run Status / Logs` to track execution and inspect output.
4. User opens `Advanced` only for low-level or rarely changed parameters.

### Top-Level Navigation

Use a left navigation rail:

```text
Project Setup
Experiment Run
Run Status / Logs
Advanced
```

Use a top toolbar for global actions:

```text
Preset | Base config | Save | Load | Validate | Build YAML | Run | Stop
```

Use a right inspector panel for selected field metadata:

```text
Field path
Current value
Default value
Active condition
Runtime consumers
Copy path
Copy value
```

Use a bottom or dedicated log panel only in `Run Status / Logs`.

---

## UI Levels

### Project Setup

Purpose: stable project configuration.

Contains:

- dataset identity;
- input adapter;
- XES/Camunda source;
- BPMN source;
- knowledge graph backend;
- Neo4j/file backend settings;
- stats source and mapping;
- graph feature mapping YAML.

This page may have subtabs:

```text
Data Source
Adapter Mapping
Knowledge Graph
Stats / Graph Mapping
```

### Experiment Run

Purpose: high-frequency experiment controls.

This is the main work page and must fit comfortably on a 1920x1080 screen without requiring vertical scrolling for the common path.

Suggested subtabs:

```text
Run Identity
Data Slice
Structure Signal
Model / Fusion
Learning
Checkpoint
Tracking
```

Core fields:

- `experiment.mode`
- `experiment.project`
- `experiment.name`
- `experiment.fraction`
- `experiment.fraction_strategy`
- `experiment.train_ratio`
- `experiment.split_strategy`
- `experiment.split_ratio`
- `experiment.version_scope_policy`
- `experiment.structural_mode`
- `experiment.statistic_enabled`
- `experiment.mask_guided_enabled`
- `model.type`
- `model.fusion_mode`
- `model.pooling_strategy`
- `model.graph_strategy`
- `training.learning_strategy`
- `training.retrain`
- `training.epochs`
- `training.batch_size`
- `training.learning_rate`
- `training.device`
- `training.load_checkpoint`
- `seed`

### Advanced

Purpose: secondary and low-level parameters.

This page may scroll, but must include search/filter and "show active only".

Suggested subtabs:

```text
Training Internals
Fusion Internals
Learning Strategy Internals
Tracing / Debug
Cache / Performance
Adapter Details
Raw YAML Blocks
```

### Run Status / Logs

Purpose: execution visibility.

Requirements:

- show only stages relevant to selected `experiment.mode`;
- no full screen of empty progress bars;
- show current active stage prominently;
- show elapsed time and ETA when available;
- show child process memory/system memory/disk cache status;
- log autoscroll can be paused;
- log filters: `All`, `Progress`, `Warnings`, `Errors`;
- copy selected log and copy full log;
- keep full log available while presenting concise progress in UI.

---

## Field UX Requirements

### Effective Config Emission

The registry must separate three concerns:

1. `catalog fields`: all known fields, including rare and mode-specific fields.
2. `visible/active fields`: fields relevant to the current UI state.
3. `emitted fields`: fields written into the generated runtime config for a run.

The generated runtime config should be clean by default. It must not emit fields
that are inactive for the current `experiment.mode`, `mapping.adapter`,
`model.fusion_mode`, `training.learning_strategy`, or related selector state.

This keeps MLflow flat params focused on the parameters that actually influenced
the run.

Rules:

- Keep all fields in `configs/ui/config_catalog.yaml`.
- Do not delete inactive fields from the catalog.
- Use `active_when` for UI visibility/enabled state.
- Use `emit_when` only when emission differs from visibility.
- If `emit_when` is absent, default to `active_when`.
- Add a debug toggle: `Include inactive advanced fields`, default `false`.
- Base config files may remain verbose; only the generated UI runtime override
  should be effective/clean.

Examples:

```yaml
path: model.struct_xattn_heads
active_when:
  model.fusion_mode: [StructXAttn]
emit_when:
  model.fusion_mode: [StructXAttn]
```

```yaml
path: training.topology_conditioned_negative_loss_weight
active_when:
  training.learning_strategy: [topology_conditioned]
```

### Copying Field Names

Every rendered field label must support:

- right-click context menu: `Copy field path`;
- right-click context menu: `Copy field value`;
- right inspector buttons: `Copy path`, `Copy value`;
- keyboard shortcut for focused field: `Ctrl+Shift+C` copies field path.

### Ukrainian Keyboard Shortcuts

Use Qt `QAction` / `QShortcut` / `QKeySequence` rather than raw character matching.

Required shortcuts:

- `Ctrl+S`: save current preset/config draft;
- `Ctrl+O`: open/load preset or config;
- `Ctrl+R`: run;
- `Ctrl+F`: focus field search;
- `Esc`: clear search or close transient panels;
- standard text shortcuts (`Ctrl+C`, `Ctrl+V`, `Ctrl+A`, `Ctrl+Z`) must work in Ukrainian layout.

### Scroll Behavior

- `Experiment Run` common workflow must not require vertical scroll on 1920x1080.
- `Advanced` may scroll.
- mouse wheel must work under cursor without requiring focus click.
- field groups should use compact two-column or cardless grid layouts.

### Conditional Visibility

Each field must have an active/visible state derived from registry metadata.

Examples:

- `model.struct_xattn_*` visible only for `model.fusion_mode=StructXAttn`;
- `training.topology_conditioned_*` visible only for `training.learning_strategy=topology_conditioned`;
- XES fields visible only for `mapping.adapter=xes`;
- Camunda fields visible only for `mapping.adapter=camunda`;
- sync/backfill fields visible only for sync modes.

---

## Checkpoint UX

### Requirement

`experiment.load_checkpoint` / `training.load_checkpoint` must be easy to set without manually constructing long paths.

### Default Behavior

Default field value:

```text
checkpoints/
```

This means the field starts at the checkpoint directory, not at a specific file.

### Auto-Fill Button

Add a button:

```text
Fill from experiment.name
```

Behavior:

1. Read `experiment.name`.
2. Build a candidate checkpoint path using existing project convention.
3. If a matching checkpoint file exists, fill the field with that path.
4. If multiple candidates exist, show a small chooser.
5. If no candidate exists, keep `checkpoints/` and show a non-blocking warning.

Recommended resolver contract:

```python
def resolve_checkpoint_candidates(
    *,
    checkpoint_root: Path,
    experiment_name: str,
) -> list[Path]:
    ...
```

The resolver must be testable independently from the UI.

---

## Proposed File Structure

### New files

- `tools/desktop_ui/__init__.py`
- `tools/desktop_ui/app.py`
  - PySide6 application entry point.
- `tools/desktop_ui/main_window.py`
  - `QMainWindow`, navigation rail, toolbar, main page switcher.
- `tools/desktop_ui/field_registry.py`
  - Loads `configs/ui/config_catalog.yaml` and normalizes field metadata.
- `tools/desktop_ui/field_state.py`
  - Stores current values, defaults, dirty state, and generated YAML override.
- `tools/desktop_ui/field_widgets.py`
  - Renders typed widgets from registry metadata.
- `tools/desktop_ui/pages/project_setup_page.py`
- `tools/desktop_ui/pages/experiment_run_page.py`
- `tools/desktop_ui/pages/advanced_page.py`
- `tools/desktop_ui/pages/run_status_page.py`
- `tools/desktop_ui/checkpoint_resolver.py`
- `tools/desktop_ui/run_process.py`
  - Process runner and progress-event parser.
- `tests/application/test_desktop_ui_field_registry.py`
- `tests/application/test_desktop_ui_checkpoint_resolver.py`
- `tests/application/test_desktop_ui_command_entrypoint.py`

### Modified files

- `main.py`
  - Add `ui` command.
- `requirements.txt`
  - Add `PySide6`.
- `requirements-current-freeze.txt`
  - Update only after environment install/freeze is intentionally refreshed.
- `AGENTS.MD`
  - Add new UI routing if this becomes active.
- `docs/UI_SPECA.MD`
  - Update supported desktop UI architecture.
- `docs/current/project-state.md`
  - Mention new parallel PySide6 UI when implemented.
- `docs/current/architecture-debt.md`
  - Update Desktop UI debt status after Phase 1/2.

---

## Phase 1: Visual Prototype

Purpose: validate layout, field grouping, keyboard/copy UX, and 1920x1080 usability before wiring full run behavior.

### Task 1: Add PySide6 dependency and entrypoint guard

**Files:**

- Modify: `requirements.txt`
- Modify: `main.py`
- Create: `tools/desktop_ui/app.py`
- Test: `tests/application/test_desktop_ui_command_entrypoint.py`

- [ ] Add `PySide6>=6.7` to `requirements.txt`.
- [ ] Add a `main.py ui` branch that imports `tools.desktop_ui.app.main`.
- [ ] If PySide6 is missing, print a clear message and exit with non-zero code.
- [ ] Test that `main.py` recognizes `ui` without importing legacy Web UI.

### Task 2: Build `DesktopFieldRegistry`

**Files:**

- Create: `tools/desktop_ui/field_registry.py`
- Test: `tests/application/test_desktop_ui_field_registry.py`

- [ ] Load `configs/ui/config_catalog.yaml`.
- [ ] Normalize each field into:
  - `path`
  - `section`
  - `label`
  - `default`
  - `enum`
  - `ui_level`
  - `ui_tab`
  - `ui_group`
  - `ui_order`
  - `required_when`
  - `active_when`
  - `emit_when`
- [ ] Apply fallback classification from `outputs/ui/desktop_ui_field_dependency_matrix.csv` while catalog lacks explicit `ui_level`.
- [ ] Test that known fields classify correctly:
  - `data.log_path` -> `project_setup`
  - `experiment.mode` -> `experiment_run`
  - `model.struct_xattn_heads` -> `advanced`
- [ ] Test that `emit_when` defaults to `active_when` when omitted.

### Task 3: Build static prototype shell

**Files:**

- Create: `tools/desktop_ui/main_window.py`
- Create: `tools/desktop_ui/pages/project_setup_page.py`
- Create: `tools/desktop_ui/pages/experiment_run_page.py`
- Create: `tools/desktop_ui/pages/advanced_page.py`
- Create: `tools/desktop_ui/pages/run_status_page.py`

- [ ] Build `QMainWindow` with top toolbar.
- [ ] Build left navigation rail.
- [ ] Build center page switcher.
- [ ] Build right field inspector.
- [ ] Use placeholder data first; no run logic.
- [ ] Verify manually on 1920x1080 that `Experiment Run` common fields fit without vertical scroll.

### Task 4: Render registry-driven fields

**Files:**

- Create: `tools/desktop_ui/field_widgets.py`
- Modify: page files from Task 3
- Test: `tests/application/test_desktop_ui_field_registry.py`

- [ ] Render string fields as `QLineEdit`.
- [ ] Render bool fields as `QCheckBox`.
- [ ] Render enum fields as `QComboBox`.
- [ ] Render YAML/multiline fields as compact editor blocks only in `Advanced` or specific YAML tabs.
- [ ] Every label opens field inspector on click.
- [ ] Every label context menu includes `Copy field path`.

### Task 5: Add field search and active-only filter

**Files:**

- Modify: `tools/desktop_ui/main_window.py`
- Modify: `tools/desktop_ui/field_registry.py`

- [ ] Add `Ctrl+F` field search.
- [ ] Add `Show active only` toggle.
- [ ] Evaluate active state from current values.
- [ ] Hide mode/fusion/adapter irrelevant fields.

### Task 6: Add checkpoint resolver prototype

**Files:**

- Create: `tools/desktop_ui/checkpoint_resolver.py`
- Test: `tests/application/test_desktop_ui_checkpoint_resolver.py`
- Modify: `tools/desktop_ui/pages/experiment_run_page.py`

- [ ] Default checkpoint field value to `checkpoints/`.
- [ ] Add `Fill from experiment.name` button.
- [ ] Resolver searches under `checkpoints/` by experiment name.
- [ ] If no match exists, keep `checkpoints/` and return warning status.
- [ ] If one match exists, fill exact path.
- [ ] If multiple matches exist, return candidates for chooser UI.

### Task 7: Prototype review gate

- [ ] Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\application\test_desktop_ui_field_registry.py -v
.\.venv\Scripts\python.exe -m pytest tests\application\test_desktop_ui_checkpoint_resolver.py -v
.\.venv\Scripts\python.exe -m pytest tests\application\test_desktop_ui_command_entrypoint.py -v
```

- [ ] Launch:

```powershell
.\.venv\Scripts\python.exe main.py ui
```

- [ ] User reviews:
  - layout;
  - field grouping;
  - copy path/value;
  - Ukrainian hotkeys;
  - 1920x1080 no-scroll common workflow.

Stop after Phase 1 until user approves visual design.

---

## Phase 2: Full Functionality

Purpose: make new UI operational while preserving legacy `experiment-ui`.

### Task 8: Implement `DesktopExperimentState`

**Files:**

- Create: `tools/desktop_ui/field_state.py`
- Test: `tests/application/test_desktop_ui_field_state.py`

- [ ] Load base config via existing `load_yaml_with_includes`.
- [ ] Load preset values.
- [ ] Merge defaults from registry.
- [ ] Track dirty state.
- [ ] Build effective config override YAML.
- [ ] Do not emit inactive fields by default.
- [ ] Emit inactive fields only when `include_inactive_advanced_fields=true`.
- [ ] Preserve active defaults when they are required for deterministic runtime behavior.
- [ ] Preserve YAML blocks without lossy string manipulation.

Required tests:

- [ ] `StructXAttn` fields are not emitted when `model.fusion_mode=ClassMeanConcat`.
- [ ] topology-conditioned fields are not emitted when `training.learning_strategy=standard`.
- [ ] XES adapter fields are not emitted when `mapping.adapter=camunda`, except shared fields.
- [ ] inactive fields are emitted when the debug toggle is explicitly enabled.
- [ ] emitted YAML still contains required active defaults.

### Task 9: Implement preset load/save

**Files:**

- Modify: `tools/desktop_ui/field_state.py`
- Modify: `tools/desktop_ui/main_window.py`
- Test: `tests/application/test_desktop_ui_field_state.py`

- [ ] Read/write `outputs/ui/experiment_ui_presets.json`.
- [ ] Keep compatibility with existing preset format where practical.
- [ ] Sort `_`-prefixed preset names first.
- [ ] Preserve current experiment state on save/load.

### Task 10: Implement validation

**Files:**

- Modify: `tools/desktop_ui/field_registry.py`
- Modify: `tools/desktop_ui/field_state.py`
- Test: `tests/application/test_desktop_ui_validation.py`

- [ ] Validate required fields by mode.
- [ ] Validate conditional fields by `required_when`.
- [ ] Validate emitted config by `emit_when`.
- [ ] Warn when a field has catalog metadata but no known runtime consumer.
- [ ] Warn when an inactive field has a non-default value that will not be emitted.
- [ ] Validate numeric ranges for `fraction`, split ratios, learning rate, batch size.
- [ ] Mark invalid fields visually.
- [ ] Show validation summary in right inspector/status area.

### Task 11: Implement command preview

**Files:**

- Modify: `tools/desktop_ui/main_window.py`
- Modify: `tools/desktop_ui/field_state.py`
- Test: `tests/application/test_desktop_ui_command_preview.py`

- [ ] Generate the same CLI command shape as legacy UI.
- [ ] Show command preview.
- [ ] Show effective config preview, not full catalog config.
- [ ] Mark omitted inactive fields in a secondary "Omitted Fields" inspector section.
- [ ] Allow copying command.
- [ ] Include generated temp config path only after run starts.

### Task 12: Implement run/stop process manager

**Files:**

- Create: `tools/desktop_ui/run_process.py`
- Modify: `tools/desktop_ui/pages/run_status_page.py`
- Test: `tests/application/test_desktop_ui_run_process.py`

- [ ] Start subprocess with generated config.
- [ ] Stream stdout/stderr.
- [ ] Parse `__BPM_PROGRESS__` events.
- [ ] Support Stop.
- [ ] Surface non-zero exit codes.

### Task 13: Implement mode-aware Run Status

**Files:**

- Modify: `tools/desktop_ui/pages/run_status_page.py`
- Test: `tests/application/test_desktop_ui_run_status.py`

- [ ] Show only stages relevant to current `experiment.mode`.
- [ ] Hide empty/unseen stages by default.
- [ ] Show active stage, stage progress, overall progress, ETA.
- [ ] Show memory/cache status when emitted.
- [ ] Avoid a full screen of empty status bars.

### Task 14: Implement log panel controls

**Files:**

- Modify: `tools/desktop_ui/pages/run_status_page.py`
- Test: `tests/application/test_desktop_ui_log_panel.py`

- [ ] Add Pause Autoscroll.
- [ ] Add filters: `All`, `Progress`, `Warnings`, `Errors`.
- [ ] Add Copy Selected and Copy All.
- [ ] Keep full raw log in memory/file while filtered view changes.

### Task 15: Full UI parity audit

**Files:**

- Create: `tools/desktop_ui/audit.py`
- Test: `tests/application/test_desktop_ui_parity_audit.py`

- [ ] Compare catalog fields against rendered fields.
- [ ] Fail if a catalog field is not rendered or intentionally hidden.
- [ ] Fail if a rendered field has no catalog metadata.
- [ ] Compare catalog fields against effective emitted config for representative states.
- [ ] Fail if an inactive mode-specific field is emitted by default.
- [ ] Fail if an active required field is omitted.
- [ ] Produce a report:

```text
outputs/ui/desktop_ui_parity_report.json
```

The report must include:

```json
{
  "catalog_fields": 0,
  "rendered_fields": 0,
  "active_fields": 0,
  "emitted_fields": 0,
  "omitted_inactive_fields": [],
  "unknown_runtime_consumers": [],
  "active_required_omissions": []
}
```

### Task 16: Documentation update

**Files:**

- Modify: `AGENTS.MD`
- Modify: `docs/UI_SPECA.MD`
- Modify: `docs/current/project-state.md`
- Modify: `docs/current/architecture-debt.md`
- Modify: `docs/runbooks/mvp2_5-commands.md`

- [ ] Document `main.py ui`.
- [ ] Mark legacy `experiment-ui` as stable legacy during transition.
- [ ] Document PySide6 UI hierarchy.
- [ ] Update Desktop UI debt status.

### Task 17: Final verification

- [ ] Run:

```powershell
.\.venv\Scripts\python.exe tools\architecture_guard.py
.\.venv\Scripts\python.exe -m pytest tests\application\test_desktop_ui_field_registry.py -v
.\.venv\Scripts\python.exe -m pytest tests\application\test_desktop_ui_checkpoint_resolver.py -v
.\.venv\Scripts\python.exe -m pytest tests\application\test_desktop_ui_command_entrypoint.py -v
.\.venv\Scripts\python.exe -m pytest tests\ -v
```

- [ ] Manual smoke:

```powershell
.\.venv\Scripts\python.exe main.py ui
.\.venv\Scripts\python.exe main.py experiment-ui
```

- [ ] Confirm both UIs can coexist.

---

## Risks and Mitigations

### Risk: PySide6 adds a large dependency

Mitigation: Add it only for the new `ui` command. Legacy CLI/train/eval must not import PySide6 unless the user launches `main.py ui`.

### Risk: UI rewrite delays research work

Mitigation: Phase 1 is only a prototype. Stop after visual review before wiring full run behavior.

### Risk: field registry metadata is incomplete

Mitigation: Use the audit CSV as fallback for Phase 1, then move stable `ui_level`, `active_when`, and `runtime_consumers` into `configs/ui/config_catalog.yaml`.

### Risk: effective config omission changes runtime behavior

Mitigation: implement omission behind `DesktopExperimentState.build_effective_config()`
with tests for representative modes, adapters, fusion modes, and learning
strategies. Keep a debug toggle to include inactive advanced fields when
investigating compatibility issues.

### Risk: MLflow params become hard to compare with older runs

Mitigation: log a compact `effective_config_policy=active_fields_only` marker
and keep an optional artifact with omitted inactive fields for audit/debug.

### Risk: copying/hotkeys behave differently across keyboard layouts

Mitigation: Use Qt actions and manually test Ukrainian layout before Phase 1 approval.

### Risk: no-scroll 1920x1080 target conflicts with too many fields

Mitigation: Keep `Experiment Run` compact and split into subtabs. Do not force all high-frequency fields into one visible grid.

---

## Acceptance Criteria

### Phase 1

- `main.py ui` opens a PySide6 prototype.
- `main.py experiment-ui` still opens legacy UI.
- `Experiment Run` common workflow fits 1920x1080 without vertical scroll.
- User can copy field path/value.
- Ukrainian-layout hotkeys work.
- Checkpoint field defaults to `checkpoints/`.
- `Fill from experiment.name` works on mocked checkpoint files.
- Effective config preview omits inactive mode/fusion/strategy fields by default.

### Phase 2

- New UI can load/save presets.
- New UI can generate effective YAML override.
- MLflow params from UI-generated runs are not polluted by inactive
  mode/fusion/strategy-specific fields.
- New UI can run/stop experiments.
- Run Status is mode-aware and not full of empty bars.
- Logs support pause autoscroll and filters.
- Full pytest passes.

---

## Recommended Execution

Implement Phase 1 first and stop for visual review. Do not start Phase 2 until the user approves the layout and field grouping.
