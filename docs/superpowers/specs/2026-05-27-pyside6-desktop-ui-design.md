# PySide6 Desktop UI Refactoring Spec

**Date:** 2026-05-27  
**Status:** Approved  
**Topic:** Dynamic Catalog-Driven PySide6 Desktop UI for `bpm_prediction`  
**Phase:** Phase 2 Implementation  

---

## 1. Overview & Goals
The objective of this design is to refactor and complete the PySide6 Desktop UI prototype to fully replace the legacy Tkinter-based `experiment_ui.py`. The new design directly addresses key pain points from the legacy implementation:
1. **Dynamic Catalog Binding:** Automatically build and render forms directly from `configs/ui/config_catalog.yaml` using a normalized metadata registry.
2. **Dynamic Visibility (Clean Configs):** Dynamically hide fields and sections not relevant to the currently selected `experiment.mode` and other parameter dependencies. Avoid exporting irrelevant fields to the generated YAML configurations to keep MLflow tracking clean.
3. **Preset Management:** Organize a tree-view preset manager grouped by `experiment.project` -> `preset_name` supporting metadata (dates, run mode, and custom user comments).
4. **Log Streaming & Progress Monitoring:** Stream live logs to a copyable console with a log-freezing feature. Provide mode-specific progress bars with runtime duration, percent completion, and ETA calculations.
5. **Polished Design:** A modern dark-slate stylesheet theme (`styles.py`) with micro-animations.

---

## 2. Component Architecture
Following the approved modular approach, the application will be structured into separate components:

```
tools/desktop_ui/
├── __init__.py
├── app.py                     # App entry point
├── styles.py                  # QSS Dark Slate theme & colors [NEW]
├── field_registry.py          # Field metadata and active/emit rule evaluation
├── field_widgets.py           # Factory to create editing widgets for fields
├── checkpoint_resolver.py     # Checkpoint filename resolver
├── preset_manager.py          # Sliding drawer, tree view, comments & metadata [NEW]
├── run_monitor.py             # Subprocess execution, QProcess, log console & progress [NEW]
└── main_window.py             # MainWindow orchestration, layout, search & visibility updates
```

### 2.1 UI Layout & Sliding Drawer
* **3-Panel Main Splitter:**
  * **Panel 1 (Left Navigation):** Navigation List for switching Stacked Pages. A button "Presets 📁" is placed at the top/bottom.
  * **Panel 2 (Center Content):** Dynamic forms with tabs (Project Setup, Experiment Run, Run Status / Logs, Advanced).
  * **Panel 3 (Right Inspector):** Metadata card of the active field.
* **Sliding Preset Drawer:**
  * A custom sidebar panel placed adjacent to Navigation.
  * Controlled by a sliding transition using `QPropertyAnimation` on the widget's width (from `0` closed to `320` pixels open) with `QEasingCurve.OutQuad` easing.
  * Holds the `QTreeWidget` (Project -> Preset), search bar, metadata labels (mode, date), a `QTextEdit` comment field, and action buttons: **Save Preset**, **Delete Preset**, and **Load Preset**.

### 2.2 Preset Storage Format
Presets are loaded and saved using `outputs/ui/experiment_ui_presets.json`. The schema is extended backward-compatibly:
```json
{
  "preset_name": {
    "values": {
      "experiment.mode": "train",
      "experiment.project": "bpm_prediction_mvp2_5",
      "experiment.name": "my_run",
      "...": "..."
    },
    "saved_at": "ISO-8601-Timestamp",
    "mode": "train",
    "project": "bpm_prediction_mvp2_5",
    "comment": "Optional author notes explaining this run profile."
  }
}
```
*Legacy presets missing `project` or `comment` default to parsing project from `values` and keeping comments blank.*

### 2.3 Smart Checkpoint Resolver
* **Sorting & Ranking Candidates:**
  * Search checkpoints directory (`checkpoints/`) for `.pth` files.
  * Filter candidates by current `experiment.name` and `model.type`.
  * Sort candidates by **modification time (newest first)** to ensure recently completed runs are listed first.
  * Prioritize/highlight candidates with `_best.pth` suffix.
* **UI Integration:**
  * When `experiment.load_checkpoint` is selected or clicked, display a dropdown list of the top 5 candidates with their file sizes and timestamps.
  * Show warnings in the Inspector if the checkpoint model configuration does not align with `model.type` or if the timestamp indicates it is outdated.

### 2.4 Field Catalog Constraints & Validation Rules
* **Single Source of Truth:** `configs/ui/config_catalog.yaml` holds all attributes, requirements (`required_when`), visibility triggers (`active_when`), description, and affects (impact) properties.
* **Audit Matrix Synced:** The audit CSV `outputs/ui/desktop_ui_field_dependency_matrix.csv` has been refreshed to sync all 241 catalog fields, including the 20 newly added MVP2.5 Stage 4.2 fields.
* **Toolbar Validation Checks:** Clicking **Validate** triggers the following checks:
  * **Temporal Drift Warning:** Warns if `experiment.mode` is set to `eval_drift` or `eval_cross_dataset` but `experiment.stats_time_policy` is set to `latest` (data leakage risk).
  * **Topology-Conditioned Model Requirements:** Raises an error if `training.learning_strategy` is set to `topology_conditioned` but `model.type` is not set to `EOPKGTopologyConditioned` (invalid training mode configuration).
  * **Stats Policy Requirement:** Verifies that `experiment.stats_time_policy` is active and defined whenever `experiment.statistic_enabled` is true.

---

## 3. Dynamic Visibility Engine
* Fields are processed dynamically by `DesktopFieldRegistry` which monitors field values.
* Changing any field triggers `_update_field_visibilities()` in `MainWindow`:
  * Evaluates `active_when` and `emit_when` conditions for all fields against current values.
  * If a field is active: row is visible and enabled.
  * If a field is inactive:
    * If `Show Hidden (Read-Only)` is **unchecked**: Hide the row.
    * If `Show Hidden (Read-Only)` is **checked**: Show the row but disable the widget (`setEnabled(False)`).
* Groups/tabs containing zero active/visible fields are collapsed automatically.

---

## 4. Run Monitoring & Log Streaming
* **Process Execution:** Uses PySide6's `QProcess` in asynchronous mode.
* **YAML File Generation:**
  * When **Run** or **Build YAML** is clicked, we compile only active and visible fields.
  * Nest the dictionary keys using dotted path parts.
  * Save the output config to: `outputs/ui/generated_configs/<project_name>/<run_name>_<timestamp>.yaml`.
  * Launch the process: `.\.venv-modern\Scripts\python.exe main.py --config <generated_yaml_path>` with environment variable `BPM_PROGRESS_EVENTS=1`.
* **Progress Dashboard (Progress Dash):**
  * Displays progress bars for steps listed in `RUN_STAGE_ORDER` that are **active** for the current `experiment.mode`.
  * Emits status updates: Pending, Running, Done, Error.
  * Formats time values:
    * Under 60s: `<secs>s` (e.g. `45s`).
    * Under 1 hour: `MM:SS` (e.g. `12:34`).
    * Over 1 hour: `HH:MM:SS` (e.g. `01:05:22`).
  * Estimates ETA dynamically using overall weighted completion rate.
* **Log Console:**
  * Uses a read-only, copy-enabled `QPlainTextEdit`.
  * Color-codes lines: red for errors, orange/yellow for warnings, gray/white for info.
  * "Freeze Logs" checkbox: disables cursor autoscroll-to-end to allow selecting and copying log snippets during active run.

---

## 5. Verification Plan
### Automated Verification
1. Run `.\.venv-modern\Scripts\python.exe tools\architecture_guard.py` to ensure clean architecture compliance (UI tools isolated from domain layers).
2. Run standard tests: `.\.venv-modern\Scripts\python.exe -m pytest tests/ -v`.
3. Build mock tests for `DesktopFieldRegistry` condition matches.

### Manual Verification
1. Start PySide6 UI: `.\.venv-modern\Scripts\python.exe -m tools.desktop_ui.app`.
2. Toggle `experiment.mode` and verify dynamic field visibility.
3. Open sliding Preset Drawer, select, load, edit comment, and delete presets.
4. Run a diagnostic experiment, verify progress bars update, logs stream, and freeze logs halts scrolling.
5. Confirm generated YAML config is correctly written to `outputs/ui/generated_configs/` and contains only active fields.
