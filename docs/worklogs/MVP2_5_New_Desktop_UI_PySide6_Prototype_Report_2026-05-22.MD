# MVP2.5 New Desktop UI PySide6 Prototype Report - 2026-05-22

## Metadata

- `status`: prototype_implemented
- `scope`: Phase 1 visual/UX prototype
- `entrypoint`: `.\.venv-modern\Scripts\python.exe main.py ui`
- `legacy_entrypoint`: `.\.venv-modern\Scripts\python.exe main.py experiment-ui`

## Summary

Implemented a parallel PySide6 desktop UI prototype without replacing the
legacy Tkinter `experiment-ui`.

The prototype is intentionally not a production run UI yet. Its current purpose
is visual review and field hierarchy validation:

- `Project Setup`
- `Experiment Run`
- `Run Status / Logs`
- `Advanced`

## Implemented

### Entrypoint

- Added `main.py ui`.
- Kept `main.py experiment-ui` unchanged for legacy UI access.
- PySide6 import is lazy and only required when `main.py ui` is launched.

### Field Registry

Created `tools/desktop_ui/field_registry.py`.

The registry:

- loads `configs/ui/config_catalog.yaml`;
- uses `outputs/ui/desktop_ui_field_dependency_matrix.csv` as fallback
  classification source;
- exposes `project_setup`, `experiment_run`, and `advanced` levels;
- normalizes `active_when` and `emit_when`;
- defaults `emit_when` to `active_when`.

### Prototype Window

Created `tools/desktop_ui/main_window.py`.

The prototype window includes:

- top toolbar;
- left navigation rail;
- maximized startup window;
- narrow-resizable left navigation panel with `50px` minimum and no maximum
  width;
- non-collapsible splitter sections to prevent accidental panel disappearance
  during resize;
- center page stack;
- right field inspector;
- search field;
- registry-driven field rendering;
- row-based single-column `Experiment Run` layout split into workflow tabs:
  `Run Identity`, `Data Slice`, `Structure Signal`, `Model / Fusion`,
  `Learning`, `Checkpoint / Tracking`, and `Other`;
- `Project Setup` layout split into workflow tabs:
  `Dataset / Adapter`, `XES Source`, `Camunda Runtime`,
  `Camunda Structure`, `Knowledge Graph`, `Stats / Mapping`, and `Other`;
- capped field widget width to avoid oversized controls on wide screens;
- scrollable `Project Setup` and `Advanced`;
- `Run Status / Logs` placeholder page;
- selectable field labels;
- field path/value copy through selectable inspector fields and inspector
  buttons;
- field path/value copy through label context menu.

### Checkpoint Resolver

Created `tools/desktop_ui/checkpoint_resolver.py`.

Implemented:

- default checkpoint value: `checkpoints/`;
- candidate lookup by `experiment.name`;
- sorted candidate output preferring `best` checkpoints.

The prototype includes a `Fill from experiment.name` button next to
`experiment.load_checkpoint`.

### Tests

Added:

- `tests/application/test_desktop_ui_field_registry.py`
- `tests/application/test_desktop_ui_checkpoint_resolver.py`
- `tests/application/test_desktop_ui_command_entrypoint.py`

## Verification

Targeted tests:

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\application\test_desktop_ui_field_registry.py tests\application\test_desktop_ui_checkpoint_resolver.py tests\application\test_desktop_ui_command_entrypoint.py -v
```

Result:

```text
13 passed
```

PySide6 import:

```powershell
.\.venv-modern\Scripts\python.exe -c "import PySide6; print(PySide6.__version__)"
```

Result:

```text
6.11.1
```

Headless PySide6 window smoke:

```powershell
$env:QT_QPA_PLATFORM='offscreen'
.\.venv-modern\Scripts\python.exe -c "... DesktopPrototypeWindow ..."
```

Result:

```text
project-tabs ['Dataset / Adapter', 'XES Source', 'Camunda Runtime', 'Camunda Structure', 'Knowledge Graph', 'Stats / Mapping', 'Other']
run-tabs ['Run Identity', 'Data Slice', 'Structure Signal', 'Model / Fusion', 'Learning', 'Checkpoint / Tracking', 'Other']
```

Full project tests:

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\ -v
```

Result:

```text
439 passed, 11 warnings
```

## Known Limitations

- Run/Stop is not wired to subprocess execution yet.
- Preset load/save is not implemented in the PySide6 prototype yet.
- Effective config preview is planned but not implemented yet.
- Run Status page is a placeholder; mode-aware progress bars come in Phase 2.
- Search currently hides matching label/widget pairs but does not collapse empty
  tab groups.
- Field active-only filtering is not implemented yet.
- Keyboard shortcut review on Ukrainian layout still requires manual UI testing.
- Visual tuning of exact group order and field density still requires manual
  review on the target 1920x1080 screen.

## Next Step

Open:

```powershell
.\.venv-modern\Scripts\python.exe main.py ui
```

Review:

- `Experiment Run` layout on 1920x1080;
- field grouping;
- copy path/value behavior;
- checkpoint default and autofill behavior;
- whether the split into `Project Setup`, `Experiment Run`, `Run Status / Logs`,
  and `Advanced` matches the real workflow.

