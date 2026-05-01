# AGENTS.MD

Machine-readable operating guide for agents working in `bpm_prediction`.

This is the first document an agent must read. Keep it short: routing, hard
rules, validation commands, and documentation drift checks only.

---

## Status

- `active_phase`: MVP2.5 Stage 4.2
- `primary_runtime`: Python local virtual environment
- `primary_interface`: CLI
- `last_updated`: 2026-05-01

---

## Mandatory First Read

Always read first:

1. `AGENTS.MD`
2. `docs/current/project-state.md`
3. `docs/current/architecture-debt.md`

Then read only task-specific documents from `Task Routing`.

Do not load all Markdown files by default.

Use `docs/index.md` only for documentation discovery, documentation
reorganization, or source-of-truth cleanup.

---

## Truth Priority

When documents conflict, use this priority:

1. `AGENTS.MD`
2. `docs/current/project-state.md`
3. `docs/current/architecture-debt.md`
4. Accepted ADRs in `docs/adr/`
5. Current architecture canon:
   - `docs/ARCHITECTURE_GUIDELINES.MD`
   - `docs/ARCHITECTURE_RULES.MD`
   - `docs/TARGET_ARCHITECTURE.MD`
6. Active MVP2.5 docs:
   - `docs/ARCHITECTURE_MVP2_5.MD`
   - `docs/DATA_MODEL_MVP2_5.MD`
   - `docs/DATA_FLOWS_MVP2_5.MD`
   - `docs/LLD_MVP2_5.MD`
   - `docs/EVF_MVP2_5.MD`
   - `docs/GNN_RUNTIME_MVP2_5.MD`
7. Adapter specs:
   - `docs/ADAPTER_XES.MD`
   - `docs/ADAPTER_CAMUNDA_SQL.MD`
8. User-facing docs:
   - `README.MD`
   - `docs/GLOSSARY.MD`
   - `docs/VARIABLES.MD`
9. Historical worklogs in `docs/worklogs/`

If implementation and documentation conflict, stop and ask unless the task is
explicitly documentation synchronization.

---

## Hard Runtime Rules

- Train/eval/infer must consume prebuilt structure and stats artifacts through
  `IKnowledgeGraphPort`.
- Train/eval/infer must not rebuild topology or stats synchronously from raw
  sources.
- Research-grade temporal runs must use:

```yaml
experiment.stats_time_policy: strict_asof
experiment.on_missing_asof_snapshot: raise
```

Exploratory runs may use fallback policies, but must be reported as
exploratory and not used as dissertation-grade evidence.

---

## Clean Architecture Rules

Layer layout:

```text
src/domain/
src/application/
src/adapters/
src/infrastructure/
tools/
```

Dependency rules:

- `src/domain/` must not import `src.application`, `src.adapters`, or
  `src.infrastructure`.
- `src/application/` must not import concrete adapters.
- `src/adapters/` must not depend on concrete application use cases.
- External systems must be accessed through ports/adapters.

Validation:

```powershell
.\.venv\Scripts\python.exe tools\architecture_guard.py
```

---

## Task Routing

### Architecture / Clean Architecture

Read:

- `docs/ARCHITECTURE_GUIDELINES.MD`
- `docs/ARCHITECTURE_RULES.MD`
- `docs/TARGET_ARCHITECTURE.MD`
- relevant accepted ADRs in `docs/adr/`

### MVP2.5 Core

For any MVP2.5 runtime, data contract, repository, stats, topology, train/eval,
GNN, XES, or Camunda task, read:

- `docs/ARCHITECTURE_MVP2_5.MD`
- `docs/DATA_MODEL_MVP2_5.MD`
- `docs/DATA_FLOWS_MVP2_5.MD`
- `docs/LLD_MVP2_5.MD`

### GNN / Tensor / Training / Drift

Read `MVP2.5 Core`, then:

- `docs/GNN_RUNTIME_MVP2_5.MD`
- `docs/EVF_MVP2_5.MD`
- `docs/VARIABLES.MD` when metrics, notation, or dissertation reasoning are involved

### Stats Snapshots

Read `MVP2.5 Core`, then:

- `docs/EVF_MVP2_5.MD`
- `docs/runbooks/mvp2_5-commands.md`

### XES Adapter

Read `MVP2.5 Core`, then:

- `docs/ADAPTER_XES.MD`
- `docs/GNN_RUNTIME_MVP2_5.MD` if target generation or lifecycle filtering is involved

### Camunda Adapter

Read `MVP2.5 Core`, then:

- `docs/ADAPTER_CAMUNDA_SQL.MD`

### Commands / Experiments / UI / Config

Read:

- `docs/runbooks/mvp2_5-commands.md`
- `configs/ui/config_catalog.yaml` for config/UI changes
- `tools/experiment_ui.py` for desktop UI changes
- `tools/web_experiment_ui_app.py` for web prototype changes
- `docs/EVF_MVP2_5.MD` for experiment or evaluation semantics

Required rule:

- Every new config key in `experiment`, `data`, `mapping`, `model`,
  `training`, `tracking`, or `sync_stats` must be added to
  `configs/ui/config_catalog.yaml`.

---

## Active Architecture Debt

Active debt source of truth:

```text
docs/current/architecture-debt.md
```

Do not duplicate the active debt list in this file.

---

## ADR Policy

ADR directory:

```text
docs/adr/
```

ADR guide:

```text
docs/adr/README.md
```

Use ADRs for durable architectural decisions. Do not use long-form worklogs as
source of truth.

---

## Historical Worklog Policy

Ignore `docs/worklogs/` by default.

Read worklogs only when directly referenced by the user, an ADR,
`docs/current/*`, or when the task explicitly requires historical analysis.

Worklogs are never source of truth.

---

## Python Environment

Use only:

```powershell
.\.venv\Scripts\python.exe <command>
```

Do not use:

```powershell
python
py -3
```

If `.venv\Scripts\python.exe` fails with
`Unable to create process using ...Python311...`, rerun the same command with
escalated permissions. The venv is not necessarily broken.

Canonical validation commands:

```powershell
.\.venv\Scripts\python.exe tools\architecture_guard.py
.\.venv\Scripts\python.exe -m pytest -m mvp1_regression -v
.\.venv\Scripts\python.exe -m pytest tests/ -v
```

---

## Experiment Validity Guardrails

For train/eval/eval_drift result analysis, verify:

- loaded checkpoint matches the intended preset, gateway mode, and activity vocab;
- `vocab_activity_size > 1`;
- mask-aware runs report meaningful `target_in_mask_rate`;
- `F1=1.0` with `target_in_mask_rate=0` or `test_oos=1` is invalid until proven otherwise;
- `gateway_mode=collapse_for_prediction` is the canonical mode for XES task-label
  prediction unless the task explicitly investigates preserve-mode behavior.

---

## Documentation Language Policy

Machine-readable agent files should be English:

- `AGENTS.MD`
- ADR metadata and section headers
- status keys
- routing keys
- validation command labels

Human-facing architecture documentation may be Ukrainian.

For Ukrainian documentation, keep section names, metadata keys, and important
machine-readable labels in English where practical.

---

## Documentation Drift Check

Before finishing any task that changes code, config, CLI behavior, data
contracts, runtime semantics, tests, or architecture, explicitly check
documentation impact.

Checklist:

- `runtime_behavior_changed`: update `docs/current/project-state.md`.
- `architecture_decision_changed`: create or update ADR in `docs/adr/`.
- `architecture_debt_changed`: update `docs/current/architecture-debt.md`.
- `cli_or_command_changed`: update `docs/runbooks/mvp2_5-commands.md` and
  `README.MD` if user-facing.
- `config_key_added_or_changed`: update `configs/ui/config_catalog.yaml`.
- `data_contract_changed`: update `docs/DATA_MODEL_MVP2_5.MD` and related runtime docs.
- `gnn_tensor_or_forward_changed`: update `docs/GNN_RUNTIME_MVP2_5.MD`.
- `docs_routing_changed`: update `AGENTS.MD` and `docs/index.md`.

The final response must include one of:

- `Documentation drift check: updated <files>`
- `Documentation drift check: no documentation changes required because <reason>`

Never silently skip this check.


