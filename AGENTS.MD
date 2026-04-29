# AGENTS.MD

Machine-readable operating guide for agents working in `bpm_prediction`.

This file is the first document an agent must read. It is intentionally short.
It routes the agent to the minimum required documentation for the current task.

---

## Status

- `owner`: Serhii Korotenko
- `project`: bpm_prediction
- `active_phase`: MVP2.5 Stage 4.2
- `primary_runtime`: Python local virtual environment
- `primary_interface`: CLI
- `last_updated`: 2026-04-27

---

## Mandatory First Read

Always read these first:

1. `AGENTS.MD`
2. `docs/current/project-state.md`
3. `docs/current/architecture-debt.md`

Then read only task-specific documents from `Task Routing`.

Do not load all Markdown files by default.

Use `docs/index.md` only when the task is about documentation discovery,
documentation reorganization, or source-of-truth cleanup.

---

## Truth Priority

Use this priority order when documents conflict:

1. `AGENTS.MD`
2. `docs/current/project-state.md`
3. `docs/current/architecture-debt.md`
4. Accepted ADRs in `docs/adr/` when present
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

Historical worklogs are not source of truth unless explicitly listed in this file
or referenced by an accepted ADR.

If implementation and documentation conflict, stop and ask unless the task is
explicitly documentation synchronization.

---

## Current Runtime Rules

MVP2.5 uses physically separated pipelines:

1. Offline knowledge preparation:
   - `ingest-topology`
   - `sync-topology`
   - `sync-stats`
   - `sync-stats-backfill`
2. Runtime ML:
   - train
   - eval
   - infer

Hard rule:

- Train/eval/infer must consume prebuilt structure and stats artifacts through
  `IKnowledgeGraphPort`.
- Train/eval/infer must not rebuild topology or stats synchronously from raw
  sources.

Research-grade temporal runs should use:

```yaml
experiment.stats_time_policy: strict_asof
experiment.on_missing_asof_snapshot: raise
```

Exploratory runs may use fallback policies, but they must be reported as
exploratory and not used as dissertation-grade evidence.

---

## Snapshot Policy

Stats snapshots are immutable and append-only.

Required snapshot identity:

```text
tenant_id + process_name + version_key + proc_def_id + knowledge_version
```

Required storage policy:

- JSON-only stats payload
- no normalized `NodeStat` or `EdgeStat` graph nodes
- no TTL
- as-of lookup selects the latest snapshot where `snapshot.as_of_ts <= requested_as_of`

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

Validation command:

```powershell
.\.venv\Scripts\python.exe tools\architecture_guard.py
```

---

## Task Routing

### Architecture Task

Read:

1. `docs/ARCHITECTURE_GUIDELINES.MD`
2. `docs/ARCHITECTURE_RULES.MD`
3. `docs/TARGET_ARCHITECTURE.MD`
4. Relevant accepted ADRs in `docs/adr/` when present

### MVP2.5 Runtime Task

Read:

1. `docs/ARCHITECTURE_MVP2_5.MD`
2. `docs/DATA_MODEL_MVP2_5.MD`
3. `docs/DATA_FLOWS_MVP2_5.MD`
4. `docs/LLD_MVP2_5.MD`
5. `docs/GNN_RUNTIME_MVP2_5.MD`

### Stats Snapshot Task

Read:

1. `docs/ARCHITECTURE_MVP2_5.MD`
2. `docs/DATA_MODEL_MVP2_5.MD`
3. `docs/DATA_FLOWS_MVP2_5.MD`
4. `docs/LLD_MVP2_5.MD`
5. `docs/EVF_MVP2_5.MD`
6. `docs/runbooks/mvp2_5-commands.md`

### GNN Model or Tensor Contract Task

Read:

1. `docs/GNN_RUNTIME_MVP2_5.MD`
2. `docs/DATA_MODEL_MVP2_5.MD`
3. `docs/LLD_MVP2_5.MD`
4. `docs/VARIABLES.MD`

### XES Adapter Task

Read:

1. `docs/ADAPTER_XES.MD`
2. `docs/DATA_MODEL_MVP2_5.MD`
3. `docs/GNN_RUNTIME_MVP2_5.MD` if target generation or lifecycle filtering is involved

### Camunda Adapter Task

Read:

1. `docs/ADAPTER_CAMUNDA_SQL.MD`
2. `docs/DATA_MODEL_MVP2_5.MD`
3. `docs/DATA_FLOWS_MVP2_5.MD`

### Experiment or Runbook Task

Read:

1. `docs/runbooks/mvp2_5-commands.md`
2. `docs/worklogs/experiment_matrix_runbook.MD` if matrix experiments are involved
3. `docs/EVF_MVP2_5.MD`

### UI or Config Catalog Task

Read:

1. `AGENTS.MD`
2. `configs/ui/config_catalog.yaml`
3. `tools/experiment_ui.py`
4. `tools/web_experiment_ui_app.py` if web prototype is involved

Required rule:

- Every new config key in `experiment`, `data`, `mapping`, `model`,
  `training`, `tracking`, or `sync_stats` must be added to
  `configs/ui/config_catalog.yaml`.

---

## Known Active Architecture Debt

Track these as active debt unless superseded by ADR:

1. `snapshot_homogeneous_batching`
   - ADR: `docs/adr/0005-snapshot-homogeneous-batching.md`
   - Current behavior: mixed snapshots may appear in one PyG batch.
   - Current workaround: first graph structural payload is used for forward.
   - Risk: not research-grade for strict temporal experiments.

2. `activity_to_topology_alignment_gate`
   - ADR: `docs/adr/0006-research-grade-activity-topology-alignment-gate.md`
   - Need stricter preflight validation for event-to-BPMN/topology alignment.
   - Research-grade runs need locked thresholds.

3. `topology_projection_alignment`
   - ADR: `docs/adr/0007-topology-projection-alignment.md`
   - Projection diagnostics and strict `on_fail` are implemented.
   - In Codex sandbox, run `.venv\Scripts\python.exe -m pytest ...` with
     escalated permissions if base interpreter process creation is blocked.
   - Keep architecture compatible with future node-identity-aware mapping.

4. `duplicate_activity_identity_ambiguity`
   - ADR: none yet.
   - Multiple BPMN prediction nodes can share one log activity label/name.
   - Treat as separate from topology projection and parallel mask semantics.
   - Future architecture should support node-identity-aware mapping.

5. `cli_composition_root_overgrowth`
   - ADR: `docs/adr/0008-cli-composition-root-boundary.md`
   - `src/cli.py` is no longer only a thin composition root.
   - Refactoring should be planned, not done opportunistically.

6. `mvp3_reliability_semaphore_not_ready`
   - OOD/Wasserstein calibration, latent artifact pipeline, and
     Green/Yellow/Red action policy are future MVP3 work.

---

## Historical Worklog Policy

`docs/worklogs/` contains historical planning and reports.

Rules:

- Do not read all worklogs by default.
- Do not treat old worklogs as current requirements.
- Use worklogs only when explicitly referenced by:
  - the user,
  - `docs/current/project-state.md`,
  - this file,
  - an accepted ADR.
- If a worklog conflicts with current architecture docs or ADRs, current docs
  and ADRs win.

Currently relevant worklogs:

1. `docs/current/architecture-debt.md`
2. `docs/worklogs/Finish_MVP2_5_Plan.MD`
3. `docs/worklogs/MVP2_5_Canonical_Doc_Sync_and_Architecture_Debt_2026-04-24.MD`
4. `docs/worklogs/MVP2_5_Dissertation_Alignment_and_Blocking_Debt_Analysis_2026-03-21.MD`
5. `docs/worklogs/MVP2_5_Stage4_2_OptionA_Unbatch_Fix_Report.MD`
6. `docs/worklogs/mismatch_fix_plan.md`
7. `docs/worklogs/change_attention_aproach.md`

---

## ADR Policy

Target location:

```text
docs/adr/
```

ADR filenames:

```text
0001-short-decision-name.md
0002-short-decision-name.md
```

ADR statuses:

- `Proposed`
- `Accepted`
- `Superseded`
- `Deprecated`

ADR template:

```md
# ADR-0000: Decision Title

Date: YYYY-MM-DD
Status: Proposed | Accepted | Superseded | Deprecated

## Context

Why this decision is needed.

## Decision

What is decided.

## Consequences

Positive and negative consequences.

## Runtime Rules

Concrete rules agents and developers must follow.

## Affected Files

- `path/to/file`

## Related

- `path/to/doc-or-worklog`
```

Recommended initial ADRs:

1. `offline-topology-runtime-separation`
2. `immutable-stats-snapshots`
3. `strict-asof-research-policy`
4. `json-only-stats-payload`
5. `snapshot-homogeneous-batching-debt`
6. `activity-to-topology-alignment-gate`
7. `cli-composition-root-boundary`
8. `python-venv-sandbox-execution`

---

## Python Environment

Use the project virtual environment:

```powershell
.\.venv\Scripts\python.exe <command>
```

Do not rely on:

```powershell
py -3
python
```

They may be missing from PATH in agent sessions.

Known Codex sandbox behavior:

- `.venv\Scripts\python.exe` may delegate to the base interpreter from
  `.venv\pyvenv.cfg`.
- On this machine, the base interpreter is under:

- The default sandbox may block process creation for that base interpreter.
- If the command fails with `Unable to create process using ...Python311...`,
  the venv is not necessarily broken.
- Rerun the same `.venv\Scripts\python.exe` command with escalated permissions.

Canonical validation commands:

```powershell
.\.venv\Scripts\python.exe tools\architecture_guard.py
.\.venv\Scripts\python.exe -m pytest -m mvp1_regression -v
.\.venv\Scripts\python.exe -m pytest tests/ -v
```

---

## Default Agent Workflow

1. Read `AGENTS.MD`.
2. Read `docs/current/project-state.md`.
3. Read only task-specific docs from `Task Routing`.
4. Check `git status --short --branch`.
5. Identify relevant files before proposing changes.
6. For code changes, state:
   - files to modify,
   - documentation contracts used,
   - validation commands.
7. Do not edit unrelated files.
8. Do not revert user changes unless explicitly requested.
9. Run focused validation where feasible.
10. Report validation results and any unverified risk.

---

## Documentation Language Policy

Machine-readable agent files should be English:

- `AGENTS.MD`
- future ADR metadata and section headers
- status keys
- routing keys
- validation command labels

Human-facing architecture documentation may be Ukrainian.

For Ukrainian documentation, keep section names, metadata keys, and important
machine-readable labels in English where practical.

---

## Documentation Update Rule

Every code, config, or runtime-behavior change must include a documentation
impact check.

Use this rule:

1. Architectural decision changed:
   - create or update an ADR in `docs/adr/`.
2. Current runtime behavior changed:
   - update `docs/current/project-state.md`.
3. Known architecture debt changed:
   - update the active debt reference and related ADR if one exists.
4. Agent routing, hard rule, validation command, or active phase changed:
   - update `AGENTS.MD`.
5. User-facing command or workflow changed:
   - update `README.MD` or the relevant runbook.
6. Config key added or renamed:
   - update `configs/ui/config_catalog.yaml` in the same change.

Do not add new long-form worklogs as source of truth. If a durable decision is
made, write an ADR. If a session note is needed, keep it historical and link it
from the relevant ADR only when it adds useful context.

## Mandatory Documentation Drift Check

Before finishing any task that changes code, config, CLI behavior, data contracts,
runtime semantics, tests, or architecture, the agent must explicitly run a
documentation impact check.

Checklist:

1. `runtime_behavior_changed`
   - If yes, update `docs/current/project-state.md`.
2. `architecture_decision_changed`
   - If yes, create/update ADR in `docs/adr/`.
3. `architecture_debt_changed`
   - If yes, update `docs/current/architecture-debt.md`.
4. `cli_or_command_changed`
   - If yes, update `docs/runbooks/mvp2_5-commands.md` and README if user-facing.
5. `config_key_added_or_changed`
   - If yes, update `configs/ui/config_catalog.yaml`.
6. `data_contract_changed`
   - If yes, update `docs/DATA_MODEL_MVP2_5.MD` and related runtime docs.
7. `gnn_tensor_or_forward_changed`
   - If yes, update `docs/GNN_RUNTIME_MVP2_5.MD`.
8. `docs_routing_changed`
   - If yes, update `AGENTS.MD` and `docs/index.md`.

The final response must include one of:

- `Documentation drift check: updated <files>`
- `Documentation drift check: no documentation changes required because <reason>`

Never silently skip this check.
