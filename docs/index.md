# Documentation Index

Documentation map for `bpm_prediction`.

This file helps humans and agents find the right document without scanning the
entire repository.

---

## Metadata

- `status`: active
- `audience`: human-and-agent
- `source_of_truth`: navigation-only
- `language_policy`: keys and section headers in English, human descriptions in Ukrainian
- `last_updated`: 2026-04-27

---

## Entry Points

### For Agents

- `AGENTS.MD`
  - **Description (ukr):** перший файл для агента. Визначає routing, truth
    priority, validation commands, ADR policy і documentation update rule.

### For Current Project State

- `docs/current/project-state.md`
  - **Description (ukr):** короткий актуальний стан MVP2.5 Stage 4.2: що вже
    реалізовано, які runtime invariants діють, які validation gates використовувати.

- `docs/current/architecture-debt.md`
  - **Description (ukr):** актуальний список архітектурного боргу з українськими
    поясненнями проблем і посиланнями на ADR.

### For Users

- `README.MD`
  - **Description (ukr):** головна user-facing точка входу: опис проекту,
    quick start і посилання на актуальні operational runbooks.

---

## Runbooks

- `docs/runbooks/mvp2_5-commands.md`
  - **Description (ukr):** актуальний каталог операційних команд MVP2.5:
    environment, validation, train/eval, topology preparation, stats snapshots,
    research-safe workflow, visualization, tools, cache maintenance і UI commands.

---

## ADR

- `docs/adr/README.md`
  - **Description (ukr):** правила ведення ADR, статуси, naming і шаблон.

- `docs/adr/0001-agent-knowledge-base-governance.md`
  - **Description (ukr):** рішення про `AGENTS.MD`, ADR і обмеження ролі worklogs.

- `docs/adr/0002-offline-topology-runtime-separation.md`
  - **Description (ukr):** рішення про фізичне розділення offline topology/stats
    preparation і runtime train/eval/infer.

- `docs/adr/0003-immutable-json-stats-snapshots.md`
  - **Description (ukr):** рішення про immutable JSON-only stats snapshots.

- `docs/adr/0004-strict-asof-research-policy.md`
  - **Description (ukr):** рішення про `strict_asof + raise` для research-grade
    temporal experiments.

- `docs/adr/0005-snapshot-homogeneous-batching.md`
  - **Description (ukr):** proposed target для snapshot-homogeneous batching.

- `docs/adr/0006-research-grade-activity-topology-alignment-gate.md`
  - **Description (ukr):** proposed target для strict activity-to-topology
    alignment gate.

- `docs/adr/0007-topology-projection-alignment.md`
  - **Description (ukr):** proposed target для alignment після topology projection.

- `docs/adr/0008-cli-composition-root-boundary.md`
  - **Description (ukr):** proposed target для декомпозиції `src/cli.py`.

---

## Architecture Canon

- `docs/ARCHITECTURE_GUIDELINES.MD`
  - **Description (ukr):** архітектурна конституція: Clean/Hexagonal principles,
    modes, MVP stages, scientific integrity gate.

- `docs/ARCHITECTURE_RULES.MD`
  - **Description (ukr):** enforceable dependency boundaries для `domain`,
    `application`, `adapters`.

- `docs/TARGET_ARCHITECTURE.MD`
  - **Description (ukr):** target blueprint незалежно від конкретного MVP:
    module layout, ports, knowledge lifecycle, enterprise direction.

---

## Active MVP2.5 Reference

- `docs/ARCHITECTURE_MVP2_5.MD`
  - **Description (ukr):** системна архітектура MVP2.5: offline ingestion,
    stats snapshots, repository-backed runtime.

- `docs/DATA_MODEL_MVP2_5.MD`
  - **Description (ukr):** DTO і artifact contracts: `ProcessStructureDTO`,
    stats snapshots, `GraphTensorContract`.

- `docs/DATA_FLOWS_MVP2_5.MD`
  - **Description (ukr):** data flows для `ingest-topology`, `sync-topology`,
    `sync-stats`, train/eval/infer.

- `docs/LLD_MVP2_5.MD`
  - **Description (ukr):** low-level repository algorithms, stats mapping,
    fallback matrix, known limitations.

- `docs/EVF_MVP2_5.MD`
  - **Description (ukr):** evaluation framework, anti-leakage protocol,
    metrics and reproducibility checklist.

- `docs/GNN_RUNTIME_MVP2_5.MD`
  - **Description (ukr):** model registry, tensor pipeline, target `y`
    generation, model forward behavior.

---

## Adapter Reference

- `docs/ADAPTER_XES.MD`
  - **Description (ukr):** XES/MXML ingestion contract, lifecycle filtering,
    duration pairing, schema resolution.

- `docs/ADAPTER_CAMUNDA_SQL.MD`
  - **Description (ukr):** Camunda 7 MSSQL/files adapter contract, runtime events,
    BPMN extraction, cleanup-aware behavior.

---

## Domain Reference

- `docs/GLOSSARY.MD`
  - **Description (ukr):** canonical glossary for POKG/EPOKG, drift, reliability,
    OOS, fusion graph, terms and concepts.

- `docs/VARIABLES.MD`
  - **Description (ukr):** mathematical notation to code naming map.

---

## UI And Tooling

- `docs/UI_SPECA.MD`
  - **Description (ukr):** UI-specific specification.

- `configs/ui/config_catalog.yaml`
  - **Description (ukr):** config key catalog for interactive tooling. Update it
    whenever config keys are added or renamed.

---

## Historical MVP Docs

Use only for backward compatibility or historical context:

- `docs/ARCHITECTURE_MVP1.MD`
- `docs/DATA_MODEL_MVP1.MD`
- `docs/DATA_FLOWS_MVP1.MD`
- `docs/LLD_MVP1.MD`
- `docs/EVF_MVP1.MD`
- `docs/ARCHITECTURE_MVP2.MD`
- `docs/DATA_MODEL_MVP2.MD`
- `docs/DATA_FLOWS_MVP2.MD`
- `docs/LLD_MVP2.MD`
- `docs/EVF_MVP2.MD`

**Description (ukr):**

Ці файли не є active MVP2.5 source of truth, але важливі для backward
compatibility, regression reasoning і розуміння еволюції.

---

## Archive

- `docs/archive/`
  - **Description (ukr):** архів застарілих документів, які більше не є
    active source of truth.

### Deprecated Agent Contexts

Use only when explicitly requested by the user:

- `docs/archive/agent-contexts/AGENT_GUIDE.MD`
- `docs/archive/agent-contexts/PROJECT_CONTEXT.md`
- `docs/archive/agent-contexts/AGENT_CONTEXT_MVP1.MD`
- `docs/archive/agent-contexts/AGENT_CONTEXT_MVP2.MD`
- `docs/archive/agent-contexts/AGENT_CONTEXT_MVP2_5.MD`

**Description (ukr):**

Ці файли більше не є first-read entrypoint для агента. Актуальний маршрут:
`AGENTS.MD` -> `docs/current/project-state.md` ->
`docs/current/architecture-debt.md`.

---

## Worklogs

- `docs/worklogs/`
  - **Description (ukr):** історичні плани, звіти і аналізи. Не source of truth
    за замовчуванням. Використовувати тільки якщо на них посилається
    `AGENTS.MD`, `docs/current/*`, ADR або користувач.

Important current-context worklogs:

- `docs/worklogs/Finish_MVP2_5_Plan.MD`
- `docs/worklogs/MVP2_5_Canonical_Doc_Sync_and_Architecture_Debt_2026-04-24.MD`
- `docs/worklogs/MVP2_5_Dissertation_Alignment_and_Blocking_Debt_Analysis_2026-03-21.MD`
- `docs/worklogs/MVP2_5_Stage4_2_OptionA_Unbatch_Fix_Report.MD`
- `docs/worklogs/mismatch_fix_plan.md`
- `docs/worklogs/change_attention_aproach.md`

---

## Roadmap Policy

### Current Rule

- `docs/ARCHITECTURE_GUIDELINES.MD` keeps the high-level MVP sequence:
  MVP1, MVP2, MVP3, MVP4, MVP5.
- `docs/TARGET_ARCHITECTURE.MD` keeps the target architecture blueprint.
- `README.MD` may describe user-facing current capabilities and operational
  entry-point commands, but the full command catalog belongs in runbooks.

### Recommendation

Create a separate roadmap file only if roadmap decisions start changing often:

```text
docs/current/roadmap.md
```

**Description (ukr):**

Зараз roadmap частково живе в `ARCHITECTURE_GUIDELINES.MD` як high-level MVP
послідовність, а target state — в `TARGET_ARCHITECTURE.MD`. Це нормально для
стабільної стратегії. Окремий `docs/current/roadmap.md` потрібен лише тоді, коли
треба регулярно оновлювати найближчі milestones, статуси блокерів, порядок
закриття debt і критерії переходу до MVP3.

### Do Not Duplicate

Do not maintain competing roadmap sections in multiple files.

If `docs/current/roadmap.md` is created later:

1. keep `ARCHITECTURE_GUIDELINES.MD` as stable high-level phase canon,
2. keep `TARGET_ARCHITECTURE.MD` as target blueprint,
3. keep `docs/current/roadmap.md` as tactical current roadmap,
4. link to it from `README.MD`, `AGENTS.MD`, and this index.

---

## Maintenance Rule

When adding, moving, or deprecating documentation:

1. update this index,
2. update `AGENTS.MD` only if routing changes,
3. add or update ADR if the move changes source-of-truth policy,
4. do not make historical worklogs source of truth.
