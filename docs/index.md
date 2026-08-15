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
- `last_updated`: 2026-05-24

---

## Entry Points

### For Active Research Path

- [docs/GNN_RUNTIME_MVP2_5.MD](GNN_RUNTIME_MVP2_5.MD)
  - **Focus:** runtime contract for `EOPKGTopologyConditioned`, dynamic
    topology-native candidate space `C_v`, and impulse activation routing.

- [docs/GNN_LEARNING_METODOLOGY.MD](GNN_LEARNING_METODOLOGY.MD)
  - **Focus:** methodology rationale for moving from fixed-head/fusion
    ablations to topology-conditioned candidate scoring under structural drift.

- [docs/current/project-state.md](current/project-state.md)
  - **Focus:** current implementation state and role grouping: primary path,
    reviewer controls, and experimental structural modes.

### For Agents

- [AGENTS.MD](../AGENTS.MD)
  - **Description (ukr):** перший файл для агента. Визначає routing, truth
    priority, validation commands, ADR policy і documentation update rule.

### For Current Project State

- [docs/current/project-state.md](current/project-state.md)
  - **Description (ukr):** короткий актуальний стан MVP2.5 Stage 4.2: що вже
    реалізовано, які runtime invariants діють, які validation gates використовувати.

- [docs/current/architecture-debt.md](current/architecture-debt.md)
  - **Description (ukr):** актуальний список архітектурного боргу з українськими
    поясненнями проблем і посиланнями на ADR.

- [docs/current/phase-2c-review-fix-plan.md](current/phase-2c-review-fix-plan.md)
  - **Focus:** actionable review findings after Phase-2C completion: runtime
    consistency, candidate topology batching, config/catalog wording, and
    validation gaps to close before treating Phase-2C as cleanly complete.

### For Users

- [README.MD](../README.MD)
  - **Description (ukr):** головна user-facing точка входу: опис проекту,
    quick start і посилання на актуальні operational runbooks.

---

## Runbooks

- [docs/runbooks/mvp2_5-commands.md](runbooks/mvp2_5-commands.md)
  - **Description (ukr):** актуальний каталог операційних команд MVP2.5:
    environment, validation, train/eval, topology preparation, stats snapshots,
    research-safe workflow, visualization, tools, cache maintenance і UI commands.

---

## ADR

- [docs/adr/README.md](adr/README.md)
  - **Description (ukr):** правила ведення ADR, статуси, naming і шаблон.

- [docs/adr/0001-agent-knowledge-base-governance.md](adr/0001-agent-knowledge-base-governance.md)
  - **Description (ukr):** рішення про `AGENTS.MD`, ADR і обмеження ролі worklogs.

- [docs/adr/0002-offline-topology-runtime-separation.md](adr/0002-offline-topology-runtime-separation.md)
  - **Description (ukr):** рішення про фізичне розділення offline topology/stats
    preparation і runtime train/eval/infer.

- [docs/adr/0003-immutable-json-stats-snapshots.md](adr/0003-immutable-json-stats-snapshots.md)
  - **Description (ukr):** рішення про immutable JSON-only stats snapshots.

- [docs/adr/0004-strict-asof-research-policy.md](adr/0004-strict-asof-research-policy.md)
  - **Description (ukr):** рішення про `strict_asof + raise` для research-grade
    temporal experiments.

- [docs/adr/0005-snapshot-homogeneous-batching.md](adr/0005-snapshot-homogeneous-batching.md)
  - **Description (ukr):** proposed target для snapshot-homogeneous batching.

- [docs/adr/0006-research-grade-activity-topology-alignment-gate.md](adr/0006-research-grade-activity-topology-alignment-gate.md)
  - **Description (ukr):** proposed target для strict activity-to-topology
    alignment gate.

- [docs/adr/0007-topology-projection-alignment.md](adr/0007-topology-projection-alignment.md)
  - **Description (ukr):** proposed target для alignment після topology projection.

- [docs/adr/0008-cli-composition-root-boundary.md](adr/0008-cli-composition-root-boundary.md)
  - **Description (ukr):** proposed target для декомпозиції `src/cli.py`.

- [docs/adr/0009-impulse-activation-topology-state-routing.md](adr/0009-impulse-activation-topology-state-routing.md) (`Status: Proposed`)
  - **Description (ukr):** proposed hypothesis / planned rework для `EOPKGTopologyConditioned`: динамічна імпульсна активація topology-native candidate graph через GNN індуктивності та топологічне трасування стану (Topology State Routing). Не є прийнятим рішенням, доки ADR має статус `Proposed`.


---

## Architecture Canon

- [docs/ARCHITECTURE_GUIDELINES.MD](ARCHITECTURE_GUIDELINES.MD)
  - **Description (ukr):** архітектурна конституція: Clean/Hexagonal principles,
    modes, MVP stages, scientific integrity gate.

- [docs/ARCHITECTURE_RULES.MD](ARCHITECTURE_RULES.MD)
  - **Description (ukr):** enforceable dependency boundaries для `domain`,
    `application`, `adapters`.

- [docs/TARGET_ARCHITECTURE.MD](TARGET_ARCHITECTURE.MD)
  - **Description (ukr):** target blueprint незалежно від конкретного MVP:
    module layout, ports, knowledge lifecycle, enterprise direction.

---

## Active MVP2.5 Reference

- [docs/ARCHITECTURE_MVP2_5.MD](ARCHITECTURE_MVP2_5.MD)
  - **Description (ukr):** системна архітектура MVP2.5: offline ingestion,
    stats snapshots, repository-backed runtime.

- [docs/DATA_MODEL_MVP2_5.MD](DATA_MODEL_MVP2_5.MD)
  - **Description (ukr):** DTO і artifact contracts: `ProcessStructureDTO`,
    stats snapshots, `GraphTensorContract`.

- [docs/DATA_FLOWS_MVP2_5.MD](DATA_FLOWS_MVP2_5.MD)
  - **Description (ukr):** data flows для `ingest-topology`, `sync-topology`,
    `sync-stats`, train/eval/infer.

- [docs/LLD_MVP2_5.MD](LLD_MVP2_5.MD)
  - **Description (ukr):** low-level repository algorithms, stats mapping,
    fallback matrix, known limitations.

- [docs/EVF_MVP2_5.MD](EVF_MVP2_5.MD)
  - **Description (ukr):** evaluation framework, anti-leakage protocol,
    metrics and reproducibility checklist.

- [docs/GNN_RUNTIME_MVP2_5.MD](GNN_RUNTIME_MVP2_5.MD)
  - **Description (ukr):** model registry, tensor pipeline, target `y`
    generation, model forward behavior.

- [docs/GNN_LEARNING_STRATEGY.MD](GNN_LEARNING_STRATEGY.MD)
  - **Description (ukr):** learning strategy contract for GNN training:
    `standard`, `topology_conditioned`, loss design, version-safe negative
    sampling, controlled forgetting, and `versioned_zero_shot` protocol.

- [docs/GNN_LEARNING_METODOLOGY.MD](GNN_LEARNING_METODOLOGY.MD)
  - **Description (ukr):** research methodology roadmap for business-valid
    topology-conditioned zero-shot adaptation: candidate-set drift, dynamic
    candidate scoring, calibration, auditability, and staged transition beyond
    fixed-head fusion ablations.

---

## Adapter Reference

- [docs/ADAPTER_XES.MD](ADAPTER_XES.MD)
  - **Description (ukr):** XES/MXML ingestion contract, lifecycle filtering,
    duration pairing, schema resolution.

- [docs/ADAPTER_CAMUNDA_SQL.MD](ADAPTER_CAMUNDA_SQL.MD)
  - **Description (ukr):** Camunda 7 MSSQL/files adapter contract, runtime events,
    BPMN extraction, cleanup-aware behavior.

---

## Domain Reference

- [docs/GLOSSARY.MD](GLOSSARY.MD)
  - **Description (ukr):** canonical glossary for POKG/EPOKG, drift, reliability,
    OOS, fusion graph, terms and concepts.

- [docs/VARIABLES.MD](VARIABLES.MD)
  - **Description (ukr):** mathematical notation to code naming map.

- [docs/research_comparation.MD](research_comparation.MD)
  - **Description (ukr):** наукова новизна та порівняльний аналіз літератури:
    зіставлення EOPKGTopologyConditioned з сучасними SOTA моделями (Lischka, Wang,
    Rizzi, ProcessGFM, SAGN, PROPHET, IPA-GNN, SNAP, TPOP) та обґрунтування плагіат-захисту.

- [docs/dissertation_math_fix.MD](dissertation_math_fix.MD)
  - **Description (ukr):** математичні та архітектурні невідповідності:
    аналіз розбіжностей між теоретичним описом у дисертації (модель злиття, онлайн-еволюція,
    семафор надійності) та практичною програмною реалізацією Impulse Activation Routing.



---

## UI And Tooling

- [docs/UI_SPECA.MD](UI_SPECA.MD)
  - **Description (ukr):** UI-specific specification.

- [configs/ui/config_catalog.yaml](../configs/ui/config_catalog.yaml)
  - **Description (ukr):** config key catalog for interactive tooling. Update it
    whenever config keys are added or renamed.

---

## Historical MVP Docs

Use only for backward compatibility or historical context:

- [docs/ARCHITECTURE_MVP1.MD](ARCHITECTURE_MVP1.MD)
- [docs/DATA_MODEL_MVP1.MD](DATA_MODEL_MVP1.MD)
- [docs/DATA_FLOWS_MVP1.MD](DATA_FLOWS_MVP1.MD)
- [docs/LLD_MVP1.MD](LLD_MVP1.MD)
- [docs/EVF_MVP1.MD](EVF_MVP1.MD)
- [docs/ARCHITECTURE_MVP2.MD](ARCHITECTURE_MVP2.MD)
- [docs/DATA_MODEL_MVP2.MD](DATA_MODEL_MVP2.MD)
- [docs/DATA_FLOWS_MVP2.MD](DATA_FLOWS_MVP2.MD)
- [docs/LLD_MVP2.MD](LLD_MVP2.MD)
- [docs/EVF_MVP2.MD](EVF_MVP2.MD)

**Description (ukr):**

Ці файли не є active MVP2.5 source of truth, але важливі для backward
compatibility, regression reasoning і розуміння еволюції.

---

## Roadmap Policy

### Current Rule

- [docs/ARCHITECTURE_GUIDELINES.MD](ARCHITECTURE_GUIDELINES.MD) keeps the high-level MVP sequence:
  MVP1, MVP2, MVP3, MVP4, MVP5.
- [docs/TARGET_ARCHITECTURE.MD](TARGET_ARCHITECTURE.MD) keeps the target architecture blueprint.
- [README.MD](../README.MD) may describe user-facing current capabilities and operational
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
