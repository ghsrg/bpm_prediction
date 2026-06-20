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

### For Agents

- `AGENTS.MD`
  - **Description (ukr):** РїРµСЂС€РёР№ С„Р°Р№Р» РґР»СЏ Р°РіРµРЅС‚Р°. Р’РёР·РЅР°С‡Р°С” routing, truth
    priority, validation commands, ADR policy С– documentation update rule.

### For Current Project State

- `docs/current/project-state.md`
  - **Description (ukr):** РєРѕСЂРѕС‚РєРёР№ Р°РєС‚СѓР°Р»СЊРЅРёР№ СЃС‚Р°РЅ MVP2.5 Stage 4.2: С‰Рѕ РІР¶Рµ
    СЂРµР°Р»С–Р·РѕРІР°РЅРѕ, СЏРєС– runtime invariants РґС–СЋС‚СЊ, СЏРєС– validation gates РІРёРєРѕСЂРёСЃС‚РѕРІСѓРІР°С‚Рё.

- `docs/current/architecture-debt.md`
  - **Description (ukr):** Р°РєС‚СѓР°Р»СЊРЅРёР№ СЃРїРёСЃРѕРє Р°СЂС…С–С‚РµРєС‚СѓСЂРЅРѕРіРѕ Р±РѕСЂРіСѓ Р· СѓРєСЂР°С—РЅСЃСЊРєРёРјРё
    РїРѕСЏСЃРЅРµРЅРЅСЏРјРё РїСЂРѕР±Р»РµРј С– РїРѕСЃРёР»Р°РЅРЅСЏРјРё РЅР° ADR.

### For Users

- `README.MD`
  - **Description (ukr):** РіРѕР»РѕРІРЅР° user-facing С‚РѕС‡РєР° РІС…РѕРґСѓ: РѕРїРёСЃ РїСЂРѕРµРєС‚Сѓ,
    quick start С– РїРѕСЃРёР»Р°РЅРЅСЏ РЅР° Р°РєС‚СѓР°Р»СЊРЅС– operational runbooks.

---

## Runbooks

- `docs/runbooks/mvp2_5-commands.md`
  - **Description (ukr):** Р°РєС‚СѓР°Р»СЊРЅРёР№ РєР°С‚Р°Р»РѕРі РѕРїРµСЂР°С†С–Р№РЅРёС… РєРѕРјР°РЅРґ MVP2.5:
    environment, validation, train/eval, topology preparation, stats snapshots,
    research-safe workflow, visualization, tools, cache maintenance С– UI commands.

---

## ADR

- `docs/adr/README.md`
  - **Description (ukr):** РїСЂР°РІРёР»Р° РІРµРґРµРЅРЅСЏ ADR, СЃС‚Р°С‚СѓСЃРё, naming С– С€Р°Р±Р»РѕРЅ.

- `docs/adr/0001-agent-knowledge-base-governance.md`
  - **Description (ukr):** СЂС–С€РµРЅРЅСЏ РїСЂРѕ `AGENTS.MD`, ADR С– РѕР±РјРµР¶РµРЅРЅСЏ СЂРѕР»С– worklogs.

- `docs/adr/0002-offline-topology-runtime-separation.md`
  - **Description (ukr):** СЂС–С€РµРЅРЅСЏ РїСЂРѕ С„С–Р·РёС‡РЅРµ СЂРѕР·РґС–Р»РµРЅРЅСЏ offline topology/stats
    preparation С– runtime train/eval/infer.

- `docs/adr/0003-immutable-json-stats-snapshots.md`
  - **Description (ukr):** СЂС–С€РµРЅРЅСЏ РїСЂРѕ immutable JSON-only stats snapshots.

- `docs/adr/0004-strict-asof-research-policy.md`
  - **Description (ukr):** СЂС–С€РµРЅРЅСЏ РїСЂРѕ `strict_asof + raise` РґР»СЏ research-grade
    temporal experiments.

- `docs/adr/0005-snapshot-homogeneous-batching.md`
  - **Description (ukr):** proposed target РґР»СЏ snapshot-homogeneous batching.

- `docs/adr/0006-research-grade-activity-topology-alignment-gate.md`
  - **Description (ukr):** proposed target РґР»СЏ strict activity-to-topology
    alignment gate.

- `docs/adr/0007-topology-projection-alignment.md`
  - **Description (ukr):** proposed target РґР»СЏ alignment РїС–СЃР»СЏ topology projection.

- `docs/adr/0008-cli-composition-root-boundary.md`
  - **Description (ukr):** proposed target РґР»СЏ РґРµРєРѕРјРїРѕР·РёС†С–С— `src/cli.py`.

- `docs/adr/0009-impulse-activation-topology-state-routing.md` (`Status: Proposed`)
  - **Description (ukr):** proposed hypothesis / planned rework РґР»СЏ `EOPKGTopologyConditioned`: РґРёРЅР°РјС–С‡РЅР° С–РјРїСѓР»СЊСЃРЅР° Р°РєС‚РёРІР°С†С–СЏ topology-native candidate graph С‡РµСЂРµР· GNN С–РЅРґСѓРєС‚РёРІРЅРѕСЃС‚С– С‚Р° С‚РѕРїРѕР»РѕРіС–С‡РЅРµ С‚СЂР°СЃСѓРІР°РЅРЅСЏ СЃС‚Р°РЅСѓ (Topology State Routing). РќРµ С” РїСЂРёР№РЅСЏС‚РёРј СЂС–С€РµРЅРЅСЏРј, РґРѕРєРё ADR РјР°С” СЃС‚Р°С‚СѓСЃ `Proposed`.


---

## Architecture Canon

- `docs/ARCHITECTURE_GUIDELINES.MD`
  - **Description (ukr):** Р°СЂС…С–С‚РµРєС‚СѓСЂРЅР° РєРѕРЅСЃС‚РёС‚СѓС†С–СЏ: Clean/Hexagonal principles,
    modes, MVP stages, scientific integrity gate.

- `docs/ARCHITECTURE_RULES.MD`
  - **Description (ukr):** enforceable dependency boundaries РґР»СЏ `domain`,
    `application`, `adapters`.

- `docs/TARGET_ARCHITECTURE.MD`
  - **Description (ukr):** target blueprint РЅРµР·Р°Р»РµР¶РЅРѕ РІС–Рґ РєРѕРЅРєСЂРµС‚РЅРѕРіРѕ MVP:
    module layout, ports, knowledge lifecycle, enterprise direction.

---

## Active MVP2.5 Reference

- `docs/ARCHITECTURE_MVP2_5.MD`
  - **Description (ukr):** СЃРёСЃС‚РµРјРЅР° Р°СЂС…С–С‚РµРєС‚СѓСЂР° MVP2.5: offline ingestion,
    stats snapshots, repository-backed runtime.

- `docs/DATA_MODEL_MVP2_5.MD`
  - **Description (ukr):** DTO С– artifact contracts: `ProcessStructureDTO`,
    stats snapshots, `GraphTensorContract`.

- `docs/DATA_FLOWS_MVP2_5.MD`
  - **Description (ukr):** data flows РґР»СЏ `ingest-topology`, `sync-topology`,
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

- `docs/GNN_LEARNING_STRATEGY.MD`
  - **Description (ukr):** learning strategy contract for GNN training:
    `standard`, `topology_conditioned`, loss design, version-safe negative
    sampling, controlled forgetting, and `versioned_zero_shot` protocol.

- `docs/GNN_LEARNING_METODOLOGY.MD`
  - **Description (ukr):** research methodology roadmap for business-valid
    topology-conditioned zero-shot adaptation: candidate-set drift, dynamic
    candidate scoring, calibration, auditability, and staged transition beyond
    fixed-head fusion ablations.

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

- `docs/research_comparation.MD`
  - **Description (ukr):** РЅР°СѓРєРѕРІР° РЅРѕРІРёР·РЅР° С‚Р° РїРѕСЂС–РІРЅСЏР»СЊРЅРёР№ Р°РЅР°Р»С–Р· Р»С–С‚РµСЂР°С‚СѓСЂРё:
    Р·С–СЃС‚Р°РІР»РµРЅРЅСЏ EOPKGTopologyConditioned Р· СЃСѓС‡Р°СЃРЅРёРјРё SOTA РјРѕРґРµР»СЏРјРё (Lischka, Wang,
    Rizzi, ProcessGFM, SAGN, PROPHET, IPA-GNN, SNAP, TPOP) С‚Р° РѕР±Т‘СЂСѓРЅС‚СѓРІР°РЅРЅСЏ РїР»Р°РіС–Р°С‚-Р·Р°С…РёСЃС‚Сѓ.

- `docs/dissertation_math_fix.MD`
  - **Description (ukr):** РјР°С‚РµРјР°С‚РёС‡РЅС– С‚Р° Р°СЂС…С–С‚РµРєС‚СѓСЂРЅС– РЅРµРІС–РґРїРѕРІС–РґРЅРѕСЃС‚С–:
    Р°РЅР°Р»С–Р· СЂРѕР·Р±С–Р¶РЅРѕСЃС‚РµР№ РјС–Р¶ С‚РµРѕСЂРµС‚РёС‡РЅРёРј РѕРїРёСЃРѕРј Сѓ РґРёСЃРµСЂС‚Р°С†С–С— (РјРѕРґРµР»СЊ Р·Р»РёС‚С‚СЏ, РѕРЅР»Р°Р№РЅ-РµРІРѕР»СЋС†С–СЏ,
    СЃРµРјР°С„РѕСЂ РЅР°РґС–Р№РЅРѕСЃС‚С–) С‚Р° РїСЂР°РєС‚РёС‡РЅРѕСЋ РїСЂРѕРіСЂР°РјРЅРѕСЋ СЂРµР°Р»С–Р·Р°С†С–С”СЋ Impulse Activation Routing.



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

Р¦С– С„Р°Р№Р»Рё РЅРµ С” active MVP2.5 source of truth, Р°Р»Рµ РІР°Р¶Р»РёРІС– РґР»СЏ backward
compatibility, regression reasoning С– СЂРѕР·СѓРјС–РЅРЅСЏ РµРІРѕР»СЋС†С–С—.

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

Р—Р°СЂР°Р· roadmap С‡Р°СЃС‚РєРѕРІРѕ Р¶РёРІРµ РІ `ARCHITECTURE_GUIDELINES.MD` СЏРє high-level MVP
РїРѕСЃР»С–РґРѕРІРЅС–СЃС‚СЊ, Р° target state вЂ” РІ `TARGET_ARCHITECTURE.MD`. Р¦Рµ РЅРѕСЂРјР°Р»СЊРЅРѕ РґР»СЏ
СЃС‚Р°Р±С–Р»СЊРЅРѕС— СЃС‚СЂР°С‚РµРіС–С—. РћРєСЂРµРјРёР№ `docs/current/roadmap.md` РїРѕС‚СЂС–Р±РµРЅ Р»РёС€Рµ С‚РѕРґС–, РєРѕР»Рё
С‚СЂРµР±Р° СЂРµРіСѓР»СЏСЂРЅРѕ РѕРЅРѕРІР»СЋРІР°С‚Рё РЅР°Р№Р±Р»РёР¶С‡С– milestones, СЃС‚Р°С‚СѓСЃРё Р±Р»РѕРєРµСЂС–РІ, РїРѕСЂСЏРґРѕРє
Р·Р°РєСЂРёС‚С‚СЏ debt С– РєСЂРёС‚РµСЂС–С— РїРµСЂРµС…РѕРґСѓ РґРѕ MVP3.

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
