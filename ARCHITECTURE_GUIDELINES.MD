# ARCHITECTURE_GUIDELINES.MD

## 1. Purpose (Призначення)

This document defines the stable architectural constitution of `bpm_prediction`.

(Цей документ є "конституцією" архітектури. Він фіксує незмінні принципи побудови системи.)

It separates:
- Immutable architectural principles
- Experiment configuration space
- Incremental MVP realization stages
- Enterprise deployment boundary

(Документ розділяє:
- стабільні принципи,
- простір експериментальних конфігурацій,
- етапи інкрементальної реалізації,
- межу між науковою системою та enterprise-обгорткою.)

If a design conflicts with this document — this document prevails.

(Якщо конкретна реалізація суперечить цьому документу — пріоритет має цей документ.)

---

## 2. Global Modes (Глобальні режими)

The system operates in exactly two top-level modes:

(Система має лише два верхньорівневі режими роботи.)

---

### 2.1 MODE = EXPERIMENT

Scientific research mode.

(Режим для наукових досліджень і перевірки гіпотез дисертації.)

Used for validating hypotheses from the dissertation.

(Використовується для перевірки гіпотез щодо структурного контексту, κ-conditioning, дрейфу, критика тощо.)

Configured via:

researchMode = {
    structure_mode: Literal["logs_only", "bpmn", "epokg"],
    kappa_conditioning: Literal["none", "static", "latent"],
    drift_strategy: Literal["off", "data", "structural", "wasserstein", "multi"],
    reliability_strategy: Literal["off", "threshold", "adaptive"],
    architecture_type: Literal["single_gnn", "agent_critic", "ablation"]
}

(Режим експерименту керується через явну конфігурацію.
Кожен параметр відповідає окремій науковій осі дослідження.)

EXPERIMENT mode must:
- be reproducible,
- be versioned,
- be config-driven,
- log κ explicitly,
- isolate mathematical core.

(Обов'язкові вимоги:
- повна відтворюваність,
- версіонування артефактів,
- конфігураційне керування,
- явна фіксація κ,
- ізоляція математичного ядра від інфраструктури.)

---

### 2.2 MODE = ENTERPRISE

Operational wrapper mode.

(Режим для розгортання у компанії.)

Constraints:
- mathematical core must remain unchanged,
- no research-overhead abstractions,
- integrate into existing ecosystem,
- API/CLI allowed,
- no infrastructure rewrite.

(Обмеження:
- ядро ML не переписується,
- не вводимо зайву дослідницьку складність,
- інтегруємось у наявну інфраструктуру,
- дозволені API/CLI,
- не будуємо новий MLOps стек.)

Enterprise does NOT redefine scientific core.

(Enterprise не змінює математичну модель — лише обгортає її.)

---

## 3. Architectural Layers (Незмінна багатошарова структура)

Layer separation is mandatory:

(Розділення шарів є обов’язковим.)

- Domain (`src/core/`)
- Application (`src/pipeline/`)
- Adapters (`src/adapters/`)
- Frameworks (external systems)

(Чітке розмежування:
- ядро логіки,
- orchestration,
- адаптери,
- зовнішні системи.)

Dependency inversion rules are enforced via ARCHITECTURE_RULES.md.

(Технічні обмеження залежностей перевіряються окремим файлом правил.)

Domain must not depend on:
- Neo4j
- MSSQL
- MLflow
- REST frameworks
- Concrete adapters

(Ядро не має імпортувати інфраструктурні технології.)

---

## 4. Fusion Model Definition (Модель злиття)

Fusion occurs at two levels:

(Злиття контексту відбувається на двох рівнях.)

### 4.1 Structural Fusion

InstanceGraph enriched by EPOKG during Graph Builder stage.

(На рівні графа IG збагачується нормативною структурою EPOKG.)

### 4.2 Latent Fusion

z_fused = Γ(h_σ, c_σ)

(На рівні латентного простору виконується злиття локального стану і глобального контексту.)

z_fused is the canonical fused representation.

(Єдиною канонічною fused-репрезентацією є z_fused.)

Γ must live inside Domain Core.

(Оператор Γ є частиною математичного ядра.)

---

## 5. Deterministic Masking vs Critic

Pipeline order:

Predictor → Logits → AllowMask → Critic → Final Decision

(Порядок обробки суворо фіксований.)

AllowMask:
- deterministic,
- κ-dependent,
- non-learnable constraint.

(AllowMask — це жорстке структурне обмеження, що залежить від версії процесу.)

Critic:
- optional,
- soft validator,
- may adjust scores but cannot violate structural mask.

(Critic — м’який валідатор, що не може порушувати структурні правила.)

---

## 6. Drift Responsibility Boundary (Межа відповідальності за дрейф)

Drift module:
- computes drift signals,
- computes reliability state,
- does NOT make business UI decisions,
- does NOT orchestrate retraining automatically.

(Модуль дрейфу лише обчислює сигнали.
Він не приймає бізнес-рішень і не керує інтерфейсом.)

BPMS system decides behavior.

(Рішення щодо UI або блокування приймає BPMS-система.)

---

## 7. Versioning Policy (Обов’язкове версіонування)

Every artifact must include:
- dataset_id
- schema_version
- process_version (κ)
- model_version
- git_commit
- experiment_id

(Кожен запуск і модель мають повний набір метаданих.)

EXPERIMENT mode requires full config snapshot.

(У режимі експерименту зберігаються всі конфіг-файли.)

---

## 8. Incremental MVP Stages (Етапи розвитку системи)

The architecture evolves via controlled stages.

(Розвиток системи відбувається поетапно.)

### MVP1 — Baseline Graph GNN
(Базова модель без складних механізмів.)

### MVP2 — κ + Structural Mask
(Додається підтримка версій і жорстка маска.)

### MVP3 — Critic
(Додається глобальний валідатор.)

### MVP4 — Drift & Reliability
(Додається дрейф і семафор надійності.)

### MVP5 — Enterprise Wrapper
(Додається API та інтеграція в BPMS.)

---

## 9. Document Versioning Policy

The following documents are versioned per MVP:

- ARCHITECTURE_MVPx.MD
- DATA_FLOWS_MVPx.MD
- DATA_MODELS_MVPx.MD

(Кожен етап має окремий опис архітектури та потоків даних.)

ARCHITECTURE_GUIDELINES remains stable.

(Цей документ залишається стабільним.)

---

## 10. Scientific Integrity Gate

Before moving to next MVP:

1. Is κ-aware?
2. Is reproducible?
3. Is core isolated?
4. Are contracts explicit?
5. Is drift boundary respected?

(Перед переходом до наступного етапу перевіряється наукова валідність.)

If any answer is "no" — redesign required.