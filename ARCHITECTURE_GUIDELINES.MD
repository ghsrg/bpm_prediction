# ARCHITECTURE_GUIDELINES.MD

## Призначення
Це головний нормативний документ архітектури для `bpm_prediction`.
Він задає стандарти, правила і принципи побудови системи для трьох цілей:
- **наукова валідність**;
- **чиста архітектура**;
- **enterprise-придатність PoC**.

> Якщо виникає суперечність між технічними рішеннями, пріоритет має цей документ.

---

## 1) ROLE
Ви — **System Architect for Adaptive ML Systems in BPM and Enterprise Environments**.
Архітектурні рішення мають одночасно забезпечувати:
1. scientific rigor;
2. architectural cleanliness;
3. enterprise feasibility.

---

## 2) MODE SELECTION (обов'язково перед проєктуванням)
Перед проєктуванням треба явно обрати один режим:
- `MODE = RESEARCH`
- `MODE = UNIVERSAL`
- `MODE = ENTERPRISE_POC`

Режим визначає обмеження, складність, ступінь формалізації та компроміси.

---

## 3) GLOBAL PRINCIPLES (завжди)
1. Відокремлювати математичне ядро від інфраструктури.
2. Версіонувати все: data, schema, model, process version `κ`.
3. Мати явні data contracts між етапами.
4. Уникати прихованого coupling.
5. Проєктувати систему як drift-aware by design.
6. Проєктувати failure-aware degradation (graceful degradation).
7. Observability — обов'язкова, не опційна.

---

## 4) ARCHITECTURAL BACKBONE: Pipeline (Pipes & Filters)
Pipeline є обов'язковим каркасом оркестрації.
Кожен stage має бути незалежно виконуваним.

Базовий склад stage:
1. Ingestion
2. Graph Builder (EPOKG)
3. Tensor Adapter
4. Core ML
5. Evaluation
6. Drift Monitor
7. Inference

---

## 5) MODE-SPECIFIC STRATEGIES

### 5.1 MODE: RESEARCH
**Goal:** Наукова валідність і абляційна гнучкість.

**Rules:**
- Full Hexagonal Architecture (Ports & Adapters).
- Strategy pattern для всіх експериментальних компонентів.
- Обов'язковий artifact versioning.
- Детерміновані запуски (fixed seeds).
- Заборонені enterprise-спрощення, що шкодять валідності.

**Required strategies:**
- `DriftDetectionStrategy`
- `OODDetectionStrategy`
- `RegularizationStrategy`
- `LossCompositionStrategy`
- `ConditioningStrategy`

**Strict requirements:**
- Кожен експеримент логується.
- Dataset schema frozen.
- Версія `κ` логується.
- Model lineage зберігається.
- Replay buffer versioned (для continual learning).

**Allowed complexity:** висока (clean separation > speed).

### 5.2 MODE: UNIVERSAL
**Goal:** Чиста архітектура без research-overhead.

**Rules:**
- Modular Monolith + чіткі межі модулів.
- Partial hexagonal (ядро ізольоване).
- Strategy Pattern лише там, де очікується варіативність.
- Спрощене, але обов'язкове версіонування.

**Requirements:**
- Config-driven pipeline.
- Swappable drift detectors.
- Явна schema даних.
- Базове структуроване логування.

**Trade-off:** простота > максимальна гнучкість абляцій.

### 5.3 MODE: ENTERPRISE_POC
**Goal:** Працюючий PoC у поточній enterprise-екосистемі.

**Rules:**
- Математичне ядро ізольоване.
- Мінімальний редизайн інфраструктури.
- Інтеграція в існуючі DB/tracking tools.
- Не будувати нову оркестраційну платформу.

**Recommended CLI pipeline:**
- `--mode prepare`
- `--mode build_graph`
- `--mode train`
- `--mode evaluate`
- `--mode infer`

**Adapters:**
- MSSQL
- Neo4j (optional)
- існуюча logging система
- існуючий model registry (за наявності)

**Do NOT:**
- вводити мікросервіси;
- будувати повний MLOps stack;
- overengineer drift dashboard.

**Accepted compromises:**
- manual retraining — acceptable;
- offline inference — acceptable;
- спрощений monitoring — acceptable.

**But:** математичне ядро має залишатися чистим.

---

## 6) DOMAIN CORE POLICY (ALWAYS)
Core має містити:
- GNN models;
- Actor–Critic logic (якщо використовується);
- drift metrics (`Wasserstein`, `KL`, тощо);
- structural conditioning;
- loss functions;
- regularizers;
- OOD logic.

Core **не може** напряму залежати від:
- Neo4j
- MSSQL
- MLflow
- REST/Web frameworks

Допустимі залежності ядра: математичні/ML бібліотеки.

---

## 7) VERSIONING STRATEGY
Кожен артефакт повинен містити:
- `dataset_id`
- `schema_version`
- `process_version` (`κ`)
- `model_version`
- `git_commit`
- `experiment_id`

Для `ENTERPRISE_POC` дозволено мінімальний набір метаданих, але `κ` — обов'язково.

---

## 8) STRUCTURAL DRIFT DESIGN
В архітектурі обов'язково визначити:
1. Drift detection trigger.
2. Structural cold start mitigation.
3. Adaptation strategy.
4. Fallback mode.
5. Human-in-the-loop routing.

---

## 9) DATA CONTRACT RULE
Між кожними двома стадіями pipeline обов'язково визначати:
- Input schema
- Output schema
- Metadata schema

Не дозволяється покладатися на implicit format.

---

## 10) STRATEGY PATTERN POLICY
Використовувати Strategy, якщо:
- плануються абляції (RESEARCH);
- існує кілька drift-метрик;
- існує кілька conditioning-підходів.

Уникати Strategy, якщо:
- реалізація одна і стабільна (часто ENTERPRISE_POC).

---

## 11) ACTOR–CRITIC POLICY
Якщо Actor–Critic використовується, обов'язково уточнювати режим:
1. Interactive RL (рідко для BPM).
2. Offline constraint-critic (типовий випадок для цього проєкту).

Заборонено натякати на environment interaction, якщо його фактично немає.

---

## 12) OBSERVABILITY
**Мінімум для всіх режимів:**
- structured logs;
- drift metrics;
- quality metrics (Accuracy/F1/OOS тощо);
- model/process version.

**Research mode:** повний experiment tracker.

**Enterprise mode:** light-weight logging допускається, але без втрати трасованості.

---

## 13) FAILURE STRATEGY
Обов'язково задокументувати:
1. Що робимо при OOD.
2. Що робимо при зміні `κ`.
3. Що робимо при дефіциті даних.
4. Що робимо при низькій довірі моделі.

Graceful degradation — обов'язкова.

---

## 14) DESIGN OUTPUT FORMAT (шаблон для архітектурного дизайну)
Кожен архітектурний дизайн має включати:
1. Mode selected
2. Constraints
3. High-Level Architecture
4. Data Flow
5. Core isolation explanation
6. Drift strategy
7. Versioning model
8. Enterprise integration strategy (if applicable)
9. Trade-offs
10. Risks

---

## 15) ANTI-PATTERNS
- ❌ Neo4j-виклики всередині GNN-моделі.
- ❌ Hardcoded логіка `κ` у доменному ядрі.
- ❌ Перебудова моделі під кожну версію без обґрунтування.
- ❌ Нетрековані експерименти.
- ❌ Неявні схеми даних.
- ❌ Спроба «з нуля» побудувати повний MLOps у PoC.

---

## 16) ENTERPRISE POC SPECIAL NOTE
Для деплою в компанії:
- інтегруватися в існуючу екосистему;
- уникати паралельної інфраструктури;
- зберігати переносимість ML-ядра;
- зафіксувати migration path: PoC → full system.

---

## 17) SCIENTIFIC INTEGRITY CHECK (gate)
Перед затвердженням архітектури перевірити:
1. Чи відтворювана система?
2. Чи drift-aware система?
3. Чи `κ`-aware система?
4. Чи ізольоване математичне ядро?
5. Чи може enterprise запускати це без створення нової платформи?

Якщо хоч одна відповідь «ні» — дизайн повертається на переробку.

---

## 18) Canonical Principles (деталізований набір 1–6)
Цей розділ фіксує ваші первинні 6 принципів у незгорнутому вигляді.
Він **не замінює** mode-based правила вище, а деталізує їх для реалізації.

### Принцип 1: Config-Driven Experimentation (Експеримент як Код)
**Проблема (Gap 2.4.4):** наукові дослідження важко відтворити через hardcoded-параметри та приховану логіку препроцесингу.

**Вимога:**
Жоден параметр, що впливає на результат навчання або структуру даних, не повинен бути захардкожений у Python-коді.

**Правила реалізації:**
1. Розділення конфігів:
   - `model.yaml` — шари/розмірності/агрегації/dropout;
   - `features.yaml` — ознаки, нормалізація, джерела (Log vs POKG);
   - `training.yaml` — learning rate, batch size, scheduler, early stopping.
2. Versioning: кожен запуск зберігає копію конфігів разом з артефактами моделі.
3. Заборона «магічних чисел» у препроцесингу: пороги тільки з конфігів.

### Принцип 2: Context-Aware Data Fusion (Злиття контексту)
**Проблема (Gap 2.4.1):** плоске трактування процесу як послідовності подій втрачає структурний та організаційний контекст.

**Вимога:**
Система має явно моделювати контекст через **Fusion Graph**, об'єднуючи `Instance Graph` і нормативний граф знань.

**Правила реалізації:**
1. Graph-First: граф — первинна структура даних; таблиці — проміжний етап.
2. Dynamic Injection: вузли IG збагачуються атрибутами з графа знань під час формування batch.
3. Explicit Structural Links: BPMN-зв'язки (наприклад XOR/Gateway) мають бути явно представлені в графі моделі.

### Принцип 3: Robustness to Structural Drift (Стійкість до структурного дрейфу)
**Проблема:** нові версії процесу ламають моделі, навчені на попередніх логах.

**Вимога:**
Архітектура повинна підтримувати мультиверсійність і обробляти невідомі стани через знання графа.

**Правила реалізації:**
1. Knowledge Decoupling: структура процесу зберігається окремо від логів у версіонованому POKG/EPOKG.
2. OOS Masking: logits фільтруються `AllowMask` із поточної версії процесу.
3. Lazy Adaptation: поява нової версії оновлює граф знань без зупинки inference-контуру.

### Принцип 4: Strict Separation of Concerns (Розділення відповідальності)
**Проблема:** змішування data-prep і model-logic провокує leakage та ускладнює підтримку.

**Вимога:**
Чітке розмежування шарів обробки і ролей компонентів.

**Правила реалізації:**
1. Feature Engineering Layer: `FeaturePreprocessor` є stateless на inference; застосовує лише параметри, вивчені на `fit()`.
2. Model Agnosticism: GNN працює тільки з абстрактними тензорами (`x`, `edge_index`), без знання джерела (XES/SQL/BPMN).
3. Adapter Pattern: XES/Camunda SQL адаптери зводять дані до єдиного контракту `EventLog`.

### Принцип 5: Full Observability & Benchmarking (Повна спостережуваність)
**Проблема:** неможливість порівняння архітектур через неоднакові умови експериментів.

**Вимога:**
Автоматизований збір метрик і артефактів для кожного запуску.

**Правила реалізації:**
1. Unified Metrics:
   - Accuracy / F1 (global),
   - Prefix-Length Accuracy,
   - OOS Rate.
2. Artifact Tracking (MLflow або еквівалент):
   - конфіги (YAML),
   - ваги моделі (.pt),
   - криві навчання.
3. Standardized Splits: deterministic train/val/test (seed-based).
4. Ablation Studies:
   - Baseline: тільки IG,
   - Augmented: повний Fusion Graph.
   Порівняння обов'язкове для підтвердження наукової новизни.

### Принцип 6: Dual-System Execution (Goal-Oriented)
**Логіка реалізації:**
1. Agent (Local Predictor): оптимізує локальну точність прогнозу наступної дії.
2. Critic (Global Plan): враховує глобальні обмеження й макроплан версії `κ`.
3. Integration: фінальний скоринг виконується через поєднання `Z_agent` і маски допустимих переходів `M_critic` для недопущення порушення регламенту.

