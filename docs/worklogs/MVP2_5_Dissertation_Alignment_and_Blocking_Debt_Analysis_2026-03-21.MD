# MVP2.5 Dissertation Alignment and Blocking Technical Debt Analysis

Date: 2026-03-21  
Status: Completed  
Scope: звірка з цілями дисертації + визначення блокерів перед валідними експериментами structure/drift і переходом до MVP3 Reliability Semaphore.

## 1. Вхідні джерела

1. `docs/Дисертація Коротенко.docx` (ключові формулювання мети/задач/семафора/OOD/Wasserstein).
2. `README.MD` (GAP 2.4.1..2.4.4, наукові принципи, експериментальна логіка).
3. `docs/EVF_MVP2.MD`, `docs/EVF_MVP2_5.MD` (метрики, drift protocol, anti-leakage).
4. `docs/ARCHITECTURE_MVP2_5.MD`, `docs/DATA_FLOWS_MVP2_5.MD` (архітектурні потоки, strict_asof).
5. `docs/worklogs/Finish_MVP2_5_Plan.MD` + Stage4.2 звіти (contract/quality gate/mapping/XES fix).
6. Поточні конфіги `configs/experiments/*` і `configs/data/*` для drift/stat сценаріїв.

## 2. Матриця відповідності цілям дисертації

## 2.1 Ціль A: Structural Awareness (інтеграція нормативної топології з емпіричними подіями)

Оцінка: `PARTIAL_READY`  
Покрито:
1. Репозиторний контур topology + snapshots реалізований.
2. `DynamicGraphBuilder` будує структурні тензори та `allowed_target_mask`.
3. Config-driven stats mapping працює.

Гап:
1. Немає жорсткої гарантії, що `activity_id` з логу коректно вирівняний з BPMN `node_id` для кожного датасету.
2. Залишається ризик “тихої” деградації структури при mismatch (історія з `use_classifier` у XES).

## 2.2 Ціль B: Стійкість до структурного дрейфу без leakage

Оцінка: `PARTIAL_READY`  
Покрито:
1. `strict_asof` семантика і snapshot policy.
2. XES-fix: auto `as_of` по `max(event_ts)` після train-cut + skip no-data snapshots.

Гап:
1. Для наукових прогонів ще не зафіксовано fail-fast policy, якщо в strict режимі немає придатного snapshot.
2. Немає фінального єдиного E2E acceptance пакета “aligned/non-aligned/strict_asof/zero-dominant”.

## 2.3 Ціль C: Епістемічний контроль (Reliability Semaphore, OOD/Wasserstein)

Оцінка: `NOT_READY`  
Покрито:
1. Є базова інфраструктура для quality diagnostics у snapshots.
2. Є OOS-метрики у MVP2 evaluation контурі.

Гап:
1. Немає production-модуля MVP3 Semaphore (Green/Yellow/Red policy).
2. Немає формального контуру калібрування порогів (bootstrap для `τ_W`, `τ_M`).
3. Немає стандартизованого latent-manifold artifact pipeline для OOD.

## 2.4 Ціль D: Відтворюваність і доказовість (Ablation/контроль експериментів)

Оцінка: `PARTIAL_READY`  
Покрито:
1. Конфіг-орієнтований запуск.
2. Stage4.2 статистичні конфіги і quality gate механізми.

Гап:
1. Потрібен єдиний “канонічний” експериментальний пакет для розділу 4 з фіксованим output-schema.
2. Потрібні обов’язкові критерії прийнятності якості snapshot перед train/eval.

## 3. Блокуючий техборг перед тестами впливу структури та дрейфу (P0)

Ці пункти потрібно закрити перед будь-якими фінальними висновками про вплив структури/дрейфу:

1. `P0-1` Activity-to-Topology Alignment Gate  
Потрібен автоматичний preflight-check:
   - `|activities ∩ node_id| / |activities|`,
   - явний поріг acceptance,
   - fail або quarantine при низькому overlap.

2. `P0-2` Strict-AsOf Fail-Fast Mode для research runs  
У strict експериментах недопустимий тихий fallback, якщо snapshot `<= as_of` не знайдено.

3. `P0-3` Snapshot-Homogeneous Batching  
Поточна Option-A (first-graph structural payload) підходить як тимчасовий runtime workaround, але недостатня для доказових drift експериментів.
Потрібно:
   - або batch bucketing за snapshot key,
   - або micro-batch by snapshot.

4. `P0-4` Locked Quality-Gate Policy для експериментів  
Зараз у частині конфігів `on_fail=allow_with_warning`.
Для доказових прогонів має бути жорстка policy:
   - `ignore_with_warning` або `raise` (залежно від етапу),
   - зафіксовані thresholds у плані експерименту.

5. `P0-5` E2E Test Gate “Structure+Time Integrity”  
Обов’язковий набір:
   - aligned vs non-aligned mapping,
   - strict_asof anti-leakage,
   - zero-dominant fallback behavior,
   - parity file/neo4j backend semantics.

6. `P0-6` Canonical Experiment Output Contract  
Перед запуском серії потрібно зафіксувати обов’язкові поля артефактів:
   - `F1`, `ECE`, `OOS`,
   - drift-window timestamps,
   - `knowledge_version`, effective `as_of`,
   - stats quality fields.

## 4. Блокери перед стартом MVP3 Reliability Semaphore

1. `MVP3-B1` Немає калібрувального протоколу OOD-порогів (`τ_W`, `τ_M`) на історично стабільному періоді.
2. `MVP3-B2` Немає стандартного pipeline збереження latent-representations для window-wise Wasserstein аналізу.
3. `MVP3-B3` Немає формалізованої policy-мапи `Green/Yellow/Red -> runtime action`.
4. `MVP3-B4` Немає окремого evaluation-пакету для помилок I/II роду Semaphore (false alarm vs missed drift).
5. `MVP3-B5` Не зафіксовано Human-in-the-loop інтерфейс/протокол для стану `Red`.

## 5. Рекомендований формат демонстрації результатів (розділ 4)

## 5.1 Закриті корпоративні дані (50+ версій, 3+ роки)

Ціль: показати реальну цінність у production-like умовах.

Рекомендований формат:
1. Публікувати тільки агреговані та анонімізовані метрики (без сирих подій).
2. Основні графіки:
   - `F1/ECE/OOS` по timeline версій,
   - drift heatmap по версіях і часових вікнах,
   - snapshot quality trend (`non_zero_ratio`, `coverage`).
3. Для кожного графіка фіксувати `knowledge_version` і effective `as_of`.

## 5.2 Відкриті дані (`data.4tu.nl`)

### Контрольний датасет (без вираженого drift)
1. `BPI_Challenge_2012` як baseline/control сценарій.

### Drift-сценарії (Loan Assessment / Business Process Drift)
1. Кожен drift-type трактувати як окремий сценарій експерименту, а не змішувати в одну timeline.
2. Робити нормалізацію часу до `relative phase`:
   - `pre-drift`,
   - `drift-onset`,
   - `post-drift`.
3. Показувати порівняння між drift-type сценаріями в єдиній панелі метрик.

Це прямо підходить як bridge до MVP3 Semaphore:
1. контрольні сценарії для false alarm,
2. drift-сценарії для sensitivity/latency detection.

## 6. Мінімальний порядок дій перед наступною фазою

1. Закрити `P0-1..P0-6` (без цього не стартувати фінальні висновки по впливу структури/дрейфу).
2. Оновити `Finish_MVP2_5_Plan.MD` до фактичного closure-стану + чітких acceptance gates.
3. Після цього запускати серію розділу 4:
   - корпоративний анонімізований трек,
   - open-data drift suite трек.
4. Лише після стабільного проходження серії переходити до MVP3 Semaphore (`MVP3-B1..B5`).
