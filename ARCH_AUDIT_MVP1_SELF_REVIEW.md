# Архітектурний аудит MVP1 (Self-Review)

## Scope
- Переглянуті зміни останніх комітів з фокусом на `ModelTrainer.run()`, `XESAdapter`, `FeatureEncoder`, `FeatureConfig`.
- База оцінки: правила шарів/залежностей (`ARCHITECTURE_RULES.md`), контракти потоків (`DATA_FLOWS_MVP1.MD`), цільова архітектура (`TARGET_ARCHITECTURE.MD`).

---

## Крок 1. Аудит поточного стану

### 1) Порушення контрактів / DDD / leakage

1. **`source_key` не порушує межі шарів і не тягне інфраструктуру в Domain.**
   - `FeatureConfig` залишився чистим DTO конфігурації (додано лише поле псевдоніма джерела), без залежностей на адаптери.
   - `FeatureEncoder` працює з доменними DTO (`RawTrace` / `event.extra`) і конфігом, а не з XES/XML API.
   - Висновок: **Clean/Hexagonal dependency direction не зламано**.

2. **Логіка мапінгу фактично розподілена між Adapter і Domain Service.**
   - `XESAdapter` використовує `source_key` для typed-extraction.
   - `FeatureEncoder` паралельно також підтримує fallback `name -> source_key`.
   - Це не порушує імпорти/залежності, але створює **дублювання schema-resolution логіки** у двох місцях pipeline.
   - Висновок: формально контракт не зламано, але є **архітектурна ерозія SRP** (одна відповідальність «канонічне резольвлення поля» розмазана).

3. **Обробка drift windows у `Trainer` знаходиться в Application (допустимо), але `Trainer` став багаторежимним orchestration-god-object.**
   - `run()` одночасно оркеструє train / eval_cross / eval_drift + checkpoint policy.
   - Це не leakage у Data Prep/Domain, але є **змішування кількох use-case сценаріїв в одному класі**, що підвищує coupling.

4. **Інференс-логіка в data-preparation не протекла.**
   - Drift window evaluation викликає `_build_loader` + `_evaluate_test` у eval-режимі; трансформація даних лишилась у тих самих pipeline-кроках.
   - Прямого змішування online inference policy з ingestion/preprocessing не виявлено.

### 2) Чистота ядра (hardcoded / dataset-specific)

1. **Є жорсткі дефолти, що прив’язані до XES-екосистеми, але не до конкретного датасету.**
   - Ключі типу `concept:name`, `time:timestamp`, `org:resource`, `lifecycle:transition`.
   - Default `complete_transitions` містить фіксований набір значень.
   - Це прийнятно для XESAdapter-контракту MVP1, але потребує винесення у schema policy для масштабування.

2. **Є hardcoded behavioral assumptions у FeatureEncoder.**
   - fallback activity feature name: `concept:name`;
   - z-score clipping до `[-3, 3]`;
   - `time2vec_work_hours` жорстко закодовано як Mon-Fri 09:00–18:00 UTC.
   - Це не dataset-specific у вузькому сенсі, але це **неявна доменна політика у коді**, а не в конфігурації.

3. **Drift window mechanics у Trainer зараз константно-лінійна (fixed non-overlap windows).**
   - Розмір вікна параметризований, але stride/policy не параметризовані.
   - Для дисертаційної відтворюваності це ок у MVP1, але обмежує експерименти для MVP2.

### 3) Відповідність Target Architecture (шлях до EOPKG / Dynamic Structural Layer)

1. **`source_key`-аліасинг — правильний крок до schema abstraction.**
   - Дозволяє відв’язати внутрішні імена фіч від зовнішніх ключів джерела.
   - Це корисний фундамент для multi-source ingest і подальших структурних шарів.

2. **Поточна реалізація — проміжна, не фінальна архітектура для MVP2.**
   - Через дублювання alias-resolution в Adapter + Encoder не сформовано єдиний canonical schema resolver.
   - Без цього Dynamic Structural Layer ризикує отримати ще більше умовних `if alias` по коду.

3. **Drift evaluation у Trainer не тупик, але потрібна декомпозиція на окремі use cases.**
   - Для Target Architecture (окремий research pipeline evaluation) поточний монолітний `Trainer` треба розділяти.

---

## Крок 2. Technical Debt та рекомендації (без виправлень у цьому кроці)

### TD-1. Розмазана відповідальність за schema/alias resolution
- **Симптом:** `source_key` резольвиться і в `XESAdapter`, і в `FeatureEncoder`.
- **Ризик:** неузгоджені правила мапінгу між ingestion та encoding, latent bugs при додаванні нових джерел.
- **Рекомендація:**
  1. Ввести окремий `SchemaResolver` (Domain service або Application-level policy) з єдиним контрактом `resolve(feature_config, payload) -> value`.
  2. Adapter має формувати канонічний payload один раз; Encoder працює тільки з канонічними ключами без fallback.
  3. Додати contract tests на еквівалентність резольвлення для event/trace features.

### TD-2. `ModelTrainer` перевантажений кількома сценаріями
- **Симптом:** один клас керує train, eval_cross_dataset, eval_drift, checkpoint restore policy.
- **Ризик:** збільшення когнітивної складності, складність еволюції під MVP2 (Research pipeline vs Runtime pipeline).
- **Рекомендація:**
  1. Розділити use cases: `TrainUseCase`, `CrossDatasetEvalUseCase`, `DriftEvalUseCase`.
  2. Винести спільні сервіси (`CheckpointService`, `DataLoaderFactory`, `MetricsEvaluator`) в Application services.
  3. Зберегти один вхідний фасад лише як thin orchestrator.

### TD-3. Неявні domain policies захардкожені в коді
- **Симптом:** z-score clipping, work-hours UTC, lifecycle complete defaults у коді.
- **Ризик:** слабка експериментальна керованість, складність відтворення різних протоколів у дисертації.
- **Рекомендація:**
  1. Перенести ці політики в конфіг (`mapping`/`training`/`evaluation`) з явними default profile.
  2. Додати versioned policy profiles (`mvp1_default`, `bpi2017_profile`, `enterprise_profile`).

### TD-4. Drift window policy недостатньо формалізована
- **Симптом:** тільки fixed non-overlap windows, без stride/overlap/event-time policy.
- **Ризик:** обмеження дослідницьких сценаріїв та порівняльності дрейф-експериментів.
- **Рекомендація:**
  1. Ввести `IDriftWindowPolicy` (fixed / sliding / expanding, trace-based або time-based).
  2. Логувати policy metadata у tracker як обов’язкові артефакти експерименту.

### TD-5. Підготовка до EOPKG/Dynamic Structural Layer поки неповна
- **Симптом:** aliasing є, але немає централізованого semantic layer між raw schema та model features.
- **Ризик:** при інтеграції EOPKG зросте coupling між ingestion, feature engineering і graph building.
- **Рекомендація:**
  1. Спроєктувати `SemanticFeatureContract` (канонічні semantic slots + provenance).
  2. Відокремити `StructuralContextProvider` (майбутній EOPKG-injection hook) від поточного `GraphBuilder`.
  3. Закласти extension points без додавання MVP2-функцій у runtime MVP1.

---

## Короткий висновок
- Критичних порушень Clean/Hexagonal/DDD-контрактів у розглянутих змінах **не виявлено**.
- Виявлено **5 зон технічного боргу**, головна з них — відсутність єдиного canonical schema resolver та перевантаження `Trainer`.
- Поточні рішення не є тупиком, але потребують цілеспрямованого рефакторингу перед MVP2, щоб уникнути каскадного росту умовної логіки.
