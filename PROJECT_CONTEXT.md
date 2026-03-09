# PROJECT_CONTEXT.md

Оновлено: 2026-03-09  
Призначення: коротка, стабільна точка входу для AI-агентів (System Prompt Extension) у проєкті `bpm_prediction`.

---

## 1. Project Purpose

`bpm_prediction` — це система прогнозування наступної активності бізнес-процесу (next-activity prediction) на основі event logs.

Поточна ціль:
- **MVP1 baseline**: logs-only GNN pipeline.
- Без EPOKG/semantic enrichment/critic/reliability semaphore (це зона MVP2+).

Основний сценарій:
1. Прочитати XES-лог.
2. Нормалізувати його до канонічних DTO.
3. Згенерувати префікси трас.
4. Побудувати графові тензори.
5. Натренувати/оцінити baseline GNN (GCN або GATv2).

---

## 2. Architecture Overview

Проєкт дотримується Clean/Hexagonal розмежування шарів:

- `src/domain/`
  - Чиста доменна логіка, DTO, доменні сервіси.
  - Не залежить від `src/application`, `src/adapters`, `src/infrastructure`.

- `src/application/`
  - Use-case orchestration (trainer), порти (інтерфейси) для адаптерів.
  - Працює через абстракції/контракти.

- `src/adapters/`
  - Реалізації зовнішніх інтеграцій (наприклад, XES ingestion).

- `src/infrastructure/`
  - Технічні утиліти/інтеграції (конфіг, трекінг).

Ключовий сервіс узгодження схем:
- `SchemaResolver` (`src/domain/services/schema_resolver.py`)
  - У MVP1 це **identity resolver**: резолвить ключі, але не робить семантичної нормалізації значень.
  - Розширення до semantic mapping (синоніми, мовні варіанти) заплановано для MVP2.

---

## 3. Main Pipelines

### 3.1 Data Flow (MVP1)

`XESAdapter` -> `RawTrace` -> `PrefixPolicy` -> `PrefixSlice` -> `BaselineGraphBuilder` -> `GraphTensorContract` -> `ModelTrainer`

Деталізація:
1. `XESAdapter.read(...)` стрімінгово читає XES (`lxml.etree.iterparse`) та формує `RawTrace`.
2. `PrefixPolicy.generate_slices(...)` будує all-prefixes слайси (`prefix -> next event`).
3. `BaselineGraphBuilder.build_graph(...)` кодує події через `FeatureEncoder` і формує тензорний контракт.
4. `ModelTrainer` виконує split, train/val/test цикл або eval режими.

### 3.2 Runtime Modes

- `train`
  - Стандартне тренування з temporal split + early stopping + test evaluation.

- `eval_cross_dataset`
  - Оцінка на іншому датасеті з завантаженим checkpoint.
  - Енкодер (vocab/scalers) має бути зафіксований станом із checkpoint.

- `eval_drift`
  - Оцінка у ковзних часових вікнах для відстеження drift-метрик.

---

## 4. Key Data Structures

### `RawTrace`
Канонічна траса після ingestion:
- `case_id`
- `process_version`
- `events: List[EventRecord]`
- `trace_attributes`

### `EncodedNodeFeatures`
Результат кодування однієї події у `FeatureEncoder`:
- `cat_indices: List[int]` (категоріальні канали для embedding)
- `num_values: List[float]` (числові канали після трансформацій)

### `GraphTensorContract`
Вхідний контракт моделі у MVP1:
- `x_cat: torch.LongTensor`
- `x_num: torch.FloatTensor`
- `edge_index: torch.LongTensor`
- `edge_type: torch.LongTensor`
- `y: torch.LongTensor`
- `batch: torch.LongTensor`
- `num_nodes: int`

---

## 5. MVP1 Constraints (Critical)

### 5.1 Imputation / Encoding Policy

MVP1 використовує просту, стабільну політику:

- **Categorical features**
  - `<UNK>` індекс = `0`.
  - Відсутні/невідомі значення мапляться у `<UNK>`.

- **Numeric features**
  - Для `z-score` каналів відсутнє значення імпутується середнім `mu`, після чого `z = 0`.
  - Тобто в тензорі модель бачить **нульовий normalized сигнал** (zero-imputation у normalized space).

- **Schema resolution**
  - **Identity Schema Resolution**: жодного semantic synonym mapping у MVP1.
  - Немає нормалізації типу `"Start" ~ "Початок" ~ "Begin"` — це **MVP2 scope**.

### 5.2 Scope Guardrails

У MVP1 заборонено:
- EPOKG/структурні маски/critic/reliability semaphore/continual learning.
- Варіанти, що ламають базовий logs-only baseline.

---

## 6. Document Map (Source of Truth)

Після реорганізації:
- Корінь:
  - `AGENT_GUIDE.MD` — правила поведінки агента.
  - `PROJECT_CONTEXT.md` — короткий runtime/context summary.
  - `README.MD` — проєктний опис для людей.

- `docs/`:
  - `ARCHITECTURE_RULES.md` — жорсткі правила залежностей і заборон.
  - `ARCHITECTURE_GUIDELINES.MD` — загальні архітектурні принципи.
  - `DATA_MODEL_MVP1.MD` — DTO/контракти даних.
  - `DATA_FLOWS_MVP1.MD` — потоки і порти.
  - `ARCHITECTURE_MVP1.MD`, `LLD_MVP1.MD`, `EVF_MVP1.MD`, `VARIABLES.MD`, `GLOSSARY.MD` — деталізація реалізації та evaluation стандарту.

Правило пріоритету при конфліктах:
1. `AGENT_GUIDE.MD`
2. `docs/ARCHITECTURE_GUIDELINES.MD`
3. `docs/ARCHITECTURE_RULES.md`
4. далі інші документи згідно AGENT_GUIDE.

---

## 7. Etalon Status

Папка `etalon/` вважається **deprecated** для поточної лінії MVP1.

Політика для агентів:
- Не використовувати `etalon/` як джерело архітектурних рішень.
- Не будувати новий код на основі `etalon/`.
- Дозволено лише точкове звернення для історичного контексту, якщо явно попросив розробник.

Рекомендація:
- На цьому етапі **краще не видаляти одразу** з репозиторію, а:
  1. Позначити як архів/legacy.
  2. Прибрати з активних правил агента.
  3. Видалити після 1-2 стабільних релізів або після міграції потрібних артефактів.

---

## 8. Operational Notes for Future Agents

- Перед суттєвими змінами перевіряти:
  - `AGENT_GUIDE.MD`
  - `docs/ARCHITECTURE_RULES.md`
  - `docs/DATA_MODEL_MVP1.MD`
  - `docs/DATA_FLOWS_MVP1.MD`

- Для архітектурної валідації запускати:
  - `py -3 tools/architecture_guard.py`

- При додаванні нових бібліотек:
  - Оновити `requirements.txt`.
  - Відобразити зміни у `AGENT_GUIDE.MD` (стек дозволених залежностей).

- Не змінювати контракти DTO без синхронного оновлення:
  - коду,
  - `AGENT_GUIDE.MD`,
  - `docs/DATA_MODEL_MVP1.MD`.

---

## 9. Near-Term Roadmap Hints (MVP1 -> MVP2)

Потенційні безпечні кроки розвитку:
- Винести/зафіксувати formal feature profiles.
- Додати тест-контури на стабільність vocabulary + encoder state restore.
- Підготувати extension point для semantic mapping у `SchemaResolver` (без активації в MVP1).
- Формалізувати drift-eval протокол і пороги інтерпретації метрик.

Цей файл має залишатися коротким, технічно точним і синхронним з реальним кодом.
