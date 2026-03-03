# ADAPTER_XES.MD

## 1. Статус
`APPROVED` (Core Foundation - All MVPs)

## 2. Призначення
Специфікація адаптера (Ingestion Layer) для читання сирих журналів подій (XES/CSV/MXML) та їх конвертації у канонічні DTO (`RawTrace`, `EventRecord`) перед передачею в Domain Layer.

## 3. Мінімальні вимоги (Requirements)
1. **Підтримка форматів:** Читання `.xes` (нативно, потоково) та `.mxml` (через конвертацію `pm4py` під капотом).
2. **Динамічний мапінг:** Використання `mapping.yaml` для гнучкого мапінгу колонок.
3. **XES Fallbacks & Classifiers:**
   - Якщо `mapping.yaml` не задає ключі, адаптер використовує стандарт IEEE XES 2.0: `concept:name` (activity), `time:timestamp` (time), `org:resource` (resource), `lifecycle:transition` (lifecycle).
   - Підтримка злиття через `<classifier>` (наприклад, `activity_id = concept:name + lifecycle:transition`), що є стандартом для логів BPI Challenge.
4. **Версіонування (κ):** Пріоритет пошуку версії процесу: `event-level` -> `trace-level` -> `log-level` -> `filename` -> значення `"default"`.
5. **Ізоляція:** Адаптер не створює графи і не перетворює дані на тензори. Він видає виключно плоскі DTO.

6. **XES Fallbacks & Classifiers:**
   - Якщо `mapping.yaml` не задає ключі, адаптер використовує стандарт IEEE XES 2.0: `concept:name` (activity), `time:timestamp` (time), `org:resource` (resource), `lifecycle:transition` (lifecycle).
   - **Classifier handling:** Якщо в лозі є елемент `<classifier name="Activity">` (або інший), то `activity_id` береться з значення classifier (найчастіше це `concept:name + lifecycle:transition`, наприклад "Submit order+complete").  
     Якщо classifier відсутній — `activity_id = значення activity_key` (зазвичай `concept:name`), а `lifecycle` зберігається окремо.
   - Це забезпечує коректну роботу з логами BPI Challenge, IEEE Contest тощо.

## 4. Жорсткі інваріанти парсингу та розрахунків

Адаптер зобов'язаний виконати нормалізацію для кожної траси після її витягування. Усі часові дельти рахуються **строго в секундах (float)**.

### 4.1. Timezone та Час
Усі `timestamp` обов'язково парсяться або конвертуються в UTC (`datetime.fromisoformat(...).astimezone(UTC)`), а потім переводяться в Unix epoch (float секунди).

### 4.2. Алгоритм обробки та розрахунку
Для кожного `case_id` масив подій сортується за `timestamp` (час початку) за зростанням.
- `position_in_trace`: Дорівнює індексу події (від 0 до N-1).
- `duration`: 
  - Якщо є `lifecycle:complete` і відповідний `start`: `(END_TIME - START_TIME).total_seconds()`.
  - Якщо подія точкова (один timestamp): `0.0`.
  - Від'ємні значення блокуються (`max(0.0, duration)`).
- `time_since_case_start`: Для `i = 0` -> `0.0`. Для інших -> `(timestamp[i] - timestamp[0])`.
- `time_since_previous_event`: Для `i = 0` -> `0.0`. Для інших -> `max(0.0, timestamp[i] - timestamp[i-1])`.

### 4.2.1 Duration pairing rules (якщо lifecycle присутній)

Для розрахунку `duration` в `EventRecord` (тільки для завершальних подій):
- Якщо є атрибут `lifecycle:instance` або `concept:instance` — pairing виконується за комбінацією `activity_id + instance_id` (найточніший варіант).
- Якщо instance відсутній — pairing за `activity_id + resource_id` (якщо resource є).
- Якщо resource теж відсутній — pairing за `activity_id` (FIFO або LIFO — беремо останній непарений `start` перед `complete`).
- Правила пріоритету pairing (за замовчуванням):
  1. Останній `start` перед `complete` (LIFO — найпоширеніший у реальних логах).
  2. Якщо кілька `start` без `complete` — вони ігноруються.
  3. Якщо `complete` без відповідного `start` — `duration = 0.0`.
  4. Від'ємні значення блокуються: `duration = max(0.0, end_time - start_time)`.

Ці правила забезпечують сумісність з більшістю реальних логів (BPI Challenge, hospital, finance тощо).

### 4.3. Вирішення колізій та життєвих циклів (Event Deduplication)
- **Stable Sorting (Tie-breaker):** При однаковому `timestamp`, вторинним ключем сортування є оригінальний порядковий номер (index) у сирому файлі. Гарантує 100% детермінованість між запусками.
- **Lifecycle Filter:** Якщо є `lifecycle:transition`, для формування вузлів (`EventRecord`) залишаються **виключно** події зі статусом `complete`. Події `start` використовуються тільки "під капотом" для розрахунку `duration`.
- **Flat Nested Attributes:** Будь-які вкладені XML/JSON атрибути сплющуються (`parent.child = value`) або зберігаються як JSON-string у словниках `extra` / `trace_attributes`.
- **Lifecycle Filter:** Якщо в логах присутнє поле `lifecycle:transition`, до фінального списку подій (`EventRecord`) потрапляють **виключно** події з transition, що позначають завершення або скасування активності. За стандартом IEEE XES (стандартний transactional model) це такі значення:
  - `complete` — нормальне завершення активності (основний випадок).
  - `ate_abort` — аборт виконання конкретної активності.
  - `pi_abort` — аборт усього процесу для цього кейсу.
  - `manualskip` — ручне пропускання активності.
  - `autoskip` — автоматичне пропускання активності.

  Ці значення вважаються "завершальними" і генерують повноцінний `EventRecord`.  
  Інші transition (`start`, `schedule`, `assign`, `reassign`, `withdraw`, `suspend`, `resume` тощо) використовуються **виключно** "під капотом" для pairing та розрахунку `duration` і **не потрапляють** у фінальний список подій.

  Якщо потрібно — список "завершальних" transition можна налаштувати в `mapping.yaml` (наприклад: `complete_transitions: ["complete", "ate_abort", "manualskip", "autoskip"]`).

  Якщо поля `lifecycle:transition` взагалі немає — кожна подія вважається атомарною (`duration = 0.0`).

### 4.4. Обробка відсутніх даних
- Відсутній `resource_id`: `"UNKNOWN"`.
- Невалідний `timestamp`: подія відкидається (`logger.warning`), траса перераховується без неї.

## 5. Структура DTO Контрактів (Data Models)
Адаптер повертає дані строго у такій структурі:

```python
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

class EventRecord(BaseModel):
    activity_id: str
    timestamp: float                   # Unix epoch (UTC)
    resource_id: str                   # "UNKNOWN", якщо відсутній
    lifecycle: Optional[str]
    position_in_trace: int             # Від 0 до N-1
    duration: float                    # Секунди
    time_since_case_start: float       # Секунди
    time_since_previous_event: float   # Секунди
    extra: Dict[str, Any]              # Усі не-маплені атрибути події
    activity_instance_id: Optional[str]   # lifecycle:instance або concept:instance, якщо є


class RawTrace(BaseModel):
    case_id: str
    process_version: str               # κ
    events: List[EventRecord]
    trace_attributes: Dict[str, Any]   # Усі атрибути рівня <trace>
```



**6. Приклад конфігурації (mapping.yaml)**
```yaml
xes_adapter:
  case_id_key: "concept:name"
  activity_key: "concept:name"         # Або ["concept:name", "lifecycle:transition"]
  timestamp_key: "time:timestamp"
  resource_key: "org:resource"
  lifecycle_key: "lifecycle:transition"
  version_key: "concept:version"       
  extra_event_fields: "*"              # "*" = зберігати всі інші атрибути в extra
  extra_trace_fields: "*"              # "*" = зберігати всі інші атрибути в trace_attributes
  ## Додаткові опції для lifecycle та pairing
  complete_transitions: ["complete", "ate_abort", "pi_abort", "manualskip", "autoskip"]
  pairing_strategy: "lifo"          # або "fifo", "by_instance"
  use_classifier: true              # якщо true — шукати та використовувати <classifier> для activity_id


```

**7. Контракти виконання**

**Input:** file_path: str, mapping_config: dict

**Output:** Iterator[RawTrace] (Генератор для потокової обробки, дозволяє кешування на рівні Application Layer).

**Парсинг:** Обов'язкове використання потокового читання (наприклад, lxml.etree.iterparse або Rust-парсер). Заборонено завантажувати весь XML-документ у пам'ять (DOM).

**Errors & Logging:** - logger.error: невалідні XML/XES схеми.
logger.warning: пропущені биті події.
logger.info: після парсингу обов'язковий підсумковий запис: "Processed X traces, Y events, Z skipped".