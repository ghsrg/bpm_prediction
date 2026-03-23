# AGENT_GUIDE.md
# Інструкція для LLM-агента / код-генератора 

Це **єдиний документ**, який ти повинен читати перед тим, як генерувати будь-який код, пропонувати зміни чи планувати рефакторинг у проєкті `bpm_prediction`.

Всі інші `.md` файли — це джерело правди. Ти **не можеш** їх ігнорувати, переписувати чи інтерпретувати по-своєму.

## 1. Порядок пріоритетності документів
Читай і виконуй у такому порядку (від найвищого пріоритету до найнижчого):

1. **AGENT_GUIDE.MD** — цей файл (ти зараз його читаєш).
2. **PROJECT_CONTEXT.MD** — короткий актуальний контекст проекту.
3. **docs/ARCHITECTURE_GUIDELINES.MD** — конституція архітектури, незмінні правила.
4. **docs/ARCHITECTURE_RULES.MD** — жорсткі правила залежностей і заборон.
5. **docs/TARGET_ARCHITECTURE.MD** — цільова архітектура (поточний target blueprint).
6. **docs/ARCHITECTURE_MVP2_5.MD** — поточна активна архітектурна фаза (MVP2.5).
7. **docs/DATA_MODEL_MVP2_5.MD** — актуальні DTO/контракти даних для MVP2.5.
8. **docs/DATA_FLOWS_MVP2_5.MD** — актуальні потоки даних для MVP2.5.
9. **docs/LLD_MVP2_5.MD** — low-level дизайн поточного етапу.
10. **docs/EVF_MVP2_5.MD** — рамка валідації для MVP2.5.
11. **docs/ADAPTER_CAMUNDA_SQL.MD** — специфіка Camunda SQL/BPMN ingestion.
12. **docs/ADAPTER_XES.MD** — специфіка XES ingestion.
13. **docs/VARIABLES.MD** — математичні позначення та snake_case мапінг.
14. **docs/GLOSSARY.MD** — канон термінів.
15. **MVP1/MVP2 docs** (ARCHITECTURE_MVP1/2, DATA_MODEL_MVP1/2, DATA_FLOWS_MVP1/2, LLD_MVP1/2, EVF_MVP1/2) — використовуй тільки для backward compatibility або історичного контексту.

**Правило вирішення конфліктів:** Якщо в цих файлах є суперечність — пріоритет завжди у того документа, що стоїть **вище у цьому списку** (ближче до пункту 1). Є файл Дисертація Коротенко.docx, це теоретичний канон. 
Але якщо суперечності критичні — зупинись і задай питання розробнику!
Якщо виявлено суперечність, або тобі незрозуміло, як саме реалізувати конкретний метод — ЗУПИНИСЬ. Ти не маєш права фантазувати. Задай питання розробнику.

**Статус legacy-матеріалів:** папка `etalon/` є deprecated і не використовується як джерело архітектурних рішень у MVP1.

## 2. Жорсткі заборони (ніколи не порушуй)
- **Архітектура:** Не пиши код, який порушує Clean/Hexagonal Architecture. Domain не імпортує адаптери, Application залежить тільки від абстракцій (портів).
- **Scope MVP1:** Не додавай `EPOKG`, структурні маски, `Critic`, `Reliability Semaphore` або механізми `Continual Learning` у MVP1 — це функціонал MVP2+. Зараз будуємо чистий Baseline GNN.
- **Модель:** Не створюй окрему модель на кожну версію процесу (κ). Завжди єдина параметрична модель `θ_base`.
- **Дані:** Не використовуй `EventLog` / `EventLogBatch` як фізичні сутності у MVP1. Використовуй потік: `RawTrace` $\to$ `PrefixSlice` $\to$ `InstanceGraph` $\to$ `GraphTensorContract`.
- **Парсинг:** Не використовуй DOM-парсинг для XES (весь файл у пам'ять). Тільки потокове читання (`lxml.etree.iterparse` / `pm4py` stream).
- **Іменування:** Не змінюй назви класів/методів з контрактів (наприклад, `IXESAdapter`, `IPrefixPolicy`, `IGraphBuilder`).
- **Сліпе кодування:** Не генеруй код без посилання на конкретний файл і рядок/розділ документації, де це описано.

## 3. Стек технологій та бібліотеки
Дозволено використовувати:
- **Core:** Python 3.10+, `pydantic` (або `dataclasses` для DTO).
- **ML & Math:** `torch`, `torch_geometric`, `numpy`, `scikit-learn`, `matplotlib` (для артефактів/візуалізації метрик).
- **Data Ingestion:** `lxml`, `pm4py`, `pandas` (обмежено, тільки для аналітики, не для core-логіки).
- **Tracking:** `mlflow`.
- **Infra/Utils:** `PyYAML`, `psutil`, `tqdm`.
Заборонено додавати нові сторонні залежності без погодження. Якщо є логічна необхідність - запутуй що потрібно і для чого.

## 4. Зафіксовані контракти даних (DTO)
Ти зобов'язаний використовувати строго цю структуру полів для базових DTO:

```python
class EventRecord(BaseModel):
    activity_id: str
    timestamp: float                   # Unix epoch float
    resource_id: str
    lifecycle: Optional[str]
    position_in_trace: int
    duration: float                    # Seconds float
    time_since_case_start: float       # Seconds float
    time_since_previous_event: float   # Seconds float
    extra: Dict[str, Any]
    activity_instance_id: Optional[str]

class RawTrace(BaseModel):
    case_id: str
    process_version: str
    events: List[EventRecord]
    trace_attributes: Dict[str, Any]

class GraphTensorContract(TypedDict):
    x_cat: torch.LongTensor       # [N_nodes, F_cat]
    x_num: torch.FloatTensor      # [N_nodes, F_num]
    edge_index: torch.LongTensor  # [2, E_edges]
    edge_type: torch.LongTensor   # [E_edges] (Для MVP1: заповнено 0)
    y: torch.LongTensor           # [1]
    batch: torch.LongTensor       # [N_nodes]
    num_nodes: int                # N_nodes
```

## 5. Що робити, коли отримуєш завдання
Кожного разу, коли розробник дає тобі задачу, виконуй такий алгоритм:

1. Прочитай завдання повністю.
2. Подивись у цьому файлі (`AGENT_GUIDE.MD`) — чи є пряма заборона або обмеження.
3. Визнач, які саме `.md` файли стосуються задачі (згідно з пріоритетним списком вище).
4. Читай **тільки** ці файли в порядку пріоритету.
5. Склади план дій у форматі:
   - Файл, який буду змінювати/створювати.
   - Контракт/розділ, на який посилаюсь.
   - Що саме роблю (коротко алгоритм).
   - Чому це відповідає дисертації/архітектурі.
6. **Покажи план розробнику перед написанням коду.**
7. Тільки після команди "ок" — пиши код.

## 6. Як посилатися на документацію в коді
У кожному новому файлі або суттєвій функції додавай коментар зверху:

- Кожен логічний блок коду необхідно коментувати: що він робить і **для чого**.
- При створенні нової суттєвої доменної змінної, якої немає в `docs/VARIABLES.MD`, додавай її в останній блок (технічні змінні в коді). Локальні змінні функцій, індекси тощо вносити у файл не потрібно.

---

## MVP2.5 Addendum (2026-03-15)

### New mandatory docs to read for MVP2.5 tasks
1. `docs/ARCHITECTURE_MVP2_5.MD`
2. `docs/DATA_MODEL_MVP2_5.MD`
3. `docs/DATA_FLOWS_MVP2_5.MD`
4. `docs/LLD_MVP2_5.MD`
5. `docs/EVF_MVP2_5.MD`
6. `docs/ADAPTER_CAMUNDA_SQL.MD`

### Mandatory execution rule
- For MVP2.5, topology extraction must run in offline ingestion pipeline (`ingest-topology`).
- Training pipeline must consume repository artifacts via `IKnowledgeGraphPort` and must not rebuild topology synchronously.

### Working directories for reports and plans
- Root must stay clean.
- New and historical reports/plans are stored under `docs/worklogs/`.

## MVP2.5 Stage 3.4 Addendum (2026-03-19)

### Mandatory docs for any task related to structure/stats/train timing
Read in this strict order:
1. `docs/ARCHITECTURE_MVP2_5.MD`
2. `docs/DATA_MODEL_MVP2_5.MD`
3. `docs/DATA_FLOWS_MVP2_5.MD`
4. `docs/LLD_MVP2_5.MD`
5. `docs/EVF_MVP2_5.MD`
6. `README.MD` section `MVP2.5 Stage 3.4 Runbook (Canonical)`

### Hard execution rules
1. Topology and stats are prepared offline (`ingest-topology`, `sync-topology`, `sync-stats`).
2. Train/eval/infer must consume repository artifacts only.
3. For temporal-safe studies, use `experiment.stats_time_policy = strict_asof` and maintain snapshot timeline.

### Snapshot policy (must not be violated)
1. Immutable snapshots only.
2. Strict key: `tenant_id + process_name + version_key + proc_def_id + knowledge_version`.
3. JSON-only stats payload (no `NodeStat`/`EdgeStat` normalization).
4. No TTL.

## Config Catalog Rule (Mandatory)

For every new configurable parameter (any new key in `experiment`, `data`, `mapping`, `model`, `training`, `tracking`, `sync_stats`):

1. Add it to `configs/ui/config_catalog.yaml` in the same change.
2. Fill metadata at once: `description`, `affects`, `default`, `enum` (if applicable), `required_in_modes`/`required_when`, and `ui` placement.
3. Expose it in `tools/experiment_ui.py` (load, edit, save, hints, and mode-dependent enable/disable where relevant).
4. Keep CLI and UI behavior consistent; no hidden config-only parameters for interactive runs.
