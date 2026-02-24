*(Технічна специфікація витягування даних з Camunda 7 + Custom Tables)*

# Adapter Specification: Camunda SQL & BPMS

**Status:** Draft v1.0
**Source System:** Camunda BPM 7.x (MSSQL) + Custom `bpm_tasks` table.

## 1. Огляд джерел даних

Адаптер працює з трьома групами таблиць:
1.  **Repository (`ACT_RE_*`):** Статичні визначення процесів та XML-файли.
2.  **History (`ACT_HI_*`):** Стандартний лог виконання Camunda.
3.  **Business Data (`bpm_tasks`):** Кастомні таблиці збагаченна процесів бізнес даними та додатковими атрибутами.

## 2. Логіка Екстракції (Extraction Logic)

### 2.1. Flow A: POKG Structure Discovery (Recursion)
Цей потік відповідає за побудову топології процесів та відновлення зв'язків `Parent-Child`.

**Алгоритм роботи:**
1.  **Entry Point:** Отримання списку кореневих процесів (з конфігу, на етапі розгортання та дебагу або авто-пошук з бази).
2.  **Recursive Fetch:** Завантаження BPMN, парсинг, пошук тегів `<callActivity>`, додавання посилань на дочірні процеси які залінковані.
3.  **Linkage:** Створення ребер `(:Task)-[:CALLS]->(:ProcessDef)` у POKG.

### 2.2. POKG Structure Discovery (Modular Approach)
Замість повної рекурсивної розгортки всіх вкладених процесів (що веде до надмірної довжини трас >100 вузлів), адаптер реалізує **локальний контекст**:

1. **Context Boundary:** Навчання та інференс проводяться в межах одного `Process Definition`.
2. **Call Activity Handling:** Вузли типу `callActivity` розглядаються як "чорні скриньки" з власним вектором стану. 
3. **Версійність:** Оскільки кожна `Call Activity` може посилатися на різні версії дочірніх процесів, система використовує `version_id` поточного рівня для фіксації графа.
4. **Взаємодія** Для повного циклу та розуміння helocopter view потрібно мати розуміння який процес викликав цей і які викликати наступні.
*Перевага:* Це забезпечує масштабованість та дозволяє моделі фокусуватися на логіці поточного рівня ієрархії.

#### Крок 1: Discovery (Звідки знати process_key?)
Система підтримує два режими виявлення точок входу:

* **Mode 1: Config-Driven (Recommended)**
    Список кореневих процесів задається явно в `config/ingestion.yaml`. Це дозволяє обмежити POKG тільки необхідними процесами.
    ```yaml
    root_processes:
      - "order_process"
      - "invoice_approval"
    ```

* **Mode 2: Auto-Discovery (Scan)**
    Якщо конфіг порожній, адаптер сканує БД на предмет усіх унікальних ключів процесів.
    ```sql
    SELECT DISTINCT KEY_ FROM ACT_RE_PROCDEF;
    ```

#### Крок 2: BPMN Extraction Query
Запит оновлено для отримання назви та версії. Виконується ітеративно для кожного `process_key` з черги.

```sql
SELECT 
    P.KEY_       AS process_key,
    P.VERSION_   AS process_version,
    P.NAME_      AS process_name,      -- Додано для читабельності в графі
    P.ID_        AS proc_def_id,       -- Унікальний ID версії (напр., order:1:123)
    B.BYTES_     AS bpmn_xml_content
FROM ACT_RE_PROCDEF P
JOIN ACT_GE_BYTEARRAY B 
    ON P.DEPLOYMENT_ID_ = B.DEPLOYMENT_ID_
    AND P.RESOURCE_NAME_ = B.NAME_
WHERE P.KEY_ = :target_process_key
-- Опційно: брати тільки останню версію або всі
ORDER BY P.VERSION_ DESC; 

```

#### Крок 3: Parsing & Child Discovery Strategy

Це критичний етап відновлення зв'язків, який виконується на стороні Python (`POKGBuilder`).

1. Отриманий `bpmn_xml_content` передається в парсер.
2. Парсер шукає елементи типу `callActivity`.
3. Витягує атрибут `calledElement` (це `process_key` дочірнього процесу).
* *Приклад:* `<bpmn:callActivity id="Activity_CheckRisk" calledElement="risk_assessment_v1" />`


4. Якщо `risk_assessment_v1` ще немає в графі, цей ключ додається в чергу **To-Do List** для виконання SQL-запиту (Крок 2).
5. Після завантаження обох процесів у Neo4j створюється зв'язок:
```cypher
MATCH (parent:Task {bpmn_id: 'Activity_CheckRisk'})
MATCH (child:ProcessDef {key: 'risk_assessment_v1'})
MERGE (parent)-[:CALLS]->(child)

```



> **Важливо про версійність Call Activity:**
> Camunda дозволяє викликати дочірній процес двома способами:
> 1. *Binding "latest":* Викликається найсвіжіша версія. У графі лінкуємо на вузол `ProcessDef` (abstract).
> 2. *Binding "version":* Викликається конкретна версія. У графі лінкуємо на конкретний `ProcessVersion`.
> 
> 
> Адаптер має зчитувати атрибут `camunda:processBinding` з XML, щоб створити правильний зв'язок.

```


* **Mapping:** `bpmn_xml_content` передається в `BPMNParser` для розбору топології.

---

### 2.2. Flow B: Отримання Логів (Runtime + Enrichment)

Використовується для побудови Instance Graph (IG).
Адаптер виконує `LEFT JOIN` стандартної історії з бізнес-таблицею.

**SQL Query:**

```sql
SELECT 
    -- 1. Core Identifiers
    H.PROC_INST_ID_              AS case_id,
    H.ACT_ID_                    AS activity_id,  -- Key for POKG Merge
    H.PROC_DEF_ID_               AS proc_def_id,  -- Version Context
    
    -- 2. Temporal Data
    H.END_TIME_                  AS timestamp,
    H.START_TIME_                AS start_time,
    H.DURATION_                  AS duration_ms,
    
    -- 3. Resource & Role (Priority Logic)
    -- Якщо в бізнес-таблиці є логін, беремо його, інакше - технічного виконавця Camunda
    COALESCE(B.user_compl_login, H.ASSIGNEE_) AS resource_id,
    
    -- Реальна посада важливіша за групу Camunda
    B.user_compl_position        AS role_id,
    
    -- 4. Business Attributes (Features)
    B.taskaction_code            AS action_code,  -- Result (Approve/Reject)
    CAST(B.overduet_work AS INT) AS is_overdue,   -- SLA Violation Flag
    H.ACT_TYPE_                  AS activity_type -- UserTask/Gateway/Service
    
FROM ACT_HI_ACTINST H
LEFT JOIN bpm_tasks B ON H.TASK_ID_ = B.externalid
WHERE 
    H.PROC_DEF_KEY_ = :process_key
    AND H.END_TIME_ IS NOT NULL -- Беремо тільки завершені події для навчання
ORDER BY H.PROC_INST_ID_, H.START_TIME_;

```

## 3. Трансформація типів (Data Casting)

Адаптер відповідає за приведення типів SQL до Python типів:

| Field | Source Type | Target Type (Python) | Note |
| --- | --- | --- | --- |
| `timestamp` | `datetime2` | `pd.Timestamp` (UTC) | timezone-aware |
| `duration_ms` | `bigint` | `float` (Seconds) | `val / 1000.0` |
| `is_overdue` | `bit/int` | `bool` | `1=True, 0=False` |
| `action_code` | `varchar` | `str` / `category` | Handle `NULL` as `<NA>` |

## 4. Обробка помилок (Error Handling)

* **Missing Business Data:** Якщо `LEFT JOIN` з `bpm_tasks` не знайшов запису (наприклад, для автоматичного `ServiceTask`), поля `role_id` та `action_code` будуть `NULL`. Адаптер має заповнити їх дефолтними значеннями: `role_id='system'`, `action_code='auto'`.
* **Corrupted Traces:** Трейси, у яких відсутній `StartEvent`, ігноруються або позначаються прапором `valid=False`.

```

---

### Файл 2: `DATA_MODEL.md` (Core)
*(Чистий контракт ядра системи)*

```markdown
# Data Model & Knowledge Graph Specification (Core)

**Status:** Draft v1.0
**Scope:** Core Internal Structures (POKG, IG, Fusion, Tensors).

Цей документ описує внутрішні структури даних, якими оперує ядро системи. Усі вхідні дані мають бути приведені до цих форматів відповідними Адаптерами.

## 1. POKG Schema (Neo4j)
"Холодне" сховище знань про структуру та контекст.

### 1.1. Вузли (Nodes)
| Label | Ключові властивості | Опис |
| :--- | :--- | :--- |
| `Task` | `id` (BPMN ID), `name`, `type` | Елементарна дія |
| `Gateway` | `id`, `type` (XOR/AND), `direction` | Точка прийняття рішень |
| `Role` | `name`, `code` | Організаційна одиниця |
| `ProcessVersion`| `proc_def_id`, `version_tag` | Якір версії |

### 1.2. Зв'язки (Relationships)
* `(:Task)-[:FLOWS_TO {prob: 0.8}]->(:Task)`
* `(:Task)-[:PERFORMED_BY]->(:Role)`
* `(:ProcessVersion)-[:CONTAINS]->(:Task)`

---

## 2. Internal Event Log (Intermediate)
Уніфікований формат DataFrame, який очікує `IGBuilder`.

| Column | Type | Description |
| :--- | :--- | :--- |
| `case_id` | `str` | Trace Identifier |
| `activity_id` | `str` | Link to POKG Node (`bpmn_id`) |
| `timestamp` | `datetime` | Completion time |
| `role_id` | `str` | Link to POKG Role |
| `resource_id` | `str` | Specific Performer ID |
| `duration_sec` | `float` | Activity Duration |
| `action_code` | `str` | Result code (Decision) |
| `is_overdue` | `bool` | SLA violation flag |

---

## 3. Fusion Graph (ML Input)
Об'єкт, що утворюється після злиття `IG` та `POKG`.

### 3.1. Structure
* **Nodes:** Всі події з `IG` + контекстні вузли з `POKG` (якщо застосовується розширення).
* **Node Attributes:** Конкатенація динамічних та статичних фіч.

### 3.2. Tensor Specification (PyTorch)
Вхідний інтерфейс для GNN моделей.

**Node Matrix ($X$):**
Розмірність: `[N, D_total]`, де `D_total` визначається конфігом `features.yaml`.
* `Features[0:4]`: Numerical (Duration, ActiveExecs...)
* `Features[4:12]`: Time Encoding (Sin/Cos)
* `Features[12:28]`: Structural Embedding (Node2Vec from POKG)
* `Features[28:36]`: Role Embedding

**Edge Index ($A$):**
Розмірність: `[2, E]`. `long` тензор індексів `source -> target`.

**Masks:**
* `batch`: Вказує, до якого графа в батчі належить вузол.
* `allow_mask` (Optional): Вектор допустимих наступних кроків (розмір `VocabSize`).

```

---

### Що далі?

Тепер, коли ми розділили "мух і котлети", ми можемо спокійно:

1. Створити ці файли.
2. Перейти до деталізації **"Adapter XES"** (якщо потрібно зараз) або відразу до **"Feature Configuration"** (опис `features.yaml`), оскільки це наступний крок після отримання `Internal Event Log`.

Який наступний документ беремо в роботу?