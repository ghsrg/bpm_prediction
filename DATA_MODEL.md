
# Data Model & Knowledge Graph Specification (DM-KGS)

**Project:** BPM Prediction Platform  
**Version:** 1.0 (Draft)  
**Status:** Architecture Baseline

Цей документ визначає схеми даних для збереження знань (Neo4j), динамічного відображення (IG) та навчання моделей (Tensors).

-----

## 1\. Концепція Потоку Даних

Перетворення даних відбувається у три етапи:

1.  **Raw Data:** SQL (Camunda) або XES $\to$ Табличний вигляд (DataFrame).
2.  **Graph Construction:**
      * **Static:** BPMN XML $\to$ **POKG** (Neo4j).
      * **Dynamic:** DataFrame $\to$ **Instance Graph** (NetworkX).
3.  **Fusion & Vectorization:** Merge(IG, POKG) $\to$ **Fusion Graph** $\to$ **PyTorch Tensors**.

-----

## 2\. POKG: Process Organizational Knowledge Graph (Neo4j)

Це "холодне" сховище структури та контексту.

### 2.1. Вузли (Nodes & Labels)

| Label | Джерело (BPMN/Code) | Атрибути (Properties) | Опис |
| :--- | :--- | :--- | :--- |
| **`ProcessDef`** | Root Process | `process_key`, `version_tag`, `name` | Визначення процесу (напр., "OrderHandling"). |
| **`Task`** | `<userTask>`, `<serviceTask>` | `bpmn_id`, `name`, `type` | Елементарна дія. |
| **`Gateway`** | `<exclusiveGateway>`, `<parallelGateway>` | `bpmn_id`, `type`, `direction` | Точка прийняття рішень (Converging/Diverging). |
| **`Event`** | `<startEvent>`, `<endEvent>` | `bpmn_id`, `type` | Початок/кінець або проміжна подія. |
| **`Role`** | `user_compl_position` (Code) | `name`, `code` | Посада виконавця (напр., "Manager"). |
| **`User`** | `user_compl_login` (Code) | `login_hash` | Конкретний виконавець (анонімізований). |
| **`Action`** | `taskaction_code` (Code) | `code`, `description` | Результат виконання задачі (напр., "Approve", "Reject"). |

### 2.2. Зв'язки (Relationships)

| Type | Від (Source) | До (Target) | Атрибути ребра |
| :--- | :--- | :--- | :--- |
| **`FLOWS_TO`** | Task/Gateway | Task/Gateway | `prob_v1` (ймовірність переходу), `avg_time` |
| **`PERFORMED_BY`** | Task | Role | `frequency` |
| **`POSSIBLE_ACTION`**| Task | Action | - |
| **`CONTAINS`** | ProcessDef | Task/Event | - |
| **`CALLS`** | Task (CallActivity) | ProcessDef | Зв'язок підпроцесів (SubProcess) |

-----

## 3\. Instance Graph (Runtime Data)

Це "гаряче" представлення конкретного трейсу (Case), побудоване з `camunda_actions`.

### 3.1. Вхідний DataFrame (Log Schema)

Ці поля повинні бути забезпечені адаптером (`mssql_connector`):

  * `PROC_INST_ID_`: ID кейсу.
  * `ACT_ID_`: ID активності (мапиться на `bpmn_id` у POKG).
  * `START_TIME_`, `END_TIME_`: Часові мітки.
  * `DURATION_`: Тривалість (фактична).
  * `SEQUENCE_COUNTER_`: Порядковий номер у трейсі.
  * **Бізнес-атрибути (з вашого коду):**
      * `user_compl_position`: Роль.
      * `user_compl_login`: Користувач.
      * `taskaction_code`: Код результату.
      * `task_status`: Статус задачі.
      * `overdue_work`: Флаг прострочення (Boolean).

### 3.2. Графова структура IG

  * **Вузли:** Події, що *вже відбулися*.
  * **Ребра:** `DIRECTLY_FOLLOWS` (послідовність виконання у часі).
  * **Тимчасові атрибути:** `time_since_start`, `time_since_prev`.

-----

## 4\. Fusion Graph: Логіка Злиття (Mapping)

Це ключовий етап перетворення даних для AI.

### 4.1. Алгоритм злиття (Merge Logic)

Для кожного вузла $u$ в Instance Graph:

1.  Знайти відповідний вузол $v$ у POKG (за `ACT_ID_` == `bpmn_id`).
2.  **Enrichment (Збагачення):**
      * Додати ембеддінг типу вузла з POKG (напр., вектор для `UserTask`).
      * Додати ембеддінг Ролі з POKG (вектор для `Manager`).
      * Додати глобальні статистики з POKG (напр., "середній час виконання цієї задачі").
3.  **Handling OOS:** Якщо вузол є в IG, але немає в POKG (напр., стара версія процесу), використовувати спеціальний токен `<UNK_TASK>`.

-----

## 5\. Tensor Interface (Spec for PyTorch)

Це формат, який очікує модель (`BaseModel.forward`).

### 5.1. Матриця ознак вузлів ($X$)

Розмірність: `[N_nodes, Input_Dim]`. Вектор кожного вузла складається з конкатенації:

| Секція | Джерело | Тип | Розмірність (Приклад) |
| :--- | :--- | :--- | :--- |
| **Numerical** | IG | `Float` | 4 (`norm_duration`, `active_execs`, `overdue_flag`, `time_since_start`) |
| **Positional** | IG | `Sinusoidal` | 8 (Time2Vec encoding часу) |
| **Struct Emb** | POKG | `Embedding` | 16 (Node2Vec вектор вузла з графа знань) |
| **Type Emb** | POKG | `OneHot` | 8 (Task vs Gateway vs Event) |
| **Role Emb** | Fusion | `Embedding` | 8 (Вектор для `user_compl_position`) |
| **Action Emb** | Fusion | `Embedding` | 4 (Вектор для `taskaction_code`) |
| **TOTAL** | - | - | **48** (Input Dimension моделі) |

### 5.2. Атрибути Ребер ($E_{attr}$)

Розмірність: `[N_edges, Edge_Dim]`.

  * `norm_duration_between`: Час переходу.
  * `structural_probability`: Ймовірність переходу з POKG (апріорне знання).

### 5.3. Цільові змінні ($Y$)

1.  **Classification Head:** `Next_Activity_Index` (Long).
2.  **Regression Head:** `Remaining_Time_Normalized` (Float).

-----

## 6\. Feature Configuration Strategy

Замість хардкоду в `graph_creator.py`, ми використовуємо `features.yaml`.

**Приклад конфігурації для вашого процесу:**

```yaml
# features.yaml
features:
  node:
    # Числові з логу (IG)
    - name: "DURATION_"
      type: "numerical"
      preprocessing: "log_norm" # ln(x+1)
      fill_na: 0.0
    
    - name: "active_executions"
      type: "numerical"
      preprocessing: "standard_scaler"

    - name: "overdue_work"
      type: "categorical" # Хоча це bool, краще як категорію 0/1
      preprocessing: "identity"

    # Категоріальні з логу (Fusion -> POKG)
    - name: "user_compl_position"
      type: "categorical"
      preprocessing: "embedding"
      params: { embedding_dim: 8, vocab_source: "pokg_roles" }

    - name: "taskaction_code"
      type: "categorical"
      preprocessing: "embedding"
      params: { embedding_dim: 4 }

  edge:
    - name: "DURATION_E" # З вашого коду (тривалість переходу)
      type: "numerical"
      preprocessing: "log_norm"
```

### Наступні кроки для реалізації:

1.  Створити **Schema Migration Script** для Neo4j (створення констрейнтів та індексів для `bpmn_id`).
2.  Написати **BPMN-to-Cypher Parser** (адаптація вашого XML-парсера для генерації Cypher-запитів).
3.  Реалізувати клас `FeaturePreprocessor`, який читає цей YAML.