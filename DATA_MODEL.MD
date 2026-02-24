# Data Model & Knowledge Graph Specification (DM-KGS)

**Project:** BPM Prediction Platform  
**Version:** 2.0 (Full Scope)  
**Status:** Approved Architecture Baseline  

Цей документ визначає специфікації даних для всіх рівнів системи: від збереження знань (POKG) до тензорів для нейромережі.

---

## 1. Класифікація Атрибутів

Для забезпечення гнучкості системи всі атрибути поділено на три рівні. Це дозволяє розділити "хардкод" ядра від налаштувань конкретного бізнес-процесу.

| Рівень | Опис | Приклади з коду | Конфігурація |
| :--- | :--- | :--- | :--- |
| **🔴 Fundamental**<br>(System Core) | Критичні поля для побудови топології графа та ідентифікації. Жорстко прописані в логіці `GraphBuilder`. | `ACT_ID_`, `PROC_INST_ID_`, `SEQUENCE_COUNTER_`, `PROC_KEY_` | **Hardcoded** |
| **🟡 Base**<br>(Standard BPM) | Стандартні метрики виконання, притаманні будь-якому процесу. Система має вбудовані препроцесори для них. | `DURATION_`, `START_TIME_`, `user_compl_login` (Resource) | `data_sources.yaml` |
| **🟢 Custom**<br>(Business Domain) | Специфічні бізнес-дані (контекст документа, результати виконання). Обробляються динамічно. | `overdue_work`, `duration_work`, `PurchasingBudget`, `CategoryL1` | `features.yaml` |

---

## 2. Ідентифікація та Ієрархія

Система підтримує складні вкладені процеси (Parent $\to$ Child).

### 2.1. Рівні Ідентифікаторів

| Сутність | ID в Системі | Джерело (Camunda) | Опис |
| :--- | :--- | :--- | :--- |
| **Document** | `doc_id` | *Business Key / Var* | Глобальний контейнер (напр., Заявка). Спільний контекст для всіх підпроцесів. |
| **Root Process** | `root_proc_id` | `ROOT_PROC_INST_ID_` | Головний екземпляр процесу. |
| **Scope (Trace)** | `proc_inst_id` | `PROC_INST_ID_` | ID конкретного рівня виконання (може бути CallActivity). |
| **Flow Node** | `node_instance_id` | `ID_` (Act Inst ID) | Унікальна подія виконання. |
| **BPMN Template** | **`bpmn_id`** | `ACT_ID_` | Посилання на вузол у POKG (Task Definition Key). |

---

## 3. POKG: Схема Бази Знань (Neo4j)

Граф, що зберігає "статику" (структуру) та "історію" (статистику).

### 3.1. Вузли (Nodes)

| Label | Основні Властивості | Опис |
| :--- | :--- | :--- |
| **`:ProcessDef`** | `process_key`, `version_tag`, `name` | Визначення процесу (напр., "Tender_v1"). |
| **`:Task`** | `bpmn_id`, `name`, `type`, `lane` | Активності (`UserTask`, `ServiceTask`, `CallActivity`). |
| **`:Gateway`** | `bpmn_id`, `gateway_type`, `direction` | Логічні розгалуження (`XOR`, `AND`). |
| **`:Event`** | `bpmn_id`, `event_type` | Події (`Start`, `End`, `Timer`). |
| **`:Role`** | `role_code` | Посада (`user_compl_position`). |
| **`:Resource`** | `user_login` | Конкретний виконавець (опційно, для детального аналізу). |

> **Важливо для валідації:** При формуванні `edge_attr` для `Fusion Graph` заборонено використовувати статистичні ознаки, що містять інформацію про майбутнє (наприклад, "середній час завершення всього кейсу"). Дозволяються лише:
> 1. Топологічні константи з BPMN.
> 2. Історичні дані, доступні на момент події $t$ (Cumulative Features).
> 3. 
### 3.2. Зв'язки (Relationships)

* **`(:Task)-[:FLOWS_TO]->(:Task)`**
    * *Stat Attributes:* `transition_prob` (частота переходу), `avg_duration_sec`.
* **`(:Task {type:'CallActivity'})-[:CALLS]->(:ProcessDef)`**
    * *Purpose:* Моделювання переходу в підпроцес (Hierarchy).
* **`(:Task)-[:PERFORMED_BY]->(:Role)`**
    * *Stat Attributes:* `execution_count`.

---

## 4. Instance Graph & Global Context (Dynamic Data)

Це дані, що надходять на вхід `Inference Controller`.

### 4.1. Глобальний Контекст Документа (Global Features)
Ці атрибути є незмінними (або повільно змінними) для всього графа. Вони задають "умови гри".

| Атрибут (Internal) | Джерело (Camunda Vars) | Тип | Обробка |
| :--- | :--- | :--- | :--- |
| `budget_amount` | `PurchasingBudget` | Float | LogNorm |
| `price_final` | `FinalPrice` | Float | LogNorm |
| `category_l1` | `CategoryL1` | Cat | Embedding |
| `supplier_class` | `ClassSSD` | Cat | Embedding |
| `company_id` | `Company_SO` | Cat | Embedding |

### 4.2. Атрибути Вузлів (Node Features)
Дані з таблиці історії (`ACT_HI_ACTINST` + `bpm_tasks`).

| Атрибут | Джерело | Тип | Опис |
| :--- | :--- | :--- | :--- |
| `duration_total` | `DURATION_` | Float | Повний час життя токена. |
| `duration_work` | `duration_work` | Float | Чистий час роботи виконавця. |
| `wait_time` | *Calc (`first_view` - `START`)* | Float | Час простою в черзі. |
| `is_overdue` | `overdue_work` | Bool | Чи порушено SLA (з `global_statistics`). |
| `action_result` | `taskaction_code` | Cat | Код результату (Approve/Reject). |
| `task_status` | `task_status` | Cat | Технічний статус (Completed/Deleted). |
| `executor_role` | `user_compl_position` | Cat | Роль з POKG (через Fusion). |
| `concurrency` | `active_executions` | Int | Кількість активних токенів у момент виконання. |

---

## 5. Feature Engineering (Tensor Specification)

Опис того, як атрибути перетворюються на тензори для моделі.

### 5.1. Time Encoding Strategy
Використовується гібридний підхід для кодування часу.

1.  **Relative Dynamics (Dynamic):**
    * Вхід: `timestamps` (Unix time).
    * Логіка (в `forward`): `delta_t = current_max_time - node_time`.
    * Енкодинг: **Time2Vec** (вивчає патерни "недавніх" подій).
2.  **Calendar Context (Static):**
    * Вираховується з `START_TIME_` під час препроцесингу.
    * Фічі: `Month` (Emb), `DayOfWeek` (Emb), `IsWeekend` (Bool), `IsBusinessHour` (Bool), `Quarter` (Emb).

### 5.2. Node Vector ($x$) Construction
Вектор кожного вузла формується конкатенацією:

```python
x_node = Concat(
    # --- 1. Dynamic Business Metrics (IG) ---
    LogNorm(duration_work),      # Чиста робота
    LogNorm(wait_time),          # Очікування
    StandardScaler(overdue_val), # Наскільки прострочено
    Identity(is_overdue_flag),   # Факт прострочення
    Identity(active_execs),      # Паралелізм

    # --- 2. Categorical Embeddings (IG + Fusion) ---
    Embedding(action_result),    # Що вирішили?
    Embedding(task_status),      # Як завершилось?
    Embedding(executor_role),    # Хто робив?

    # --- 3. Structural Context (POKG) ---
    Node2Vec(bpmn_id),           # Де це в схемі?
    OneHot(node_type),           # Task vs Gateway

    # --- 4. Calendar Context ---
    Embedding(month), Embedding(weekday), ...
)
````

### 5.3. Global Vector ($u$)

Вектор документа, який впливає на весь граф (через механізм Attention або конкатенацію після пулінгу).

```python
u_global = Concat(
    LogNorm(budget), 
    LogNorm(price), 
    Embedding(category), 
    Embedding(supplier)
)
```

-----

## 6\. Tensor Interface (PyTorch Geometric)

Фінальний контракт даних, що передається у метод `model.forward(data)`.

1.  **`x` (Node Features):**

      * Shape: `[num_nodes, node_dim]`
      * Type: `torch.float32`

2.  **`edge_index` (Adjacency):**

      * Shape: `[2, num_edges]`
      * Type: `torch.long` (COO Format)

3.  **`edge_attr` (Edge Features):**

      * Shape: `[num_edges, edge_dim]`
      * Content: `[LogNorm(duration_transition), POKG_Probability]`

4.  **`doc_features` (Global Context):**

      * Shape: `[batch_size, global_dim]`
      * *Примітка:* Це окремий тензор, не частина графа, передається в модель окремим аргументом.

5.  **`timestamps` (Raw Time):**

      * Shape: `[num_nodes, 1]`
      * Призначення: Для розрахунку `Time2Vec` всередині моделі.

6.  **`batch`:**

      * Shape: `[num_nodes]`
      * Index mapping вузлів до графів у батчі.

7.  **`y` (Targets):**

      * `y_class`: `[batch_size]` (Next Activity Index).
      * `y_time`: `[batch_size]` (Remaining Time Normalized).

<!-- end list -->

```
```
