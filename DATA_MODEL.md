# Data Model & Knowledge Graph Specification (DM-KGS)

**Project:** BPM Prediction Platform  
**Status:** Draft v1.1  
**Scope:** POKG, Instance Graph, Fusion Logic, Tensor Structures  

Цей документ описує схему даних системи. Атрибути поділено на три рівні для забезпечення гнучкості та масштабованості.

---

## 1. Класифікація Атрибутів (Attribute Stratification)

Щоб система була одночасно жорсткою (для коду) і гнучкою (для бізнесу), ми розділяємо всі дані на три категорії:

1.  **🔴 Fundamental (System Core)**
    * **Опис:** Критичні поля, без яких система впаде (Hardcoded логіка). Використовуються для зв'язування графів, ідентифікації вузлів та побудови топології.
    * **Приклади:** `node_id`, `process_id`, `source`, `target`.
    * **Налаштування:** Не змінюються.

2.  **🟡 Base (Standard BPM)**
    * **Опис:** Стандартні атрибути, притаманні будь-якому бізнес-процесу (і в XES, і в Camunda). Система має дефолтну логіку для них (наприклад, розрахунок часу), але їх назви можна переназначити.
    * **Приклади:** `timestamp`, `duration`, `resource`, `role`.
    * **Налаштування:** Мапяться у `data_sources.yaml`.

3.  **🟢 Custom (Business Specific)**
    * **Опис:** Унікальні поля конкретного процесу (наприклад, "сума замовлення", "тип скарги"). Система про них не знає, поки вони не описані в конфігу.
    * **Приклади:** `amount`, `risk_level`, `customer_type`.
    * **Налаштування:** Описуються у `features.yaml` для перетворення в тензори.

---

## 2. POKG: Схема Бази Знань (Neo4j)

Це граф, що зберігає структуру та контекст. Вузли створюються парсером BPMN, атрибути наповнюються з логів (офлайн).

### 2.1. Вузли (Nodes)

| Label | Fundamental (Обов'язкові) | Base (Стандартні) | Custom (Приклад) |
| :--- | :--- | :--- | :--- |
| **`:Task`** | `bpmn_id` (з XML)<br>`element_type` (Task) | `name` (Human readable)<br>`lane` (Swimlane) | `risk_weight` |
| **`:Gateway`** | `bpmn_id`<br>`gateway_type` (XOR/AND) | `direction` (Diverging) | - |
| **`:Event`** | `bpmn_id`<br>`event_type` (Start/End) | - | - |
| **`:Role`** | `role_id` (Hash/Code) | `name` (e.g. "Manager") | `hourly_rate` |
| **`:Version`** | `tag` (v1.0) | `valid_from`<br>`valid_to` | `deployer_id` |

### 2.2. Зв'язки (Relationships)

| Type | Source $\to$ Target | Fundamental Attrs | Base / Statistical Attrs |
| :--- | :--- | :--- | :--- |
| **`:FLOWS_TO`** | Task $\to$ Task | - | `count` (скільки разів йшли)<br>`avg_duration` (сер. час переходу)<br>`probability` (вага) |
| **`:PERFORMED_BY`** | Task $\to$ Role | - | `frequency` |
| **`:BELONGS_TO`** | Task $\to$ Version | - | - |

> **Примітка:** Статистичні атрибути (`avg_duration`, `probability`) оновлюються спеціальним Worker-ом, який агрегує історичні Instance Graphs.

---

## 3. Instance Graph (IG) Specification

Це граф конкретного виконання (Trace), що будується в пам'яті (NetworkX).

### 3.1. Вхідний DataFrame (Mapping)
Адаптери (`CamundaAdapter` / `XESAdapter`) повинні привести сирі дані до внутрішнього стандарту.

| Internal Field | Camunda Column (Source) | XES Attribute (Source) | Тип |
| :--- | :--- | :--- | :--- |
| **`case_id`** 🔴 | `PROC_INST_ID_` | `trace:concept:name` | String |
| **`activity_id`** 🔴 | `ACT_ID_` | `concept:name` | String |
| **`seq_num`** 🔴 | `SEQUENCE_COUNTER_` | *Index in trace* | Int |
| **`timestamp`** 🟡 | `END_TIME_` | `time:timestamp` | Datetime |
| **`duration`** 🟡 | `DURATION_` | *Calc: end - start* | Float |
| **`resource`** 🟡 | `user_compl_login` | `org:resource` | String |
| **`role`** 🟡 | `user_compl_position` | `org:role` | String |
| **`result_code`** 🟢 | `taskaction_code` | `lifecycle:transition` | Cat |
| **`is_overdue`** 🟢 | `overdue_work` | - | Bool |

### 3.2. Графова структура
* **Вузли:** Відповідають подіям у лозі. ID вузла = `case_id` + `seq_num`.
* **Ребра:** `DIRECTLY_FOLLOWS` ($Node_t \to Node_{t+1}$).
* **Атрибути вузла:** Всі поля з таблиці вище зберігаються як properties словника `networkx`.

---

## 4. Fusion Graph & Tensor Mapping

Це найважливіша частина для ML. Тут описується, як атрибути перетворюються на матрицю $X$.

### 4.1. Вектор Вузла ($X$)
Вектор формується конкатенацією (Concat) оброблених фіч. Конфігурація задається в `features.yaml`.

| Feature Group | Source | Attribute Name | Processing Method | Output Dim (Приклад) |
| :--- | :--- | :--- | :--- | :--- |
| **Structural** | **POKG** | `node2vec_embedding` | *Pre-calculated in Neo4j* | 16 |
| **Org Context** | **Fusion** | `role` (`user_compl_position`) | `Embedding(Vocab)` | 8 |
| **Dynamic** | **IG** | `duration` (`DURATION_`) | `LogNorm` ($\ln(x+1)$) | 1 |
| **Dynamic** | **IG** | `timestamp` | `Time2Vec` (Sin/Cos) | 8 |
| **Custom** | **IG** | `result_code` (`taskaction_code`) | `OneHot` | 5 |
| **Custom** | **IG** | `is_overdue` | `Identity` (0/1) | 1 |
| **Total** | | | | **39** |

### 4.2. Конфігурація (`features.yaml`)
Цей файл керує тим, які **Custom** та **Base** поля потрапляють у модель.

```yaml
# features.yaml example

system_config:
  # Fundamental mapping (Hardwired logic uses these keys)
  activity_id_col: "ACT_ID_"
  case_id_col: "PROC_INST_ID_"

features:
  # Base & Custom features definition
  - name: "DURATION_"           # Колонка в DataFrame
    type: "numerical"
    source: "log"               # Брати з поточного логу
    preprocessing: "log_norm"
  
  - name: "user_compl_position"
    type: "categorical"
    source: "log"
    preprocessing: "embedding"
    params: { dim: 8, vocab_key: "roles" }

  - name: "avg_duration"        # Атрибут з POKG (Context)
    type: "numerical"
    source: "pokg"              # Підтягується через Fusion
    preprocessing: "minmax"
````

-----

## 5\. Tensor Specifications (PyTorch Geometric)

Інтерфейс, який очікують моделі (`GNN.forward()`).

1.  **`x` (Node Features):**

      * Type: `torch.float32`
      * Shape: `[num_nodes, feature_dim]` (наприклад, `[N, 39]`)

2.  **`edge_index` (Adjacency):**

      * Type: `torch.long`
      * Shape: `[2, num_edges]`
      * Format: COO (Coordinate format)

3.  **`edge_attr` (Edge Features):**

      * Type: `torch.float32`
      * Shape: `[num_edges, edge_dim]`
      * Content: `[probability, avg_time_norm]` (з POKG).

4.  **`batch` (Graph Indicator):**

      * Type: `torch.long`
      * Shape: `[num_nodes]`
      * Description: Індекс графа в батчі, до якого належить вузол.

5.  **`y` (Target):**

      * *Next Activity:* `torch.long`, Shape `[1]` (Class Index).
      * *Time:* `torch.float32`, Shape `[1]` (Normalized Duration).

<!-- end list -->

```
```