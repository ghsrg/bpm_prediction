# System Architecture Document (SAD)

**Project:** BPM Prediction Platform (`bpm_prediction`)  
**Status:** Draft v1.0  
**Authors:** Serhii Korotenko  

Цей документ надає всебічний архітектурний огляд платформи прогнозування бізнес-процесів. Він призначений для розробників, архітекторів та дослідників.

- [System Architecture Document (SAD)](#system-architecture-document-sad)
  - [1. Business Context (Бізнес-контекст)](#1-business-context-бізнес-контекст)
    - [1.1. Основні бізнес-функції](#11-основні-бізнес-функції)
    - [1.2. Зацікавлені сторони (Stakeholders)](#12-зацікавлені-сторони-stakeholders)
  - [2. System Context (Контекстна діаграма)](#2-system-context-контекстна-діаграма)
  - [6. Фізична архітектура та Інфраструктура](#6-фізична-архітектура-та-інфраструктура)
    - [6.1. Container Model (C4)](#61-container-model-c4)
    - [6.2. Топологія розгортання (Deployment)](#62-топологія-розгортання-deployment)
  - [7. Дані і Сховище](#7-дані-і-сховище)
    - [7.1. Моделі даних](#71-моделі-даних)
    - [7.2. Управління даними](#72-управління-даними)
  - [8. API та Інтеграції](#8-api-та-інтеграції)
    - [8.1. Протоколи](#81-протоколи)
    - [8.2. Контракт API (Приклад)](#82-контракт-api-приклад)
  - [9. Безпека](#9-безпека)
  - [10. Observability (Спостережуваність)](#10-observability-спостережуваність)
  - [11. DevOps / Deployment](#11-devops--deployment)
  - [12. Ризики та Обмеження](#12-ризики-та-обмеження)
  - [13. Майбутній розвиток (Roadmap)](#13-майбутній-розвиток-roadmap)


---

## 1. Business Context (Бізнес-контекст)

### 1.1. Основні бізнес-функції
Система вирішує проблему невизначеності у виконанні бізнес-процесів, надаючи інструменти для:
1.  **Predictive Monitoring:** Прогнозування наступного кроку (Next Activity) та часу завершення (Time Prediction) для активних процесів.
2.  **Structural Drift Adaptation:** Забезпечення точності прогнозів навіть при зміні версій процесів завдяки POKG.
3.  **Process Simulation Benchmarking:** Наукове порівняння ефективності різних ML-архітектур (GNN vs LSTM).

### 1.2. Зацікавлені сторони (Stakeholders)
* **Бізнес-користувачі (BPMS):** Отримують підказки (Recommendations) та попередження про порушення SLA.
* **Data Scientists:** Використовують платформу для експериментів з новими архітектурами GNN.
* **Process Owners:** Аналізують розходження між планом (BPMN) та фактом (Logs).

---

## 2. System Context (Контекстна діаграма)

Система `bpm_prediction` є "інтелектуальним ядром", яке вбудовується в ландшафт підприємства.

```mermaid
C4Context
    title System Context Diagram for BPM Prediction Platform

    Person(admin, "Data Scientist / Admin", "Configures experiments, manages POKG")
    System(bpm_pred, "BPM Prediction Core", "Fusion Graph Engine + GNN Models")
    
    System_Ext(camunda, "Camunda BPMS", "Orchestrates processes, stores runtime data (SQL)")
    System_Ext(neo4j, "Neo4j Graph DB", "Stores EPOKG (Enterprise Process-Org Graph)")
    System_Ext(files, "File Storage", "XES Logs, BPMN files, Checkpoints")

    Rel(admin, bpm_pred, "Configures YAML, Runs Pipelines")
    Rel(bpm_pred, camunda, "Reads Logs (SQL/API), Returns Predictions (REST)")
    Rel(bpm_pred, neo4j, "Reads Context, Updates Statistics")
    Rel(bpm_pred, files, "Reads XES/BPMN, Saves Models")

-----

## 3\. Вимоги (Requirements)

### 3.1. Функціональні вимоги

1.  **Ingestion:** Парсинг BPMN (XML) та Event Logs (XES, MSSQL).
2.  **Knowledge Management:** Побудова та оновлення POKG у Neo4j.
3.  **Fusion:** Динамічне злиття IG (Instance Graph) та POKG для формування тензорів.
4.  **Training:** Навчання GNN моделей з конфігурованими гіперпараметрами.
5.  **Inference:** Надання прогнозу (клас + ймовірність + час) за \<200ms.
6.  **OOS Handling:** Детекція та маркування прогнозів, що виходять за межі POKG.

### 3.2. Нефункціональні вимоги (NFR)

  * **Reproducibility (Відтворюваність):** 100% ідентичні результати при фіксованому `seed` та конфігу.
  * **Modularity:** Можливість додавання нової GNN архітектури без зміни пайплайнів.
  * **Interoperability:** Робота з даними Camunda незалежно від версії процесу.
  * **Scalability (Data):** Підтримка графів до 10^5 вузлів (Neo4j GDS).

-----

## 4\. Архітектурне бачення

### 4.1. Архітектурний стиль

Обрано стиль **Modular Monolith** з чіткою слоистою структурою (Layered Architecture).

  * *Чому не Microservices?* На етапі Research/PoC важлива простота дебагу, спільна пам'ять для графів та швидкість розробки. Розділення на сервіси (Inference API vs Training Worker) можливе в майбутньому.

### 4.2. Принципи

  * **Clean Architecture:** Бізнес-логіка (Models, Graph Builders) не залежить від фреймворків доставки (API, CLI).
  * **Config-Driven Development:** Вся поведінка (фічі, моделі) визначається у YAML, а не в коді.
  * **Graph-First:** Граф є первинною структурою даних; таблиці використовуються лише як сировина.

-----

## 5\. Логічна архітектура (Logical View)

Система складається з 5 основних модулів (Bounded Contexts).

1.  **Data Access Layer (Adapters):** Абстракція над I/O (SQL, File, Neo4j).
2.  **Graph Construction Layer:** Логіка побудови IG, POKG та Fusion.
3.  **Feature Engineering Layer:** Перетворення графів у тензори (Stateless).
4.  **Model Layer:** GNN архітектури (PyTorch Geometric).
5.  **Orchestration Layer:** Пайплайни (Train, Eval, Predict).

### Діаграма взаємодії (Inference Flow)

```mermaid
sequenceDiagram
    participant Client (BPMS)
    participant API
    participant IGBuilder
    participant FusionEngine
    participant Neo4j
    participant Model

    Client->>API: POST /predict (trace_events)
    API->>IGBuilder: Build IG(trace)
    IGBuilder-->>API: InstanceGraph
    API->>FusionEngine: Merge(IG)
    FusionEngine->>Neo4j: GetContext(ProcessID, Version)
    Neo4j-->>FusionEngine: SubGraph (Nodes+Embeddings)
    FusionEngine-->>API: FusionGraph (Tensors)
    API->>Model: Forward(FusionGraph)
    Model-->>API: Logits / Time
    API-->>Client: Prediction JSON
```

-----

## 6\. Фізична архітектура та Інфраструктура

Ми використовуємо контейнеризацію для ізоляції середовища.

### 6.1. Container Model (C4)

  * **Container 1: `bpm-core` (Python 3.10):** Містить код додатка, PyTorch, API (FastAPI).
  * **Container 2: `neo4j-db`:** База даних Neo4j з плагіном APOC та GDS.
  * **Container 3 (Optional): `mlflow`:** Сервер для трекінгу експериментів.

### 6.2. Топологія розгортання (Deployment)

  * **Environment:** Docker Compose (для Dev/PoC).
  * **Hardware:** Рекомендовано наявність GPU (CUDA) для прискорення навчання GNN, але підтримується CPU-only режим для інференсу.

-----

## 7\. Дані і Сховище

### 7.1. Моделі даних

  * **Relational (Input):** `ACT_HI_ACTINST` (Camunda) — джерело сирих подій.
  * **Graph (Storage - Neo4j):**
      * Labels: `:Task`, `:Gateway`, `:Role`, `:ProcessVersion`.
      * Relationships: `:SEQUENCE_FLOW`, `:PERFORMED_BY`, `:BELONGS_TO`.
  * **Tensors (Runtime):** `PyG Data Object` (`x`, `edge_index`, `edge_attr`, `y`).

### 7.2. Управління даними

  * **Кешування:** Побудовані графи для навчання кешуються у форматі `.pt` (Pickle) для прискорення повторних експериментів.
  * **Міграції:** Схема POKG є гнучкою (schemaless), версійність підтримується через властивості вузлів.

-----

## 8\. API та Інтеграції

### 8.1. Протоколи

  * **External (Inference):** REST API (JSON) через HTTP.
  * **Internal (Storage):** Bolt Protocol (для Neo4j), JDBC/ODBC (для MSSQL).

### 8.2. Контракт API (Приклад)

**POST /api/v1/predict**

```json
{
  "process_id": "order_process",
  "trace_events": [
    {"activity": "start", "timestamp": "2024-01-01T10:00:00"},
    {"activity": "check_funds", "timestamp": "2024-01-01T10:05:00"}
  ]
}
```

**Response:**

```json
{
  "next_activity": "approve_order",
  "confidence": 0.89,
  "predicted_duration_sec": 120,
  "is_oos": false
}
```

-----

## 9\. Безпека

Для рівня PoC реалізовано базові заходи:

  * **API Authentication:** API Key у заголовку запиту (`X-API-Key`).
  * **Secrets Management:** Паролі до БД (MSSQL, Neo4j) передаються виключно через змінні оточення (`.env`), які не комітяться в Git.
  * **Network:** Neo4j не експонується в зовнішню мережу, доступ тільки всередині Docker network.

-----

## 10\. Observability (Спостережуваність)

  * **Логування:** Структуровані логи (рівні INFO, ERROR, DEBUG) з ротацією файлів. Включають `trace_id` для відладки конкретного прогнозу.
  * **Метрики навчання:** Логування Loss, Accuracy, F1 у консоль та CSV/MLFlow.
  * **Evaluation Reports:** Автоматична генерація Confusion Matrices та графіків залежності точності від довжини префікса.

-----

## 11\. DevOps / Deployment

  * **CI/CD:** GitHub Actions (Linting, Unit Tests).
  * **Containerization:** `Dockerfile` з оптимізацією шарів (окремо встановлення важких бібліотек PyTorch).
  * **Release Strategy:** Версіонування Docker імейджів тегами (v0.1, v0.2).

-----

## 12\. Ризики та Обмеження

1.  **Cold Start Problem:** Якщо в POKG немає статистики для нової версії процесу, точність прогнозу часу може бути низькою (використовуються середні значення).
2.  **Latency:** Побудова Fusion Graph "на льоту" може займати час. *Мітигація:* Кешування ембеддінгів POKG в пам'яті.
3.  **Data Drift:** Якщо реальна поведінка різко змінюється без зміни BPMN, модель потребує перенавчання (Retraining Trigger).

-----

## 13\. Майбутній розвиток (Roadmap)

1.  **Short-term (1-3 міс):**
      * Реалізація повного пайплайну POKG population з BPMN.
      * Інтеграція REST API сервісу.
2.  **Mid-term (3-6 міс):**
      * Додавання механізму *Reinforcement Memory* (динамічне забування).
      * Інтеграція з MLFlow для моніторингу метрик у продакшені.
3.  **Long-term:**
      * Перехід на розподілене навчання (Distributed Training) для великих графів.
      * Автоматична генерація пояснень (Explainable AI) на основі графів (чому обрано цей шлях?).

<!-- end list -->

```
```