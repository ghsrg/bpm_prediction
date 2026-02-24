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
  - [3. Вимоги (Requirements)](#3-вимоги-requirements)
    - [3.1. Функціональні вимоги](#31-функціональні-вимоги)
    - [3.2. Нефункціональні вимоги (NFR)](#32-нефункціональні-вимоги-nfr)
  - [4. Архітектурне бачення](#4-архітектурне-бачення)
    - [4.1. Стиль та Принципи](#41-стиль-та-принципи)
    - [4.2. Technology Stack](#42-technology-stack)
  - [5. Логічна архітектура](#5-логічна-архітектура)
    - [Діаграма взаємодії (Inference Flow)](#діаграма-взаємодії-inference-flow)
      - [Component: Feature Engineering Layer](#component-feature-engineering-layer)
      - [Component: Fusion Engine](#component-fusion-engine)
      - [Component: Model Registry (Inference)](#component-model-registry-inference)
  - [6. Фізична архітектура та Інфраструктура](#6-фізична-архітектура-та-інфраструктура)
    - [6.1. Середовище розгортання (Deployment Topology)](#61-середовище-розгортання-deployment-topology)
  - [7. Дані](#7-дані)
    - [7.1. Сховище (Neo4j)](#71-сховище-neo4j)
    - [7.2. Вхідні дані](#72-вхідні-дані)
  - [8. Інтеграція (Phasing)](#8-інтеграція-phasing)
    - [Етап 1: PoC (Manual / CLI)](#етап-1-poc-manual--cli)
    - [Етап 2: Strategic (REST API)](#етап-2-strategic-rest-api)
  - [9. Observability (Спостережуваність)](#9-observability-спостережуваність)
  - [10. Roadmap](#10-roadmap)


---

## 1. Business Context (Бізнес-контекст)

### 1.1. Основні бізнес-функції
Система виступає інтелектуальним ядром для BPMS і вирішує завдання:
1.  **Predictive Monitoring:** Прогнозування наступної активності (Next Activity) та часу завершення (Time Prediction) для запущених процесів.
2.  **Anomaly Detection:** Виявлення відхилень від нормативного шляху (POKG) у реальному часі.
3.  **Knowledge Consolidation:** Об'єднання розрізнених версій процесів у єдиний граф знань (EPOKG).
4.  **Scientific Benchmarking:** Забезпечення відтворюваних експериментів для порівняння Logs-only vs BPMN-aware підходів.

### 1.2. Зацікавлені сторони (Stakeholders)
* **Бізнес-користувачі:** Отримують ранні попередження про затримки або порушення процесу.
* **Data Scientists:** Використовують платформу як лабораторний стенд для перевірки гіпотез (GAPs).
* **IT/DevOps:** Забезпечують інтеграцію з Camunda та розгортання в корпоративному контурі.

---

## 2. System Context (Контекстна діаграма)

Діаграма відображає місце системи в ландшафті підприємства та її зв'язки.

```mermaid
C4Context
    title System Context Diagram

    Person(researcher, "Researcher / DS", "Runs experiments via PyCharm/CLI, analyzes MLFlow")
    Person(biz_user, "Business User", "Consumes predictions via BPMS UI")

    System(bpm_core, "BPM Prediction Core", "Main Logic: Fusion Graph Engine + GNN Models")

    System_Ext(camunda, "Camunda BPMS", "Source of Runtime Events (SQL) & BPMN Models")
    System_Ext(neo4j, "Neo4j Graph DB", "Stores EPOKG (Process Knowledge & Context)")
    System_Ext(mlflow, "MLFlow (Local)", "Experiment Tracking (Metrics, Params, Artifacts)")
    System_Ext(files, "File Storage", "XES Logs, Cache (.pt), Configs")

    Rel(researcher, bpm_core, "Configures & Runs Pipelines")
    Rel(bpm_core, camunda, "Reads Runtime Data (MSSQL/API)")
    Rel(bpm_core, neo4j, "Queries Context & Updates Stats")
    Rel(bpm_core, mlflow, "Logs Metrics & Models")
    Rel(bpm_core, files, "Reads/Writes Data")
    Rel(camunda, bpm_core, "Sends Prediction Request (Future REST)")
    Rel(bpm_core, biz_user, "Returns Prediction")
````

-----

## 3\. Вимоги (Requirements)

### 3.1. Функціональні вимоги

1.  **Data Ingestion:**
      * Імпорт BPMN моделей (XML) -\> Neo4j.
      * Імпорт логів (XES для науки, MSSQL для PoC).
2.  **Knowledge Graph (POKG):** Підтримка версійності процесів та зв'язків з ресурсами.
3.  **Fusion Engine:** Динамічне збагачення вектора стану процесу (IG) даними з графа знань.
4.  **Prediction:**
      * *Mode 1 (PoC):* Користувач вказує `process_instance_id`, система будує прогноз.
      * *Mode 2 (Target):* REST API приймає запит з ID.
5.  **Observability:** Трекінг всіх запусків та метрик у MLFlow.

### 3.2. Нефункціональні вимоги (NFR)

  * **Environment Constraints:** Робота в ізольованій корпоративній мережі (Windows VM) з обмеженим доступом до інтернету (Proxy).
  * **Reproducibility:** Повна повторюваність експериментів (Fixed Seeds, Config-Driven).
  * **Modularity:** Чітке розділення логіки побудови графів та ML-моделей.

-----

## 4\. Архітектурне бачення

### 4.1. Стиль та Принципи

  * **Modular Monolith (Layered):** Обрано для спрощення розробки та дебагу в умовах локального розгортання.
  * **Config-Driven Development:** Вся конфігурація (фічі, гіперпараметри, шляхи) винесена у YAML файли. Хардкод заборонено.
  * **Separation of Concerns:** Моделі (Neural Networks) нічого не знають про джерела даних (SQL/XES). Вони працюють виключно з тензорами.

### 4.2. Реалізація управлінського циклу (Closed-loop)

Згідно з методологією, архітектура реалізує замкнений контур управління:
* **Sensor (Сенсор):** Адаптер Camunda/XES, що генерує `Instance Graph`.
* **Processor (Агент):** GNN-модель, що формує локальний прогноз.
* **Governor (Критик/Семафор):** * Модуль **Epistemic Caution** (оцінює $\mathcal{W}_1$ відстань).
    * Модуль **Structural Validator** (перевіряє прогноз на відповідність топології POKG).
* **Actuator (Механізм адаптації):** Система донавчання (Continual Learning), що оновлює параметри $\theta$ при детекції дрейфу.

### 4.3. Technology Stack

  * **Python: 3.10
  * **ML: PyTorch 2.x, PyTorch Geometric
  * **Data: Pandas, Polars (для швидкої обробки логів)
  * **Storage: Neo4j 5.x + GDS Library
  * **Connectors: neo4j (driver), pyodbc (MSSQL)
  * **інструмент для трекінгу експериментів MLflow

-----

## 5\. Логічна архітектура

Система складається з 5 шарів (Layers):

1.  **Adapters Layer:** "Перехідники" до зовнішнього світу (MSSQL Connector, Neo4j Driver, XES Reader).
2.  **Graph Engine Layer:** Ядро системи.
      * `IG Builder`: Будує граф виконання.
      * `Fusion Builder`: Реалізує логіку злиття (Merge).
      * `POKG Manager`: Керує оновленням знань у Neo4j.
3. **Semantic Alignment Layer** (Abstract) Для забезпечення розширюваності системи (XES, SAP, Oracle) вводиться шар семантичного вирівнювання:
   * **Internal Canonical ID:** Всі вузли всередині системи оперують уніфікованими хешами активностей.
   * **Mapping Strategy:** Адаптери (CamundaSQL, XES) відповідають за мапінг специфічних імен (наприклад, `UserTask_1` в Camunda) до канонічних назв у `POKG`.
   * **Scope:** Повна автоматизація мапінгу для зовнішніх систем (Fuzzy Matching) винесена за рамки даного проєкту, проте архітектура підтримує ін'єкцію `mapping_table.json`.
4.  **Feature Layer:** Stateless-трансформери даних у тензори (згідно з `features.yaml`).
5 **Model Layer:** GNN архітектури (на базі PyTorch Geometric).
6 **Orchestration Layer:**
      * `Pipelines`: Сценарії навчання/тестування.
      * `Inference Service`: Обробка одиничних запитів.


### Діаграма взаємодії (Inference Flow)

*Показано цільовий потік обробки запиту на прогноз.*

```mermaid
sequenceDiagram
    participant Client as Client (BPMS/CLI)
    participant Ctrl as Inference Controller
    participant Orch as Orchestrator
    participant Fusion as Fusion Engine
    participant Neo4j as Neo4j DB
    participant Reg as Model Registry
    participant Feat as Feat. Preprocessor

    Note over Client, Reg: 1. Initialization (Lazy)
    Client->>Ctrl: POST /predict (Trace + ProcessID)
    Ctrl->>Orch: Handle Request
    
    Orch->>Reg: GetModel(ProcessID)
    alt Model not loaded
        Reg->>Reg: Load Checkpoint (.pt)
        Reg->>Reg: Load Preprocessor State (.json)
    end
    Reg-->>Orch: Model Instance + Preprocessor
    
    Note over Orch, Neo4j: 2. Data Gathering & Fusion
    par Prepare Inputs
        Orch->>Fusion: Build IG(Runtime Trace)
        Fusion->>Neo4j: Fetch Context & ValidTransitions
        Neo4j-->>Fusion: Subgraph + Mask
    end
    
    Fusion->>Fusion: Merge (IG + POKG)
    Fusion-->>Orch: Raw FusionGraph (NetworkX) + AllowMask
    
    Note over Orch, Feat: 3. Feature Engineering
    Orch->>Feat: Transform(RawGraph, State)
    Feat-->>Orch: Tensors (X, Edge_Index)
    
    Note over Orch, Reg: 4. Prediction & Logic
    Orch->>Reg: Forward(Tensors)
    Reg-->>Orch: Raw Logits / Time
    
    Orch->>Orch: Apply AllowMask (Filter OOS)
    Orch->>Orch: Softmax -> Final Prediction
    
    Orch-->>Ctrl: Result JSON
    Ctrl-->>Client: Response
```

#### Component: Feature Engineering Layer

  * **Відповідальність:** Перетворення атрибутів графа у тензори.
  * **Принцип роботи:** Stateless-трансформація, яка використовує стан (`min/max`, `vocab`), зафіксований під час навчання.
  * **Artifacts:** `preprocessor_state.json` — зберігається разом з версією моделі. Це гарантує, що інференс математично ідентичний навчанню.

#### Component: Fusion Engine

  * **Відповідальність:** Побудова графа для моделі та валідація переходів.
  * **OOS Handling:** Разом із графом повертає `AllowMask` — маску допустимих переходів згідно з поточною версією процесу в POKG. Це дозволяє фільтрувати галюцинації моделі.

#### Component: Model Registry (Inference)

  * **Стратегія:** Lazy Loading (LRU Cache).
  * **Поведінка:** Моделі завантажуються в RAM лише при першому зверненні до конкретного `process_id`. Це дозволяє підтримувати десятки процесів на обмеженому обладнанні (Dev VM).

-----

## 6\. Фізична архітектура та Інфраструктура

### 6.1. Середовище розгортання (Deployment Topology)

**Поточний стан (Dev / PoC):**

  * **Infrastructure:** Windows VM.
  * **Execution:**
      * Python-код запускається через Local Interpreter / venv.
      * **Neo4j:** Desktop/Server версія, встановлена локально (localhost:7687).
      * **MLFlow:** Локальний запуск (`mlflow ui`), збереження артефактів у локальну папку `./mlruns`.
  * **Integration:** Прямий доступ до MSSQL через sql.

**Цільовий стан (Production / Scaling):**

  * **Containerization:** Docker Compose.
      * Service `app`: Python API (FastAPI).
      * Service `db`: Neo4j Container.
      * Service `tracking`: MLFlow Server + Postgres (meta).

-----

## 7\. Дані

### 7.1. Сховище (Neo4j)

POKG моделюється як Labeled Property Graph.

  * **Nodes:** `Task`, `Gateway`, `StartEvent`, `EndEvent`, `Role`, `Lane`.
  * **Edges:** `FLOWS_TO` (Control Flow), `CANDIDATE_GROUP` (Org link).
  * **Properties:** `activity_id`, `version_tag`, `avg_duration`, `execution_count`.

### 7.2. Вхідні дані

  * **Науковий режим:** XES файли.
  * **PoC режим:** Таблиці Camunda (`ACT_HI_ACTINST`, `ACT_RE_PROCDEF`).

-----

## 8\. Інтеграція (Phasing)

### Етап 1: PoC (Manual / CLI)

  * **Вхід:** Оператор вказує ID активного процесу в конфігурації або через CLI аргумент.
  * **Процес:** Система підключається до MSSQL, витягує історію цього процесу, будує граф і видає прогноз у консоль/лог.

### Етап 2: Strategic (REST API)

  * **Вхід:** HTTP POST запит.
  * **API Spec (Draft):**
    ```http
    POST /api/v1/prediction
    Content-Type: application/json
    {
        "process_instance_id": "a1b2-c3d4...",
        "model_version": "production_v1"
    }
    ```
  * **Вихід:** JSON з прогнозом та confidence score.

-----

## 9\. Observability (Спостережуваність)

Через обмеження мережі (Proxy) використовується локальний стек.

1.  **Logging:**
      * Бібліотека `loguru`.
      * Логи пишуться у файли з ротацією (`logs/app.log`) та дублюються в stdout.
2.  **Experiment Tracking (MLFlow):**
      * Логування параметрів (`learning_rate`, `hidden_dim`, `fusion_mode`).
      * Логування метрик (`accuracy`, `f1_score`, `oos_rate`).
      * Збереження навчених моделей (Artifacts) для відтворюваності.

-----

## 10\. Roadmap

  * **Phase 1 (Current):** Реалізація Fusion Pipeline на локальній машині. Навчання на XES. Тест через СLI.
  * **Phase 2:** Розгортання REST API. 
  * **Phase 3:** Контейнеризація та перехід на серверне середовище.

<!-- end list -->

```
```