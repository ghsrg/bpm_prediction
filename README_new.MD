# BPM Prediction Platform: High-Level Design (HLD)

**bpm_prediction** — це науково-дослідницька платформа для прогнозування поведінки бізнес-процесів (наступна активність, час виконання, аномалії, результат) з використанням Графових Нейронних Мереж (GNN).

Система реалізує концепцію **Fusion Graph** — динамічного об'єднання фактичного стану процесу (Log-based) зі структурними та організаційними знаннями (Knowledge Graph), що дозволяє підвищити точність прогнозів у складних, багатоверсійних процесах.

- [BPM Prediction Platform: High-Level Design (HLD)](#bpm-prediction-platform-high-level-design-hld)
  - [1. Глосарій та Концептуальна Модель](#1-глосарій-та-концептуальна-модель)
    - [1.1. POKG (Process Organizational Knowledge Graph)](#11-pokg-process-organizational-knowledge-graph)
    - [1.2. Instance Graph (IG)](#12-instance-graph-ig)
    - [1.3. Fusion Graph (FG)](#13-fusion-graph-fg)
  - [2. Архітектура Системи (Clean Architecture)](#2-архітектура-системи-clean-architecture)
  - [3. Деталізація Компонентів](#3-деталізація-компонентів)
    - [Layer 1: Data Access (Adapters)](#layer-1-data-access-adapters)
    - [Layer 2: Graph Construction (Core Logic)](#layer-2-graph-construction-core-logic)
    - [Layer 3: Feature Engineering (Config-Driven)](#layer-3-feature-engineering-config-driven)
    - [Layer 4: Model Zoo](#layer-4-model-zoo)
    - [Layer 5: Orchestration \& Experiments](#layer-5-orchestration--experiments)
  - [4. Сценарії Використання (Workflows)](#4-сценарії-використання-workflows)
    - [4.1. POKG Initialization (Cold Start)](#41-pokg-initialization-cold-start)
    - [4.2. Scientific Benchmarking (Написання статті)](#42-scientific-benchmarking-написання-статті)
    - [4.3. Production Inference](#43-production-inference)
  - [5. Структура Репозиторію](#5-структура-репозиторію)
  - [6. Технічний Стек](#6-технічний-стек)

---



## 1. Глосарій та Концептуальна Модель

Система будується навколо трьох ключових графових абстракцій. Розуміння їхньої взаємодії є фундаментом архітектури.

### 1.1. POKG (Process Organizational Knowledge Graph)
**"Статичний" фундамент знань.** Це граф, що зберігається у базі даних **Neo4j** і описує, як процес *повинен* виконуватися та хто за нього відповідає.

* **Формалізація:** $POKG = \langle V, E, A, T, \theta \rangle$
* **Вузли ($V$):**
    * `ProcessElement`: Task, Gateway, Event (з BPMN).
    * `OrgElement`: Role, User, Unit, Resource.
    * `PolicyElement`: Business Rule, SLA Constraint.
    * `VersionNode`: Вузол-якір для підтримки версійності (v1, v2).
* **Зв'язки ($E$):**
    * `SEQUENCE_FLOW`: Нормативний перехід (A -> B).
    * `PERFORMED_BY`: Зв'язок Task -> Role/Unit.
    * `GOVERNED_BY`: Зв'язок Task -> Policy.
* **Час ($T$):** Атрибути `valid_from`, `valid_to` на ребрах для підтримки еволюції процесу (Concept Drift).
* **Статистика ($\theta$):** Накопичені історичні дані на ребрах (ймовірність переходу, середній час виконання), що оновлюються офлайн-джобами.

### 1.2. Instance Graph (IG)
**Динамічне відображення виконання.** Граф, що будується "на льоту" з логів подій для конкретного кейсу (Trace).

* **Джерела:**
    * *Науковий режим:* Файли `.xes` (Event Logs).
    * *Промисловий режим:* SQL/REST API (Runtime data з Camunda).
* **Вузли:** Конкретні події (Events), що вже відбулися.
* **Ребра:**
    * `DIRECTLY_FOLLOWS`: Хронологічна послідовність.
    * `TEMPORAL_EDGE`: Зв'язки між подіями, що близькі у часі.

### 1.3. Fusion Graph (FG)
**Гібридний об'єкт для GNN.** Це результат проєкції IG на POKG. Саме цей граф перетворюється на тензори для нейромережі.

* **Логіка Fusion:**
    1.  Береться вузол події з IG (напр., "Task A, completed by User John").
    2.  Знаходиться відповідний вузол у POKG (BPMN Task "Task A").
    3.  Вектор ознак вузла формується як конкатенація:
        * *Динамічні фічі:* Duration, Timestamp (з IG).
        * *Структурні фічі:* Node2Vec embedding вузла з POKG.
        * *Орг-фічі:* Embedding ролі User John (з POKG).
        * *Статистичні фічі:* Історична частота цього кроку (з POKG).

---

## 2. Архітектура Системи (Clean Architecture)

Проєкт поділено на 5 ізольованих шарів. Дані рухаються строго зверху вниз: від сирих джерел до моделей.

```mermaid
flowchart TD
    subgraph Data Sources
        XES[XES / CSV Logs]
        SQL[Camunda SQL]
        BPMN[BPMN XML]
    end

    subgraph Layer 1: Data Access
        LA[Log Adapter]
        BA[BPMN Parser]
        NA[Neo4j Connector]
    end

    subgraph Layer 2: Graph Engine
        PB[POKG Builder]
        IGB[Instance Graph Builder]
        FB[Fusion Builder]
    end

    subgraph Layer 3: Feature Engineering
        FP[Feature Preprocessor]
        Conf[features.yaml]
    end

    subgraph Layer 4: Model Zoo
        GNN[GNN Models]
        Heads[Prediction Heads]
    end

    subgraph Layer 5: Orchestration
        Exp[Experiment Runner]
        Inf[Inference Service]
    end

    XES --> LA
    SQL --> LA
    BPMN --> BA
    
    BA --> PB
    PB --> NA
    NA --> FB
    
    LA --> IGB
    IGB --> FB
    
    FB --> FP
    Conf -.-> FP
    FP --> GNN
    GNN --> Heads
    Heads --> Exp
    Heads --> Inf
````

-----

## 3\. Деталізація Компонентів

### Layer 1: Data Access (Adapters)

Цей шар абстрагує систему від форматів даних.

  * **`LogAdapter`:** Інтерфейс для читання логів.
      * *Implementations:* `XESAdapter`, `CamundaSQLAdapter`, `CSVAdapter`.
      * *Output:* Pandas/Polars DataFrame зі стандартизованими колонками (`case_id`, `activity`, `timestamp`, `resource`, `lifecycle`).
  * **`BPMNParser`:** Розбирає XML файли `.bpmn`. Витягує топологію (nodes, flows), типи гейтвеїв та атрибути для наповнення POKG.
  * **`Neo4jConnector`:** Обгортка над драйвером БД для виконання Cypher-запитів (збереження графа, вибірка контексту).

### Layer 2: Graph Construction (Core Logic)

Тут живе основна бізнес-логіка графових перетворень.

  * **`POKG Builder`:**
      * Завантажує BPMN-структуру в Neo4j.
      * Створює версійні зрізи (Version Nodes).
      * Запускає *Offline Statistics Update* (оновлення ваг ребер на основі історичних логів).
  * **`IG Builder`:**
      * Приймає трейс (список подій).
      * Формує `NetworkX` граф виконання (вузли + часові ребра).
  * **`Fusion Builder`:**
      * Реалізує алгоритм злиття (Merge).
      * Робить запит у Neo4j: *"Для процесу X версії Y дай мені ембеддінги всіх активностей та ролей"*.
      * Збагачує `IG` отриманими даними.

### Layer 3: Feature Engineering (Config-Driven)

Шар перетворення графів у тензори PyTorch. Ключова особливість — **декларативна конфігурація**. Код не містить хардкоду нормалізації.

**Файл `config/processes/<proc_name>/features.yaml`:**

```yaml
# Конфігурація фіч для Fusion Graph
node_features:
  # Динамічні атрибути (з логу)
  - input: "DURATION"
    type: "numerical"
    processing: { method: "log_norm", fill_na: 0 }
  
  # Контекстні атрибути (з POKG через Fusion)
  - input: "user_role_id"
    type: "categorical"
    processing: { method: "embedding", params: { vocab_size: 50, dim: 16 } }
  
  - input: "bpmn_node_type" # (Task, XOR-Gateway, etc.)
    type: "categorical"
    processing: { method: "onehot", params: { num_classes: 8 } }

edge_features:
  - input: "historical_transition_prob" # Статистика з POKG
    type: "numerical"
    processing: { method: "minmax" }
```

  * **`FeaturePreprocessor`:** Клас, який зчитує цей YAML, навчається (Fit) на тренувальному датасеті (зберігає min/max, словники) і трансформує (Transform) графи у об'єкти `torch_geometric.data.Data`.

### Layer 4: Model Zoo

Реалізація моделей на PyTorch Geometric.

  * **`BaseModel`:** Абстрактний клас (інтерфейси `forward`, `training_step`, `validation_step`).
  * **Backbones (Енкодери):**
      * `GCN`, `GraphSAGE`, `GAT` (статичні графи).
      * `TGAT`, `TemporalGAT` (з урахуванням часу Time2Vec).
      * `DeepGCN` (для глибоких графів).
  * **Heads (Декодери):**
      * `NextActivityHead` (Linear + Softmax).
      * `TimePredictionHead` (Linear + ReLU).
      * `AnomalyDetectionHead` (Autoencoder structure).

### Layer 5: Orchestration & Experiments

  * **`ExperimentRunner`:** Читає `experiment.yaml`, запускає цикл навчання для набору моделей та параметрів (Grid Search для науки).
  * **`TrainPipeline`:** Стандартний цикл навчання (Epochs, Batching, Early Stopping, Checkpointing).
  * **Metrics & Reporting:** Розрахунок Accuracy, F1, MAE, а також специфічних метрик (Out-of-Scope Rate). Генерація діаграм для статей.

-----

## 4\. Сценарії Використання (Workflows)

### 4.1. POKG Initialization (Cold Start)

Цей процес виконується один раз при додаванні нового процесу або його версії.

1.  Завантажується `.bpmn` файл.
2.  `BPMNParser` розбирає структуру.
3.  `POKGBuilder` створює вузли в Neo4j, лінкує їх до версії.
4.  (Опційно) `LogAdapter` проганяє історичні логи для розрахунку статистики переходів, яка записується у властивості ребер POKG.

### 4.2. Scientific Benchmarking (Написання статті)

Мета: Порівняти Logs-only (Baseline) vs BPMN-aware (Fusion) підходи.

1.  В `experiment.yaml` задаються режими: `[logs_only, bpmn_fusion]`.
2.  `IGBuilder` створює графи з XES.
3.  У режимі `bpmn_fusion`, `FusionBuilder` підтягує структуру з POKG.
4.  `ExperimentRunner` тренує моделі на обох наборах даних.
5.  Генеруються звіти: "Accuracy vs Prefix Length", "OOS Rate Comparison".

### 4.3. Production Inference

1.  Зовнішня система (BPMS) надсилає поточний трейс (JSON).
2.  Створюється IG.
3.  `FusionBuilder` запитує Neo4j про актуальний контекст для цього Process ID.
4.  Завантажується навчена модель (Checkpoint).
5.  Повертається прогноз: `{ "next_activity": "Task_B", "confidence": 0.92, "time_remaining": 1200s }`.

-----

## 5\. Структура Репозиторію

```text
bpm_prediction/
├── config/                     # Конфігураційні файли
│   ├── global/                 # Налаштування БД, логування
│   └── processes/              # Параметри для конкретних процесів
│       └── order_process/
│           ├── features.yaml   # Які фічі використовувати
│           └── model.yaml      # Гіперпараметри моделі
├── data/                       # Локальне сховище
│   ├── raw/                    # Вхідні XES, BPMN
│   └── processed/              # Кешовані тензори (.pt)
├── src/
│   ├── adapters/               # LogAdapter, Neo4jClient
│   ├── core/                   # Базові класи та типи
│   ├── graph/                  # IGBuilder, FusionBuilder, POKGManager
│   ├── features/               # FeaturePreprocessor, Tokenizers
│   ├── models/                 # GNN архітектури (PyG)
│   ├── pipelines/              # Train, Eval loops
│   └── utils/                  # Візуалізація, метрики
├── experiments/                # Описи експериментів (YAML)
├── notebooks/                  # Jupyter для аналізу результатів
└── main.py                     # CLI точка входу
```

-----

## 6\. Технічний Стек

  * **Core:** Python 3.10+
  * **Deep Learning:** PyTorch, PyTorch Geometric (PyG), PyTorch Lightning (optional for scaling).
  * **Graph DB:** Neo4j + Graph Data Science (GDS) Library.
  * **Data Analysis:** Pandas, Polars, NetworkX.
  * **Visualization:** Plotly (інтерактивні графіки), Matplotlib (графіки для статей).
  * **Config:** Hydra або OmegaConf (робота з YAML).

<!-- end list -->

```
```