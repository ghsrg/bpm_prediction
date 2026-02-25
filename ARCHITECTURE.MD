# ARCHITECTURE.MD

## 1) Призначення
Цей документ описує цільову архітектуру `bpm_prediction` і синхронізований з:
- `ARCHITECTURE_GUIDELINES.MD` (головний набір правил);
- `GLOSSARY.MD` (канон термінів);
- `VARIABLES.MD` (канон позначень і змінних).

Документ фіксує реалізаційний профіль системи, а не лише загальні принципи.

Документ фіксує реалізаційний профіль системи, а не лише загальні принципи.

## 2) Mode selected
**Обраний режим для поточної хвилі реалізації:** `ENTERPRISE_POC`.

Причина вибору:
1. Потрібен працюючий PoC в існуючому enterprise-контурі.
2. Потрібно зберегти наукову відтворюваність і трасованість для розділу 4 дисертації.
3. Не будуємо нову інфраструктуру з нуля, але ізолюємо математичне ядро.

---

## 3) Архітектурний стиль (узгоджено)
1. **Clean Architecture (Layered)** — базова модель розділення відповідальностей.
2. **Hexagonal (Ports & Adapters)** — механізм інверсії залежностей для зовнішніх інтеграцій.
3. **Pipeline (Pipes & Filters)** — основний каркас виконання use-case сценаріїв.
4. **Strategy Pattern** — для drift/conditioning/loss у місцях варіативності.

---

## 4) High-Level Architecture (Clean Architecture view)

### 4.1 Domain Layer (Entities + Science Core)
Містить:
- доменні сутності (`EventLog`, `ProcessGraph`, `TensorGraph`, `Prediction`, `DriftSignal`);
- GNN-моделі прогнозування;
- Actor–Critic/Constraint-Critic логіку;
- OOD/drift-логіку і структурний regularization;
- функції втрат (`loss_task`, `loss_reg`) і контроль `beta_relax`.

Domain оперує лише тензорами/доменними DTO і **не має прямих залежностей** від Neo4j, MSSQL, MLflow, REST/CLI.

### 4.2 Application Layer (Use Cases)
- Use cases: `Train Pipeline`, `Inference`, `Drift Monitoring`, `Evaluation`;
- оркестрація сценаріїв `prepare/build_graph/train/evaluate/infer`;
- виклик портів доступу до інфраструктури;
- контроль режимів `semaphore_mode` (`green`/`yellow`/`red`).

### 4.3 Interface Adapters Layer
- **Camunda SQL Adapter**: читання Camunda7 MSSQL.
- **XES Reader Adapter**: імпорт XES логів.
- **Business Data Adapter**: читання зовнішніх бізнес-даних (за наявності).
- **Neo4j Gateway**: доступ до EPOKG/POKG.
- **MLflow Adapter**: трекінг метрик/артефактів.
- **Report Writer Adapter**: генерація звітів.
- **REST/CLI Controllers**: вхідні контролери для use cases.

### 4.4 Frameworks & Drivers (Infrastructure)
- MSSQL Camunda 7 DB;
- XES Files Storage;
- External Business Data System;
- Neo4j Storage;
- MLflow Tracker;
- Research Reports Storage;
- REST API/CLI runtime.

---

## 5) Pipeline stages (обов'язковий каркас)
1. **Ingestion** — читання логу процесу/батча.
2. **Graph Builder (EPOKG context)** — побудова IG і зв'язок з EPOKG.
3. **Tensor Adapter** — перехід до `torch_geometric.data.Data`.
4. **Core ML** — inference/train крок.
5. **Evaluation** — метрики якості/OOS.
6. **Drift Monitor** — обчислення drift-індикаторів.
7. **Inference** — фінальний прогноз + режим семафора.

---

## 6) Data Flow (runtime)
1. Запит надходить через CLI або REST API з `process_instance_id` і source-type.
2. Source-aware adapter (`camunda7_mssql_adapter` / `xes_adapter` / інший) читає сирі події.
3. `Data Converter` нормалізує вхід у єдиний контракт `EventLogBatch`.
4. `event_stream` + process context формують `process_graph`.
5. З EPOKG додається нормативний контекст і будується Fusion Graph.
6. Tensor Adapter формує `node_features`, `adj_matrix`/`edge_index`.
7. Core обчислює логіти/прогноз + uncertainty/drift сигнали.
8. `AllowMask` відсікає недопустимі переходи (OOS masking).
9. Reliability Semaphore обчислює `reliability_score` і встановлює `semaphore_mode`.
10. Prediction/metadata логуються в tracker і повертаються у викликаючий контур (CLI/API).

---

## 7) Reliability Semaphore (узгодження порогів)

### 7.1 Вхідні сигнали
- `wasserstein_drift` (`\mathcal{W}_1`) — структурний дрейф;
- `data_drift_score` (`D_data`);
- `concept_drift_score` (`D_concept`);
- додаткові сигнали невизначеності моделі.

### 7.2 Ієрархія порогів
1. `ood_threshold` (`\tau_{ood}`) — локальний поріг структурного OOD (по `\mathcal{W}_1`).
2. `warning_threshold` (`\tau_{warn}`) — агрегований жовтий режим.
3. `critical_threshold` (`\tau_{crit}`) — агрегований червоний режим.

Інваріант: `\tau_{warn} < \tau_{crit}`.

### 7.3 Поведінка за режимами
- **green:** стандартний inference.
- **yellow:** обмежений inference + сигнал на адаптацію.
- **red:** Human-in-the-Loop / призупинка авто-рішення.

---

## 8) Adaptation strategy
1. **Zero-Shot Structural Adaptation** — для мінорного дрейфу (оновлення графового контексту без перенавчання).
2. **Few-Shot Adaptation (Controlled Fine-Tuning)** — для глибокого OOD:
   - накопичення буфера до `min_buffer_size` (`n_min`);
   - контрольоване донавчання;
   - валідація перед поверненням у автоматичний inference.

---

## 9) Knowledge Infusion Operator
`knowledge_infusion_operator` (`\Gamma`) реалізується як `nn.Module` (або `nn.Sequential`), а не як статичний вектор.

Базова форма:
```python
fusion_repr = self.knowledge_infusion_operator(local_state_vec, global_context_vec)
```

де:
- `local_state_vec` (`h_\sigma`) — локальний стан кейсу;
- `global_context_vec` (`c_\sigma`) — глобальний контекст із EPOKG.

---

## 10) Data contracts (мінімально обов'язкові)
Між кожними stage визначаються:
1. Input schema.
2. Output schema.
3. Metadata schema.

Не допускається implicit format.

---

## 11) Versioning model
Кожний артефакт має містити:
- `dataset_id`
- `schema_version`
- `process_version` (`kappa`/`version_id`)
- `model_version`
- `git_commit`
- `experiment_id`

Додатково для експериментів:
- копії конфігів (`model.yaml`, `features.yaml`, `training.yaml`);
- `preprocessor_state` для відтворюваності inference.

---

## 12) Observability
Мінімальний набір логування на `inference/evaluate`:
- `data_drift_score`
- `concept_drift_score`
- `wasserstein_drift`
- `reliability_score`
- `semaphore_mode`
- Accuracy/F1/OOS (де застосовно)

Логування виконується на кожному batch (або іншому атомарному кроці).

---

## 13) Enterprise integration strategy (PoC)
Рекомендований контур запуску:
- CLI: `--mode prepare/build_graph/train/evaluate/infer`
- REST API (PoC): `POST /api/v1/infer-by-instance` з `process_instance_id` та `source_type`.

Правило маршрутизації даних:
1. API/CLI викликає orchestration service.
2. Orchestration service обирає потрібний data adapter (`camunda7_mssql_adapter` або `xes_adapter`).
3. Дані проходять через `Data Converter` до єдиного контракту.
4. Далі запускається стандартний pipeline до inference.

Інтеграція в існуючу екосистему:
- MSSQL Camunda 7 (обов'язково для PoC);
- XES-файли в директорії даних для відкритих датасетів;
- Neo4j (за наявності);
- існуючий трекер/registry (якщо є).

Не допускається:
- прямий зв'язок raw-джерела з EPOKG без конвертора контракту;
- мікросервісний оверінжиніринг;
- побудова повного MLOps-stack у PoC-фазі.

---

## 14) Risks & trade-offs
1. **Trade-off:** чиста ізоляція Core може уповільнювати першу ітерацію, але знижує технічний борг.
2. **Risk:** неявні схеми між stage спричинять нестабільні inference-помилки.
3. **Risk:** відсутність трекінгу `kappa` руйнує валідність експериментів дрейфу.
4. **Risk:** змішування adapter-логіки всередині Core ламає переносимість і тестованість.

---

## 15) Next architectural step
Наступний крок: деталізувати `Contracts & Abstract Base Classes` для портів/адаптерів і формалізувати схеми потоків даних для кожного pipeline-stage.

---

## 16) Clean Architecture (Layered View)

```mermaid
flowchart TB

%% ================================
%% OUTER LAYER — Frameworks & Drivers
%% ================================

subgraph Frameworks_and_Drivers["Frameworks & Drivers (Infrastructure)"]
    CamundaDB[(MSSQL Camunda 7 DB)]
    XES[(XES Files Storage)]
    BusinessSys[(External Business Data System)]
    Neo4j[(Neo4j EPOKG Storage)]
    MLflow[(MLflow Tracker)]
    Reports[(Research Reports Storage)]
    REST[(REST API)]
    CLI[(CLI Runner)]
end

%% ================================
%% INTERFACE ADAPTERS
%% ================================

subgraph Interface_Adapters["Interface Adapters"]
    SQLReader[Camunda SQL Adapter]
    XESReader[XES Reader Adapter]
    BusinessAdapter[Business Data Adapter]

    GraphGateway[Neo4j Gateway]
    TrackerAdapter[MLflow Adapter]
    ReportAdapter[Report Writer Adapter]

    APIController[REST Controller]
    CLIController[CLI Controller]
end

%% ================================
%% APPLICATION LAYER (USE CASES)
%% ================================

subgraph Application_Layer["Application / Use Cases"]
    TrainUC[Train Pipeline Use Case]
    InferUC[Inference Use Case]
    DriftUC[Drift Monitoring Use Case]
    EvalUC[Evaluation Use Case]
end

%% ================================
%% DOMAIN (ENTITIES + SCIENCE CORE)
%% ================================

subgraph Domain_Layer["Domain Core (Pure Logic / Math)"]

    %% Core domain models
    EventLog[EventLog Entity]
    EnrichedCase[Case Enrichment Model]
    ProcessGraph[Process Graph Model]
    TensorGraph[TensorGraph Representation]
    Prediction[Prediction Entity]
    DriftSignal[Drift Signal Model]

    %% Scientific components
    EPOKGBuilder[EPOKG Graph Builder]
    FeatureEncoder[Feature Encoding Pipeline]
    AgentCritic[Agent–Critic GNN Module]
    DriftDetector[Drift Detection Module]
    ReliabilityPolicy[Reliability & Semaphore Policy]
end

%% ================================
%% INFRASTRUCTURE IMPLEMENTS PORTS
%% ================================

CamundaDB --> SQLReader
XES --> XESReader
BusinessSys --> BusinessAdapter
Neo4j --> GraphGateway
MLflow --> TrackerAdapter
Reports --> ReportAdapter
REST --> APIController
CLI --> CLIController

%% ================================
%% ADAPTERS USED BY APPLICATION
%% ================================

SQLReader --> TrainUC
XESReader --> TrainUC
BusinessAdapter --> TrainUC

SQLReader --> InferUC
BusinessAdapter --> InferUC

GraphGateway --> TrainUC
GraphGateway --> InferUC

TrackerAdapter --> TrainUC
TrackerAdapter --> DriftUC
TrackerAdapter --> EvalUC

ReportAdapter --> EvalUC

APIController --> InferUC
CLIController --> TrainUC

%% ================================
%% APPLICATION CALLS DOMAIN
%% ================================

TrainUC --> EventLog
TrainUC --> EnrichedCase
TrainUC --> EPOKGBuilder
TrainUC --> FeatureEncoder
TrainUC --> TensorGraph
TrainUC --> AgentCritic

InferUC --> EventLog
InferUC --> EnrichedCase
InferUC --> EPOKGBuilder
InferUC --> FeatureEncoder
InferUC --> TensorGraph
InferUC --> AgentCritic
InferUC --> ReliabilityPolicy

DriftUC --> DriftDetector
DriftUC --> DriftSignal

EvalUC --> Prediction

%% ================================
%% DOMAIN INTERNAL FLOW
%% ================================

EventLog --> EnrichedCase
EnrichedCase --> EPOKGBuilder
EPOKGBuilder --> ProcessGraph
ProcessGraph --> FeatureEncoder
FeatureEncoder --> TensorGraph
TensorGraph --> AgentCritic
AgentCritic --> Prediction
Prediction --> ReliabilityPolicy
Prediction --> DriftDetector
DriftDetector --> DriftSignal
```

---

## 17) Component Architecture (Hexagonal View)

```mermaid
flowchart TB

%% ================================
%% 1. FRAMEWORKS & DRIVERS (OUTER)
%% ================================

subgraph Frameworks["Frameworks & Drivers"]
    CamundaDB[(MSSQL Camunda 7 DB)]
    XESFiles[(XES Storage)]
    Neo4j[(Neo4j EPOKG)]
    MLflow[(MLflow)]
    Reports[(Reports Storage)]
    REST[(REST API)]
    CLI[(CLI)]
end

%% ================================
%% 2. INTERFACE ADAPTERS (HEXAGON)
%% ================================

subgraph Adapters["Interface Adapters (Ports & Adapters)"]

    %% Outbound Adapters
    SQLAdapter[Camunda SQL Adapter]
    XESAdapter[XES Reader Adapter]
    GraphAdapter[Neo4j Adapter]
    TrackerAdapter[MLflow Adapter]
    ReportAdapter[Report Adapter]

    %% Inbound Adapters
    APIController[REST Controller]
    CLIController[CLI Controller]
end

%% ================================
%% 3. APPLICATION (USE CASES)
%% ================================

subgraph Application["Application / Use Cases"]

    TrainUC[Train Use Case]
    InferUC[Inference Use Case]
    DriftUC[Drift Monitoring Use Case]
    EvalUC[Evaluation Use Case]
end

%% ================================
%% 4. DOMAIN (PURE CORE)
%% ================================

subgraph Domain["Domain Core (Pure Logic)"]

    EventNormalizer[EventLog Normalizer]
    GraphBuilder[EPOKG Graph Builder]
    TensorAdapter[Tensor Adapter]
    AgentCritic[Agent-Critic GNN]
    DriftDetector[Drift Detector]
    ReliabilityPolicy[Reliability Policy]
end

%% ===================================
%% INFRA IMPLEMENTS ADAPTERS
%% ===================================

CamundaDB --> SQLAdapter
XESFiles --> XESAdapter
Neo4j --> GraphAdapter
MLflow --> TrackerAdapter
Reports --> ReportAdapter
REST --> APIController
CLI --> CLIController

%% ===================================
%% ADAPTERS CALL APPLICATION
%% ===================================

APIController --> InferUC
CLIController --> TrainUC

%% ===================================
%% APPLICATION USES ADAPTERS (OUTBOUND)
%% ===================================

TrainUC --> SQLAdapter
TrainUC --> XESAdapter
TrainUC --> GraphAdapter
TrainUC --> TrackerAdapter

InferUC --> SQLAdapter
InferUC --> GraphAdapter
InferUC --> TrackerAdapter

DriftUC --> TrackerAdapter
EvalUC --> ReportAdapter
EvalUC --> TrackerAdapter

%% ===================================
%% APPLICATION CALLS DOMAIN
%% ===================================

TrainUC --> EventNormalizer
TrainUC --> GraphBuilder
TrainUC --> TensorAdapter
TrainUC --> AgentCritic

InferUC --> EventNormalizer
InferUC --> GraphBuilder
InferUC --> TensorAdapter
InferUC --> AgentCritic
InferUC --> ReliabilityPolicy

DriftUC --> DriftDetector
EvalUC --> AgentCritic

%% ===================================
%% DOMAIN INTERNAL FLOW
%% ===================================

EventNormalizer --> GraphBuilder
GraphBuilder --> TensorAdapter
TensorAdapter --> AgentCritic
AgentCritic --> DriftDetector
AgentCritic --> ReliabilityPolicy
```

### Ключова вводна (операційна модель)
Архітектура підтримує **єдині операційні сценарії** (`Train`, `Inference`, `Evaluation`), але допускає багатовимірну конфігурацію дослідницьких режимів через стратегії:
- структурного кодування;
- латентного кондиціонування версій;
- детекції дрейфу;
- політики надійності.

---

## 18) Data Flow (Use Case Scenarios)

### 18.1 TRAIN DATA FLOW
```mermaid
sequenceDiagram
    participant CLI
    participant TrainUC
    participant DataAdapter
    participant BusinessAdapter
    participant DomainPipeline
    participant GNN
    participant Tracker

    CLI->>TrainUC: run(dataset_id, ExperimentConfig)

    TrainUC->>DataAdapter: read_logs(source_type)
    TrainUC->>BusinessAdapter: enrich_cases()

    TrainUC->>DomainPipeline: normalize_events(config.structure_mode)

    DomainPipeline->>DomainPipeline: build_EPOKG(config.structure_mode)

    DomainPipeline->>DomainPipeline: encode_features(config.kappa_conditioning)

    DomainPipeline->>GNN: train_step(config.architecture_type)

    GNN-->>TrainUC: loss, metrics

    TrainUC->>Tracker: log_metrics()
    TrainUC->>Tracker: save_model()
```

### 18.2 INFERENCE (CLI / REST API)
```mermaid
sequenceDiagram
    participant API
    participant InferUC
    participant DataAdapter
    participant BusinessAdapter
    participant DomainPipeline
    participant GNN
    participant Drift
    participant Reliability
    participant Tracker

    API->>InferUC: predict(instance_id, model_version)

    InferUC->>DataAdapter: read_prefix()
    InferUC->>BusinessAdapter: enrich_case()

    InferUC->>DomainPipeline: build_subgraph()

    DomainPipeline->>DomainPipeline: encode_features()

    DomainPipeline->>GNN: forward_pass()

    GNN-->>InferUC: prediction, confidence

    InferUC->>Drift: evaluate(config.drift_strategy)

    InferUC->>Reliability: apply_policy(config.reliability_strategy)

    InferUC->>Tracker: log_inference()

    InferUC-->>API: PredictionDTO
```

### 18.3 EXPERIMENTAL EVALUATION
```mermaid
sequenceDiagram
    participant CLI
    participant EvalUC
    participant Dataset
    participant DomainPipeline
    participant GNN
    participant Metrics
    participant Tracker
    participant Report

    CLI->>EvalUC: evaluate(dataset_id, ExperimentConfig)

    EvalUC->>Dataset: load_test_split()

    loop for each prefix
        EvalUC->>DomainPipeline: build_graph()
        DomainPipeline->>GNN: forward_pass()
        GNN-->>EvalUC: prediction
    end

    EvalUC->>Metrics: compute(Accuracy, MacroF1, AUC, OOS)

    EvalUC->>Tracker: log_metrics()

    EvalUC->>Report: generate_report()
```

### 18.4 RETRAINING FLOW
```mermaid
sequenceDiagram
    participant DriftMonitor
    participant RetrainUC
    participant TrainUC
    participant Tracker

    DriftMonitor->>RetrainUC: trigger_if_threshold()

    RetrainUC->>TrainUC: run(new_dataset, ExperimentConfig)

    TrainUC-->>RetrainUC: new_model_version

    RetrainUC->>Tracker: log_retraining_event()
```

### 18.5 Research Axes (частина `ExperimentConfig`)

```python
class ExperimentConfig:
    structure_mode: Literal["logs_only", "bpmn", "epokg"]
    kappa_conditioning: Literal["none", "static", "latent"]
    drift_strategy: Literal["off", "data", "structural", "wasserstein", "multi"]
    reliability_strategy: Literal["off", "threshold", "adaptive"]
    architecture_type: Literal["single_gnn", "agent_critic", "ablation"]
```

#### Як інтегрується
- `Train` / `Experimental Evaluation` / `Retraining` приймають `ExperimentConfig` явно.
- `Inference` працює або в `production config`, або в `experimental config` (для research).

#### Ключова архітектурна ідея
- **Use Cases не множаться.**
- **Стратегії множаться.**

| Use Case                | Конфігурований? | Мета               |
| ----------------------- | --------------- | ------------------ |
| Train                   | ✅               | Навчання           |
| Inference               | частково        | Runtime prediction |
| Experimental Evaluation | ✅               | Порівняння режимів |
| Retraining              | ✅               | Adaptive update    |

---

## 19) Reliability Semaphore State Machine

```mermaid
stateDiagram-v2
    [*] --> Green_Mode: Start Inference

    Green_Mode --> Yellow_Mode: W_1 > tau_warn
    Yellow_Mode --> Green_Mode: Drift Stabilized

    Yellow_Mode --> Red_Mode: W_1 > tau_crit (Major Drift)
    Green_Mode --> Red_Mode: W_1 > tau_crit

    state Red_Mode {
        [*] --> Suspend_Auto_Inference
        Suspend_Auto_Inference --> Human_in_the_Loop: Route to Expert
        Human_in_the_Loop --> Buffer_Accumulation: Collect n_min samples
        Buffer_Accumulation --> Controlled_Fine_Tuning: Few-Shot Adaptation
        Controlled_Fine_Tuning --> [*]: Validation Passed
    }

    Red_Mode --> Green_Mode: Adaptation Complete (Zero/Few-Shot)
```

---

## 20) Physical Directory Structure
Обов'язкова структура репозиторію для дотримання Clean Architecture:

```text
bpm_prediction/
├── src/
│   ├── core/                  # Domain Math (без інфраструктурних залежностей)
│   │   ├── models/            # GNN, Agent, Critic (PyTorch)
│   │   ├── semaphore/         # OOD math, Wasserstein, Thresholds
│   │   └── interfaces/        # Abstract Base Classes (Ports)
│   ├── adapters/              # Infrastructure
│   │   ├── data/              # camunda7_mssql_adapter, xes_adapter, neo4j_storage
│   │   ├── api/               # rest_api_adapter
│   │   ├── tracking/          # mlflow_tracker
│   │   └── reporting/         # report_writer, chart_exporter
│   ├── pipeline/              # Orchestration (Stages)
│   │   ├── ingestion_router.py
│   │   ├── graph_builder.py
│   │   ├── data_converter.py
│   │   ├── tensor_adapter.py
│   │   ├── trainer.py
│   │   ├── evaluator.py
│   │   └── report_builder.py
│   └── cli.py                 # Entry point (--mode train/evaluate/infer)
├── data/                      # Local cache (.pt files, raw logs)
├── reports/                   # Tables/plots for dissertation section 4
├── tests/
└── ARCHITECTURE.MD
```

---

## 21) Research Reporting & Section 4 Outputs
Для підтримки розділу 4 дисертації в pipeline обов'язковий окремий `Evaluation & Reporting` крок.

### 21.1 Обов'язкові артефакти
1. Таблиці метрик по режимах (Baseline vs Augmented).
2. Графіки динаміки дрейфу (`wasserstein_drift`, `data_drift_score`, `concept_drift_score`).
3. Графіки стабільності семафора (`semaphore_mode` у часі).
4. Порівняльні графіки якості (Accuracy/F1/OOS) для різних `process_version (κ)`.

### 21.2 Вимоги до трасованості звітів
Кожен згенерований звіт має містити metadata:
- `experiment_id`
- `dataset_id`
- `schema_version`
- `process_version (κ)`
- `model_version`
- `git_commit`

### 21.3 Розміщення звітів
- Локально: `reports/`.
- У трекері: artifacts в MLflow (або сумісному tracker).

