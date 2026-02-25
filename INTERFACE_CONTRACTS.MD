# INTERFACE_CONTRACTS.MD

## Статус
`IN_PROGRESS (DOC-007)`

## 1) Призначення
Документ фіксує **мінімальний HLD-комплект контрактів** для DOC-007:
- порти (interfaces) між Application/Core/Adapters;
- базові Abstract Base Classes (ABC);
- єдині DTO/схеми `input/output/metadata` для pipeline-stage.

Базовий канон змісту: `Дисертація Коротенко.docx`.

---

## 2) Scope (Sprint A / DOC-007)
Покриваємо обов'язкові стадії:
1. Ingestion
2. Graph Builder
3. Tensor Adapter
4. Core ML
5. Evaluation
6. Drift Monitor
7. Inference

---

## 3) Canonical DTO

### 3.1 `RunMetadataDTO` (run-level)
```python
class RunMetadataDTO(TypedDict):
    experiment_id: str
    git_commit: str
```

### 3.2 `StageMetadataDTO` (stage-level)
```python
class StageMetadataDTO(TypedDict, total=False):
    # required for all stages
    dataset_id: str
    schema_version: str
    process_version: str   # κ

    # optional/conditional per stage
    model_version: str
    checkpoint_id: str
    epokg_snapshot_id: str
    graph_build_version: str
    feature_config_version: str
    drift_strategy_version: str
    evaluation_profile: str
    routing_decision: str
```

### 3.3 `EventRecord`
```python
class EventRecord(TypedDict):
    case_id: str
    activity_id: str
    event_time: str   # ISO-8601
    resource_id: str | None
    process_version: str  # κ
    payload: dict[str, Any]
```

### 3.4 `EventLogBatch`
```python
class EventLogBatch(TypedDict):
    source_type: Literal["camunda7_mssql", "xes", "other"]
    batch_id: str
    events: list[EventRecord]
    run_metadata: RunMetadataDTO
    stage_metadata: StageMetadataDTO
```

### 3.5 `GraphNodeDTO` / `GraphEdgeDTO`
```python
EdgeType = Literal[
    "control_flow", "resource", "data", "normative", "observed"
]

class GraphNodeDTO(TypedDict):
    node_id: str
    node_type: Literal["activity", "event", "gateway", "resource", "document", "other"]
    attrs: dict[str, Any]
    kappa: str
    ref_bpmn_id: str | None

class GraphEdgeDTO(TypedDict):
    src_id: str
    dst_id: str
    edge_type: EdgeType
```

### 3.6 `ProcessGraphDTO`
```python
class ProcessGraphDTO(TypedDict):
    nodes: list[GraphNodeDTO]
    edges: list[GraphEdgeDTO]
    process_version: str
    run_metadata: RunMetadataDTO
    stage_metadata: StageMetadataDTO
```

#### Invariants for `ProcessGraphDTO`
1. Кожен вузол має `node_id`, `node_type`, `attrs`, `kappa`.
2. Кожне ребро має валідний `edge_type` із фіксованого набору `EdgeType`.
3. Для кожного `edge.src_id`/`edge.dst_id` відповідні вузли існують у `nodes`.
4. `process_version` має дорівнювати `stage_metadata.process_version`.

### 3.7 `TensorGraphDTO`
```python
class TensorGraphDTO(TypedDict):
    x: "Tensor"
    edge_index: "Tensor"
    edge_attr: "Tensor | None"
    y: "Tensor | None"
    run_metadata: RunMetadataDTO
    stage_metadata: StageMetadataDTO
```

### 3.8 `DriftInputDTO` / `DriftResultDTO`
```python
class DriftInputDTO(TypedDict):
    batch: EventLogBatch
    graph: TensorGraphDTO | None
    distribution_features: dict[str, float]
    embedding_summary: dict[str, float]
    counters: dict[str, int]

class DriftResultDTO(TypedDict):
    data_drift_score: float
    concept_drift_score: float
    wasserstein_drift: float
    ood_flag: bool
    run_metadata: RunMetadataDTO
    stage_metadata: StageMetadataDTO
```

### 3.9 `InferenceResultDTO`
```python
class InferenceResultDTO(TypedDict):
    predicted_activity: str
    confidence: float
    reliability_score: float
    semaphore_mode: Literal["green", "yellow", "red"]
    ood_flag: bool
    run_metadata: RunMetadataDTO
    stage_metadata: StageMetadataDTO
```

### 3.10 `TrainStepResultDTO`
```python
class TrainStepResultDTO(TypedDict):
    loss_total: float
    loss_task: float
    loss_reg: float
    lr: float
    grad_norm: float | None
    run_metadata: RunMetadataDTO
    stage_metadata: StageMetadataDTO  # must include model_version/checkpoint_id
```

---

## 4) Ports / Interfaces (ABC level)

### 4.1 Data Readers
```python
class IDataReader(ABC):
    @abstractmethod
    def read_by_instance(self, process_instance_id: str, run_meta: RunMetadataDTO) -> EventLogBatch: ...

    @abstractmethod
    def read_batch(self, query: dict[str, Any], run_meta: RunMetadataDTO) -> EventLogBatch: ...
```

```python
class IXESReader(ABC):
    @abstractmethod
    def read_xes(self, file_path: str, run_meta: RunMetadataDTO) -> EventLogBatch: ...
```

### 4.2 Data Normalization
```python
class IDataConverter(ABC):
    @abstractmethod
    def normalize(self, raw_batch: EventLogBatch) -> EventLogBatch: ...
```

### 4.3 Graph / Tensor
```python
class IGraphBuilder(ABC):
    @abstractmethod
    def build_process_graph(self, batch: EventLogBatch) -> ProcessGraphDTO: ...
```

```python
class ITensorAdapter(ABC):
    @abstractmethod
    def to_tensor_graph(self, graph: ProcessGraphDTO) -> TensorGraphDTO: ...
```

### 4.4 Core ML
```python
class ICorePredictor(ABC):
    @abstractmethod
    def infer(self, graph: TensorGraphDTO) -> InferenceResultDTO: ...

    @abstractmethod
    def train_step(self, graph: TensorGraphDTO) -> TrainStepResultDTO: ...
```

### 4.5 Drift / Reliability
```python
class IDriftMonitor(ABC):
    @abstractmethod
    def evaluate_drift(self, drift_input: DriftInputDTO) -> DriftResultDTO: ...
```

```python
class IReliabilitySemaphore(ABC):
    @abstractmethod
    def compute(self, prediction: InferenceResultDTO, drift: DriftResultDTO) -> InferenceResultDTO: ...
```

### 4.6 Tracking / Storage / Reporting
```python
class ITracker(ABC):
    @abstractmethod
    def log_metrics(self, metrics: dict[str, float], run_meta: RunMetadataDTO, stage_meta: StageMetadataDTO) -> None: ...

    @abstractmethod
    def log_artifact(self, path: str, run_meta: RunMetadataDTO, stage_meta: StageMetadataDTO) -> None: ...
```

```python
class IStorage(ABC):
    @abstractmethod
    def save_model(self, model_blob: bytes, run_meta: RunMetadataDTO, stage_meta: StageMetadataDTO) -> str: ...
```

```python
class IReportWriter(ABC):
    @abstractmethod
    def write_eval_report(self, payload: dict[str, Any], run_meta: RunMetadataDTO, stage_meta: StageMetadataDTO) -> str: ...
```

---

## 5) Metadata policy (canonical)

### 5.1 Run-level metadata (для всього запуску)
Обов'язково задається на старті запуску й має бути доступне на всіх stage:
- `experiment_id`
- `git_commit`

### 5.2 Stage-level required (для кожного stage)
- `dataset_id`
- `schema_version`
- `process_version` (`κ`)

### 5.3 Stage-level optional/conditional
- `model_version`, `checkpoint_id` — для `Core ML`, `Evaluation`, `Inference`.
- `epokg_snapshot_id`, `graph_build_version` — для `Graph Builder`.
- `feature_config_version` — для `Tensor Adapter`.
- `drift_strategy_version` — для `Drift Monitor`.
- `evaluation_profile` — для `Evaluation`.

---

## 6) Definition of Ready для імплементації
1. Кожен adapter імплементує відповідний port.
2. `EventLogBatch` — єдина вхідна форма до Graph Builder.
3. `InferenceResultDTO` містить `reliability_score` + `semaphore_mode`.
4. Жоден інфраструктурний виклик не потрапляє всередину Core.

---

## 7) Validation Tests (Mandatory)

### Test A — End-to-End DTO Chain
Перевірити безперервність DTO:
`EventLogBatch → ProcessGraphDTO → TensorGraphDTO → InferenceResultDTO`.
Не допускаються пропуски required-ідентифікаторів (`case_id`, `process_version (κ)`, `dataset_id`, `schema_version`).

### Test B — Metadata Consistency
Правила metadata мають бути ідентичними в `INTERFACE_CONTRACTS.MD` і `DATA_FLOWS.MD`.
Run-level vs Stage-level вимоги не можуть конфліктувати.

### Test C — Module Coverage
Кожен модуль, що згадується в `DATA_FLOWS.MD`, повинен мати відповідний Port/Interface у `INTERFACE_CONTRACTS.MD`.
Phantom-модулі заборонені.

### Test D — Contract Concreteness
Для кожного DTO мають бути визначені мінімальні інваріанти та required-поля.
Мінімум для `ProcessGraphDTO`: required-ключі вузлів/ребер і allowed `edge_type`.

**Definition of Done:** заборонено merge/PR, якщо будь-який тест A–D не проходить.
