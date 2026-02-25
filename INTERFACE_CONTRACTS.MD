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

### 3.1 `EventRecord`
```python
class EventRecord(TypedDict):
    case_id: str
    activity_id: str
    event_time: str   # ISO-8601
    resource_id: str | None
    process_version: str  # κ
    payload: dict[str, Any]
```

### 3.2 `EventLogBatch`
```python
class EventLogBatch(TypedDict):
    source_type: Literal["camunda7_mssql", "xes", "other"]
    batch_id: str
    events: list[EventRecord]
    metadata: dict[str, Any]  # dataset_id, schema_version, process_version, etc.
```

### 3.3 `ProcessGraphDTO`
```python
class ProcessGraphDTO(TypedDict):
    nodes: list[dict[str, Any]]
    edges: list[tuple[str, str, str]]  # (src, dst, edge_type)
    process_version: str
    metadata: dict[str, Any]
```

### 3.4 `TensorGraphDTO`
```python
class TensorGraphDTO(TypedDict):
    x: Any
    edge_index: Any
    edge_attr: Any | None
    y: Any | None
    metadata: dict[str, Any]
```

### 3.5 `InferenceResultDTO`
```python
class InferenceResultDTO(TypedDict):
    predicted_activity: str
    confidence: float
    reliability_score: float
    semaphore_mode: Literal["green", "yellow", "red"]
    ood_flag: bool
    metadata: dict[str, Any]
```

---

## 4) Ports / Interfaces (ABC level)

### 4.1 Data Readers
```python
class IDataReader(ABC):
    @abstractmethod
    def read_by_instance(self, process_instance_id: str) -> EventLogBatch: ...

    @abstractmethod
    def read_batch(self, query: dict[str, Any]) -> EventLogBatch: ...
```

```python
class IXESReader(ABC):
    @abstractmethod
    def read_xes(self, file_path: str) -> EventLogBatch: ...
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
    def train_step(self, graph: TensorGraphDTO) -> dict[str, float]: ...
```

### 4.5 Drift / OOD
```python
class IDriftMonitor(ABC):
    @abstractmethod
    def evaluate_drift(self, batch_meta: dict[str, Any]) -> dict[str, float]: ...
```

### 4.6 Tracking / Storage / Reporting
```python
class ITracker(ABC):
    @abstractmethod
    def log_metrics(self, metrics: dict[str, float], metadata: dict[str, Any]) -> None: ...

    @abstractmethod
    def log_artifact(self, path: str, metadata: dict[str, Any]) -> None: ...
```

```python
class IStorage(ABC):
    @abstractmethod
    def save_model(self, model_blob: bytes, metadata: dict[str, Any]) -> str: ...
```

```python
class IReportWriter(ABC):
    @abstractmethod
    def write_eval_report(self, payload: dict[str, Any]) -> str: ...
```

---

## 5) Metadata contract (обов'язковий мінімум)
У `metadata` для кожного stage обов'язково передається:
- `dataset_id`
- `schema_version`
- `process_version` (`κ`)
- `model_version` (де застосовно)
- `git_commit`
- `experiment_id`

---

## 6) Definition of Ready для імплементації
1. Кожен adapter імплементує відповідний port.
2. `EventLogBatch` — єдина вхідна форма до Graph Builder.
3. `InferenceResultDTO` містить `reliability_score` + `semaphore_mode`.
4. Жоден інфраструктурний виклик не потрапляє всередину Core.
