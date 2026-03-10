# DATA_MODEL_MVP2.MD

## 0. Canon & Naming Rules
- **Scope:** MVP2 (EOPKG Fusion GNN & Zero-Shot Knowledge Injection).
- **Інваріант сумісності:** Усі зміни структури даних зберігають повну зворотну сумісність із MVP1 (використання `Optional`).
- Symbolic notation follows `VARIABLES.MD`.

---

## 1. External Boundary Objects (Adapters Layer)

Об'єкти `EventRecord` та `RawTrace` залишаються сумісними з MVP1, але для MVP2 поле `process_version` ($\kappa$) стає критично важливим маршрутизатором для графа знань.

### 1.1 Boundary DTO Composition (Cascade)

```mermaid
classDiagram
    classDef mvp2 fill:#e6ffed,stroke:#2ea043,stroke-width:2px,color:#000;

    class RawTrace {
      +case_id: str
      +process_version: str (κ)
      +trace_attributes: Dict[str, Any]
      +events: List[EventRecord]
    }

    class EventRecord {
      +activity_id: str
      +timestamp: float
      +resource_id: str
      +lifecycle: Optional[str]
      +position_in_trace: int
      +extra: Dict[str, Any]
    }

    class PrefixSlice {
      +case_id: str
      +process_version: str (κ)
      +prefix_events: List[EventRecord]
      +target_event: EventRecord
    }

    RawTrace "1" *-- "1..*" EventRecord
    PrefixSlice "1" *-- "1..*" EventRecord
    RawTrace --> PrefixSlice : sliced into
```

---

## 2. Нові Knowledge Graph DTOs (Domain Layer)

Для передачі нормативної структури (BPMN-каркасу) від TopologyExtractor (або Neo4j у майбутньому) до `DynamicGraphBuilder` вводиться новий контракт.

### 2.1 ProcessStructureDTO
```python
class ProcessStructureDTO(BaseModel):
    version: str # Відповідає \kappa
    allowed_edges: List[Tuple[str, str]] # Список дозволених переходів (Activity A -> Activity B)
    edge_statistics: Optional[Dict[Tuple[str, str], Dict[str, float]]] = None # Опціональна статистика для MVP2+
```

---

## 3. Graph Tensor Contract (The Core Extension)

Найважливіша зміна MVP2: розширення єдиного формату даних, який приймає нейромережа, додаванням нормативних масок.

### 3.1 Розширений контракт
```python
class GraphTensorContract(TypedDict):
    # --- MVP1 (Обов'язкові поля локального графа G_obs) ---
    x_cat: torch.LongTensor      # X_cat \in \mathbb{N}^{N \times F_cat}
    x_num: torch.FloatTensor     # X_num \in \mathbb{R}^{N \times F_num}
    edge_index: torch.LongTensor # edge_index \in \mathbb{N}^{2 \times E}
    edge_type: torch.LongTensor  # edge_type \in \mathbb{N}^{E}
    y: torch.LongTensor          # y \in \mathbb{N}^{1}
    batch: torch.LongTensor      # batch \in \mathbb{N}^{N}
    num_nodes: int               # N
    
    # --- MVP2 (Опціональні поля нормативного графа G_struct) ---
    structural_edge_index: Optional[torch.LongTensor] = None # \in \mathbb{N}^{2 \times E_struct}
    structural_edge_weight: Optional[torch.FloatTensor] = None 
    version_emb_idx: Optional[torch.LongTensor] = None       # Індекс \kappa для embedding шару
```

### 3.2 Graph Tensor Attribute Cascade

```mermaid
flowchart LR
    classDef mvp2 fill:#e6ffed,stroke:#2ea043,stroke-width:2px,color:#000;

    PS[PrefixSlice] --> DGB[DynamicGraphBuilder]
    KG[ProcessStructureDTO]:::mvp2 -.->|get_structure version| DGB
    
    DGB --> XC[X_cat]
    DGB --> XN[X_num]
    DGB --> EI[edge_index]
    DGB --> ET[edge_type]
    
    DGB --> SEI[structural_edge_index]:::mvp2
    DGB --> SEW[structural_edge_weight]:::mvp2
    DGB --> VEI[version_emb_idx]:::mvp2
    
    DGB --> Y[y]
    DGB --> B[batch]
    DGB --> N[num_nodes]
```

---

## 4. Experiment Configuration Schema (MVP2 Updates)

Конфігурація експерименту розширюється осями дослідження з `ARCHITECTURE_GUIDELINES.MD`.

### 4.1 Додані поля конфігурації
```yaml
experiment:
  # ... параметри MVP1 (mode, windows) ...
  # Нові дослідницькі осі MVP2:
  structure_mode: "epokg"         # logs_only | bpmn | epokg
  kappa_conditioning: "static"    # none | static | latent

model:
  type: "EOPKGFusionGNN"          # Динамічно завантажується через ModelFactory
  hidden_dim: 64
  # ...
```

### 4.2 Config Cascade Structure (MVP2)
```mermaid
flowchart TD
    classDef mvp2 fill:#e6ffed,stroke:#2ea043,stroke-width:2px,color:#000;

    Root[Experiment YAML] --> EX[experiment]
    Root --> M[model]
    
    EX --> EXM[mode]
    EX --> SM[structure_mode]:::mvp2
    EX --> KC[kappa_conditioning]:::mvp2
    EX --> EXW[drift_window_size]
    
    M --> MT[type: EOPKGFusionGNN]:::mvp2
    M --> HD[hidden_dim]
```

---

## 5. Object Composition Schema (Що є частиною чого)

Ця глобальна діаграма відображає структурну залежність (композицію) об'єктів моделі даних для MVP2. Вона показує, як нормативні знання (EOPKG) інтегруються в існуючий потік тензорів.

```mermaid
classDiagram
    classDef mvp2 fill:#e6ffed,stroke:#2ea043,stroke-width:2px,color:#000;

    class EventRecord {
        +activity_id: String
        +timestamp: Float
        +position_in_trace: Int
        +extra: Dict
    }

    class RawTrace {
        +case_id: String
        +process_version: String
        +events: EventRecord[]
    }

    class PrefixSlice {
        +process_version: String
        +prefix_events: EventRecord[]
        +target_event: EventRecord
    }

    class ProcessStructureDTO:::mvp2 {
        +version: String
        +allowed_edges: Tuple[]
        +edge_statistics: Dict
    }

    class InstanceGraph {
        +nodes: EventRecord[]
        +edges: Tuple[]
    }

    class GraphTensorContract {
        +x_cat: Tensor
        +x_num: Tensor
        +edge_index: Tensor
        +edge_type: Tensor
        +y: Tensor
        +batch: Tensor
        %% MVP2 Extensions
        +structural_edge_index: Tensor?:::mvp2
        +structural_edge_weight: Tensor?:::mvp2
        +version_emb_idx: Tensor?:::mvp2
    }

    class TrainingBatch {
        +graph_batch: Batch
    }

    RawTrace *-- EventRecord : contains (1..N)
    PrefixSlice *-- EventRecord : contains (1..N)
    PrefixSlice *-- EventRecord : target (1)

    InstanceGraph *-- EventRecord : built from
    ProcessStructureDTO ..> GraphTensorContract : injects structural mask
    
    GraphTensorContract *-- InstanceGraph : encoded from
    TrainingBatch *-- GraphTensorContract : contains (1..N)
```

## 6. Tensor Shape Invariants (Оновлено)
Для одного графа (суміш $G_{obs}$ та $G_{struct}$):
\[
X_{cat} \in \mathbb{N}^{N \times F_{cat}}, \quad X_{num} \in \mathbb{R}^{N \times F_{num}}
\]
\[
edge\_index_{obs} \in \mathbb{N}^{2 \times E_{obs}}
\]
\[
edge\_index_{struct} \in \mathbb{N}^{2 \times E_{struct}} \quad \text{(Optional)}
\]
Цільова змінна:
\[
y \in \{0,\dots,C-1\}
\]