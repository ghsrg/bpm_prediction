# ARCHITECTURE_MVP2.MD

## 1. Scope (Призначення)
MVP2 розширює Baseline-архітектуру (MVP1) шляхом впровадження Enterprise Operational Process Knowledge Graph (EOPKG) та оператора злиття ($\Gamma$).
Головна мета: довести гіпотезу про те, що використання нормативного графа ($G_{struct}$) зменшує деградацію моделі під час структурного дрейфу.

## 2. Головний архітектурний інваріант (Зворотна сумісність)
Впровадження MVP2 **не повинно ламати MVP1**. 
- Всі існуючі контракти розширюються через `Optional` поля.
- Якщо дані з EOPKG відсутні (наприклад, `structural_edge_index is None`), система (включаючи `forward` моделі) повинна автоматично перемикатися в режим Baseline (MVP1).

## 3. Нові компоненти системи
1. **IKnowledgeGraphPort (Domain Layer)**: Порт для отримання $G_{struct}$.
2. **TopologyExtractorService (Application Layer)**: Реалізація порту для MVP2. Екстрагує нормативну топологію безпосередньо з `train_traces` (без Data Leakage цільових змінних на тестовій вибірці) для симуляції BPMN-каркасу.
3. **DynamicGraphBuilder (Application Layer)**: Розширює `BaselineGraphBuilder`. Додає маски з EOPKG у тензорний контракт, якщо включено відповідний режим у конфігу.
4. **EOPKGFusionGNN (Domain Layer)**: Нова архітектура моделі. Реалізує оператор $\Gamma$, який застосовує увагу (Attention) або гейтинг (Gating) на основі $G_{struct}$.

## 4. Clean Architecture (Layered View)

```mermaid
flowchart TB
    classDef mvp2 fill:#e6ffed,stroke:#2ea043,stroke-width:2px,color:#000;

    subgraph Frameworks
        XES[XES Files]
    end

    subgraph Adapters
        XESAdapter[XES Adapter]
    end

    subgraph Application
        TracePreprocessor[TracePreprocessor]
        PrefixPolicy[PrefixPolicy]
        GraphBuilder[IGraphBuilder]
        DynamicGraphBuilder[DynamicGraphBuilder]:::mvp2
        TopologyExtractor[TopologyExtractorService]:::mvp2
        Trainer[Trainer]
        Evaluator[Evaluator]
    end

    subgraph Domain
        SchemaResolver[SchemaResolver]
        InstanceGraph[InstanceGraph]
        FeatureEncoder[FeatureEncoder]
        GNNModel[GNNModelPort]
        EOPKGFusionGNN[EOPKGFusionGNN]:::mvp2
        IKnowledgeGraphPort[IKnowledgeGraphPort]:::mvp2
        Predictor[Predictor]
    end

    XES --> XESAdapter
    XESAdapter --> TracePreprocessor
    
    %% MVP2 Zero-Shot flow
    XESAdapter -.->|train_traces only| TopologyExtractorService
    TopologyExtractorService -.->|Extracts Topology| IKnowledgeGraphPort
    
    TracePreprocessor --> PrefixPolicy
    PrefixPolicy --> GraphBuilder
    PrefixPolicy --> DynamicGraphBuilder
    
    GraphBuilder --> InstanceGraph
    DynamicGraphBuilder --> InstanceGraph
    DynamicGraphBuilder -.->|get_structure version| IKnowledgeGraphPort
    
    InstanceGraph --> FeatureEncoder
    FeatureEncoder --> GNNModel
    FeatureEncoder --> EOPKGFusionGNN
    
    SchemaResolver --> XESAdapter
    SchemaResolver --> FeatureEncoder
    
    GNNModel --> Predictor
    EOPKGFusionGNN --> Predictor
    
    Predictor --> Trainer
    Predictor --> Evaluator
```

### Пояснення шарів (Оновлено для MVP2)
*Нові компоненти MVP2 підсвічені зеленим.*
- **Application:** Додано `TopologyExtractorService` для безпечної екстракції топології (без Leakage) та `DynamicGraphBuilder`, який вміє звертатися до графа знань.
- **Domain:** Додано порт `IKnowledgeGraphPort` (для інверсії залежностей бази знань) та нову модель `EOPKGFusionGNN` з оператором $\Gamma$.

---

## 5. Component Architecture (Hexagonal View)

```mermaid
flowchart LR
    classDef mvp2 fill:#e6ffed,stroke:#2ea043,stroke-width:2px,color:#000;

    subgraph Core
        Resolver[SchemaResolver]
        Graph[InstanceGraph]
        Encoder[FeatureEncoder]
        Model[GNNModelPort]
        FusionModel[EOPKGFusionGNN]:::mvp2
    end

    subgraph Ports
        DataPort[IXESAdapter]
        TrainPort[TrainingPort]
        PredictPort[PredictionPort]
        GraphPort[IGraphBuilder]
        KGPort[IKnowledgeGraphPort]:::mvp2
    end

    subgraph Adapters
        XESAdapter[XESAdapter]
        CLIAdapter[CLI Runner]
        TopologyExtractorService[TopologyExtractorService]:::mvp2
    end

    XESAdapter --> DataPort
    CLIAdapter --> TrainPort
    CLIAdapter --> PredictPort

    DataPort --> Resolver
    TopologyExtractorAdapter -.->|Implements| KGPort
    
    Resolver --> Graph
    Graph --> Encoder
    Encoder --> GraphPort
    GraphPort --> Model
    GraphPort --> FusionModel
    
    KGPort -.->|Provides G_struct| FusionModel
    
    Model --> PredictPort
    FusionModel --> PredictPort
```

---

## 6.1 Training Flow (із Zero-Shot Topology Extraction)

```mermaid
sequenceDiagram
    participant XES as XESAdapter
    participant TopologyExtractor as TopologyExtractorService
    participant KG as IKnowledgeGraphPort
    participant TR as Trainer
    participant PP as PrefixPolicy
    participant DGB as DynamicGraphBuilder
    participant FE as FeatureEncoder
    participant M as FusionGNN

    rect rgb(230, 255, 237)
    Note right of XES: MVP2: Zero-Shot Knowledge Injection
    XES->>TopologyExtractor: train_traces (k=1)
    TopologyExtractor->>TopologyExtractor: Симуляція BPMN (унікальні ребра)
    TopologyExtractor->>KG: save_structure(ProcessStructureDTO)
    end

    XES->>TR: Iterator[RawTrace]
    TR->>PP: generate_slices(trace)
    PP->>DGB: PrefixSlice (з версією k)
    
    rect rgb(230, 255, 237)
    DGB->>KG: get_structure(k)
    KG-->>DGB: allowed_edges
    end
    
    DGB->>FE: encode_event(...)
    DGB-->>TR: Розширений GraphTensorContract
    TR->>M: forward/backward (Злиття G_obs + G_struct)
```

## 6.2 Eval Drift Flow (З урахуванням EOPKG)

```mermaid
sequenceDiagram
    participant InputTrace
    participant PrefixPolicy
    participant DynamicGraphBuilder
    participant KGPort as IKnowledgeGraphPort
    participant Model as EOPKGFusionGNN
    participant Evaluator

    InputTrace->>PrefixPolicy: trace/prefix (версія k+n)
    PrefixPolicy->>DynamicGraphBuilder: prefix graph
    DynamicGraphBuilder->>KGPort: get_structure(k+n)
    
    alt Structure Exists
        KGPort-->>DynamicGraphBuilder: ProcessStructureDTO
        DynamicGraphBuilder->>Model: GTC (with structural_edge_index)
        Model->>Model: forward (Attention/Fusion)
    else Structure Unknown (Drift)
        KGPort-->>DynamicGraphBuilder: None
        DynamicGraphBuilder->>Model: GTC (without structural_edges)
        Model->>Model: forward (Fallback до MVP1 Baseline)
    end
    
    Model->>Evaluator: logits
    Evaluator->>Evaluator: compute F1/Accuracy/ECE
```

## 6.3 End-to-End Pipeline

```mermaid
flowchart TD
    classDef mvp2 fill:#e6ffed,stroke:#2ea043,stroke-width:2px,color:#000;

    Event --> NormalizedEvent
    NormalizedEvent --> RawTrace
    
    RawTrace -->|Only Train Data| TopologyExtractor:::mvp2
    TopologyExtractor --> ProcessStructureDTO:::mvp2
    
    RawTrace --> TracePreprocessor
    TracePreprocessor --> PrefixPolicy
    PrefixPolicy --> PrefixSlice
    PrefixSlice --> InstanceGraph
    InstanceGraph --> SchemaResolver
    SchemaResolver --> FeatureEncoder
    
    ProcessStructureDTO -.-> DynamicGraphBuilder:::mvp2
    FeatureEncoder --> DynamicGraphBuilder
    
    DynamicGraphBuilder --> ExpandedTensorContract[GraphTensorContract + Masks]:::mvp2
    ExpandedTensorContract --> EOPKGFusionGNN:::mvp2
```