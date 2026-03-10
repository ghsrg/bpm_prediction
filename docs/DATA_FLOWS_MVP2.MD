# DATA_FLOWS_MVP2.MD

## 1. Zero-Shot EOPKG Extraction Flow (Тренувальна фаза)
Щоб уникнути Data Leakage, $G_{struct}$ будується як "Oracle" виключно на тренувальній вибірці:
```mermaid
sequenceDiagram
    participant Splitter
    participant OracleDiscoveryService
    participant EOPKG_Memory
    
    Splitter->>OracleDiscoveryService: train_traces (тільки версія k=1)
    OracleDiscoveryService->>OracleDiscoveryService: Екстракція унікальних пар (A->B)
    OracleDiscoveryService->>EOPKG_Memory: Збереження ProcessStructureDTO(k=1)
```

## 2. Dynamic Graph Building Flow (Інференс / Навчання)
```mermaid
sequenceDiagram
    participant PrefixPolicy
    participant DynamicGraphBuilder
    participant IKnowledgeGraphPort
    participant Model
    
    PrefixPolicy->>DynamicGraphBuilder: PrefixSlice (з версією k)
    DynamicGraphBuilder->>IKnowledgeGraphPort: get_structure(k)
    IKnowledgeGraphPort-->>DynamicGraphBuilder: ProcessStructureDTO
    DynamicGraphBuilder->>DynamicGraphBuilder: Побудова structural_edge_index
    DynamicGraphBuilder->>Model: Розширений GraphTensorContract
    Model->>Model: forward() з урахуванням маски (або fallback)
```