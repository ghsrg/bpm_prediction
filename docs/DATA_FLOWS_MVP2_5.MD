# DATA_FLOWS_MVP2_5.MD

Updated: 2026-03-17  
Status: ACTIVE

## 1. Scope
This document defines runtime data flows for MVP2.5 with strict separation between:
- topology ingestion,
- model train and eval.

## 2. Flow A: ingest-topology
```mermaid
flowchart TD
    C1[CLI ingest-topology --config] --> L1[Load YAML includes]
    L1 --> A1[Resolve adapter xes or camunda]
    A1 --> D1{Camunda + structure.source=bpmn and structure_from_logs=false}
    D1 -->|yes| B1A[CamundaBpmnAdapter fetch catalog and BPMN XML]
    B1A --> B1B[BpmnStructureParserService parse and flatten]
    B1B --> Q1{Parse success}
    Q1 -->|yes| P1[IKnowledgeGraphPort save_process_structure]
    Q1 -->|no| Q2[Append quarantine record]
    D1 -->|no| R1[Read traces]
    R1 --> F1[Apply fraction]
    F1 --> S1[Apply ingest split train or full]
    S1 --> T1[TopologyExtractorService extract_from_logs]
    T1 --> P1
    P1 --> B1[File repository JSON artifact]
    Q2 --> O1[Write ingestion summary JSON]
    B1 --> O1
```

## 3. Flow B: train/eval
```mermaid
flowchart TD
    C2[CLI main --config] --> L2[Load YAML includes]
    L2 --> A2[Resolve trace adapter]
    A2 --> R2[Read traces]
    R2 --> P2[Prefix generation]
    P2 --> E2[FeatureEncoder fit and encode]
    E2 --> G2[DynamicGraphBuilder]
    G2 --> K2[IKnowledgeGraphPort get_process_structure]
    K2 --> M2[Build GraphTensorContract]
    M2 --> D2[Model forward and loss]
    D2 --> X2[Metrics and tracker]
```

## 4. Train flow with fallback branch
```mermaid
flowchart LR
    G[DynamicGraphBuilder] --> K[Repository lookup by process and version]
    K -->|found| S[Build allowed_target_mask]
    K -->|missing| F[Fallback no structure fields]
    S --> T[Tensor contract]
    F --> T
```

## 5. Sequence for repository-backed visualization
```mermaid
sequenceDiagram
    participant U as User
    participant V as visualize-topology
    participant R as IKnowledgeGraphPort
    participant P as plot_topology

    U->>V: run visualize-topology --config
    V->>R: list_versions(process_name)
    R-->>V: available_versions
    V->>R: get_graph_for_visualization(process_name, version, min_freq)
    R-->>V: nx.DiGraph
    V->>P: render graphviz or pm4py
    P-->>U: image file or viewer
```

## 6. Data leakage control points
- Ingestion split uses chronological train subset by default.
- Training split is independent and re-applied in train pipeline.
- No evaluation records are used to build topology when `ingest_split=train`.
- File-backed topology artifacts make split boundary explicit and reproducible.

## 7. Deterministic diagnostics
For Camunda BPMN ingest summary:
1. `quarantined_procdefs` is explicit.
2. `quarantine_error_codes` is a stable sorted list.
3. `quarantine_records[*].source_hint` is available for operator debugging.
