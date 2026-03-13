# ARCHITECTURE_MVP2.MD

## 1. Scope
MVP2 extends MVP1 with Zero-Shot Knowledge Injection through EOPKG and introduces a Dual-Encoder architecture for `EOPKGGATv2`.

Primary goal:
1. Keep MVP1 fully runnable.
2. Inject structural process knowledge into model inference.
3. Improve drift robustness (F1 stability, ECE stability, OOS reduction).

## 2. Backward Compatibility Invariant
MVP2 is additive and must not break MVP1:
1. `GraphTensorContract` is extended only with optional fields.
2. If structural tensors are absent, model falls back to baseline path.
3. Evaluator must compute base metrics even when OOS data is unavailable.

## 3. Core Components
1. `IKnowledgeGraphPort` (Domain port): provides structural process topology.
2. `TopologyExtractorService` (Application service): extracts topology from train traces.
3. `DynamicGraphBuilder` (Domain service): produces `allowed_target_mask` and structural tensors.
4. `EOPKGGATv2` (Domain model): Dual-Encoder + Soft Cross-Attention.
5. `ModelTrainer` (Application use case): orchestrates training/evaluation, slicing, OOS.

## 4. Clean Architecture (Layered View)

```mermaid
flowchart TB
    classDef mvp2 fill:#e6ffed,stroke:#2ea043,stroke-width:2px,color:#000;

    subgraph Frameworks
        XES[XES files]
    end

    subgraph Adapters
        XESAdapter[XESAdapter]
    end

    subgraph Application
        Prep[Data Preparation]
        Trainer[ModelTrainer]
        Eval[Evaluator logic in Trainer]
        TopologyExtractor[TopologyExtractorService]:::mvp2
    end

    subgraph Domain
        PrefixPolicy[PrefixPolicy]
        Encoder[FeatureEncoder]
        DGB[DynamicGraphBuilder]:::mvp2
        KGPort[IKnowledgeGraphPort]:::mvp2
        Baselines[BaselineGCN/BaselineGATv2]
        EOPKGGATv2[EOPKGGATv2 Dual-Encoder]:::mvp2
    end

    XES --> XESAdapter
    XESAdapter --> Prep
    Prep --> PrefixPolicy
    PrefixPolicy --> DGB
    Prep --> Encoder
    Encoder --> DGB

    Prep -.->|train split| TopologyExtractor
    TopologyExtractor -.->|ProcessStructureDTO| KGPort
    KGPort -.-> DGB

    DGB --> Trainer
    Trainer --> Baselines
    Trainer --> EOPKGGATv2
    Trainer --> Eval
```

## 5. Component Architecture (Hexagonal View)

```mermaid
flowchart LR
    classDef mvp2 fill:#e6ffed,stroke:#2ea043,stroke-width:2px,color:#000;

    subgraph Ports
      IXES[IXESAdapter]
      IGB[IGraphBuilder]
      IKG[IKnowledgeGraphPort]:::mvp2
      ITR[ITracker]
    end

    subgraph Adapters
      XES[XESAdapter]
      MLflow[MLflowTracker]
    end

    subgraph Application
      TR[ModelTrainer]
      TE[TopologyExtractorService]:::mvp2
    end

    subgraph Domain
      DGB[DynamicGraphBuilder]:::mvp2
      M1[Baseline models]
      M2[EOPKGGATv2 Dual-Encoder]:::mvp2
    end

    XES --> IXES
    MLflow --> ITR

    TR --> IXES
    TR --> IGB
    TR --> ITR
    TR --> M1
    TR --> M2

    DGB --> IKG
    TE --> IKG
    DGB --> TR
```

## 6. Dual-Encoder Model Structure (Main Diagram)

This is the key Sprint 3 architecture.

### 6.1 Tensor Routing Overview

```mermaid
flowchart LR
    classDef obs fill:#e6f0ff,stroke:#1f6feb,stroke-width:2px,color:#000;
    classDef struct fill:#e6ffed,stroke:#2ea043,stroke-width:2px,color:#000;
    classDef fuse fill:#fff4e5,stroke:#b26a00,stroke-width:2px,color:#000;

    XCAT[x_cat, x_num, edge_index, batch]:::obs --> OBSENC[Observed Encoder\nGATv2Conv stack]:::obs
    OBSENC --> HOBS["H_obs: [B, hidden_dim]"]:::obs

    SEI[structural_edge_index]:::struct --> SENC[Structural Encoder\nstruct_gnn]:::struct
    SX[struct_x optional]:::struct --> SXSEL[if None -> struct_node_emb]:::struct
    SXSEL --> SENC
    SEW[structural_edge_weight optional]:::struct -. optional .-> SENC
    SENC --> HNORM["H_norm: [C, struct_hidden_dim]"]:::struct

    HOBS --> Q["Q: [B,1,hidden_dim"]]:::fuse
    HNORM --> PROJ["struct_to_attn\n[C,hidden_dim]"]:::fuse
    PROJ --> KV["expand -> [B,C,hidden_d im]"]:::fuse

    Q --> ATTN[MultiheadAttention\nbatch_first=True]:::fuse
    KV --> ATTN
    ATTN --> CXT["context: [B,hidden_dim]\nattn_weights: [B,H,1,C]"]:::fuse

    CXT --> CPROJ["attn_to_struct\n[B,struct_hidden_dim]"]:::fuse
    HOBS --> CONCAT["concat(H_obs, context_struct)"]:::fuse
    CPROJ --> CONCAT

    CONCAT --> FUSION["Fusion MLP\nLinear(hidden+struct_hidden -> hidden)+ReLU"]:::fuse
    FUSION --> CLS[Classifier -> logits [B,C]]:::fuse
```

### 6.2 Forward Contract Rules
1. If `structural_edge_index` is missing/empty: baseline fallback.
2. If `struct_x` is missing: use `struct_node_emb` (expected in Sprint 3).
3. Cross-attention weights are stored in `self.last_cross_attn_weights` for XAI.
4. `allowed_target_mask` is not used by model forward; it is used by evaluator (OOS).

## 7. Runtime Flows

### 7.1 Training Flow

```mermaid
sequenceDiagram
    participant X as XESAdapter
    participant T as TopologyExtractorService
    participant D as DynamicGraphBuilder
    participant M as EOPKGGATv2
    participant R as ModelTrainer

    X->>R: RawTrace iterator
    R->>T: extract_from_logs(train_traces)
    T-->>D: ProcessStructureDTO (by version)
    R->>D: build_graph(prefix)
    D-->>R: GraphTensorContract (with mask + structural edges)
    R->>M: forward(contract)
    M-->>R: logits (+last_cross_attn_weights)
```

### 7.2 Eval Drift Flow

```mermaid
sequenceDiagram
    participant R as ModelTrainer
    participant D as DynamicGraphBuilder
    participant M as EOPKGGATv2

    R->>R: chronological windows
    loop each window
      R->>D: build_graph(prefixes)
      D-->>R: contracts (may contain/no structural tensors)
      R->>M: forward
      M-->>R: logits
      R->>R: compute F1/ECE/OOS/slices
    end
```

## 8. Scope Limits and MVP3 Boundary
1. Sprint 3 scope is only `EOPKGGATv2` Dual-Encoder refactor.
2. `EOPKGGCN` stays on previous fusion path (compatibility mode).
3. Rich `struct_x` features from graph DB (resources, policies, advanced stats) are Sprint 4+.
4. Agent-Critic and dynamic beta regularization remain MVP3.
