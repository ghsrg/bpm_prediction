# DATA_FLOWS_MVP2.MD

**Project:** bpm_prediction  
**Scope:** MVP2 Sprint 3 (Dual-Encoder `EOPKGGATv2`)  
**Purpose:** Canonical lifecycle flows for topology extraction, tensor injection, and drift-time evaluation.

---

## 1. Data Flow Architecture (Updated for MVP2)

MVP2 adds a structural knowledge branch: train traces are used to extract version-scoped topology, then `DynamicGraphBuilder` injects structural tensors into contracts.

```mermaid
flowchart TD
    classDef mvp2 fill:#e6ffed,stroke:#2ea043,stroke-width:2px,color:#000;

    Log[(XES File)] --> Adapter[XESAdapter.read]
    Adapter --> RT[RawTrace]

    RT -.->|train split only| TopologyExtractor[TopologyExtractorService]:::mvp2
    TopologyExtractor -.->|ProcessStructureDTO by version| KGPort[IKnowledgeGraphPort]:::mvp2

    RT --> PP[PrefixPolicy.generate_slices]
    PP --> PS[PrefixSlice]
    PS --> DGB[DynamicGraphBuilder.build_graph]:::mvp2
    KGPort -.->|get_process_structure version| DGB

    DGB --> GTC[GraphTensorContract + structural fields]:::mvp2
    GTC --> TRAIN[Trainer.train]
    GTC --> CROSS[Trainer.eval_cross_dataset]
    GTC --> DRIFT[Trainer.eval_drift]
```

Architecture invariant:
- `SchemaResolver` and feature lookup order remain unchanged from MVP1.
- MVP2 injects optional contract fields only, preserving baseline execution path.

---

## 2. Data Flow Overview (Lifecycle)

```mermaid
flowchart TD
    classDef mvp2 fill:#e6ffed,stroke:#2ea043,stroke-width:2px,color:#000;

    subgraph 1. Ingestion Layer
        ER[EventRecord] -->|group by case_id| RT[RawTrace]
    end

    subgraph 1.5. Knowledge Extraction Layer
        RT -.->|train-only fit| TEO[TopologyExtractorService]:::mvp2
        TEO --> PSDTO[ProcessStructureDTO registry]:::mvp2
    end

    subgraph 2. Prefix Generation Layer
        RT -->|PrefixPolicy| PS[PrefixSlice with process_version]
    end

    subgraph 3. Graph Engineering Layer
        PS --> DGB[DynamicGraphBuilder]
        PSDTO -.->|lookup by version| DGB
        DGB --> GTC[GraphTensorContract]
    end

    subgraph 4. Execution Modes
        GTC -->|train/val/test| TB[TrainingBatch]
        GTC -->|eval_cross_dataset| EC[Cross metrics]
        GTC -->|eval_drift| DW[Window metrics]
    end
```

---

## 3. Eval Drift Critical Path (Sliding Windows + Tail Drop)

Drift evaluation is chronological and leakage-safe. Windows are generated with:
- `size = drift_window_size`
- `step = drift_window_sliding or size`
- short tail windows (`len(window) < size`) are dropped.

```mermaid
flowchart LR
    classDef mvp2 fill:#e6ffed,stroke:#2ea043,stroke-width:2px,color:#000;

    A[Sort traces by first event timestamp] --> B[Generate windows size/step]
    B --> C[Drop short tail windows]
    C --> D[Build DataLoader per window]
    D --> E[Model forward + metrics]
    E --> F[window_macro_f1, window_test_ece, window OOS/slices]
```

Formalization:
\[
W_k = \mathcal{T}[s_k:s_k+w), \quad s_{k+1}=s_k+\Delta, \quad \Delta=\text{sliding}\;\text{or}\;w
\]

---

## 4. Dynamic Graph Building Flow (Train/Inference)

```mermaid
sequenceDiagram
    participant PP as PrefixPolicy
    participant DGB as DynamicGraphBuilder
    participant KG as IKnowledgeGraphPort
    participant TR as ModelTrainer
    participant M as Model (Baseline/EOPKG)

    PP->>DGB: PrefixSlice(case_id, process_version, prefix, target)
    DGB->>KG: get_process_structure(version)
    KG-->>DGB: ProcessStructureDTO or None
    DGB->>DGB: Build x_cat/x_num/edge_index/y
    DGB->>DGB: Build allowed_target_mask [C]
    DGB->>DGB: Build structural_edge_index/weight
    DGB-->>TR: GraphTensorContract
    TR->>M: forward(contract)
```

---

## 5. Runtime Integrity Gates

1. Train fit scope: topology extraction is built from train traces only.
2. Contract purity: model receives tensors only (`GraphTensorContract`), no Python objects.
3. Fallback safety: if structural tensors are absent, model executes baseline path without crash.
4. OOS safety: evaluator computes OOS only when `allowed_target_mask` exists; otherwise `test_oos=None`.

---

## 6. Dual-Encoder Attention Flow (Main Tensor Diagram)

```mermaid
flowchart LR
    classDef obs fill:#e6f0ff,stroke:#1f6feb,stroke-width:2px,color:#000;
    classDef struct fill:#e6ffed,stroke:#2ea043,stroke-width:2px,color:#000;
    classDef fuse fill:#fff4e5,stroke:#b26a00,stroke-width:2px,color:#000;

    C1[Contract: x_cat,x_num,edge_index,batch]:::obs --> O1["Observed encoder -> H_obs [B,H]"]:::obs

    C2[Contract: structural_edge_index]:::struct --> S1[Structural encoder]:::struct
    C3[Contract: struct_x optional]:::struct --> S0["if None -> struct_node_emb [C,S]"]:::struct
    S0 --> S1
    S1 --> HN["H_norm [C,S]"]:::struct

    HN --> P["struct_to_attn -> [C,H]"]:::fuse
    P --> KV["expand -> [B,C,H]"]:::fuse
    O1 --> Q["Q [B,1,H]"]:::fuse
    Q --> A[MultiheadAttention]:::fuse
    KV --> A
    A --> CTX["context [B,H], attn_weights [B,heads,1,C]"]:::fuse

    CTX --> T1["attn_to_struct -> [B,S]"]:::fuse
    O1 --> T2["concat(H_obs, context_struct)"]:::fuse
    T1 --> T2
    T2 --> FUS[Fusion MLP]:::fuse
    FUS --> OUT["logits [B,C]"]:::fuse
```

Notes:
- Structural branch is active when `structural_edge_index` is present.
- `struct_x` is optional in Sprint 3 and is replaced by structural embeddings when absent.
- Attention weights are retained in model state for XAI diagnostics.
