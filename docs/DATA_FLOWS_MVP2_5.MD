# DATA_FLOWS_MVP2_5.MD

Updated: 2026-03-26
Status: ACTIVE

## 1. Scope
Runtime and offline data flows for MVP2.5 after Stage 3.4.

Detailed runtime model and tensor semantics are documented in:
1. `docs/GNN_RUNTIME_MVP2_5.MD`.

## 2. Flow A - ingest-topology (single dataset)
```mermaid
flowchart TD
    A1[CLI ingest-topology] --> A2[Load config and includes]
    A2 --> A3[Select adapter xes or camunda]
    A3 --> A4[Read traces or BPMN source]
    A4 --> A5[Build ProcessStructureDTO]
    A5 --> A6[IKnowledgeGraphPort save_process_structure]
    A6 --> A7[Write summary JSON]
```

## 3. Flow B - sync-topology (bulk)
```mermaid
flowchart TD
    B1[CLI sync-topology] --> B2[Resolve source catalog]
    B2 --> B3[Iterate process definitions or files]
    B3 --> B4[Build ProcessStructureDTO per process and version]
    B4 --> B5[IKnowledgeGraphPort save_process_structure]
    B5 --> B6[Sync summary and warnings]
```

## 4. Flow C - sync-stats (offline enrichment)
```mermaid
flowchart TD
    C1[CLI sync-stats] --> C2[Load runtime events from Camunda adapter]
    C2 --> C3[Load structure by process and version]
    C3 --> C4[Compute window metrics]
    C4 --> C5[Build stats_index and diagnostics]
    C5 --> C6[save_process_structure_snapshot]
    C6 --> C7[Immutable knowledge_version created]
    C7 --> C8[Summary JSON]
```

## 5. Flow D - train/eval/infer consumption
```mermaid
flowchart TD
    D1[Read traces] --> D2[PrefixPolicy]
    D2 --> D3[FeatureEncoder]
    D3 --> D4[DynamicGraphBuilder]
    D4 --> D5{stats_time_policy}
    D5 -->|strict_asof| D6[get_process_structure_as_of]
    D5 -->|latest| D7[get_process_structure]
    D6 --> D8[Build struct_x and structural tensors]
    D7 --> D8
    D8 --> D9[GraphTensorContract]
    D9 --> D10[Model forward]
```

## 6. Timestamp Semantics

### 6.1 Snapshot creation timestamp
1. `sync-stats --as-of <ISO>`: snapshot uses provided timestamp.
2. `sync-stats` without `--as-of`: snapshot uses derived `effective_as_of=max(event_ts)` from selected train-cut events per process scope.

### 6.2 Snapshot retrieval timestamp
1. `strict_asof`: builder uses prefix last-event timestamp as lookup time.
2. `latest`: builder ignores prefix time and reads latest structure/snapshot.

### 6.3 Selection rule
For `get_process_structure_as_of`:
1. select latest snapshot where `snapshot.as_of_ts <= requested_as_of`.
2. if no snapshot matches, repository fallback path is applied (current implementation behavior).

## 7. Stats Windows and Scopes
Computed during `sync-stats`:
1. windows: `last_7d`, `last_30d`, `last_90d`, `all_time`
2. scopes: `version`, `process`
3. flattened index key: `window.scope.metric`

## 8. Leakage Control Points
1. Structure and stats are generated offline.
2. Experiment timeline is controlled via `--as-of` snapshots.
3. Runtime consumes repository artifacts only.
4. To avoid future leakage in research timelines, maintain historical snapshot cadence and isolated namespace/storage for experiments.

## 9. Visualization Flow
```mermaid
sequenceDiagram
    participant U as User
    participant V as visualize-topology
    participant R as IKnowledgeGraphPort
    participant P as Renderer

    U->>V: run command with config and version
    V->>R: get_graph_for_visualization(process, version, min_freq)
    R-->>V: graph
    V->>P: render graph
    P-->>U: image output
```
