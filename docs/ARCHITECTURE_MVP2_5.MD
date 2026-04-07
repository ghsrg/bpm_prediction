# ARCHITECTURE_MVP2_5.MD

Updated: 2026-03-26
Status: ACTIVE (Stage 4.2 implemented, closure sync in progress)

## 1. Purpose
This document is the canonical architecture baseline for MVP2.5.

MVP2.5 currently includes:
1. Offline topology ingestion (`ingest-topology`, `sync-topology`).
2. Offline statistics enrichment (`sync-stats`).
3. Repository-backed structure consumption in train/eval/infer via `IKnowledgeGraphPort`.
4. Snapshot-based historical lookup (`as_of`) for temporal-safe experiments.

Runtime model/tensor details are specified in:
1. `docs/GNN_RUNTIME_MVP2_5.MD`.

## 2. Core Architecture Principle
The system is split into independent pipelines:
1. Ingestion pipeline: reads raw sources and persists structure/snapshots.
2. Runtime ML pipeline: reads traces and consumes prebuilt artifacts only.

Training must not build topology/stats synchronously from raw sources.

## 3. Hexagonal View
```mermaid
flowchart LR
    subgraph Drivers
        CLI_INGEST[CLI ingest-topology]
        CLI_SYNC[CLI sync-topology]
        CLI_STATS[CLI sync-stats]
        CLI_TRAIN[CLI train and eval]
        TOOL_VIS[CLI visualize-topology]
    end

    subgraph Application
        UC_INGEST[IngestTopologyUseCase]
        UC_SYNC[SyncTopologyUseCase]
        UC_STATS[SyncStatsUseCase]
        UC_TRAIN[TrainEvalUseCase]
        UC_VIS[VisualizeTopologyUseCase]
    end

    subgraph Ports
        PORT_TRACE[IXESAdapter and ICamundaTracePort]
        PORT_RUNTIME[ICamundaRuntimePort]
        PORT_KG[IKnowledgeGraphPort]
        PORT_TRACK[ITracker]
    end

    subgraph Domain
        SVC_EXTRACT[TopologyExtractorService]
        SVC_PARSER[BpmnStructureParserService]
        SVC_BUILDER[DynamicGraphBuilder]
        SVC_ENCODER[FeatureEncoder and PrefixPolicy]
        SVC_MODEL[GNN Models]
    end

    subgraph Adapters
        AD_XES[XESAdapter]
        AD_CAM_TRACE[CamundaTraceAdapter]
        AD_CAM_RT[CamundaRuntimeAdapter]
        AD_REPO_FILE[FileBasedKnowledgeGraphRepository]
        AD_REPO_MEM[InMemoryNetworkXRepository]
        AD_REPO_NEO[Neo4jKnowledgeGraphRepository]
        AD_TRACK[MLflowTracker]
    end

    CLI_INGEST --> UC_INGEST
    CLI_SYNC --> UC_SYNC
    CLI_STATS --> UC_STATS
    CLI_TRAIN --> UC_TRAIN
    TOOL_VIS --> UC_VIS

    UC_INGEST --> PORT_TRACE
    UC_INGEST --> SVC_EXTRACT
    UC_INGEST --> SVC_PARSER
    SVC_EXTRACT --> PORT_KG
    SVC_PARSER --> PORT_KG

    UC_SYNC --> PORT_TRACE
    UC_SYNC --> SVC_EXTRACT
    SVC_EXTRACT --> PORT_KG

    UC_STATS --> PORT_RUNTIME
    UC_STATS --> PORT_KG

    UC_TRAIN --> PORT_TRACE
    UC_TRAIN --> SVC_ENCODER
    UC_TRAIN --> SVC_BUILDER
    SVC_BUILDER --> PORT_KG
    UC_TRAIN --> SVC_MODEL
    UC_TRAIN --> PORT_TRACK

    UC_VIS --> PORT_KG

    PORT_TRACE --> AD_XES
    PORT_TRACE --> AD_CAM_TRACE
    PORT_RUNTIME --> AD_CAM_RT
    PORT_KG --> AD_REPO_FILE
    PORT_KG --> AD_REPO_MEM
    PORT_KG --> AD_REPO_NEO
    PORT_TRACK --> AD_TRACK
```

## 4. Pipeline Separation

### 4.1 Structure ingestion (offline)
1. Read traces or BPMN source.
2. Build `ProcessStructureDTO`.
3. Save structure through `IKnowledgeGraphPort`.

### 4.2 Topology bulk sync (offline)
1. Iterate all configured process definitions/files.
2. Save/update structure by `(process_name, version)`.

### 4.3 Stats sync (offline)
1. Read runtime activity events from Camunda runtime adapter.
2. Load existing structure from repository.
3. Compute Tier A stats windows (`last_7d`, `last_30d`, `last_90d`, `all_time`) relative to `as_of`.
4. Persist immutable snapshot (`knowledge_version`) in repository.

### 4.4 Train/eval/infer (online runtime)
1. Read traces and build prefixes.
2. Resolve structure (+ stats snapshot) from repository.
3. Build tensors (`GraphTensorContract`) and run model forward.

## 5. Stage 3.4 Snapshot Policy

1. Immutable snapshots are append-only.
2. Snapshot key is strict:
   - `tenant_id + process_name + version_key + proc_def_id + knowledge_version`.
3. Stats storage is JSON-only payload in snapshot node/document.
4. `as_of` lookup returns nearest snapshot with timestamp `<= as_of`.
5. TTL is not used.

## 6. Timestamp Semantics

1. `sync-stats --as-of <ISO>`: snapshot timestamp is forced to provided value.
2. `sync-stats` without `--as-of`: effective snapshot timestamp is derived from selected train-cut events (`max(event_ts)` per process scope).
3. In train/eval with `experiment.stats_time_policy = strict_asof`:
   - builder uses prefix last-event timestamp as runtime `as_of`.
4. In `latest` policy:
   - builder uses latest available structure/snapshot.

## 7. Backward Compatibility and Fallback

1. MVP1 path must remain functional.
2. Missing structure/snapshot does not crash unless strict load mode is explicitly enabled.
3. EOPKG stats enrichment is additive to existing contracts.

## 8. Physical Structure (relevant)
```text
bpm_prediction/
  configs/
    data/
    experiments/
    model/
  data/
    knowledge_graph/
    camunda_exports/
  src/
    adapters/ingestion/
    application/use_cases/
    domain/services/
    infrastructure/repositories/
  tools/
    ingest_topology.py
    sync_topology.py
    sync_stats.py
    visualize_topology.py
```

## 9. Runtime GNN Reference
For model-specific behavior and tensor assembly details, use:
1. `docs/GNN_RUNTIME_MVP2_5.MD`
