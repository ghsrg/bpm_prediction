# ADR-0002: Offline Topology and Runtime Separation

Date: 2026-04-27
Status: Accepted

## Context

MVP2.5 separates knowledge preparation from model runtime to prevent leakage,
keep experiments reproducible, and make knowledge backends interchangeable.

The active architecture documents and worklogs consistently state that topology
and statistics must be prepared offline before train/eval/infer.

## Decision

Topology and statistics preparation are offline pipelines.

Train/eval/infer consume prebuilt repository artifacts through
`IKnowledgeGraphPort`.

Train/eval/infer must not rebuild topology or statistics synchronously from raw
sources.

## Consequences

Positive:

1. clearer anti-leakage boundary,
2. reproducible knowledge artifacts,
3. backend portability across file, in-memory, and Neo4j repositories,
4. simpler runtime responsibility for model training and evaluation.

Negative:

1. users must run preparation commands before research-grade experiments,
2. missing artifacts require explicit fallback or fail-fast policy,
3. more operational discipline is required around artifact timelines.

## Runtime Rules

1. Use `ingest-topology` or `sync-topology` to prepare structure artifacts.
2. Use `sync-stats` or `sync-stats-backfill` to prepare stats snapshots.
3. Runtime graph builders may load structure/stats through `IKnowledgeGraphPort`.
4. Runtime graph builders must not call raw source adapters to rebuild topology.

## Affected Files

- `main.py`
- `tools/ingest_topology.py`
- `tools/sync_topology.py`
- `tools/sync_stats.py`
- `tools/sync_stats_backfill.py`
- `src/domain/services/dynamic_graph_builder.py`
- `src/infrastructure/repositories/`

## Related

- `docs/ARCHITECTURE_MVP2_5.MD`
- `docs/DATA_FLOWS_MVP2_5.MD`
- `docs/TARGET_ARCHITECTURE.MD`
- `docs/worklogs/Finish_MVP2_5_Plan.MD`

