# ADR-0003: Immutable JSON Stats Snapshots

Date: 2026-04-27
Status: Accepted

## Context

Stage 3.4 and Stage 4.2 introduced repository-backed stats snapshots for
temporal-safe enrichment. The canonical docs require immutable snapshots,
strict identity keys, JSON-only payloads, and as-of lookup.

This supports reproducible experiments and avoids coupling the runtime tensor
path to backend-specific normalized statistic subgraphs.

## Decision

Stats snapshots are immutable and append-only.

Snapshot identity is:

```text
tenant_id + process_name + version_key + proc_def_id + knowledge_version
```

Stats payload is JSON-only.

Do not model stats as normalized `NodeStat` or `EdgeStat` graph nodes in the
current MVP2.5 contract.

Do not use TTL for stats snapshots.

## Consequences

Positive:

1. snapshot provenance is explicit,
2. historical lookup is reproducible,
3. file and Neo4j backends can share the same logical payload contract,
4. tensor assembly can use one `metadata.stats_index` contract.

Negative:

1. snapshot storage grows append-only,
2. cleanup must be explicit and policy-driven if needed later,
3. queries over individual statistic fields are less graph-native.

## Runtime Rules

1. `sync-stats` writes immutable snapshots.
2. Snapshot payload fields remain JSON payloads.
3. `metadata.stats_index` is the canonical access map for tensor assembly.
4. As-of lookup selects the latest snapshot where `snapshot.as_of_ts <= requested_as_of`.
5. No TTL-based expiry is allowed for MVP2.5 stats snapshots.

## Affected Files

- `tools/sync_stats.py`
- `tools/sync_stats_backfill.py`
- `src/domain/entities/process_structure.py`
- `src/infrastructure/repositories/knowledge_graph_repository_factory.py`
- `src/infrastructure/repositories/file_based_knowledge_graph_repository.py`
- `src/infrastructure/repositories/neo4j_knowledge_graph_repository.py`
- `src/domain/services/dynamic_graph_builder.py`

## Related

- `docs/DATA_MODEL_MVP2_5.MD`
- `docs/LLD_MVP2_5.MD`
- `docs/DATA_FLOWS_MVP2_5.MD`
- `docs/worklogs/Finish_MVP2_5_Plan.MD`

