# PROJECT_CONTEXT.md

Updated: 2026-03-21
Purpose: operational context for current MVP2.5 state after Stage 4.2 updates.

## 1. Current State

Implemented and active:
1. Offline structure ingestion (`ingest-topology`, `sync-topology`).
2. Offline stats enrichment (`sync-stats`) with immutable snapshots.
3. Repository-backed runtime consumption (`IKnowledgeGraphPort`) in train/eval.
4. Stage 3.2 BPMN structure ingestion and Stage 3.1 runtime adapters.
5. Stage 3.4 strict snapshot policy:
   - immutable
   - strict composite key
   - JSON-only stats payload
   - as-of lookup
   - no TTL.
6. Stage 4.2 stats contract/runtime integration:
   - universal stats contract v1 (`stats_contract_version=1.0`)
   - producer quality gate in `sync-stats`
   - consumer quality gate in `DynamicGraphBuilder`
   - config-driven stats mapping with `encoding` support (`identity|log1p|z-score`).

## 2. MVP1 Baseline Contract (must stay green)

Stable path:
`XESAdapter -> RawTrace -> PrefixPolicy -> PrefixSlice -> GraphBuilder -> GraphTensorContract -> Trainer`

Mandatory checks:
1. `py -3 tools/architecture_guard.py`
2. `pytest -m mvp1_regression -v`
3. `pytest tests/ -v`

## 3. Runtime Semantics (Stage 3.4 + 4.2)

1. `sync-stats --as-of <ISO>` creates historical snapshot for selected cutoff.
2. Without `--as-of`, effective timestamp is derived from available process events (auto `max(event_ts)` after train-cut).
3. `stats_time_policy = strict_asof` uses prefix last-event timestamp for lookup.
4. `stats_time_policy = latest` uses latest snapshot/structure.
5. Stats quality gate may disable stats tensors without failing the whole run (policy-driven fallback).

## 4. Config Ownership

1. `data`/`mapping`: source, adapter, backend, mapping details.
2. `experiment`: run mode and split/timing controls.
3. `mapping.graph_feature_mapping`: stats feature matrix mapping + per-feature `encoding`.
4. `sync_stats.quality_gate` and `mapping.graph_feature_mapping.stats_quality_gate`: producer/consumer quality policy.

## 5. Current Priority

Immediate next step:
1. synchronize canonical docs with Stage 4.2 implementation details.
2. finalize MVP2.5 closure checklist (`Finish_MVP2_5_Plan.MD`) against completed Stage 4.2 worklogs.
3. run dissertation-goal alignment review and technical-debt audit.

## 6. Priority of Truth

1. `AGENT_GUIDE.MD`
2. `PROJECT_CONTEXT.md`
3. `docs/ARCHITECTURE_GUIDELINES.MD`
4. `docs/ARCHITECTURE_RULES.MD`
5. `docs/TARGET_ARCHITECTURE.MD`
6. `docs/ARCHITECTURE_MVP2_5.MD`
7. `docs/DATA_MODEL_MVP2_5.MD`
8. `docs/DATA_FLOWS_MVP2_5.MD`
9. `docs/LLD_MVP2_5.MD`
10. `docs/EVF_MVP2_5.MD`
