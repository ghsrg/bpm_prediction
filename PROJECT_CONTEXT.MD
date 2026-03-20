# PROJECT_CONTEXT.md

Updated: 2026-03-19
Purpose: operational context for current MVP2.5 state after Stage 3.4 freeze.

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

## 2. MVP1 Baseline Contract (must stay green)

Stable path:
`XESAdapter -> RawTrace -> PrefixPolicy -> PrefixSlice -> GraphBuilder -> GraphTensorContract -> Trainer`

Mandatory checks:
1. `py -3 tools/architecture_guard.py`
2. `pytest -m mvp1_regression -v`
3. `pytest tests/ -v`

## 3. Stage 3.4 Runtime Semantics

1. `sync-stats --as-of <ISO>` creates historical snapshot for selected cutoff.
2. Without `--as-of`, snapshot timestamp is current UTC.
3. `stats_time_policy = strict_asof` uses prefix last-event timestamp for lookup.
4. `stats_time_policy = latest` uses latest snapshot/structure.

## 4. Config Ownership

1. `data`/`mapping`: source, adapter, backend, mapping details.
2. `experiment`: run mode and split/timing controls.

## 5. Current Priority

Immediate next step after docs freeze:
1. full integration of stats features into train/inference/eval_drift via dedicated graph feature encoding policy (fit/transform/state).

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
