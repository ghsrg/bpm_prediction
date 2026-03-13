# MVP2.5 Stage 3.1 Detailed Plan

Date: 2026-03-13  
Status: Planned  
Scope: Camunda Runtime Adapters + Canonical Instance Graph (cleanup-resilient)

## 0. Sources of Truth Used

1. `mvp2_5_Stage3_Plan.MD` (Stage 3 split, ports, fallback direction, optional BPMS isolation).
2. `docs/TARGET_ARCHITECTURE.MD` (Clean/Hexagonal boundaries, ports/adapters, resilience and logging).
3. `AGENT_GUIDE.MD` (contract-first, no architecture violations, explicit planning before code).

This plan is strictly for **Stage 3.1** only.

---

## 1. Stage 3.1 Objective and Boundaries

## 1.1 Objective

Build a resilient runtime ingestion pipeline for Camunda 7 that always returns a valid graph artifact for GNN, even under aggressive history cleanup.

## 1.2 Hard guarantees

1. `CamundaRuntimeAdapter` and `InstanceGraphAssemblerService` **never raise** because of missing/partially cleaned history.
2. Pipeline always returns one of:
   - `mode="canonical"` (activity- or execution-centric), or
   - `mode="fallback_structure_only"`.
3. All fetch/build methods are idempotent.
4. Fallback behavior is explicit, logged, and measurable.

## 1.3 Non-goals of Stage 3.1

1. No BPMN XML structural parsing implementation (Stage 3.2).
2. No Neo4j implementation (later stage).
3. No mandatory BPMS enrichment in runtime core.

---

## 2. Traceability Matrix to Mandatory Camunda Nuances

| Requirement | Planned handling |
|---|---|
| 1. `ACT_HI_ACTINST` != execution tree | Combine historic activity instances (base) + bounded runtime execution tree summary (parallel/scope refinement). |
| 2. Execution tree explosion | Depth-limited traversal + keep only leaf/multi-instance-body/key scopes + aggregate in collapsed projection. |
| 3. Unreliable watermark | Combined watermark = `max(start_time, end_time)` + lookback window + optional `REV_` guard. |
| 4. Multi-instance vars | Fetch from `ACT_RU_VARIABLE` + `ACT_HI_VARINST` by execution hierarchy, merge with body/child semantics. |
| 5. Boundary events semantics | Distinct edge types: `cancellation_edge` (interrupting) vs `fork_edge` (non-interrupting). |
| 6. Delayed identity links | LEFT JOIN strategy, nullable-safe DTO fields and late-binding merge. |
| 7. History cleanup resilience | `REMOVAL_TIME_` guard where available + auto fallback on missing column + coverage checks + structure-only fallback. |
| 8. General requirements | Always valid output, canonical mode switch (`activity-centric`/`execution-centric`), new config flags, `removal_time` in DTO, idempotent fetch/build. |

---

## 3. Planned Architecture Changes (Stage 3.1)

## 3.1 New/Updated Ports

1. `ICamundaRuntimePort`
   - robust fetch API with diagnostics payload.
2. `IInstanceGraphPort`
   - idempotent save/upsert for canonical graph + projection + diagnostics.
3. Existing `IKnowledgeGraphPort`
   - used only for fallback `structure_only` graph seeds.

## 3.2 Services

1. `CamundaRuntimeAdapter`
   - SQL execution, schema-aware query builder, cleanup-aware fallback behavior.
2. `InstanceGraphAssemblerService`
   - converts fetched runtime artifacts into canonical IG + collapsed projection.
3. `RuntimeFetchDiagnosticsService` (can be private helper)
   - computes coverage and fallback indicators.

## 3.3 Repositories

1. `InMemoryInstanceGraphRepository` for Stage 3.1 development/test.
2. Interface remains future-compatible for Neo4j implementation.

---

## 4. Config Extensions (Required)

Add under `camunda` / `instance_graph` / `runtime_fetch` blocks:

```yaml
camunda:
  history_cleanup_aware: true
  lookback_hours: 2
  removal_time_guard_default: true
  fetch_batch_size: 10000
  fallback_min_coverage_ratio: 0.05

instance_graph:
  canonical_mode: "activity-centric"     # activity-centric | execution-centric
  execution_tree_depth_limit: 4           # recommended 3..5
  max_execution_nodes_per_case: 5000
  call_activity_mode: "supernode"        # supernode | hierarchical

runtime_fetch:
  enable_rev_guard: true
  rev_guard_source: "ACT_RU_EXECUTION"
  on_missing_removal_time: "auto_fallback"   # auto_fallback | strict
```

Behavior defaults:
1. `history_cleanup_aware=true` by default.
2. `execution_tree_depth_limit=4` default.
3. `canonical_mode="activity-centric"` default (faster and safer).

---

## 5. DTO/Contract Updates

## 5.1 ProcessEventDTO

Add optional diagnostics field:
1. `removal_time: Optional[datetime] = None`

Keep existing optional fields for resilience:
1. `assignee`, `candidate_groups`, `task_id`, `execution_id`, `parent_execution_id`, `call_proc_inst_id`.

## 5.2 Fetch diagnostics DTO (new)

`RuntimeFetchDiagnosticsDTO`:
1. `rows_raw`
2. `rows_after_cleanup_filter`
3. `rows_after_dedup`
4. `cleaned_instances_percent`
5. `history_coverage_percent`
6. `fallback_triggered`
7. `fallback_reason`
8. `removal_time_supported_tables`

## 5.3 Build result DTO (new)

`InstanceGraphBuildResultDTO`:
1. `graph`
2. `projection`
3. `mode` (`canonical_activity`, `canonical_execution`, `fallback_structure_only`)
4. `diagnostics`

---

## 6. SQL and Cleanup-Resilient Query Strategy

## 6.1 SQL file structure

`src/adapters/ingestion/camunda/sql/runtime/`
1. `historic_activity_events.sql`
2. `historic_tasks.sql`
3. `identity_links.sql`
4. `execution_tree.sql`
5. `multi_instance_variables.sql`
6. `call_links.sql`

## 6.2 REMOVAL_TIME_ guard strategy

For each ACT_HI_* query:
1. Attempt with predicate:
   - `AND (REMOVAL_TIME_ IS NULL OR REMOVAL_TIME_ > CURRENT_TIMESTAMP)`
2. If SQL error indicates missing column:
   - log warning,
   - mark table as `removal_time_not_supported`,
   - retry query without this predicate.

Implementation detail:
1. Prefer one-time column introspection via `INFORMATION_SCHEMA.COLUMNS` cache.
2. Keep retry-based fallback anyway for DB variance safety.

## 6.3 Watermark-safe filtering

Use effective event timestamp:
1. `event_ts = COALESCE(END_TIME_, START_TIME_)` for event ordering.
2. Incremental read window starts at:
   - `last_watermark_ts - lookback_hours`.
3. Optional REV guard:
   - track `max_rev_seen` and include events where `REV_ >= max_rev_seen - rev_lookback_margin` if available.

## 6.4 Mandatory diagnostics per fetch

Compute and store:
1. `cleaned_instances_percent`
2. `history_coverage_percent`
3. `fallback_triggered`

Coverage computation policy:
1. If raw count known: `coverage = after_filter / raw_count`.
2. If raw count unknown: estimate from case/activity continuity heuristics.
3. If estimation impossible: mark `coverage=unknown` and fallback only on empty critical payload.

---

## 7. Runtime Ingestion Algorithm (Stage 3.1)

## 7.1 Step A - Fetch process/version scope

1. Resolve target `proc_def_id` set.
2. Bind all runtime/history fetches to explicit proc_def_ids.

## 7.2 Step B - Fetch base activity instances

1. Pull `ACT_HI_ACTINST` as business-level backbone.
2. Preserve key fields: `ACT_INST_ID_`, `ACTIVITY_ID_`, `START_TIME_`, `END_TIME_`, `TASK_ID_`, `CALL_PROC_INST_ID_`, `REMOVAL_TIME_`.

## 7.3 Step C - Fetch execution-tree refinement (bounded)

1. Pull `ACT_RU_EXECUTION` (or historical equivalent where required).
2. Traverse/assemble only up to `execution_tree_depth_limit`.
3. Keep:
   - leaf executions,
   - multi-instance body scopes,
   - key scope nodes (`IS_SCOPE_`, `IS_CONCURRENT_`).
4. Drop deep non-informative branches; summarize into aggregated counters.

## 7.4 Step D - Fetch multi-instance variables

1. Read `loopCounter`, `nrOfInstances`, `nrOfActiveInstances`, `nrOfCompletedInstances` from:
   - `ACT_RU_VARIABLE` (runtime),
   - `ACT_HI_VARINST` (history backup).
2. Merge by execution id and scope semantics.

## 7.5 Step E - Fetch delayed identity links safely

1. LEFT JOIN identity links to tasks/executions.
2. Keep NULL-safe defaults (empty list/groups).
3. No hard failure on missing links.

## 7.6 Step F - Merge, deduplicate, and normalize DTOs

1. Build `ProcessEventDTO` rows.
2. Dedup key:
   - `(case_id, act_inst_id|task_id, activity_def_id, start_time, end_time)`.
3. Persist idempotently.

## 7.7 Step G - Health gate and fallback decision

Trigger fallback if any critical condition true:
1. Empty event payload after fetch.
2. Coverage ratio `< fallback_min_coverage_ratio` (default 5%).
3. Severe inconsistency detected (e.g., no activity IDs for majority of rows).

Fallback action:
1. Log warning, do not raise.
2. Build `fallback_structure_only` graph using `IKnowledgeGraphPort` structure + available batch stats.

---

## 8. Canonical IG Build Modes

## 8.1 Mode A - activity-centric (default)

1. Node = activity instance event.
2. Edge = temporal successor inside case.
3. Execution tree contributes only metadata:
   - `parallel_count`, `scope_depth`, `loop_count`.
4. Best for scale and cleanup resilience.

## 8.2 Mode B - execution-centric (accurate)

1. Node set includes execution-scoped activity instances.
2. Edge types:
   - temporal edges,
   - parent-child execution scope edges,
   - call edges (supernode hooks).
3. Enforced guards:
   - depth limit,
   - per-case node cap.
4. If cap exceeded -> degrade to activity-centric with warning.

---

## 9. Boundary Events and Edge Typing Rules

1. Interrupting boundary event:
   - create `cancellation_edge` from interrupted activity path.
2. Non-interrupting boundary event:
   - create `fork_edge` to parallel boundary path.
3. Keep edge-type semantics in both canonical and projection outputs.

---

## 10. Fallback Graph Design (`fallback_structure_only`)

When history is missing/mostly cleaned:
1. Load structure DTO by process/version from `IKnowledgeGraphPort`.
2. Build graph from `allowed_edges` and structure nodes.
3. Enrich node/edge features with latest available aggregated stats (if present).
4. Set metadata:
   - `mode="fallback_structure_only"`
   - `fallback_triggered=true`
   - `fallback_reason`
5. Return valid graph object (never exception due to missing runtime history).

---

## 11. Idempotency Rules

1. Fetch methods are deterministic for same `(proc_def_id set, time window, config)`.
2. Lookback-induced duplicates are removed by event dedup key.
3. Save methods use upsert semantics with stable keys.
4. Re-running same interval updates diagnostics only, not duplicate graph nodes.

---

## 12. Observability and Operational Logging

Mandatory logs:
1. Start/end per fetch step with row counts.
2. Whether `REMOVAL_TIME_` guard was applied or skipped.
3. Coverage and cleanup percentages.
4. Canonical mode selected / degraded.
5. Fallback trigger reason.

Long operations (>10s):
1. `tqdm` progress bars for batched fetch/assembly phases.

Metrics per run:
1. `cleaned_instances_percent`
2. `history_coverage_percent`
3. `fallback_triggered`
4. `canonical_mode`
5. `execution_tree_truncated_count`

---

## 13. File-Level Implementation Plan

## 13.1 New files (planned)

1. `src/domain/ports/camunda_runtime_port.py`
2. `src/domain/ports/instance_graph_port.py`
3. `src/domain/entities/process_event.py` (or extend existing DTO module)
4. `src/domain/entities/runtime_fetch_diagnostics.py`
5. `src/application/services/instance_graph_assembler_service.py`
6. `src/adapters/ingestion/camunda_runtime_adapter.py`
7. `src/infrastructure/repositories/in_memory_instance_graph_repository.py`
8. `src/adapters/ingestion/camunda/sql/runtime/*.sql`

## 13.2 Existing files to update

1. `src/cli.py`
   - add config keys (`history_cleanup_aware`, `execution_tree_depth_limit`, `canonical_mode`, `call_activity_mode`).
2. `src/application/use_cases/trainer.py`
   - wire runtime ingestion path to always accept fallback result.
3. `configs/experiments/mvp2_*.yaml`
   - add new camunda/runtime config blocks with defaults.
4. `docs/DATA_MODEL_MVP2.MD`
   - document `ProcessEventDTO.removal_time` and diagnostics.
5. `docs/DATA_FLOWS_MVP2.MD`
   - document fallback and dual canonical modes.

---

## 14. Test Plan for Stage 3.1

## 14.1 Adapter resilience tests

1. `test_runtime_adapter_removal_time_supported_query`
2. `test_runtime_adapter_missing_removal_time_column_auto_fallback`
3. `test_runtime_adapter_empty_after_cleanup_returns_no_exception`
4. `test_runtime_adapter_combined_watermark_with_lookback_dedup`

## 14.2 Canonical build tests

1. `test_instance_graph_activity_centric_build`
2. `test_instance_graph_execution_centric_build_depth_limit`
3. `test_execution_tree_explosion_degrades_to_activity_mode`
4. `test_multi_instance_variables_merge_from_runtime_and_history`
5. `test_boundary_interrupting_creates_cancellation_edge`
6. `test_boundary_non_interrupting_creates_fork_edge`
7. `test_identity_links_null_safe_left_join`

## 14.3 Fallback and diagnostics tests

1. `test_fallback_structure_only_on_low_coverage`
2. `test_fallback_structure_only_on_empty_history`
3. `test_diagnostics_metrics_presence`
4. `test_idempotent_double_run_no_duplicate_nodes`

## 14.4 Regression gates

1. `pytest -m mvp1_regression -v`
2. `pytest tests/domain/test_eopkg_models_forward.py tests/domain/test_eopkg_gatv2_dual_encoder.py -v`
3. Stage 3.1 new suite:
   - `pytest tests/adapters/camunda tests/application/test_instance_graph_* -v`

---

## 15. Better Options and Recommendations

## 15.1 Recommended now (low risk)

1. Default canonical mode: `activity-centric`.
2. Execution tree used as bounded refinement only.
3. Fallback-to-structure policy enabled by default.
4. Supernode call-activity mode for runtime link propagation.

## 15.2 Optional optimization after baseline stabilizes

1. Add precomputed daily snapshot table/materialized view for heavy joins.
2. Add adaptive depth limit by case complexity.
3. Introduce hybrid sampling for huge execution trees.

---

## 16. Acceptance Criteria (Stage 3.1 DoD)

1. Stage 3.1 runtime ingestion returns valid graph object in all tested scenarios, including 99% cleaned history.
2. No hard failure on missing `REMOVAL_TIME_` columns.
3. Fallback mode produces usable graph for GNN input.
4. Canonical modes are selectable and stable (`activity-centric` / `execution-centric`).
5. Diagnostics and cleanup coverage metrics are always emitted.
6. Idempotency tests are green.
7. MVP1 regression shield remains green.
