# MVP2.5 Stage 3 Plan (Updated)

Date: 2026-03-13  
Status: Planned  
Stage Name: Camunda Adapters Writing to the Same Port

## 1. Scope and Fixed Decisions

1. `bpmn_xml_content` is the single source for `ProcessStructure` (structural graph).
2. Runtime statistics are computed only by a separate periodic job, never inside train/eval runtime loop.
3. BPMS integration is optional and isolated; core experiments must run with `enrichment.enabled=false`.
4. Structure <-> runtime anchor is `activity_def_id` (`ACT_ID_` / `TASK_DEF_KEY_`); `task_id` remains runtime instance identifier only.
5. Parallel executions and cycles are mandatory in InstanceGraph modeling.
6. Version-safe binding is mandatory: structure and runtime must be matched by `PROC_DEF_ID_` first, not only by process key.

---

## 2. Key Risks and How We Mitigate Them

## 2.1 BPMN XML Parsing Risks

1. Invalid XML, namespace drift, broken BLOB payloads.
2. Loss of Camunda extensions under naive XML parsing.
3. Large BPMN models causing repeated parse overhead.
4. `isExecutable=false`, missing service task implementation, incomplete conditional expressions.

Mitigation:
1. Strict parser pipeline:
   - decode bytes with guarded fallback (`utf-8`, `utf-8-sig`, fail-fast if invalid),
   - namespace normalization and validation (`bpmn`, `camunda`, `xsi`, `bpmndi`),
   - ignore BPMNDI for control-flow graph extraction.
2. Two-parser strategy:
   - primary: `lxml` custom Camunda-aware parser,
   - optional stricter validator hook (future Java-side `Bpmn.readModelFromXml` service).
3. Parse cache keyed by `(proc_def_id, deployment_id, resource_name)`.
4. Quarantine strategy for bad definitions:
   - store parser error + proc_def_id in diagnostics storage,
   - skip broken definition without stopping pipeline.

## 2.2 Structure <-> Runtime Binding Risks

1. Mismatched versions when binding by `proc_key` only.
2. Multi-instance: same activity definition produces many runtime instances.
3. Parallel token execution tree differs from static BPMN.
4. CallActivity creates child process instance with independent lifecycle.

Mitigation:
1. Always join runtime events to exact `PROC_DEF_ID_`.
2. Use canonical node-per-execution IG with `task_id` / `act_inst_id`.
3. Persist both:
   - canonical execution graph,
   - collapsed projection for fast aggregation.
4. CallActivity support:
   - Stage 3 default: Supernode aggregated child-state features,
   - next option: Hierarchical parent-child graph mode.

## 2.3 Ongoing Statistics Risks

1. Heavy recompute in runtime loops.
2. Race conditions and duplicate counters.
3. Drift between runtime events and structural DTO stats.

Mitigation:
1. Dedicated `ProcessStatisticsRefreshService` with watermark.
2. Incremental delta updates only.
3. Idempotent aggregation keys `(proc_def_id, activity_def_id, window_id)`.

## 2.4 Runtime Cleanup and Scale Risks

1. `REMOVAL_TIME_` can be absent or mostly `NULL` for legacy data (typical for pre-7.10 history/TTL setups).
2. Cleanup backlog can make coverage unstable during the day (records marked for cleanup but not yet deleted).
3. `ACT_RU_EXECUTION` can be empty for completed/cleaned instances while history is still partially available.
4. Parent/child CallActivity history can be asymmetrically cleaned and produce stale child aggregates.
5. Extreme parallel multi-instance can explode row counts even in activity-centric mode.

Mitigation:
1. If `REMOVAL_TIME_` is unsupported or effectively not populated for legacy slices, treat those records as durable ("eternal") and do not use that as fallback trigger by itself.
2. Add fallback hysteresis:
   - trigger fallback only after low coverage appears in 2+ consecutive fetch cycles, or
   - use stabilized coverage rule `max(previous_coverage, current_coverage)` for transient cleanup oscillations.
3. In execution-centric mode, auto-degrade to activity-centric when runtime execution linkage is missing for more than configured threshold (default 50%).
4. Mark child-derived CallActivity features as `potentially_stale=true` (or skip them) under low coverage/fallback mode.
5. Add anti-explosion guard for high-MI cases:
   - per-case row caps,
   - fetch-time aggregation/sampling by `activity_def_id` + loop buckets.

---

## 3. Stage Split (Final)

## 3.1 Stage 3.1 - Runtime InstanceGraph Ingestion

Goal: ingest Camunda runtime/history SQL data and build execution-level IG.

## 3.2 Stage 3.2 - BPMN Structure Ingestion from `bpmn_xml_content`

Goal: parse process structure, call bindings, and write structural DTO via knowledge port.

## 3.3 Stage 3.3 - Periodic Statistics + Version Refresh

Goal: incremental stats refresh and structure enrichment outside runtime loop.

## 3.4 Optional Track - BPMS Enrichment Adapter

Goal: enrich runtime and structure attributes without coupling core Camunda ingestion path.

---

## 4. Architecture Boundaries (Do Not Mix)

## 4.1 Mandatory Core Camunda Flow

1. `CamundaRuntimeAdapter` -> `IInstanceGraphPort`
2. `CamundaBpmnAdapter` -> `IKnowledgeGraphPort`
3. `ProcessStatisticsRefreshService` reads both ports and writes enriched structural statistics back to `IKnowledgeGraphPort`.

## 4.2 Optional BPMS Flow

1. `BusinessDataAdapter` -> `IBusinessDataPort`
2. `BusinessDataEnricherService` enriches:
   - instance events (runtime attributes),
   - structure DTO (node/edge/business context).
3. Controlled only by config switch.

---

## 5. Contracts and DTOs

## 5.1 Port Contracts

1. `ICamundaRuntimePort`
   - `fetch_historic_activity_events(...)`
   - `fetch_historic_task_events(...)`
   - `fetch_identity_links(...)`
   - `fetch_call_links(...)`
   - `fetch_runtime_execution_tree(...)`
2. `ICamundaBpmnPort`
   - `fetch_procdef_catalog(...)`
   - `fetch_bpmn_xml(proc_def_id: str)`
3. `IInstanceGraphPort`
   - `save_instance_events(process_name, version_key, events)`
   - `save_instance_graph(process_name, version_key, graph)`
   - `save_instance_projection(process_name, version_key, projection)`
   - `get_instance_events(..., since=None, until=None)`
   - `get_last_watermark(...)`
   - `set_last_watermark(...)`
4. `IKnowledgeGraphPort`
   - `save_process_structure(...)`
   - `get_process_structure(...)`
   - `get_graph_for_visualization(process_name, version_key, min_edge_frequency=0)`
5. `IBusinessDataPort` (optional)
   - `fetch_task_context(task_ids)`
   - `fetch_user_context(user_ids)`
   - `fetch_case_context(case_ids)`

## 5.2 DTO Updates

1. `ProcessEventDTO`
   - `case_id`, `task_id`, `act_inst_id`, `activity_def_id`, `activity_name`, `activity_type`
   - `start_time`, `end_time`, `duration_ms`
   - `assignee`, `candidate_groups`
   - `proc_def_id`, `proc_def_key`, `proc_def_version`
   - `execution_id`, `parent_execution_id`, `call_proc_inst_id`
2. `ProcessStructureDTO`
   - `version`
   - `allowed_edges`
   - `edge_statistics`
   - `node_statistics`
   - `node_attributes`
   - `edge_attributes`
   - `call_activity_bindings`
3. `InstanceStatisticsDTO`
   - `node_execution_count`
   - `edge_transition_count`
   - `loop_counts_by_activity_def_id`
   - `parallel_count_by_activity_def_id`
   - `duration_stats`

---

## 6. Detailed Handling for Parallel, Cycles, CallActivity

## 6.1 Parallel and Cycles

1. Canonical IG model:
   - one node per execution instance (`act_inst_id` or `task_id`),
   - edge by event time ordering plus parent/execution relations.
2. Cycle handling:
   - repeated occurrences of same `activity_def_id` stay as distinct nodes in canonical IG,
   - collapsed projection aggregates `loop_count`.
3. Parallel handling:
   - separate nodes for concurrent tokens,
   - `parallel_count` computed from overlapping intervals and execution tree.

## 6.2 CallActivity Strategy (Recommended)

Default in Stage 3:
1. Supernode mode:
   - parent IG keeps CallActivity node,
   - attach aggregated child-process features (status, duration, running nodes, incidents).
2. Why:
   - stable complexity,
   - supports parallel parent+child execution,
   - low risk for first rollout.

Planned extension:
1. Hierarchical mode:
   - explicit parent-child graph link,
   - child embedding fed into parent GNN branch.

## 6.3 Runtime Cleanup Resilience Rules (Stage 3.1)

1. Legacy `REMOVAL_TIME_` behavior:
   - if column is missing or mostly unfilled for old slices, data is treated as durable history,
   - low coverage must not be inferred from missing `REMOVAL_TIME_` alone.
2. Coverage hysteresis:
   - fallback is triggered only after 2+ consecutive low-coverage fetches,
   - transient oscillations from cleanup backlog should not immediately switch mode.
3. Execution-tree availability gate:
   - in `execution-centric` mode, if execution linkage is unavailable for >50% of sampled cases, auto-degrade to `activity-centric` with warning.
4. CallActivity cleanup asymmetry:
   - if parent-child linkage is incomplete due to cleanup, child-derived aggregates are skipped or flagged as stale.

## 6.4 High-Volume Multi-Instance Protection

1. Add per-case upper bounds for processed events to avoid memory spikes.
2. For extreme MI (`1000+` parallel repetitions), enable fetch-time reduction:
   - aggregate by `activity_def_id`,
   - keep loop-counter buckets,
   - preserve transition signal for collapsed projection.
3. If limits are exceeded, keep canonical graph valid and emit truncation diagnostics.

---

## 7. BPMN XML Ingestion Plan (Stage 3.2)

Pipeline:
1. Load bytes from `ACT_GE_BYTEARRAY` using exact `proc_def_id` binding.
2. Decode and validate XML.
3. Parse control-flow elements:
   - start/end/intermediate/boundary events,
   - user/service/script/businessRule/send/receive/manual tasks,
   - gateways (XOR/AND/OR/event-based),
   - subprocesses, callActivity,
   - sequenceFlow with condition.
4. Parse Camunda extensions:
   - `asyncBefore`, `asyncAfter`, properties, listeners, I/O mappings metadata anchors.
5. Ignore BPMNDI for graph topology.
6. Build `ProcessStructureDTO` and persist to `IKnowledgeGraphPort`.

Failure policy:
1. Invalid XML -> quarantine record + warning log + skip process definition.
2. Missing optional extension fields -> soft warning, continue.
3. Broken call binding -> mark unresolved binding in DTO for later diagnostics.

---

## 8. Runtime Stats Refresh Plan (Stage 3.3)

`ProcessStatisticsRefreshService`:
1. Reads new events since watermark from `IInstanceGraphPort`.
2. Computes delta metrics:
   - node counts,
   - transition counts,
   - loop counts,
   - parallel counts,
   - duration aggregates.
3. Merges stats into structure DTO in `IKnowledgeGraphPort`.
4. Commits new watermark.

Scheduling:
1. cron or interval based.
2. per-process/per-version filtering.
3. lookback window for late-arriving records.

Critical guarantee:
1. no stats recomputation in online train/eval/predict loops.

---

## 9. SQL Storage and Draft Queries

SQL location:
`src/adapters/ingestion/camunda/sql/`

Subfolders:
1. `runtime/`
2. `structure/`
3. `stats/`
4. `enrichment/` (BPMS optional)

Required query packs:
1. Runtime history events by `PROC_DEF_ID_` and window.
2. Task definition mapping (`TASK_DEF_KEY_` -> `activity_def_id`).
3. Identity links for candidate users/groups.
4. BPMN bytes by deployment and resource name.
5. Call hierarchy (`SUPER_PROCESS_INSTANCE_ID_`, `CALL_PROC_INST_ID_`).
6. Multi-instance helpers (`loopCounter`, execution tree).
7. `REMOVAL_TIME_` support probe (via metadata/introspection) to switch cleanup predicates safely.
8. High-volume MI fallback query (aggregated/sampled mode) for per-case explosion protection.

---

## 10. Recommended Improvements Beyond Baseline

1. Keep parser implementation pluggable:
   - `lxml` parser now,
   - optional JVM strict validator adapter later.
2. Introduce parser diagnostics artifact:
   - unresolved bindings,
   - dropped malformed nodes/flows count.
3. Add `mode=supernode|hierarchical` for CallActivity handling in config.
4. Keep graph retrieval through `IKnowledgeGraphPort` only, including visualization path.
5. In low-coverage/fallback modes, skip child-specific CallActivity aggregates or mark them `potentially_stale` to avoid false confidence.

---

## 11. Configuration Design

```yaml
data:
  source: "camunda"
  process_name: "procurement"

camunda:
  enabled: true
  dsn: "..."
  process_filters: ["procurement"]
  from_ts: null
  to_ts: null
  history_cleanup_aware: true
  fallback_min_coverage_ratio: 0.05
  coverage_hysteresis_fetches: 2
  legacy_removal_time_policy: "treat_as_eternal"  # treat_as_eternal | strict
  bpmn:
    parser: "lxml"                  # lxml | jvm_validator
    strict_namespaces: true
    quarantine_on_error: true

instance_graph:
  mode: "hybrid"                    # canonical + collapsed
  canonical_mode: "activity-centric"  # activity-centric | execution-centric
  execution_tree_depth_limit: 4
  execution_missing_degrade_threshold: 0.5
  max_execution_nodes_per_case: 5000
  extreme_mi_row_threshold: 10000
  extreme_mi_strategy: "aggregate"   # aggregate | sample
  call_activity_mode: "supernode"   # supernode | hierarchical

stats_refresh:
  enabled: true
  cron: "*/30 * * * *"
  lookback_hours: 24

enrichment:
  enabled: false
  provider: "bpms"
```

---

## 12. Implementation Steps

## 12.1 Stage 3.1

1. Implement `CamundaRuntimeAdapter` + runtime SQL loader.
2. Implement `InstanceGraphAssemblerService`:
   - canonical IG from activity instances,
   - collapsed projection with loop/parallel counters.
3. Persist both canonical and projection artifacts via `IInstanceGraphPort`.
4. Add call links in runtime payload (`CALL_PROC_INST_ID_`, parent ids).
5. Add cleanup-awareness path:
   - `REMOVAL_TIME_` predicate where supported,
   - auto retry without predicate for legacy schemas,
   - legacy policy `treat_as_eternal` for historical slices with missing/NULL removal timestamps.
6. Add fallback hysteresis for unstable cleanup windows (2+ consecutive low-coverage fetches).
7. Add auto-degradation `execution-centric -> activity-centric` when runtime execution linkage coverage is below threshold.
8. Add child-aggregate safety for CallActivity (skip/flag stale aggregates under low coverage or fallback mode).
9. Add high-MI explosion guard with aggregated/sampled fetch strategy.

## 12.2 Stage 3.2

1. Implement `CamundaBpmnAdapter` + structure SQL loader.
2. Implement `BpmnStructureParserService` (Camunda-aware, namespace-safe).
3. Build and persist `ProcessStructureDTO`.
4. Resolve callActivity binding metadata and persist unresolved states for diagnostics.

## 12.3 Stage 3.3

1. Implement `ProcessStatisticsRefreshService`.
2. Add watermark persistence.
3. Add incremental merge into structure DTO.
4. Expose scheduler entrypoint.

## 12.4 Optional BPMS

1. Add `IBusinessDataPort` + no-op adapter.
2. Add enrichment hooks in assembler/refresh services.
3. Enforce full backward compatibility when disabled.

---

## 13. Test Plan (Expanded)

## 13.1 Parser and BPMN Robustness

1. invalid XML -> quarantine and continue.
2. missing/odd namespaces -> controlled failure/warning.
3. extensionElements parsing retained.
4. boundary events and subprocess parsing correctness.
5. callActivity binding resolution modes (`latest`, `version`, `deployment`).

## 13.2 Runtime and IG

1. parallel execution produces separate canonical nodes.
2. cycle occurrence increments loop counters.
3. activity_def_id mapping remains stable across task instances.
4. callActivity links child process correctly in supernode mode.
5. execution-centric mode auto-degrades to activity-centric when execution linkage coverage is too low.
6. extreme MI case triggers aggregation/sampling guard without OOM and still returns valid graph.

## 13.3 Stats Refresh

1. incremental watermark update.
2. idempotent repeated refresh.
3. runtime loop does not trigger heavy stats job.
4. cleanup backlog oscillation does not trigger immediate fallback due to hysteresis rule.

## 13.4 Optional BPMS

1. disabled enrichment causes zero behavioral changes.
2. partial BPMS payload does not break core ingestion.
3. join-key mismatch produces warning only.

## 13.5 Cleanup resilience add-ons

1. missing `REMOVAL_TIME_` column uses retry path and does not crash.
2. legacy records with `REMOVAL_TIME_ = NULL` are treated as durable history and do not cause false fallback.
3. parent/child cleanup asymmetry marks CallActivity child aggregates as stale or skips them safely.
4. low coverage fallback requires 2+ consecutive failures (hysteresis).

## 13.6 Non-Regression Gates

1. `pytest -m mvp1_regression -v`
2. `pytest tests/domain/test_eopkg_models_forward.py tests/domain/test_eopkg_gatv2_dual_encoder.py -v`
3. `pytest tests/ -v`

---

## 14. Definition of Done

1. Stage 3.1:
   - canonical and collapsed IG are built from Camunda runtime data,
   - parallel and cycle behavior is correctly represented,
   - cleanup-aware logic works for both modern and legacy Camunda histories.
2. Stage 3.2:
   - BPMN bytes are parsed into valid `ProcessStructureDTO`,
   - structure-runtime anchor works by `activity_def_id` and `proc_def_id`.
3. Stage 3.3:
   - periodic statistics refresh works incrementally with watermark,
   - no heavy stats in online loops.
4. BPMS:
   - adapter remains optional and isolated.
5. Quality:
   - all new tests pass,
   - high-volume MI protection avoids graph-construction blowups,
   - MVP1 regression shield remains green.
