# MVP2.5 Stage 3.4 Detailed Plan

Updated: 2026-03-19  
Status: Planned  
Scope: `sync-stats` enrichment pipeline (Camunda-first), immutable knowledge snapshots, GNN-ready statistical feature layer.

## 0. Sources of Truth

1. `docs/worklogs/mvp2_5_general_plan.MD`
2. `docs/worklogs/mvp2_5_Stage3_3_Plan.MD`
3. `docs/worklogs/mvp2_5_Stage3_3_production-hardening_Plan.MD`
4. `docs/DATA_MODEL_MVP2_5.MD`
5. `docs/LLD_MVP2_5.MD`
6. `docs/ADAPTER_CAMUNDA_SQL.MD`

---

## 1. Stage 3.4 Goal

Build a production-grade `sync-stats` service that:
1. reads already ingested process structure (from `sync-topology` output),
2. enriches nodes/edges with Camunda runtime statistics (no BPMN re-parsing),
3. writes immutable versioned snapshots to knowledge storage,
4. exposes GNN-first features for training/eval/inference,
5. remains resilient to history cleanup and partial data loss.

Hard rule: structure topology comes from Stage 3.2. Stage 3.4 only enriches it.

---

## 2. In Scope / Out of Scope

## 2.1 In Scope

1. New command: `sync-stats`.
2. Camunda runtime extraction for node/edge/case aggregates.
3. Multi-window + decay statistics.
4. Confidence/freshness/coverage diagnostics.
5. Snapshot persistence and retrieval.

## 2.2 Out of Scope

1. BPMN parsing logic changes.
2. BPMS runtime connector implementation (only specification in this plan).
3. Neo4j-specific optimization (covered by Stage 4.x).

---

## 3. Target Runtime Flow

1. Load process structure (`process_name`, `version_key`, `proc_def_id`) from KG port.
2. Fetch Camunda runtime/history events for same identity scope.
3. Build windowed aggregates (`7d`, `30d`, `90d`, `all_time_decay`).
4. Compute robust stats, transition probabilities, resource patterns.
5. Merge into `ProcessStructureDTO` in stats sections only.
6. Write immutable snapshot `knowledge_version = kN`.
7. Emit diagnostics summary (coverage, fallback, cleanup impact).

## 3.1 Storage identity: process/tenant/version

Statistics are stored and queried by strict identity tuple:
1. `tenant_id` (nullable, but explicit in key model)
2. `process_name` (Camunda process key / dataset namespace)
3. `version_key` (`vNN`)
4. `proc_def_id` (one DTO per concrete definition)
5. `knowledge_version` (immutable snapshot sequence)

Rules:
1. No cross-tenant merge.
2. No cross-process merge.
3. No cross-version overwrite.
4. New refresh always appends a new `knowledge_version`, never mutates old snapshot.

Backend examples:
1. File backend: `.../<tenant_or_default>/<process_name>/<version_key>/<knowledge_version>.json`
2. Neo4j backend: labels/relations keyed by `tenant_id + process_name + version_key + proc_def_id`, stats by `knowledge_version`.

---

## 4. Work Packages (Implementation Steps)

## WP1. Contracts and Config Freeze

Files:
1. `docs/DATA_MODEL_MVP2_5.MD` (if needed, additive fields only)
2. `src/domain/entities/...` DTO module(s)
3. `configs/data/*.yaml` templates

Actions:
1. Freeze stats sections in `ProcessStructureDTO`:
   - `node_stats.windows`
   - `edge_stats.windows`
   - `gnn_features`
   - `stats_diagnostics`
2. Add `sync_stats` config block:
   - `enabled`
   - `windows_days`
   - `enable_decay`
   - `decay_factor_per_day`
   - `history_cleanup_aware`
   - `lookback_hours`
   - `coverage_threshold`
   - `confidence_weights`
3. Keep backward compatibility: if stats absent, consumers use defaults.

Exit criteria:
1. DTO schema stable and documented.
2. Config validation catches invalid windows/weights.

## WP2. Runtime Data Fetch Layer (Camunda)

Files:
1. `src/adapters/ingestion/camunda/sql/runtime/*.sql`
2. `src/adapters/ingestion/camunda/runtime_fetcher.py` (or equivalent)

Actions:
1. Implement batch SQL fetch by `(proc_def_id, time range)`.
2. Use cleanup guard where available:
   - `(REMOVAL_TIME_ IS NULL OR REMOVAL_TIME_ > NOW())`
3. Implement compatibility fallback when `REMOVAL_TIME_` missing.
4. Collect diagnostics:
   - `history_rows_total`
   - `history_rows_after_guard`
   - `history_coverage_percent`
   - `cleaned_instances_percent`

Exit criteria:
1. No per-node N+1 SQL.
2. Stable fetch on both old and new Camunda schemas.

## WP3. Aggregation Engine

Files:
1. `src/application/services/process_statistics_refresh_service.py`
2. `src/domain/services/stats_aggregator.py` (if split)

Actions:
1. Map runtime records to structure anchors via `activity_def_id`.
2. Compute node-level, edge-level, and case-level aggregates.
3. Implement robust metrics:
   - median, IQR, P95, winsorized mean, MAD, skewness, kurtosis.
4. Implement transition probabilities and entropy.
5. Implement loop and parallelism indicators.

Exit criteria:
1. Deterministic output for same input snapshot.
2. Aggregates are numeric-stable and NaN-safe.

## WP4. Snapshot Merge and Persistence

Files:
1. `src/infrastructure/repositories/*knowledge_graph*`
2. `src/application/use_cases/sync_stats.py` (or command handler)

Actions:
1. Merge stats into existing structure without topology mutation.
2. Append immutable snapshot with incremented `knowledge_version`.
3. Save refresh metadata:
   - `snapshot_ts`
   - `watermark_used`
   - `source_mode`
   - `fallback_triggered`

Exit criteria:
1. Previous snapshots remain unchanged.
2. New snapshot is queryable by version.

## WP5. CLI + Reporting + Tests

Files:
1. `main.py`
2. `tools/sync_stats.py` (or integrated CLI subcommand)
3. `tests/application/test_sync_stats_*.py`
4. `docs/worklogs/mvp2_5_Stage3_4_Report.MD` (future report)

Actions:
1. Add command: `python main.py sync-stats --config ... --out ...`.
2. Output JSON summary per run.
3. Add progress reporting with stage counters.
4. Add full test matrix (see Section 8).

Exit criteria:
1. Green tests.
2. Summary report includes per-process coverage and updated snapshot ids.

---

## 5. Required Config (Draft)

```yaml
mapping:
  knowledge_graph:
    backend: "file"             # file | neo4j
    path: "data/knowledge_graph"

  camunda_adapter:
    source: "files"             # files | mssql
    process_name: "procurement"
    tenant_id: null

sync_stats:
  enabled: true
  target_scope: "single_version"       # single_version | all_versions_for_process
  windows_days: [7, 30, 90]
  enable_decay: true
  decay_factor_per_day: 0.995
  history_cleanup_aware: true
  lookback_hours: 24
  coverage_threshold: 0.05
  fallback_policy: "snapshot_with_warning"   # snapshot_with_warning | skip
  enable_gnn_features: true
  enable_resource_features: true
  confidence_weights:
    sample_size: 0.4
    freshness: 0.3
    coverage: 0.3
```

Notes:
1. For full process sync across versions use `target_scope: all_versions_for_process`.
2. For tenant-aware runs set `camunda_adapter.tenant_id` or `tenant_filters`.
3. If tenant is not provided, repository uses reserved namespace `default`.

---

## 6. Full Camunda Statistics Catalog (What can be extracted)

## 6.1 Node-level (activity_def_id anchored)

| Group | Metric | Source | Aggregations | Typical output field |
|---|---|---|---|---|
| Throughput | executions_count | ACT_HI_ACTINST | count | node_stats.exec_count |
| Duration | duration_ms | ACT_HI_ACTINST | mean, median, min, max, p50, p75, p90, p95, p99, std, var, iqr, mad, winsorized_mean | node_stats.duration.* |
| Activity SLA | sla_breach_rate | derived | ratio, rolling ratio | node_stats.sla.breach_rate |
| Waiting | wait_before_start_ms | derived from predecessor end/start | mean, median, p95 | node_stats.wait.* |
| Rework | repeats_per_case | derived by activity repeats in case | mean, median, p95 | node_stats.rework.* |
| Loop | loop_density | activity repeats / case | mean, percentile | node_stats.loop_density |
| Active backlog | active_instances | runtime/history blend | current, avg per hour/day | node_stats.backlog.* |
| Errors | incident_rate | ACT_HI_INCIDENT (if available) | ratio, count | node_stats.quality.incident_rate |
| Cancellation | canceled_rate | delete reason / interrupting boundary | ratio | node_stats.quality.cancel_rate |
| Parallelism | parallel_token_count | ACT_RU_EXECUTION + HI mapping | mean, max, variance | node_stats.parallel.* |
| Freshness | last_seen_ts | derived | max timestamp, age_days | node_stats.freshness.* |
| Stability | drift_vs_prev_snapshot | derived | delta_pct | node_stats.drift.* |

## 6.2 Edge-level (source -> target)

| Group | Metric | Source | Aggregations | Typical output field |
|---|---|---|---|---|
| Transition volume | transition_count | ordered events | count | edge_stats.count |
| Transition probability | P(target|source) | derived | normalized ratio | gnn_features.transition_prob |
| Transition latency | delta_ms | predecessor->successor time | mean, median, p95, iqr | edge_stats.latency.* |
| Transition reliability | failed_after_edge_rate | incidents/deletions after edge | ratio | edge_stats.reliability.* |
| Rare path score | path_rarity | derived from frequency | percentile rank | edge_stats.rarity |
| Condition selectivity | condition_true_ratio | gateway+condition traces | ratio | edge_stats.condition.selectivity |
| Bounce-back | reverse_transition_rate | A->B then B->A | ratio | edge_stats.loopback_rate |

## 6.3 Case-level (process instance)

| Group | Metric | Source | Aggregations | Typical output field |
|---|---|---|---|---|
| Case cycle time | case_duration_ms | ACT_HI_PROCINST | mean, median, p95, p99 | case_stats.duration.* |
| Case complexity | nodes_visited, edges_visited | derived | mean, percentiles | case_stats.complexity.* |
| Variant | variant_id | sequence hash | top-k frequency, entropy | case_stats.variant.* |
| Rework depth | repeated steps count | derived | mean, max | case_stats.rework.* |
| Parallel spread | max concurrent branches | execution tree | mean, max | case_stats.parallel.* |
| Open case age | age_ms for active | runtime | mean, p95 | case_stats.active_age.* |

## 6.4 Resource-level (Camunda-only)

| Group | Metric | Source | Aggregations | Typical output field |
|---|---|---|---|---|
| Assignee load | tasks per assignee | ACT_HI_TASKINST | count per window, moving avg | resource_stats.assignee_load |
| Group load | tasks per candidate group | ACT_HI_IDENTITYLINK | count, share | resource_stats.group_load |
| Handover | user/group transition matrix | task/activity sequence | count, normalized prob, entropy | resource_stats.handover.* |
| Touch time | human work duration | task start/end | mean, median, p95 | resource_stats.touch_time.* |
| Automation ratio | service vs user task share | ACT_TYPE_ | ratio | resource_stats.automation_ratio |
| Performer diversity | unique assignees per activity | task history | count, gini | resource_stats.diversity.* |

## 6.5 Multi-window and decay modes

For every eligible metric:
1. fixed windows: `last_7d`, `last_30d`, `last_90d`.
2. cumulative decayed view: `all_time_decay`.

Aggregation families:
1. Count-based: sum, rate, normalized rate.
2. Time-based: mean, median, quantiles, robust estimators.
3. Distribution: std, var, iqr, mad, skewness, kurtosis.
4. Information-theoretic: entropy, gini, concentration.
5. Drift: delta vs previous snapshot/version.

---

## 7. GNN-First Feature Set (Recommended Priorities)

## 7.1 Data config mapping seed for GNN ingestion

To feed graph stats into tensors, data config must declare mapping (same idea as IG mapping):

```yaml
mapping:
  graph_feature_mapping:
    node_numeric:
      - name: node_exec_count_30d
        source: "process_structure.node_stats.windows.last_30d.exec_count"
      - name: node_duration_p95_30d
        source: "process_structure.node_stats.windows.last_30d.duration.p95"
      - name: node_loop_density_30d
        source: "process_structure.node_stats.windows.last_30d.loop_density"
    edge_numeric:
      - name: edge_transition_prob_30d
        source: "process_structure.edge_stats.windows.last_30d.transition_probability"
      - name: edge_latency_median_30d
        source: "process_structure.edge_stats.windows.last_30d.latency.median"
    global_numeric:
      - name: coverage_percent
        source: "process_structure.stats_diagnostics.history_coverage_percent"
```

Rules:
1. Mapping is explicit and versioned in data config, not hardcoded in model.
2. Missing attributes are zero-imputed with warning counter.
3. FeatureEncoder keeps deterministic order of mapped features.
4. Same mapping contract must work for both file and neo4j backends.

## Tier A (strong baseline, highest practical impact)

1. `transition_probability[source][target]`
2. `node_exec_count_windowed`
3. `node_duration_median/p95`
4. `edge_latency_median/p95`
5. `loop_density`
6. `parallel_token_variance_30d`
7. `resource_handover_entropy`
8. `freshness_score`
9. `confidence_score`

## Tier B (usually useful, medium complexity)

1. `winsorized_mean_duration`
2. `rare_path_score`
3. `condition_complexity_index`
4. `performer_diversity`
5. `automation_ratio`
6. `variant_entropy`

## Tier C (advanced/experimental)

1. `kurtosis/skewness` for duration tails.
2. `gateway branch imbalance index`.
3. `reverse_transition_rate`.
4. `edge anomaly z-score`.

Practical note:
1. In predictive process monitoring, robust time features + transition probabilities + rework/parallel signals consistently outperform plain frequency-only encodings.
2. Resource handover and workload signals usually improve human-task heavy processes.

---

## 8. BPMS Parameters to Add Later (Optional, but high value)

BPMS is not implemented in Stage 3.4, but schema should stay extensible.

## 8.1 Candidate BPMS enrichments

| Domain | Example fields | Aggregations | Why useful |
|---|---|---|---|
| Business amount | contract_amount, invoice_amount | raw, log-scale, quantiles, bucket ratios | strong driver of path and SLA |
| Risk | risk_score, compliance_risk_flag | mean, max, ratio | helps drift and anomaly separation |
| Priority | business_priority, urgency | distribution, transition-conditioned rates | impacts queue and delay |
| Customer/vendor | segment, country, tier | entropy, per-segment cycle time | contextual behavior shifts |
| Product/service | category, service_line | path frequencies by category | structural route differences |
| Org context | department, legal_entity | rates, handover matrix | organizational bottlenecks |
| Staffing | planned role vs actual role | mismatch rate | explains delays/rework |
| Financial outcomes | penalty, discount, loss | mean, tail risk | optimize cost-aware prediction |
| Policy | policy_version, rule_pack | version-conditioned metrics | governance and model alignment |

## 8.2 Integration principle for future

1. Keep BPMS as separate adapter and separate feature namespace.
2. Join key should be explicit (`business_key`, `task_id`, external mapping table).
3. Never block Camunda-only pipeline when BPMS is unavailable.

---

## 9. Test Plan for Stage 3.4

## 9.1 Unit tests

1. Window slicing correctness (`7d/30d/90d/all_time_decay`).
2. Decay formula correctness across snapshot intervals.
3. Robust stat functions (median, iqr, winsorized mean, MAD).
4. Transition probability normalization (sum to 1 per source, within tolerance).
5. Confidence/freshness computation monotonicity.

## 9.2 Integration tests

1. `sync-topology` -> `sync-stats` -> snapshot append.
2. Missing runtime rows -> warning + snapshot fallback (no crash).
3. Cleanup-aware fetch with and without `REMOVAL_TIME_` columns.
4. Multi-tenant process scope isolation.
5. Backward compatibility: training still runs when stats absent.

## 9.3 Regression tests

1. `pytest -m mvp1_regression -v`
2. Existing MVP2.5 sync tests.
3. New tests for DTO schema stability (required keys, additive-only evolution).

---

## 10. Deliverables Checklist

1. `sync-stats` CLI mode.
2. Camunda runtime aggregation service.
3. Extended snapshot schema with stats blocks.
4. Diagnostics JSON summary and logs.
5. Test suite for windows/decay/robustness/fallback.
6. Stage report with benchmark of one real dataset.

---

## 11. Open Decisions for User Selection

1. Which Tier A/B/C features to include in first production pass.
2. Which BPMS fields are available and reliable for joins.
3. Default fallback policy when coverage is low (`snapshot_with_warning` vs `skip`).
4. Snapshot cadence (`hourly`, `daily`, hybrid by process criticality).
