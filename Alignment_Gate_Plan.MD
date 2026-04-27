# Activity-to-Topology Alignment Gate Plan

Date: 2026-04-24  
Status: Analysis and implementation plan only  
Scope: eliminate weak/unsafe activity-to-topology alignment behavior in `sync-stats` and cover it with real-object tests

## 1. Goal

Make `Activity-to-Topology Alignment Gate`:

1. deterministic,
2. explainable,
3. robust to known real-world naming shifts,
4. strict enough for research-grade runs,
5. covered by integration-style tests using real XES, real `ProcessStructureDTO`, real repository objects, and real `sync_stats_main` runs.

This plan does not change code yet. It defines the target design and test pack.

## 2. Current State

Current implementation lives in [tools/sync_stats.py](C:/Users/korsr/PycharmProjects/bpm_prediction/tools/sync_stats.py).

Key observations:

1. `_build_alignment_summary(...)` currently compares normalized `event.activity_def_id` against `_dto_activity_ids(dto)`.
2. `_dto_activity_ids(dto)` currently takes `dto.nodes[].id` only, or falls back to `allowed_edges`.
3. Matching is effectively exact-string matching after trim/null normalization.
4. `node_coverage` is computed against all DTO node ids, including structural-only nodes such as gateways/start/end when they are present in BPMN-derived topology.
5. Existing tests prove the gate is wired, but they mostly validate binary mismatch/pass behavior and do not cover realistic naming shifts deeply.

Relevant existing evidence:

1. Current gate config parsing: [tools/sync_stats.py](C:/Users/korsr/PycharmProjects/bpm_prediction/tools/sync_stats.py:475)
2. Current summary logic: [tools/sync_stats.py](C:/Users/korsr/PycharmProjects/bpm_prediction/tools/sync_stats.py:784)
3. Current gate application path: [tools/sync_stats.py](C:/Users/korsr/PycharmProjects/bpm_prediction/tools/sync_stats.py:1516)
4. Existing application tests: [tests/application/test_sync_stats_tool.py](C:/Users/korsr/PycharmProjects/bpm_prediction/tests/application/test_sync_stats_tool.py:694)
5. Existing end-to-end BPMN -> XES -> stats -> GNN cycle: [tests/integration/test_bpmn_xes_stats_to_gnn_cycle.py](C:/Users/korsr/PycharmProjects/bpm_prediction/tests/integration/test_bpmn_xes_stats_to_gnn_cycle.py:1)

## 3. Problem Breakdown

### 3.1 Exact-id matching is too weak

Known real mismatches are not model bugs, but naming representation shifts:

1. XES classifier adds lifecycle suffixes such as `Approve+COMPLETE`.
2. BPMN node id may be `Task_Approve`, while event activity is `Approve`.
3. Case and separator drift: `Task A`, `Task_A`, `task-a`.
4. Some logs align to BPMN `node.name`, not BPMN `node.id`.

With current exact-id logic, these scenarios become false negatives.

### 3.2 `node_coverage` is semantically wrong for BPMN topology

Current `node_coverage = matched_unique / all_nodes`.

That is unsafe because `all_nodes` may include:

1. `startEvent`
2. `endEvent`
3. gateways
4. other structural-only nodes that are never expected to appear in event logs

So a perfectly healthy log can still look under-covered.

### 3.3 Current gate is tool-local and not reusable

The logic sits inside `tools/sync_stats.py`, which makes it:

1. harder to test as a reusable policy,
2. harder to reason about as a domain rule,
3. harder to reuse later in preflight checks or research pipelines.

### 3.4 Diagnostics are not rich enough

Current summary gives ratios and top unmatched activities, but not enough to answer:

1. which matcher rule resolved an activity,
2. whether matching used `id` or `name`,
3. whether a match was ambiguous,
4. whether failure came from exact mismatch or from structural-only node inflation.

## 4. Target State

Introduce a deterministic alignment service with explicit matching policy and richer diagnostics.

### 4.1 Core principle

The gate should validate alignment against **loggable activity nodes**, not against every structural node in the BPMN graph.

### 4.2 Safe matching policy

Matching should be deterministic and conservative, with ordered strategies:

1. exact match to `node.id`,
2. exact match to `node.name` when unique,
3. normalized match with case-folding and separator collapse,
4. optional classifier/lifecycle suffix stripping for XES-driven alignment,
5. optional adapter-specific canonicalization rules.

The gate should not use fuzzy string similarity or edit distance in the first implementation. That would be too risky and hard to explain.

### 4.3 Ambiguity must fail closed

If one event activity maps to more than one node candidate after normalization, the gate must not silently choose one.

It should:

1. mark the activity as ambiguous,
2. report diagnostics,
3. fail or quarantine according to `on_fail`.

## 5. Proposed Architecture

### 5.1 Extract a dedicated domain service

Create a reusable service, for example:

1. `src/domain/services/activity_topology_alignment_service.py`

Responsibility:

1. build alignment candidate index from `ProcessStructureDTO`,
2. normalize event activities and node candidates,
3. evaluate alignment metrics,
4. return structured diagnostics.

Why domain service:

1. inputs are domain entities (`ProcessEventDTO`, `ProcessStructureDTO`),
2. logic is pure and deterministic,
3. tooling should orchestrate it, not own it.

### 5.2 Introduce explicit policy objects

Recommended shape:

1. `AlignmentGateConfig`
2. `AlignmentMatchPolicy`
3. `AlignmentSummary`

These may start as dataclasses or typed dicts.

Minimum policy fields:

1. `min_event_match_ratio`
2. `min_unique_activity_coverage`
3. `min_node_coverage`
4. `candidate_node_fields`
5. `ignore_structural_only_nodes`
6. `strip_classifier_suffix`
7. `normalize_case`
8. `collapse_separators`
9. `on_fail`

### 5.3 Define loggable node filtering

The service should separate:

1. loggable nodes
2. structural-only nodes

Default loggable set should include executable/activity-like BPMN tags such as:

1. `userTask`
2. `manualTask`
3. `serviceTask`
4. `scriptTask`
5. `businessRuleTask`
6. `sendTask`
7. `receiveTask`
8. `callActivity`
9. `subProcess`

Default structural-only set should exclude from alignment denominator:

1. `startEvent`
2. `endEvent`
3. all gateways
4. boundary/intermediate helper nodes unless there is a proven event-log mapping rule

This is the most important semantic fix for `node_coverage`.

### 5.4 Keep compatibility modes

Do not break current runs abruptly. Introduce matcher profiles:

1. `legacy_exact`
2. `safe_normalized`
3. `research_strict`

Recommended behavior:

1. `legacy_exact` preserves current semantics for comparison/debug.
2. `safe_normalized` becomes the practical default for `sync-stats`.
3. `research_strict` uses the same canonicalization, but stricter thresholds and `on_fail=raise` or `skip_snapshot`.

## 6. Matching Rules

### 6.1 Canonicalization rules

Canonicalization should be deterministic and transparent:

1. trim,
2. lowercase,
3. collapse spaces, `_`, `-` into one normalized separator,
4. optionally remove classifier suffixes like `+complete`, `+start`,
5. optionally strip known BPMN technical prefixes only when explicitly enabled.

Important:

1. prefix stripping such as `Task_` should not be globally hardcoded without a policy switch,
2. `node.name` fallback should be allowed only when unique after normalization,
3. if both `node.id` and `node.name` produce conflicting candidates, report ambiguity.

### 6.2 Match provenance

`AlignmentSummary` should include how matches were obtained:

1. `matched_by_exact_id`
2. `matched_by_exact_name`
3. `matched_by_normalized_id`
4. `matched_by_normalized_name`
5. `matched_by_classifier_strip`
6. `ambiguous_activity_count`
7. `unmatched_event_activities_top`

This is necessary to trust the gate and debug dataset-specific policies.

## 7. Implementation Plan

### Phase 1. Freeze semantics and metrics

1. Document that alignment denominator uses only loggable nodes.
2. Freeze a canonical definition for:
   - `event_match_ratio`
   - `unique_activity_coverage`
   - `node_coverage`
3. Define that ambiguity is a failure condition.
4. Decide which matcher profile is default for exploratory runs.

Deliverable:

1. updated contract note in docs and plan-approved matching semantics.

### Phase 2. Extract service from `tools/sync_stats.py`

1. Move alignment computation into a domain service.
2. Keep `tools/sync_stats.py` responsible only for:
   - config parsing,
   - orchestration,
   - persistence,
   - summary/report formatting.
3. Keep old behavior behind `legacy_exact` profile during migration.

Deliverable:

1. alignment logic no longer buried inside CLI tool code.

### Phase 3. Add canonical candidate index

1. Build candidate index from DTO nodes using:
   - `id`
   - `name`
   - normalized forms
2. Filter candidate nodes by loggable BPMN types.
3. Add ambiguity detection.
4. Record match provenance.

Deliverable:

1. `AlignmentSummary` with richer diagnostics and correct denominator semantics.

### Phase 4. Integrate gate into `sync-stats`

1. Replace current `_build_alignment_summary(...)` usage with the new service.
2. Preserve current `on_fail` options:
   - `write_with_flag`
   - `skip_snapshot`
   - `raise`
3. Extend persisted `metadata.stats_contract.alignment` with diagnostic details.

Deliverable:

1. same high-level gate behavior, better semantics and better evidence.

### Phase 5. Lock research profile

For research-grade runs define one locked profile:

1. ignore structural-only nodes,
2. enable safe canonicalization,
3. fail on ambiguity,
4. require high thresholds.

Recommended starting thresholds:

1. `min_event_match_ratio >= 0.9`
2. `min_unique_activity_coverage >= 0.9`
3. `min_node_coverage >= 0.8`

`node_coverage` may be set to `0.9` after observing real datasets, but `0.8` is a safer initial value when optional tasks exist.

## 8. Test Strategy

Main principle: no mock-only confidence. The backbone must be real-object tests.

### 8.1 Test layers

1. Small pure-service tests for matcher correctness.
2. Real-object application tests via `sync_stats_main`, `FileBasedKnowledgeGraphRepository`, real `ProcessStructureDTO`, and generated XES.
3. End-to-end integration tests via `sync_topology_main` + `sync_stats_main` on BPMN + XES.

Unit tests are supplementary. Real-object tests are mandatory.

## 9. Real-Object Test Matrix

### T1. Exact aligned baseline

Purpose:

1. prove healthy pass path.

Setup:

1. DTO node ids exactly match XES activity ids.
2. thresholds are high.

Expected:

1. gate passes,
2. all alignment ratios near `1.0`,
3. stats snapshot is written.

### T2. Classifier suffix shift

Purpose:

1. validate the known `Approve+COMPLETE` style mismatch.

Setup:

1. BPMN/DTO nodes use raw task ids or names,
2. XES is read with classifier-enabled activity ids or manually generated suffixed activity ids.

Expected:

1. `legacy_exact` rejects,
2. canonicalized matcher with `strip_classifier_suffix=true` passes,
3. summary shows `matched_by_classifier_strip > 0`.

### T3. BPMN id vs node name shift

Purpose:

1. validate real BPMN `id` vs business label mismatch.

Setup:

1. DTO node id = `Task_Approve`,
2. DTO node name = `Approve`,
3. XES activity = `Approve`.

Expected:

1. exact-id mode rejects,
2. name-aware unique matching passes,
3. provenance shows name-based resolution.

### T4. Case and separator shift

Purpose:

1. handle `Task A` vs `task_a` vs `task-a`.

Setup:

1. same business activity represented with different case/separator style.

Expected:

1. normalized matcher passes,
2. exact legacy mode rejects or partially rejects.

### T5. Structural-only nodes do not depress `node_coverage`

Purpose:

1. eliminate false negative due to gateways/start/end.

Setup:

1. topology from real BPMN with gateways/start/end,
2. log contains only executable tasks.

Expected:

1. gate computes denominator over loggable nodes only,
2. `node_coverage` reflects executable-task coverage rather than total BPMN node count.

This scenario should reuse a real BPMN topology similar to [tests/integration/test_bpmn_xes_stats_to_gnn_cycle.py](C:/Users/korsr/PycharmProjects/bpm_prediction/tests/integration/test_bpmn_xes_stats_to_gnn_cycle.py:1).

### T6. Partial overlap

Purpose:

1. verify threshold behavior, not just binary zero-vs-one mismatch.

Setup:

1. two activities align,
2. one activity is unknown or shifted beyond allowed normalization.

Expected:

1. ratios land in intermediate range,
2. pass/fail depends on threshold,
3. unmatched top list contains the bad activity.

### T7. Ambiguous name collision

Purpose:

1. prove fail-closed behavior.

Setup:

1. two DTO nodes normalize to the same candidate label,
2. XES activity maps to both.

Expected:

1. gate rejects,
2. summary records ambiguity,
3. no silent auto-resolution happens.

### T8. `on_fail=skip_snapshot`

Purpose:

1. keep current operational policy verified under realistic mismatch.

Setup:

1. failing alignment scenario from T2/T3/T7,
2. `on_fail=skip_snapshot`.

Expected:

1. summary reports skipped snapshot,
2. repository structure remains without written stats.

### T9. `on_fail=write_with_flag`

Purpose:

1. keep exploratory behavior available.

Setup:

1. failing alignment scenario,
2. `on_fail=write_with_flag`.

Expected:

1. snapshot is written,
2. metadata contains `is_aligned=false`,
3. diagnostics are persisted.

### T10. `on_fail=raise`

Purpose:

1. verify research fail-fast mode.

Setup:

1. failing alignment scenario,
2. `on_fail=raise`.

Expected:

1. tool returns error,
2. no snapshot is persisted.

## 10. Proposed Test File Layout

### Pure service tests

1. `tests/domain/test_activity_topology_alignment_service.py`

Purpose:

1. matcher ordering,
2. ambiguity detection,
3. denominator semantics,
4. summary/provenance correctness.

### Real-object sync-stats tests

1. `tests/application/test_sync_stats_alignment_gate_real_objects.py`

Use:

1. generated XES,
2. `FileBasedKnowledgeGraphRepository`,
3. real `ProcessStructureDTO`,
4. real `sync_stats_main`.

This should be the main test pack for shift scenarios T1-T10.

### BPMN + XES end-to-end integration

1. `tests/integration/test_alignment_gate_bpmn_xes_e2e.py`

Use:

1. real BPMN file,
2. `sync_topology_main`,
3. `sync_stats_main`,
4. repository reload,
5. optional `DynamicGraphBuilder` smoke check after aligned snapshot.

This validates that topology extraction shape and alignment semantics stay compatible.

## 11. Suggested Implementation Order

1. Freeze semantics for loggable-node denominator.
2. Extract alignment service without changing behavior.
3. Add matcher profiles and provenance diagnostics.
4. Add real-object test pack for exact, classifier, name, separator, partial, ambiguous scenarios.
5. Switch `sync-stats` default from implicit exact-id behavior to `safe_normalized`.
6. Add one locked research config example and acceptance thresholds in docs.

## 12. Acceptance Criteria

The debt can be considered closed when all points below are true:

1. alignment logic is no longer embedded only inside `tools/sync_stats.py`,
2. `node_coverage` no longer counts structural-only BPMN nodes in the acceptance denominator,
3. classifier suffix and `id` vs `name` mismatch scenarios are either resolved or explicitly rejected with clear diagnostics,
4. ambiguity fails closed,
5. real-object tests cover exact, shifted, partial, and ambiguous cases,
6. `write_with_flag`, `skip_snapshot`, and `raise` are all verified on real mismatch scenarios,
7. one research-grade profile is documented with locked thresholds.

## 13. Risks and Guardrails

1. Do not introduce fuzzy matching in phase 1. It can silently create false positives.
2. Do not auto-strip prefixes like `Task_` globally without explicit policy. It is dataset-sensitive.
3. Do not use all BPMN nodes in `node_coverage`.
4. Do not let `node.name` fallback silently resolve ambiguous duplicates.
5. Keep `legacy_exact` available until the new matcher is proven by the real-object test pack.

## 14. Recommended Immediate Next Task

Implementation task should start with:

1. extracting a dedicated alignment service,
2. fixing denominator semantics to loggable nodes only,
3. adding the real-object test pack before switching default matcher profile.
