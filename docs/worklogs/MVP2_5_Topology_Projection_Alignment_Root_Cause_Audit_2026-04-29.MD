# MVP2.5 Topology Projection Alignment Root-Cause Audit

Date: 2026-04-29
Status: proposed implementation plan, no code changes in this audit
Scope: `collapse_for_prediction`, topology-to-activity alignment, structural tensors, allowed target mask

## Executive Summary

`collapse_for_prediction` is not just a gateway-removal option. It changes the process topology before the model receives `structural_edge_index`, `structural_edge_weight`, `struct_x`, and `allowed_target_mask`. The current implementation can produce valid tensors for the happy path, but it does not produce an explicit post-projection alignment artifact. Because of that, a drop in F1 after removing gateways can be caused by at least three different things that are currently hard to distinguish:

1. real modeling effect from less structural context;
2. projected topology losing edges/nodes due to missing `activity_vocab` mappings;
3. silent index repair or stale projection cache hiding a topology/class-space mismatch.

Recommendation: treat this as one technical debt item named `topology_projection_alignment`, but implement it in stages. Do not merge it with the separate parallel-branch target-mask semantics problem from `mismatch_fix_plan.md`.

Follow-up clarification: duplicate BPMN tasks with the same log activity label are a separate debt named `duplicate_activity_identity_ambiguity`. `topology_projection_alignment` should remain compatible with a future node-identity-aware mapping, but it should not solve duplicate-label disambiguation in the same implementation.

## Skills Used

- `brainstorming`: split the problem into architecture boundaries and implementation options.
- `systematic-debugging`: root-cause audit by tracing data flow from config to tensors and model forward.
- `writing-plans`: implementation plan with tests, checkpoints, and acceptance criteria.

## Relevant Existing Documentation

- `docs/adr/0007-topology-projection-alignment.md`
  - Defines the intended target state: any projection that mutates topology must produce explicit post-projection alignment before research-grade runs.
- `docs/current/architecture-debt.md`
  - Tracks `topology_projection_alignment` as active P0.
  - Tracks `activity_to_topology_alignment_gate` as closed; future projection-specific work belongs here.
- `docs/worklogs/MVP2_5_Canonical_Doc_Sync_and_Architecture_Debt_2026-04-24.MD`
  - Records the original debt: `collapse_for_prediction` can break indexing with structure and needs alignment after structure modification.
- `docs/worklogs/mismatch_fix_plan.md`
  - Describes a related but different issue: target-mask mismatch under parallelism where the next completed activity may be valid even if it is not structurally adjacent.

## Current Code Path

Config entry point:

- `configs/ui/config_catalog.yaml`
  - `mapping.graph_feature_mapping.topology_projection.gateway_mode`
  - supported values: `preserve`, `collapse_for_prediction`

Runtime path:

- `src/domain/services/dynamic_graph_builder.py`
  - `DynamicGraphBuilder.__init__` resolves `self.topology_gateway_mode`.
  - `build_graph()` resolves `activity_vocab`, compiles topology, builds `allowed_target_mask`, and attaches structural tensors.
  - `_resolve_compiled_topology()` builds:
    - `allowed_masks_by_src`
    - `structural_edge_index`
    - `structural_edge_weight`
    - `struct_x`
  - `_project_allowed_edge_paths_for_prediction()` performs the actual collapse.
  - `_classify_projection_nodes()` treats tasks/subprocesses as prediction nodes and gateways/events as transparent.
  - `_build_struct_x()` creates rows in `activity_vocab` class-space.

Model path:

- `src/domain/models/eopkg_models.py`
  - The model consumes `structural_edge_index` and `struct_x`.
  - If `structural_edge_index.max() >= node_count`, it currently applies modulo repair:
    - `structural_edge_index = structural_edge_index % node_count`
  - This can hide a structural-index mismatch instead of failing fast.

Batching path:

- `src/application/use_cases/trainer.py`
  - `_select_structural_payload_for_forward()` currently selects one structural payload from a PyG batch.
  - This is related to structural correctness but belongs to separate debt: `snapshot_homogeneous_batching`.

## What `collapse_for_prediction` Does

Input topology may contain BPMN technical nodes:

```text
TaskA -> Gateway_XOR -> TaskB
```

`collapse_for_prediction` projects through transparent nodes and emits prediction-space edges:

```text
TaskA -> TaskB
```

This is necessary because the model predicts activities/classes, while BPMN structure can contain nodes that are not target classes: gateways, start events, end events, intermediate events.

The important nuance: after collapse, topology is no longer the same graph as the BPMN graph. It is a derived projection. Therefore all downstream artifacts must be proven to use the same class-space:

- edge endpoints must be `activity_vocab` indices;
- mask indices must be `activity_vocab` indices;
- `struct_x` rows must match the same `activity_vocab` indices;
- edge weights must correspond to projected paths, not only direct BPMN edges.

## Root Cause

Root cause is missing post-projection alignment as a first-class contract.

The code currently relies on implicit invariants:

1. every projected prediction node is present in `activity_vocab`;
2. every structural edge endpoint maps to the same class index used by `y`;
3. `struct_x` rows are ordered by the same class-space as `structural_edge_index`;
4. projected edge weights remain meaningful when a gateway path is collapsed;
5. the topology cache is invalidated whenever projection-relevant topology metadata changes;
6. invalid structural indices will never reach the model.

Some happy-path tests prove these invariants for simple cases. The production pipeline does not yet prove them for every run.

## Specific Failure Modes

### 1. Silent Edge Loss On Missing Vocab

In `_resolve_compiled_topology()`, projected edges are skipped when either endpoint is absent from `activity_vocab`:

```python
src_idx = activity_vocab.get(src_token)
dst_idx = activity_vocab.get(dst_token)
if src_idx is None or dst_idx is None:
    continue
```

This avoids crashes, but it can silently remove structural signal. If many projected edges are skipped, F1 can drop while the run still looks technically successful.

Required fix: collect diagnostics for skipped edges and fail/warn according to configured `on_fail`.

### 2. Fallback To Raw Edges When `dto.nodes` Is Missing

`_project_allowed_edge_paths_for_prediction()` falls back to original `allowed_edges` if `dto.nodes` is absent.

For a BPMN-derived structure, this means gateway/event edges can reach the class-space mapping step. Since gateways/events are usually absent from `activity_vocab`, those edges can then be skipped.

Required fix: diagnostics must report that projection was requested but could not classify nodes.

### 3. No Projection Artifact

The compiler returns tensors and masks but no durable explanation of the projection:

- which source nodes were transparent;
- which projected edges were created;
- which source paths contributed to each projected edge;
- which projected edges were skipped;
- whether all projected endpoints map to `activity_vocab`;
- whether `struct_x` rows and structural edge indices share the same class-space.

Required fix: introduce a projection diagnostics artifact and include it in run outputs/logs. Do not rely only on per-sample tensors.

### 4. Cache Key Omits Node Metadata

`_topology_cache_key()` includes gateway mode, activity vocab, mapping config, normalized edges, and stats identity. It does not include a fingerprint of `dto.nodes`.

In `collapse_for_prediction`, node metadata controls classification into `prediction` vs `transparent`. If `allowed_edges` stay the same but node metadata changes, the cached projected topology can become stale.

Required fix: include normalized projection-relevant node metadata in the cache key.

### 5. Model Silently Repairs Invalid Structural Indices

`EoPkgModel.forward()` applies modulo when structural edge indices exceed the node count. That keeps the forward pass running, but it corrupts topology semantics.

For research-grade runs, invalid structural indices should fail fast with a clear diagnostic.

Required fix: replace silent modulo repair with explicit validation policy. If fallback is needed for legacy runs, make it opt-in and observable.

### 6. Projection Alignment And Parallel Target Semantics Are Coupled In Symptoms

F1 drop after topology changes may look like the target-mask issue from `mismatch_fix_plan.md`, but these are separate causes:

- `topology_projection_alignment`: Are structural tensors and masks internally aligned after topology projection?
- `duplicate_activity_identity_ambiguity`: Does one log activity label map to more than one BPMN prediction node?
- `target_mask_parallel_semantics`: Is the observed next completed event valid under parallel branch semantics?

Required fix: implement projection diagnostics first so later mask-semantics work is not debugging through a moving topology target.

### 7. Duplicate Activity Labels Are A Separate Identity Problem

If two parallel BPMN tasks have the same name and the log only stores `concept:name`, the current class-space cannot know which concrete BPMN node produced the event. Structure may know the node ids, but `y`, activity-level stats, and `allowed_target_mask` collapse them into one activity class.

This is not fixed by `collapse_for_prediction`. It is also not the same as parallel target-mask semantics. Parallelism affects valid ordering; duplicate labels affect identity. The projection alignment work should only detect and report that a projected BPMN node-to-activity-label mapping is non-injective, then preserve compatibility for a future identity layer.

## Existing Test Coverage

Good coverage already exists for happy paths:

- `tests/domain/test_dynamic_graph_builder_masks.py`
  - verifies collapsed gateway path produces prediction mask;
  - verifies collapse stops at the next prediction node;
  - verifies active mask is unioned with structural mask.
- `tests/integration/test_bpmn_xes_stats_to_gnn_cycle.py`
  - verifies BPMN/XES path can collapse `GatewayChoice`;
  - verifies projected TaskA -> TaskB and TaskA -> TaskC edges;
  - verifies projected edge weights;
  - verifies `struct_x` rows for tasks.

Missing coverage:

- diagnostics for skipped edges;
- diagnostics for missing `dto.nodes`;
- cache invalidation when only node metadata changes;
- fail-fast behavior for invalid structural indices;
- run-level artifact proving projection alignment;
- explicit policy around `on_fail`.

## Is It Critical?

For the current `loan_v1_v4_simulated` sync-stat backfill result, this is not the cause of low `node_coverage`. That earlier issue was explained by BPMN nodes that are not expected to appear in logs, especially gateways/events and simulator coverage gaps.

For model training and F1 comparisons, this is critical. If removing gateways changes F1, we currently cannot prove whether the model got a valid projected topology or a silently degraded topology. That makes experimental conclusions weak.

Priority remains P0 for research-grade experiments.

## Should This Be Split?

Recommended split:

1. Keep `topology_projection_alignment` as one technical debt item.
2. Implement it in stages:
   - projection compiler diagnostics;
   - builder integration and fail policy;
   - model structural-index validation;
   - experiment/run artifact and docs.
3. Keep `duplicate_activity_identity_ambiguity` separate.
4. Keep `target_mask_parallel_semantics` separate.
5. Keep `snapshot_homogeneous_batching` separate.

Reason: the projection issue has one coherent boundary: topology projection must output tensors only after proving class-space alignment. Splitting the debt into unrelated tickets would make the same invariant harder to enforce. Staging the implementation is enough.

## Preferred Design

Add a small projection/alignment component rather than expanding `DynamicGraphBuilder` further.

Suggested component:

```text
src/domain/services/topology_projection_alignment.py
```

Responsibility:

- classify nodes for projection;
- build projected edges and source paths;
- map projected endpoints to `activity_vocab`;
- detect non-injective BPMN node -> activity label mapping and report it as external identity debt;
- build diagnostics;
- validate class-space alignment;
- expose a fail/warn result to `DynamicGraphBuilder`.

Suggested result object:

```python
@dataclass(frozen=True)
class TopologyProjectionResult:
    projected_edge_paths: dict[tuple[str, str], list[list[str]]]
    prediction_nodes: set[str]
    transparent_nodes: set[str]
    diagnostics: TopologyProjectionDiagnostics
```

Suggested diagnostics:

```python
@dataclass(frozen=True)
class TopologyProjectionDiagnostics:
    gateway_mode: str
    node_count: int
    edge_count: int
    prediction_node_count: int
    transparent_node_count: int
    projected_edge_count: int
    skipped_projected_edges: list[dict[str, str]]
    missing_vocab_nodes: list[str]
    duplicate_activity_labels: dict[str, list[str]]
    missing_node_metadata: bool
    struct_x_rows: int | None
    num_classes: int
    structural_edge_index_max: int | None
    is_aligned: bool
    failure_reasons: list[str]
```

Do not put a large diagnostics dictionary into every per-sample tensor contract unless the tensor-purity rules are intentionally updated. Prefer:

- internal diagnostics in compiled topology cache;
- run-level JSON artifact;
- strict failure before graph contract is returned when `on_fail: raise`.

Suggested config:

```yaml
mapping:
  graph_feature_mapping:
    topology_projection:
      gateway_mode: collapse_for_prediction
      diagnostics_enabled: true
      on_fail: raise   # raise | warn | disable_struct
```

## Implementation Plan

### Phase 1 - Add Projection Alignment Diagnostics Tests

- [ ] Add unit tests for a pure projection/alignment component.
- [ ] Cover simple gateway collapse:
  - source: `TaskA -> Gateway -> TaskB`;
  - expected projected edge: `TaskA -> TaskB`;
  - expected transparent node: `Gateway`;
  - expected `is_aligned: true`.
- [ ] Cover missing activity vocab:
  - source projected edge endpoint absent from `activity_vocab`;
  - expected `skipped_projected_edges` and `is_aligned: false`.
- [ ] Cover missing `dto.nodes`:
  - `gateway_mode: collapse_for_prediction`;
  - no node metadata available;
  - expected `missing_node_metadata: true` and failure/warn depending policy.
- [ ] Cover multiple transparent paths:
  - two gateway paths from same source to same destination;
  - expected diagnostics preserve all source paths.

Verification command:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/domain/test_topology_projection_alignment.py
```

### Phase 2 - Extract Projection Logic From `DynamicGraphBuilder`

- [ ] Create `src/domain/services/topology_projection_alignment.py`.
- [ ] Move projection classification and BFS path logic out of `DynamicGraphBuilder`.
- [ ] Keep behavior equivalent for current passing tests.
- [ ] Return `TopologyProjectionResult` with diagnostics.
- [ ] Do not change model behavior in this phase.

Verification command:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/domain/test_dynamic_graph_builder_masks.py tests/domain/test_topology_projection_alignment.py
```

### Phase 3 - Integrate Diagnostics And `on_fail`

- [ ] Extend graph feature mapping parser to read:
  - `topology_projection.diagnostics_enabled`;
  - `topology_projection.on_fail`.
- [ ] In `_resolve_compiled_topology()`, validate diagnostics before returning compiled tensors.
- [ ] Implement policies:
  - `raise`: stop run with actionable error;
  - `warn`: keep run but record diagnostics;
  - `disable_struct`: remove structural payload and continue with logged reason.
- [ ] Add run-level JSON output for projection diagnostics in experiment/sync stats paths.

Verification command:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/domain/test_dynamic_graph_builder_masks.py tests/integration/test_bpmn_xes_stats_to_gnn_cycle.py
```

### Phase 4 - Fix Cache Invalidation

- [ ] Add projection-relevant `dto.nodes` fingerprint to `_topology_cache_key()`.
- [ ] Include only stable fields needed for classification:
  - node id;
  - type;
  - activity_type;
  - bpmn_tag if used.
- [ ] Add regression test:
  - same `allowed_edges`;
  - changed node type from gateway-like to task-like or reverse;
  - expected cache miss and changed projection result.

Verification command:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/domain/test_dynamic_graph_builder_masks.py
```

### Phase 5 - Replace Silent Structural Index Repair

- [ ] Audit `EoPkgModel.forward()` structural edge index validation.
- [ ] Replace modulo repair with explicit policy.
- [ ] For strict runs, raise a clear error:
  - edge index max;
  - node count;
  - shape of `struct_x`;
  - config mode.
- [ ] If legacy fallback is still needed, gate it behind a config flag and emit diagnostics.

Verification command:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/domain tests/integration/test_bpmn_xes_stats_to_gnn_cycle.py
```

### Phase 6 - Update Documentation

- [ ] Update `docs/adr/0007-topology-projection-alignment.md` from `Proposed` to `Accepted` after implementation.
- [ ] Update `docs/current/architecture-debt.md` with implemented state and residual risks.
- [ ] Add a worklog with:
  - tests added;
  - diagnostics artifact examples;
  - any config migration notes.
- [ ] Update `AGENTS.MD` only if first-read agent rules change.

## Acceptance Criteria

- `collapse_for_prediction` produces a projection diagnostics artifact.
- Every projected edge endpoint is validated against `activity_vocab`.
- `structural_edge_index`, `struct_x`, and `allowed_target_mask` are proven to share class-space.
- Duplicate BPMN node labels are detected as `duplicate_activity_identity_ambiguity`, not silently treated as solved projection alignment.
- Missing node metadata and skipped projected edges are visible.
- Strict mode fails before training if projection cannot be proven aligned.
- Topology projection cache invalidates when node classification metadata changes.
- Model forward no longer silently modulo-repairs structural edge indices in strict mode.
- Existing loan BPMN/XES happy-path tests continue to pass.

## Recommended Next Step

Implement Phase 1 first: pure projection/alignment diagnostics tests. This creates a stable target before touching `DynamicGraphBuilder`, and it makes the later refactor measurable instead of speculative.
