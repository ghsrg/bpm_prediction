# Parallel Process-State Mask Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current last-completed-node mask with a process-state-aware mask for parallel/interleaved XES and Camunda-like logs.

**Architecture:** The mask must be derived from runtime process state, not only from `successors(last_completed_activity)`. For lifecycle-rich logs, reconstruct active task state from `assign/start/complete` events and inject it into both `allowed_target_mask` and `struct_prefix_state_x`. For logs without lifecycle/active state, provide a clearly labeled relaxed reachability fallback.

**Tech Stack:** Python 3.11, PyTorch/PyG, existing `XESAdapter`, `PrefixPolicy`, `DynamicGraphBuilder`, `ModelTrainer`, MLflow diagnostics.

---

## Context

The complex parallel dataset audit showed that `target_in_mask_rate` drops to about `0.53` even when trace-level and event-level `concept:version` are consistent. The issue is not wrong version grouping. The issue is that in parallel BPMN regions the next completed task can already be active after the previous completed task:

```text
parallel split -> B, C, D active
B complete
C complete  # valid next log event, but not a BPMN successor of B
```

The current mask approximates:

```text
M_t = successors(last_completed_activity)
```

For parallel logs this is incomplete. The process-state-aware mask should be:

```text
M_t = active_not_completed_tasks(t) union newly_enabled_successors(t)
```

where `active_not_completed_tasks(t)` is reconstructed from lifecycle state or provided by the simulator/BPMS.

## File Structure

- Modify: `src/adapters/ingestion/xes_adapter.py`
  - Preserve lifecycle-derived runtime metadata needed by downstream graph building.
- Modify: `src/domain/entities/event_record.py`
  - Ensure active-state metadata can be carried in `EventRecord.extra` without changing core fields.
- Modify: `src/domain/entities/prefix_slice.py`
  - No required schema change unless a typed prefix-level active-set field is introduced.
- Modify: `src/domain/services/dynamic_graph_builder.py`
  - Build `active_candidates` from prefix events.
  - Union active candidates into `allowed_target_mask` and `candidate_allowed_target_mask`.
  - Add active-state channels to `struct_prefix_state_x`.
- Modify: `src/application/use_cases/trainer.py`
  - Log active-mask diagnostics during train/eval/eval_drift.
- Modify: `configs/ui/config_catalog.yaml`
  - Add config keys for process-state mask mode.
- Modify: `outputs/ui/experiment_ui_presets.json`
  - Add/adjust presets only after implementation is validated.
- Test: `tests/domain/test_dynamic_graph_builder_parallel_active_mask.py`
- Test: `tests/application/test_trainer_parallel_active_mask_metrics.py`
- Docs: `docs/GNN_RUNTIME_MVP2_5.MD`, `docs/DATA_MODEL_MVP2_5.MD`, `docs/dissertation_math_fix.MD`, `docs/dissertation_changes.MD`

## Config Contract

Add:

```yaml
training.process_state_mask_enabled: false
training.process_state_mask_source: lifecycle_active_set
training.process_state_mask_fallback: direct_successors
training.process_state_mask_include_direct_successors: true
training.process_state_mask_include_active_candidates: true
```

Allowed values:

- `process_state_mask_source: lifecycle_active_set`
  - Reconstruct active tasks from `assign/start/complete` and `sim:activity_instance_id`.
- `process_state_mask_source: event_active_candidates`
  - Use explicit active candidates emitted by simulator/BPMS.
- `process_state_mask_source: relaxed_reachability`
  - Approximate fallback for lifecycle-poor XES. Must be reported as approximate.

## Task 1: Failing Domain Test for Parallel Active Mask

**Files:**

- Create: `tests/domain/test_dynamic_graph_builder_parallel_active_mask.py`
- Modify later: `src/domain/services/dynamic_graph_builder.py`

- [ ] **Step 1: Write the failing test**

```python
from __future__ import annotations

import torch

from src.domain.entities.event_record import EventRecord
from src.domain.entities.prefix_slice import PrefixSlice
from src.domain.entities.process_structure import ProcessStructureDTO
from src.domain.entities.raw_trace import RawTrace
from src.domain.services.dynamic_graph_builder import DynamicGraphBuilder
from src.domain.services.feature_encoder import FeatureEncoder
from src.infrastructure.repositories.in_memory_networkx_repository import InMemoryNetworkXRepository


def _event(idx: int, activity: str, lifecycle: str, instance: str) -> EventRecord:
    return EventRecord(
        activity_id=activity,
        timestamp=float(1700000000 + idx),
        resource_id="R1",
        lifecycle=lifecycle,
        position_in_trace=idx,
        duration=1.0,
        time_since_case_start=float(idx),
        time_since_previous_event=1.0 if idx > 0 else 0.0,
        extra={
            "concept:name": activity,
            "lifecycle:transition": lifecycle,
            "sim:activity_instance_id": instance,
            "org:resource": "R1",
        },
        activity_instance_id=instance,
    )


def _trace() -> RawTrace:
    return RawTrace(
        case_id="c1",
        process_version="v1",
        events=[
            _event(0, "A", "complete", "A_1"),
            _event(1, "B", "start", "B_1"),
            _event(2, "C", "start", "C_1"),
            _event(3, "B", "complete", "B_1"),
            _event(4, "C", "complete", "C_1"),
        ],
        trace_attributes={},
    )


def test_parallel_active_candidate_is_allowed_after_sibling_complete(mock_feature_configs):
    encoder = FeatureEncoder(feature_configs=mock_feature_configs, traces=[_trace()])
    repository = InMemoryNetworkXRepository()
    repository.save_process_structure(
        "v1",
        ProcessStructureDTO(
            version="v1",
            allowed_edges=[("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")],
            nodes=[
                {"id": "A", "bpmn_tag": "task", "type": "task", "label": "A"},
                {"id": "B", "bpmn_tag": "task", "type": "task", "label": "B"},
                {"id": "C", "bpmn_tag": "task", "type": "task", "label": "C"},
                {"id": "D", "bpmn_tag": "task", "type": "task", "label": "D"},
            ],
        ),
    )
    prefix = PrefixSlice(
        case_id="c1",
        process_version="v1",
        prefix_events=[
            _event(0, "A", "complete", "A_1"),
            _event(1, "B", "start", "B_1"),
            _event(2, "C", "start", "C_1"),
            _event(3, "B", "complete", "B_1"),
        ],
        target_event=_event(4, "C", "complete", "C_1"),
    )

    contract = DynamicGraphBuilder(
        feature_encoder=encoder,
        knowledge_port=repository,
        graph_feature_mapping={"topology_projection": {"gateway_mode": "collapse_for_prediction"}},
        process_state_mask_enabled=True,
    ).build_graph(prefix)

    activity_vocab = encoder.categorical_vocabs[encoder.activity_feature_name]
    assert bool(contract["allowed_target_mask"][activity_vocab["C"]]) is True
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\domain\test_dynamic_graph_builder_parallel_active_mask.py -v
```

Expected: failure because `DynamicGraphBuilder.__init__` does not accept `process_state_mask_enabled` yet or because `C` is not included in `allowed_target_mask`.

## Task 2: Add Process-State Mask Reconstruction

**Files:**

- Modify: `src/domain/services/dynamic_graph_builder.py`
- Test: `tests/domain/test_dynamic_graph_builder_parallel_active_mask.py`

- [ ] **Step 1: Add constructor options**

Add parameters to `DynamicGraphBuilder.__init__`:

```python
process_state_mask_enabled: bool = False,
process_state_mask_source: str = "lifecycle_active_set",
process_state_mask_include_direct_successors: bool = True,
process_state_mask_include_active_candidates: bool = True,
```

Store normalized values:

```python
self.process_state_mask_enabled = bool(process_state_mask_enabled)
self.process_state_mask_source = str(process_state_mask_source or "lifecycle_active_set").strip().lower()
self.process_state_mask_include_direct_successors = bool(process_state_mask_include_direct_successors)
self.process_state_mask_include_active_candidates = bool(process_state_mask_include_active_candidates)
```

- [ ] **Step 2: Add active-set reconstruction helper**

Add method:

```python
def _active_candidates_from_prefix(self, prefix: PrefixSlice) -> set[str]:
    active_by_instance: dict[str, str] = {}
    for event in prefix.prefix_events:
        extra = event.extra if isinstance(event.extra, dict) else {}
        lifecycle = str(extra.get("lifecycle:transition", event.lifecycle) or "").strip().lower()
        activity = str(
            self.feature_encoder.resolve_event_feature(
                event_extra=extra,
                feature_name=self.feature_encoder.activity_feature_name,
                default=event.activity_id,
            )
        ).strip()
        instance_id = str(
            extra.get("sim:activity_instance_id")
            or getattr(event, "activity_instance_id", "")
            or f"{activity}:{event.position_in_trace}"
        ).strip()
        if not activity or not instance_id:
            continue
        if lifecycle in {"assign", "start"}:
            active_by_instance[instance_id] = activity
        elif lifecycle == "complete":
            active_by_instance.pop(instance_id, None)
    return {activity for activity in active_by_instance.values() if activity}
```

- [ ] **Step 3: Union active candidates into fixed-label mask**

After `active_candidates = self._extract_active_candidates(...)`, add:

```python
if self.process_state_mask_enabled and self.process_state_mask_source == "lifecycle_active_set":
    active_candidates.update(self._active_candidates_from_prefix(prefix))
```

Then keep existing union logic:

```python
allowed_mask = torch.logical_or(allowed_mask, active_mask)
```

- [ ] **Step 4: Union active candidates into candidate-local mask**

Reuse existing candidate-index union:

```python
for token in active_candidates:
    for candidate_idx in self._candidate_indices_for_token(compiled=compiled, token=token):
        if 0 <= int(candidate_idx) < int(candidate_allowed_mask.numel()):
            candidate_allowed_mask[int(candidate_idx)] = True
```

- [ ] **Step 5: Run focused test**

Run:

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\domain\test_dynamic_graph_builder_parallel_active_mask.py -v
```

Expected: PASS.

## Task 3: Add Active-State Channels to Structural Prefix State

**Files:**

- Modify: `src/domain/services/dynamic_graph_builder.py`
- Test: `tests/domain/test_dynamic_graph_builder_parallel_active_mask.py`

- [ ] **Step 1: Extend `struct_prefix_state_x` channel semantics**

Current impulse state includes:

```text
is_last_event
prefix_executed_count_log1p
prefix_recency_norm
```

Add:

```text
is_active_candidate
active_count_log1p
```

- [ ] **Step 2: Update builder helper signature**

Change `_build_struct_prefix_state_x` to accept active candidates:

```python
def _build_struct_prefix_state_x(
    self,
    *,
    prefix: PrefixSlice,
    activity_vocab: Dict[str, int],
    struct_node_to_class_index: torch.Tensor,
    active_candidates: set[str] | None = None,
) -> torch.Tensor:
```

- [ ] **Step 3: Populate active channels**

Inside `_build_struct_prefix_state_x`, after resolving class indices, set:

```python
active_candidates = active_candidates or set()
active_indices = {
    int(activity_vocab[token])
    for token in active_candidates
    if token in activity_vocab
}
```

For each structural node class index:

```python
is_active = 1.0 if int(class_idx) in active_indices else 0.0
active_count_log1p = math.log1p(1.0) if is_active > 0.0 else 0.0
```

Append these values to the per-node row.

- [ ] **Step 4: Add shape assertion test**

Add to `tests/domain/test_dynamic_graph_builder_parallel_active_mask.py`:

```python
def test_parallel_active_state_adds_struct_prefix_channels(mock_feature_configs):
    # Reuse setup from test_parallel_active_candidate_is_allowed_after_sibling_complete.
    # Assert struct_prefix_state_x has at least five channels and active candidate C has active flag.
```

Expected assertion:

```python
assert contract["struct_prefix_state_x"].shape[1] >= 5
```

## Task 4: Wire Config from CLI

**Files:**

- Modify: `src/cli.py`
- Modify: `configs/ui/config_catalog.yaml`
- Test: `tests/application/test_cli_flatten_config.py` or a new focused config test

- [ ] **Step 1: Add config extraction in CLI**

When creating `DynamicGraphBuilder`, pass:

```python
process_state_mask_enabled=_as_bool(training_cfg.get("process_state_mask_enabled"), default=False),
process_state_mask_source=training_cfg.get("process_state_mask_source", "lifecycle_active_set"),
process_state_mask_include_direct_successors=_as_bool(
    training_cfg.get("process_state_mask_include_direct_successors"),
    default=True,
),
process_state_mask_include_active_candidates=_as_bool(
    training_cfg.get("process_state_mask_include_active_candidates"),
    default=True,
),
```

- [ ] **Step 2: Add catalog keys**

Add to `configs/ui/config_catalog.yaml`:

```yaml
training.process_state_mask_enabled:
  path: training.process_state_mask_enabled
  label: 'Training: Process-state mask'
  level: experiment_run
  group: Training Objective
  widget: checkbox
  default: false
  description: Enables active-task process-state reconstruction for allowed target masks.
  affects: allowed_target_mask, candidate_allowed_target_mask, struct_prefix_state_x.

training.process_state_mask_source:
  path: training.process_state_mask_source
  label: 'Training: Process-state mask source'
  level: advanced
  group: Training Objective
  widget: dropdown
  default: lifecycle_active_set
  options: [lifecycle_active_set, event_active_candidates, relaxed_reachability]
  description: Source used to build process-state-aware masks.
```

- [ ] **Step 3: Run catalog/UI tests**

Run:

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\application\test_experiment_ui_catalog_fields.py tests\application\test_desktop_ui_field_registry.py -v
```

Expected: PASS.

## Task 5: Add Diagnostics

**Files:**

- Modify: `src/application/use_cases/trainer.py`
- Test: `tests/application/test_trainer_parallel_active_mask_metrics.py`

- [ ] **Step 1: Add forward-stat counters**

Log:

```text
process_state_mask_active_candidate_count_mean
process_state_mask_target_added_by_active_set_rate
process_state_mask_target_in_direct_successors_rate
process_state_mask_target_in_active_set_rate
```

- [ ] **Step 2: Add minimal trainer test**

Create a batch where target is not direct successor of last completed node but is in active set. Assert:

```python
assert logged_metrics["process_state_mask_target_in_active_set_rate"] == 1.0
```

- [ ] **Step 3: Run focused trainer tests**

Run:

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\application\test_trainer_parallel_active_mask_metrics.py -v
```

Expected: PASS.

## Task 6: Documentation Update

**Files:**

- Modify: `docs/GNN_RUNTIME_MVP2_5.MD`
- Modify: `docs/DATA_MODEL_MVP2_5.MD`
- Modify: `docs/dissertation_math_fix.MD`
- Modify: `docs/dissertation_changes.MD`

- [ ] **Step 1: Update runtime docs**

Document:

```text
Direct successor mask is valid for sequential processes.
Parallel/interleaved processes require process-state-aware masks.
```

- [ ] **Step 2: Update data model docs**

Document required lifecycle metadata:

```text
lifecycle:transition
sim:activity_instance_id
active_candidates (optional explicit field)
```

- [ ] **Step 3: Update dissertation notes**

Add formula:

```text
M_t = A_t union S_t
```

where:

- `A_t` is the active task set;
- `S_t` is the set of newly enabled structural successors.

## Task 7: Verification Run

**Files:**

- No new code files.
- Uses existing presets after config update.

- [ ] **Step 1: Run unit and focused integration tests**

Run:

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\domain\test_dynamic_graph_builder_parallel_active_mask.py tests\domain\test_dynamic_graph_builder_masks.py -v
.\.venv-modern\Scripts\python.exe tools\architecture_guard.py
```

Expected: PASS and `[ARCH_GUARD] OK`.

- [ ] **Step 2: Re-run complex drift audit**

Run `_EOPKG_TC` complex drift with:

```yaml
training.process_state_mask_enabled: true
training.process_state_mask_source: lifecycle_active_set
```

Expected:

```text
target_in_mask_rate increases materially above 0.53
test_oos decreases
strict_macro_f1 increases only if ranking is also adequate
strict_error_but_allowed_rate may stay non-zero in parallel regions
```

## Risk Notes

- A process-state mask improves oracle validity but can also expand candidate cardinality. Strict F1 may not improve proportionally if ranking among active parallel tasks remains ambiguous.
- For pure complete-only XES without lifecycle events, this method cannot reconstruct active state reliably. Use `relaxed_reachability` and report it as approximate.
- If timestamps are tied, event ordering in XES can still be arbitrary. Metrics should include a parallel ambiguity rate.

## Phase 2: Relaxed Reachability Mask for Flattened Parallel XES

### Why Phase 2 Is Needed

The implemented `lifecycle_active_set` path only adds tasks that are already
started and not completed. For the current complex XES adapter path this does
not change drift metrics because `XESAdapter` already emits
`active_activities_after_complete`, and `DynamicGraphBuilder` already merged
those active candidates into masks before this plan.

The remaining `target_not_in_mask` cases require a broader approximation:

```text
direct successors of last completed
+ active already-started tasks
+ enabled-but-not-yet-started sibling tasks in recent open parallel regions
```

Without a true BPMN token/marking replay, the MVP fallback is
`process_state_mask_source=relaxed_reachability`.

### Relaxed Reachability Semantics

This mode is explicitly approximate. It does not claim exact BPMN join/split
semantics. It builds a bounded candidate set from the collapsed prediction graph:

```text
M_t =
  successors(last_completed_activity)
  union active_candidates_after_complete
  union reachable_successors(recent_prefix_anchors, depth <= D)
```

Where:

- `recent_prefix_anchors` are the last `K` completed activity labels in the
  current prefix, including the last completed activity.
- `reachable_successors` uses the same projected/collapsed topology graph used
  for prediction masks.
- `D` is small, default `1`, to capture sibling branch tasks from recent split
  anchors without turning the mask into the full candidate universe.
- The mode must report expanded cardinality and target-added-by-relaxed rates.

Recommended initial config:

```yaml
training.process_state_mask_enabled: true
training.process_state_mask_source: relaxed_reachability
training.process_state_mask_include_direct_successors: true
training.process_state_mask_include_active_candidates: true
training.process_state_mask_relaxed_lookback_events: 8
training.process_state_mask_relaxed_max_depth: 1
training.process_state_mask_relaxed_max_cardinality_ratio: 0.35
```

If `max_cardinality_ratio` is exceeded, cap the relaxed additions by BFS order
and log a warning counter. This prevents the fallback from becoming an
unbounded "allow almost everything" mask.

### Task 8: Add Relaxed Reachability Config Contract

**Files:**

- Modify: `configs/ui/config_catalog.yaml`
- Modify: `src/cli.py`
- Modify: `src/domain/services/dynamic_graph_builder.py`
- Test: `tests/application/test_experiment_ui_catalog_fields.py`

- [ ] **Step 1: Add builder constructor parameters**

Add:

```python
process_state_mask_relaxed_lookback_events: int = 8,
process_state_mask_relaxed_max_depth: int = 1,
process_state_mask_relaxed_max_cardinality_ratio: float = 0.35,
```

Normalize:

```python
self.process_state_mask_relaxed_lookback_events = max(
    1,
    int(process_state_mask_relaxed_lookback_events),
)
self.process_state_mask_relaxed_max_depth = max(
    1,
    int(process_state_mask_relaxed_max_depth),
)
self.process_state_mask_relaxed_max_cardinality_ratio = min(
    1.0,
    max(0.01, float(process_state_mask_relaxed_max_cardinality_ratio)),
)
```

- [ ] **Step 2: Wire CLI config**

Pass from `training_cfg` into `DynamicGraphBuilder`:

```python
process_state_mask_relaxed_lookback_events=int(
    training_cfg.get("process_state_mask_relaxed_lookback_events", 8)
),
process_state_mask_relaxed_max_depth=int(
    training_cfg.get("process_state_mask_relaxed_max_depth", 1)
),
process_state_mask_relaxed_max_cardinality_ratio=float(
    training_cfg.get("process_state_mask_relaxed_max_cardinality_ratio", 0.35)
),
```

- [ ] **Step 3: Add catalog keys**

Add:

```yaml
training.process_state_mask_relaxed_lookback_events:
  path: training.process_state_mask_relaxed_lookback_events
  default: 8
  type: int
  description: Number of recent completed prefix events used as relaxed reachability anchors.
  required_when:
    training.process_state_mask_source: relaxed_reachability

training.process_state_mask_relaxed_max_depth:
  path: training.process_state_mask_relaxed_max_depth
  default: 1
  type: int
  description: BFS depth over collapsed prediction topology for relaxed mask expansion.
  required_when:
    training.process_state_mask_source: relaxed_reachability

training.process_state_mask_relaxed_max_cardinality_ratio:
  path: training.process_state_mask_relaxed_max_cardinality_ratio
  default: 0.35
  type: float
  description: Maximum fraction of candidate universe allowed after relaxed expansion.
  required_when:
    training.process_state_mask_source: relaxed_reachability
```

- [ ] **Step 4: Extend catalog test**

Update `test_catalog_includes_process_state_mask_fields` to assert the three
new keys exist and are gated by `training.process_state_mask_source:
relaxed_reachability`.

### Task 9: Build Relaxed Candidate Expansion in DynamicGraphBuilder

**Files:**

- Modify: `src/domain/services/dynamic_graph_builder.py`
- Test: `tests/domain/test_dynamic_graph_builder_parallel_process_state_mask.py`

- [ ] **Step 1: Write failing completion-only parallel test**

Add test:

```python
def test_relaxed_reachability_allows_not_yet_started_parallel_sibling(mock_feature_configs):
    # Topology: A -> B, A -> C, B -> D, C -> D.
    # Prefix is completion-only: A complete, B complete.
    # Target is C complete.
    # Direct successor of B is D, active set is empty.
    # relaxed_reachability with lookback=2 must include C through anchor A.
```

Expected:

```python
assert bool(contract["allowed_target_mask"][activity_vocab["C"]]) is True
assert bool(contract["candidate_allowed_target_mask"][activity_vocab["C"]]) is True
```

- [ ] **Step 2: Add recent prefix anchor helper**

Add:

```python
def _recent_prefix_activity_tokens(self, prefix: PrefixSlice, *, limit: int) -> list[str]:
    tokens: list[str] = []
    target_feature = self.feature_encoder.activity_feature_name
    for event in reversed(prefix.prefix_events):
        token = str(
            self.feature_encoder.resolve_event_feature(
                event_extra=event.extra if isinstance(event.extra, dict) else {},
                feature_name=target_feature,
                default=event.activity_id,
            )
        ).strip()
        if token:
            tokens.append(token)
        if len(tokens) >= limit:
            break
    return tokens
```

- [ ] **Step 3: Add relaxed reachability helper**

Use `compiled["candidate_allowed_masks_by_src"]` for topology-native candidate
space and `compiled["allowed_masks_by_src"]` for fixed label space:

```python
def _relaxed_reachability_candidates(
    self,
    *,
    prefix: PrefixSlice,
    compiled: Dict[str, Any],
    activity_vocab: Dict[str, int],
) -> set[str]:
    anchors = self._recent_prefix_activity_tokens(
        prefix,
        limit=self.process_state_mask_relaxed_lookback_events,
    )
    result: set[str] = set()
    for anchor in anchors:
        frontier = self._candidate_indices_for_token(compiled=compiled, token=anchor)
        visited = set(frontier)
        for _depth in range(self.process_state_mask_relaxed_max_depth):
            next_frontier: list[int] = []
            for src_candidate_idx in frontier:
                mask = compiled.get("candidate_allowed_masks_by_src", {}).get(int(src_candidate_idx))
                if not isinstance(mask, torch.Tensor):
                    continue
                for dst_idx in torch.nonzero(mask, as_tuple=False).reshape(-1).tolist():
                    if int(dst_idx) in visited:
                        continue
                    visited.add(int(dst_idx))
                    next_frontier.append(int(dst_idx))
                    labels = compiled.get("candidate_labels", ())
                    if 0 <= int(dst_idx) < len(labels):
                        result.add(str(labels[int(dst_idx)]))
            frontier = next_frontier
    return self._cap_relaxed_candidates(result=result, compiled=compiled)
```

- [ ] **Step 4: Add cardinality cap helper**

Add deterministic cap:

```python
def _cap_relaxed_candidates(self, *, result: set[str], compiled: Dict[str, Any]) -> set[str]:
    candidate_count = int(compiled.get("candidate_count", 0) or 0)
    if candidate_count <= 0:
        return result
    max_count = max(1, int(math.ceil(candidate_count * self.process_state_mask_relaxed_max_cardinality_ratio)))
    if len(result) <= max_count:
        return result
    return set(sorted(result)[:max_count])
```

Note: deterministic alphabetical cap is acceptable for MVP diagnostics. If it
hurts quality, replace with BFS-order-preserving cap in a follow-up.

- [ ] **Step 5: Merge relaxed candidates**

In `_process_state_active_candidates`:

```python
if self.process_state_mask_source == "relaxed_reachability":
    return self._relaxed_reachability_candidates(
        prefix=prefix,
        compiled=compiled,
        activity_vocab=activity_vocab,
    )
```

This requires changing `_process_state_active_candidates` signature to accept
`compiled` and `activity_vocab`.

- [ ] **Step 6: Preserve active-set behavior**

For `lifecycle_active_set`, behavior must remain unchanged:

```python
return self._active_candidates_from_lifecycle(prefix)
```

For `event_active_candidates`, behavior must remain unchanged:

```python
return self._extract_active_candidates(prefix.prefix_events[-1].extra)
```

### Task 10: Add Relaxed Reachability Diagnostics

**Files:**

- Modify: `src/domain/services/dynamic_graph_builder.py`
- Modify: `src/application/use_cases/trainer.py`
- Test: `tests/application/test_trainer_parallel_active_mask_metrics.py`

- [ ] **Step 1: Add contract debug scalars**

Attach optional scalar fields to each graph contract:

```python
contract["process_state_mask_source"] = self.process_state_mask_source
contract["process_state_mask_active_candidate_count"] = int(len(active_candidates))
contract["process_state_mask_relaxed_candidate_count"] = int(len(relaxed_candidates))
contract["process_state_mask_direct_candidate_count"] = int(direct_candidate_count)
```

If current `GraphTensorContract` batching cannot preserve string fields, keep
only numeric tensor fields:

```python
contract["process_state_mask_active_candidate_count"] = torch.tensor([len(active_candidates)], dtype=torch.long)
contract["process_state_mask_relaxed_candidate_count"] = torch.tensor([len(relaxed_candidates)], dtype=torch.long)
contract["process_state_mask_direct_candidate_count"] = torch.tensor([direct_candidate_count], dtype=torch.long)
```

- [ ] **Step 2: Log aggregate metrics**

In trainer/evaluator metrics, log:

```text
process_state_mask_active_candidate_count_mean
process_state_mask_relaxed_candidate_count_mean
process_state_mask_direct_candidate_count_mean
process_state_mask_cardinality_delta_mean
```

- [ ] **Step 3: Add target attribution metrics**

For eval/eval_drift, log:

```text
target_in_direct_mask_rate
target_added_by_active_set_rate
target_added_by_relaxed_reachability_rate
```

These metrics are required to prove whether relaxed reachability is actually
responsible for any `target_in_mask_rate` increase.

### Task 11: Verification Plan for Relaxed Reachability

**Files:**

- No new code files.
- Uses existing complex dataset and MLflow metrics.

- [ ] **Step 1: Run focused tests**

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\domain\test_dynamic_graph_builder_parallel_process_state_mask.py tests\application\test_experiment_ui_catalog_fields.py -v
```

Expected: PASS.

- [ ] **Step 2: Run mask-only drift comparison**

Use the same checkpoint and same dataset, compare:

```yaml
training.process_state_mask_enabled: true
training.process_state_mask_source: lifecycle_active_set
```

against:

```yaml
training.process_state_mask_enabled: true
training.process_state_mask_source: relaxed_reachability
training.process_state_mask_relaxed_lookback_events: 8
training.process_state_mask_relaxed_max_depth: 1
training.process_state_mask_relaxed_max_cardinality_ratio: 0.35
```

Expected for relaxed reachability:

```text
drift_window_target_in_mask_rate increases above current ~0.53
drift_window_test_oos decreases if model predictions follow mask
drift_window_strict_error_but_allowed_rate may increase because mask is wider
drift_window_macro_f1 may improve before strict_macro_f1 improves
```

- [ ] **Step 3: Decide if training is needed**

Only retrain after relaxed reachability shows a material
`target_in_mask_rate` increase. If the mask does not improve, training will not
fix this layer.

## Phase 2 Risk Notes

- `relaxed_reachability` can over-expand masks. Track cardinality and report it
  as approximate, not exact BPMN conformance.
- For dissertation-grade strict BPMN semantics, this should later be replaced
  by true BPMN marking replay over collapsed prediction nodes.
- If strict F1 does not improve but `target_in_mask_rate` improves, the problem
  moves from mask validity to candidate ranking among valid parallel tasks.

## Self-Review

- Spec coverage: covers active candidates, reconstructed marking, relaxed fallback, impulse/mask impact, tests, config, docs.
- Placeholder scan: no `TBD` or unresolved implementation placeholders.
- Type consistency: uses existing project terms `allowed_target_mask`, `candidate_allowed_target_mask`, `struct_prefix_state_x`, `DynamicGraphBuilder`, `PrefixSlice`.
