# MVP2.5 State-Aware Relaxed Reachability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refine `training.process_state_mask_source=relaxed_reachability` so it improves `target_in_mask_rate` without over-expanding the allowed set and hurting strict next-activity metrics.

**Architecture:** Keep relaxed reachability as an approximate fallback for flattened/interleaved XES logs without reliable BPMN marking. Add state-aware filtering over recent-prefix reachable candidates: direct successors and active candidates remain authoritative, while completed relaxed-only candidates are suppressed only when they are not active, not direct successors, and have no loop/rework evidence. Add diagnostics that separate raw relaxed expansion, completed-candidate suppression, target suppression risk, and final mask effect.

**Tech Stack:** Python 3.11, PyTorch/PyG, `DynamicGraphBuilder`, existing `GraphTensorContract`, `ModelTrainer` diagnostics, MLflow metrics, desktop UI config catalog.

---

## Context

Recent experiments showed that simply reducing relaxed reachability aggression
(`lookback`, `depth`, `max_cardinality_ratio`) does not solve the issue.

Observed behavior from the complex parallel dataset:

- `relaxed_reachability` improves `target_in_mask_rate`.
- It strongly reduces `test_oos`.
- It can worsen `drift_window_strict_macro_f1`.
- It increases `strict_error_but_allowed_rate` and `ambiguous_prefix_rate`.

Interpretation:

```text
current relaxed mask = direct successors(last) union active candidates union reachable_from_recent_anchors
```

This is too broad for completion-only parallel XES. It may include tasks that
were valid from an older split anchor but have already completed in the prefix.
The model is then not OOS, but it is allowed to choose stale sibling tasks. That
converts invalid predictions into "allowed but wrong" predictions.

The next refinement should make relaxed reachability state-aware:

```text
M_final =
  direct_successors(last_completed)
  union active_candidates
  union state_filtered_relaxed_candidates
```

where:

```text
state_filtered_relaxed_candidates =
  reachable_from_recent_open_anchors
  - completed_relaxed_only_candidates
```

Direct successors and active candidates must not be removed by the completed
filter because loops, rework, and BPMS retries can legitimately revisit a task.
The implementation must not use a plain `completed_tokens: set[str]` as the only
state signal. A token can be completed in an earlier instance while another
instance of the same task is active now. Suppression must use:

```text
active_tokens
completed_tokens
active_instance_ids
completed_instance_ids
direct_successors(last_completed)
loop/rework evidence
```

A relaxed candidate may be suppressed only if all are true:

```text
token in completed_tokens
token not in active_tokens
token not in explicit active_candidates
token not in direct_successors(last_completed)
token has no loop/rework evidence
```

## Non-Goals

- Do not implement full BPMN marking replay in this plan.
- Do not change `struct_prefix_state_x` shape.
- Do not change model architecture or loss.
- Do not change `mask_guided_enabled` semantics.
- Do not treat relaxed reachability results as dissertation-grade exact BPMN
  semantics. It remains an approximation until BPMN marking replay exists.

## File Structure

- Modify: `src/domain/services/dynamic_graph_builder.py`
  - Add prefix process-state extraction: completed tokens, active tokens,
    completed instance IDs, and active instance IDs.
  - Add open-anchor selection.
  - Add completed-candidate suppression for relaxed-only candidates.
  - Add raw/suppressed/final relaxed diagnostics and target-suppression
    diagnostics to the graph contract.
- Modify: `src/cli.py`
  - Pass new `training.process_state_mask_relaxed_*` keys into `DynamicGraphBuilder`.
- Modify: `src/application/use_cases/trainer.py`
  - Aggregate and log new process-state relaxed diagnostics.
- Modify: `configs/ui/config_catalog.yaml`
  - Register new config keys for desktop UI and YAML validation.
- Test: `tests/domain/test_dynamic_graph_builder_parallel_process_state_mask.py`
  - Extend existing relaxed reachability tests.
- Test: `tests/application/test_experiment_ui_catalog_fields.py`
  - Verify new config catalog keys and gating.
- Test: `tests/application/test_drift_one_pass_evaluation.py`
  - Verify diagnostics propagation does not break one-pass drift records.
- Docs: `docs/GNN_RUNTIME_MVP2_5.MD`
  - Document state-aware relaxed reachability behavior and limits.
- Docs: `docs/current/project-state.md`
  - Add runtime update after implementation.
- Docs: `docs/dissertation_changes.MD`
  - Add short thesis note: relaxed masks trade OOS for ambiguity unless state-aware suppression is used.

## Config Contract

Keep existing keys:

```yaml
training.process_state_mask_enabled: true
training.process_state_mask_source: relaxed_reachability
training.process_state_mask_include_direct_successors: true
training.process_state_mask_include_active_candidates: true
training.process_state_mask_relaxed_lookback_events: 8
training.process_state_mask_relaxed_max_depth: 1
training.process_state_mask_relaxed_max_cardinality_ratio: 0.35
```

Add:

```yaml
training.process_state_mask_relaxed_suppress_completed: true
training.process_state_mask_relaxed_anchor_policy: open_successors
training.process_state_mask_relaxed_loop_policy: keep_direct_successor_repeats
```

Allowed values:

- `training.process_state_mask_relaxed_anchor_policy`
  - `recent_prefix`: current behavior; use recent prefix events as anchors.
  - `open_successors`: skip anchors whose direct successors are already completed, except direct successor / active / loop exceptions.
- `training.process_state_mask_relaxed_loop_policy`
  - `keep_direct_successor_repeats`: completed tasks may stay in the mask only when they are direct successors of the last completed candidate or active candidates.

Default behavior should remain backward-compatible enough for existing presets,
but the recommended research default should be:

```yaml
training.process_state_mask_relaxed_suppress_completed: true
training.process_state_mask_relaxed_anchor_policy: open_successors
training.process_state_mask_relaxed_loop_policy: keep_direct_successor_repeats
```

## Task 1: Domain Tests for Completed Sibling Suppression

**Files:**

- Modify: `tests/domain/test_dynamic_graph_builder_parallel_process_state_mask.py`
- Modify later: `src/domain/services/dynamic_graph_builder.py`

- [ ] **Step 1: Add a failing test where completed siblings are suppressed**

Append this test to `tests/domain/test_dynamic_graph_builder_parallel_process_state_mask.py`:

```python
def test_relaxed_reachability_suppresses_completed_parallel_siblings(mock_feature_configs):
    events = [
        _event(0, "A", instance_id="A_1"),
        _event(1, "B", instance_id="B_1"),
        _event(2, "C", instance_id="C_1"),
        _event(3, "D", instance_id="D_1"),
    ]
    encoder = FeatureEncoder(
        feature_configs=mock_feature_configs,
        traces=[_trace("train", "v1", events)],
    )
    repository = InMemoryNetworkXRepository()
    repository.save_process_structure(
        "v1",
        ProcessStructureDTO(
            version="v1",
            allowed_edges=[
                ("A", "B"),
                ("A", "C"),
                ("B", "D"),
                ("C", "D"),
            ],
        ),
    )

    prefix = PrefixSlice(
        case_id="eval",
        process_version="v1",
        prefix_events=events[:3],
        target_event=events[3],
    )

    contract = DynamicGraphBuilder(
        feature_encoder=encoder,
        knowledge_port=repository,
        process_state_mask_enabled=True,
        process_state_mask_source="relaxed_reachability",
        process_state_mask_relaxed_lookback_events=3,
        process_state_mask_relaxed_max_depth=1,
        process_state_mask_relaxed_max_cardinality_ratio=1.0,
        process_state_mask_relaxed_suppress_completed=True,
        process_state_mask_relaxed_anchor_policy="open_successors",
    ).build_graph(prefix)

    activity_vocab = encoder.categorical_vocabs[encoder.activity_feature_name]
    b_idx = int(activity_vocab["B"])
    c_idx = int(activity_vocab["C"])
    d_idx = int(activity_vocab["D"])

    assert bool(contract["allowed_target_mask"][d_idx]) is True
    assert bool(contract["candidate_allowed_target_mask"][d_idx]) is True
    assert bool(contract["allowed_target_mask"][b_idx]) is False
    assert bool(contract["candidate_allowed_target_mask"][b_idx]) is False
    assert bool(contract["allowed_target_mask"][c_idx]) is False
    assert bool(contract["candidate_allowed_target_mask"][c_idx]) is False
    assert int(contract["process_state_mask_relaxed_raw_candidate_count"].item()) >= 2
    assert int(contract["process_state_mask_relaxed_suppressed_completed_count"].item()) >= 1
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\domain\test_dynamic_graph_builder_parallel_process_state_mask.py::test_relaxed_reachability_suppresses_completed_parallel_siblings -v
```

Expected: fail because the new constructor parameters and diagnostics do not
exist yet, or because completed sibling candidates are still included.

## Task 2: Prefix Process-State and Open-Anchor Helpers

**Files:**

- Modify: `src/domain/services/dynamic_graph_builder.py`
- Test: `tests/domain/test_dynamic_graph_builder_parallel_process_state_mask.py`

- [ ] **Step 1: Add constructor options**

Extend `DynamicGraphBuilder.__init__`:

```python
process_state_mask_relaxed_suppress_completed: bool = True,
process_state_mask_relaxed_anchor_policy: str = "open_successors",
process_state_mask_relaxed_loop_policy: str = "keep_direct_successor_repeats",
```

Store normalized values:

```python
self.process_state_mask_relaxed_suppress_completed = bool(
    process_state_mask_relaxed_suppress_completed
)
self.process_state_mask_relaxed_anchor_policy = str(
    process_state_mask_relaxed_anchor_policy or "open_successors"
).strip().lower()
if self.process_state_mask_relaxed_anchor_policy not in {"recent_prefix", "open_successors"}:
    self.process_state_mask_relaxed_anchor_policy = "open_successors"
self.process_state_mask_relaxed_loop_policy = str(
    process_state_mask_relaxed_loop_policy or "keep_direct_successor_repeats"
).strip().lower()
if self.process_state_mask_relaxed_loop_policy not in {"keep_direct_successor_repeats"}:
    self.process_state_mask_relaxed_loop_policy = "keep_direct_successor_repeats"
```

- [ ] **Step 2: Add prefix process-state helper**

Add method near `_recent_prefix_activity_tokens`:

```python
def _prefix_process_state(self, prefix: PrefixSlice) -> dict[str, set[str]]:
    completed_tokens: set[str] = set()
    active_tokens: set[str] = set()
    completed_instance_ids: set[str] = set()
    active_instance_ids: set[str] = set()
    active_token_by_instance: dict[str, str] = {}
    target_feature = self.feature_encoder.activity_feature_name
    for event in prefix.prefix_events:
        extra = event.extra if isinstance(event.extra, dict) else {}
        lifecycle = str(extra.get("lifecycle:transition") or event.lifecycle or "").strip().lower()
        token = str(
            self.feature_encoder.resolve_event_feature(
                event_extra=extra,
                feature_name=target_feature,
                default=event.activity_id,
            )
        ).strip()
        if not token:
            continue
        instance_id = str(
            extra.get("sim:activity_instance_id")
            or event.activity_instance_id
            or f"{token}:{event.position_in_trace}"
        ).strip()
        if lifecycle in {"assign", "start"}:
            active_tokens.add(token)
            if instance_id:
                active_instance_ids.add(instance_id)
                active_token_by_instance[instance_id] = token
        elif not lifecycle or lifecycle in {"complete", "completed"}:
            completed_tokens.add(token)
            if instance_id:
                completed_instance_ids.add(instance_id)
                active_instance_ids.discard(instance_id)
                previous_token = active_token_by_instance.pop(instance_id, None)
                if previous_token is not None and previous_token not in active_token_by_instance.values():
                    active_tokens.discard(previous_token)
    return {
        "completed_tokens": completed_tokens,
        "active_tokens": active_tokens,
        "completed_instance_ids": completed_instance_ids,
        "active_instance_ids": active_instance_ids,
    }
```

This helper intentionally keeps token-level and instance-level state separate.
The suppression decision uses token-level state, while instance IDs are retained
for diagnostics and future simulator/BPMS extensions.

- [ ] **Step 3: Add direct successor label helper**

Add method:

```python
def _direct_successor_labels_for_token(self, *, compiled: Dict[str, Any], token: str) -> set[str]:
    candidate_masks_by_src = compiled.get("candidate_allowed_masks_by_src", {})
    candidate_labels = tuple(str(item) for item in compiled.get("candidate_labels", ()))
    result: set[str] = set()
    for src_candidate_idx in self._candidate_indices_for_token(compiled=compiled, token=token):
        mask = (
            candidate_masks_by_src.get(int(src_candidate_idx))
            if isinstance(candidate_masks_by_src, dict)
            else None
        )
        if not isinstance(mask, torch.Tensor):
            continue
        for dst_idx_raw in torch.nonzero(mask, as_tuple=False).reshape(-1).tolist():
            dst_idx = int(dst_idx_raw)
            if 0 <= dst_idx < len(candidate_labels):
                result.add(str(candidate_labels[dst_idx]))
    return result
```

- [ ] **Step 4: Run the focused failing test**

Run:

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\domain\test_dynamic_graph_builder_parallel_process_state_mask.py::test_relaxed_reachability_suppresses_completed_parallel_siblings -v
```

Expected: still fail because filtering is not applied yet.

## Task 3: State-Aware Relaxed Candidate Filtering

**Files:**

- Modify: `src/domain/services/dynamic_graph_builder.py`
- Test: `tests/domain/test_dynamic_graph_builder_parallel_process_state_mask.py`

- [ ] **Step 1: Return relaxed candidate diagnostics from helper**

Change `_relaxed_reachability_candidates` to return a tuple:

```python
def _relaxed_reachability_candidates(
    self,
    *,
    prefix: PrefixSlice,
    compiled: Dict[str, Any],
    activity_vocab: Dict[str, int],
    active_candidates: set[str],
) -> tuple[set[str], dict[str, int]]:
```

Initialize diagnostics:

```python
diagnostics = {
    "raw_candidate_count": 0,
    "suppressed_completed_count": 0,
    "final_candidate_count": 0,
    "target_suppressed_by_completed_filter_count": 0,
    "anchor_count": 0,
    "skipped_closed_anchor_count": 0,
}
```

- [ ] **Step 2: Add open-anchor filtering**

Inside `_relaxed_reachability_candidates`, after computing `anchors`, derive:

```python
prefix_state = self._prefix_process_state(prefix)
completed_tokens = set(prefix_state["completed_tokens"])
active_tokens = set(prefix_state["active_tokens"])
active_tokens.update(str(item).strip() for item in active_candidates if str(item).strip())
last_token = anchors[0] if anchors else ""
last_direct_successors = self._direct_successor_labels_for_token(
    compiled=compiled,
    token=last_token,
)
```

When `self.process_state_mask_relaxed_anchor_policy == "open_successors"`,
skip an anchor only if all of its direct successors are already completed and
none of those successors is active. Do not skip anchors with self-loop/rework
evidence:

```python
anchor_direct_successors = self._direct_successor_labels_for_token(
    compiled=compiled,
    token=anchor,
)
anchor_has_loop_evidence = str(anchor).strip() in anchor_direct_successors
if (
    self.process_state_mask_relaxed_anchor_policy == "open_successors"
    and anchor_direct_successors
    and anchor_direct_successors.issubset(completed_tokens)
    and not anchor_direct_successors.intersection(active_tokens)
    and not anchor_has_loop_evidence
):
    diagnostics["skipped_closed_anchor_count"] += 1
    continue
diagnostics["anchor_count"] += 1
```

- [ ] **Step 3: Apply completed suppression only to relaxed-only candidates**

After raw BFS result is built:

```python
raw_result = set(result)
diagnostics["raw_candidate_count"] = int(len(raw_result))
```

Suppress completed candidates only if they are not active, not explicit active
candidates, not direct successors of the last completed token, and have no loop
or rework evidence:

```python
if self.process_state_mask_relaxed_suppress_completed:
    keep_completed = set(last_direct_successors)
    keep_completed.update(active_tokens)
    filtered = {
        token for token in raw_result
        if token not in completed_tokens
        or token in keep_completed
        or self._candidate_has_loop_or_rework_evidence(compiled=compiled, token=token)
    }
    diagnostics["suppressed_completed_count"] = int(len(raw_result) - len(filtered))
else:
    filtered = raw_result
```

Track whether the target was suppressed:

```python
target_token = ""
if prefix.target_event is not None:
    target_extra = prefix.target_event.extra if isinstance(prefix.target_event.extra, dict) else {}
    target_token = str(
        self.feature_encoder.resolve_event_feature(
            event_extra=target_extra,
            feature_name=self.feature_encoder.activity_feature_name,
            default=prefix.target_event.activity_id,
        )
    ).strip()
if target_token and target_token in raw_result and target_token not in filtered:
    diagnostics["target_suppressed_by_completed_filter_count"] = 1
```

Then cap:

```python
final = self._cap_relaxed_candidates(result=filtered, compiled=compiled)
diagnostics["final_candidate_count"] = int(len(final))
return final, diagnostics
```

- [ ] **Step 4: Add loop/rework evidence helper**

Add method:

```python
def _candidate_has_loop_or_rework_evidence(self, *, compiled: Dict[str, Any], token: str) -> bool:
    token = str(token).strip()
    if not token:
        return False
    direct_successors = self._direct_successor_labels_for_token(compiled=compiled, token=token)
    if token in direct_successors:
        return True
    candidate_masks_by_src = compiled.get("candidate_allowed_masks_by_src", {})
    if not isinstance(candidate_masks_by_src, dict):
        return False
    for predecessor_label in tuple(str(item) for item in compiled.get("candidate_labels", ())):
        successor_labels = self._direct_successor_labels_for_token(
            compiled=compiled,
            token=predecessor_label,
        )
        if token in successor_labels and predecessor_label == token:
            return True
    return False
```

- [ ] **Step 5: Adapt caller and contract diagnostics**

In `_process_state_active_candidates`, return only candidates for non-relaxed
paths. For relaxed mode, introduce a new helper or store diagnostics in
`build_graph`. The simplest low-risk change is to call relaxed reachability
directly from `build_graph`:

```python
process_state_candidates = set()
relaxed_diagnostics = {
    "raw_candidate_count": 0,
    "suppressed_completed_count": 0,
    "anchor_count": 0,
    "skipped_closed_anchor_count": 0,
}
if self.process_state_mask_enabled and self.process_state_mask_include_active_candidates:
    if self.process_state_mask_source == "relaxed_reachability":
        process_state_candidates, relaxed_diagnostics = self._relaxed_reachability_candidates(
            prefix=prefix,
            compiled=compiled,
            activity_vocab=activity_vocab,
            active_candidates=active_candidates,
        )
        relaxed_candidates = set(process_state_candidates)
    else:
        process_state_candidates = self._process_state_active_candidates(
            prefix=prefix,
            compiled=compiled,
            activity_vocab=activity_vocab,
        )
    active_candidates.update(process_state_candidates)
```

Attach diagnostics:

```python
contract["process_state_mask_relaxed_raw_candidate_count"] = torch.tensor(
    [int(relaxed_diagnostics.get("raw_candidate_count", 0))],
    dtype=torch.long,
)
contract["process_state_mask_relaxed_suppressed_completed_count"] = torch.tensor(
    [int(relaxed_diagnostics.get("suppressed_completed_count", 0))],
    dtype=torch.long,
)
contract["process_state_mask_relaxed_final_candidate_count"] = torch.tensor(
    [int(relaxed_diagnostics.get("final_candidate_count", len(relaxed_candidates)))],
    dtype=torch.long,
)
contract["process_state_mask_target_suppressed_by_completed_filter_count"] = torch.tensor(
    [int(relaxed_diagnostics.get("target_suppressed_by_completed_filter_count", 0))],
    dtype=torch.long,
)
contract["process_state_mask_relaxed_anchor_count"] = torch.tensor(
    [int(relaxed_diagnostics.get("anchor_count", 0))],
    dtype=torch.long,
)
contract["process_state_mask_relaxed_skipped_closed_anchor_count"] = torch.tensor(
    [int(relaxed_diagnostics.get("skipped_closed_anchor_count", 0))],
    dtype=torch.long,
)
```

- [ ] **Step 5: Run domain tests**

Run:

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\domain\test_dynamic_graph_builder_parallel_process_state_mask.py -v
```

Expected: all tests in this file pass.

## Task 4: Preserve Loop/Rework Direct Successor Behavior

**Files:**

- Modify: `tests/domain/test_dynamic_graph_builder_parallel_process_state_mask.py`
- Modify: `src/domain/services/dynamic_graph_builder.py`

- [ ] **Step 1: Add test for repeated direct successor**

Append:

```python
def test_relaxed_reachability_keeps_completed_direct_successor_for_loop(mock_feature_configs):
    events = [
        _event(0, "A", instance_id="A_1"),
        _event(1, "B", instance_id="B_1"),
        _event(2, "A", instance_id="A_2"),
    ]
    encoder = FeatureEncoder(
        feature_configs=mock_feature_configs,
        traces=[_trace("train", "v1", events)],
    )
    repository = InMemoryNetworkXRepository()
    repository.save_process_structure(
        "v1",
        ProcessStructureDTO(
            version="v1",
            allowed_edges=[
                ("A", "B"),
                ("B", "A"),
            ],
        ),
    )

    prefix = PrefixSlice(
        case_id="eval",
        process_version="v1",
        prefix_events=events[:2],
        target_event=events[2],
    )

    contract = DynamicGraphBuilder(
        feature_encoder=encoder,
        knowledge_port=repository,
        process_state_mask_enabled=True,
        process_state_mask_source="relaxed_reachability",
        process_state_mask_relaxed_suppress_completed=True,
        process_state_mask_relaxed_anchor_policy="open_successors",
    ).build_graph(prefix)

    activity_vocab = encoder.categorical_vocabs[encoder.activity_feature_name]
    a_idx = int(activity_vocab["A"])

    assert bool(contract["allowed_target_mask"][a_idx]) is True
    assert bool(contract["candidate_allowed_target_mask"][a_idx]) is True
```

- [ ] **Step 2: Run test**

Run:

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\domain\test_dynamic_graph_builder_parallel_process_state_mask.py::test_relaxed_reachability_keeps_completed_direct_successor_for_loop -v
```

Expected: pass. If it fails, ensure direct successors are unioned after relaxed
completed suppression and are not removed by the relaxed filter.

- [ ] **Step 3: Add test for active repeat after previous completion**

Append:

```python
def test_relaxed_reachability_keeps_completed_token_when_new_instance_is_active(mock_feature_configs):
    events = [
        _event(0, "A", lifecycle="complete", instance_id="A_1"),
        _event(1, "B", lifecycle="start", instance_id="B_1"),
        _event(2, "B", lifecycle="complete", instance_id="B_1"),
        _event(3, "B", lifecycle="start", instance_id="B_2"),
        _event(4, "C", lifecycle="complete", instance_id="C_1"),
        _event(5, "B", lifecycle="complete", instance_id="B_2"),
    ]
    encoder = FeatureEncoder(
        feature_configs=mock_feature_configs,
        traces=[_trace("train", "v1", events)],
    )
    repository = InMemoryNetworkXRepository()
    repository.save_process_structure(
        "v1",
        ProcessStructureDTO(
            version="v1",
            allowed_edges=[
                ("A", "B"),
                ("A", "C"),
                ("B", "D"),
                ("C", "D"),
            ],
        ),
    )

    prefix = PrefixSlice(
        case_id="eval",
        process_version="v1",
        prefix_events=events[:5],
        target_event=events[5],
    )

    contract = DynamicGraphBuilder(
        feature_encoder=encoder,
        knowledge_port=repository,
        process_state_mask_enabled=True,
        process_state_mask_source="relaxed_reachability",
        process_state_mask_relaxed_lookback_events=5,
        process_state_mask_relaxed_max_depth=1,
        process_state_mask_relaxed_max_cardinality_ratio=1.0,
        process_state_mask_relaxed_suppress_completed=True,
        process_state_mask_relaxed_anchor_policy="open_successors",
    ).build_graph(prefix)

    activity_vocab = encoder.categorical_vocabs[encoder.activity_feature_name]
    b_idx = int(activity_vocab["B"])

    assert bool(contract["allowed_target_mask"][b_idx]) is True
    assert bool(contract["candidate_allowed_target_mask"][b_idx]) is True
    assert int(contract["process_state_mask_target_suppressed_by_completed_filter_count"].item()) == 0
```

- [ ] **Step 4: Run repeat/active test**

Run:

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\domain\test_dynamic_graph_builder_parallel_process_state_mask.py::test_relaxed_reachability_keeps_completed_token_when_new_instance_is_active -v
```

Expected: pass. If it fails, completed suppression is too coarse and must keep
tokens present in `active_tokens` or explicit `active_candidates`.

## Task 5: CLI and Config Catalog Wiring

**Files:**

- Modify: `src/cli.py`
- Modify: `configs/ui/config_catalog.yaml`
- Modify: `tests/application/test_experiment_ui_catalog_fields.py`

- [ ] **Step 1: Add CLI wiring**

In the `DynamicGraphBuilder` construction in `src/cli.py`, add:

```python
process_state_mask_relaxed_suppress_completed=bool(
    training_cfg.get("process_state_mask_relaxed_suppress_completed", True)
),
process_state_mask_relaxed_anchor_policy=str(
    training_cfg.get("process_state_mask_relaxed_anchor_policy", "open_successors")
),
process_state_mask_relaxed_loop_policy=str(
    training_cfg.get("process_state_mask_relaxed_loop_policy", "keep_direct_successor_repeats")
),
```

- [ ] **Step 2: Add config catalog entries**

Add entries under `training.*` near existing relaxed reachability fields:

```yaml
  training.process_state_mask_relaxed_suppress_completed:
    path: training.process_state_mask_relaxed_suppress_completed
    label: Suppress completed relaxed candidates
    type: bool
    default: true
    level: advanced
    group: Training / Process State Mask
    active_when:
      training.process_state_mask_source: relaxed_reachability
    description: >
      Removes already-completed relaxed-only candidates from the approximate
      relaxed reachability mask. Direct successors and active candidates remain
      allowed to preserve loop/rework behavior.

  training.process_state_mask_relaxed_anchor_policy:
    path: training.process_state_mask_relaxed_anchor_policy
    label: Relaxed anchor policy
    type: enum
    default: open_successors
    enum:
      - recent_prefix
      - open_successors
    level: advanced
    group: Training / Process State Mask
    active_when:
      training.process_state_mask_source: relaxed_reachability
    description: >
      Controls which recent prefix activities can seed relaxed reachability.
      open_successors skips anchors only when their direct successors are
      already completed, not active, and have no loop/rework evidence.

  training.process_state_mask_relaxed_loop_policy:
    path: training.process_state_mask_relaxed_loop_policy
    label: Relaxed loop policy
    type: enum
    default: keep_direct_successor_repeats
    enum:
      - keep_direct_successor_repeats
    level: advanced
    group: Training / Process State Mask
    active_when:
      training.process_state_mask_source: relaxed_reachability
    description: >
      Keeps completed candidates when they are direct successors of the current
      last activity, preserving loop/rework paths while suppressing stale
      relaxed-only candidates.
```

- [ ] **Step 3: Extend catalog test**

In `tests/application/test_experiment_ui_catalog_fields.py`, add the three new
fields to the existing process-state-mask assertion list and verify:

```python
assert catalog["training.process_state_mask_relaxed_anchor_policy"]["enum"] == [
    "recent_prefix",
    "open_successors",
]
assert catalog["training.process_state_mask_relaxed_loop_policy"]["enum"] == [
    "keep_direct_successor_repeats",
]
```

- [ ] **Step 4: Run catalog test**

Run:

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\application\test_experiment_ui_catalog_fields.py -v
```

Expected: pass.

## Task 6: MLflow Diagnostics Propagation

**Files:**

- Modify: `src/application/use_cases/trainer.py`
- Test: `tests/application/test_drift_one_pass_evaluation.py`

- [ ] **Step 1: Extend process-state diagnostics bucket**

Where the trainer initializes process-state mask stats, add sums for:

```python
"process_state_mask_relaxed_raw_candidate_count_sum": 0.0,
"process_state_mask_relaxed_suppressed_completed_count_sum": 0.0,
"process_state_mask_relaxed_final_candidate_count_sum": 0.0,
"process_state_mask_target_suppressed_by_completed_filter_count_sum": 0.0,
"process_state_mask_relaxed_anchor_count_sum": 0.0,
"process_state_mask_relaxed_skipped_closed_anchor_count_sum": 0.0,
```

- [ ] **Step 2: Accumulate graph contract values**

Next to existing accumulation for `process_state_mask_relaxed_candidate_count`,
read:

```python
raw_count = contract.get("process_state_mask_relaxed_raw_candidate_count")
suppressed_count = contract.get("process_state_mask_relaxed_suppressed_completed_count")
final_count = contract.get("process_state_mask_relaxed_final_candidate_count")
target_suppressed_count = contract.get("process_state_mask_target_suppressed_by_completed_filter_count")
anchor_count = contract.get("process_state_mask_relaxed_anchor_count")
skipped_anchor_count = contract.get("process_state_mask_relaxed_skipped_closed_anchor_count")
```

For each tensor, detach and sum safely:

```python
if isinstance(raw_count, torch.Tensor):
    bucket["process_state_mask_relaxed_raw_candidate_count_sum"] = float(
        bucket.get("process_state_mask_relaxed_raw_candidate_count_sum", 0.0)
        + raw_count.detach().float().sum().item()
    )
```

Repeat for the other five fields.

- [ ] **Step 3: Log mean diagnostics**

Where existing process-state means are emitted, add:

```python
process_state_mask_relaxed_raw_candidate_count
process_state_mask_relaxed_suppressed_completed_count
process_state_mask_relaxed_final_candidate_count
process_state_mask_relaxed_anchor_count
process_state_mask_relaxed_skipped_closed_anchor_count
process_state_mask_target_suppressed_by_completed_filter_rate
process_state_mask_completed_suppression_rate
```

Each should divide by the same sample count used for the existing active/relaxed
candidate count metrics.

Use:

```python
process_state_mask_target_suppressed_by_completed_filter_rate = (
    target_suppressed_count_sum / max(sample_count, 1)
)
process_state_mask_completed_suppression_rate = (
    suppressed_completed_count_sum / max(raw_candidate_count_sum, 1.0)
)
```

The target suppression rate is a safety metric. Any value above zero means the
new completed-candidate filter may be too aggressive for the current dataset or
loop/rework semantics.

- [ ] **Step 4: Run drift one-pass test**

Run:

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\application\test_drift_one_pass_evaluation.py -v
```

Expected: pass.

## Task 7: Documentation Updates

**Files:**

- Modify: `docs/GNN_RUNTIME_MVP2_5.MD`
- Modify: `docs/current/project-state.md`
- Modify: `docs/dissertation_changes.MD`

- [ ] **Step 1: Update runtime docs**

In `docs/GNN_RUNTIME_MVP2_5.MD`, replace the relaxed reachability description
with:

```markdown
`training.process_state_mask_source=relaxed_reachability` is an approximate
fallback for flattened/interleaved XES logs without reliable lifecycle active
state. It expands the mask from recent prefix anchors through bounded topology
reachability, then suppresses already-completed relaxed-only candidates when
`training.process_state_mask_relaxed_suppress_completed=true`.

Direct successors of the last completed activity, explicit active candidates,
and tokens with currently active instances remain allowed even if they appeared
earlier in the prefix. This protects loop, rework, multiple-instance, and BPMS
retry behavior.
```

- [ ] **Step 2: Update project-state runtime note**

Add a new dated runtime update:

```markdown
## Runtime Update 2026-06-03: State-Aware Relaxed Reachability

`relaxed_reachability` now supports completed-candidate suppression,
active-instance protection, and open-successor anchor filtering. This reduces
over-expanded allowed masks in parallel/interleaved completion-only XES logs,
where naive relaxed reachability can improve OOS while hurting strict
next-activity accuracy by allowing stale parallel sibling tasks.
```

- [ ] **Step 3: Update dissertation notes**

Append a short note:

```markdown
- Relaxed topology-derived masks can improve OOS/calibration by keeping the
  model inside the process topology, but if the mask is too broad it converts
  OOS errors into allowed-but-wrong strict errors. For parallel XES without true
  BPMN marking, state-aware suppression of completed relaxed-only candidates is
  required before treating mask improvements as evidence of predictive benefit.
```

## Task 8: Focused Verification

**Files:**

- No source edits in this task.

- [ ] **Step 1: Run focused domain and catalog tests**

Run:

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\domain\test_dynamic_graph_builder_parallel_process_state_mask.py tests\application\test_experiment_ui_catalog_fields.py -v
```

Expected: pass.

- [ ] **Step 2: Run one-pass drift regression**

Run:

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\application\test_drift_one_pass_evaluation.py -v
```

Expected: pass.

- [ ] **Step 3: Run architecture guard**

Run:

```powershell
.\.venv-modern\Scripts\python.exe tools\architecture_guard.py
```

Expected: `[ARCH_GUARD] OK`.

## Manual Experiment Plan

No retraining is required to evaluate this mask refinement. Re-run drift with
the same checkpoint and compare three configurations:

### A. Lifecycle/direct baseline

```yaml
training.process_state_mask_enabled: true
training.process_state_mask_source: lifecycle_active_set
```

### B. Current relaxed behavior compatibility

```yaml
training.process_state_mask_enabled: true
training.process_state_mask_source: relaxed_reachability
training.process_state_mask_relaxed_suppress_completed: false
training.process_state_mask_relaxed_anchor_policy: recent_prefix
```

### C. New state-aware relaxed behavior

```yaml
training.process_state_mask_enabled: true
training.process_state_mask_source: relaxed_reachability
training.process_state_mask_relaxed_suppress_completed: true
training.process_state_mask_relaxed_anchor_policy: open_successors
training.process_state_mask_relaxed_loop_policy: keep_direct_successor_repeats
```

Primary success criteria:

- `drift_window_target_in_mask_rate` remains materially higher than lifecycle/direct baseline.
- `drift_window_test_oos` remains materially lower than lifecycle/direct baseline.
- `drift_window_ambiguous_prefix_rate` decreases versus current relaxed behavior.
- `drift_window_strict_error_but_allowed_rate` decreases versus current relaxed behavior.
- `drift_window_strict_macro_f1` improves or at least stops degrading versus current relaxed behavior.
- `process_state_mask_target_suppressed_by_completed_filter_rate == 0`.
- `process_state_mask_completed_suppression_rate > 0` on the complex parallel dataset.

Acceptable trade-off:

- `target_in_mask_rate` may drop slightly versus current relaxed behavior if the removed candidates are stale completed siblings.

Failure interpretation:

- If target inclusion improves but strict-F1 still degrades, the issue is no
  longer simple mask over-expansion. Next step should be true BPMN marking
  replay or simulator-provided active/enabled candidate state.
- If `process_state_mask_target_suppressed_by_completed_filter_rate` is non-zero after diagnostics
  are added, the completed suppression is too aggressive for loops/rework and
  must be constrained to parallel-region candidates rather than all completed
  prefix tokens.

## Self-Review

- Spec coverage: covers state-aware relaxed filtering, config, CLI, diagnostics,
  tests, docs, and manual experiment interpretation.
- Placeholder scan: no `TBD` or unspecified implementation steps remain.
- Type consistency: new graph contract fields are named consistently across
  builder, trainer diagnostics, and tests.
- Scope check: this plan intentionally avoids full BPMN marking replay and model
  architecture changes.
