# EOPKGTopologyConditioned Stage 2 True Candidate Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current fixed-label projection compatibility path with an opt-in true topology candidate contract where train/eval/drift can compute loss and metrics directly on BPMN candidate ids for the current topology version.

**Architecture:** Keep existing `EOPKGTopologyConditioned.forward()` and `training.dynamic_candidate_contract_enabled=true` projection path for backward compatibility. Add a new Stage 2 true mode that treats prediction candidates as topology target nodes (`BPMN node ids`) rather than fixed activity classes. The model emits node-level candidate logits `[B, C_v]`; trainer/evaluator maps targets to candidate sets and computes candidate-level loss/metrics without projecting to `[B, C_train]`.

**Tech Stack:** Python 3.11, PyTorch, PyTorch Geometric, MLflow metrics/traces, existing `GraphTensorContract`, `CandidatePredictionOutput`, `ModelTrainer`, `DynamicGraphBuilder`.

**Blocking reviewer correction:** true candidate-id mode must not reuse the
old "first structural payload in batch" assumption. Once logits have shape
`[B, C_v]`, mixed-topology batches can have different candidate axes per row,
so using the first graph's topology for the whole batch silently corrupts loss
and metrics. Stage 2 true therefore requires a batch topology policy before
candidate-level training is enabled.

---

## 1. Current Diagnosis

The latest `_EOPKGTC-UN` run proved that Stage 2 compatibility is wired, but it is not yet a true candidate contract:

```text
vocab_activity_size = 37
candidate_dynamic_count = 37
duplicate_candidate_count_max = 1
candidate_class_score_mean_abs == candidate_node_score_mean_abs
```

Because `candidate_logits [B, C_v]` were projected back into `[B, C_train]`, and `C_v == C_train`, the result was mathematically identical to Stage 1.

Therefore the next step must stop using fixed-label projection as the primary loss/evaluation path.

---

## 2. Design Decisions

### 2.1 Candidate Axis

Use node-level topology candidates:

```text
candidate axis C_v = target BPMN nodes in current topology version
candidate_ids[j] = BPMN node id
candidate_class_index[j] = global activity class index for that BPMN node
candidate_node_index[j] = structural node row index
candidate_logits[:, j] = node_logits[:, candidate_node_index[j]]
```

Rationale:

- supports duplicate activity labels as separate BPMN candidates;
- supports added/removed BPMN nodes;
- keeps gateways/events/non-target nodes available for message passing but excluded from prediction candidates;
- avoids collapsing true candidate identity back into class-level fixed vocab too early.

### 2.2 Target Semantics

Current logs usually provide activity label/class target, not exact BPMN node id. Therefore the first true candidate loss must be set-aware:

```text
target_candidate_mask[b, j] = candidate_class_index[j] == y[b]
```

Loss:

```text
L_candidate =
  -logsumexp(candidate_logits[b, target_candidate_mask[b]])
  + logsumexp(candidate_logits[b, all candidates])
```

If exactly one target candidate matches, this is normal CE. If multiple duplicate BPMN nodes share the target label, this is MIL/set-aware CE.

### 2.3 Compatibility

Add a new explicit mode instead of changing the meaning of the existing boolean:

```yaml
training.candidate_contract_mode: fixed_projection | candidate_id
```

Backward compatibility:

```text
missing key -> fixed_projection if training.dynamic_candidate_contract_enabled=true
missing key -> fixed_label for old models
```

Recommended runtime:

```yaml
model.type: EOPKGTopologyConditioned
training.candidate_contract_mode: candidate_id
```

The existing `training.dynamic_candidate_contract_enabled` remains a compatibility alias for `fixed_projection`.

### 2.4 Scope

Included:

1. Candidate identity contract.
2. Target mapping.
3. Candidate-level loss.
4. Candidate-level evaluator.
5. Drift/trace/MLflow metrics.

Excluded:

- semantic text/BPMN metadata embeddings;
- non-parametric semantic retrieval head;
- duration/resource/risk multi-task heads;
- exact BPMN node target supervision when logs do not contain exact node ids.

### 2.5 Source / Topology View Alignment

Stage 2 true assumes that logs and structure are already synchronized by config
and data preparation.

Valid source/view combinations:

```yaml
Camunda-like source:
  gateway_mode: preserve
  logs contain the same technical/gateway/runtime nodes that the topology view exposes

XES task-label source:
  gateway_mode: collapse_for_prediction
  structure is projected to the same task-like prediction view as the XES log
```

Invalid combinations, such as XES logs with full preserve topology when XES does
not contain gateway/runtime labels, are configuration/data-preparation errors.
They must not be compensated inside the prediction model.

Contract invariant:

```text
candidate eligibility is not decided by BPMN type inside the model;
candidate eligibility is encoded by the prebuilt mapping;
struct_node_to_class_index[row] >= 0 means this structural row maps to a target/log vocab class;
struct_node_to_class_index[row] == -1 means this row has no target/log vocab class in the selected view;
candidate_ids are structural row ids aligned to the selected, already-prepared topology view.
```

`struct_node_to_class_index` is therefore not a gateway hardcode. It is the
tensor-friendly mapping:

```text
structural row index -> target/log vocabulary index
```

The model only uses it to map candidate rows to targets for loss/evaluation.
It must not infer business semantics such as "gateway is invalid" by itself.

Required tests:

```text
aligned view:
  every candidate row with struct_node_to_class_index >= 0 can be mapped to y;
  rows with -1 can participate in message passing but cannot be selected as y.

misaligned view:
  if target y is absent from candidate mapping, target_in_candidate_set_rate drops;
research profile should report/fail this as alignment/config issue, not model failure.
```

### 2.6 Batch Topology Policy

True candidate-id mode requires one of these policies:

```yaml
training.candidate_batch_topology_policy: single_topology_required | group_by_topology
```

Initial implementation:

```text
single_topology_required:
  every batch must contain exactly one process_version/as_of/topology identity;
  trainer raises before forward if more than one identity is detected.

group_by_topology:
  future extension;
  split a mixed batch into topology-homogeneous sub-batches, run each separately,
  then concatenate losses/metrics.
```

Stage 2 true should start with `single_topology_required`. This is stricter
than the existing structural payload contract, but it prevents silent
dimension and candidate-id errors.

Required invariant:

```text
candidate_contract_mode=candidate_id
=> one candidate axis per forward/loss call
=> batch_topology_unique_count == 1
```

If versioned batching is unavailable or a loader produces mixed topology rows,
training/eval must fail with a clear error instead of falling back to the first
payload.

### 2.7 MIL Loss Diagnostics

MIL/set-aware target loss is necessary when logs provide only activity labels
and topology contains duplicate BPMN nodes with the same label. However,
`logsumexp(target candidates)` can hide duplicate collapse if all duplicate
candidate logits remain identical.

Add diagnostics:

```text
candidate_target_set_size_mean
candidate_target_set_size_max
candidate_target_logit_entropy_mean
candidate_target_logit_variance_mean
candidate_duplicate_target_rate
```

Interpretation:

```text
target_set_size > 1 and target_logit_variance ~= 0 across epochs
=> structural encoder is not separating duplicate nodes;
=> candidate-id mode has not solved duplicate activity ambiguity yet.
```

### 2.8 Missing Target Guard

`missing_target` is not a normal training case. It means the log target cannot
be found in the candidate set for the selected topology view.

Add:

```yaml
training.candidate_missing_target_fail_threshold: 0.10
```

Behavior:

```text
training:
  if missing_target_rate > threshold in any train batch/epoch, raise ValueError.

evaluation/drift:
  never crash by default, but report candidate_missing_target_rate and
  target_in_candidate_set_rate; dissertation-grade runs must treat high missing
  target as invalid alignment/config.
```

The loss may still return a finite tensor for edge cases, but the trainer must
not allow long training runs where most rows provide zero useful gradient.

---

## 3. File Structure

### Create

- `src/application/services/candidate_target_mapping.py`
  - maps fixed class targets to topology-local candidate masks/indexes;
  - computes set-aware CE and candidate-level metrics helpers.

- `tests/application/test_candidate_target_mapping.py`
  - unit tests for target mapping, duplicate labels, missing targets, CE.
  - unit tests for MIL duplicate entropy/variance diagnostics.
  - unit tests for missing target threshold guard.

### Modify

- `src/domain/entities/tensor_contract.py`
  - add optional `struct_node_ids`, `candidate_ids`, `target_candidate_mask`, `target_candidate_index`.

- `src/domain/entities/candidate_prediction.py`
  - extend `CandidatePredictionOutput` with node-level candidate ids/indexes and target mapping helpers.

- `src/domain/services/dynamic_graph_builder.py`
  - include stable structural node ids in compiled topology and graph contracts.

- `src/domain/models/topology_conditioned.py`
  - add node-level `forward_candidate()` output.
  - keep class-level output for `forward()` and compatibility projection.

- `src/application/use_cases/trainer.py`
  - parse `training.candidate_contract_mode`;
  - enforce candidate batch topology policy;
  - enforce candidate missing target threshold in training;
  - route candidate-id mode through candidate-level loss and evaluator;
  - keep old projection path unchanged.

- `src/application/services/structural_trace_payload_builder.py`
  - add candidate ids, target candidate set, predicted candidate id to traces.

- `configs/ui/config_catalog.yaml`
  - add `training.candidate_contract_mode`.
  - add `training.candidate_batch_topology_policy`.
  - add `training.candidate_missing_target_fail_threshold`.

- `docs/GNN_RUNTIME_MVP2_5.MD`
  - document true candidate-id mode.

- `docs/GNN_LEARNING_METODOLOGY.MD`
  - mark Phase 2 candidate contract as implemented when finished.

- `docs/current/project-state.md`
  - update runtime state.

- `docs/current/architecture-debt.md`
  - move `dynamic_candidate_prediction_contract` from active P0 to partially closed or update remaining debt.

- `docs/dissertation_changes.MD`
  - add concise methodological note: class-level targets require set-aware candidate loss until exact BPMN node targets exist.

---

## 4. Task 1: Candidate Identity Contract

**Files:**

- Modify: `src/domain/entities/tensor_contract.py`
- Modify: `src/domain/entities/candidate_prediction.py`
- Test: `tests/domain/test_candidate_prediction_contract.py`

- [ ] **Step 1: Write failing tests for node-level candidate output**

Create or extend `tests/domain/test_candidate_prediction_contract.py`:

```python
import torch

from src.domain.entities.candidate_prediction import CandidatePredictionOutput


def test_candidate_output_maps_class_targets_to_candidate_mask_with_duplicates():
    output = CandidatePredictionOutput(
        candidate_logits=torch.tensor([[0.2, 1.0, 0.7]], dtype=torch.float32),
        candidate_class_index=torch.tensor([4, 5, 5], dtype=torch.long),
        node_logits=torch.tensor([[0.2, 1.0, 0.7]], dtype=torch.float32),
        node_to_candidate_index=torch.tensor([0, 1, 2], dtype=torch.long),
        node_to_class_index=torch.tensor([4, 5, 5], dtype=torch.long),
        candidate_node_index=torch.tensor([0, 1, 2], dtype=torch.long),
        candidate_ids=("task_a", "task_b_left", "task_b_right"),
    )

    mask = output.map_targets_to_candidate_mask(torch.tensor([5], dtype=torch.long))

    assert mask.tolist() == [[False, True, True]]


def test_candidate_output_reports_missing_target_when_not_in_candidate_set():
    output = CandidatePredictionOutput(
        candidate_logits=torch.tensor([[0.2, 1.0]], dtype=torch.float32),
        candidate_class_index=torch.tensor([4, 5], dtype=torch.long),
        node_logits=torch.tensor([[0.2, 1.0]], dtype=torch.float32),
        node_to_candidate_index=torch.tensor([0, 1], dtype=torch.long),
        node_to_class_index=torch.tensor([4, 5], dtype=torch.long),
        candidate_node_index=torch.tensor([0, 1], dtype=torch.long),
        candidate_ids=("task_a", "task_b"),
    )

    mask = output.map_targets_to_candidate_mask(torch.tensor([9], dtype=torch.long))

    assert mask.tolist() == [[False, False]]
```

- [ ] **Step 2: Run failing tests**

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\domain\test_candidate_prediction_contract.py -v
```

Expected: fail because `candidate_node_index`, `candidate_ids`, and `map_targets_to_candidate_mask()` are missing.

- [ ] **Step 3: Extend `CandidatePredictionOutput`**

Modify `src/domain/entities/candidate_prediction.py`:

```python
@dataclass(frozen=True)
class CandidatePredictionOutput:
    candidate_logits: torch.Tensor
    candidate_class_index: torch.LongTensor
    node_logits: torch.Tensor
    node_to_candidate_index: torch.LongTensor
    node_to_class_index: torch.LongTensor
    candidate_node_index: torch.LongTensor | None = None
    candidate_ids: tuple[str, ...] = ()
    fixed_label_logits: torch.Tensor | None = None

    @property
    def candidate_count(self) -> int:
        return int(self.candidate_logits.size(1))

    def map_targets_to_candidate_mask(self, targets: torch.Tensor) -> torch.BoolTensor:
        flat_targets = targets.detach().to(device=self.candidate_class_index.device, dtype=torch.long).view(-1)
        class_index = self.candidate_class_index.to(device=flat_targets.device, dtype=torch.long).view(1, -1)
        return class_index.eq(flat_targets.view(-1, 1))

    def map_targets_to_candidate_index(self, targets: torch.Tensor, missing_value: int = -1) -> torch.LongTensor:
        mask = self.map_targets_to_candidate_mask(targets)
        mapped = torch.full((mask.size(0),), int(missing_value), dtype=torch.long, device=mask.device)
        has_one = mask.sum(dim=1) == 1
        if bool(has_one.any()):
            mapped[has_one] = torch.argmax(mask[has_one].long(), dim=1)
        return mapped
```

- [ ] **Step 4: Extend `GraphTensorContract`**

Add optional fields:

```python
struct_node_ids: NotRequired[list[str] | None]
candidate_ids: NotRequired[list[str] | None]
target_candidate_mask: NotRequired[torch.BoolTensor | None]
target_candidate_index: NotRequired[torch.LongTensor | None]
```

- [ ] **Step 5: Run tests**

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\domain\test_candidate_prediction_contract.py tests\domain\test_eopkg_topology_conditioned.py -v
```

Expected: pass.

---

## 5. Task 2: Propagate BPMN Node Candidate IDs

**Files:**

- Modify: `src/domain/services/topology_projection_alignment.py`
- Modify: `src/domain/services/dynamic_graph_builder.py`
- Modify: `src/application/use_cases/trainer.py`
- Test: `tests/application/test_dynamic_graph_builder_candidate_ids.py`
- Test: `tests/application/test_trainer_forward_stats_logging.py`

- [ ] **Step 1: Write failing DynamicGraphBuilder test**

Create `tests/application/test_dynamic_graph_builder_candidate_ids.py`:

```python
import torch


def test_dynamic_graph_builder_contract_includes_struct_node_ids(dynamic_builder_fixture):
    contract = dynamic_builder_fixture.build_one_contract()

    assert "struct_node_ids" in contract
    assert isinstance(contract["struct_node_ids"], list)
    assert len(contract["struct_node_ids"]) == int(contract["struct_node_to_class_index"].numel())
```

If there is no existing fixture with this exact name, implement the test using the smallest existing `DynamicGraphBuilder` fixture pattern from `tests/integration/test_bpmn_xes_stats_to_gnn_cycle.py` and assert on the returned graph contract.

- [ ] **Step 2: Run failing test**

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\application\test_dynamic_graph_builder_candidate_ids.py -v
```

Expected: fail because `struct_node_ids` is not present.

- [ ] **Step 3: Add source/topology alignment tests**

Extend `tests/application/test_dynamic_graph_builder_candidate_ids.py` with two
explicit cases:

```python
def test_candidate_ids_are_driven_by_struct_node_to_class_mapping_not_bpmn_type():
    contract = build_contract_with_prepared_mapping(
        struct_node_ids=["task_a", "gateway_xor", "task_b"],
        struct_node_to_class_index=torch.tensor([0, 1, 2], dtype=torch.long),
    )

    assert contract["struct_node_ids"] == ["task_a", "gateway_xor", "task_b"]
    assert contract["struct_node_to_class_index"].tolist() == [0, 1, 2]


def test_message_passing_only_rows_are_encoded_by_minus_one_mapping():
    contract = build_contract_with_prepared_mapping(
        struct_node_ids=["task_a", "context_only_node", "task_b"],
        struct_node_to_class_index=torch.tensor([0, -1, 1], dtype=torch.long),
    )

    assert contract["struct_node_ids"][1] == "context_only_node"
    assert int(contract["struct_node_to_class_index"][1].item()) == -1
```

These tests intentionally avoid hardcoding gateway semantics. They verify that
candidate eligibility follows the prepared mapping.

Expected: fail until `struct_node_ids` and `struct_node_to_class_index` are
preserved together.

- [ ] **Step 4: Add prediction node ids to topology compilation**

In `DynamicGraphBuilder._resolve_compiled_topology()`, derive a stable node list aligned to `struct_node_to_class_index`.

For `collapse_for_prediction`, current projection uses
`struct_node_to_class_index = torch.arange(num_classes)`, so a safe first
implementation is:

```python
class_to_token = {idx: token for token, idx in activity_vocab.items()}
struct_node_ids = [
    str(class_to_token.get(int(idx), f"__class_{int(idx)}__"))
    for idx in range(num_classes)
]
```

For any source mode, structural rows must follow the selected topology view and
candidate mapping must follow the source log vocabulary:

```text
rows mapped to source target vocabulary: struct_node_to_class_index >= 0
rows not mapped to source target vocabulary: struct_node_to_class_index == -1
```

If the current compiler cannot yet expose full row ids and mapping for a source
mode, keep the implementation scoped to the existing projected rows and add an
architecture debt note before claiming full support for that source/view
combination.

When projection compiler exposes duplicate BPMN node rows later, this list must
be replaced by actual BPMN node ids from topology rows.

Add to `compiled`:

```python
"struct_node_ids": tuple(struct_node_ids),
```

Attach to contract:

```python
struct_node_ids = compiled.get("struct_node_ids")
if isinstance(struct_node_ids, (list, tuple)):
    contract["struct_node_ids"] = [str(item) for item in struct_node_ids]
```

- [ ] **Step 5: Pass `struct_node_ids` through trainer contract conversion**

Modify `ModelTrainer._select_structural_payload_for_forward()` to return `struct_node_ids` in addition to tensors, and `_data_to_contract()` to set:

```python
if isinstance(struct_node_ids, list):
    contract["struct_node_ids"] = struct_node_ids
```

For `candidate_contract_mode != candidate_id`, keep current first-payload
behavior for backward compatibility.

For `candidate_contract_mode == candidate_id`, do not silently select the first
payload when the batch contains more than one topology identity. This mode must
be guarded by Task 2A before Task 4 is enabled.

- [ ] **Step 6: Add trainer contract extraction test**

Add to `tests/application/test_trainer_forward_stats_logging.py`:

```python
def test_data_to_contract_preserves_struct_node_ids():
    trainer = _make_minimal_trainer()
    data = _minimal_data()
    data.struct_node_ids = ["task_a", "task_b"]
    data.struct_node_to_class_index = torch.tensor([0, 1], dtype=torch.long)
    data.structural_edge_index = torch.zeros((2, 0), dtype=torch.long)

    contract = trainer._data_to_contract(data)

    assert contract["struct_node_ids"] == ["task_a", "task_b"]
```

Use existing local helper names in that test file; if helper names differ, add the minimal model/trainer setup already used by neighboring tests.

- [ ] **Step 7: Run tests**

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\application\test_dynamic_graph_builder_candidate_ids.py tests\application\test_trainer_forward_stats_logging.py -v
```

Expected: pass.

---

## 5A. Task 2A: Candidate Batch Topology Guard

**Files:**

- Modify: `src/application/use_cases/trainer.py`
- Modify: `src/domain/entities/tensor_contract.py`
- Modify: `src/domain/services/dynamic_graph_builder.py`
- Test: `tests/application/test_trainer_candidate_batch_topology.py`

- [ ] **Step 1: Define topology identity in contracts**

Add optional identity fields to graph/data contracts where available:

```python
process_id: NotRequired[str | None]
process_version: NotRequired[str | None]
as_of_snapshot: NotRequired[str | None]
topology_identity: NotRequired[str | None]
```

Build `topology_identity` from stable source metadata:

```text
topology_identity = process_id + "::" + process_version + "::" + gateway_mode + "::" + as_of_snapshot
```

If exact fields differ in current code, use the already available
`process_version` and selected topology/gateway mode. The identity only needs
to be stable enough to detect mixed candidate axes inside one batch.

- [ ] **Step 2: Add failing mixed-topology test**

Create `tests/application/test_trainer_candidate_batch_topology.py`:

```python
def test_candidate_id_mode_rejects_mixed_topology_batch():
    trainer = _make_candidate_id_trainer(
        config={
            "candidate_contract_mode": "candidate_id",
            "candidate_batch_topology_policy": "single_topology_required",
        }
    )
    data = _minimal_batched_data_with_topology_identities(["loan::v1", "loan::v2"])

    with pytest.raises(ValueError, match="candidate_id.*mixed topology"):
        trainer._data_to_contract(data)
```

Use existing minimal trainer/data helpers if names differ.

- [ ] **Step 3: Implement unique topology detection**

Add helper in `ModelTrainer`:

```python
def _extract_topology_identities(self, data: Any) -> tuple[str, ...]:
    raw = getattr(data, "topology_identity", None)
    if raw is None:
        raw = getattr(data, "process_version", None)
    return _normalize_batch_string_values(raw)
```

Then guard:

```python
if self.candidate_contract_mode == "candidate_id":
    identities = self._extract_topology_identities(data)
    if self.candidate_batch_topology_policy == "single_topology_required":
        unique = sorted(set(identities))
        if len(unique) > 1:
            raise ValueError(
                "candidate_contract_mode=candidate_id received a mixed topology batch. "
                f"unique_topology_identities={unique}. Use versioned batching/sampling "
                "or implement candidate_batch_topology_policy=group_by_topology."
            )
```

If identity metadata is missing in candidate-id mode, fail in research profile
or warn/fail according to an explicit policy. Do not assume that missing
identity means homogeneous.

- [ ] **Step 4: Add homogeneous batch test**

```python
def test_candidate_id_mode_accepts_single_topology_batch():
    trainer = _make_candidate_id_trainer(...)
    data = _minimal_batched_data_with_topology_identities(["loan::v2", "loan::v2"])

    contract = trainer._data_to_contract(data)

    assert contract["topology_identity"] == "loan::v2"
```

- [ ] **Step 5: Add future debt for grouped execution**

Update `docs/current/architecture-debt.md` during implementation:

```text
candidate_batch_topology_grouping:
  current_behavior: candidate_id mode requires homogeneous topology batches.
  remaining_debt: group mixed batches by topology identity and run candidate
  forward/loss per group.
```

- [ ] **Step 6: Run tests**

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\application\test_trainer_candidate_batch_topology.py tests\application\test_trainer_forward_stats_logging.py -v
```

Expected: pass.

---

## 6. Task 3: Node-Level `forward_candidate()`

**Files:**

- Modify: `src/domain/models/topology_conditioned.py`
- Test: `tests/domain/test_eopkg_topology_conditioned.py`

- [ ] **Step 1: Write failing tests for node-level candidates**

Add to `tests/domain/test_eopkg_topology_conditioned.py`:

```python
def test_topology_conditioned_forward_candidate_uses_node_level_candidate_axis_for_duplicates():
    model = _make_topology_conditioned_model(output_dim=3)
    contract = _make_contract(
        struct_node_to_class_index=torch.tensor([0, 1, 1, -1], dtype=torch.long),
        struct_node_ids=["task_a", "task_b_left", "task_b_right", "gateway"],
    )

    output = model.forward_candidate(contract)

    assert output.candidate_count == 3
    assert output.candidate_ids == ("task_a", "task_b_left", "task_b_right")
    assert output.candidate_class_index.tolist() == [0, 1, 1]
    assert output.candidate_node_index.tolist() == [0, 1, 2]
    assert output.node_to_candidate_index.tolist() == [0, 1, 2, -1]
```

Existing helper names may differ; use current helpers in the same file. The core expectation is that duplicate class `1` produces two candidates, not one pooled class.

- [ ] **Step 2: Run failing test**

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\domain\test_eopkg_topology_conditioned.py::test_topology_conditioned_forward_candidate_uses_node_level_candidate_axis_for_duplicates -v
```

Expected: fail because current dynamic pooling returns unique class candidates.

- [ ] **Step 3: Implement node-level candidate extraction**

In `src/domain/models/topology_conditioned.py`, replace or add a new helper:

```python
def _node_level_candidates(
    self,
    node_scores: torch.Tensor,
    node_to_class: torch.Tensor,
    struct_node_ids: list[str] | None,
) -> tuple[torch.Tensor, torch.LongTensor, torch.LongTensor, torch.LongTensor, tuple[str, ...]]:
    node_to_class = node_to_class.to(device=node_scores.device, dtype=torch.long)
    valid = (node_to_class >= 0) & (node_to_class < self.output_dim)
    candidate_node_index = torch.nonzero(valid, as_tuple=False).view(-1).long()
    candidate_logits = node_scores[:, candidate_node_index]
    candidate_class_index = node_to_class[candidate_node_index].long()
    node_to_candidate_index = torch.full_like(node_to_class, -1, dtype=torch.long)
    if candidate_node_index.numel() > 0:
        node_to_candidate_index[candidate_node_index] = torch.arange(
            int(candidate_node_index.numel()),
            device=node_scores.device,
            dtype=torch.long,
        )
    ids: list[str] = []
    raw_ids = list(struct_node_ids or [])
    for node_idx in candidate_node_index.detach().cpu().tolist():
        if 0 <= int(node_idx) < len(raw_ids):
            ids.append(str(raw_ids[int(node_idx)]))
        else:
            ids.append(f"node_{int(node_idx)}")
    return candidate_logits, candidate_class_index, node_to_candidate_index, candidate_node_index, tuple(ids)
```

Use it inside `forward_candidate()`:

```python
struct_node_ids = contract.get("struct_node_ids")
candidate_logits, candidate_class_index, node_to_candidate_index, candidate_node_index, candidate_ids = (
    self._node_level_candidates(node_scores, node_to_class, struct_node_ids if isinstance(struct_node_ids, list) else None)
)
```

Return:

```python
CandidatePredictionOutput(
    candidate_logits=candidate_logits,
    candidate_class_index=candidate_class_index,
    node_logits=node_scores,
    node_to_candidate_index=node_to_candidate_index,
    node_to_class_index=node_to_class,
    candidate_node_index=candidate_node_index,
    candidate_ids=candidate_ids,
    fixed_label_logits=fixed_label_logits,
)
```

Keep `forward()` unchanged.

- [ ] **Step 4: Preserve compatibility projection tests**

Existing tests must still pass:

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\domain\test_eopkg_topology_conditioned.py -v
```

Expected: pass, with updated assertions where old `forward_candidate()` expected unique class candidates.

---

## 7. Task 4: Candidate Target Mapping And Loss

**Files:**

- Create: `src/application/services/candidate_target_mapping.py`
- Modify: `src/application/use_cases/trainer.py`
- Test: `tests/application/test_candidate_target_mapping.py`
- Test: `tests/application/test_evaluator_metrics.py`

- [ ] **Step 1: Write failing mapping/loss tests**

Create `tests/application/test_candidate_target_mapping.py`:

```python
import torch

from src.application.services.candidate_target_mapping import (
    candidate_set_cross_entropy,
    candidate_target_summary,
)


def test_candidate_set_cross_entropy_accepts_duplicate_target_candidates():
    logits = torch.tensor([[0.0, 2.0, 1.5]], dtype=torch.float32)
    target_mask = torch.tensor([[False, True, True]], dtype=torch.bool)

    loss = candidate_set_cross_entropy(logits, target_mask)

    expected = -(torch.logsumexp(logits[:, 1:3], dim=1) - torch.logsumexp(logits, dim=1)).mean()
    assert loss == torch.approx(expected.item())


def test_candidate_target_summary_reports_missing_target():
    target_mask = torch.tensor([[False, False], [True, False]], dtype=torch.bool)
    pred = torch.tensor([1, 0], dtype=torch.long)

    summary = candidate_target_summary(target_mask=target_mask, pred_candidate_index=pred)

    assert summary["target_in_candidate_set_rate"] == 0.5
    assert summary["candidate_hit_rate"] == 0.5


def test_candidate_target_summary_reports_duplicate_target_entropy_and_variance():
    logits = torch.tensor([[0.1, 0.5, 0.5, -0.2]], dtype=torch.float32)
    target_mask = torch.tensor([[False, True, True, False]], dtype=torch.bool)
    pred = torch.tensor([1], dtype=torch.long)

    summary = candidate_target_summary(
        target_candidate_mask=target_mask,
        pred_candidate_index=pred,
        candidate_logits=logits,
    )

    assert summary["candidate_target_set_size_max"] == 2.0
    assert summary["candidate_duplicate_target_rate"] == 1.0
    assert summary["candidate_target_logit_variance_mean"] == 0.0
    assert summary["candidate_target_logit_entropy_mean"] > 0.0
```

- [ ] **Step 2: Run failing tests**

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\application\test_candidate_target_mapping.py -v
```

Expected: fail because service does not exist.

- [ ] **Step 3: Implement service**

Create `src/application/services/candidate_target_mapping.py`:

```python
from __future__ import annotations

from typing import Any

import torch


def candidate_set_cross_entropy(
    candidate_logits: torch.Tensor,
    target_candidate_mask: torch.Tensor,
    *,
    missing_target_penalty: float = 0.0,
) -> torch.Tensor:
    if candidate_logits.dim() != 2:
        raise ValueError("candidate_logits must have shape [B, C_v].")
    mask = target_candidate_mask.to(device=candidate_logits.device, dtype=torch.bool)
    if tuple(mask.shape) != tuple(candidate_logits.shape):
        raise ValueError("target_candidate_mask must match candidate_logits shape.")

    has_target = mask.any(dim=1)
    if not bool(has_target.any()):
        # This keeps the returned tensor differentiable but intentionally has
        # zero gradient. The trainer must separately guard high missing target
        # rates because this indicates broken log/topology mapping.
        return candidate_logits.sum() * 0.0 + float(missing_target_penalty)

    safe_target_logits = candidate_logits.masked_fill(~mask, float("-inf"))
    numerator = torch.logsumexp(safe_target_logits[has_target], dim=1)
    denominator = torch.logsumexp(candidate_logits[has_target], dim=1)
    loss = -(numerator - denominator).mean()
    if missing_target_penalty and bool((~has_target).any()):
        missing_ratio = (~has_target).float().mean().to(device=candidate_logits.device)
        loss = loss + float(missing_target_penalty) * missing_ratio
    return loss


def candidate_target_summary(
    *,
    target_candidate_mask: torch.Tensor,
    pred_candidate_index: torch.Tensor,
    candidate_logits: torch.Tensor | None = None,
) -> dict[str, float]:
    mask = target_candidate_mask.to(dtype=torch.bool)
    pred = pred_candidate_index.to(device=mask.device, dtype=torch.long).view(-1)
    if mask.dim() != 2:
        raise ValueError("target_candidate_mask must have shape [B, C_v].")
    if pred.numel() != mask.size(0):
        raise ValueError("pred_candidate_index must have one value per row.")
    has_target = mask.any(dim=1)
    rows = torch.arange(mask.size(0), device=mask.device)
    pred_valid = (pred >= 0) & (pred < mask.size(1))
    hits = torch.zeros(mask.size(0), dtype=torch.bool, device=mask.device)
    if bool(pred_valid.any()):
        hits[pred_valid] = mask[rows[pred_valid], pred[pred_valid]]
    target_set_size = mask.sum(dim=1).to(dtype=torch.float32)
    duplicate_target = target_set_size > 1
    target_entropy_values: list[float] = []
    target_variance_values: list[float] = []
    if candidate_logits is not None and mask.numel() > 0:
        logits = candidate_logits.detach().to(device=mask.device, dtype=torch.float32)
        for row in range(mask.size(0)):
            row_mask = mask[row]
            if bool(row_mask.any()):
                row_logits = logits[row, row_mask]
                probs = torch.softmax(row_logits, dim=0)
                entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum()
                variance = row_logits.var(unbiased=False) if row_logits.numel() > 1 else row_logits.new_tensor(0.0)
                target_entropy_values.append(float(entropy.item()))
                target_variance_values.append(float(variance.item()))
    entropy_mean = float(sum(target_entropy_values) / len(target_entropy_values)) if target_entropy_values else 0.0
    variance_mean = float(sum(target_variance_values) / len(target_variance_values)) if target_variance_values else 0.0
    return {
        "target_in_candidate_set_rate": float(has_target.float().mean().item()) if mask.size(0) else 0.0,
        "prediction_in_candidate_set_rate": float(pred_valid.float().mean().item()) if mask.size(0) else 0.0,
        "candidate_hit_rate": float(hits.float().mean().item()) if mask.size(0) else 0.0,
        "candidate_missing_target_rate": float((~has_target).float().mean().item()) if mask.size(0) else 0.0,
        "candidate_target_set_size_mean": float(target_set_size.mean().item()) if mask.size(0) else 0.0,
        "candidate_target_set_size_max": float(target_set_size.max().item()) if mask.size(0) else 0.0,
        "candidate_duplicate_target_rate": float(duplicate_target.float().mean().item()) if mask.size(0) else 0.0,
        "candidate_target_logit_entropy_mean": entropy_mean,
        "candidate_target_logit_variance_mean": variance_mean,
    }
```

- [ ] **Step 4: Add candidate mode config parsing**

Modify `ModelTrainer.__init__`:

```python
self.candidate_contract_mode = str(config.get("candidate_contract_mode", "") or "").strip().lower()
if not self.candidate_contract_mode:
    self.candidate_contract_mode = "fixed_projection" if self.dynamic_candidate_contract_enabled else "fixed_label"
if self.candidate_contract_mode not in {"fixed_label", "fixed_projection", "candidate_id"}:
    raise ValueError("training.candidate_contract_mode must be fixed_label, fixed_projection, or candidate_id.")
self.candidate_batch_topology_policy = str(
    config.get("candidate_batch_topology_policy", "single_topology_required") or "single_topology_required"
).strip().lower()
if self.candidate_batch_topology_policy not in {"single_topology_required", "group_by_topology"}:
    raise ValueError("training.candidate_batch_topology_policy must be single_topology_required or group_by_topology.")
self.candidate_missing_target_fail_threshold = float(
    config.get("candidate_missing_target_fail_threshold", 0.10)
)
```

Add helper:

```python
def _use_true_candidate_contract(self) -> bool:
    return self.candidate_contract_mode == "candidate_id"
```

- [ ] **Step 5: Add trainer candidate loss path**

In `_run_epoch()`, when `self._use_true_candidate_contract()`:

1. call `self.model.forward_candidate(contract)`;
2. build `target_candidate_mask = output.map_targets_to_candidate_mask(targets)`;
3. compute candidate target summary with logits;
4. fail training if `candidate_missing_target_rate > candidate_missing_target_fail_threshold`;
5. compute `loss = candidate_set_cross_entropy(output.candidate_logits, target_candidate_mask)`;
6. predictions are `argmax(output.candidate_logits, dim=1)`;
7. for legacy fixed-label metrics, map predicted candidate to class:

```python
summary = candidate_target_summary(
    target_candidate_mask=target_candidate_mask,
    pred_candidate_index=pred_candidate,
    candidate_logits=output.candidate_logits,
)
missing_rate = float(summary["candidate_missing_target_rate"])
if training and missing_rate > self.candidate_missing_target_fail_threshold:
    raise ValueError(
        "candidate_id target mapping failed: "
        f"candidate_missing_target_rate={missing_rate:.4f} exceeds "
        f"threshold={self.candidate_missing_target_fail_threshold:.4f}."
    )
pred_class = output.candidate_class_index[pred_candidate]
```

Do not use `_candidate_output_to_fixed_logits()` in this mode.

- [ ] **Step 6: Add failing trainer test**

Add to `tests/application/test_evaluator_metrics.py`:

```python
def test_train_epoch_candidate_id_mode_uses_set_aware_candidate_loss():
    model = _DynamicCandidateModelWithDuplicateCandidates()
    trainer = ModelTrainer(
        xes_adapter=_DummyAdapter(),
        prefix_policy=_DummyPrefixPolicy(),
        graph_builder=_DummyGraphBuilder(),
        model=model,
        log_path="in_memory.xes",
        config={
            "mode": "train",
            "device": "cpu",
            "show_progress": False,
            "tqdm_disable": True,
            "candidate_contract_mode": "candidate_id",
        },
    )

    loss, macro_f1, _, _ = trainer._run_epoch(
        DataLoader([_sample_with_target(2)], batch_size=1),
        optimizer=torch.optim.Adam(model.parameters(), lr=0.01),
        training=True,
    )

    assert model.forward_candidate_called is True
    assert loss >= 0.0
    assert macro_f1 == pytest.approx(1.0)
```

Model helper returns `candidate_class_index=[1,2,2]` and highest candidate logit on the second `2`.

Add a guard test:

```python
def test_train_epoch_candidate_id_mode_fails_when_missing_target_rate_exceeds_threshold():
    model = _DynamicCandidateModelWithDuplicateCandidates(candidate_class_index=[1, 2, 2])
    trainer = _make_candidate_id_trainer(
        model=model,
        config={"candidate_missing_target_fail_threshold": 0.10},
    )

    with pytest.raises(ValueError, match="candidate_missing_target_rate"):
        trainer._run_epoch(
            DataLoader([_sample_with_target(9)], batch_size=1),
            optimizer=torch.optim.Adam(model.parameters(), lr=0.01),
            training=True,
        )
```

- [ ] **Step 7: Run tests**

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\application\test_candidate_target_mapping.py tests\application\test_evaluator_metrics.py -v
```

Expected: pass.

---

## 8. Task 5: Candidate-Level Evaluator Metrics

**Files:**

- Modify: `src/application/use_cases/trainer.py`
- Test: `tests/application/test_evaluator_metrics.py`
- Test: `tests/application/test_drift_one_pass_evaluation.py`

- [ ] **Step 1: Add evaluator test for candidate metrics**

Add to `tests/application/test_evaluator_metrics.py`:

```python
def test_evaluate_test_candidate_id_mode_reports_candidate_metrics():
    model = _DynamicCandidateModelWithDuplicateCandidates()
    trainer = ModelTrainer(
        xes_adapter=_DummyAdapter(),
        prefix_policy=_DummyPrefixPolicy(),
        graph_builder=_DummyGraphBuilder(),
        model=model,
        log_path="in_memory.xes",
        config={
            "mode": "eval_drift",
            "device": "cpu",
            "show_progress": False,
            "tqdm_disable": True,
            "candidate_contract_mode": "candidate_id",
        },
    )

    metrics = trainer._evaluate_test(DataLoader([_sample_with_target(2)], batch_size=1))

    assert metrics["candidate_hit_rate"] == pytest.approx(1.0)
    assert metrics["target_in_candidate_set_rate"] == pytest.approx(1.0)
    assert metrics["prediction_in_candidate_set_rate"] == pytest.approx(1.0)
    assert "candidate_nll" in metrics
    assert "candidate_ece" in metrics
```

- [ ] **Step 2: Run failing test**

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\application\test_evaluator_metrics.py::test_evaluate_test_candidate_id_mode_reports_candidate_metrics -v
```

Expected: fail because metrics are missing.

- [ ] **Step 3: Implement candidate evaluation branch**

In `_evaluate_test()`:

If `candidate_contract_mode == "candidate_id"`:

```python
output = self.model.forward_candidate(contract)
candidate_logits = output.candidate_logits
target_candidate_mask = output.map_targets_to_candidate_mask(targets)
candidate_probs = torch.softmax(candidate_logits, dim=1)
pred_candidate = torch.argmax(candidate_probs, dim=1)
pred_class = output.candidate_class_index[pred_candidate]
```

Accumulate:

```text
candidate_hit_rate
target_in_candidate_set_rate
prediction_in_candidate_set_rate
candidate_missing_target_rate
candidate_nll
candidate_ece
candidate_entropy
candidate_count_mean
candidate_count_min
candidate_count_max
candidate_target_set_size_mean
candidate_target_set_size_max
candidate_duplicate_target_rate
candidate_target_logit_entropy_mean
candidate_target_logit_variance_mean
```

For legacy comparison, still compute existing class-level metrics with `pred_class` and `targets`.

Do not compute candidate CE for rows where `target_candidate_mask` is empty;
count them in `candidate_missing_target_rate`. In training, fail when the
missing rate exceeds `training.candidate_missing_target_fail_threshold`.

- [ ] **Step 4: Add candidate ECE helper**

Reuse current ECE implementation if it accepts class probabilities and labels. For set targets, use candidate confidence and hit flag:

```python
confidence = candidate_probs.max(dim=1).values
hit = target_candidate_mask[rows, pred_candidate]
candidate_ece = _ece_from_confidence_and_correctness(confidence, hit.float())
```

If no such helper exists, add a private static helper in `ModelTrainer`:

```python
@staticmethod
def _expected_calibration_error_from_hits(confidence: torch.Tensor, hit: torch.Tensor, n_bins: int = 10) -> float:
    ...
```

- [ ] **Step 5: Drift one-pass records**

Modify `_collect_drift_inference_records()` so records include:

```python
candidate_pred_index
candidate_target_mask_any
candidate_hit
candidate_confidence
candidate_nll
candidate_entropy
candidate_count
candidate_target_set_size
candidate_target_logit_entropy
candidate_target_logit_variance
```

Add/extend `tests/application/test_drift_one_pass_evaluation.py` to assert these keys are present in candidate mode.

- [ ] **Step 6: Run tests**

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\application\test_evaluator_metrics.py tests\application\test_drift_one_pass_evaluation.py -v
```

Expected: pass.

---

## 9. Task 6: Candidate-Level Mask/OOS Semantics

**Files:**

- Modify: `src/application/use_cases/trainer.py`
- Modify: `src/application/services/candidate_target_mapping.py`
- Test: `tests/application/test_candidate_target_mapping.py`
- Test: `tests/application/test_evaluator_metrics.py`

- [ ] **Step 1: Define candidate allowed mask conversion**

Add helper:

```python
def class_allowed_mask_to_candidate_allowed_mask(
    allowed_target_mask: torch.Tensor,
    candidate_class_index: torch.Tensor,
) -> torch.BoolTensor:
    mask = allowed_target_mask.to(dtype=torch.bool, device=candidate_class_index.device)
    class_idx = candidate_class_index.to(dtype=torch.long, device=mask.device)
    if mask.dim() != 2:
        raise ValueError("allowed_target_mask must have shape [B, C_train].")
    valid = (class_idx >= 0) & (class_idx < mask.size(1))
    result = torch.zeros((mask.size(0), class_idx.numel()), dtype=torch.bool, device=mask.device)
    if bool(valid.any()):
        result[:, valid] = mask[:, class_idx[valid]]
    return result
```

- [ ] **Step 2: Add tests**

```python
def test_class_allowed_mask_to_candidate_allowed_mask_handles_duplicate_class_candidates():
    allowed = torch.tensor([[False, True, False]], dtype=torch.bool)
    candidate_class_index = torch.tensor([1, 1, 2], dtype=torch.long)

    result = class_allowed_mask_to_candidate_allowed_mask(allowed, candidate_class_index)

    assert result.tolist() == [[True, True, False]]
```

- [ ] **Step 3: Apply in candidate evaluator**

When `allowed_target_mask` exists:

```python
candidate_allowed_mask = class_allowed_mask_to_candidate_allowed_mask(
    allowed_target_mask,
    output.candidate_class_index,
)
candidate_oos = ~candidate_allowed_mask[rows, pred_candidate]
candidate_target_in_mask = (candidate_allowed_mask & target_candidate_mask).any(dim=1)
candidate_strict_error_but_allowed = (~candidate_hit) & (~candidate_oos)
```

Log:

```text
candidate_oos_rate
candidate_target_in_mask_rate
candidate_strict_error_but_allowed_rate
candidate_allowed_cardinality_mean
```

- [ ] **Step 4: Keep mask-guided inference separate**

Do not apply current fixed-label `mask_guided_enabled` directly to `candidate_logits` until candidate-level hard/soft mask semantics are explicitly tested. For this stage:

```text
candidate_contract_mode=candidate_id + mask_guided_enabled=true
=> raise ValueError or warn and disable mask-guided logits
```

Recommended first implementation: raise with a clear message:

```python
if self._use_true_candidate_contract() and self.mask_guided_enabled:
    raise ValueError("candidate_contract_mode=candidate_id does not yet support mask_guided_enabled.")
```

This prevents silently mixing fixed-label mask logic with candidate logits.

- [ ] **Step 5: Run tests**

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\application\test_candidate_target_mapping.py tests\application\test_evaluator_metrics.py -v
```

Expected: pass.

---

## 10. Task 7: Traces And MLflow Candidate Attributes

**Files:**

- Modify: `src/application/services/structural_trace_payload_builder.py`
- Modify: `src/application/use_cases/trainer.py`
- Test: `tests/application/test_structural_trace_payload_builder.py`
- Test: `tests/application/test_trainer_selective_tracing.py`

- [ ] **Step 1: Add trace payload test**

Add to `tests/application/test_structural_trace_payload_builder.py`:

```python
def test_trace_payload_includes_true_candidate_fields():
    payload = build_structural_prediction_trace_payload(
        model=_CandidateTraceModel(),
        contract={
            "y": torch.tensor([2]),
            "candidate_ids": ["task_a", "task_b_left", "task_b_right"],
            "target_candidate_mask": torch.tensor([[False, True, True]], dtype=torch.bool),
        },
        logits=torch.tensor([[0.1, 0.2, 0.9]]),
        effective_logits=torch.tensor([[0.1, 0.2, 0.9]]),
        probs=torch.softmax(torch.tensor([[0.1, 0.2, 0.9]]), dim=1),
        row=0,
        stage="test",
        reason="strict_error",
        reverse_vocab={2: "Approve"},
    )

    assert payload["candidate"]["predicted_candidate_id"] == "task_b_right"
    assert payload["candidate"]["target_candidate_ids"] == ["task_b_left", "task_b_right"]
```

- [ ] **Step 2: Implement payload fields**

Add top-level `candidate` section:

```python
"candidate": {
    "candidate_contract_mode": "...",
    "predicted_candidate_index": int(...),
    "predicted_candidate_id": "...",
    "target_candidate_ids": [...],
    "target_in_candidate_set": bool(...),
    "candidate_hit": bool(...),
    "candidate_count": int(...),
}
```

Detach tensors before converting:

```python
tensor.detach().cpu().tolist()
```

- [ ] **Step 3: Add flat attributes**

Extend `build_structural_prediction_trace_attributes()`:

```python
candidate = payload.get("candidate", {})
attrs["candidate_contract_mode"] = str(candidate.get("candidate_contract_mode", ""))
attrs["predicted_candidate_id"] = str(candidate.get("predicted_candidate_id", ""))
attrs["target_in_candidate_set"] = _optional_bool(candidate.get("target_in_candidate_set"))
attrs["candidate_hit"] = _optional_bool(candidate.get("candidate_hit"))
attrs["candidate_count"] = _safe_attr_float(candidate.get("candidate_count"))
```

- [ ] **Step 4: Run tests**

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\application\test_structural_trace_payload_builder.py tests\application\test_trainer_selective_tracing.py -v
```

Expected: pass.

---

## 11. Task 8: Config, UI Catalog, And Safety Guards

**Files:**

- Modify: `configs/ui/config_catalog.yaml`
- Modify: `src/application/use_cases/trainer.py`
- Test: `tests/application/test_experiment_ui_catalog_fields.py`
- Test: `tests/application/test_learning_strategy_config.py`

- [ ] **Step 1: Add config catalog test**

Add to `tests/application/test_experiment_ui_catalog_fields.py`:

```python
def test_catalog_includes_candidate_contract_mode():
    catalog = yaml.safe_load(Path("configs/ui/config_catalog.yaml").read_text(encoding="utf-8"))["fields"]
    field = catalog["training.candidate_contract_mode"]

    assert field["enum"] == ["fixed_label", "fixed_projection", "candidate_id"]
    assert field["required_when"]["model.type"] == "EOPKGTopologyConditioned"


def test_catalog_includes_candidate_id_safety_fields():
    fields = yaml.safe_load(Path("configs/ui/config_catalog.yaml").read_text(encoding="utf-8"))["fields"]

    assert fields["training.candidate_batch_topology_policy"]["enum"] == [
        "single_topology_required",
        "group_by_topology",
    ]
    assert "training.candidate_missing_target_fail_threshold" in fields
```

- [ ] **Step 2: Add catalog entry**

```yaml
training.candidate_contract_mode:
  path: training.candidate_contract_mode
  section: training
  label: 'Training: Candidate contract mode'
  description: Prediction/loss contract for EOPKGTopologyConditioned. fixed_label uses forward(); fixed_projection uses forward_candidate() projected to C_train; candidate_id uses true topology candidate logits without fixed-vocab projection.
  affects: Controls whether topology candidates are evaluated as fixed labels or true BPMN candidate ids.
  default: fixed_label
  enum:
  - fixed_label
  - fixed_projection
  - candidate_id
  required_in_modes: []
  required_when:
    model.type: EOPKGTopologyConditioned
  ui:
    tab: general
    group: advanced
    priority: 100
    order: 1015

training.candidate_batch_topology_policy:
  path: training.candidate_batch_topology_policy
  section: training
  label: 'Training: Candidate batch topology policy'
  description: Batch topology policy for candidate_id mode. single_topology_required fails mixed topology batches; group_by_topology is a future extension.
  affects: Prevents silent candidate-axis corruption when a batch contains multiple process versions/topology snapshots.
  default: single_topology_required
  enum:
  - single_topology_required
  - group_by_topology
  required_when:
    training.candidate_contract_mode: candidate_id
  ui:
    tab: advanced
    group: training
    priority: 100
    order: 1016

training.candidate_missing_target_fail_threshold:
  path: training.candidate_missing_target_fail_threshold
  section: training
  label: 'Training: Candidate missing target fail threshold'
  description: Maximum allowed missing-target rate in candidate_id training before the run fails as a log/topology mapping error.
  affects: Prevents silent zero-gradient training when targets are absent from the topology candidate set.
  default: 0.10
  type: float
  required_when:
    training.candidate_contract_mode: candidate_id
  ui:
    tab: advanced
    group: training
    priority: 100
    order: 1017
```

- [ ] **Step 3: Safety guard**

In `ModelTrainer.__init__`, if `candidate_contract_mode == "candidate_id"`:

```python
if not hasattr(self.model, "forward_candidate"):
    raise ValueError("training.candidate_contract_mode=candidate_id requires a model with forward_candidate().")
if self.mask_guided_enabled:
    raise ValueError("candidate_contract_mode=candidate_id does not yet support mask_guided_enabled.")
if self.candidate_batch_topology_policy == "group_by_topology":
    raise ValueError("candidate_batch_topology_policy=group_by_topology is planned but not implemented.")
if not (0.0 <= self.candidate_missing_target_fail_threshold <= 1.0):
    raise ValueError("candidate_missing_target_fail_threshold must be between 0.0 and 1.0.")
```

- [ ] **Step 4: Run tests**

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\application\test_experiment_ui_catalog_fields.py tests\application\test_learning_strategy_config.py -v
```

Expected: pass.

---

## 12. Task 9: Documentation Updates

**Files:**

- Modify: `docs/GNN_RUNTIME_MVP2_5.MD`
- Modify: `docs/GNN_LEARNING_METODOLOGY.MD`
- Modify: `docs/current/project-state.md`
- Modify: `docs/current/architecture-debt.md`
- Modify: `docs/dissertation_changes.MD`
- Add: `docs/worklogs/MVP2_5_EOPKGTopologyConditioned_Stage2_True_Candidate_Contract_Report_2026-05-25.MD`

- [ ] **Step 1: Runtime docs**

Add:

```text
training.candidate_contract_mode:
  fixed_label       -> legacy forward() [B,C_train]
  fixed_projection  -> forward_candidate() then sparse projection [B,C_train]
  candidate_id      -> forward_candidate() [B,C_v] with candidate ids and set-aware target mapping
```

Document tensor contract:

```text
candidate_logits [B,C_v]
candidate_ids tuple[str, ...]
candidate_class_index [C_v]
candidate_node_index [C_v]
target_candidate_mask [B,C_v]
topology_identity str
candidate batch topology policy
candidate missing target fail threshold
```

- [ ] **Step 2: Methodology docs**

Update `docs/GNN_LEARNING_METODOLOGY.MD`:

```text
Phase 2 true candidate-id contract: implemented when candidate_contract_mode=candidate_id.
Remaining limitation: if logs do not contain exact BPMN node ids, duplicate labels use MIL target sets.
Blocking guard: candidate-id training requires homogeneous topology batches
until group_by_topology execution is implemented.
```

- [ ] **Step 3: Debt docs**

Update `docs/current/architecture-debt.md`:

```text
dynamic_candidate_prediction_contract:
  current_behavior: candidate_id mode implemented
  remaining_debt: exact BPMN node target supervision, semantic prototypes, and group_by_topology mixed-batch execution
```

- [ ] **Step 4: Dissertation notes**

Add concise note:

```text
Stage 2 true changes evaluation target from fixed activity class to topology candidate set.
For duplicate labels, current logs provide set supervision, not exact BPMN node supervision.
MIL duplicate-set diagnostics are required because logsumexp can hide duplicate-node collapse.
This distinction is important for interpreting strict-F1 under duplicate activity identity ambiguity.
```

- [ ] **Step 5: Worklog report**

Create report with:

```text
what changed
how to enable
what metrics to inspect
known limitations
test commands and results
```

---

## 13. Task 10: Verification

**Files:** no code changes.

- [ ] **Step 1: Run targeted tests**

```powershell
.\.venv-modern\Scripts\python.exe -m pytest `
  tests\domain\test_candidate_prediction_contract.py `
  tests\domain\test_eopkg_topology_conditioned.py `
  tests\application\test_candidate_target_mapping.py `
  tests\application\test_evaluator_metrics.py `
  tests\application\test_trainer_candidate_batch_topology.py `
  -v
```

Expected: pass.

- [ ] **Step 2: Run drift/evaluator tests**

```powershell
.\.venv-modern\Scripts\python.exe -m pytest `
  tests\application\test_drift_one_pass_evaluation.py `
  tests\application\test_trainer_selective_tracing.py `
  tests\application\test_structural_trace_payload_builder.py `
  -v
```

Expected: pass.

- [ ] **Step 3: Run config/UI tests**

```powershell
.\.venv-modern\Scripts\python.exe -m pytest `
  tests\application\test_experiment_ui_catalog_fields.py `
  tests\application\test_cli_flatten_config.py `
  -v
```

Expected: pass.

- [ ] **Step 4: Run architecture guard**

```powershell
.\.venv-modern\Scripts\python.exe tools\architecture_guard.py
```

Expected:

```text
[ARCH_GUARD] OK: architecture dependency rules satisfied.
```

- [ ] **Step 5: Run full suite**

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\ -v
```

Expected: all pass. Existing external dependency deprecation warnings are acceptable if no new warning category is introduced.

---

## 14. First Experiment After Implementation

Create one preset:

```yaml
preset_name: _EOPKGTCid-UN
project: mvp2_6-test
experiment:
  mode: train
  structural_mode: true
  mask_guided_enabled: false
  statistic_enabled: false
  train_ratio: 0.5
  fraction: 0.3
  fraction_strategy: versioned
  split_strategy: versioned
  version_scope_policy: train_cut
model:
  type: EOPKGTopologyConditioned
  observed_encoder: GATv2
  struct_encoder: GATv2
  candidate_scoring: cosine
  candidate_pooling: logmeanexp
  pooling_strategy: last_node
training:
  learning_strategy: standard
  candidate_contract_mode: candidate_id
  candidate_batch_topology_policy: single_topology_required
  candidate_missing_target_fail_threshold: 0.10
  dynamic_candidate_contract_enabled: false
```

Then drift preset:

```yaml
preset_name: _EOPKGTCid-UN-drift
experiment:
  mode: eval_drift
  load_checkpoint: checkpoints/<run-name>/best_model.pth
training:
  candidate_contract_mode: candidate_id
```

Inspect:

```text
candidate_hit_rate
target_in_candidate_set_rate
prediction_in_candidate_set_rate
candidate_oos_rate
candidate_ece
candidate_nll
candidate_count_mean/min/max
candidate_target_set_size_mean/max
candidate_duplicate_target_rate
candidate_target_logit_entropy_mean
candidate_target_logit_variance_mean
strict_macro_f1
macro_f1
test_oos
ece
nll
```

Expected first interpretation:

- if `target_in_candidate_set_rate < 1`, topology/log target alignment is still incomplete;
- if `candidate_missing_target_rate > 0.10` during training, the run is invalid and should fail rather than continue;
- if duplicate target sets exist and `candidate_target_logit_variance_mean` stays near `0`, the model is not separating duplicate BPMN candidates;
- if `candidate_hit_rate` improves but strict-F1 does not, duplicate-label target ambiguity is active;
- if candidate metrics equal Stage 1 again, candidate ids are still effectively class ids and structural node identity has not been propagated correctly;
- if OOS/ECE improve but strict-F1 does not, topology helps reliability but not exact ranking.

---

## 15. Plan Self-Review

- The plan directly covers the requested five blocks:
  1. Candidate Identity Contract: Tasks 1-3.
  2. Target Mapping: Task 4.
  3. Candidate-Level Loss: Task 4.
  4. Candidate-Level Evaluator: Tasks 5-6.
  5. Drift/Trace/MLflow: Task 7.
- The plan does not remove old behavior; it adds an explicit mode.
- The plan avoids semantic prototypes until candidate-id evaluation is stable.
- The plan handles duplicate activity labels with set-aware MIL CE because current logs do not reliably provide exact BPMN node targets.
- The plan adds MIL diagnostics because set-aware `logsumexp` can otherwise hide duplicate-node collapse.
- The plan treats missing target rows as mapping/alignment failures in training, with an explicit fail threshold.
- The plan explicitly blocks `mask_guided_enabled` in candidate-id mode until candidate-level mask-guided logits are separately designed and tested.
- The plan no longer permits mixed-topology candidate-id batches to use the first structural payload. Stage 2 true starts with `single_topology_required`; `group_by_topology` remains explicit future debt.
