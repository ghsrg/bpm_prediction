# MVP2.5 Topology-Native Candidate Identity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` or `superpowers:subagent-driven-development` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** replace the current fixed-vocabulary `candidate_id` bridge with a topology-native candidate contract so `EOPKGTopologyConditioned` can evaluate candidates from the current BPMN topology, including future-version nodes that are absent from the training activity vocabulary.

**Architecture:** separate the observed/log activity vocabulary from the topology candidate universe. The observed encoder may keep the fixed train vocabulary for prefix features, but candidate scoring, target mapping, candidate masks, OOS, and calibration must operate on topology-local candidate IDs. Fixed-label metrics remain compatibility diagnostics, not the primary evidence for zero-shot topology transfer.

**Tech Stack:** Python, PyTorch, PyTorch Geometric, project `GraphTensorContract`, `CandidatePredictionOutput`, `DynamicGraphBuilder`, `EOPKGTopologyConditioned`, MLflow metrics/traces, pytest.

---

## Why This Supersedes the P0/P1 Flow Penalty Plan

The implemented P0/P1 flow penalty plan added useful diagnostics, but the latest audit shows the deeper blocker:

```text
version_scope_policy=train_cut
  -> activity_vocab contains only train versions
  -> unseen future targets map to <UNK>
  -> topology edges with unseen labels are skipped
  -> candidate_dynamic_count remains equal to train vocab size
  -> EOPKGTopologyConditioned scores fixed labels, not real future topology candidates
```

Therefore, increasing topology-flow penalty weight cannot prove the main hypothesis. It only regularizes a candidate space that is still constrained by the training vocabulary.

Current evidence:

- `candidate_dynamic_count = 37` even in drift versions;
- `candidate_class_index = [0..36]`;
- future-version candidates are not represented as independent candidates;
- `struct_node_emb = nn.Embedding(output_dim, hidden_dim)` encourages class-identity shortcut learning;
- `wrong_version` and `drop_edges` negative losses are skipped for `candidate_id`.

This plan keeps the useful P0/P1 diagnostics, but changes the next implementation target.

---

## Target Contract

### Current Contract

```text
candidate_logits:      [B, C_v]
candidate_class_index: [C_v] -> global train activity vocab index
target y:              global train activity vocab index
```

This fails when the topology has candidates that are not in the train activity vocab.

### Required Contract

```text
candidate_logits:        [B, K_v]
candidate_ids:           K_v stable topology candidate IDs
candidate_labels:        K_v human-readable labels
candidate_class_index:   K_v optional compatibility mapping to train vocab, -1 for unseen labels
target_candidate_mask:   [B, K_v] true when candidate matches target label/node
candidate_allowed_mask:  [B, K_v] true when candidate is reachable from current prefix state
```

Primary loss:

```text
L_candidate = -logsumexp(candidate_logits[target_candidate_mask])
              + logsumexp(candidate_logits[all candidates])
```

Primary metrics:

```text
candidate_strict_accuracy
candidate_strict_macro_f1_by_label
candidate_oos_rate
candidate_invalid_probability_mass
candidate_missing_target_rate
candidate_unseen_target_rate
candidate_unseen_prediction_rate
candidate_ece
candidate_nll
```

Compatibility metrics:

```text
fixed_label_strict_macro_f1
fixed_label_oos
fixed_label_ece
```

Compatibility metrics must be marked invalid/partial when target or prediction maps to `candidate_class_index=-1`.

---

## Scope for This Implementation

### In Scope

1. Preserve all topology prediction nodes as candidates, even when absent from `activity_vocab`.
2. Preserve projected topology edges between candidate nodes, even when labels are unseen in the train vocab.
3. Add topology-local candidate metadata to the graph contract.
4. Map targets to candidates by topology candidate label for XES logs.
5. Train `EOPKGTopologyConditioned` with candidate-local target masks, not global class targets.
6. Add candidate-local metrics/traces and mark fixed-label metrics as compatibility only.
7. Keep `gateway_mode` semantics unchanged:
   - `preserve`: Camunda-like logs include technical/service nodes;
   - `collapse_for_prediction`: XES task-label logs use prediction-node projection.
8. Keep topology-homogeneous batching as required.
9. Attach `target_label` to graph contracts so XES labels can train unseen topology candidates without relying on `<UNK>` target indexes.
10. Preserve candidate metadata through PyG batching/collation without assuming every field is a tensor.

### Out of Scope

1. Semantic text embeddings / Sentence-BERT prototypes.
2. True BPMN node-id target supervision from logs.
3. Multi-instance duplicate disambiguation beyond label-set CE.
4. Full wrong-version negative loss across different candidate axes.
5. Full `versioned_zero_shot` experiment mode.

These are next methodology stages after this contract is correct.

---

## External Audit Reconciliation

The external audit plan is mostly aligned with this plan. Required additions are:

- `candidate_target_mapping.py` already exists, so this is a modification, not a new file.
- The graph contract must explicitly carry `target_label` because `y=0` can mean `<UNK>` after `train_cut`; candidate-native training must not infer unseen targets from `y`.
- Candidate IDs/labels are not tensors, so batching must define how they are stored and retrieved for topology-homogeneous batches.
- Tests must cover both gateway modes:
  - `gateway_mode=collapse_for_prediction` for XES task-label logs;
  - `gateway_mode=preserve` for Camunda-like logs where technical nodes may appear in logs.
- Manual verification must check that candidate counts and unseen rates actually change across versions.

No additional open question blocks implementation. The accepted assumption remains:

```text
XES targets are activity labels, not unique BPMN node IDs.
If multiple topology candidates share the same label, they form one target set.
```

---

## File Map

### Modify

- `src/domain/entities/candidate_prediction.py`
  - extend `CandidatePredictionOutput` with topology-native candidate metadata and target-mask helpers.

- `src/domain/services/dynamic_graph_builder.py`
  - build topology-native candidates independent of `activity_vocab`;
  - keep all candidate nodes/edges after projection;
  - attach candidate IDs, labels, class compatibility indexes, and allowed masks in candidate space.

- `src/domain/services/topology_projection_alignment.py`
  - if needed, expose projection output suitable for topology-native candidate lists without treating missing train vocab as skipped topology.

- `src/domain/models/topology_conditioned.py`
  - stop using `struct_node_emb(output_dim)` as the primary candidate identity path;
  - score candidates by topology node axis;
  - return candidate metadata in `CandidatePredictionOutput`.

- `src/application/services/candidate_target_mapping.py`
  - add candidate-local target-mask and prediction mapping helpers.

- `src/application/services/topology_conditioned_learning.py`
  - add candidate-local CE, OOS, calibration, and allowed-flow helpers.

- `src/application/use_cases/trainer.py`
  - train/evaluate `candidate_contract_mode=candidate_id` using candidate-local masks;
  - log candidate-native metrics;
  - keep fixed-label projection only as compatibility diagnostics.

- `src/application/services/structural_trace_payload_builder.py`
  - include candidate IDs, labels, unseen flags, target-mask status, and candidate-local top-k.

- `configs/ui/config_catalog.yaml`
  - expose any new `training.*` or `model.*` keys.

- `docs/GNN_RUNTIME_MVP2_5.MD`
  - document topology-native candidate contract.

- `docs/GNN_LEARNING_STRATEGY.MD`
  - document why `topology_conditioned` requires topology-native candidates.

- `docs/current/architecture-debt.md`
  - keep semantic grounding, node-id supervision, and cross-axis negative sampling as explicit future debt.

- `docs/dissertation_changes.MD`
  - add the methodology conclusion about train-vocab collapse and topology-native candidate identity.

### Tests

- `tests/domain/test_dynamic_graph_builder_topology_native_candidates.py`
- `tests/domain/test_eopkg_topology_conditioned.py`
- `tests/application/test_candidate_target_mapping.py`
- `tests/application/test_topology_conditioned_learning.py`
- `tests/application/test_trainer_topology_conditioned_candidate_id.py`
- `tests/application/test_evaluator_metrics.py`
- `tests/application/test_trainer_selective_tracing.py`

---

## New / Updated Config Keys

No new key is required for the default behavior if `candidate_contract_mode=candidate_id` is intended to become topology-native.

Add one explicit transition key only if backward compatibility is needed:

```yaml
training.candidate_identity_mode: topology_native
```

Allowed values:

```text
fixed_vocab_bridge
topology_native
```

Recommended default:

```yaml
training.candidate_identity_mode: topology_native
```

Guardrail:

```text
candidate_contract_mode=candidate_id + candidate_identity_mode=fixed_vocab_bridge
```

should log a warning:

```text
fixed_vocab_bridge is compatibility-only and cannot validate zero-shot topology transfer with unseen future candidates.
```

If this key is added, it must be added to `configs/ui/config_catalog.yaml`.

---

## Task 1: Lock the Current Failure with Tests

**Files:**

- Create: `tests/domain/test_dynamic_graph_builder_topology_native_candidates.py`
- Modify: `tests/domain/test_eopkg_topology_conditioned.py`

- [ ] **Step 1: Write failing topology preservation test**

Create a minimal topology with train vocab containing only `A`, `B`, and future topology containing `A -> C -> B`.

Expected behavior after this plan:

```python
def test_topology_native_candidates_keep_unseen_future_node_and_edges():
    contract = build_contract_with_activity_vocab(
        activity_vocab={"<UNK>": 0, "A": 1, "B": 2},
        topology_edges=[("A", "C"), ("C", "B")],
        topology_nodes=[
            {"id": "node_a", "label": "A", "type": "task"},
            {"id": "node_c", "label": "C", "type": "task"},
            {"id": "node_b", "label": "B", "type": "task"},
        ],
        gateway_mode="collapse_for_prediction",
    )

    assert contract["candidate_ids"] == ["node_a", "node_c", "node_b"]
    assert contract["candidate_labels"] == ["A", "C", "B"]
    assert contract["candidate_class_index"].tolist() == [1, -1, 2]
    assert contract["struct_node_to_candidate_index"].tolist() == [0, 1, 2]
    assert contract["structural_edge_index"].shape[1] == 2
```

Run:

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\domain\test_dynamic_graph_builder_topology_native_candidates.py -v
```

Expected now: fail because unseen node `C` is skipped or collapsed.

- [ ] **Step 2: Write failing model candidate count test**

```python
def test_topology_conditioned_forward_candidate_uses_topology_candidate_axis_with_unseen_label():
    model = build_small_eopkg_topology_conditioned(output_dim=3)
    contract = build_graph_contract(
        candidate_ids=["node_a", "node_c", "node_b"],
        candidate_labels=["A", "C", "B"],
        candidate_class_index=torch.tensor([1, -1, 2]),
        struct_node_to_candidate_index=torch.tensor([0, 1, 2]),
        struct_node_to_class_index=torch.tensor([1, -1, 2]),
    )

    output = model.forward_candidate(contract)

    assert output.candidate_logits.shape[1] == 3
    assert output.candidate_class_index.tolist() == [1, -1, 2]
    assert output.candidate_ids == ("node_a", "node_c", "node_b")
```

Expected now: fail because `CandidatePredictionOutput` has no candidate IDs and the model pools only known class indexes.

---

## Task 2: Extend CandidatePredictionOutput Contract

**Files:**

- Modify: `src/domain/entities/candidate_prediction.py`
- Test: `tests/application/test_candidate_target_mapping.py`

- [ ] **Step 1: Add metadata fields**

Extend the dataclass:

```python
@dataclass(frozen=True)
class CandidatePredictionOutput:
    candidate_logits: torch.Tensor
    candidate_class_index: torch.LongTensor
    node_logits: torch.Tensor
    node_to_candidate_index: torch.LongTensor
    node_to_class_index: torch.LongTensor
    fixed_label_logits: torch.Tensor | None = None
    candidate_ids: tuple[str, ...] = ()
    candidate_labels: tuple[str, ...] = ()
    candidate_is_unseen: torch.BoolTensor | None = None
```

- [ ] **Step 2: Keep backward-compatible helpers**

Keep current `map_targets_to_candidate_mask()` for fixed-vocab compatibility.

Add label-based target mapping:

```python
def map_target_labels_to_candidate_mask(self, target_labels: list[str]) -> torch.BoolTensor:
    labels = [str(item).strip() for item in self.candidate_labels]
    rows = []
    for raw in target_labels:
        token = str(raw).strip()
        rows.append([label == token for label in labels])
    return torch.tensor(rows, dtype=torch.bool, device=self.candidate_logits.device)
```

This is the XES-compatible mapping until logs contain exact BPMN node IDs.

- [ ] **Step 3: Add tests**

```python
def test_candidate_output_maps_unseen_label_by_candidate_label_not_vocab_index():
    output = CandidatePredictionOutput(
        candidate_logits=torch.zeros(2, 3),
        candidate_class_index=torch.tensor([1, -1, 2]),
        node_logits=torch.zeros(2, 3),
        node_to_candidate_index=torch.tensor([0, 1, 2]),
        node_to_class_index=torch.tensor([1, -1, 2]),
        candidate_ids=("node_a", "node_c", "node_b"),
        candidate_labels=("A", "C", "B"),
    )

    mask = output.map_target_labels_to_candidate_mask(["C", "B"])

    assert mask.tolist() == [[False, True, False], [False, False, True]]
```

Run:

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\application\test_candidate_target_mapping.py tests\domain\test_eopkg_topology_conditioned.py -v
```

---

## Task 3: Build Topology-Native Candidate Payloads

**Files:**

- Modify: `src/domain/services/dynamic_graph_builder.py`
- Modify: `src/domain/services/baseline_graph_builder.py` if baseline contracts do not already expose `target_label`
- Test: `tests/domain/test_dynamic_graph_builder_topology_native_candidates.py`

- [ ] **Step 1: Add candidate extraction helper**

Add a helper near `_resolve_compiled_topology`:

```python
def _topology_prediction_candidates(
    self,
    *,
    dto: ProcessStructureDTO,
    activity_vocab: Dict[str, int],
) -> dict[str, Any]:
    projection = TopologyProjectionCompiler(gateway_mode=self.topology_gateway_mode).project(
        dto=dto,
        activity_vocab={},
    )
    # If compiler still requires vocab, use node-id projection only for candidate discovery,
    # but do not filter final candidates by train activity_vocab.
```

Implementation rule:

- candidate ID = BPMN node ID after gateway projection;
- candidate label = normalized prediction label / task name used by XES target;
- `candidate_class_index = activity_vocab.get(candidate_label, -1)`;
- `candidate_is_unseen = candidate_class_index < 0`;
- candidate order must be stable and deterministic.

- [ ] **Step 2: Keep `gateway_mode` semantics**

For `collapse_for_prediction`, candidate nodes are only prediction/task nodes after transparent gateway projection.

For `preserve`, candidate nodes include the nodes that are expected to appear in Camunda-like logs.

Do not hardcode XES vs Camunda in the model. This remains a config/data alignment responsibility.

- [ ] **Step 3: Preserve edges independent of train vocab**

Replace this failure mode:

```python
src_idx = activity_vocab.get(src_token)
dst_idx = activity_vocab.get(dst_token)
if src_idx is None or dst_idx is None:
    continue
```

with candidate-local indexes:

```python
src_candidate_idx = candidate_id_to_index.get(src_candidate_id_or_label)
dst_candidate_idx = candidate_id_to_index.get(dst_candidate_id_or_label)
if src_candidate_idx is None or dst_candidate_idx is None:
    record_projection_warning_but_do_not_use_activity_vocab_as_filter()
    continue
structural_src.append(src_candidate_idx)
structural_dst.append(dst_candidate_idx)
```

- [ ] **Step 4: Attach contract fields**

Compiled topology must include:

```python
compiled = {
    "candidate_ids": tuple(candidate_ids),
    "candidate_labels": tuple(candidate_labels),
    "candidate_class_index": torch.tensor(candidate_class_index, dtype=torch.long),
    "candidate_is_unseen": torch.tensor(candidate_is_unseen, dtype=torch.bool),
    "struct_node_to_candidate_index": torch.tensor(struct_node_to_candidate_index, dtype=torch.long),
    "struct_node_to_class_index": torch.tensor(candidate_class_index, dtype=torch.long),
    ...
}
```

`struct_node_to_class_index` may stay for compatibility, but the model must prefer `struct_node_to_candidate_index`.

- [ ] **Step 5: Attach target label to every graph contract**

Every graph used by train/eval/drift must carry the raw next-activity label:

```python
contract["target_label"] = str(target_token).strip()
```

Reason:

```text
y == 0 can mean <UNK> when the target is absent from train activity_vocab.
candidate-native training must map the raw label to topology candidates.
```

For PyG `Data`, if direct string attributes are not collated safely in existing loaders, store an integer row pointer plus a sidecar metadata list in the dataset/shard metadata. The implementation must include a test proving that a batch with `y=<UNK>` still exposes `target_label="C"` to the trainer.

- [ ] **Step 6: Preserve candidate metadata through batching**

Because `candidate_ids` and `candidate_labels` are strings, topology-homogeneous batching must guarantee one candidate metadata set per batch.

Required behavior:

```text
if batch topology identity is homogeneous:
    candidate_ids/candidate_labels can be taken from the shared topology payload
else:
    group_by_topology must split the batch before model forward
```

Add a guard:

```python
if candidate_contract_mode == "candidate_id" and candidate_identity_mode == "topology_native":
    assert_homogeneous_topology_batch(data)
```

Do not attempt to concatenate candidate metadata from mixed topologies.

- [ ] **Step 7: Cache key update**

The topology cache key must include:

```text
candidate_identity_schema_version
topology_gateway_mode
process_version
stats_snapshot_identity
activity_vocab_fingerprint
```

because compatibility mapping to `candidate_class_index` still depends on `activity_vocab`.

- [ ] **Step 8: Gateway-mode tests**

Add tests that prove candidate extraction follows the configured topology/log alignment:

```python
def test_topology_native_candidates_collapse_for_prediction_uses_xes_task_labels():
    ...

def test_topology_native_candidates_preserve_keeps_camunda_visible_nodes():
    ...
```

The model must not decide which mode is correct. Incorrect XES/preserve or Camunda/collapse combinations remain invalid config/data alignment.

- [ ] **Step 9: Run tests**

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\domain\test_dynamic_graph_builder_topology_native_candidates.py tests\domain\test_dynamic_graph_builder_masks.py -v
```

Expected: new topology-native tests pass; existing mask tests still pass or are updated to assert both fixed and candidate masks where relevant.

---

## Task 4: Candidate-Local Target and Allowed Masks

**Files:**

- Modify: `src/application/services/candidate_target_mapping.py`
- Modify: `src/application/services/topology_conditioned_learning.py`
- Test: `tests/application/test_candidate_target_mapping.py`
- Test: `tests/application/test_topology_conditioned_learning.py`

- [ ] **Step 1: Add candidate-local target helper**

```python
def candidate_target_mask_from_labels(
    *,
    target_labels: list[str],
    candidate_labels: tuple[str, ...],
    device: torch.device,
) -> torch.BoolTensor:
    rows = []
    normalized_candidates = [str(label).strip() for label in candidate_labels]
    for raw in target_labels:
        target = str(raw).strip()
        rows.append([candidate == target for candidate in normalized_candidates])
    return torch.tensor(rows, dtype=torch.bool, device=device)
```

- [ ] **Step 2: Add candidate set CE**

```python
def candidate_set_cross_entropy_from_mask(
    candidate_logits: torch.Tensor,
    target_candidate_mask: torch.Tensor,
    *,
    missing_target_penalty: float = 0.0,
) -> torch.Tensor:
    if candidate_logits.shape != target_candidate_mask.shape:
        raise ValueError("target_candidate_mask must match candidate_logits shape.")
    has_target = target_candidate_mask.any(dim=1)
    if not bool(has_target.any()):
        return candidate_logits.sum() * 0.0 + float(missing_target_penalty)
    logits = candidate_logits[has_target]
    mask = target_candidate_mask[has_target]
    numerator = torch.logsumexp(logits.masked_fill(~mask, float("-inf")), dim=1)
    denominator = torch.logsumexp(logits, dim=1)
    loss = -(numerator - denominator)
    if bool((~has_target).any()) and missing_target_penalty > 0:
        penalty = logits.new_full((int((~has_target).sum().item()),), float(missing_target_penalty))
        loss = torch.cat([loss, penalty])
    return loss.mean()
```

- [ ] **Step 3: Add missing-target guard**

Training must log:

```text
candidate_missing_target_rate
candidate_unseen_target_rate
```

and fail when:

```yaml
training.candidate_missing_target_fail_threshold: 0.10
```

is exceeded.

If this config key is added, add it to `configs/ui/config_catalog.yaml`.

- [ ] **Step 4: Candidate allowed mask**

Build `candidate_allowed_mask [B, K_v]` from topology-local reachability, not by projecting fixed `allowed_target_mask [B, C_train]`.

Compatibility projection from global mask remains available only when `candidate_class_index >= 0` for all relevant candidates.

- [ ] **Step 5: Tests**

```python
def test_candidate_set_ce_trains_unseen_candidate_label():
    logits = torch.tensor([[0.0, 2.0, 0.0]], requires_grad=True)
    mask = torch.tensor([[False, True, False]])
    loss = candidate_set_cross_entropy_from_mask(logits, mask)
    loss.backward()
    assert logits.grad is not None
```

Run:

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\application\test_candidate_target_mapping.py tests\application\test_topology_conditioned_learning.py -v
```

---

## Task 5: Update EOPKGTopologyConditioned to Score Topology Candidates

**Files:**

- Modify: `src/domain/models/topology_conditioned.py`
- Test: `tests/domain/test_eopkg_topology_conditioned.py`

- [ ] **Step 1: Stop treating `output_dim` as candidate count**

`output_dim` remains the fixed-label compatibility dimension.

Candidate count comes from:

```python
candidate_class_index = contract["candidate_class_index"]
candidate_ids = contract["candidate_ids"]
candidate_labels = contract["candidate_labels"]
```

- [ ] **Step 2: Replace primary identity embedding**

Do not use:

```python
self.struct_node_emb = nn.Embedding(self.output_dim, self.hidden_dim)
```

as the primary candidate representation for topology-native scoring.

Stage MVP options:

1. Use `struct_x` + topology message passing as the primary representation.
2. Add a non-parametric deterministic candidate feature block from topology metadata where available.
3. Keep `struct_node_emb` only as compatibility embedding for known labels and combine it with an unseen/shared candidate embedding.

Recommended MVP:

```text
h0 = struct_input_proj(struct_x_or_zeros)
   + known_label_embedding(candidate_class_index where >= 0)
   + unseen_label_embedding(where candidate_class_index == -1)
```

This is not yet semantic zero-shot, but it prevents unseen candidates from being physically absent.

- [ ] **Step 3: Pool by candidate index, not class index**

Use:

```python
node_to_candidate_index = contract["struct_node_to_candidate_index"]
```

instead of:

```python
node_to_class = contract["struct_node_to_class_index"]
```

Candidate logits:

```python
candidate_logits[:, candidate_idx] = aggregate(node_scores for candidate_idx)
```

Fixed-label projection:

```python
fixed_label_logits = project_candidate_logits_to_fixed_labels(
    candidate_logits,
    candidate_class_index,
    output_dim=self.output_dim,
)
```

Candidates with `candidate_class_index=-1` are ignored in fixed-label projection, but preserved in candidate logits.

- [ ] **Step 4: Return metadata**

```python
return CandidatePredictionOutput(
    candidate_logits=candidate_logits,
    candidate_class_index=candidate_class_index,
    node_logits=node_scores,
    node_to_candidate_index=node_to_candidate_index,
    node_to_class_index=node_to_class,
    fixed_label_logits=fixed_label_logits,
    candidate_ids=tuple(candidate_ids),
    candidate_labels=tuple(candidate_labels),
    candidate_is_unseen=candidate_class_index < 0,
)
```

- [ ] **Step 5: Tests**

Run:

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\domain\test_eopkg_topology_conditioned.py -v
```

Expected:

- unseen candidate appears in `candidate_logits`;
- `fixed_label_logits` does not crash;
- `candidate_dynamic_count` can exceed known train label count.

---

## Task 6: Trainer Candidate-Local Loss Path

**Files:**

- Modify: `src/application/use_cases/trainer.py`
- Test: `tests/application/test_trainer_topology_conditioned_candidate_id.py`

- [ ] **Step 1: Extract target labels for candidate-id mode**

The trainer currently has `y` as fixed vocab index. For topology-native candidate loss, it also needs target labels.

Preferred source order:

1. graph contract field `target_label`;
2. topology-homogeneous batch sidecar metadata if direct string collation is not available;
3. fallback `reverse_activity_vocab[y]` only when target is known and not `<UNK>`.

If the target is `<UNK>` and no target label exists, mark missing target and fail above threshold.

- [ ] **Step 2: Use candidate-local CE**

For:

```yaml
training.candidate_contract_mode: candidate_id
training.candidate_identity_mode: topology_native
```

use:

```python
candidate_output = self._forward_candidate_output(contract)
target_mask = candidate_target_mask_from_labels(
    target_labels=batch_target_labels,
    candidate_labels=candidate_output.candidate_labels,
    device=candidate_output.candidate_logits.device,
)
loss = candidate_set_cross_entropy_from_mask(candidate_output.candidate_logits, target_mask)
```

Do not use `candidate_output.map_targets_to_candidate_mask(y)` as the primary path in topology-native mode.

- [ ] **Step 3: Fixed-label metrics remain compatibility**

If all targets and predictions have known class indexes, compute fixed-label metrics.

If target or prediction maps to `-1`, log:

```text
fixed_label_metric_coverage_rate
candidate_unseen_target_rate
candidate_unseen_prediction_rate
```

Do not silently count unseen future candidates as `<UNK>` for primary metrics.

- [ ] **Step 4: Tests**

```python
def test_trainer_candidate_id_uses_target_label_when_y_is_unknown():
    # y == 0 (<UNK>), target_label == "C", candidate_labels include "C"
    # Expected: missing_target_rate == 0 and loss uses candidate C.
```

Run:

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\application\test_trainer_topology_conditioned_candidate_id.py -v
```

---

## Task 7: Candidate-Local Evaluation, Drift Metrics, and Traces

**Files:**

- Modify: `src/application/use_cases/trainer.py`
- Modify: `src/application/services/structural_trace_payload_builder.py`
- Test: `tests/application/test_evaluator_metrics.py`
- Test: `tests/application/test_trainer_selective_tracing.py`

- [ ] **Step 1: Add candidate-local prediction labels**

For each prediction:

```python
pred_candidate_idx = torch.argmax(candidate_logits, dim=1)
pred_candidate_id = candidate_ids[pred_candidate_idx]
pred_candidate_label = candidate_labels[pred_candidate_idx]
```

- [ ] **Step 2: Candidate-local strict correctness**

For XES label-supervised mode:

```text
correct when pred_candidate_label == target_label
```

For future BPMN node-supervised mode:

```text
correct when pred_candidate_id == target_node_id
```

Current MVP uses label correctness.

- [ ] **Step 3: Candidate-local OOS**

Use `candidate_allowed_mask [B, K_v]`.

```text
candidate_oos = predicted candidate is not allowed by current topology from prefix state
```

Do not derive candidate OOS from fixed-label `allowed_target_mask` if unseen candidates exist.

- [ ] **Step 4: Candidate-local ECE/NLL**

Compute from `softmax(candidate_logits)`.

For set targets:

```text
target probability = sum probability over target_candidate_mask
NLL = -log(target probability)
```

- [ ] **Step 5: Trace payload**

Add flat trace attributes:

```text
candidate_id
candidate_label
candidate_is_unseen
target_candidate_found
target_candidate_count
candidate_oos
fixed_label_metric_available
```

Add JSON details:

```json
{
  "candidate_top_k": [
    {"candidate_id": "...", "label": "...", "score": 1.23, "prob": 0.41, "is_allowed": true}
  ]
}
```

All tensors must be detached before JSON conversion.

- [ ] **Step 6: Tests**

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\application\test_evaluator_metrics.py tests\application\test_trainer_selective_tracing.py -v
```

---

## Task 8: Candidate-Space Negative Sampling MVP

**Files:**

- Modify: `src/application/use_cases/trainer.py`
- Modify: `src/application/services/topology_conditioned_learning.py`
- Test: `tests/application/test_trainer_topology_conditioned_candidate_id.py`

- [ ] **Step 1: Re-enable `drop_edges` for candidate-id only when candidate axis is unchanged**

`drop_edges` same-version corruption keeps the same candidate IDs, so it is safe for topology-native candidate mode.

Condition:

```text
candidate_identity_mode=topology_native
corruption=drop_edges_same_version
candidate_ids_correct == candidate_ids_corrupt
```

- [ ] **Step 2: Keep wrong-version disabled for now**

Wrong-version topology can have different candidate axes. Do not re-enable until cross-axis alignment is defined.

Log:

```text
topology_conditioned_wrong_version_skipped_candidate_axis_mismatch_batches
```

- [ ] **Step 3: Loss**

Use CE delta only if target exists in both correct and corrupted topology:

```text
L_drop = relu(margin + CE_correct - CE_corrupt)
```

If target is removed by corruption, use allowed-flow penalty instead of CE delta.

- [ ] **Step 4: Tests**

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\application\test_trainer_topology_conditioned_candidate_id.py -v
```

---

## Task 9: Documentation and Debt Updates

**Files:**

- Modify: `docs/GNN_RUNTIME_MVP2_5.MD`
- Modify: `docs/GNN_LEARNING_STRATEGY.MD`
- Modify: `docs/current/architecture-debt.md`
- Modify: `docs/dissertation_changes.MD`

- [ ] **Step 1: Runtime docs**

Document three candidate layers:

```text
fixed_label: old global activity vocabulary
fixed_projection: dynamic logits projected to global labels
candidate_id/topology_native: topology-local candidate IDs
```

- [ ] **Step 2: Learning strategy docs**

Document:

```text
topology_conditioned requires topology-native candidate identity for zero-shot claims.
```

`standard` can still train baseline and compatibility models.

- [ ] **Step 3: Debt**

Add or update active debt:

- semantic candidate prototypes for unseen labels;
- exact BPMN node-id supervision for Camunda-like logs;
- cross-axis wrong-version negative sampling;
- candidate-space metric reporting in MLflow UI;
- missing-version metadata fallback when version is timestamp-only.

- [ ] **Step 4: Dissertation notes**

Add:

```text
The previous candidate_id implementation still depended on the training activity vocabulary, which collapses unseen future activities to <UNK> and prunes future topology edges. A business-valid zero-shot method must score topology-native candidates from the current process version, then project to fixed labels only for compatibility.
```

---

## Task 10: Verification

**Files:** no new source files.

- [ ] **Step 1: Focused domain tests**

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\domain\test_dynamic_graph_builder_topology_native_candidates.py tests\domain\test_eopkg_topology_conditioned.py -v
```

- [ ] **Step 2: Focused application tests**

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\application\test_candidate_target_mapping.py tests\application\test_topology_conditioned_learning.py tests\application\test_trainer_topology_conditioned_candidate_id.py -v
```

- [ ] **Step 3: Evaluator/tracing tests**

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\application\test_evaluator_metrics.py tests\application\test_trainer_selective_tracing.py -v
```

- [ ] **Step 4: Architecture guard**

```powershell
.\.venv-modern\Scripts\python.exe tools\architecture_guard.py
```

Expected:

```text
[ARCH_GUARD] OK
```

- [ ] **Step 5: Full tests**

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\ -v
```

Expected: pass. Known third-party warnings are acceptable unless new project warnings appear.

- [ ] **Step 6: Manual MLflow/run verification**

Run a small train/drift experiment with:

```yaml
model.type: EOPKGTopologyConditioned
training.candidate_contract_mode: candidate_id
training.candidate_identity_mode: topology_native
training.candidate_batch_topology_policy: group_by_topology
experiment.version_scope_policy: train_cut
```

Verify in logs/MLflow:

```text
candidate_dynamic_count changes when topology version changes
candidate_class_index contains -1 for unseen future candidates when expected
candidate_unseen_target_rate is non-zero on future versions with new labels
candidate_missing_target_rate remains near 0 for aligned XES/collapse runs
fixed_label_metric_coverage_rate drops below 1 when unseen candidates appear
```

---

## First Experiment After Implementation

Use `_EOPKGTC-UN` and `_EOPKGTC-UN-drift` style presets.

Required settings:

```yaml
model.type: EOPKGTopologyConditioned
training.candidate_contract_mode: candidate_id
training.candidate_identity_mode: topology_native
training.candidate_batch_topology_policy: group_by_topology
experiment.version_scope_policy: train_cut
experiment.mask_guided_enabled: false
experiment.statistic_enabled: false
```

Expected signs the contract is fixed:

```text
candidate_dynamic_count differs by version when topology differs
candidate_class_index may contain -1 for unseen future candidates
candidate_unseen_target_rate > 0 when future targets are outside train vocab
candidate_missing_target_rate ~= 0 if topology/log alignment is correct
fixed_label_metric_coverage_rate < 1 when unseen labels exist
candidate-level metrics are still available
```

If these signs do not appear, the system is still not testing real topology-native zero-shot behavior.

---

## Interpretation Rules

### If candidate-level F1 improves but fixed-label F1 does not

This is acceptable. Fixed-label metrics cannot represent unseen candidates without extending the global label space.

### If candidate-level OOS decreases but strict F1 does not

Topology helps keep predictions feasible, but exact next-activity ambiguity remains. This supports the dissertation point:

```text
topology constrains the valid region, but does not fully resolve exact next activity in ambiguous prefixes.
```

### If candidate_missing_target_rate is high

The run is invalid as alignment evidence. The issue is data/topology synchronization, not model quality.

### If unseen candidates exist but predictions never choose them

The contract is fixed, but the model still lacks semantic grounding. Move to semantic-topological prototypes / retrieval head.

### If candidate_dynamic_count remains fixed and no candidate has `class_index=-1`

The implementation did not solve the root issue. Re-check `DynamicGraphBuilder._resolve_compiled_topology` and topology candidate extraction.

---

## Stage 2 After This Plan

If this plan confirms that candidate identity is correct but transfer is still weak, proceed to:

1. semantic candidate prototypes from BPMN metadata;
2. non-parametric retrieval head over candidate prototypes;
3. edge-aware candidate scoring / flow-conditioned routing;
4. few-shot side adapter for new versions;
5. `versioned_zero_shot` lifecycle mode.

Do not tune loss weights aggressively before the candidate universe is verified.

---

## Self-Review

- This plan directly addresses the external audit items:
  - `<UNK>` collapse;
  - skipped future nodes/edges;
  - fixed `struct_node_emb(output_dim)` shortcut;
  - skipped candidate-id negative sampling.
- The plan keeps `gateway_mode=preserve` vs `collapse_for_prediction` as a data/config synchronization concern, not a prediction-class hardcode.
- The plan separates primary candidate-native metrics from compatibility fixed-label metrics.
- The plan avoids claiming semantic zero-shot until candidate identity and metric coverage are correct.
