# MVP2.5 P0/P1 Topology Flow Penalty Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` or `superpowers:subagent-driven-development` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** make `EOPKGTopologyConditioned` diagnostics truthful and add a topology-flow-aware training/evaluation layer that tests whether the model uses the current version's process flow, not only historical activity frequencies.

**Architecture:** keep the existing `candidate_contract_mode=candidate_id` path. Do not rework candidate mapping or MIL for the current `loan_v1_v4_simulated` dataset because BPMN activity labels are currently 1:1 with prediction nodes inside each version. Add candidate-level allowed-flow diagnostics and an optional topology penalty loss driven by `allowed_target_mask`.

**Tech Stack:** Python, PyTorch, PyTorch Geometric, MLflow metrics/traces, project config catalog, pytest.

---

## Current Diagnosis

### What is already solved

- `EOPKGTopologyConditioned.forward_candidate()` returns topology-local candidate logits `[B, C_v]`.
- `training.candidate_contract_mode=candidate_id` trains with set-aware CE directly over candidate logits.
- Topology-homogeneous batching prevents mixing incompatible candidate axes in one batch.
- In the current loan BPMN versions, `candidate_target_duplicate_count_max=1`, so the duplicate/MIL issue is not the blocker for this dataset.

### What is not solved

The model can still behave like a historical frequency learner:

```text
prefix + known v1/v2 logs -> likely next activity from observed history
```

instead of a topology-conditioned router:

```text
prefix + current topology edges -> candidates allowed by current process flow
```

If `C_v = 37` for every version and only edges change, strict-F1 degradation and OOS growth after drift mean the model is not sufficiently using the current `allowed_target_mask` / topology flow in its decision.

### Scope decision

This plan focuses on:

1. `P0`: trace/metadata correctness so the debug evidence is interpretable.
2. `P1`: topology-flow diagnostics and loss so the training objective explicitly penalizes invalid-flow probability mass.

This plan does not implement:

- full `group_by_topology` mixed-batch splitting;
- semantic candidate prototypes;
- node-level target identity from logs;
- new/removed class zero-shot retrieval head;
- negative topology sampling for `candidate_id`.

Those stay separate methodology steps after the flow penalty experiment.

---

## File Map

### Modify

- `src/domain/models/topology_conditioned.py`
  - expose stable model metadata used by traces and MLflow diagnostics.

- `src/application/services/structural_trace_payload_builder.py`
  - include truthful topology-conditioned model attributes in trace payload and flat span attributes.

- `src/application/services/candidate_target_mapping.py`
  - keep target mapping helpers unchanged unless a shared candidate mask projection helper is added here.

- `src/application/services/topology_conditioned_learning.py`
  - add pure tensor helpers for candidate-level allowed-flow metrics/loss.

- `src/application/use_cases/trainer.py`
  - map global `allowed_target_mask [B, C_train]` to candidate-space mask `[B, C_v]`;
  - log candidate flow diagnostics during train/val/test/drift;
  - optionally add topology-flow penalty to loss.

- `configs/ui/config_catalog.yaml`
  - add all new `training.*` config keys.

- `docs/GNN_RUNTIME_MVP2_5.MD`
  - document candidate-id flow penalty and metrics.

- `docs/GNN_LEARNING_STRATEGY.MD`
  - document how `topology_conditioned` uses topology-flow loss.

- `docs/current/project-state.md`
  - update runtime state after implementation.

- `docs/dissertation_changes.MD`
  - add the methodology conclusion: topology must be part of the objective, not only a candidate source.

### Tests

- `tests/application/test_structural_trace_payload_builder.py`
- `tests/application/test_topology_conditioned_learning.py`
- `tests/application/test_candidate_target_mapping.py` if mask projection helper is placed there.
- `tests/application/test_trainer_candidate_contract.py` or nearest existing trainer candidate tests.
- `tests/application/test_evaluator_metrics.py` if drift/eval metric aggregation changes.

---

## Proposed Config Keys

Add these keys under `training`:

```yaml
training.topology_flow_penalty_enabled: false
training.topology_flow_penalty_weight: 0.0
training.topology_flow_penalty_type: invalid_probability_mass
training.topology_flow_penalty_margin: 0.1
training.topology_flow_penalty_fail_on_missing_mask: true
```

Recommended first experiment values:

```yaml
training.learning_strategy: topology_conditioned
training.candidate_contract_mode: candidate_id
training.topology_flow_penalty_enabled: true
training.topology_flow_penalty_weight: 0.05
training.topology_flow_penalty_type: invalid_probability_mass
training.topology_flow_penalty_fail_on_missing_mask: true
```

Rationale:

- `invalid_probability_mass` is bounded and stable.
- `margin` is useful later, but can over-sharpen logits and worsen calibration.
- The penalty is explicit, not silently tied to `experiment.mask_guided_enabled`.

---

## Task 1: P0 Trace Metadata Correctness

**Files:**

- Modify: `src/domain/models/topology_conditioned.py`
- Modify: `src/application/services/structural_trace_payload_builder.py`
- Test: `tests/application/test_structural_trace_payload_builder.py`

- [ ] **Step 1: Write failing trace metadata test**

Add or update a test that builds a trace payload for a minimal topology-conditioned model stub:

```python
class TopologyConditionedStub:
    structural_mode = True
    fusion_mode = "TopologyConditionedCandidateScoring"
    candidate_scoring_mode = "cosine"
    candidate_pooling = "direct"

def test_trace_payload_marks_topology_conditioned_model_as_structural():
    payload = build_structural_prediction_trace_payload(
        stage="eval_drift_one_pass",
        global_index=0,
        contract=minimal_contract,
        logits=torch.zeros(1, 3),
        effective_logits=torch.zeros(1, 3),
        probs=torch.full((1, 3), 1 / 3),
        targets=torch.tensor([1]),
        predictions=torch.tensor([1]),
        reverse_vocab={0: "A", 1: "B", 2: "C"},
        model=TopologyConditionedStub(),
        reason="version_first",
        row=0,
    )
    assert payload["run"]["model_type"] == "TopologyConditionedStub"
    assert payload["run"]["structural_mode"] is True
    assert payload["run"]["fusion_mode"] == "TopologyConditionedCandidateScoring"
```

Run:

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\application\test_structural_trace_payload_builder.py -v
```

Expected: fail because current traces may report empty `fusion_mode` / `structural_mode=false` for topology-conditioned models.

- [ ] **Step 2: Expose model metadata**

In `EOPKGTopologyConditioned`, set trace-friendly fields during initialization:

```python
self.structural_mode = True
self.fusion_mode = "TopologyConditionedCandidateScoring"
self.candidate_scoring_mode = str(candidate_scoring_mode)
self.candidate_pooling = str(candidate_pooling)
```

Use existing constructor parameter names from `topology_conditioned.py`.

- [ ] **Step 3: Add payload fallbacks**

In `structural_trace_payload_builder.py`, make `structural_mode` robust:

```python
model_type = str(model.__class__.__name__)
structural_mode = bool(getattr(model, "structural_mode", model_type == "EOPKGTopologyConditioned"))
fusion_mode = str(
    getattr(model, "fusion_mode", "")
    or getattr(model, "candidate_scoring_mode", "")
    or ("TopologyConditionedCandidateScoring" if model_type == "EOPKGTopologyConditioned" else "")
)
```

Also add flat attributes if present:

```python
"candidate_scoring_mode": str(run.get("candidate_scoring_mode", "")),
"candidate_pooling": str(run.get("candidate_pooling", "")),
```

- [ ] **Step 4: Run targeted tests**

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\application\test_structural_trace_payload_builder.py -v
```

Expected: pass.

---

## Task 2: Candidate-Space Allowed Flow Projection

**Files:**

- Modify: `src/application/services/topology_conditioned_learning.py`
- Test: `tests/application/test_topology_conditioned_learning.py`

- [ ] **Step 1: Write failing projection tests**

Add tests for projecting global label mask to candidate mask:

```python
def test_candidate_allowed_mask_projects_global_mask_to_candidate_axis():
    allowed = torch.tensor([[True, False, True, False]])
    candidate_class_index = torch.tensor([2, 0, 3])

    mask = candidate_allowed_mask_from_global(allowed, candidate_class_index)

    assert mask.tolist() == [[True, True, False]]
```

Add guard test:

```python
def test_candidate_allowed_mask_rejects_out_of_range_class_index():
    allowed = torch.ones(1, 2, dtype=torch.bool)
    candidate_class_index = torch.tensor([0, 3])

    with pytest.raises(ValueError, match="candidate_class_index"):
        candidate_allowed_mask_from_global(allowed, candidate_class_index)
```

- [ ] **Step 2: Implement pure helper**

Add:

```python
def candidate_allowed_mask_from_global(
    allowed_target_mask: torch.Tensor,
    candidate_class_index: torch.Tensor,
) -> torch.Tensor:
    if allowed_target_mask.dim() != 2:
        raise ValueError("allowed_target_mask must have shape [B, C_train].")
    class_index = candidate_class_index.to(device=allowed_target_mask.device, dtype=torch.long).view(-1)
    if class_index.numel() <= 0:
        raise ValueError("candidate_class_index must not be empty.")
    valid = class_index >= 0
    if bool(valid.any()):
        max_index = int(class_index[valid].max().detach().cpu().item())
        if max_index >= int(allowed_target_mask.size(1)):
            raise ValueError(
                "candidate_class_index contains values outside allowed_target_mask class axis: "
                f"max={max_index} classes={int(allowed_target_mask.size(1))}."
            )
    projected = torch.zeros(
        (allowed_target_mask.size(0), class_index.numel()),
        device=allowed_target_mask.device,
        dtype=torch.bool,
    )
    if bool(valid.any()):
        projected[:, valid] = allowed_target_mask.to(dtype=torch.bool)[:, class_index[valid]]
    return projected
```

- [ ] **Step 3: Run focused tests**

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\application\test_topology_conditioned_learning.py -v
```

Expected: pass.

---

## Task 3: Candidate Flow Diagnostics

**Files:**

- Modify: `src/application/services/topology_conditioned_learning.py`
- Test: `tests/application/test_topology_conditioned_learning.py`

- [ ] **Step 1: Write failing diagnostics tests**

```python
def test_candidate_allowed_flow_summary_reports_invalid_mass_and_oos():
    logits = torch.tensor([[3.0, 1.0, 4.0]])
    candidate_allowed_mask = torch.tensor([[True, False, False]])

    summary = candidate_allowed_flow_summary(logits, candidate_allowed_mask)

    assert summary["candidate_oos_rate"] == 1.0
    assert summary["candidate_invalid_probability_mass"] > 0.5
    assert summary["candidate_valid_invalid_logit_margin"] < 0.0
```

- [ ] **Step 2: Implement diagnostics helper**

Add:

```python
def candidate_allowed_flow_summary(
    candidate_logits: torch.Tensor,
    candidate_allowed_mask: torch.Tensor,
) -> dict[str, float]:
    if candidate_logits.dim() != 2:
        raise ValueError("candidate_logits must have shape [B, C_v].")
    mask = candidate_allowed_mask.to(device=candidate_logits.device, dtype=torch.bool)
    if mask.shape != candidate_logits.shape:
        raise ValueError("candidate_allowed_mask must have the same shape as candidate_logits.")
    probs = torch.softmax(candidate_logits, dim=1)
    invalid = ~mask
    invalid_mass = probs.masked_fill(~invalid, 0.0).sum(dim=1)
    valid_mass = probs.masked_fill(~mask, 0.0).sum(dim=1)
    pred = torch.argmax(candidate_logits, dim=1)
    pred_allowed = mask.gather(1, pred.view(-1, 1)).view(-1)
    valid_logits = candidate_logits.masked_fill(~mask, float("-inf"))
    invalid_logits = candidate_logits.masked_fill(~invalid, float("-inf"))
    valid_max = torch.where(mask.any(dim=1), valid_logits.max(dim=1).values, torch.zeros_like(invalid_mass))
    invalid_max = torch.where(invalid.any(dim=1), invalid_logits.max(dim=1).values, torch.zeros_like(invalid_mass))
    margin = valid_max - invalid_max
    return {
        "candidate_oos_rate": float((~pred_allowed).float().mean().detach().cpu().item()) if pred_allowed.numel() else 0.0,
        "candidate_invalid_probability_mass": float(invalid_mass.mean().detach().cpu().item()) if invalid_mass.numel() else 0.0,
        "candidate_valid_probability_mass": float(valid_mass.mean().detach().cpu().item()) if valid_mass.numel() else 0.0,
        "candidate_valid_invalid_logit_margin": float(margin.mean().detach().cpu().item()) if margin.numel() else 0.0,
    }
```

- [ ] **Step 3: Run focused tests**

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\application\test_topology_conditioned_learning.py -v
```

Expected: pass.

---

## Task 4: Candidate Topology Flow Penalty Loss

**Files:**

- Modify: `src/application/services/topology_conditioned_learning.py`
- Test: `tests/application/test_topology_conditioned_learning.py`

- [ ] **Step 1: Write failing loss tests**

```python
def test_candidate_topology_flow_penalty_invalid_probability_mass_is_bounded():
    logits = torch.tensor([[4.0, 0.0, 3.0]], requires_grad=True)
    mask = torch.tensor([[True, False, False]])

    loss = candidate_topology_flow_penalty_loss(
        logits,
        mask,
        penalty_type="invalid_probability_mass",
        margin=0.1,
    )

    assert 0.0 <= float(loss.detach().item()) <= 1.0
    loss.backward()
    assert logits.grad is not None
```

```python
def test_candidate_topology_flow_penalty_margin_penalizes_invalid_max():
    logits = torch.tensor([[1.0, 3.0]], requires_grad=True)
    mask = torch.tensor([[True, False]])

    loss = candidate_topology_flow_penalty_loss(
        logits,
        mask,
        penalty_type="margin",
        margin=0.5,
    )

    assert float(loss.detach().item()) > 0.0
```

- [ ] **Step 2: Implement loss**

Add:

```python
def candidate_topology_flow_penalty_loss(
    candidate_logits: torch.Tensor,
    candidate_allowed_mask: torch.Tensor,
    *,
    penalty_type: str = "invalid_probability_mass",
    margin: float = 0.1,
) -> torch.Tensor:
    if candidate_logits.dim() != 2:
        raise ValueError("candidate_logits must have shape [B, C_v].")
    mask = candidate_allowed_mask.to(device=candidate_logits.device, dtype=torch.bool)
    if mask.shape != candidate_logits.shape:
        raise ValueError("candidate_allowed_mask must have the same shape as candidate_logits.")
    penalty_type = str(penalty_type).strip().lower()
    invalid = ~mask
    if penalty_type == "invalid_probability_mass":
        probs = torch.softmax(candidate_logits, dim=1)
        return probs.masked_fill(~invalid, 0.0).sum(dim=1).mean()
    if penalty_type == "margin":
        valid_logits = candidate_logits.masked_fill(~mask, float("-inf"))
        invalid_logits = candidate_logits.masked_fill(~invalid, float("-inf"))
        valid_max = torch.where(mask.any(dim=1), valid_logits.max(dim=1).values, candidate_logits.new_zeros(candidate_logits.size(0)))
        invalid_max = torch.where(invalid.any(dim=1), invalid_logits.max(dim=1).values, candidate_logits.new_zeros(candidate_logits.size(0)))
        return torch.relu(invalid_max - valid_max + float(margin)).mean()
    raise ValueError("penalty_type must be invalid_probability_mass or margin.")
```

- [ ] **Step 3: Run focused tests**

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\application\test_topology_conditioned_learning.py -v
```

Expected: pass.

---

## Task 5: Trainer Integration for Candidate Flow Metrics

**Files:**

- Modify: `src/application/use_cases/trainer.py`
- Test: nearest existing trainer candidate tests, or create `tests/application/test_trainer_candidate_flow_metrics.py`

- [ ] **Step 0: Add required imports**

In `src/application/use_cases/trainer.py`, extend the existing import from
`src.application.services.topology_conditioned_learning`:

```python
from src.application.services.topology_conditioned_learning import (
    allowed_set_mass_leakage,
    allowed_set_loss,
    candidate_allowed_flow_summary,
    candidate_allowed_mask_from_global,
    candidate_topology_flow_penalty_loss,
    margin_negative_loss,
    version_weighted_cross_entropy,
)
```

- [ ] **Step 1: Write failing trainer metric test**

Create a test that runs a minimal candidate-id train/eval step with:

- candidate logits `[B, C_v]`;
- `candidate_class_index`;
- global `allowed_target_mask`;
- one invalid high-score prediction.

Expected tracker metrics include:

```text
train_candidate_oos_rate
train_candidate_invalid_probability_mass
train_candidate_valid_invalid_logit_margin
```

- [ ] **Step 2: Add trainer helper**

Add a private helper:

```python
def _candidate_allowed_mask(
    self,
    *,
    candidate_output: CandidatePredictionOutput,
    allowed_mask: torch.Tensor | None,
) -> torch.Tensor | None:
    if not isinstance(allowed_mask, torch.Tensor):
        return None
    return candidate_allowed_mask_from_global(allowed_mask, candidate_output.candidate_class_index)
```

- [ ] **Step 3: Accumulate diagnostics in train/validation loop**

In `_run_epoch`, after `candidate_output` and `allowed_mask` are available:

```python
candidate_allowed_mask = self._candidate_allowed_mask(
    candidate_output=candidate_output,
    allowed_mask=allowed_mask,
)
if candidate_allowed_mask is not None:
    flow_summary = candidate_allowed_flow_summary(logits.detach(), candidate_allowed_mask)
    self._accumulate_candidate_flow_stats(forward_stats, flow_summary)
```

Use detached logits for diagnostics to avoid retaining graphs.

- [ ] **Step 4: Implement `_accumulate_candidate_flow_stats`**

Add a dedicated accumulator near the other trainer stats helpers:

```python
def _accumulate_candidate_flow_stats(self, bucket: Dict[str, Any], summary: Dict[str, float]) -> None:
    bucket["candidate_flow_batches"] = int(bucket.get("candidate_flow_batches", 0)) + 1
    for key in (
        "candidate_oos_rate",
        "candidate_invalid_probability_mass",
        "candidate_valid_probability_mass",
        "candidate_valid_invalid_logit_margin",
    ):
        bucket[f"{key}_sum"] = float(bucket.get(f"{key}_sum", 0.0)) + float(summary.get(key, 0.0))
```

This helper is intentionally separate from `_accumulate_topology_conditioned_stats`
because these are diagnostics, not objective-loss summaries.

- [ ] **Step 5: Log epoch metrics**

Extend `_log_forward_stats` to log:

```text
<prefix>_candidate_oos_rate
<prefix>_candidate_invalid_probability_mass
<prefix>_candidate_valid_probability_mass
<prefix>_candidate_valid_invalid_logit_margin
```

- [ ] **Step 6: Run targeted trainer tests**

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\application -k "candidate and trainer" -v
```

Expected: pass.

---

## Task 6: Trainer Integration for Topology Flow Penalty

**Files:**

- Modify: `src/application/use_cases/trainer.py`
- Modify: config parsing object for learning strategy if separated from trainer.
- Test: `tests/application/test_trainer_candidate_flow_metrics.py`

- [ ] **Step 1: Write failing loss integration test**

Use a model stub that returns fixed candidate logits where invalid candidate probability is high. Configure:

```python
training = {
    "learning_strategy": "topology_conditioned",
    "candidate_contract_mode": "candidate_id",
    "topology_flow_penalty_enabled": True,
    "topology_flow_penalty_weight": 0.5,
    "topology_flow_penalty_type": "invalid_probability_mass",
}
```

Assert total loss is greater than candidate CE alone and metric `train_topology_flow_penalty_loss` is logged.

- [ ] **Step 2: Parse config keys**

In trainer init or `LearningStrategyConfig`, parse:

```python
self.topology_flow_penalty_enabled = bool(config.get("topology_flow_penalty_enabled", False))
self.topology_flow_penalty_weight = float(config.get("topology_flow_penalty_weight", 0.0) or 0.0)
self.topology_flow_penalty_type = str(config.get("topology_flow_penalty_type", "invalid_probability_mass"))
self.topology_flow_penalty_margin = float(config.get("topology_flow_penalty_margin", 0.1) or 0.1)
self.topology_flow_penalty_fail_on_missing_mask = bool(config.get("topology_flow_penalty_fail_on_missing_mask", True))
```

- [ ] **Step 3: Apply loss only when explicitly enabled**

In `_run_epoch`:

```python
if (
    candidate_output is not None
    and self.learning_strategy_config.is_topology_conditioned
    and self.topology_flow_penalty_enabled
    and self.topology_flow_penalty_weight > 0.0
):
    candidate_allowed_mask = self._candidate_allowed_mask(...)
    if candidate_allowed_mask is None:
        if self.topology_flow_penalty_fail_on_missing_mask:
            raise ValueError("topology_flow_penalty requires allowed_target_mask.")
    else:
        flow_loss = candidate_topology_flow_penalty_loss(
            logits,
            candidate_allowed_mask,
            penalty_type=self.topology_flow_penalty_type,
            margin=self.topology_flow_penalty_margin,
        )
        loss = loss + self.topology_flow_penalty_weight * flow_loss
        self._accumulate_topology_conditioned_stats(
            forward_stats,
            key_prefix="topology_flow_penalty",
            correct_ce=base_ce_loss,
            objective_loss=flow_loss,
        )
```

Do not pass `value=...`. The current `_accumulate_topology_conditioned_stats`
signature is:

```python
def _accumulate_topology_conditioned_stats(
    self,
    bucket: Dict[str, Any],
    *,
    key_prefix: str,
    correct_ce: torch.Tensor,
    negative_ce: torch.Tensor | None = None,
    objective_loss: torch.Tensor | None = None,
    margin: float = 0.0,
) -> None:
    ...
```

For a single-objective regularizer such as flow penalty, pass `correct_ce`
for context and `objective_loss=flow_loss`; leave `negative_ce=None`.

- [ ] **Step 4: Keep `experiment.mask_guided_enabled` independent**

Do not require mask-guided inference. The flow penalty consumes the same topology-derived mask as supervision, but it does not clamp logits at inference.

- [ ] **Step 5: Run targeted tests**

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\application -k "candidate and topology" -v
```

Expected: pass.

---

## Task 7: Eval/Drift Candidate Flow Metrics

**Files:**

- Modify: `src/application/use_cases/trainer.py`
- Test: `tests/application/test_evaluator_metrics.py`

- [ ] **Step 1: Write failing eval metric test**

For `candidate_contract_mode=candidate_id`, assert `_evaluate_test` returns/logs:

```text
candidate_oos_rate
candidate_invalid_probability_mass
candidate_valid_invalid_logit_margin
```

where predictions are measured in candidate space before projection back to global labels.

- [ ] **Step 2: Add evaluation aggregation**

In `_evaluate_test`, after `candidate_output` and `allowed_mask`:

```python
candidate_allowed_mask = self._candidate_allowed_mask(
    candidate_output=candidate_output,
    allowed_mask=allowed_mask,
)
if candidate_allowed_mask is not None:
    summary = candidate_allowed_flow_summary(raw_logits, candidate_allowed_mask)
    candidate_flow_metric_lists["candidate_oos_rate"].append(summary["candidate_oos_rate"])
    ...
```

Log final averages with `test_` prefix.

- [ ] **Step 3: Extend `DriftInferenceRecords` explicitly**

`DriftInferenceRecords` is a typed compact dataclass record object, so new drift
metrics must be added as first-class fields and initialized in the constructor
call inside `_collect_drift_inference_records`.

Add fields:

```python
candidate_oos_flags: np.ndarray
candidate_invalid_probability_mass: np.ndarray
candidate_valid_probability_mass: np.ndarray
candidate_valid_invalid_logit_margin: np.ndarray
```

Initialize collection lists before the inference loop:

```python
all_candidate_oos_flags: list[float] = []
all_candidate_invalid_probability_mass: list[float] = []
all_candidate_valid_probability_mass: list[float] = []
all_candidate_valid_invalid_logit_margin: list[float] = []
```

When `candidate_output` or candidate allowed mask is unavailable, append neutral
values:

```python
all_candidate_oos_flags.extend([0.0] * batch_size)
all_candidate_invalid_probability_mass.extend([0.0] * batch_size)
all_candidate_valid_probability_mass.extend([0.0] * batch_size)
all_candidate_valid_invalid_logit_margin.extend([0.0] * batch_size)
```

When available, append per-row values derived from candidate-space logits and
candidate allowed mask. Do not store tensors in the dataclass; detach and move
to NumPy immediately.

Update the return:

```python
return DriftInferenceRecords(
    ...
    candidate_oos_flags=np.asarray(all_candidate_oos_flags, dtype=np.float32),
    candidate_invalid_probability_mass=np.asarray(all_candidate_invalid_probability_mass, dtype=np.float32),
    candidate_valid_probability_mass=np.asarray(all_candidate_valid_probability_mass, dtype=np.float32),
    candidate_valid_invalid_logit_margin=np.asarray(all_candidate_valid_invalid_logit_margin, dtype=np.float32),
    ...
)
```

- [ ] **Step 4: Add one-pass drift aggregation**

In `_collect_drift_inference_records`, include per-record or per-batch values sufficient to aggregate drift windows:

```text
candidate_pred_allowed
candidate_invalid_probability_mass
candidate_valid_invalid_logit_margin
```

Then in drift-window aggregation log:

```text
drift_window_candidate_oos
drift_window_candidate_invalid_probability_mass
drift_window_candidate_valid_invalid_logit_margin
```

- [ ] **Step 5: Run evaluator tests**

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\application\test_evaluator_metrics.py tests\application\test_drift_one_pass_evaluation.py -v
```

Expected: pass.

---

## Task 8: Config Catalog and UI Exposure

**Files:**

- Modify: `configs/ui/config_catalog.yaml`
- Test: existing config catalog tests, if present.

- [ ] **Step 1: Add catalog entries**

Add entries for:

```text
training.topology_flow_penalty_enabled
training.topology_flow_penalty_weight
training.topology_flow_penalty_type
training.topology_flow_penalty_margin
training.topology_flow_penalty_fail_on_missing_mask
```

Recommended UI placement:

```yaml
ui_level: advanced
tab: Experiment Run
group: Learning Strategy
active_when:
  training.learning_strategy: topology_conditioned
  training.candidate_contract_mode: candidate_id
```

- [ ] **Step 2: Run catalog/UI smoke tests**

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\ -k "config_catalog or desktop_ui or experiment_ui" -v
```

Expected: pass or no matching tests.

---

## Task 9: Documentation Update

**Files:**

- Modify: `docs/GNN_RUNTIME_MVP2_5.MD`
- Modify: `docs/GNN_LEARNING_STRATEGY.MD`
- Modify: `docs/current/project-state.md`
- Modify: `docs/dissertation_changes.MD`

- [ ] **Step 1: Update runtime doc**

Document:

- `candidate_id` now has candidate-space flow diagnostics;
- `allowed_target_mask [B, C_train]` is projected to candidate axis through `candidate_class_index`;
- flow penalty is a train objective, not inference masking.

- [ ] **Step 2: Update learning strategy doc**

Document:

```text
standard: no topology-flow loss
topology_conditioned: may enable topology_flow_penalty to penalize invalid-flow probability mass
```

- [ ] **Step 3: Update dissertation notes**

Add a short thesis:

```text
Candidate-id scoring alone does not prove zero-shot topology transfer when candidate labels are stable across versions. The learning objective must explicitly train flow sensitivity: the model should lose probability mass on candidates invalid under the current topology. This separates topology as candidate vocabulary from topology as routing constraint.
```

---

## Task 10: Verification

**Files:** no new source files.

- [ ] **Step 1: Run focused tests**

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\application\test_topology_conditioned_learning.py tests\application\test_structural_trace_payload_builder.py -v
```

Expected: pass.

- [ ] **Step 2: Run trainer/evaluator tests**

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\application -k "candidate or drift or evaluator or topology_conditioned" -v
```

Expected: pass.

- [ ] **Step 3: Run architecture guard**

```powershell
.\.venv-modern\Scripts\python.exe tools\architecture_guard.py
```

Expected: `[ARCH_GUARD] OK`.

- [ ] **Step 4: Run full test suite if time permits**

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\ -v
```

Expected: pass. Known third-party deprecation warnings are acceptable unless new project warnings appear.

---

## Experiment After Implementation

### First sanity run

Use the same preset family as `_EOPKGTC-UN`, but enable:

```yaml
training.learning_strategy: topology_conditioned
training.candidate_contract_mode: candidate_id
training.topology_flow_penalty_enabled: true
training.topology_flow_penalty_weight: 0.05
training.topology_flow_penalty_type: invalid_probability_mass
```

Expected signs:

- `candidate_missing_target_rate = 0`;
- `candidate_target_in_candidate_set_rate = 1`;
- `train_candidate_invalid_probability_mass` decreases across epochs;
- `validation_candidate_oos_rate` decreases or stays lower than no-penalty run;
- NLL/ECE must not explode.

### Drift comparison

Compare against:

- baseline run;
- `_EOPKGTC-UN` candidate-id standard run;
- previous half-projection run `8900633b257943b3b795be1f7d43dca6`.

Primary metrics:

- `drift_window_strict_macro_f1`;
- `drift_window_candidate_oos`;
- `drift_window_candidate_invalid_probability_mass`;
- `drift_window_test_oos`;
- `drift_window_test_nll`;
- `drift_window_test_ece`.

Interpretation:

- If strict-F1 does not improve but candidate-OOS and invalid mass drop, topology helps constrain the feasible area but still does not solve ambiguous exact next activity.
- If invalid mass does not drop, the flow penalty is too weak, miswired, or `allowed_target_mask` does not represent the current topology correctly.
- If invalid mass drops but NLL/ECE worsens, margin/logit sharpness is too aggressive; use `invalid_probability_mass` and lower weight.

---

## Risks and Guardrails

### Risk: double-counting mask effect

`experiment.mask_guided_enabled` clamps or changes inference logits. `topology_flow_penalty` changes train loss. They must stay independently switchable.

### Risk: shape mismatch hidden by current dataset

Today `C_v=37` and `C_train=37`, but the code must still explicitly project from global mask to candidate axis through `candidate_class_index`.

### Risk: penalty improves OOS but hurts strict-F1

This is a valid scientific result. It means topology is acting as a feasibility constraint, not as a complete exact-next-activity predictor.

### Risk: future new candidates

This plan still projects to known global labels. Full new-candidate zero-shot requires semantic-topological retrieval or dynamic metric head. Keep this outside the current P0/P1 implementation.

---

## Self-Review

- `P0` is scoped to trace metadata correctness and does not change model behavior.
- `P1` does not reimplement candidate-id contract; it adds flow-aware diagnostics and an optional loss.
- The plan keeps mask-guided inference separate from train-time topology-flow penalty.
- The plan avoids MIL complexity for the current loan dataset where BPMN labels are 1:1 with nodes.
- The plan keeps future work explicit: semantic prototypes, true new-candidate support, and negative topology sampling remain separate methodology steps.
