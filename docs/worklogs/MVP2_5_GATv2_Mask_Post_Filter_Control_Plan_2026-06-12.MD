# GATv2 Mask Post-Filter Control Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reviewer-control baseline `GATv2 + Mask`: a standard fixed-vocabulary `BaselineGATv2` classifier whose inference logits are post-filtered by the current topology mask.

**Architecture:** Keep the model as a logs-only fixed-head classifier over `C_train`; do not add topology-native candidate scoring. Add an explicit mask policy switch so evaluation can force hard post-filtering without relying on accumulated mask reliability. Create train and drift presets that reuse the synthetic `_Base-UN-f1` / `_Base-UN-f1-drift` setup but isolate the effect of topology post-filtering.

**Tech Stack:** Python, PyTorch, PyTorch Geometric, existing `ModelTrainer`, `DynamicGraphBuilder`, Experiment UI presets, pytest.

---

## Research Purpose

This control answers the reviewer question:

> Is EOPKG better only because invalid activities are filtered after prediction?

The intended comparison table becomes:

| Model | Candidate Space | Topology Use | Expected Interpretation |
|---|---|---|---|
| `LSTM` | fixed `C_train` | none | classical sequence baseline |
| `GATv2` | fixed `C_train` | none | logs-only graph baseline |
| `GATv2 + Mask` | fixed `C_train` | inference post-filter | tests whether topology mask alone is enough |
| `EOPKG-WI` | topology-native `C^(v)` | candidate scoring, no impulse | tests dynamic candidate space without impulse |
| `EOPKG` | topology-native `C^(v)` | impulse + candidate scoring | full method |

Expected result:

1. `GATv2 + Mask` should reduce OOS compared with plain `GATv2`.
2. `GATv2 + Mask` should not match EOPKG on future/unseen activities because the fixed head cannot emit classes outside `C_train`.
3. Calibration may improve only partially because the masked distribution is a renormalized fixed-vocabulary distribution.

---

## Current Behavior To Preserve

Current code path:

1. `experiment.mask_guided_enabled` is copied to `training.mask_guided_enabled` in `src/cli.py`.
2. `ModelTrainer._apply_mask_guided_logits()` supports:
   - `off`: no effect;
   - `soft`: subtract `training.mask_guided_soft_penalty` from invalid logits;
   - `hard`: set invalid logits to a very low value.
3. `ModelTrainer._resolve_mask_guided_policy()` currently decides hard/soft using mask reliability.
4. In `eval_drift` loaded from checkpoint, hard mode may not activate because reliability state is not guaranteed to be available in the current process.

Required new behavior:

1. Existing `auto` behavior remains default-compatible.
2. New explicit policy can force hard post-filtering in eval/drift:

```yaml
training.mask_guided_policy: hard
training.mask_guided_apply_in_eval: true
experiment.mask_guided_enabled: true
```

3. The masked GATv2 control must still be fixed-vocabulary:

```yaml
model.type: BaselineGATv2
training.candidate_contract_mode: fixed_label
training.dynamic_candidate_contract_enabled: false
experiment.structural_mode: false
```

Topology is used only to build `allowed_target_mask`; the model itself does not consume structural tensors.

---

## Files To Modify

- Modify: `src/application/use_cases/trainer.py`
  - Add explicit `training.mask_guided_policy`.
  - Preserve current auto reliability behavior.
  - Log whether mask policy was `off`, `soft`, or `hard`.

- Modify: `configs/ui/config_catalog.yaml`
  - Add `training.mask_guided_policy` enum: `auto`, `soft`, `hard`, `off`.

- Modify: `outputs/ui/experiment_ui_presets.json`
  - Add `_Base-MN-f1`.
  - Add `_Base-MN-f1-drift`.

- Modify: `docs/GNN_RUNTIME_MVP2_5.MD`
  - Clarify that `GATv2 + Mask` is a reviewer-control fixed-vocabulary post-filter, not topology-native scoring.

- Modify: `docs/current/project-state.md`
  - Add short current-state note that mask post-filter control exists for fixed-vocabulary baselines.

- Test: `tests/application/test_evaluator_metrics.py`
  - Add unit coverage for forced hard policy in eval.
  - Add regression that hard mask changes fixed-vocabulary prediction from invalid high-logit class to valid lower-logit class.

- Test: `tests/application/test_experiment_ui_catalog_fields.py`
  - Add catalog coverage for `training.mask_guided_policy`.

---

## Task 1: Add Explicit Mask Policy Config

**Files:**
- Modify: `src/application/use_cases/trainer.py`
- Test: `tests/application/test_evaluator_metrics.py`

- [ ] **Step 1: Write failing test for forced hard eval policy**

Add this test to `tests/application/test_evaluator_metrics.py` near existing mask-guided tests:

```python
def test_mask_guided_policy_can_force_hard_in_eval_without_reliability_state():
    trainer = ModelTrainer(
        xes_adapter=_DummyAdapter(),
        prefix_policy=_DummyPrefixPolicy(),
        graph_builder=_DummyGraphBuilder(),
        model=_ConstantClassZero3(),
        log_path="in_memory.xes",
        config={
            "mode": "eval_drift",
            "device": "cpu",
            "show_progress": False,
            "tqdm_disable": True,
            "mask_guided_enabled": True,
            "mask_guided_policy": "hard",
            "mask_guided_apply_in_eval": True,
        },
    )

    assert trainer._resolve_mask_guided_policy(
        training=False,
        batch_target_in_mask_rate=None,
        batch_samples=4,
    ) == "hard"
```

- [ ] **Step 2: Run RED test**

Run:

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\application\test_evaluator_metrics.py::test_mask_guided_policy_can_force_hard_in_eval_without_reliability_state -v
```

Expected: FAIL because `mask_guided_policy` is not parsed yet and eval falls back to current soft/auto behavior.

- [ ] **Step 3: Implement minimal config parsing**

In `ModelTrainer.__init__`, after `self.mask_guided_enabled` and before threshold fields, add:

```python
self.mask_guided_policy = str(config.get("mask_guided_policy", "auto") or "auto").strip().lower()
if self.mask_guided_policy not in {"auto", "off", "soft", "hard"}:
    raise ValueError("training.mask_guided_policy must be auto, off, soft, or hard.")
```

- [ ] **Step 4: Update policy resolution**

At the beginning of `_resolve_mask_guided_policy()`, replace the first branch with:

```python
if not self.mask_guided_enabled:
    return "off"
if self.mask_guided_policy in {"off", "soft", "hard"}:
    if not training and not self.mask_guided_apply_in_eval:
        return "off"
    return self.mask_guided_policy
```

Leave the existing train/eval reliability logic below as the `auto` path.

- [ ] **Step 5: Run GREEN test**

Run:

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\application\test_evaluator_metrics.py::test_mask_guided_policy_can_force_hard_in_eval_without_reliability_state -v
```

Expected: PASS.

- [ ] **Step 6: Run existing mask tests**

Run:

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\application\test_evaluator_metrics.py::test_mask_guided_policy_hard_when_reliable_and_soft_when_not tests\application\test_evaluator_metrics.py::test_mask_guided_logits_hard_and_soft_behaviour -v
```

Expected: PASS.

---

## Task 2: Prove Fixed-Vocabulary Post-Filter Changes Prediction

**Files:**
- Modify: `tests/application/test_evaluator_metrics.py`
- No production file changes if Task 1 is correct.

- [ ] **Step 1: Write regression test for hard post-filter**

Add this test near the mask-guided tests:

```python
def test_mask_guided_hard_post_filter_changes_fixed_vocab_prediction_to_allowed_class():
    trainer = ModelTrainer(
        xes_adapter=_DummyAdapter(),
        prefix_policy=_DummyPrefixPolicy(),
        graph_builder=_DummyGraphBuilder(),
        model=_ConstantClassZero3(),
        log_path="in_memory.xes",
        config={
            "mode": "eval_drift",
            "device": "cpu",
            "show_progress": False,
            "tqdm_disable": True,
            "mask_guided_enabled": True,
            "mask_guided_policy": "hard",
            "mask_guided_apply_in_eval": True,
        },
    )

    logits = torch.tensor([[9.0, 1.0, 0.5]], dtype=torch.float32)
    allowed_mask = torch.tensor([[False, True, True]], dtype=torch.bool)
    policy = trainer._resolve_mask_guided_policy(
        training=False,
        batch_target_in_mask_rate=None,
        batch_samples=1,
    )

    effective_logits = trainer._apply_mask_guided_logits(
        logits=logits,
        allowed_mask=allowed_mask,
        policy=policy,
    )

    assert int(torch.argmax(logits, dim=1).item()) == 0
    assert int(torch.argmax(effective_logits, dim=1).item()) == 1
```

- [ ] **Step 2: Run test**

Run:

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\application\test_evaluator_metrics.py::test_mask_guided_hard_post_filter_changes_fixed_vocab_prediction_to_allowed_class -v
```

Expected: PASS.

- [ ] **Step 3: Run focused evaluator tests**

Run:

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\application\test_evaluator_metrics.py -v
```

Expected: PASS.

---

## Task 3: Expose `training.mask_guided_policy` In Config Catalog

**Files:**
- Modify: `configs/ui/config_catalog.yaml`
- Modify: `tests/application/test_experiment_ui_catalog_fields.py`

- [ ] **Step 1: Add failing catalog test**

Add this test to `tests/application/test_experiment_ui_catalog_fields.py`:

```python
def test_catalog_includes_mask_guided_policy():
    catalog = _load_catalog()
    field = catalog["training.mask_guided_policy"]

    assert field["enum"] == ["auto", "off", "soft", "hard"]
    assert field["default"] == "auto"
    assert field["ui"]["group"] == "core"
```

- [ ] **Step 2: Run RED test**

Run:

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\application\test_experiment_ui_catalog_fields.py::test_catalog_includes_mask_guided_policy -v
```

Expected: FAIL because the catalog entry does not exist.

- [ ] **Step 3: Add catalog entry**

In `configs/ui/config_catalog.yaml`, place this near existing `training.mask_guided_apply_in_eval`:

```yaml
  training.mask_guided_policy:
    affects: Controls whether topology masks are ignored, softly penalized, hard-filtered, or selected automatically.
    default: auto
    description: Mask policy for logits. Use hard for reviewer-control post-filter baselines.
    enum:
    - auto
    - off
    - soft
    - hard
    label: 'Training: Mask policy'
    path: training.mask_guided_policy
    required_in_modes: []
    required_when: {}
    section: training
    ui:
      group: core
      order: 6
      priority: 1
      tab: general
```

- [ ] **Step 4: Run GREEN test**

Run:

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\application\test_experiment_ui_catalog_fields.py::test_catalog_includes_mask_guided_policy -v
```

Expected: PASS.

---

## Task 4: Add Synthetic `GATv2 + Mask` Presets

**Files:**
- Modify: `outputs/ui/experiment_ui_presets.json`

- [ ] **Step 1: Create train preset from `_Base-UN-f1`**

Add `_Base-MN-f1` by copying `_Base-UN-f1` and changing only these fields:

```json
{
  "preset_name": "_Base-MN-f1",
  "experiment_name": "_Base-MN-f1-42",
  "tracking.run_name": "_Base-MN-f1-42",
  "tracking.tags.model_family": "baseline_gatv2_mask",
  "experiment.mask_guided_enabled": "true",
  "training.mask_guided_policy": "hard",
  "training.mask_guided_apply_in_eval": "true",
  "training.mask_guided_hard_threshold": "1.0",
  "training.mask_guided_soft_penalty": "2.0"
}
```

Keep these fields unchanged from `_Base-UN-f1`:

```json
{
  "model.type": "BaselineGATv2",
  "experiment.structural_mode": "false",
  "training.candidate_contract_mode": "fixed_label",
  "training.dynamic_candidate_contract_enabled": "false"
}
```

Do not change synthetic dataset fields, fraction, split, gateway mode, checkpoint naming convention, seed, or training budget from `_Base-UN-f1`. This keeps the comparison isolated to mask post-filtering.

- [ ] **Step 2: Decide checkpoint strategy**

Use one of two valid strategies:

1. **Inference-only mask control:** train `_Base-UN-f1`, then run drift with the same checkpoint using `_Base-MN-f1-drift`.
2. **Same-preset train/eval:** train `_Base-MN-f1`; training still uses normal CE but validation/test masks may be active.

Recommended for article control: strategy 1, because it isolates inference post-filtering from training.

- [ ] **Step 3: Create drift preset**

Add `_Base-MN-f1-drift` by copying `_Base-UN-f1-drift` and changing only these fields:

```json
{
  "preset_name": "_Base-MN-f1-drift",
  "experiment_name": "_Base-MN-f1-drift-42",
  "tracking.run_name": "_Base-MN-f1-drift-42",
  "mode": "eval_drift",
  "experiment.mask_guided_enabled": "true",
  "training.mask_guided_policy": "hard",
  "training.mask_guided_apply_in_eval": "true"
}
```

Keep `experiment.load_checkpoint` aligned with the checkpoint produced by `_Base-UN-f1`. This makes `_Base-MN-f1-drift` an inference-only mask control over the same trained fixed-vocabulary GATv2 model.

- [ ] **Step 4: Validate JSON and key fields**

Run:

```powershell
.\.venv-modern\Scripts\python.exe - <<'PY'
import json
from pathlib import Path
data = json.loads(Path("outputs/ui/experiment_ui_presets.json").read_text(encoding="utf-8"))
for name in ["_Base-MN-f1", "_Base-MN-f1-drift"]:
    assert name in data, name
    flat = {}
    for group in data[name]["payload"].values():
        if isinstance(group, dict):
            flat.update(group)
    assert flat["model.type"] == "BaselineGATv2"
    assert str(flat["training.candidate_contract_mode"]).lower() == "fixed_label"
    assert str(flat["experiment.mask_guided_enabled"]).lower() == "true"
    assert str(flat["training.mask_guided_policy"]).lower() == "hard"
print("OK")
PY
```

Expected: `OK`.

---

## Task 5: Add Runtime Documentation

**Files:**
- Modify: `docs/GNN_RUNTIME_MVP2_5.MD`
- Modify: `docs/current/project-state.md`

- [ ] **Step 1: Update runtime mask section**

In `docs/GNN_RUNTIME_MVP2_5.MD`, update section `7. Loss, Evaluation, and Mask Application` with this wording:

```markdown
`GATv2 + Mask` is a reviewer-control baseline. It keeps the fixed-vocabulary
`BaselineGATv2` output head over `C_train`, but applies the current topology
mask to logits at inference time. With `training.mask_guided_policy=hard`,
classes outside `allowed_target_mask` are suppressed before softmax/argmax.
This can reduce OOS errors, but it cannot predict topology-native candidates
that do not exist in `C_train`.
```

- [ ] **Step 2: Update current project state**

In `docs/current/project-state.md`, add one bullet under `gnn_runtime`:

```markdown
`GATv2 + Mask` is available as a fixed-vocabulary reviewer-control baseline:
it uses `BaselineGATv2` with inference-time hard topology post-filtering, while
retaining the static `C_train` output head.
```

- [ ] **Step 3: Documentation drift check**

Confirm the implementation final response includes:

```text
Documentation drift check: updated docs/GNN_RUNTIME_MVP2_5.MD, docs/current/project-state.md, configs/ui/config_catalog.yaml
```

---

## Task 6: Focused Verification

**Files:**
- No file modifications.

- [ ] **Step 1: Run mask/evaluator tests**

Run:

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\application\test_evaluator_metrics.py -v
```

Expected: PASS.

- [ ] **Step 2: Run config catalog tests**

Run:

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\application\test_experiment_ui_catalog_fields.py -v
```

Expected: PASS.

- [ ] **Step 3: Run architecture guard**

Run:

```powershell
.\.venv-modern\Scripts\python.exe tools\architecture_guard.py
```

Expected:

```text
[ARCH_GUARD] OK
```

- [ ] **Step 4: Manual experiment smoke sequence**

Run sequence:

1. Train `_Base-UN-f1`.
2. Run `_Base-MN-f1-drift` using the `_Base-UN-f1` checkpoint.
3. Compare against `_Base-UN-f1-drift`, `_Base-MN-f1-drift`, `EOPKG-WI`, and full `EOPKG` synthetic runs.

Expected metric pattern:

```text
GATv2 + Mask OOS < GATv2 OOS
GATv2 + Mask strict/macro F1 <= EOPKG strict/macro F1 on unseen/future regions
GATv2 + Mask fixed_label_* remains limited by C_train
```

---

## Self-Review

### Spec Coverage

- Reviewer-control `GATv2 + Mask`: covered in Tasks 1, 2, 4.
- Forced inference post-filter: covered in Task 1.
- Fixed-vocabulary limitation retained: covered in Tasks 4 and 5.
- Tests: covered in Tasks 1, 2, 3, 6.
- Documentation: covered in Task 5.

### Placeholder Scan

No implementation task contains unspecified behavior. The only run-dependent value is `experiment.load_checkpoint` in `_Base-MN-f1-drift`; it must point to the checkpoint produced by `_Base-UN-f1`.

### Type Consistency

- `training.mask_guided_policy` uses string enum `auto|off|soft|hard`.
- `_resolve_mask_guided_policy()` continues returning `off|soft|hard`.
- `_apply_mask_guided_logits()` requires no signature change.
- `GATv2 + Mask` remains `candidate_contract_mode=fixed_label`.
