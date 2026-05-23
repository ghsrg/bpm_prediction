# MVP2.5 StructXAttn Fusion Stabilization Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `test-driven-development`
> and `verification-before-completion` during implementation. Execute tasks
> step-by-step. Do not combine architectural stabilization with new training
> methodology changes.

**Goal:** stabilize `model.fusion_mode=StructXAttn` residual fusion so structural
cross-attention can be evaluated without Post-LN scale amplification.

**Architecture:** keep `StructXAttn` as the topology-conditioned token-level
fusion mode, but make the residual merge explicit and configurable. Preserve the
current behavior as a compatibility mode, introduce a stable Pre-LN context
merge for research runs, and add optional delta-ratio clamping plus diagnostics
that show actual structural contribution before and after merge.

**Tech Stack:** Python, PyTorch, PyTorch Geometric, MLflow metrics/traces,
pytest, YAML config catalog.

---

## 1. Problem Statement

Observed run:

```text
run_id: d9d019d3827548cc920554edbf47a75f
fusion_mode: StructXAttn
learning_strategy: topology_conditioned
```

New diagnostics showed:

```text
eval_drift_one_pass_struct_xattn_pre_norm_to_observed_ratio  ~= 0.25
eval_drift_one_pass_struct_xattn_post_norm_to_observed_ratio ~= 6.75
drift_window_oos_confidence_mean avg ~= 0.95
```

**Diagnosis (ukr):**

Поточний residual merge:

```text
delta = scale * gate * context
output = LayerNorm(node_hidden + delta)
```

робить структурний внесок контрольованим до `LayerNorm`, але після
`LayerNorm(node_hidden + delta)` різко роздуває фактичну зміну hidden-state.
Це створює overconfident negative transfer: після structural drift модель
впевнено прогнозує OOS-класи.

## 2. Scope

### In Scope

1. Add configurable `StructXAttn` merge semantics.
2. Add a stable `pre_norm_context` merge mode.
3. Add optional `delta_ratio_max` clamp.
4. Preserve old behavior as `post_norm_residual`.
5. Keep existing diagnostics and add/clarify output delta diagnostics.
6. Update config catalog and runtime docs.
7. Add tests for merge modes, clamp behavior, config wiring, and diagnostics.

### Out of Scope

1. No GIB / InfoNCE / latent contrastive implementation in this change.
2. No side-tuning / GraphBridge implementation in this change.
3. No new `learning_strategy` behavior in this change.
4. No changes to `versioned_zero_shot` orchestration.
5. No changes to topology negative sampler.

**Reason (ukr):** спочатку потрібно прибрати механічний scale artifact. Якщо
одночасно змінити loss methodology, буде неможливо пояснити, що саме вплинуло
на drift metrics.

## 3. Proposed Runtime Contract

New config keys:

```yaml
model.struct_xattn_merge_mode: post_norm_residual
model.struct_xattn_delta_ratio_max: null
```

Allowed merge modes:

```text
post_norm_residual
pre_norm_context
residual_only
```

### `post_norm_residual`

Compatibility mode. Current behavior:

```text
context = attention(Q=node_hidden, K=structural_kv, V=structural_kv)
delta = scale * gate * context
output = LayerNorm(node_hidden + delta)
```

Expected diagnostic pattern:

```text
pre_norm_delta_to_observed_ratio can be small
output_delta_to_observed_ratio can be large
```

Use only for reproducing historical runs.

### `pre_norm_context`

Recommended research mode:

```text
context_norm = LayerNorm(context)
delta = scale * gate * context_norm
delta = clamp_delta_ratio(delta, node_hidden, max_ratio)   # optional
output = node_hidden + delta
```

Important: no `LayerNorm(output)` after residual merge.

Expected diagnostic pattern:

```text
output_delta_to_observed_ratio ~= pre_merge_delta_to_observed_ratio
```

### `residual_only`

Diagnostic ablation:

```text
delta = scale * gate * context
delta = clamp_delta_ratio(delta, node_hidden, max_ratio)   # optional
output = node_hidden + delta
```

This isolates whether context normalization itself matters.

## 4. Metric Semantics

Keep old metrics for compatibility:

```text
struct_xattn_context_mean_abs
struct_xattn_delta_mean_abs
struct_xattn_to_observed_ratio
```

Clarify preferred metrics:

```text
struct_xattn_raw_context_mean_abs
struct_xattn_pre_norm_delta_mean_abs
struct_xattn_post_norm_delta_mean_abs
struct_xattn_raw_to_observed_ratio
struct_xattn_pre_norm_to_observed_ratio
struct_xattn_post_norm_to_observed_ratio
```

After this plan:

- `pre_norm_delta_*` means delta before final assignment to `output`.
- `post_norm_delta_*` means `output - node_hidden`, even if the selected mode
  does not use post-merge LayerNorm.
- For `pre_norm_context`, `post_norm_delta_*` should be close to
  `pre_norm_delta_*`.
- `struct_xattn_to_observed_ratio` remains an alias for
  `post_norm_to_observed_ratio`.

## 5. File Map

### Modify

- `src/domain/models/eopkg_models.py`
  - add constructor args;
  - validate merge mode;
  - add context LayerNorm modules;
  - implement merge helper;
  - implement optional delta ratio clamp;
  - preserve diagnostics.

- `src/cli.py`
  - pass new `model.struct_xattn_*` config keys into `create_model`.

- `configs/ui/config_catalog.yaml`
  - add catalog entries for new config keys.

- `docs/GNN_RUNTIME_MVP2_5.MD`
  - document merge modes and recommended research setting.

- `docs/current/project-state.md`
  - add a short current-state note after implementation.

### Tests

- `tests/domain/test_eopkg_struct_xattn.py`
  - merge mode behavior;
  - delta clamp behavior;
  - diagnostics behavior.

- `tests/application/test_cli_flatten_config.py`
  - config wiring into model kwargs.

- `tests/application/test_experiment_ui_catalog_fields.py`
  - catalog includes new keys.

- `tests/application/test_structural_trace_payload_builder.py`
  - no tensor leakage and new diagnostic fields remain trace-safe.

## 6. Implementation Tasks

### Task 1: Add Failing Tests For Merge Mode Contract

**Files:**

- Modify: `tests/domain/test_eopkg_struct_xattn.py`

Add tests:

```python
def test_struct_xattn_rejects_unknown_merge_mode():
    with pytest.raises(ValueError, match="struct_xattn_merge_mode"):
        _model(struct_xattn_merge_mode="unknown")


def test_struct_xattn_pre_norm_context_avoids_post_merge_layernorm_amplification():
    torch.manual_seed(41)
    model = _model(
        struct_xattn_merge_mode="pre_norm_context",
        struct_xattn_scale_init=0.1,
        struct_xattn_scale_max=0.1,
    )
    model.eval()

    with torch.no_grad():
        model(_contract())

    pre_ratio = float(model.last_struct_xattn_pre_norm_to_observed_ratio.item())
    post_ratio = float(model.last_struct_xattn_post_norm_to_observed_ratio.item())

    assert pre_ratio > 0.0
    assert post_ratio > 0.0
    assert abs(post_ratio - pre_ratio) < 1e-5
```

Run:

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\domain\test_eopkg_struct_xattn.py::test_struct_xattn_rejects_unknown_merge_mode tests\domain\test_eopkg_struct_xattn.py::test_struct_xattn_pre_norm_context_avoids_post_merge_layernorm_amplification -v
```

Expected: fail before implementation.

### Task 2: Implement Merge Mode Parsing

**Files:**

- Modify: `src/domain/models/eopkg_models.py`

Add constructor args:

```python
struct_xattn_merge_mode: str = "post_norm_residual",
struct_xattn_delta_ratio_max: float | None = None,
```

Validate:

```python
self.struct_xattn_merge_mode = str(struct_xattn_merge_mode or "post_norm_residual").strip().lower()
if self.struct_xattn_merge_mode not in {"post_norm_residual", "pre_norm_context", "residual_only"}:
    raise ValueError(
        "Unsupported model.struct_xattn_merge_mode "
        f"'{self.struct_xattn_merge_mode}'. "
        "Available: ['post_norm_residual', 'pre_norm_context', 'residual_only']"
    )
self.struct_xattn_delta_ratio_max = (
    None if struct_xattn_delta_ratio_max is None else float(struct_xattn_delta_ratio_max)
)
if self.struct_xattn_delta_ratio_max is not None and self.struct_xattn_delta_ratio_max <= 0.0:
    raise ValueError("model.struct_xattn_delta_ratio_max must be positive or null.")
```

Run Task 1 tests again.

Expected: unknown mode test passes; behavior test may still fail until merge
helper exists.

### Task 3: Add Context Norm Modules

**Files:**

- Modify: `src/domain/models/eopkg_models.py`

Add modules near current StructXAttn norms:

```python
self.struct_xattn_l1_context_norm = nn.LayerNorm(hidden_dim * 4)
self.struct_xattn_l2_context_norm = nn.LayerNorm(hidden_dim)
```

No behavior change yet.

Run existing smoke test:

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\domain\test_eopkg_struct_xattn.py::test_struct_xattn_forward_shape_and_diagnostics -v
```

Expected: pass.

### Task 4: Extract Residual Merge Helper

**Files:**

- Modify: `src/domain/models/eopkg_models.py`

Add method:

```python
def _clamp_struct_xattn_delta_ratio(
    self,
    *,
    delta: torch.Tensor,
    observed: torch.Tensor,
) -> torch.Tensor:
    if self.struct_xattn_delta_ratio_max is None:
        return delta
    observed_mean = observed.detach().abs().mean().clamp_min(1e-12)
    delta_mean = delta.detach().abs().mean()
    max_delta_mean = observed_mean * float(self.struct_xattn_delta_ratio_max)
    if float(delta_mean.item()) <= float(max_delta_mean.item()):
        return delta
    factor = (max_delta_mean / delta_mean).to(device=delta.device, dtype=delta.dtype)
    return delta * factor


def _merge_struct_xattn_context(
    self,
    *,
    observed: torch.Tensor,
    context: torch.Tensor,
    gate: torch.Tensor | None,
    scale: torch.Tensor,
    residual_norm: nn.LayerNorm,
    context_norm: nn.LayerNorm,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if self.struct_xattn_merge_mode == "pre_norm_context":
        merge_context = context_norm(context)
        delta = scale * merge_context
        if gate is not None:
            delta = delta * gate
        delta = self._clamp_struct_xattn_delta_ratio(delta=delta, observed=observed)
        return observed + delta, delta, delta

    if self.struct_xattn_merge_mode == "residual_only":
        delta = scale * context
        if gate is not None:
            delta = delta * gate
        delta = self._clamp_struct_xattn_delta_ratio(delta=delta, observed=observed)
        return observed + delta, delta, delta

    delta = scale * context
    if gate is not None:
        delta = delta * gate
    delta = self._clamp_struct_xattn_delta_ratio(delta=delta, observed=observed)
    pre_norm_output = observed + delta
    output = residual_norm(pre_norm_output) if self.struct_xattn_use_layer_norm else pre_norm_output
    return output, delta, output - observed
```

Replace inline merge inside `_apply_struct_xattn_layer` with helper call.

Run:

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\domain\test_eopkg_struct_xattn.py -v
```

Expected: all StructXAttn domain tests pass.

### Task 5: Add Delta Ratio Clamp Tests

**Files:**

- Modify: `tests/domain/test_eopkg_struct_xattn.py`

Add:

```python
def test_struct_xattn_delta_ratio_max_clamps_output_delta():
    torch.manual_seed(43)
    model = _model(
        struct_xattn_merge_mode="pre_norm_context",
        struct_xattn_scale_init=1.0,
        struct_xattn_scale_max=1.0,
        struct_xattn_delta_ratio_max=0.05,
    )
    model.eval()

    with torch.no_grad():
        model(_contract())

    ratio = float(model.last_struct_xattn_post_norm_to_observed_ratio.item())
    assert ratio <= 0.0501
```

Run:

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\domain\test_eopkg_struct_xattn.py::test_struct_xattn_delta_ratio_max_clamps_output_delta -v
```

Expected: pass after Task 4.

### Task 6: Wire Config Through CLI

**Files:**

- Modify: `src/cli.py`
- Modify: `tests/application/test_cli_flatten_config.py`

Add to model kwargs construction:

```python
"struct_xattn_merge_mode": str(model_cfg.get("struct_xattn_merge_mode", "post_norm_residual")),
"struct_xattn_delta_ratio_max": model_cfg.get("struct_xattn_delta_ratio_max"),
```

Add test assertions in the existing flatten config test that already covers
`struct_xattn_*`:

```python
assert kwargs["struct_xattn_merge_mode"] == "pre_norm_context"
assert kwargs["struct_xattn_delta_ratio_max"] == 0.3
```

Run:

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\application\test_cli_flatten_config.py -v
```

Expected: pass.

### Task 7: Update Config Catalog

**Files:**

- Modify: `configs/ui/config_catalog.yaml`
- Modify: `tests/application/test_experiment_ui_catalog_fields.py`

Add catalog entries:

```yaml
model.struct_xattn_merge_mode:
  path: model.struct_xattn_merge_mode
  label: StructXAttn merge mode
  type: choice
  default: post_norm_residual
  choices:
    - post_norm_residual
    - pre_norm_context
    - residual_only
  group: StructXAttn
  ui_level: advanced
  active_when:
    model.type: EOPKGGATv2
    model.fusion_mode: StructXAttn
  description: Controls residual merge semantics for structural cross-attention.

model.struct_xattn_delta_ratio_max:
  path: model.struct_xattn_delta_ratio_max
  label: StructXAttn delta ratio max
  type: float_or_null
  default: null
  group: StructXAttn
  ui_level: advanced
  active_when:
    model.type: EOPKGGATv2
    model.fusion_mode: StructXAttn
  description: Optional cap for mean_abs(delta) / mean_abs(observed).
```

Run:

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\application\test_experiment_ui_catalog_fields.py tests\application\test_desktop_ui_field_registry.py -v
```

Expected: pass.

### Task 8: Update Runtime Docs

**Files:**

- Modify: `docs/GNN_RUNTIME_MVP2_5.MD`
- Modify: `docs/current/project-state.md`
- Optionally modify: `docs/dissertation_changes.MD`

Document:

```yaml
model.struct_xattn_merge_mode: pre_norm_context
model.struct_xattn_delta_ratio_max: 0.3
```

Recommended research config after stabilization:

```yaml
model:
  fusion_mode: StructXAttn
  struct_xattn_layers: after_each_conv
  struct_xattn_merge_mode: pre_norm_context
  struct_xattn_scale_init: 0.1
  struct_xattn_scale_max: 0.1
  struct_xattn_use_gate: true
  struct_xattn_gate_init_bias: -2.0
  struct_xattn_delta_ratio_max: 0.3
```

Document interpretation:

- if `pre_norm_to_observed_ratio <= 0.3` and `post_norm_to_observed_ratio` is
  still high, merge is still unstable;
- if OOS confidence remains high after stabilization, the next issue is
  training methodology/loss, not residual scaling;
- if OOS confidence drops but strict-F1 does not improve, topology is safer but
  still not ranking the correct allowed target.

### Task 9: Run Targeted Verification

Run:

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\domain\test_eopkg_struct_xattn.py tests\application\test_cli_flatten_config.py tests\application\test_experiment_ui_catalog_fields.py tests\application\test_desktop_ui_field_registry.py tests\application\test_structural_trace_payload_builder.py -v
```

Expected:

```text
all selected tests pass
```

Run architecture guard:

```powershell
.\.venv-modern\Scripts\python.exe tools\architecture_guard.py
```

Expected:

```text
[ARCH_GUARD] OK
```

### Task 10: Recommended Experiment Matrix

After implementation, run these controlled experiments before changing losses:

```text
A current baseline:
  struct_xattn_merge_mode=post_norm_residual
  struct_xattn_delta_ratio_max=null

B stable merge:
  struct_xattn_merge_mode=pre_norm_context
  struct_xattn_delta_ratio_max=null

C stable bounded merge:
  struct_xattn_merge_mode=pre_norm_context
  struct_xattn_delta_ratio_max=0.3

D stricter bounded merge:
  struct_xattn_merge_mode=pre_norm_context
  struct_xattn_delta_ratio_max=0.15
```

Use the same checkpoint/training setup where possible. If checkpoint shape is
unchanged, old checkpoints remain loadable because only behavior/config changes
are added. However, for final comparison, retrain each variant because changing
merge semantics changes learned optimization dynamics.

Primary metrics:

```text
eval_drift_one_pass_struct_xattn_pre_norm_to_observed_ratio
eval_drift_one_pass_struct_xattn_post_norm_to_observed_ratio
drift_window_oos_confidence_mean
drift_window_test_oos
drift_window_test_ece
drift_window_test_set_nll
drift_window_strict_macro_f1
drift_window_macro_f1
drift_window_strict_error_but_allowed_rate
```

Success criteria:

```text
post_norm_to_observed_ratio no longer explodes above pre_norm_to_observed_ratio
OOS confidence decreases materially on v3/v4
ECE/NLL improve or stop degrading relative to post_norm_residual
strict-F1 does not degrade versus post_norm_residual
```

If these pass but strict-F1 still does not improve, proceed to the next research
stage: latent contrastive/GIB or better topology negatives. Do not continue
tuning `scale` blindly.

## 7. Risks

### Risk: old checkpoints use old semantics

Old checkpoints can load if the new modules do not add required parameters or
if load is non-strict. Adding context norm modules introduces parameters. To
avoid checkpoint incompatibility, use `nn.LayerNorm(..., elementwise_affine=False)`
for context normalization, or handle missing keys explicitly.

Recommended MVP choice:

```python
self.struct_xattn_l1_context_norm = nn.LayerNorm(hidden_dim * 4, elementwise_affine=False)
self.struct_xattn_l2_context_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
```

### Risk: `struct_xattn_use_layer_norm` semantics become confusing

Keep it as compatibility flag for `post_norm_residual`. For
`pre_norm_context`, context normalization is always part of the mode and does
not depend on `struct_xattn_use_layer_norm`.

### Risk: clamp hides training pathology

Clamp is diagnostic/protective, not a replacement for methodology. Always log
ratios and compare clamped versus unclamped runs.

### Risk: `after_each_conv` doubles the merge effect

For `after_each_conv`, clamp applies per layer. The total effect can still be
larger than the per-layer cap. Interpret `l1` and `l2` diagnostics separately.

## 8. Documentation Drift Checklist

Implementation must update:

- `configs/ui/config_catalog.yaml` because new config keys are added.
- `docs/GNN_RUNTIME_MVP2_5.MD` because GNN forward semantics change.
- `docs/current/project-state.md` because current runtime behavior changes.
- `docs/dissertation_changes.MD` if the change affects dissertation narrative.

No ADR is required unless `pre_norm_context` becomes the canonical default for
all StructXAttn research presets.

## 9. Handoff Notes

This plan deliberately stabilizes the fusion mechanism only. It does not claim
that StructXAttn will outperform baseline after the change. The expected first
benefit is safer behavior under drift:

```text
lower OOS confidence
lower ECE/NLL
controlled structural contribution
cleaner evidence for the next learning-methodology experiment
```

