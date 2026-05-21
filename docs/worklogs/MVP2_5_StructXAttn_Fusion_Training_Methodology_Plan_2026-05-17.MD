# MVP2.5 StructXAttn Fusion And Training Methodology Plan

Date: 2026-05-17
Status: planned
Scope: `model.fusion_mode: StructXAttn`

## Goal

Реалізувати новий structural fusion mode, який інтегрує нормативну топологію не лише на рівні фінальних логітів або pooled context, а на рівні node/token representations observed IG encoder.

Цільова гіпотеза:

- observed IG encoder має формувати репрезентації префікса з урахуванням актуальної структури процесу;
- структура має впливати на hidden state до pooling/classifier;
- модель має бути навчена не ігнорувати structural stream;
- коректна структура має давати нижчий loss, ніж контрольовано зіпсована структура.

## Current Code Audit

### Model

Main file:

```text
src/domain/models/eopkg_models.py
```

Current `EOPKGGATv2` supports:

- `ClassMeanAttention`
- `ClassMeanConcat`
- `ClassAwareStructuralScoring`
- `TopologyStateEncoder`
- `TopologyStateGraphEncoder`
- `StructuralPriorEncoder`

Observed encoder currently:

```text
conv1: GATv2Conv(input_dim -> hidden_dim * 4)
conv2: GATv2Conv(hidden_dim * 4 -> hidden_dim)
pooling_strategy: last_node | global_mean
classifier(obs_context)
```

Structural path currently builds structural node embeddings from:

- `struct_x`
- structural node identity embedding
- optional stats projection scaled by `model.structural_stats_beta`
- `structural_edge_index`
- `structural_edge_weight`

Then it runs:

```text
struct_gnn(struct_nodes, structural_edge_index, structural_edge_weight)
```

Main limitation in current modes:

- `ClassMeanAttention` uses one global `obs_context` query. This is not token-level guidance.
- `ClassAwareStructuralScoring` affects final logits, not observed node representations.
- `StructuralPriorEncoder` adds a pooled structural prior after observed pooling.
- `TopologyStateEncoder` / `TopologyStateGraphEncoder` bypass the observed IG branch.

Therefore no current mode implements:

```text
observed node hidden state -> query
structural node hidden state -> key/value
cross-attention update before final pooling/classifier
```

### Trainer

Main file:

```text
src/application/use_cases/trainer.py
```

Current training supports:

- base CE loss;
- optional mask-guided logits;
- optional structural auxiliary loss for `ClassAwareStructuralScoring`;
- forward diagnostics logging for structural logits, topology state, topology graph, structural prior.

There is no current mechanism that compares correct topology against corrupted topology during training.

### Data Contract

Main contract docs:

```text
docs/DATA_MODEL_MVP2_5.MD
```

Existing fields are enough for MVP StructXAttn:

- `x`
- `edge_index`
- `batch`
- `struct_x`
- `structural_edge_index`
- `structural_edge_weight`
- `struct_node_to_class_index`
- `allowed_target_mask`

No new graph tensor is required for initial implementation.

Important current limitation:

- trainer selects structural payload from the first graph in a batch;
- mixed `(process_version, as_of_snapshot)` batches can use one structural payload for multiple samples;
- this is already tracked as architecture debt: `snapshot_homogeneous_batching`.

StructXAttn should not try to solve this debt in the first implementation. It must document the limitation and keep compatibility with existing batching.

## Core Design

## New Fusion Mode

Canonical name:

```yaml
model.fusion_mode: StructXAttn
```

Internal normalized name:

```text
struct_xattn
```

Supported aliases:

```text
StructXAttn
StructuralCrossAttention
LayerwiseStructuralCrossAttention
DualStreamTokenAttention
```

The canonical user-facing name should remain `StructXAttn`.

## Forward Pass

Initial MVP pass:

```text
observed node features
  -> GATv2 observed encoder
  -> observed node hidden states H_ig [N, D]

struct_x + structural topology
  -> structural node builder
  -> struct_gnn
  -> structural node hidden states H_struct [V, D]

StructXAttn:
  Q = W_Q H_ig
  K = W_K H_struct
  V = W_V H_struct

  A = softmax(Q K^T / sqrt(D))
  C = A V
  H_ig' = LayerNorm(H_ig + gamma * Dropout(C))

pool(H_ig') -> classifier -> logits
```

This is token-level cross-attention because every observed node gets its own query. It is not equivalent to current `ClassMeanAttention`, where one pooled `obs_context` queries the whole structure.

## Layer Placement

Recommended staged implementation:

### Stage 1: post-observed-layer StructXAttn

Inject after `conv2`, before pooling:

```text
conv1 -> conv2 -> StructXAttn -> pooling -> classifier
```

Reason:

- minimal architectural risk;
- keeps observed encoder intact;
- hidden dimension is already `hidden_dim`, so attention dimensions are simple;
- tests can isolate whether structural token guidance changes node states and logits.

Limitation:

Stage 1 does not make `conv1` and `conv2` structurally aware while they perform
message passing over the instance graph. It can only correct or enrich the final
observed node states after both GATv2 layers have already encoded local v1
patterns. Therefore Stage 1 is a safe integration and smoke-validation step, not
the final architectural test of the drift-transfer hypothesis.

### Stage 2: true layerwise StructXAttn

Target architecture after Stage 1 is stable:

```text
conv1 -> StructXAttn_1 -> conv2 -> StructXAttn_2 -> pooling -> classifier
```

This needs separate projection blocks for `hidden_dim * 4` and `hidden_dim`, therefore it is not the first implementation target.

Reason:

- structural guidance enters the observed IG stream during message passing;
- each observed layer can condition its representation on the current process
  topology;
- this is the architecture that actually tests whether macro-structure can guide
  the micro/instance encoder under structural drift.

Stage 2 should be promoted to the primary research architecture if Stage 1
passes basic correctness checks.

Recommended Stage 2 config:

```yaml
model.struct_xattn_layers: after_each_conv
model.struct_xattn_layer1_dim: auto
model.struct_xattn_layer2_dim: auto
model.struct_xattn_share_projections: false
```

Projection policy:

```text
conv1 output dim = hidden_dim * 4
conv2 output dim = hidden_dim
```

Therefore Stage 2 needs per-layer adapters:

```text
StructXAttn_1:
  Q from hidden_dim * 4
  K/V from structural hidden_dim projected to hidden_dim * 4

StructXAttn_2:
  Q from hidden_dim
  K/V from structural hidden_dim
```

Do not share projections between `StructXAttn_1` and `StructXAttn_2` in the
first implementation. Shared projections can be tested later as regularization.

## Batch Semantics

Observed graph batches are PyG-style concatenated node tensors with `batch [N]`.

StructXAttn should process each sample in the batch separately:

```text
for graph_id in unique(batch):
    H_ig_b = H_ig[batch == graph_id]
    H_struct_b = selected structural payload for batch
    H_ig_b' = cross_attention(H_ig_b, H_struct_b)
```

Initial implementation may reuse the currently selected structural payload for the batch. This preserves existing runtime behavior but means StructXAttn inherits the current mixed-snapshot batching limitation.

Future improvement:

```text
snapshot-homogeneous batching
or
per-sample structural payload batching
```

## New Model Config

Add to model config and UI catalog:

```yaml
model.fusion_mode: StructXAttn
model.struct_xattn_layers: post_conv2
model.struct_xattn_heads: 4
model.struct_xattn_dropout: 0.1
model.struct_xattn_scale_init: 0.1
model.struct_xattn_scale_max: 2.0
model.struct_xattn_use_layer_norm: true
model.struct_xattn_use_gate: true
model.struct_xattn_gate_init_bias: -2.0
```

Scale handling:

```text
gamma = clamp(learnable_scale, 0, struct_xattn_scale_max)
```

Gate handling:

```text
gate = sigmoid(W_gate([H_ig || C]) + bias)
H_ig' = LayerNorm(H_ig + gamma * gate * C)
```

Rationale:

- small `scale_init` prevents early negative transfer;
- `scale_max` prevents structural branch domination;
- gate lets the observed branch suppress noisy topology guidance.

## Training Methodology

## Why Extra Methodology Is Required

If model only sees correct topology during training, it can still learn to ignore `H_struct` because observed IG features are often enough for same-version prediction.

To force structural dependency, training must include a contrastive topology objective:

```text
correct topology should produce lower CE than corrupted topology
```

## Correct-vs-Corrupted Topology Loss

For each training batch:

```text
logits_correct = model(prefix, topology_correct)
CE_correct = CE(logits_correct, y)

logits_corrupt = model(prefix, topology_corrupt)
CE_corrupt = CE(logits_corrupt, y)

L_contrastive = relu(margin + CE_correct - CE_corrupt)
L_total = CE_correct + lambda_struct * L_contrastive
```

Interpretation:

- if correct topology is already better by at least `margin`, no extra penalty;
- if corrupted topology is equal or better, the model is penalized;
- this creates a direct gradient pressure to use structural information.

Recommended initial config:

```yaml
training.struct_xattn_contrastive_enabled: true
training.struct_xattn_contrastive_weight: 0.03
training.struct_xattn_contrastive_margin: 0.05
```

The weight should start small. This objective intentionally makes corrupted topology worse, so high weight can destabilize the observed branch.

## Contrastive Loss Stability Guardrails

Reviewer note:

```text
CE-margin contrastive loss is useful, but it can create gradient pathology if
CE_corrupt becomes too low and the model receives a strong penalty through the
same observed encoder weights.
```

This is a valid risk. Do not treat the CE-margin objective as the only possible
training method.

Initial implementation should keep CE-margin as the simplest measurable
objective, but it must include guardrails:

```yaml
training.struct_xattn_contrastive_weight: 0.03
training.struct_xattn_contrastive_margin: 0.05
training.struct_xattn_contrastive_warmup_epochs: 3
training.struct_xattn_contrastive_max_loss: 0.5
training.struct_xattn_gradient_clip_norm: 1.0
```

Recommended smooth variant:

```text
L_contrastive = tau * softplus((margin + CE_correct - CE_corrupt) / tau)
```

Initial value:

```yaml
training.struct_xattn_contrastive_temperature: 0.1
```

Reason:

- `ReLU` creates a hard kink at the margin boundary;
- `softplus` gives smoother gradients;
- clipping prevents one corrupted batch from dominating the epoch.

## Latent Contrastive Alternative

If CE-margin destabilizes training, switch to latent-space topology
discrimination.

Core idea:

```text
same prefix + correct topology -> H_ig_correct'
same prefix + corrupted topology -> H_ig_corrupt'
```

Then train the model so the structural effect is distinguishable before final
classification.

Candidate objective:

```text
z_correct = pool(H_ig_correct')
z_corrupt = pool(H_ig_corrupt')
L_latent = max(0, cosine(z_correct, z_corrupt) - target_similarity)
```

Safer InfoNCE-style option:

```text
anchor = z_correct
positive = stopgrad(z_correct from weak/no-corruption view)
negative = z_corrupt
L_InfoNCE = -log exp(sim(anchor, positive)/tau) /
            (exp(sim(anchor, positive)/tau) + exp(sim(anchor, negative)/tau))
```

Important constraint:

Do not blindly minimize cosine similarity between all hidden states from correct
and corrupted topologies. The same observed prefix still contains the same event
history, so forcing representations to be maximally different can destroy useful
observed semantics.

Recommended order:

1. Implement CE-margin with warmup, softplus, clipping, and metrics.
2. Log latent cosine distance between correct and corrupted representations.
3. Add latent InfoNCE only if CE-margin causes unstable training curves or
   negative transfer.

Additional diagnostics:

```text
training_struct_xattn_correct_corrupt_latent_cosine
training_struct_xattn_correct_corrupt_latent_l2
training_struct_xattn_contrastive_active_rate
training_struct_xattn_contrastive_clipped_rate
```

## Structural Corruption Policy

Corruption must operate on already prepared tensors. It must not rebuild topology or query raw sources.

Initial policy:

```yaml
training.struct_xattn_corruption_policy: edge_drop
training.struct_xattn_corruption_edge_drop_prob: 0.15
training.struct_xattn_corruption_feature_noise_std: 0.0
```

Supported MVP policies:

```text
none
edge_drop
feature_noise
edge_drop_plus_feature_noise
```

Do not implement node label shuffling in the first version. With duplicate activities it can create invalid or misleading negatives.

## Attention Regularization

Attention regularization is optional and should not be part of first acceptance.

Potential later objective:

```text
encourage attention mass on structurally allowed candidate nodes/classes
```

Reason to defer:

- `duplicate_activity_identity_ambiguity` means one activity class can map to multiple structural nodes;
- exact target node may be unobservable from logs;
- an over-strict attention target can punish valid parallel/duplicate branches.

MVP should log attention diagnostics first, then decide whether attention regularization is justified.

## Applicability To Other Fusion Modes

The methodology has two levels.

### General Principle

The principle applies to all structural fusion modes:

```text
If a mode claims to use structure, correct topology should outperform corrupted topology.
```

This can be used as a diagnostic or ablation for every structural mode.

### Full Training Methodology

The full correct-vs-corrupted training objective should be implemented first only for `StructXAttn`.

Reason:

- `StructXAttn` has an explicit structural conditioning path inside observed node representations;
- corrupted topology directly changes hidden states before pooling;
- the contrastive loss has a clear mechanism to train structural dependency.

Mode-specific applicability:

| fusion_mode | Apply full contrastive training now? | Notes |
|---|---:|---|
| `StructXAttn` | yes | Primary target. Correct topology should improve node representations. |
| `ClassAwareStructuralScoring` | partial later | Corruption delta is useful, but it affects final structural logits only. Existing structural aux loss already covers part of this role. |
| `ClassMeanAttention` | diagnostic only initially | Global query bottleneck may make corruption signal weak. |
| `ClassMeanConcat` | diagnostic only initially | Pooled context is weakly identifiable. |
| `StructuralPriorEncoder` | diagnostic only initially | Useful for ablation, but not a token-level guidance method. |
| `TopologyStateEncoder` | no for primary objective | Observed IG branch is bypassed, so it does not train IG to use structure. |
| `TopologyStateGraphEncoder` | no for primary objective | Same reason: topology-only graph branch, not dual-stream token guidance. |

Conclusion:

```text
Implement generic corruption/delta helpers where practical, but enable training loss only for StructXAttn in the first version.
```

## Diagnostics

Add model diagnostics:

```text
last_struct_xattn_context_mean_abs
last_struct_xattn_delta_mean_abs
last_struct_xattn_to_observed_ratio
last_struct_xattn_attention_entropy
last_struct_xattn_scale
last_struct_xattn_gate_mean
```

Add trainer metrics:

```text
training_struct_xattn_context_mean_abs
training_struct_xattn_delta_mean_abs
training_struct_xattn_to_observed_ratio
training_struct_xattn_attention_entropy
training_struct_xattn_scale
training_struct_xattn_gate_mean
```

Add contrastive metrics:

```text
training_struct_xattn_correct_ce
training_struct_xattn_corrupt_ce
training_struct_xattn_correct_vs_corrupt_ce_delta
training_struct_xattn_contrastive_loss
training_struct_xattn_prediction_flip_rate_under_corruption
```

Interpretation:

- `correct_vs_corrupt_ce_delta > 0`: correct topology helps.
- `delta <= 0`: model ignores structure or corrupted topology is not actually harmful.
- `to_observed_ratio ~= 0`: structure is too weak.
- `to_observed_ratio >> 1`: structure dominates and may cause negative transfer.
- low entropy with bad F1: structural overconfidence.

## Implementation Plan

## Task 1: Config Plumbing

Files:

```text
src/domain/models/eopkg_models.py
src/cli.py
configs/ui/config_catalog.yaml
tests/application/test_cli_flatten_config.py
```

Changes:

1. Add `StructXAttn` aliases to fusion mode normalization.
2. Add new model kwargs to `EOPKGGATv2.__init__`.
3. Pass new kwargs through `_build_model_factory_kwargs`.
4. Add UI catalog entries.
5. Add tests that CLI config forwards all StructXAttn model keys.

Acceptance:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/application/test_cli_flatten_config.py -v
```

## Task 2: Model Forward Path

Files:

```text
src/domain/models/eopkg_models.py
tests/domain/test_eopkg_struct_xattn.py
```

Changes:

1. Add StructXAttn modules:

```text
struct_xattn_q_proj
struct_xattn_k_proj
struct_xattn_v_proj
struct_xattn_out_proj
struct_xattn_gate
struct_xattn_norm
```

2. Add `_apply_struct_xattn(node_hidden, batch, h_struct, layer_name)` helper.
3. Implement Stage 1 injection after observed `conv2` and before pooling.
4. Prepare the observed encoder refactor so Stage 2 can inject after `conv1`
   and after `conv2` without changing public model contracts.
4. Preserve current behavior for all existing fusion modes.
5. If `structural_mode=false`, ignore StructXAttn and use observed-only path.
6. If required structural tensors are missing, warn and use observed-only fallback, same as existing structural modes.
7. Record diagnostics before returning logits.

Tests:

```text
test_struct_xattn_forward_shape
test_struct_xattn_structural_mode_false_matches_observed_path
test_struct_xattn_missing_structure_falls_back_to_observed
test_struct_xattn_changing_structure_changes_logits
test_struct_xattn_diagnostics_are_recorded
test_existing_fusion_modes_still_forward
```

Acceptance:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/domain/test_eopkg_struct_xattn.py -v
.\.venv\Scripts\python.exe -m pytest tests/domain/test_eopkg_models_forward.py -v
```

## Task 2B: Stage 2 Layerwise StructXAttn

Files:

```text
src/domain/models/eopkg_models.py
tests/domain/test_eopkg_struct_xattn.py
```

Changes:

1. Add `model.struct_xattn_layers` support:

```text
post_conv2
after_each_conv
```

2. Split observed encoder into explicit layer steps:

```text
x1 = conv1(...)
x1 = activation/dropout
x1 = optional StructXAttn_1(x1, h_struct)
x2 = conv2(...)
x2 = activation/dropout
x2 = optional StructXAttn_2(x2, h_struct)
```

3. Add per-layer attention modules because dimensions differ:

```text
struct_xattn_l1: observed dim hidden_dim * 4
struct_xattn_l2: observed dim hidden_dim
```

4. Add diagnostics by layer:

```text
last_struct_xattn_l1_delta_mean_abs
last_struct_xattn_l2_delta_mean_abs
last_struct_xattn_l1_entropy
last_struct_xattn_l2_entropy
```

Tests:

```text
test_struct_xattn_after_each_conv_forward_shape
test_struct_xattn_after_each_conv_records_layer_metrics
test_struct_xattn_after_each_conv_changes_hidden_states_before_pooling
test_struct_xattn_post_conv2_and_after_each_conv_are_config_distinct
```

Acceptance:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/domain/test_eopkg_struct_xattn.py -v
```

## Task 3: Trainer Diagnostics

Files:

```text
src/application/use_cases/trainer.py
tests/application/test_trainer_forward_stats_logging.py
```

Changes:

1. Extend forward stats accumulator with StructXAttn metrics.
2. Add `_accumulate_struct_xattn_stats`.
3. Call it in train, validation, test, and one-pass drift evaluation paths.
4. Log metrics to tracker with split prefixes.

Acceptance:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/application/test_trainer_forward_stats_logging.py -v
```

## Task 4: Corrupted Topology Builder

Files:

```text
src/application/use_cases/trainer.py
tests/application/test_trainer_struct_xattn_training.py
```

Changes:

1. Add config parsing:

```yaml
training.struct_xattn_contrastive_enabled
training.struct_xattn_contrastive_weight
training.struct_xattn_contrastive_margin
training.struct_xattn_corruption_policy
training.struct_xattn_corruption_edge_drop_prob
training.struct_xattn_corruption_feature_noise_std
```

2. Add `_build_corrupted_structural_contract(contract, epoch, batch_idx)`.
3. Implement `edge_drop` without mutating the original contract.
4. Implement `feature_noise` without changing shapes.
5. Keep corrupted `allowed_target_mask` unchanged.
6. Use deterministic torch generator when seed is available.

Tests:

```text
test_corruption_does_not_mutate_original_contract
test_edge_drop_preserves_edge_index_shape_contract
test_feature_noise_preserves_struct_x_shape
test_corruption_can_be_disabled
```

Acceptance:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/application/test_trainer_struct_xattn_training.py -v
```

## Task 5: Contrastive Topology Loss

Files:

```text
src/application/use_cases/trainer.py
tests/application/test_trainer_struct_xattn_training.py
```

Changes:

1. During training only, after correct forward and base CE:

```text
corrupted_contract = _build_corrupted_structural_contract(...)
corrupt_logits = model(corrupted_contract)
corrupt_effective_logits = same mask policy as correct path
CE_corrupt = CE(corrupt_effective_logits, targets)
L_contrastive = relu(margin + CE_correct - CE_corrupt)
loss = CE_correct + weight * L_contrastive
```

Use the smoothed/clipped implementation by default:

```text
raw_gap = margin + CE_correct - CE_corrupt
L_contrastive_raw = temperature * softplus(raw_gap / temperature)
L_contrastive = clamp(L_contrastive_raw, max=max_loss)
loss = CE_correct + weight * L_contrastive
```

2. Enable this only when:

```text
training.struct_xattn_contrastive_enabled = true
model.fusion_mode = StructXAttn
model.structural_mode = true
```

3. Log contrastive metrics.
4. Do not run corrupted forward during validation, test, or eval_drift by default.
5. Support warmup:

```text
if epoch < struct_xattn_contrastive_warmup_epochs:
    weight_effective = 0
else:
    weight_effective = configured_weight
```

6. Add latent-distance metrics without using them as loss in the first
   implementation.

Tests:

```text
test_contrastive_loss_runs_only_for_struct_xattn
test_contrastive_loss_not_used_in_eval
test_contrastive_metrics_are_logged
test_zero_weight_keeps_base_loss_effectively_unchanged
test_contrastive_loss_uses_warmup
test_contrastive_loss_is_clipped
test_latent_correct_corrupt_distance_metrics_are_logged
```

Acceptance:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/application/test_trainer_struct_xattn_training.py -v
```

## Task 6: Presets

Files:

```text
configs/experiments/
configs/ui/config_catalog.yaml
```

Add initial no-statistics presets:

```text
_XAttn-UN
_XAttn-MN
```

Recommended initial values:

```yaml
project: mvp2_6-test
experiment.structural_mode: true
model.fusion_mode: StructXAttn
model.gnn_type: EOPKGGATv2
model.pooling_strategy: last_node
model.structural_score_mode: bilinear_with_prior
model.struct_xattn_layers: post_conv2
model.struct_xattn_scale_init: 0.1
model.struct_xattn_scale_max: 2.0
training.struct_xattn_contrastive_enabled: true
training.struct_xattn_contrastive_weight: 0.03
training.struct_xattn_contrastive_margin: 0.05
data.train_ratio: 0.25
data.fraction: 0.3
```

After Stage 1 smoke validation, add layerwise presets:

```text
_XAttnL-UN
_XAttnL-MN
```

With:

```yaml
model.struct_xattn_layers: after_each_conv
```

Masked variant:

```yaml
training.mask_guided_enabled: true
```

Unmasked variant:

```yaml
training.mask_guided_enabled: false
```

## Task 7: Documentation

Files:

```text
docs/GNN_RUNTIME_MVP2_5.MD
docs/current/project-state.md
docs/dissertation_changes.MD
docs/current/architecture-debt.md
```

Updates:

1. Add StructXAttn architecture diagram and explanation.
2. Distinguish it from `ClassMeanAttention`.
3. Document training methodology:

```text
correct topology vs corrupted topology
```

4. Document that full methodology is enabled first only for StructXAttn.
5. Add dissertation note:

```text
Simple structural fusion is insufficient; structure must be made causally useful through training objective or perturbation methodology.
```

6. If implementation exposes snapshot batching as a blocker for StructXAttn, reference existing `snapshot_homogeneous_batching` debt instead of creating a duplicate debt item.

## Task 8: Verification

Run targeted tests:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/domain/test_eopkg_struct_xattn.py -v
.\.venv\Scripts\python.exe -m pytest tests/application/test_trainer_struct_xattn_training.py -v
.\.venv\Scripts\python.exe -m pytest tests/application/test_trainer_forward_stats_logging.py -v
.\.venv\Scripts\python.exe -m pytest tests/application/test_cli_flatten_config.py -v
```

Run architecture guard:

```powershell
.\.venv\Scripts\python.exe tools\architecture_guard.py
```

If time allows, run broader regression:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -v
```

## Experimental Validation Plan

## Smoke

Use fraction `0.01` first.

Compare:

```text
_Base-UN
_CAS-UN
_Top-UN
_Pri-UN
_XAttn-UN
```

Required checks:

- run reaches metrics;
- no tensor shape mismatch;
- StructXAttn diagnostics are non-zero;
- contrastive metrics are logged.

## Main

Use:

```yaml
data.train_ratio: 0.25
data.fraction: 0.3
```

Train on v1 and run drift on v1-v4.

Compare:

- same-version strict macro F1;
- drift strict macro F1;
- non-strict macro F1;
- OOS;
- ECE;
- prediction entropy;
- `correct_vs_corrupt_ce_delta`;
- `prediction_flip_rate_under_corruption`.

## Acceptance Criteria

Do not claim drift improvement only from F1.

StructXAttn is promising only if:

```text
correct_vs_corrupt_ce_delta > 0
struct_xattn_to_observed_ratio is non-zero but not dominating
same-version quality does not collapse
OOS/ECE improve or remain stable
drift degradation is not worse than baseline
```

Strong evidence:

```text
v2-v4 strict macro F1 improves over Base-UN and CAS-UN across multiple seeds
```

Weak but useful evidence:

```text
strict F1 unchanged, but OOS/ECE improves and corrupted topology is measurably worse
```

Negative result:

```text
correct_vs_corrupt_ce_delta <= 0 after training
```

This means the model still ignores structure or corruption policy is not meaningful.

## Risks

### Training Cost

Contrastive topology training requires a second forward pass during training.

Expected effect:

```text
training time increases by up to ~1.5x-2x for StructXAttn runs
```

Eval and drift should not become slower unless optional corruption diagnostics are enabled for evaluation.

### Duplicate Activity Ambiguity

StructXAttn attends to structural nodes, not classes. If multiple structural nodes share one activity label, attention can split across duplicates.

Do not force exact target-node attention in the first version.

### Mixed Snapshot Batches

Current trainer can use one structural payload for a batch. This is acceptable for first implementation but must be considered when interpreting results.

If StructXAttn shows promising signal, next engineering task should be:

```text
snapshot-homogeneous batching or per-sample structural payload batching
```

### Negative Transfer

If `struct_xattn_scale` or contrastive loss weight is too high, model can overfit topology and degrade on new versions.

Initial values must be conservative.

## Recommendation

Implement StructXAttn in two phases:

1. `post_conv2` token-level cross-attention with diagnostics and no contrastive loss.
2. `after_each_conv` layerwise StructXAttn.
3. Correct-vs-corrupted topology contrastive training.

Then evaluate:

- baseline effect of architectural fusion;
- difference between late-token and layerwise structural guidance;
- additional effect of training methodology.

Do not enable this methodology globally for all fusion modes. Keep the general corrupted-topology delta as a reusable diagnostic, but the training objective should initially be StructXAttn-only.
