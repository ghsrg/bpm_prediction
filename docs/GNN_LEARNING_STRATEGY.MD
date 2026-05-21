# GNN Learning Strategy

Machine-readable and human-readable specification for GNN learning strategies
in MVP2.5.

This document defines how model training is organized. It is intentionally
separate from `fusion_mode`: fusion describes how topology enters the model
forward pass, while learning strategy describes how the model is trained to use
that topology.

---

## Metadata

- `status`: implemented
- `audience`: human-and-agent
- `source_of_truth`: true
- `language_policy`: keys and section headers in English, explanations in Ukrainian
- `last_updated`: 2026-05-21
- `related_docs`:
  - `docs/GNN_RUNTIME_MVP2_5.MD`
  - `docs/EVF_MVP2_5.MD`
  - `docs/VARIABLES.MD`
  - `docs/current/project-state.md`
  - `docs/current/architecture-debt.md`

---

## Purpose

Цей документ фіксує поділ між трьома рівнями:

```text
fusion_mode = як topology входить у forward model
learning_strategy = як організовано навчання, loss-и, negatives, sampling, retention
experiment.mode = який життєвий або оцінювальний сценарій виконується
```

Це потрібно, щоб окремо оцінювати:

- вплив архітектури злиття структури (`fusion_mode`);
- вплив методології навчання (`learning_strategy`);
- вплив сценарію запуску або evaluation protocol (`experiment.mode`).

Основна дослідницька гіпотеза:

```text
P(next_activity | observed_prefix, topology_version)
```

Модель має навчитися використовувати topology як умову прогнозу. Коли
з'являється нова версія процесу `vN`, topology `vN` може бути доступна раніше,
ніж накопичиться достатньо логів `vN` для навчання.

---

## Config Contract

Canonical config key:

```yaml
training:
  learning_strategy: standard | topology_conditioned
```

### `standard`

`standard` має повністю відтворювати поточну поведінку навчання.

Правило:

```text
training.learning_strategy=standard must not change existing train/eval results.
```

### `topology_conditioned`

`topology_conditioned` є новою методологією навчання, яка змушує модель
використовувати topology як meaningful condition, а не як декоративний
додатковий tensor.

Implementation status:

```text
implemented in trainer-level services and ModelTrainer train loop
```

Current implementation adds config parsing, a topology payload pool from
already prebuilt train graph tensors, train-time `L_allowed_set`,
known-version wrong-topology negatives, same-version physical `drop_edges`
negatives, version-weighted CE/rehearsal retention, and tracker diagnostics.

Required guard:

```text
if training.learning_strategy == topology_conditioned
and experiment.structural_mode == false:
    fail-fast
```

Baseline observed-only experiments therefore use:

```yaml
experiment:
  structural_mode: false
training:
  learning_strategy: standard
```

Structure-aware experiments may use:

```yaml
experiment:
  structural_mode: true
training:
  learning_strategy: standard | topology_conditioned
```

---

## Fusion Mode Compatibility

`topology_conditioned` is a learning strategy, not a fusion mode. It can wrap
multiple structural `fusion_mode` values, but expected effect depends on how
strongly the selected fusion exposes topology to trainable prediction behavior.

### `supported`

These modes have a meaningful topology channel for the proposed strategy:

```text
StructXAttn
ClassAwareStructuralScoring
TopologyStateEncoder
TopologyStateGraphEncoder
```

Interpretation:

- `StructXAttn`: topology can affect observed hidden states;
- `ClassAwareStructuralScoring`: topology can affect per-class structural logits;
- `TopologyStateEncoder`: topology-state path predicts class logits directly;
- `TopologyStateGraphEncoder`: topology graph is the primary prediction graph.

### `allowed_but_weak`

These modes may run with `topology_conditioned`, but the topology exposure is
weaker:

```text
ClassMeanAttention
ClassMeanConcat
StructuralPriorEncoder
```

Runtime behavior:

```text
allowed_but_weak modes should run, but log warning/tag:
learning_strategy_support_level=allowed_but_weak
```

Research interpretation:

Це ablation: чи допомагає training methodology навіть тоді, коли topology
потрапляє в модель через слабший global/pooled context.

### `forbidden`

```text
experiment.structural_mode=false
```

For `topology_conditioned`, this must fail-fast.

---

## Topology-Conditioned Loss Contract

Target total loss:

```text
L_total =
  L_ce_correct
  + lambda_wrong * L_wrong_version_negative
  + lambda_drop * L_drop_edges_negative
  + lambda_allowed * L_allowed_set
  + lambda_retention * L_retention
```

### `L_ce_correct`

Supervised loss on the correct topology:

```text
L_ce_correct = CE(model(prefix, topology_correct), y)
```

This remains the core predictive objective.

### `L_wrong_version_negative`

Purpose:

Модель має давати гірший prediction loss, якщо для того самого prefix подано
topology іншої вже відомої версії.

Formula:

```text
logits_correct = model(prefix, topology_correct)
logits_wrong = model(prefix, topology_wrong_known_version)

L_wrong =
  max(0, margin_wrong + CE_correct - CE_wrong)
```

Activation rule:

```text
active only when another known topology version exists in the current phase
```

Leakage guard:

```text
future topology must never be used as wrong-version negative
```

Mixed-batch rule:

```text
prefer version-homogeneous training mini-batches for this loss
```

If a mixed-version batch reaches the trainer, the wrong-version term must not
be silently disabled. The first implementation may skip only this term for that
batch, but it must log explicit diagnostics. Per-version subgroup contrast is
deferred because the current model contract carries one structural payload per
batch.

### `L_drop_edges_negative`

Purpose:

Модель має реагувати на локальне псування поточної topology, навіть якщо в
поточній фазі доступна тільки одна версія.

Formula:

```text
logits_dropped = model(prefix, topology_correct_with_dropped_edges)

L_drop =
  max(0, margin_drop + CE_correct - CE_dropped)
```

First implementation policy:

```yaml
edge_corruption_policy: drop_edges
```

This is physical edge corruption in the first implementation. Soft Edge Drop
would be preferable for BPM reachability, but it requires the structural GNN
forward path to consume `structural_edge_weight`; current `EOPKGGATv2`
structural message passing does not.

Rules:

- only edges are dropped;
- never drop all edges;
- node ids are not changed;
- `struct_node_to_class_index` is not changed;
- activity labels are not changed;
- stats features are not changed.

### `L_allowed_set`

Purpose:

Train-time topology consistency loss. It is not the same as inference-time
masking.

Formula:

```text
p_allowed = sum softmax(logits_correct)[j] for j where allowed_mask[j] = 1
L_allowed = -log(clamp(p_allowed, eps, 1.0))
```

Interpretation:

Модель штрафується, якщо probability mass виходить за межі allowed topology
set. Вона не штрафується за вибір іншої допустимої активності у паралельній або
неоднозначній ділянці.

Required guard:

```text
if learning_strategy=topology_conditioned
and allowed_set_loss_enabled=true
and allowed_target_mask is missing or invalid:
    fail-fast
```

### `L_retention`

First implementation policy:

```text
version-weighted CE/rehearsal
```

No distillation, teacher model, or stored old logits in the first version.

Example:

```text
L_ce_correct =
  mean(version_weight(v_i) * CE_i)
```

Default conceptual weights:

```text
current version -> high weight
recent previous version -> medium weight
old version -> low weight
obsolete version -> optional zero
```

Business interpretation:

Це `controlled forgetting`, не повна заборона забування. Старі версії ще деякий
час можуть бути активними, але поступово втрачають бізнесову вагу.

---

## Negative Topology Sampler

Target sampler:

```yaml
topology_conditioning:
  negative_sampler: version_safe_mixed
  wrong_version_negative_enabled: true
  drop_edges_negative_enabled: true
```

### Version-Safe Rule

At phase `t`:

```text
known_topologies = topology versions already known to the business at phase t
future_topologies = topology versions available only for offline eval holdout
```

Allowed for training negatives:

```text
known_topologies only
```

Forbidden for training negatives:

```text
future_topologies
```

### Phase Examples

For `known=[v1]`:

```text
wrong_version_topology: inactive
drop_edges_same_version(v1): active
```

For `known=[v1, v2]`:

```text
wrong_version_topology: v1 <-> v2
drop_edges_same_version(v1 or v2): active
```

For `known=[v1, v2, v3]`:

```text
wrong_version_topology: among v1/v2/v3
drop_edges_same_version: active
```

---

## Allowed Set Loss vs Mask-Guided Inference

These mechanisms must remain separate.

### `L_allowed_set`

```text
train-time loss shaping
does not change inference logits directly
can be active when experiment.mask_guided_enabled=false
```

### `experiment.mask_guided_enabled`

```text
runtime logit constraint or penalty
directly affects prediction surface when enabled
independent research hypothesis
```

Recommended experiment matrix:

```text
A. standard + no mask
B. standard + mask
C. topology_conditioned + no mask
D. topology_conditioned + mask
```

Interpretation:

- `mask` measures constraint effect;
- `topology_conditioned` measures learning effect;
- combined runs measure whether effects are additive or interfering.

---

## Versioned Zero-Shot Protocol

`versioned_zero_shot` is an experiment/lifecycle mode, not a model architecture.

Target config:

```yaml
experiment:
  mode: versioned_zero_shot
  versioned_zero_shot:
    phase_policy: rolling_operational
    adaptation_mode: continue_from_previous_checkpoint
    rehearsal_enabled: true
    rehearsal_ratio: 0.05
    rehearsal_source: previous_versions
    future_tail_eval: true
```

### Rolling Operational Semantics

This protocol should simulate a business stream:

```text
new version appears -> topology is available first
logs of the new version accumulate gradually
old versions continue for a while, then fade out
```

It should not behave like purely academic cumulative training over all old logs.

### Phase Lifecycle

For each new version `vN`:

1. `pre_adaptation_zero_shot_eval`
   - checkpoint from previous phase;
   - topology `vN` is used for inference;
   - logs `vN` were not used for training;
   - measures zero-shot value of topology.
2. `adaptation_train`
   - continue from previous checkpoint;
   - train on new logs `vN` plus rehearsal buffer from previous versions;
   - use `topology_conditioned` losses if configured.
3. `post_adaptation_eval`
   - evaluate current and recent known versions;
   - in offline simulation, evaluate future tail as holdout only.
4. `advance_phase`
   - add `vN` to known versions;
   - adapted checkpoint becomes starting point for next phase.

### Future Tail Evaluation

At each phase, evaluate all future versions as zero-shot holdout:

```text
known=[v1]       -> future tail: v2, v3, v4
known=[v1,v2]    -> future tail: v3, v4
known=[v1,v2,v3] -> future tail: v4
```

Guard:

```text
future versions may be used for eval inference only
future versions must not affect train, negatives, early stopping, checkpoint selection, or hyperparameter tuning
```

### Version Distance

First implementation:

```text
version_distance = ordinal(eval_version) - ordinal(latest_known_version)
```

Example:

```text
known latest = v1, eval = v4 -> version_distance = 3
```

If versions do not have parseable ordinal ids, see
`version_order_metadata_policy` in `docs/current/architecture-debt.md`.

---

## Checkpoint Selection

Future-tail evaluation must never affect checkpoint selection.

Recommended first policy:

```yaml
versioned_zero_shot:
  checkpoint_selection_policy: weighted_known_validation
  checkpoint_metric: validation_macro_f1
  checkpoint_smoothing_window: 3
  checkpoint_min_epoch: 5
  tie_breakers:
    - validation_strict_macro_f1
    - validation_ece
    - validation_oos_rate
  future_eval_affects_checkpoint: false
```

Interpretation:

- checkpoint is selected only from known-version validation;
- validation weights align with retention policy;
- future versions are evaluation-only;
- raw early spikes should not dominate checkpoint choice.

---

## Metrics And Diagnostics

### Prediction Quality

```text
macro_f1
strict_macro_f1
accuracy
top3_hit_rate
nll
ece
```

### Drift And Zero-Shot Transfer

```text
eval_relation_to_train = seen | future_zero_shot
version_distance
known_versions
eval_version
zero_shot_tail_macro_f1
zero_shot_tail_strict_macro_f1
degradation_from_latest_seen
```

### Mask And Topology Consistency

```text
target_in_mask_rate
pred_in_mask_rate
oos_rate
strict_error_but_allowed_rate
allowed_set_nll
allowed_mass_mean
```

### Topology Dependency

```text
correct_ce
wrong_version_ce
drop_edges_ce
wrong_version_ce_delta = wrong_version_ce - correct_ce
drop_edges_ce_delta = drop_edges_ce - correct_ce
wrong_version_margin_violation_rate
drop_edges_margin_violation_rate
```

Interpretation:

```text
wrong_version_ce_delta > 0 means correct topology helps more than wrong topology
drop_edges_ce_delta > 0 means model reacts to topology damage
```

### Retention And Forgetting

```text
version_weight_current
version_weight_recent
version_weight_old
seen_version_macro_f1_by_version
forgetting_delta_by_version
```

---

## Research Success Criteria

Do not use a single best `strict_macro_f1` spike as the main success criterion.
It can select unstable early epochs.

Primary research criterion:

```text
topology_conditioned has lower degradation slope by version_distance
than standard or baseline under the same fusion/mask/seed setup
```

Report slopes for:

```text
macro_f1
strict_macro_f1
oos_rate
ece
pred_in_mask_rate
target_in_mask_rate
allowed_mass
```

Comparison rules:

```text
same fusion_mode
same mask setting
same seeds
standard vs topology_conditioned
```

---

## Open Debt And Future Extensions

### `version_order_metadata_policy`

Current design assumes parseable ordinal versions such as `v1`, `v2`, `v3`.

Future target:

```text
derive version order from metadata:
effective_from | deployment_ts | topology_version_seq
```

### `negative_sampler_extended_corruption`

First implementation only supports:

```text
drop_edges
```

Future target may add:

```text
rewire_edges
node_feature_corruption
gateway_policy_corruption
soft_edge_drop
mixed_batch_subgroup_contrast
```

These must not be added until the first `drop_edges` and `wrong_version`
experiments demonstrate useful topology dependency.

`soft_edge_drop` additionally requires an edge-weight-aware structural encoder.
`mixed_batch_subgroup_contrast` requires per-version microbatch splitting or
per-sample structural payload contracts.

### `retention_distillation`

First implementation uses version-weighted CE/rehearsal only.

Future target may add:

```text
teacher checkpoint distillation
logit replay
representation replay
```

This is deferred to avoid overcomplicating the first dissertation-grade
methodology.

---

## Maintenance Rule

When implementation changes learning behavior:

1. update this file,
2. update `docs/GNN_RUNTIME_MVP2_5.MD` if model forward or fusion behavior changes,
3. update `docs/EVF_MVP2_5.MD` if evaluation semantics change,
4. update `configs/ui/config_catalog.yaml` for any new config key,
5. update `docs/current/project-state.md` for current-state behavior,
6. update `docs/current/architecture-debt.md` for deferred strategy debt.
