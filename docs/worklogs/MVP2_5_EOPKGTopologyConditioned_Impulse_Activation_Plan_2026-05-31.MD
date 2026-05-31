# EOPKGTopologyConditioned Impulse Activation Routing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` (recommended) or
> `superpowers:executing-plans` to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `EOPKGTopologyConditioned` with topology-native Impulse
Activation Routing so the model predicts over the candidate nodes of the
current process version, not over the fixed training activity vocabulary.

**Architecture:** Keep the observed prefix encoder active to produce
`obs_context`. Project the current execution pointer from
`struct_prefix_state_x` into the topology node states as an impulse, propagate
that impulse through the current BPMN topology via structural GNN message
passing, and score each topology-local candidate node with bilinear/cosine
matching against `obs_context`. The primary loss and inference path remains
`candidate_logits [B, K_v]`.

**Tech Stack:** PyTorch 2.x, PyTorch Geometric, existing
`EOPKGTopologyConditioned` candidate contract.

---

## Scope Correction

This plan replaces the earlier incorrect framing that targeted `EOPKGGATv2`.

Do **not** implement this as another `EOPKGGATv2.fusion_mode` that returns:

```text
observed_logits [B, C_train] + structural_logits [B, C_train]
```

That would return to the fixed-vocabulary failure mode discovered during the
candidate identity audit. The corrected implementation target is:

```text
model.type: EOPKGTopologyConditioned
training.candidate_contract_mode: candidate_id
training.candidate_identity_mode: topology_native
```

Primary output:

```text
candidate_logits: [B, K_v]
candidate_ids: topology-local BPMN node ids for the current process version
candidate_labels: labels used for XES target matching
candidate_class_index: compatibility mapping to train activity vocab, -1 for unseen labels
```

Compatibility projection to fixed activity labels is allowed only for legacy
metrics and reporting. It must not be the main loss or inference path.

---

## Research Framing

This plan aligns with three external research directions:

1. **DFG/GNN Predictive Process Monitoring:** graph representations can reduce
   the information loss of sequence-only PPM on complex processes with loops and
   non-sequential relations.
2. **Adaptive/Heterogeneous Graph PPM:** next-activity prediction benefits when
   graph structure is an active modelling factor, not a passive late feature.
3. **PPM Resilience Under Change:** business-valid PPM needs an update path for
   evolving process versions; topology-native candidate scoring is the zero-shot
   part of that lifecycle, while later `versioned_zero_shot` training handles
   incremental logs.

---

## Proposed Changes

### Domain Model

#### [MODIFY] `src/domain/models/eopkg_topology_conditioned.py` or existing `EOPKGTopologyConditioned` source

Add an `impulse_activation_routing` candidate scoring mode inside
`EOPKGTopologyConditioned`.

The implementation must:

1. Keep the observed prefix encoder active:

   ```text
   node_hidden = observed_encoder(prefix_graph)
   obs_context = pool(node_hidden)
   ```

2. Require `struct_prefix_state_x` when impulse routing is enabled:

   ```text
   struct_prefix_state_x: [B, K_v, F_prefix_state]
   ```

3. Build dynamic topology node initial states:

   ```text
   h0_j = base_candidate_features_j
          + known_or_unseen_label_embedding_j
          + impulse_proj(struct_prefix_state_x[:, j, :])
   ```

4. Run structural message passing over the current topology:

   ```text
   h_nodes = struct_encoder(h0, structural_edge_index, edge_weight?)
   ```

5. Score topology-local candidates:

   ```text
   candidate_logits[:, j] = score(obs_context, h_nodes[:, j])
   ```

6. Return `CandidatePredictionOutput` with:

   ```text
   candidate_logits [B, K_v]
   candidate_ids
   candidate_labels
   candidate_class_index
   candidate_is_unseen
   diagnostics
   ```

### Configuration

#### [MODIFY] `configs/ui/config_catalog.yaml`

Add or confirm these keys:

```yaml
model.candidate_scoring: cosine | bilinear
model.topology_conditioning_mode: static_candidates | impulse_activation_routing
model.impulse_activation_enabled: true | false
model.impulse_state_channels:
  - is_last_event
  - prefix_executed_count_log1p
  - prefix_recency_norm
model.impulse_scale_init: 0.1
model.impulse_scale_max: 2.0
```

Recommended default for this experimental branch:

```yaml
model.type: EOPKGTopologyConditioned
training.candidate_contract_mode: candidate_id
training.candidate_identity_mode: topology_native
model.topology_conditioning_mode: impulse_activation_routing
```

Do not add `ImpulseActivationRouting` to `model.fusion_mode`; this is not an
`EOPKGGATv2` fusion mode.

### Trainer / Loss

#### [MODIFY] `src/application/use_cases/trainer.py`

Keep existing candidate-id training:

```text
candidate_target_mask_from_labels(...)
candidate_set_cross_entropy_from_mask(...)
```

Add diagnostics for impulse routing:

```text
impulse_activation_mean_abs
impulse_activation_max_abs
impulse_to_base_node_ratio
candidate_logit_valid_margin
candidate_oos_rate
candidate_target_missing_rate
```

The trainer must fail fast when:

```text
candidate_contract_mode = candidate_id
candidate_identity_mode = topology_native
model.topology_conditioning_mode = impulse_activation_routing
```

and any required field is missing:

```text
struct_prefix_state_x
candidate_ids
candidate_labels
candidate_target_mask
```

### Tracing

#### [MODIFY] `src/application/services/structural_trace_payload_builder.py`

Extend selected traces with topology-native impulse diagnostics:

```text
top_k_candidate_ids
top_k_candidate_labels
top_k_candidate_logits
target_candidate_ids
last_event_candidate_id
impulse_activation_norm
candidate_path_distance_from_last_event
candidate_is_allowed_by_current_topology
```

Flat MLflow span attributes must include:

```text
fusion_family = topology_conditioned
topology_conditioning_mode = impulse_activation_routing
candidate_identity_mode = topology_native
strict_correct
candidate_oos
target_missing
```

---

## Task 1: Regression Tests for Topology-Native Impulse Contract

**Files:**

- Modify: `tests/domain/test_eopkg_topology_conditioned.py`

- [ ] **Step 1: Write failing tests**

Add tests that verify:

1. `EOPKGTopologyConditioned` with impulse routing returns
   `candidate_logits [B, K_v]`, not `[B, C_train]`.
2. Moving `is_last_event` in `struct_prefix_state_x` changes candidate logits.
3. Unseen candidate labels (`candidate_class_index = -1`) remain scoreable.
4. Missing `struct_prefix_state_x` fails fast only when impulse routing is
   enabled.

- [ ] **Step 2: Verify red**

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\domain\test_eopkg_topology_conditioned.py -k "impulse" -v
```

Expected: tests fail because impulse routing is not implemented.

---

## Task 2: Implement Dynamic Impulse Node Initialization

**Files:**

- Modify: `src/domain/models/eopkg_topology_conditioned.py` or current model file

- [ ] **Step 1: Add model config fields**

Add fields for:

```text
topology_conditioning_mode
impulse_scale_init
impulse_scale_max
impulse_state_channels
```

- [ ] **Step 2: Add impulse projection**

Implement:

```text
impulse_proj: Linear(F_prefix_state -> hidden_dim)
impulse_scale: learnable bounded scalar
```

- [ ] **Step 3: Combine base candidate state and impulse**

Use a stable residual merge:

```text
base = candidate_base_projection(...)
impulse = impulse_scale * impulse_proj(struct_prefix_state_x)
h0 = layer_norm(base + impulse)
```

Track diagnostics before and after normalization.

- [ ] **Step 4: Run domain tests**

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\domain\test_eopkg_topology_conditioned.py -v
```

---

## Task 3: Structural Propagation Over Current Topology

**Files:**

- Modify: `src/domain/models/eopkg_topology_conditioned.py` or current model file

- [ ] **Step 1: Reuse existing structural encoder where possible**

The structural encoder must run over the current topology-local candidate graph:

```text
h_nodes = struct_encoder(h0, structural_edge_index, edge_weight)
```

Do not rebuild topology inside the model.

- [ ] **Step 2: Keep topology-homogeneous batch assumption**

This Stage 3 remains compatible with the current
`candidate_batch_topology_policy=group_by_topology` implementation. Do not
introduce mixed-topology candidate axes inside one batch.

- [ ] **Step 3: Add diagnostics**

Record:

```text
candidate_dynamic_count
topology_identity
structural_edge_count
impulse_to_base_node_ratio
```

---

## Task 4: Candidate Scoring Without Fixed-Vocab Pooling

**Files:**

- Modify: `src/domain/models/eopkg_topology_conditioned.py` or current model file
- Modify: `src/application/use_cases/trainer.py` if output plumbing is needed

- [ ] **Step 1: Score candidates directly**

Supported scoring:

```text
cosine(obs_context, h_nodes[j]) / temperature
bilinear(obs_context, h_nodes[j])
```

- [ ] **Step 2: Return candidate-native output**

Return:

```text
CandidatePredictionOutput(candidate_logits=[B, K_v], ...)
```

- [ ] **Step 3: Do not pool to class logits as primary path**

No primary:

```text
LogSumExp(candidate_logits -> class_logits)
observed_logits + structural_logits
```

Projection to fixed labels may remain only as compatibility diagnostics.

---

## Task 5: Candidate-Level Training and Metrics

**Files:**

- Modify: `src/application/use_cases/trainer.py`
- Modify: `src/application/services/topology_conditioned_learning.py`

- [ ] **Step 1: Use candidate target masks**

Training loss:

```text
L_candidate = candidate_set_cross_entropy_from_mask(candidate_logits, target_mask)
```

- [ ] **Step 2: Preserve allowed-set topology penalty**

Keep candidate-level topology penalties, but report them separately from
mask-guided inference:

```text
candidate_allowed_flow_penalty
candidate_oos_rate
candidate_target_in_mask_rate
```

- [ ] **Step 3: Add missing-target guard**

Fail or warn according to config if:

```text
candidate_target_missing_rate > threshold
```

Recommended threshold for research runs:

```yaml
training.candidate_target_missing_rate_max: 0.05
```

---

## Task 6: Config Catalog and Desktop UI

**Files:**

- Modify: `configs/ui/config_catalog.yaml`
- Modify: `tools/experiment_ui.py`
- Modify: `tools/desktop_ui/` if the new UI has the relevant field registry

- [ ] **Step 1: Add config keys**

Add all new `model.*` and `training.*` keys from this plan.

- [ ] **Step 2: Place fields in UI**

Recommended UI grouping:

```text
Experiment Run -> Model / Topology Conditioned
Advanced -> Diagnostics / Candidate Contract
```

- [ ] **Step 3: Validate catalog fields**

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\application\test_experiment_ui_catalog_fields.py -q
```

---

## Task 7: Documentation Updates

**Files:**

- Modify: `docs/GNN_RUNTIME_MVP2_5.MD`
- Modify: `docs/GNN_LEARNING_STRATEGY.MD`
- Modify: `docs/GNN_LEARNING_METODOLOGY.MD`
- Modify: `docs/current/project-state.md`
- Modify: `docs/dissertation_changes.MD`
- Modify: `docs/current/architecture-debt.md` if debt changes

- [ ] **Step 1: Document the corrected architecture**

Make clear:

```text
Impulse Activation Routing belongs to EOPKGTopologyConditioned.
It is not an EOPKGGATv2 fusion mode.
It predicts in topology-native candidate space [B, K_v].
```

- [ ] **Step 2: Add dissertation note**

Record that this is a planned hypothesis to test whether dynamic topology
routing improves zero-shot transfer across process versions.

- [ ] **Step 3: Update debt**

If Stage 3 keeps topology-homogeneous batching only, retain the long-term debt
for true mixed-topology batches.

---

## Verification Plan

Run targeted checks first:

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\domain\test_eopkg_topology_conditioned.py -v
.\.venv-modern\Scripts\python.exe -m pytest tests\application\test_trainer_topology_conditioned_learning.py -v
.\.venv-modern\Scripts\python.exe -m pytest tests\application\test_experiment_ui_catalog_fields.py -q
.\.venv-modern\Scripts\python.exe tools\architecture_guard.py
```

Full verification before merge:

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\ -v
```

---

## Expected Experiment

Minimal first run:

```yaml
model.type: EOPKGTopologyConditioned
model.topology_conditioning_mode: impulse_activation_routing
training.candidate_contract_mode: candidate_id
training.candidate_identity_mode: topology_native
training.candidate_batch_topology_policy: group_by_topology
experiment.gateway_mode: collapse_for_prediction
experiment.statistic_enabled: false
```

Primary comparison:

```text
Baseline sequence model
EOPKGTopologyConditioned static candidate scoring
EOPKGTopologyConditioned impulse activation routing
```

Primary metrics:

```text
candidate_strict_accuracy / strict_macro_f1 compatibility
candidate_oos_rate
candidate_target_missing_rate
candidate_nll / candidate_ece
drift degradation slope by version distance
```

The expected scientific value is not immediate perfect zero-shot performance.
The first success criterion is evidence that changing the current topology
changes candidate ranking in the right direction without collapsing to the
fixed training vocabulary.
