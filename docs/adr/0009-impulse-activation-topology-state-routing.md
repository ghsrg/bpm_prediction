# ADR-0009: EOPKGTopologyConditioned Impulse Activation and Topology State Routing

Date: 2026-05-31
Status: Proposed

## Context

The previous implementations of structural GNN guidance in the EOPKG model
family suffered from a fundamental architectural mismatch between the model's
structure and the underlying business process mining methodology:

1. **Late Fusion Mismatch (`ClassAwareStructuralScoring`):** The GNN-encoded structural node features remain completely static relative to the prefix trace execution step. The GNN operates only on the process schema topology, meaning it does not know where the execution pointer currently resides in the graph. The prefix state only enters at the very end in a bilinear score calculation, causing instability and negative transfer under structural concept drift.
2. **Early Fusion/Bypassing Sequence Mismatch (`TopologyStateEncoder`):** Although it projects the execution pointer (`struct_prefix_state_x`) onto the topology before GNN encoding, it completely bypasses the sequential/observed trace encoder. The model loses all instance-level execution history (times, loops, sequence patterns) and overfits to the training topology.

To align the codebase with the dissertation methodology, the next planned
iteration should extend `EOPKGTopologyConditioned`, not `EOPKGGATv2`. The model
must trace the sequential prefix history using the observed encoder, project
the current execution pointer as an "Impulse Activation" onto the structural
GNN initial states, propagate the activation wave through the current
topology-native candidate graph using GNN message passing layers
("inductances"), and perform bilinear/cosine candidate routing over
`candidate_logits [B, K_v]`.

## Decision

Status is `Proposed`; this ADR records a planned hypothesis, not an accepted
runtime decision.

If accepted, we will:
1. Extend `EOPKGTopologyConditioned` with a topology conditioning mode named
   `impulse_activation_routing`. This is not an `EOPKGGATv2.fusion_mode`.
2. Keep the observed sequence branch (observed sequence encoder) intact to capture the dynamic instance-level trace history.
3. Incorporate `struct_prefix_state_x` (specifically the `is_last_event` channel, which acts as the execution pointer) into the initial states of the structural GNN.
4. Run GNN message passing (using GATv2Conv layers) over the topology, propagating the activation signal from the current execution pointer to downstream candidate nodes.
5. Fuse the sequential execution context (`obs_context`) and the dynamically
   propagated structural states (`h_struct`) via bilinear/cosine candidate
   scoring over topology-local candidate nodes.
6. Keep `candidate_logits [B, K_v]` as the primary loss and inference path.
   Fixed-label projection is allowed only for compatibility diagnostics.
7. Defer the integration of the statistics latent space (compressing stats and
   topology via a graph autoencoder/VGAE) to a subsequent plan, keeping the
   initial phase focused on zero-shot routing without stats.

## Consequences

### Positive
- **Methodological Alignment:** The model directly corresponds to the dissertation definition of "Impulse Activation through GNN Inductances" and "Topology State Routing".
- **Zero-Shot Adaptability:** GNN message passing propagates the execution pointer to new downstream tasks or through modified route gates, allowing the GNN to retrieve downstream paths on unseen process versions.
- **Unified Dual-Encoder:** Retains both sequential history (sequence encoder) and dynamic topology conditioning (structural GNN).

### Negative
- **Compute Overhead:** Running GNN message passing dynamically for every execution step (batch size context) instead of caching static topology embeddings. This is mitigated by our O(1) slicing optimization in `ModelTrainer`.
- **Dependency:** Requires `struct_prefix_state_x` in the graph contract.

## Runtime Rules

1. When `model.type = EOPKGTopologyConditioned` and
   `model.topology_conditioning_mode = impulse_activation_routing`, the model
   must require `struct_prefix_state_x` in the contract and fail fast if it is
   missing or invalid.
2. The sequence branch must remain active to calculate `obs_context`.
3. Backward compatibility with other fusion modes (`ClassMeanAttention`, etc.) and the baseline fallback path must be preserved.

## Affected Files

- `EOPKGTopologyConditioned` model source
- [config_catalog.yaml](file:///c:/Users/korsr/PycharmProjects/bpm_prediction/configs/ui/config_catalog.yaml)
- [dissertation_changes.MD](file:///c:/Users/korsr/PycharmProjects/bpm_prediction/docs/dissertation_changes.MD)
- [README.md (ADR)](file:///c:/Users/korsr/PycharmProjects/bpm_prediction/docs/adr/README.md)

## Related

- [dissertation_changes.MD](file:///c:/Users/korsr/PycharmProjects/bpm_prediction/docs/dissertation_changes.MD)
- [project-state.md](file:///c:/Users/korsr/PycharmProjects/bpm_prediction/docs/current/project-state.md)
- [GNN_RUNTIME_MVP2_5.MD](file:///c:/Users/korsr/PycharmProjects/bpm_prediction/docs/GNN_RUNTIME_MVP2_5.MD)
