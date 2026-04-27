# ADR-0005: Snapshot-Homogeneous Batching

Date: 2026-04-27
Status: Proposed

## Context

Stage 4.2 forwards structural statistics through `struct_x`,
`structural_edge_index`, and `structural_edge_weight`. These tensors represent a
repository snapshot context, not only a sample-local observed graph.

Current PyG batching can mix samples from different stats snapshots in one batch.
The implemented Option-A workaround selects the first graph structural payload
for forward and emits a warning when snapshot identities are mixed.

Current behavior is implemented in:

- `src/application/use_cases/trainer.py`
  - `_warn_if_mixed_snapshot_versions`
  - `_select_structural_payload_for_forward`
  - `_data_to_contract`
- `tests/application/test_trainer_forward_stats_logging.py`

## Decision

Target contract:

One structural forward context must correspond to one stats snapshot identity.

Mixed snapshot batches must not be silently treated as semantically equivalent.

The current Option-A first-graph selection is accepted only as a transitional
runtime workaround, not as the final research-grade batching contract.

## Consequences

Positive:

1. strict temporal experiments become easier to defend,
2. structural statistics cannot accidentally represent multiple temporal states
   in one forward context,
3. forward diagnostics become semantically cleaner,
4. drift experiments can reason about one `knowledge_version` and `as_of` per
   structural batch.

Negative:

1. batching may become less efficient if samples must be bucketed,
2. DataLoader and sampler logic becomes more complex,
3. strict behavior may expose missing or uneven snapshot coverage earlier.

## Runtime Rules

Current transitional behavior:

1. mixed snapshot versions may appear in one PyG batch,
2. runtime emits a `UserWarning`,
3. structural payload is selected from the first graph in the batch.

Target behavior:

1. research-grade runs should use snapshot-homogeneous batches,
2. implementation may use sampler bucketing by snapshot identity,
3. implementation may split mixed batches into homogeneous micro-batches before
   forward,
4. strict research mode may fail fast on mixed structural snapshot contexts,
5. warning-only behavior must remain explicitly marked as transitional.

## Affected Files

- `src/application/use_cases/trainer.py`
- `tests/application/test_trainer_forward_stats_logging.py`
- future sampler or batching module

## Related

- `docs/worklogs/MVP2_5_Stage4_2_OptionA_Unbatch_Fix_Report.MD`
- `docs/worklogs/MVP2_5_Canonical_Doc_Sync_and_Architecture_Debt_2026-04-24.MD`
- `docs/worklogs/MVP2_5_Dissertation_Alignment_and_Blocking_Debt_Analysis_2026-03-21.MD`
- `docs/LLD_MVP2_5.MD`

