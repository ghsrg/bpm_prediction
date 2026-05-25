# MVP2.5 Topology-Homogeneous Candidate Batching Report

- `date`: 2026-05-25
- `status`: implemented
- `scope`: minimal batching prerequisite for future `candidate_contract_mode=candidate_id`
- `plan`: `docs/worklogs/MVP2_5_Single_Topology_Required_Guard_Plan_2026-05-25.MD`

---

## Summary

Implemented topology-homogeneous mini-batching for the future true
candidate-id prediction path.

The core rule is:

```text
candidate_contract_mode=candidate_id
=> batch topology identity = process_version_idx + stats_snapshot_version_idx
=> one batch contains one topology identity
```

This prevents a `[B, C_v]` candidate axis from mixing candidates from different
process versions or stats snapshots.

---

## Changes

### Trainer Runtime

- Added `training.candidate_contract_mode` parsing:
  - `fixed_label`
  - `fixed_projection`
  - `candidate_id`
- Preserved backward compatibility:
  - missing mode + `dynamic_candidate_contract_enabled=true` maps to
    `fixed_projection`;
  - missing mode otherwise maps to `fixed_label`.
- Replaced the version-only sampler with `_TopologyHomogeneousBatchSampler`.
- For indexable graph datasets and `candidate_contract_mode=candidate_id`,
  DataLoader emits topology-homogeneous batches. This includes in-memory
  datasets and `ShardedGraphDataset` disk-cache sources.
- New sharded graph cache shards record lightweight `topology_segments` so the
  topology sampler can build groups from metadata instead of loading every full
  graph and hydrating structural payloads before inference.
- For `shuffle=True`, groups are shuffled by topology identity and sample order
  inside each group.
- For `shuffle=False`, dataset order is preserved and batches are cut only at
  topology identity boundaries or `batch_size`.
- Added `_data_to_contract()` safety guard:
  - mixed candidate-id batches raise `ValueError`;
  - legacy modes keep current first-structural-payload behavior.

### Config Catalog

Added:

```yaml
training.candidate_contract_mode
training.candidate_batch_topology_policy
```

`group_by_topology` is visible as a reserved value but is not implemented yet.

### Documentation

Updated:

- `docs/current/architecture-debt.md`
- `docs/current/project-state.md`
- `docs/GNN_RUNTIME_MVP2_5.MD`
- `docs/dissertation_changes.MD`

---

## How To Use

For current compatibility experiments:

```yaml
training.dynamic_candidate_contract_enabled: true
```

or:

```yaml
training.candidate_contract_mode: fixed_projection
```

For future true candidate-level experiments:

```yaml
training.candidate_contract_mode: candidate_id
training.candidate_batch_topology_policy: single_topology_required
```

Important: `candidate_id` still does not implement candidate-level CE/eval in
this change. This change only makes the future path batch-safe and runnable
once candidate-level loss/eval is added.

---

## Known Limitations

- `group_by_topology` is not implemented.
- In `candidate_id` mode, topology correctness takes precedence over
  `_ShardAwareRandomSampler`; sharded disk-cache datasets use the same topology
  sampler as other indexable graph sources. This may reduce shard-aware random
  access optimization, but it prevents invalid mixed-topology candidate axes.
- Existing sharded cache entries created before `topology_segments` may still
  require a one-time fallback shard scan. Rebuilding the graph cache writes the
  lightweight topology metadata and avoids this startup scan.
- Candidate-level loss/evaluation and true BPMN node candidate ids are separate
  Stage 2 true tasks.

---

## Verification

Targeted tests passed:

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\application\test_trainer_forward_stats_logging.py::<new candidate batching tests> -v
.\.venv-modern\Scripts\python.exe -m pytest tests\application\test_experiment_ui_catalog_fields.py::<candidate catalog tests> -v
```

Full verification commands should be run before merging:

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\application\test_trainer_forward_stats_logging.py -v
.\.venv-modern\Scripts\python.exe -m pytest tests\application\test_experiment_ui_catalog_fields.py -v
.\.venv-modern\Scripts\python.exe tools\architecture_guard.py
.\.venv-modern\Scripts\python.exe -m pytest tests\ -v
```
