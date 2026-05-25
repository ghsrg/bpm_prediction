# Topology-Homogeneous Candidate Batch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans`
> or `superpowers:subagent-driven-development` to implement this plan
> task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build topology-homogeneous batching for true candidate-id
training/evaluation so `[B, C_v]` candidate axes are never mixed inside one
forward/loss call.

**Architecture:** Keep existing first-structural-payload behavior for legacy
and compatibility modes. For `training.candidate_contract_mode=candidate_id`,
the DataLoader must produce homogeneous mini-batches grouped by topology
identity. A fail-fast guard remains as a safety net. Full in-trainer
`group_by_topology` execution for arbitrary mixed batches is explicitly
deferred as architecture debt.

**Tech Stack:** Python 3.11, PyTorch Geometric `Data`/`DataLoader`,
`ModelTrainer`, existing `process_version_idx` and `stats_snapshot_version_idx`
batch metadata.

---

## 1. Scope

Included:

- Add config keys:
  - `training.candidate_contract_mode`
  - `training.candidate_batch_topology_policy`
- Replace/extend the current version-only sampler with a topology-homogeneous
  sampler keyed by `process_version_idx + stats_snapshot_version_idx`.
- Use topology-homogeneous batching automatically for
  `candidate_contract_mode=candidate_id`.
- Preserve drift/eval sample order by using consecutive homogeneous chunks
  when `shuffle=False`.
- Implement `single_topology_required` fail-fast behavior as a safety net.
- Preserve old behavior for:
  - `fixed_label`
  - `fixed_projection`
  - `training.dynamic_candidate_contract_enabled=true`
- Add tests that candidate-id loaders do not emit mixed `process_version_idx`
  or mixed `stats_snapshot_version_idx` batches.
- Add tests that a manually mixed candidate-id batch still fails.
- Add config catalog entries.
- Update docs/debt.

Excluded:

- In-trainer `group_by_topology` sub-batch execution for already mixed batches.
- Candidate-level CE/evaluator/traces.
- Candidate BPMN node id propagation.
- Semantic prototypes or retrieval head.

---

## 2. Runtime Contract

New config:

```yaml
training:
  candidate_contract_mode: fixed_label | fixed_projection | candidate_id
  candidate_batch_topology_policy: single_topology_required | group_by_topology
```

Compatibility rules:

```text
missing candidate_contract_mode:
  if dynamic_candidate_contract_enabled=true -> fixed_projection
  else -> fixed_label

candidate_contract_mode=fixed_label:
  use model.forward()

candidate_contract_mode=fixed_projection:
  use model.forward_candidate() and project to [B, C_train]

candidate_contract_mode=candidate_id:
  requires homogeneous topology batch before candidate-id loss/eval is allowed
```

Initial policy behavior:

```text
single_topology_required:
  DataLoader emits topology-homogeneous mini-batches;
  trainer raises if a mixed batch still reaches _data_to_contract()

group_by_topology:
  future extension;
  trainer would split arbitrary mixed batches internally
```

Topology identity for the guard:

```text
process_version_idx + stats_snapshot_version_idx
```

Rationale:

- `process_version_idx` captures topology version.
- `stats_snapshot_version_idx` captures stats/as-of snapshot when present.
- A change in either value can change the structural candidate axis or
  structural payload semantics.

If `stats_snapshot_version_idx` is missing, batching and guard still use
`process_version_idx`. If `process_version_idx` is missing in `candidate_id`
mode, fail with a clear mapping/config error instead of assuming homogeneous
batching.

Order policy:

```text
training shuffle=True:
  group all samples by topology identity, shuffle group order and samples
  inside each group.

eval/drift shuffle=False:
  preserve dataset order and split only where topology identity changes or
  batch_size is reached. This prevents drift-window record reordering.
```

---

## 3. Files

Modify:

- `src/application/use_cases/trainer.py`
- `configs/ui/config_catalog.yaml`
- `tests/application/test_trainer_forward_stats_logging.py`
- `tests/application/test_experiment_ui_catalog_fields.py`
- `docs/current/architecture-debt.md`
- `docs/GNN_RUNTIME_MVP2_5.MD`
- `docs/current/project-state.md`
- `docs/dissertation_changes.MD`

Add:

- Optional, only if existing test file becomes too large:
  `tests/application/test_trainer_candidate_batch_topology.py`

---

## 4. Task 1: Config Parsing And Mode Normalization

**Files:**

- Modify: `src/application/use_cases/trainer.py`
- Test: `tests/application/test_trainer_forward_stats_logging.py`

- [ ] **Step 1: Add failing test for default compatibility mode**

Add to `tests/application/test_trainer_forward_stats_logging.py`:

```python
def test_candidate_contract_mode_defaults_to_fixed_projection_when_dynamic_enabled():
    trainer = ModelTrainer(
        xes_adapter=_DummyAdapter(),
        prefix_policy=_DummyPrefixPolicy(),
        graph_builder=_DummyGraphBuilder(),
        model=_TrainableBinaryModel(),
        log_path="in_memory.xes",
        config={
            "mode": "train",
            "device": "cpu",
            "show_progress": False,
            "tqdm_disable": True,
            "dynamic_candidate_contract_enabled": True,
        },
    )

    assert trainer.candidate_contract_mode == "fixed_projection"
```

- [ ] **Step 2: Add failing test for explicit candidate-id config**

```python
def test_candidate_contract_mode_accepts_candidate_id_and_default_topology_policy():
    trainer = ModelTrainer(
        xes_adapter=_DummyAdapter(),
        prefix_policy=_DummyPrefixPolicy(),
        graph_builder=_DummyGraphBuilder(),
        model=_TrainableBinaryModel(),
        log_path="in_memory.xes",
        config={
            "mode": "train",
            "device": "cpu",
            "show_progress": False,
            "tqdm_disable": True,
            "candidate_contract_mode": "candidate_id",
        },
    )

    assert trainer.candidate_contract_mode == "candidate_id"
    assert trainer.candidate_batch_topology_policy == "single_topology_required"
```

- [ ] **Step 3: Run failing tests**

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\application\test_trainer_forward_stats_logging.py::test_candidate_contract_mode_defaults_to_fixed_projection_when_dynamic_enabled tests\application\test_trainer_forward_stats_logging.py::test_candidate_contract_mode_accepts_candidate_id_and_default_topology_policy -v
```

Expected: fail because `candidate_contract_mode` is not parsed yet.

- [ ] **Step 4: Implement parsing in `ModelTrainer.__init__`**

Place immediately after `self.dynamic_candidate_contract_enabled`:

```python
self.candidate_contract_mode = str(config.get("candidate_contract_mode", "") or "").strip().lower()
if not self.candidate_contract_mode:
    self.candidate_contract_mode = (
        "fixed_projection" if self.dynamic_candidate_contract_enabled else "fixed_label"
    )
if self.candidate_contract_mode not in {"fixed_label", "fixed_projection", "candidate_id"}:
    raise ValueError(
        "training.candidate_contract_mode must be fixed_label, fixed_projection, or candidate_id."
    )

self.candidate_batch_topology_policy = str(
    config.get("candidate_batch_topology_policy", "single_topology_required")
    or "single_topology_required"
).strip().lower()
if self.candidate_batch_topology_policy not in {"single_topology_required", "group_by_topology"}:
    raise ValueError(
        "training.candidate_batch_topology_policy must be single_topology_required or group_by_topology."
    )
if self.candidate_contract_mode == "candidate_id" and self.candidate_batch_topology_policy == "group_by_topology":
    raise ValueError(
        "training.candidate_batch_topology_policy=group_by_topology is planned but not implemented."
    )
```

- [ ] **Step 5: Run tests**

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\application\test_trainer_forward_stats_logging.py::test_candidate_contract_mode_defaults_to_fixed_projection_when_dynamic_enabled tests\application\test_trainer_forward_stats_logging.py::test_candidate_contract_mode_accepts_candidate_id_and_default_topology_policy -v
```

Expected: pass.

---

## 5. Task 2: Topology-Homogeneous Batch Sampler

**Files:**

- Modify: `src/application/use_cases/trainer.py`
- Test: `tests/application/test_trainer_forward_stats_logging.py`

- [ ] **Step 1: Add failing sampler tests**

Add to `tests/application/test_trainer_forward_stats_logging.py`:

```python
def test_topology_homogeneous_sampler_groups_by_version_and_snapshot_when_shuffling():
    samples = [
        _sample(0, snapshot_idx=7),
        _sample(1, snapshot_idx=8),
        _sample(2, snapshot_idx=7),
        _sample(3, snapshot_idx=8),
    ]
    samples[0].process_version_idx = torch.tensor([0], dtype=torch.long)
    samples[1].process_version_idx = torch.tensor([0], dtype=torch.long)
    samples[2].process_version_idx = torch.tensor([1], dtype=torch.long)
    samples[3].process_version_idx = torch.tensor([0], dtype=torch.long)

    sampler = _TopologyHomogeneousBatchSampler(samples, batch_size=2, shuffle=True, seed=42)

    for batch_indices in sampler:
        identities = {
            (
                int(samples[idx].process_version_idx.view(-1)[0].item()),
                int(samples[idx].stats_snapshot_version_idx.view(-1)[0].item()),
            )
            for idx in batch_indices
        }
        assert len(identities) == 1
```

```python
def test_topology_homogeneous_sampler_preserves_order_when_not_shuffling():
    samples = [
        _sample(0, snapshot_idx=7),
        _sample(1, snapshot_idx=7),
        _sample(2, snapshot_idx=8),
        _sample(3, snapshot_idx=8),
        _sample(4, snapshot_idx=7),
    ]
    for sample in samples:
        sample.process_version_idx = torch.tensor([0], dtype=torch.long)

    sampler = _TopologyHomogeneousBatchSampler(samples, batch_size=3, shuffle=False, seed=42)

    assert list(sampler) == [[0, 1], [2, 3], [4]]
```

- [ ] **Step 2: Run failing sampler tests**

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\application\test_trainer_forward_stats_logging.py::test_topology_homogeneous_sampler_groups_by_version_and_snapshot_when_shuffling tests\application\test_trainer_forward_stats_logging.py::test_topology_homogeneous_sampler_preserves_order_when_not_shuffling -v
```

Expected: fail because `_TopologyHomogeneousBatchSampler` does not exist.

- [ ] **Step 3: Replace version-only sampler with topology sampler**

In `src/application/use_cases/trainer.py`, replace
`_VersionHomogeneousBatchSampler` with:

```python
class _TopologyHomogeneousBatchSampler(Sampler[List[int]]):
    """Yield mini-batches grouped by process_version_idx and stats snapshot."""

    def __init__(self, dataset: Sequence[Data], *, batch_size: int, shuffle: bool, seed: int = 42) -> None:
        self._dataset = dataset
        self._batch_size = max(1, int(batch_size))
        self._shuffle = bool(shuffle)
        self._base_seed = int(seed)
        self._iteration = 0
        self._groups = self._build_groups(dataset)

    @staticmethod
    def _graph_key(graph: Data) -> str:
        raw_version = getattr(graph, "process_version_idx", None)
        if isinstance(raw_version, torch.Tensor) and raw_version.numel() > 0:
            version = f"v:{int(raw_version.view(-1)[0].item())}"
        else:
            version = "v:__missing__"
        raw_snapshot = getattr(graph, "stats_snapshot_version_idx", None)
        if isinstance(raw_snapshot, torch.Tensor) and raw_snapshot.numel() > 0:
            snapshot = f"s:{int(raw_snapshot.view(-1)[0].item())}"
        else:
            snapshot = "s:__none__"
        return f"{version}|{snapshot}"

    @classmethod
    def _build_groups(cls, dataset: Sequence[Data]) -> Dict[str, List[int]]:
        groups: Dict[str, List[int]] = {}
        for idx, graph in enumerate(dataset):
            groups.setdefault(cls._graph_key(graph), []).append(int(idx))
        return groups

    def __iter__(self) -> Iterable[List[int]]:
        if not self._shuffle:
            current_key: str | None = None
            current_batch: List[int] = []
            for idx, graph in enumerate(self._dataset):
                key = self._graph_key(graph)
                if current_batch and (key != current_key or len(current_batch) >= self._batch_size):
                    yield current_batch
                    current_batch = []
                current_key = key
                current_batch.append(int(idx))
            if current_batch:
                yield current_batch
            return

        rng = random.Random(self._base_seed + self._iteration)
        self._iteration += 1
        keys = list(self._groups)
        rng.shuffle(keys)
        for key in keys:
            indices = list(self._groups[key])
            rng.shuffle(indices)
            for start in range(0, len(indices), self._batch_size):
                yield indices[start : start + self._batch_size]

    def __len__(self) -> int:
        if self._shuffle:
            return sum(int(math.ceil(len(indices) / float(self._batch_size))) for indices in self._groups.values())
        total = 0
        last_key: str | None = None
        current_size = 0
        for graph in self._dataset:
            key = self._graph_key(graph)
            if current_size and (key != last_key or current_size >= self._batch_size):
                total += 1
                current_size = 0
            last_key = key
            current_size += 1
        return total + (1 if current_size else 0)
```

- [ ] **Step 4: Run sampler tests**

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\application\test_trainer_forward_stats_logging.py::test_topology_homogeneous_sampler_groups_by_version_and_snapshot_when_shuffling tests\application\test_trainer_forward_stats_logging.py::test_topology_homogeneous_sampler_preserves_order_when_not_shuffling -v
```

Expected: pass.

---

## 6. Task 3: Wire Candidate-ID DataLoader Batching

**Files:**

- Modify: `src/application/use_cases/trainer.py`
- Test: `tests/application/test_trainer_forward_stats_logging.py`

- [ ] **Step 1: Add failing DataLoader test**

```python
def test_candidate_id_dataloader_emits_topology_homogeneous_batches():
    samples = [
        _sample(0, snapshot_idx=7),
        _sample(1, snapshot_idx=8),
        _sample(2, snapshot_idx=7),
        _sample(3, snapshot_idx=8),
    ]
    samples[0].process_version_idx = torch.tensor([0], dtype=torch.long)
    samples[1].process_version_idx = torch.tensor([0], dtype=torch.long)
    samples[2].process_version_idx = torch.tensor([1], dtype=torch.long)
    samples[3].process_version_idx = torch.tensor([0], dtype=torch.long)

    trainer = ModelTrainer(
        xes_adapter=_DummyAdapter(),
        prefix_policy=_DummyPrefixPolicy(),
        graph_builder=_DummyGraphBuilder(),
        model=_TrainableBinaryModel(),
        log_path="in_memory.xes",
        config={
            "mode": "train",
            "device": "cpu",
            "show_progress": False,
            "tqdm_disable": True,
            "candidate_contract_mode": "candidate_id",
            "batch_size": 2,
        },
    )

    loader = trainer._create_data_loader(samples, shuffle=True)

    for batch in loader:
        identities = set(
            zip(
                batch.process_version_idx.view(-1).long().tolist(),
                batch.stats_snapshot_version_idx.view(-1).long().tolist(),
            )
        )
        assert len(identities) == 1
```

- [ ] **Step 2: Run failing DataLoader test**

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\application\test_trainer_forward_stats_logging.py::test_candidate_id_dataloader_emits_topology_homogeneous_batches -v
```

Expected: fail because `_create_data_loader_from_source()` only uses the
version-homogeneous sampler for topology-conditioned learning strategy, not
for `candidate_id`.

- [ ] **Step 3: Wire sampler in `_create_data_loader_from_source()`**

Change sampler selection:

```python
use_topology_homogeneous_batch_sampler = (
    isinstance(source, Sequence)
    and not isinstance(source, (str, bytes))
    and (
        self.candidate_contract_mode == "candidate_id"
        or (self.learning_strategy_config.is_topology_conditioned and shuffle)
    )
)
if use_topology_homogeneous_batch_sampler:
    batch_sampler = _TopologyHomogeneousBatchSampler(
        source,
        batch_size=self.batch_size,
        shuffle=bool(shuffle),
        seed=self.seed,
    )
    effective_shuffle = False
```

Keep `_ShardAwareRandomSampler` behavior unchanged for `ShardedGraphDataset`.
For sharded candidate-id datasets, rely on the fail-fast guard first; sharded
topology-homogeneous batching is future work unless the shard index exposes
per-sample metadata cheaply.

- [ ] **Step 4: Run DataLoader test**

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\application\test_trainer_forward_stats_logging.py::test_candidate_id_dataloader_emits_topology_homogeneous_batches -v
```

Expected: pass.

---

## 7. Task 4: Homogeneous Topology Guard Safety Net

**Files:**

- Modify: `src/application/use_cases/trainer.py`
- Test: `tests/application/test_trainer_forward_stats_logging.py`

- [ ] **Step 1: Add failing test for manually mixed process versions**

Add to `tests/application/test_trainer_forward_stats_logging.py`:

```python
def test_candidate_id_mode_rejects_mixed_process_version_batch():
    sample_a = _sample(0, snapshot_idx=7)
    sample_b = _sample(1, snapshot_idx=7)
    sample_a.process_version_idx = torch.tensor([0], dtype=torch.long)
    sample_b.process_version_idx = torch.tensor([1], dtype=torch.long)
    loader = DataLoader([sample_a, sample_b], batch_size=2, shuffle=False)
    batch = next(iter(loader))

    trainer = ModelTrainer(
        xes_adapter=_DummyAdapter(),
        prefix_policy=_DummyPrefixPolicy(),
        graph_builder=_DummyGraphBuilder(),
        model=_TrainableBinaryModel(),
        log_path="in_memory.xes",
        config={
            "mode": "train",
            "device": "cpu",
            "show_progress": False,
            "tqdm_disable": True,
            "candidate_contract_mode": "candidate_id",
        },
    )

    with pytest.raises(ValueError, match="candidate_id.*mixed topology"):
        trainer._data_to_contract(batch)
```

- [ ] **Step 2: Add failing test for manually mixed stats snapshots**

```python
def test_candidate_id_mode_rejects_mixed_stats_snapshot_batch():
    sample_a = _sample(0, snapshot_idx=7)
    sample_b = _sample(1, snapshot_idx=8)
    sample_a.process_version_idx = torch.tensor([0], dtype=torch.long)
    sample_b.process_version_idx = torch.tensor([0], dtype=torch.long)
    loader = DataLoader([sample_a, sample_b], batch_size=2, shuffle=False)
    batch = next(iter(loader))

    trainer = ModelTrainer(
        xes_adapter=_DummyAdapter(),
        prefix_policy=_DummyPrefixPolicy(),
        graph_builder=_DummyGraphBuilder(),
        model=_TrainableBinaryModel(),
        log_path="in_memory.xes",
        config={
            "mode": "train",
            "device": "cpu",
            "show_progress": False,
            "tqdm_disable": True,
            "candidate_contract_mode": "candidate_id",
        },
    )

    with pytest.raises(ValueError, match="candidate_id.*mixed topology"):
        trainer._data_to_contract(batch)
```

- [ ] **Step 3: Add passing test for homogeneous candidate-id batch**

```python
def test_candidate_id_mode_accepts_homogeneous_topology_batch():
    sample_a = _sample(0, snapshot_idx=7)
    sample_b = _sample(1, snapshot_idx=7)
    sample_a.process_version_idx = torch.tensor([0], dtype=torch.long)
    sample_b.process_version_idx = torch.tensor([0], dtype=torch.long)
    loader = DataLoader([sample_a, sample_b], batch_size=2, shuffle=False)
    batch = next(iter(loader))

    trainer = ModelTrainer(
        xes_adapter=_DummyAdapter(),
        prefix_policy=_DummyPrefixPolicy(),
        graph_builder=_DummyGraphBuilder(),
        model=_TrainableBinaryModel(),
        log_path="in_memory.xes",
        config={
            "mode": "train",
            "device": "cpu",
            "show_progress": False,
            "tqdm_disable": True,
            "candidate_contract_mode": "candidate_id",
        },
    )

    contract = trainer._data_to_contract(batch)

    assert contract.get("batch_topology_unique_count") == 1
```

- [ ] **Step 4: Run failing tests**

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\application\test_trainer_forward_stats_logging.py::test_candidate_id_mode_rejects_mixed_process_version_batch tests\application\test_trainer_forward_stats_logging.py::test_candidate_id_mode_rejects_mixed_stats_snapshot_batch tests\application\test_trainer_forward_stats_logging.py::test_candidate_id_mode_accepts_homogeneous_topology_batch -v
```

Expected: mixed tests fail because no guard exists; homogeneous test fails
because the contract does not expose `batch_topology_unique_count`.

- [ ] **Step 5: Add helper methods to `ModelTrainer`**

Place near `_warn_if_mixed_snapshot_versions`:

```python
def _tensor_values_as_int_tuple(self, value: Any) -> tuple[int, ...]:
    if isinstance(value, torch.Tensor):
        return tuple(int(item) for item in value.view(-1).long().cpu().tolist())
    if isinstance(value, (list, tuple)):
        result: list[int] = []
        for item in value:
            try:
                result.append(int(item))
            except (TypeError, ValueError):
                continue
        return tuple(result)
    return ()

def _candidate_batch_topology_identities(self, data: Data) -> tuple[str, ...]:
    versions = self._tensor_values_as_int_tuple(getattr(data, "process_version_idx", None))
    snapshots = self._tensor_values_as_int_tuple(getattr(data, "stats_snapshot_version_idx", None))
    if not versions:
        return ()
    if snapshots and len(snapshots) == len(versions):
        return tuple(f"version:{version}|snapshot:{snapshot}" for version, snapshot in zip(versions, snapshots))
    return tuple(f"version:{version}|snapshot:__none__" for version in versions)

def _guard_candidate_batch_topology(self, data: Data, contract: GraphTensorContract) -> None:
    if self.candidate_contract_mode != "candidate_id":
        return
    identities = self._candidate_batch_topology_identities(data)
    if not identities:
        raise ValueError(
            "candidate_contract_mode=candidate_id requires process_version_idx in each batch "
            "to verify topology homogeneity."
        )
    unique = tuple(sorted(set(identities)))
    contract["batch_topology_unique_count"] = int(len(unique))  # type: ignore[typeddict-unknown-key]
    contract["batch_topology_identities"] = list(unique)  # type: ignore[typeddict-unknown-key]
    if self.candidate_batch_topology_policy == "single_topology_required" and len(unique) > 1:
        raise ValueError(
            "candidate_contract_mode=candidate_id received a mixed topology batch; "
            f"unique_topology_identities={list(unique)}. Use versioned batching/sampling "
            "or implement candidate_batch_topology_policy=group_by_topology."
        )
```

- [ ] **Step 6: Call guard in `_data_to_contract()`**

At the end of `_data_to_contract()`, after `process_version_labels` are added
and before `return contract`:

```python
self._guard_candidate_batch_topology(data, contract)
return contract
```

- [ ] **Step 7: Run tests**

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\application\test_trainer_forward_stats_logging.py::test_candidate_id_mode_rejects_mixed_process_version_batch tests\application\test_trainer_forward_stats_logging.py::test_candidate_id_mode_rejects_mixed_stats_snapshot_batch tests\application\test_trainer_forward_stats_logging.py::test_candidate_id_mode_accepts_homogeneous_topology_batch -v
```

Expected: pass.

---

## 8. Task 5: Preserve Legacy Behavior

**Files:**

- Modify: `tests/application/test_trainer_forward_stats_logging.py`

- [ ] **Step 1: Add regression test for old mixed snapshot warning**

Existing test `test_data_to_contract_warns_when_batch_has_mixed_snapshot_versions`
must continue to pass when `candidate_contract_mode` is not `candidate_id`.
Add an explicit assertion:

```python
assert trainer.candidate_contract_mode == "fixed_label"
```

- [ ] **Step 2: Add regression test for fixed_projection**

```python
def test_fixed_projection_mode_does_not_reject_mixed_topology_batch():
    sample_a = _sample(0, snapshot_idx=7)
    sample_b = _sample(1, snapshot_idx=8)
    sample_a.process_version_idx = torch.tensor([0], dtype=torch.long)
    sample_b.process_version_idx = torch.tensor([1], dtype=torch.long)
    loader = DataLoader([sample_a, sample_b], batch_size=2, shuffle=False)
    batch = next(iter(loader))

    trainer = ModelTrainer(
        xes_adapter=_DummyAdapter(),
        prefix_policy=_DummyPrefixPolicy(),
        graph_builder=_DummyGraphBuilder(),
        model=_TrainableBinaryModel(),
        log_path="in_memory.xes",
        config={
            "mode": "train",
            "device": "cpu",
            "show_progress": False,
            "tqdm_disable": True,
            "candidate_contract_mode": "fixed_projection",
        },
    )

    with pytest.warns(UserWarning, match="Mixed stats snapshots in one batch detected"):
        contract = trainer._data_to_contract(batch)

    assert contract.get("batch_topology_unique_count") is None
```

- [ ] **Step 3: Run regression tests**

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\application\test_trainer_forward_stats_logging.py::test_data_to_contract_warns_when_batch_has_mixed_snapshot_versions tests\application\test_trainer_forward_stats_logging.py::test_fixed_projection_mode_does_not_reject_mixed_topology_batch -v
```

Expected: pass.

---

## 9. Task 6: Config Catalog

**Files:**

- Modify: `configs/ui/config_catalog.yaml`
- Modify: `tests/application/test_experiment_ui_catalog_fields.py`

- [ ] **Step 1: Add failing catalog tests**

Add to `tests/application/test_experiment_ui_catalog_fields.py`:

```python
def test_catalog_includes_candidate_contract_mode():
    catalog = yaml.safe_load(Path("configs/ui/config_catalog.yaml").read_text(encoding="utf-8"))["fields"]
    field = catalog["training.candidate_contract_mode"]

    assert field["enum"] == ["fixed_label", "fixed_projection", "candidate_id"]


def test_catalog_includes_candidate_batch_topology_policy():
    catalog = yaml.safe_load(Path("configs/ui/config_catalog.yaml").read_text(encoding="utf-8"))["fields"]
    field = catalog["training.candidate_batch_topology_policy"]

    assert field["enum"] == ["single_topology_required", "group_by_topology"]
    assert field["default"] == "single_topology_required"
```

- [ ] **Step 2: Run failing tests**

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\application\test_experiment_ui_catalog_fields.py::test_catalog_includes_candidate_contract_mode tests\application\test_experiment_ui_catalog_fields.py::test_catalog_includes_candidate_batch_topology_policy -v
```

Expected: fail because catalog fields are missing.

- [ ] **Step 3: Add catalog entries**

Add near `training.dynamic_candidate_contract_enabled`:

```yaml
  training.candidate_contract_mode:
    path: training.candidate_contract_mode
    section: training
    label: "Candidate contract mode"
    description: "Prediction/loss contract. fixed_label uses the legacy fixed activity head; fixed_projection uses dynamic candidates projected back to fixed labels; candidate_id requires homogeneous topology batches and is reserved for true candidate-level training/evaluation."
    default: fixed_label
    enum:
      - fixed_label
      - fixed_projection
      - candidate_id
    ui:
      tab: advanced
      group: training
      order: 3194

  training.candidate_batch_topology_policy:
    path: training.candidate_batch_topology_policy
    section: training
    label: "Candidate batch topology policy"
    description: "Batch topology policy for candidate_id mode. single_topology_required fails mixed topology batches; group_by_topology is reserved for a future implementation."
    default: single_topology_required
    enum:
      - single_topology_required
      - group_by_topology
    ui:
      tab: advanced
      group: training
      order: 3195
```

- [ ] **Step 4: Run catalog tests**

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\application\test_experiment_ui_catalog_fields.py::test_catalog_includes_candidate_contract_mode tests\application\test_experiment_ui_catalog_fields.py::test_catalog_includes_candidate_batch_topology_policy -v
```

Expected: pass.

---

## 10. Task 7: Architecture Debt And Docs

**Files:**

- Modify: `docs/current/architecture-debt.md`
- Modify: `docs/GNN_RUNTIME_MVP2_5.MD`
- Modify: `docs/current/project-state.md`
- Modify: `docs/dissertation_changes.MD`

- [ ] **Step 1: Update architecture debt**

Add under `dynamic_candidate_prediction_contract` or near
`snapshot_homogeneous_batching`:

```markdown
### candidate_batch_topology_grouping

- `status`: active
- `priority`: P0
- `current_behavior`: `candidate_contract_mode=candidate_id` uses
  topology-homogeneous DataLoader batching and still fails manually mixed
  batches.
- `target_state`: trainer can split mixed batches by topology identity and run
  candidate forward/loss per group, then aggregate loss and metrics.

**Description (ukr):**

У true candidate-id режимі різні версії процесу можуть мати різні candidate
axes `[C_v]`. Тому один forward/loss не може безпечно використовувати перший
structural payload для всього batch. Мінімальна реалізація зупиняє такі batch
через topology-homogeneous sampler; guard залишається safety net. Повна
реалізація `group_by_topology` має вміти приймати вже змішаний batch,
розбивати його на sub-batches за `process_version_idx + stats_snapshot_version_idx`
і агрегувати loss/metrics.
```

- [ ] **Step 2: Update GNN runtime docs**

Add a short subsection:

```markdown
### Candidate Batch Topology Policy

`training.candidate_contract_mode=candidate_id` requires one topology identity
per batch. The identity is derived from `process_version_idx` and
`stats_snapshot_version_idx`. The DataLoader uses topology-homogeneous batching
for in-memory datasets. Mixed batches still fail under
`training.candidate_batch_topology_policy=single_topology_required`.

`group_by_topology` is reserved and must not be used until implemented.
```

- [ ] **Step 3: Update project state**

Add:

```markdown
- `candidate_id_topology_homogeneous_batching`: planned/implemented DataLoader
  batching groups candidate-id mini-batches by process version and stats
  snapshot before true candidate-id loss/eval.
```

- [ ] **Step 4: Update dissertation notes**

Add:

```markdown
True candidate-id prediction requires topology-homogeneous training/evaluation
batches. Mini-batches must be grouped by process version and stats snapshot
before forward/loss. Mixed batches can silently corrupt candidate axes because
`C_v` depends on process version and snapshot. This is a methodological
validity condition, not only an engineering optimization.
```

---

## 11. Task 8: Verification

**Files:** no code changes.

- [ ] **Step 1: Run targeted trainer tests**

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\application\test_trainer_forward_stats_logging.py -v
```

Expected: pass.

- [ ] **Step 2: Run catalog tests**

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\application\test_experiment_ui_catalog_fields.py -v
```

Expected: pass.

- [ ] **Step 3: Run dynamic candidate related tests**

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\application\test_evaluator_metrics.py tests\domain\test_eopkg_topology_conditioned.py -v
```

Expected: pass.

- [ ] **Step 4: Run architecture guard**

```powershell
.\.venv-modern\Scripts\python.exe tools\architecture_guard.py
```

Expected:

```text
[ARCH_GUARD] OK: architecture dependency rules satisfied.
```

- [ ] **Step 5: Run full tests**

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\ -v
```

Expected: all pass.

---

## 12. Self-Review

- The plan does not implement candidate-level loss/eval.
- The plan removes the fatal mixed-batch ambiguity before Stage 2 true work continues.
- The plan makes candidate-id experiments runnable by forming homogeneous
  batches instead of only failing mixed batches.
- The plan preserves legacy first-payload behavior outside `candidate_id`.
- `group_by_topology` remains explicitly recorded as debt for internal splitting
  of already mixed batches.
- The sampler and guard use already available batch metadata:
  `process_version_idx` and `stats_snapshot_version_idx`.
- If later candidate axis also depends on gateway mode or topology projection
  fingerprint, those fields should be added to topology identity before
  enabling `group_by_topology`.
