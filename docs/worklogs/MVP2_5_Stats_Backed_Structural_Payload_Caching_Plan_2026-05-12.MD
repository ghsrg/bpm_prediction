# Stats-Backed Structural Payload Caching Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` or `executing-plans` to implement this plan task-by-task. Use `superpowers:test-driven-development` or `test-driven-development` for every implementation task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent full `eval_drift + structural_mode + statistic_enabled` runs from exhausting Neo4j, Neo4j driver connections, RAM, or virtual memory by caching stats payloads and structural tensors at the correct snapshot granularity.

**Architecture:** Treat this as one P0 architecture debt with two implementation stages. Stage 1 makes stats snapshot resolution snapshot-aware and stops loading/copying heavy DTO payloads per exact prefix timestamp. Stage 2 deduplicates structural tensors in graph dataset shards so prefix graphs reference shared payloads instead of carrying repeated tensors.

**Tech Stack:** Python, PyTorch, PyTorch Geometric `Data`, Neo4j Python driver, local sharded graph dataset cache, pytest.

---

## Context

The failing runtime pattern is:

```yaml
experiment.mode: eval_drift
experiment.structural_mode: true
experiment.statistic_enabled: true
experiment.graph_dataset_disk_spill_enabled: true
```

Observed evidence:

- `Base-UN-drift-loan` passed.
- `S_Str-UNc-drift-loan` passed.
- `S_Str-USc-loan` passed.
- `S_Str-USc-drift-loan` failed during `Build test graphs`.
- Windows Event Log reported low virtual memory and `python.exe` at about 32.8 GB virtual memory.
- Captured error:

```text
Failed to read from defunct connection ... ConnectionAbortedError(10053)
```

The failure happens during graph dataset construction, before model forward. Therefore `model.structural_score_mode: bilinear_with_prior` is not the root cause. The likely causes are:

1. heavy stats snapshot payload access is cached at the wrong granularity or loaded too aggressively from Neo4j;
2. repeated structural payload is materialized into many prefix graphs and graph shards.

---

## Design Decision

Use one unified debt:

```text
stats_backed_structural_payload_caching
```

with two stages:

1. `snapshot-aware DTO/stats payload cache`
2. `structural payload deduplication`

Do not solve `snapshot_homogeneous_batching` here. This work must stay compatible with that debt by exposing explicit payload identities that a future sampler can use for homogeneous batching.

---

## Files To Modify

### Stage 1 Files

- Modify: `src/infrastructure/repositories/neo4j_knowledge_graph_repository.py`
  - Split lightweight snapshot timeline lookup from heavy stats payload loading.
  - Add bounded payload cache keyed by resolved snapshot identity.
  - Add diagnostics counters for timeline loads and payload loads.

- Modify: `src/domain/services/dynamic_graph_builder.py`
  - Stop using exact prefix `as_of_ts` as the only DTO cache key.
  - Add resolved snapshot identity helpers.
  - Add cache diagnostics for DTO/topology cache hit/miss and unique resolved snapshots.

- Modify: `tests/infrastructure/test_neo4j_knowledge_graph_repository.py`
  - Add tests proving timeline lookup does not select heavy JSON fields.
  - Add tests proving repeated as-of lookups for the same resolved snapshot load heavy payload once.

- Create or modify: `tests/domain/test_dynamic_graph_builder_cache.py`
  - Add tests proving many prefix timestamps mapped to one snapshot do not grow DTO/topology caches linearly.

### Stage 2 Files

- Modify: `src/cli.py`
  - Add structural payload registry for sharded graph cache construction.
  - Save shards in a backward-compatible format that supports deduplicated payloads.
  - Update `_iter_graphs_from_dataset_payload` and tensor diagnostics to rehydrate payloads.

- Modify: `src/application/use_cases/trainer.py`
  - Teach `ShardedGraphDataset` to load deduplicated shard payloads.
  - Reattach shared structural tensors to returned `Data` objects.
  - Preserve support for legacy shard files that are plain `list[Data]`.

- Modify: `tests/application/test_cli_graph_dataset_cache.py`
  - Add sharded cache tests for deduplicated structural payloads.

- Modify: `tests/application/test_trainer_preloaded_data.py`
  - Add trainer-side lazy shard loading tests for deduplicated payload shards.

### Documentation Files

- Modify: `docs/current/architecture-debt.md`
  - Keep `stats_backed_structural_payload_caching` as active P0 debt until implementation is verified.

- Modify after implementation: `docs/current/project-state.md`
  - Document that stats-backed structural drift now uses snapshot-aware payload caching.

- Modify after implementation: `docs/GNN_RUNTIME_MVP2_5.MD`
  - Document structural payload identity, deduplicated shard format, and runtime diagnostics.

---

## Stage 1: Snapshot-Aware DTO And Stats Payload Cache

### Task 1.1: Add Neo4j Lightweight Snapshot Timeline Tests

**Files:**

- Modify: `tests/infrastructure/test_neo4j_knowledge_graph_repository.py`

- [ ] **Step 1: Write failing test for lightweight timeline query**

Add this test near existing as-of repository tests:

```python
def test_neo4j_snapshot_timeline_query_does_not_load_heavy_payloads():
    repo = _build_repo()
    captured = []

    def _fake_read(self, *, operation, query, params):
        captured.append((operation, query, params))
        if operation == "load_stats_timeline_with_proc_def":
            return [
                {
                    "knowledge_version": "k000001",
                    "as_of_ts": "2026-03-01T00:00:00Z",
                    "kv_seq": 1,
                }
            ]
        return []

    repo._run_read = MethodType(_fake_read, repo)
    row = repo._lookup_stats_snapshot_identity(
        process_name="loan_v1_v4_simulated",
        tenant_id="default",
        version_key="v1",
        proc_def_id="loan_v1",
        as_of_utc=datetime(2026, 3, 2, tzinfo=timezone.utc),
    )

    assert row is not None
    timeline_query = next(query for operation, query, _ in captured if operation == "load_stats_timeline_with_proc_def")
    assert "node_stats_json" not in timeline_query
    assert "edge_stats_json" not in timeline_query
    assert "gnn_features_json" not in timeline_query
    assert "stats_diagnostics_json" not in timeline_query
```

- [ ] **Step 2: Run failing test**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/infrastructure/test_neo4j_knowledge_graph_repository.py::test_neo4j_snapshot_timeline_query_does_not_load_heavy_payloads -v
```

Expected: fail because `_lookup_stats_snapshot_identity` does not exist yet.

### Task 1.2: Implement Neo4j Lightweight Timeline And Payload Cache

**Files:**

- Modify: `src/infrastructure/repositories/neo4j_knowledge_graph_repository.py`

- [ ] **Step 1: Add cache fields in `__init__`**

Add bounded caches and counters after `_snapshot_timeline_cache`:

```python
self._snapshot_payload_cache: dict[tuple[str, str, str, str, str], dict[str, Any] | None] = {}
self._snapshot_payload_cache_max_entries = 512
self._diagnostics: dict[str, int] = {
    "snapshot_timeline_loads": 0,
    "snapshot_timeline_cache_hits": 0,
    "snapshot_payload_loads": 0,
    "snapshot_payload_cache_hits": 0,
}
```

- [ ] **Step 2: Add diagnostic accessors**

```python
def cache_diagnostics(self) -> dict[str, int]:
    return dict(self._diagnostics)

@staticmethod
def _bounded_cache_put(cache: dict[Any, Any], key: Any, value: Any, *, max_entries: int) -> None:
    cache[key] = value
    while len(cache) > max_entries:
        cache.pop(next(iter(cache)))
```

- [ ] **Step 3: Change `_load_snapshot_timeline` query projection**

Return only identity fields:

```cypher
RETURN
  ss.knowledge_version AS knowledge_version,
  toString(ss.as_of_ts) AS as_of_ts,
  ss.kv_seq AS kv_seq
ORDER BY ss.as_of_ts ASC, ss.kv_seq ASC
```

Do the same for `load_stats_timeline_no_proc_def`.

- [ ] **Step 4: Rename lookup method internally**

Create `_lookup_stats_snapshot_identity(...)` with the current bisect logic. It should return a lightweight row containing:

```python
{
    "knowledge_version": "k000123",
    "as_of_ts": "2026-03-01T00:00:00Z",
    "kv_seq": 123,
    "_as_of_epoch": 1772323200.0,
    "_with_proc_def": True,
}
```

- [ ] **Step 5: Add heavy payload loader**

Add:

```python
def _load_stats_snapshot_payload(
    self,
    *,
    process_name: str,
    tenant_id: str,
    version_key: str,
    proc_def_id: str,
    identity_row: dict[str, Any],
) -> dict[str, Any] | None:
    knowledge_version = self._optional_text(identity_row.get("knowledge_version"))
    if knowledge_version is None:
        return None
    cache_key = (process_name, tenant_id, version_key, proc_def_id, knowledge_version)
    if cache_key in self._snapshot_payload_cache:
        self._diagnostics["snapshot_payload_cache_hits"] += 1
        cached = self._snapshot_payload_cache[cache_key]
        return dict(cached) if isinstance(cached, dict) else None

    self._diagnostics["snapshot_payload_loads"] += 1
    rows = self._run_read(
        operation="load_stats_snapshot_payload",
        query="""
        MATCH (ss:StatsSnapshot {
          process_name: $process_name,
          tenant_id: $tenant_id,
          version_key: $version_key,
          proc_def_id: $proc_def_id,
          knowledge_version: $knowledge_version
        })
        RETURN
          ss.knowledge_version AS knowledge_version,
          toString(ss.as_of_ts) AS as_of_ts,
          ss.node_stats_json AS node_stats_json,
          ss.edge_stats_json AS edge_stats_json,
          ss.gnn_features_json AS gnn_features_json,
          ss.stats_diagnostics_json AS stats_diagnostics_json,
          ss.metadata_json AS metadata_json
        LIMIT 1
        """,
        params={
            "process_name": process_name,
            "tenant_id": tenant_id,
            "version_key": version_key,
            "proc_def_id": proc_def_id,
            "knowledge_version": knowledge_version,
        },
    )
    payload = dict(rows[0]) if rows else None
    self._bounded_cache_put(
        self._snapshot_payload_cache,
        cache_key,
        payload,
        max_entries=self._snapshot_payload_cache_max_entries,
    )
    return dict(payload) if isinstance(payload, dict) else None
```

If fallback without `proc_def_id` is needed, use a second operation `load_stats_snapshot_payload_no_proc_def` and cache key proc_def token `"__no_proc_def__"`.

- [ ] **Step 6: Wire `get_process_structure_as_of`**

Replace direct heavy timeline row usage with:

```python
identity_row = self._lookup_stats_snapshot_identity(...)
row = self._load_stats_snapshot_payload(..., identity_row=identity_row) if identity_row else None
```

Keep existing behavior when no row exists:

```python
return self._mark_missing_asof_snapshot(base, as_of_utc)
```

- [ ] **Step 7: Run targeted repository tests**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/infrastructure/test_neo4j_knowledge_graph_repository.py -v
```

Expected: pass.

### Task 1.3: Test Repeated As-Of Lookups Load Payload Once

**Files:**

- Modify: `tests/infrastructure/test_neo4j_knowledge_graph_repository.py`

- [ ] **Step 1: Add failing payload-cache test**

```python
def test_get_process_structure_asof_loads_payload_once_for_same_resolved_snapshot():
    repo = _build_repo()
    base_dto = ProcessStructureDTO(
        version="v1",
        proc_def_id="PROC_DEF_1",
        allowed_edges=[("A", "B")],
        metadata={"base": "yes"},
    )
    repo.get_process_structure = MethodType(
        lambda self, version, process_name=None: base_dto.model_copy(deep=True),
        repo,
    )
    operations = []

    def _fake_read(self, *, operation, query, params):
        operations.append(operation)
        if operation == "load_stats_timeline_with_proc_def":
            return [
                {"knowledge_version": "k000010", "as_of_ts": "2026-03-01T00:00:00Z", "kv_seq": 10}
            ]
        if operation == "load_stats_snapshot_payload":
            return [
                {
                    "knowledge_version": "k000010",
                    "as_of_ts": "2026-03-01T00:00:00Z",
                    "node_stats_json": '{"windows":{"all_time":{"version":{"exec_count":{"A":1}}}}}',
                    "edge_stats_json": "{}",
                    "gnn_features_json": "{}",
                    "stats_diagnostics_json": "{}",
                    "metadata_json": '{"snapshot_meta":"ok"}',
                }
            ]
        return []

    repo._run_read = MethodType(_fake_read, repo)

    first = repo.get_process_structure_as_of(
        "v1",
        process_name="loan_v1_v4_simulated",
        as_of_ts=datetime(2026, 3, 2, tzinfo=timezone.utc),
    )
    second = repo.get_process_structure_as_of(
        "v1",
        process_name="loan_v1_v4_simulated",
        as_of_ts=datetime(2026, 3, 3, tzinfo=timezone.utc),
    )

    assert first is not None
    assert second is not None
    assert operations.count("load_stats_timeline_with_proc_def") == 1
    assert operations.count("load_stats_snapshot_payload") == 1
    diagnostics = repo.cache_diagnostics()
    assert diagnostics["snapshot_timeline_cache_hits"] >= 1
    assert diagnostics["snapshot_payload_cache_hits"] >= 1
```

- [ ] **Step 2: Run the new test**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/infrastructure/test_neo4j_knowledge_graph_repository.py::test_get_process_structure_asof_loads_payload_once_for_same_resolved_snapshot -v
```

Expected: pass after Task 1.2.

### Task 1.4: Make DynamicGraphBuilder Cache Diagnostics Snapshot-Aware

**Files:**

- Modify: `src/domain/services/dynamic_graph_builder.py`
- Create or modify: `tests/domain/test_dynamic_graph_builder_cache.py`

- [ ] **Step 1: Write failing builder cache test**

```python
def test_dynamic_builder_cache_diagnostics_tracks_unique_snapshot_identity(mock_feature_configs):
    traces = [_trace("c1", "v1", ["A", "B", "C"], base_ts=1700000000.0)]
    encoder = FeatureEncoder(feature_configs=mock_feature_configs, traces=traces)
    dto = ProcessStructureDTO(
        version="v1",
        allowed_edges=[("A", "B"), ("B", "C")],
        metadata={"knowledge_version": "k000010", "as_of_ts": "2026-03-01T00:00:00+00:00"},
    )
    port = _CountingAsOfPort(dto)
    builder = DynamicGraphBuilder(
        feature_encoder=encoder,
        knowledge_port=port,
        process_name="loan_v1_v4_simulated",
        stats_time_policy="strict_asof",
        graph_feature_mapping={"enabled": True},
        cache_policy="full",
    )

    for offset in range(20):
        prefix = _prefix("v1", ["A", "B"], target="C", last_ts=1700000000.0 + offset * 60.0)
        builder.build_graph(prefix)

    diagnostics = builder.cache_diagnostics()
    assert diagnostics["unique_snapshot_identities"] == 1
    assert diagnostics["topology_cache_misses"] == 1
    assert diagnostics["topology_cache_hits"] >= 19
    assert diagnostics["dto_cache_entries"] <= 1
```

Define local helpers `_trace`, `_prefix`, and `_CountingAsOfPort` in the test file using existing patterns from `tests/integration/test_stats_forward_verification_pack.py`.

- [ ] **Step 2: Run failing test**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/domain/test_dynamic_graph_builder_cache.py::test_dynamic_builder_cache_diagnostics_tracks_unique_snapshot_identity -v
```

Expected: fail because `cache_diagnostics()` does not exist or DTO cache still grows by exact timestamp.

- [ ] **Step 3: Implement diagnostics and bounded DTO cache change**

Add counters:

```python
self._cache_diagnostics = {
    "dto_cache_hits": 0,
    "dto_cache_misses": 0,
    "topology_cache_hits": 0,
    "topology_cache_misses": 0,
}
self._resolved_snapshot_identities: set[tuple[str, str, str | None, str | None]] = set()
```

Add:

```python
def cache_diagnostics(self) -> dict[str, int]:
    return {
        **{key: int(value) for key, value in self._cache_diagnostics.items()},
        "dto_cache_entries": int(len(self._dto_cache)),
        "topology_cache_entries": int(len(self._topology_cache)),
        "unique_snapshot_identities": int(len(self._resolved_snapshot_identities)),
    }
```

Change DTO cache behavior:

- exact `as_of_ts` may be used only as a temporary lookup key if needed;
- after DTO is returned, compute:

```python
snapshot = self._stats_snapshot_metadata(dto)
snapshot_key = (
    self.process_name or "__auto__",
    str(dto.version),
    self._clean_optional_text(snapshot.get("knowledge_version")) or None,
    self._clean_optional_text(snapshot.get("as_of_ts")) or None,
)
```

- store returned DTO under `snapshot_key`;
- if the same `snapshot_key` is already cached, return it and avoid growing exact timestamp entries;
- keep `_dto_cache_max_entries` bounded.

- [ ] **Step 4: Run dynamic builder cache tests**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/domain/test_dynamic_graph_builder_cache.py tests/integration/test_stats_forward_verification_pack.py -v
```

Expected: pass.

### Stage 1 Completion Gate

- [ ] **Run Stage 1 verification**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/infrastructure/test_neo4j_knowledge_graph_repository.py tests/domain/test_dynamic_graph_builder_cache.py tests/integration/test_stats_forward_verification_pack.py -v
.\.venv\Scripts\python.exe tools\architecture_guard.py
```

Expected:

- repository tests pass;
- dynamic graph builder cache tests pass;
- architecture guard passes.

Do not proceed to Stage 2 until this gate passes.

---

## Stage 2: Structural Payload Deduplication In Sharded Graph Dataset

### Task 2.1: Define Deduplicated Shard Format

**Files:**

- Modify: `src/cli.py`
- Modify: `tests/application/test_cli_graph_dataset_cache.py`

Target shard payload format:

```python
{
    "schema": 2,
    "format": "dedup_structural_payloads",
    "graphs": [Data(...), Data(...)],
    "structural_payloads": {
        "payload_key": {
            "struct_x": torch.Tensor,
            "structural_edge_index": torch.Tensor,
            "structural_edge_weight": torch.Tensor,
            "struct_node_to_class_index": torch.Tensor,
        }
    },
}
```

Each graph stores:

```python
data.structural_payload_key = "payload_key"
```

or, if PyG serialization rejects string attributes in any path:

```python
data.structural_payload_key_idx = torch.tensor([idx], dtype=torch.long)
```

and the shard stores `structural_payload_key_index`.

Prefer string `structural_payload_key` first because it is easier to inspect.

- [ ] **Step 1: Write failing cache test**

Add:

```python
def test_sharded_cache_deduplicates_structural_payloads(tmp_path: Path):
    shared_struct_x = torch.tensor([[1.0], [2.0]], dtype=torch.float32)
    shared_edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    shared_edge_weight = torch.tensor([0.5], dtype=torch.float32)
    shared_node_to_class = torch.tensor([0, 1], dtype=torch.long)

    graph_a = Data(y=torch.tensor([1]), struct_x=shared_struct_x, structural_edge_index=shared_edge_index)
    graph_a.structural_edge_weight = shared_edge_weight
    graph_a.struct_node_to_class_index = shared_node_to_class
    graph_b = Data(y=torch.tensor([1]), struct_x=shared_struct_x, structural_edge_index=shared_edge_index)
    graph_b.structural_edge_weight = shared_edge_weight
    graph_b.struct_node_to_class_index = shared_node_to_class

    payload_key = _structural_payload_key_from_data(graph_a)
    stripped_a, payloads = _strip_structural_payload(graph_a, {})
    stripped_b, payloads = _strip_structural_payload(graph_b, payloads)

    assert payload_key in payloads
    assert len(payloads) == 1
    assert not hasattr(stripped_a, "struct_x")
    assert not hasattr(stripped_b, "struct_x")
    assert getattr(stripped_a, "structural_payload_key") == payload_key
```

- [ ] **Step 2: Run failing test**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/application/test_cli_graph_dataset_cache.py::test_sharded_cache_deduplicates_structural_payloads -v
```

Expected: fail because helper functions do not exist.

### Task 2.2: Implement Payload Key, Strip, And Rehydrate Helpers

**Files:**

- Modify: `src/cli.py`

- [ ] **Step 1: Add helper `_structural_payload_key_from_tensors`**

```python
def _tensor_digest(tensor: torch.Tensor | None) -> str:
    if not isinstance(tensor, torch.Tensor):
        return "none"
    cpu = tensor.detach().cpu().contiguous()
    payload = {
        "shape": list(cpu.shape),
        "dtype": str(cpu.dtype),
        "bytes": cpu.numpy().tobytes().hex(),
    }
    text = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()

def _structural_payload_key_from_data(data: Data) -> str:
    payload = {
        "struct_x": _tensor_digest(getattr(data, "struct_x", None)),
        "structural_edge_index": _tensor_digest(getattr(data, "structural_edge_index", None)),
        "structural_edge_weight": _tensor_digest(getattr(data, "structural_edge_weight", None)),
        "struct_node_to_class_index": _tensor_digest(getattr(data, "struct_node_to_class_index", None)),
        "stats_snapshot_version_idx": _tensor_digest(getattr(data, "stats_snapshot_version_idx", None)),
    }
    text = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()
```

If NumPy access is unavailable for a tensor dtype, use `bytes(cpu.storage().untyped())` or `cpu.numpy().tobytes()` only for CPU numeric tensors used here.

- [ ] **Step 2: Add strip helper**

```python
def _strip_structural_payload(data: Data, registry: dict[str, dict[str, torch.Tensor]]) -> tuple[Data, dict[str, dict[str, torch.Tensor]]]:
    has_payload = any(
        isinstance(getattr(data, name, None), torch.Tensor)
        for name in ("struct_x", "structural_edge_index", "structural_edge_weight", "struct_node_to_class_index")
    )
    if not has_payload:
        return data, registry
    key = _structural_payload_key_from_data(data)
    if key not in registry:
        registry[key] = {
            name: getattr(data, name).detach().cpu()
            for name in ("struct_x", "structural_edge_index", "structural_edge_weight", "struct_node_to_class_index")
            if isinstance(getattr(data, name, None), torch.Tensor)
        }
    stripped = data.clone()
    for name in ("struct_x", "structural_edge_index", "structural_edge_weight", "struct_node_to_class_index"):
        if hasattr(stripped, name):
            delattr(stripped, name)
    stripped.structural_payload_key = key
    return stripped, registry
```

- [ ] **Step 3: Add rehydrate helper**

```python
def _attach_structural_payload(data: Data, registry: dict[str, dict[str, torch.Tensor]]) -> Data:
    key = getattr(data, "structural_payload_key", None)
    if key is None:
        return data
    payload = registry.get(str(key))
    if not isinstance(payload, dict):
        return data
    for name, tensor in payload.items():
        if isinstance(tensor, torch.Tensor):
            setattr(data, name, tensor)
    return data
```

- [ ] **Step 4: Run helper test**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/application/test_cli_graph_dataset_cache.py::test_sharded_cache_deduplicates_structural_payloads -v
```

Expected: pass.

### Task 2.3: Save Deduplicated Shards

**Files:**

- Modify: `src/cli.py`

- [ ] **Step 1: Update `_flush_buffer` in `_build_graph_dataset_sharded`**

Before `torch.save`, build per-shard registry:

```python
stripped_buffer: list[Data] = []
structural_payloads: dict[str, dict[str, torch.Tensor]] = {}
for item in buffer:
    stripped, structural_payloads = _strip_structural_payload(item, structural_payloads)
    stripped_buffer.append(stripped)

shard_payload = {
    "schema": 2,
    "format": "dedup_structural_payloads",
    "graphs": stripped_buffer,
    "structural_payloads": structural_payloads,
}
torch.save(shard_payload, tmp_path)
```

Keep legacy behavior only for a feature flag if needed. Default should be dedup for sharded cache.

- [ ] **Step 2: Add shard row payload count**

When appending `shard_rows`, include:

```python
"structural_payloads": int(len(structural_payloads)),
```

- [ ] **Step 3: Update `_iter_graphs_from_dataset_payload`**

When a shard loads as dict:

```python
if isinstance(loaded, dict) and loaded.get("format") == "dedup_structural_payloads":
    graphs = loaded.get("graphs", [])
    registry = loaded.get("structural_payloads", {})
    for item in graphs:
        if isinstance(item, Data):
            yield _attach_structural_payload(item, registry)
    continue
```

Keep existing list support.

- [ ] **Step 4: Add roundtrip test**

Extend `test_graph_dataset_cache_sharded_roundtrip` or add:

```python
def test_iter_graphs_rehydrates_deduplicated_structural_payload(tmp_path: Path):
    # Build one shard manually using the new dict format.
    # Load through _iter_graphs_from_dataset_payload.
    # Assert struct_x and structural_edge_index are present on returned Data.
```

Use concrete tensors:

```python
assert torch.equal(items[0].struct_x, torch.tensor([[1.0], [2.0]]))
assert torch.equal(items[0].structural_edge_index, torch.tensor([[0], [1]]))
```

- [ ] **Step 5: Run CLI cache tests**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/application/test_cli_graph_dataset_cache.py -v
```

Expected: pass.

### Task 2.4: Load Deduplicated Shards In Trainer

**Files:**

- Modify: `src/application/use_cases/trainer.py`
- Modify: `tests/application/test_trainer_preloaded_data.py`

- [ ] **Step 1: Write failing trainer shard test**

Add:

```python
def test_sharded_graph_dataset_rehydrates_deduplicated_structural_payload(tmp_path: Path):
    entry_dir = tmp_path / "entry"
    shard_dir = entry_dir / "test_shards"
    shard_dir.mkdir(parents=True)
    graph = Data(y=torch.tensor([1], dtype=torch.long))
    graph.structural_payload_key = "payload-a"
    payload = {
        "schema": 2,
        "format": "dedup_structural_payloads",
        "graphs": [graph],
        "structural_payloads": {
            "payload-a": {
                "struct_x": torch.tensor([[1.0], [2.0]], dtype=torch.float32),
                "structural_edge_index": torch.tensor([[0], [1]], dtype=torch.long),
                "structural_edge_weight": torch.tensor([0.5], dtype=torch.float32),
                "struct_node_to_class_index": torch.tensor([0, 1], dtype=torch.long),
            }
        },
    }
    torch.save(payload, shard_dir / "test_00001.pt")
    dataset = ShardedGraphDataset.from_payload(
        {
            "kind": "sharded_cache_split",
            "entry_dir": str(entry_dir),
            "split": "test",
            "graphs": 1,
            "shards": [{"path": "test_shards/test_00001.pt", "count": 1}],
        }
    )

    item = dataset[0]
    assert torch.equal(item.struct_x, torch.tensor([[1.0], [2.0]], dtype=torch.float32))
    assert torch.equal(item.structural_edge_index, torch.tensor([[0], [1]], dtype=torch.long))
```

- [ ] **Step 2: Run failing test**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/application/test_trainer_preloaded_data.py::test_sharded_graph_dataset_rehydrates_deduplicated_structural_payload -v
```

Expected: fail because `ShardedGraphDataset` expects shard payload to be a list.

- [ ] **Step 3: Update `ShardedGraphDataset._load_shard`**

Support both formats:

```python
loaded = torch.load(shard_path, map_location="cpu")
if isinstance(loaded, dict) and loaded.get("format") == "dedup_structural_payloads":
    graphs = loaded.get("graphs", [])
    registry = loaded.get("structural_payloads", {})
    if not isinstance(graphs, list) or not isinstance(registry, dict):
        raise ValueError(f"Invalid deduplicated shard payload for {shard_path}.")
    hydrated = []
    for graph in graphs:
        if not isinstance(graph, Data):
            raise ValueError(f"Invalid graph item in {shard_path}.")
        key = getattr(graph, "structural_payload_key", None)
        payload = registry.get(str(key)) if key is not None else None
        if isinstance(payload, dict):
            for name, tensor in payload.items():
                if isinstance(tensor, torch.Tensor):
                    setattr(graph, name, tensor)
        hydrated.append(graph)
    loaded = hydrated
if not isinstance(loaded, list):
    raise ValueError(f"Invalid shard payload type for {shard_path}.")
```

- [ ] **Step 4: Run trainer shard tests**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/application/test_trainer_preloaded_data.py -v
```

Expected: pass.

### Task 2.5: Add Payload Dedup Diagnostics

**Files:**

- Modify: `src/cli.py`
- Modify: `docs/GNN_RUNTIME_MVP2_5.MD`

- [ ] **Step 1: Add shard metadata totals**

In `_save_graph_dataset_cache_sharded`, compute from split shard rows:

```python
"structural_payloads": {
    "train": sum(int(row.get("structural_payloads", 0)) for row in train_split.get("shards", [])),
    "validation": sum(int(row.get("structural_payloads", 0)) for row in val_split.get("shards", [])),
    "test": sum(int(row.get("structural_payloads", 0)) for row in test_split.get("shards", [])),
}
```

- [ ] **Step 2: Log build summary**

After each sharded split build, log:

```python
logger.info(
    "GRAPH_DATASET_STRUCT_PAYLOADS split=%s graphs=%d shards=%d structural_payloads=%d",
    split_key,
    int(dataset_payload.get("graphs", 0)),
    len(dataset_payload.get("shards", [])),
    sum(int(row.get("structural_payloads", 0)) for row in dataset_payload.get("shards", [])),
)
```

- [ ] **Step 3: Document diagnostics**

In `docs/GNN_RUNTIME_MVP2_5.MD`, add a short section:

```markdown
### Stats-backed structural payload caching

Full drift runs with `statistic_enabled=true` use snapshot-aware stats payload
loading and deduplicated sharded structural payloads. `GRAPH_DATASET_STRUCT_PAYLOADS`
reports how many unique structural payloads were written per split.
```

### Stage 2 Completion Gate

- [ ] **Run Stage 2 verification**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/application/test_cli_graph_dataset_cache.py tests/application/test_trainer_preloaded_data.py -v
.\.venv\Scripts\python.exe -m pytest tests/integration/test_stats_forward_verification_pack.py tests/application/test_trainer_forward_stats_logging.py -v
.\.venv\Scripts\python.exe tools\architecture_guard.py
```

Expected:

- sharded cache tests pass;
- trainer preloaded-data tests pass;
- stats forward verification passes;
- architecture guard passes.

---

## End-To-End Verification

After both stages:

- [ ] **Run focused regression suite**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/infrastructure/test_neo4j_knowledge_graph_repository.py tests/domain/test_dynamic_graph_builder_cache.py tests/application/test_cli_graph_dataset_cache.py tests/application/test_trainer_preloaded_data.py tests/integration/test_stats_forward_verification_pack.py -v
```

- [ ] **Run architecture guard**

```powershell
.\.venv\Scripts\python.exe tools\architecture_guard.py
```

- [ ] **Run small drift smoke test**

Use the existing `S_Str-USc-drift-loan` preset with:

```yaml
experiment.fraction: 0.01
experiment.structural_mode: true
experiment.statistic_enabled: true
experiment.graph_dataset_disk_spill_enabled: true
```

Expected:

- test graph build completes;
- `GRAPH_DATASET_STRUCT_PAYLOADS` logs show unique payload count much smaller than graph count;
- Neo4j diagnostics show timeline/payload loads roughly bounded by versions and snapshots, not by prefix graph count.

- [ ] **Run full drift validation**

Run full `S_Str-USc-drift-loan` only after smoke test passes.

Expected:

- no Neo4j `defunct connection`;
- no system-wide low virtual memory event;
- graph build completes;
- MLflow logs still report `test_struct_x_max_abs > 0` and stats snapshot metadata.

---

## Documentation Updates After Implementation

- [ ] Update `docs/current/project-state.md`
  - Add current-state note that stats-backed structural drift uses snapshot-aware payload caching and deduplicated sharded structural payloads.

- [ ] Update `docs/GNN_RUNTIME_MVP2_5.MD`
  - Document structural payload identity and diagnostics.

- [ ] Update `docs/current/architecture-debt.md`
  - Change `stats_backed_structural_payload_caching.status` from `active` to `closed` only after full drift validation passes.

- [ ] Update `docs/dissertation_changes.MD`
  - Add short note: dissertation-grade structural drift experiments require snapshot-aware stats payload caching to avoid runtime/system bias and incomplete runs.

---

## Risk Notes

- Do not remove legacy shard support. Existing cached runs may still use `list[Data]` shards.
- Do not solve batch homogeneity in this implementation. Only expose payload identity and keep the path compatible with future homogeneous sampling.
- Do not move Neo4j access into train/eval forward. Runtime must still consume repository artifacts through `IKnowledgeGraphPort`.
- Do not change model math in this task. `ClassAwareStructuralScoring`, `bilinear_with_prior`, and auxiliary losses are out of scope.

---

## Self-Review

- `Spec coverage`: both confirmed failure modes are covered: Neo4j/stats payload pressure in Stage 1 and repeated structural tensor payloads in Stage 2.
- `Completeness scan`: no unresolved markers remain; tasks include concrete files, tests, and commands.
- `Type consistency`: planned names are consistent: `cache_diagnostics`, `_lookup_stats_snapshot_identity`, `_load_stats_snapshot_payload`, `_strip_structural_payload`, `_attach_structural_payload`, `structural_payload_key`.
- `Scope check`: this is one coherent architecture debt with two dependent stages. It should not be split into unrelated plans because Stage 2 depends on Stage 1 identities and diagnostics.
