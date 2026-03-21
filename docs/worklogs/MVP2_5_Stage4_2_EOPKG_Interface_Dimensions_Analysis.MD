# MVP2.5 Stage4.2 - Interface & Dimensions Analysis (IG / Structure / Stats)

Date: 2026-03-20  
Status: Analysis completed

## 1. Current runtime interfaces

### 1.1 Instance graph (IG) path

Runtime path:
1. Adapter (`XES` / `Camunda`) -> `RawTrace`
2. `PrefixPolicy` -> `PrefixSlice`
3. `DynamicGraphBuilder` -> `GraphTensorContract`
4. `Trainer` converts contract to `torch_geometric.data.Data`
5. `DataLoader` batches samples.

Core IG tensors:
1. `x_cat`, `x_num`
2. `edge_index`, `edge_type`
3. `y`, `batch`, `num_nodes`

### 1.2 Structure + stats path (current)

`DynamicGraphBuilder` also writes:
1. `structural_edge_index`, `structural_edge_weight`
2. `struct_x` (when `graph_feature_mapping.enabled=true` and stats allowed)
3. snapshot scalars (`stats_snapshot_version_seq`, `stats_snapshot_as_of_epoch`)

Then `Trainer` stores these per sample inside each `Data` object.

## 2. Why `struct_x node count mismatch` appears

Observed warning:
`struct_x node count mismatch (got=19968 expected=156)`

Root cause:
1. `struct_x` is generated per sample with shape `[C, F]`, where `C = activity vocab size` (here `156`).
2. During PyG batching, per-sample tensors are concatenated along dimension 0.
3. For batch size `B=128`, collated shape becomes `[B*C, F] = [19968, F]`.
4. EOPKG model expects `struct_x.size(0) == self.num_classes == C`.

So EOPKG receives batched-concatenated structure stats instead of one structural tensor.

## 3. Code points

1. Per-sample injection into Data:
   - `src/application/use_cases/trainer.py:608`
   - `src/application/use_cases/trainer.py:639`
   - `src/application/use_cases/trainer.py:642`
   - `src/application/use_cases/trainer.py:645`
2. Same pattern in prebuilt CLI dataset:
   - `src/cli.py:307`
   - `src/cli.py:350`
   - `src/cli.py:353`
   - `src/cli.py:356`
3. EOPKG expectation for struct node count:
   - `src/domain/models/eopkg_models.py:228`
   - `src/domain/models/eopkg_models.py:234`
   - `src/domain/models/eopkg_models.py:236`
4. `struct_x` built as `[num_classes, n_features]`:
   - `src/domain/services/dynamic_graph_builder.py:393`
   - `src/domain/services/dynamic_graph_builder.py:394`

## 4. Interpretation of your logs

1. `struct_x_batches=7` means stats matrix is being passed to forward.
2. `struct_feature_dims=14` means config-driven stats matrix is active (14 node stats features).
3. `struct_rows=19968` confirms batch-concatenation (`128 * 156`).
4. Warning means EOPKG currently discards this matrix and falls back to learned structural embeddings.

## 5. Target interface semantics (required)

Expected semantic contract:
1. IG is per-sample and batched normally.
2. Structural graph + stats are structural context and must be aligned by class-space (`C`) not by IG node count.
3. If model is EOPKG:
   - always pass structural graph
   - pass stats attributes when available
   - keep dimensions compatible with `C`.

## 6. Fix strategy options

### Option A (quick patch, low risk now)

In trainer before forward:
1. detect `struct_x` as `[B*C, F]`
2. reshape to `[B, C, F]`
3. if all slices equal (or near-equal), take first `[C, F]`
4. same for structural edges by selecting one graph copy.

Pros:
1. fast unblock, warnings disappear
2. EOPKG starts using stats now for homogeneous snapshots

Cons:
1. unsafe when a batch mixes different snapshot IDs.

### Option B (correct by contract, recommended)

Refactor to split contracts:
1. IG tensors remain in PyG `Batch`
2. Structural context is resolved once per batch by snapshot key (or grouped by snapshot)
3. enforce homogeneous snapshot per batch (sampler/bucketing) or process per-snapshot micro-batches.

Pros:
1. correct semantics for `strict_asof`
2. no hidden mixing across snapshots

Cons:
1. larger implementation scope.

## 7. Recommended next step

For MVP2.5 closure:
1. implement Option A immediately to remove fallback and verify gains
2. add runtime guard:
   - if batch has >1 snapshot -> warning + deterministic policy (split or fallback)
3. schedule Option B as MVP3 contract hardening.
