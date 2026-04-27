# MVP2.5 Canonical Doc Sync and Architecture Debt

Date: 2026-04-24  
Status: Active reference

## 1. Purpose

This note records the currently acknowledged architecture/documentation debt that should stay visible while MVP2.5 canonical docs are being synchronized.

It does not introduce runtime changes.

## 2. Locked decisions

1. Canonical module layout is the real repository layout:
   - `src/domain/`
   - `src/application/`
   - `src/adapters/`
   - `src/infrastructure/`
2. Different runtime fallback policies are allowed for exploratory runs.
3. For research-grade temporal runs, preferred policy is:
   - `experiment.stats_time_policy = strict_asof`
   - `experiment.on_missing_asof_snapshot = raise`
4. Dual sequential retraining modes for observed/structural branches are a separate future task.

## 3. Debt: Snapshot-Homogeneous Batching

### 3.1 Current behavior

Current runtime keeps the temporary Option-A policy:
1. mixed snapshot versions may appear in one PyG batch,
2. runtime emits a warning,
3. structural payload for forward is taken from the first graph in the batch.

This is documented in:
1. `docs/LLD_MVP2_5.MD`
2. `docs/worklogs/MVP2_5_Stage4_2_OptionA_Unbatch_Fix_Report.MD`

### 3.2 Why this is a problem

Observed graph tensors are naturally batched per sample, but structural payload (`struct_x`, `structural_edge_index`, `structural_edge_weight`) is not sample-local in the same sense.

When one batch mixes different snapshot identities:
1. structural statistics can belong to different `knowledge_version` / `as_of` states,
2. first-graph selection becomes a lossy heuristic,
3. forward pass is no longer semantically clean for strict temporal experiments,
4. drift experiments become harder to defend scientifically because one batch may partially represent multiple temporal worlds.

This is acceptable as a temporary runtime workaround, but not as a final research-grade batching contract.

### 3.3 Recommended solution direction

Preferred target is snapshot-homogeneous batching:
1. group samples by snapshot identity before batching,
2. or split mixed batches into snapshot-homogeneous micro-batches before forward,
3. keep structural payload resolved once per homogeneous batch,
4. fail or re-bucket explicitly instead of silently relying on first-graph payload.

Minimal target contract:
1. one forward batch -> one structural snapshot identity,
2. no mixed `knowledge_version` inside one structural forward context,
3. warning-only policy should become transitional, not final.

## 4. Debt: Activity-to-Topology Alignment Gate

### 4.1 Problem

Structural branch quality depends on correct alignment between:
1. activity ids from traces,
2. BPMN / topology node ids used by repository structure,
3. stats indexes projected into class space.

If alignment is weak:
1. `allowed_target_mask` remains formally present,
2. `struct_x` may be populated,
3. but the structural signal can silently degrade or become misleading.

### 4.2 Required direction

Research-grade runs need an explicit preflight gate with locked thresholds for:
1. event match ratio,
2. unique activity coverage,
3. node coverage.

Policy can stay configurable for exploratory runs, but dissertation-grade runs should use a locked strict profile.

## 5. Debt: Topology Projection
при collaps_for_predioction ломається індексанція зі структурою. Треба щоб правильно робилась індексація був Alignment після модифікації структури
Треба Продумати реалізацію


## 6. Debt: `src/cli.py` Overgrowth

### 6.1 Current issue

`src/cli.py` is no longer only a thin composition root.

It currently combines:
1. config loading and override logic,
2. trace preparation and cascade splitting,
3. graph dataset cache orchestration,
4. telemetry/profile preparation,
5. model/trainer wiring.

This increases coupling and makes architectural intent less obvious.

### 6.2 Why this matters

The file is becoming an operational god-object at the application boundary:
1. harder to test in isolation,
2. harder to evolve run modes cleanly,
3. higher risk of mixing orchestration policy with reusable application services,
4. documentation and code navigation drift more easily.

### 6.3 Recommended direction

Plan refactoring, but do not execute it yet:
1. keep `src/cli.py` as the composition root only,
2. move data-preparation orchestration into dedicated application services,
3. move cache-specific orchestration into focused helper/service modules,
4. keep run-profile/report assembly in explicit application-level helpers.

This debt is acknowledged but deferred.
