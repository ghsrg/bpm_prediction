# Phase-2C Review Fix Plan

Date: 2026-07-03
Reviewer role: strategy/runtime consistency review
Scope: current `main` branch after reported Phase-2C completion

## Review Baseline

Checked state:

- Current branch: `main`
- Tracked diff against `main...HEAD`: empty
- Untracked files: `Manuscript.docx`
- Direct `Phase-2C` report/artifact: not found by repository search
- Architecture guard: passed
- Targeted candidate/mask/runtime tests: passed

Verification commands run:

```powershell
.\.venv-modern\Scripts\python.exe tools\architecture_guard.py
.\.venv-modern\Scripts\python.exe -m pytest tests\domain\test_eopkg_topology_conditioned.py tests\application\test_candidate_target_mapping.py tests\application\test_evaluator_metrics.py tests\application\test_drift_one_pass_evaluation.py tests\application\test_experiment_ui_catalog_fields.py -q
```

Observed result:

```text
[ARCH_GUARD] OK
82 passed, 3 warnings
```

The warnings are known PyTorch / PyTorch-Geometric deprecation warnings and do
not block Phase-2C validation.

## Findings

### P1. Phase-2C completion is not discoverable as a source-of-truth artifact

**Evidence**

Repository search for `Phase-2C`, `Phase 2C`, and related spellings did not
find a completion report, accepted ADR, or current-state status entry.

**Why this matters**

The runtime can be correct, but future agents and reviewers cannot determine
what Phase-2C promised, what was implemented, what was deferred, and which tests
prove readiness. This is a process/reproducibility risk, not a runtime failure.

**Fix plan**

1. Create a short current-state status entry for Phase-2C, either in
   `docs/current/project-state.md` or as a linked file under `docs/current/`.
2. Include:
   - implemented scope;
   - explicitly deferred scope;
   - required presets/config flags;
   - validation commands and last known results;
   - relationship to article/research path.
3. Link it from `docs/index.md`.

**Acceptance criteria**

- `rg -n "Phase-2C|Phase 2C" docs/current docs/index.md` finds the status.
- A new agent can identify the valid runtime mode without reading historical
  worklogs.

---

### P1. `group_by_topology` is advertised more strongly than runtime supports

**Evidence**

Runtime:

```text
src/application/use_cases/trainer.py
training.candidate_batch_topology_policy=group_by_topology is planned but not implemented.
```

Catalog/UI docs:

```text
configs/ui/config_catalog.yaml
group_by_topology builds topology-homogeneous batches by process version and stats snapshot.
```

Matrix:

```text
configs/ui/desktop_ui_field_dependency_matrix.csv
group_by_topology builds topology-homogeneous batches...
```

Tests currently assert that the enum contains `group_by_topology`, but do not
assert that the UI/catalog marks it as reserved or non-runnable.

**Why this matters**

This is the highest strategy-consistency issue. The accepted current strategy is
safe topology-homogeneous batching plus `single_topology_required` guard. Full
in-trainer mixed-batch splitting is deferred. The UI/catalog should not imply
that selecting `group_by_topology` will work.

**Fix plan**

Immediate low-risk fix:

1. Keep runtime behavior unchanged.
2. Update `configs/ui/config_catalog.yaml` description:
   - `single_topology_required`: working mode; DataLoader forms homogeneous
     batches where possible and guard fails mixed batches.
   - `group_by_topology`: reserved/future extension; currently rejected by
     trainer.
3. Update `configs/ui/desktop_ui_field_dependency_matrix.csv` accordingly.
4. Update `docs/GNN_RUNTIME_MVP2_5.MD` and
   `docs/current/architecture-debt.md` wording if any sentence implies
   `group_by_topology` is runnable.
5. Adjust `tests/application/test_experiment_ui_catalog_fields.py` to assert
   reserved wording or remove `group_by_topology` from the selectable enum until
   implemented.

Larger future fix:

1. Implement true mixed-batch splitting inside the trainer:
   - split incoming batch by `process_version_idx + stats_snapshot_version_idx`;
   - run `forward_candidate()` per sub-batch;
   - aggregate loss/metrics with sample-weighted means;
   - preserve trace and forward-stat attribution per subgroup.

**Acceptance criteria**

- Selecting `group_by_topology` is either impossible in the UI or clearly
  marked as reserved and fails with an expected message.
- Documentation and config catalog no longer state that it builds batches today.
- Targeted UI/catalog tests cover this behavior.

---

### P1. Candidate identity default is inconsistent between CLI/catalog and direct graph builder use

**Evidence**

CLI/trainer/catalog default:

```text
training.candidate_identity_mode: topology_native
src/cli.py uses topology_native default
ModelTrainer uses topology_native default
```

Domain builder default:

```text
src/domain/services/dynamic_graph_builder.py
candidate_identity_mode: str = "fixed_vocab_bridge"
```

**Why this matters**

For the current article/research path, `candidate_id + topology_native` is the
intended zero-shot candidate identity contract. Direct service usage, custom
tools, or tests that instantiate `DynamicGraphBuilder` without passing the mode
can silently fall back to fixed-vocabulary candidate identity.

This does not necessarily break the CLI path, but it is a strategy mismatch.

**Fix plan**

1. Decide the intended default boundary:
   - Option A: set `DynamicGraphBuilder` default to `topology_native`;
   - Option B: keep `fixed_vocab_bridge` for backward compatibility, but make
     every candidate-id runtime construction pass `topology_native` explicitly
     and document the builder-level default as legacy.
2. Add a regression test that constructs the Phase-2C graph-builder path through
   the CLI/composition root and verifies:
   - `candidate_identity_mode == topology_native`;
   - future/unseen candidates remain in `candidate_ids`;
   - future candidate edges are not filtered by train vocabulary.
3. If Option B is selected, add a note to `docs/GNN_RUNTIME_MVP2_5.MD` that
   service defaults are backward-compatible, while research presets must pass
   `topology_native`.

**Acceptance criteria**

- No Phase-2C preset can accidentally run with `fixed_vocab_bridge`.
- Direct graph-builder tests make the boundary explicit.

---

### P2. Current-state docs are stale relative to implemented June features

**Evidence**

`docs/current/project-state.md` and `docs/current/architecture-debt.md` still
carry old `last_updated` dates and contain a long chronological update history,
but do not provide a compact Phase-2C readiness summary.

**Why this matters**

The main runtime path is now:

```text
EOPKGTopologyConditioned
candidate_id
topology_native
impulse_activation_routing
stable candidate-native metrics
GATv2+Mask reviewer control
```

This exists in the docs, but is scattered. Agents can still infer it, but the
state is not crisp enough for handoff.

**Fix plan**

1. Add a short "Current validated research runtime" section near the top of
   `docs/current/project-state.md`.
2. Move or compress old chronological runtime updates that are not needed for
   first-read orientation.
3. Keep `docs/current/architecture-debt.md` focused on active debt:
   - group_by_topology mixed-batch splitting;
   - semantic grounding;
   - exact BPMN node supervision;
   - version metadata ordering;
   - real-world external validation.

**Acceptance criteria**

- A first-read agent can identify the current valid Phase-2C experiment stack
  in under one minute.
- Old historical details do not obscure current behavior.

---

### P2. Full-suite validation should be refreshed before release/merge claims

**Evidence**

Targeted tests pass now. Historical reports mention full-suite results, but no
fresh full-suite run was executed during this review.

**Why this matters**

Targeted tests are enough for this review, but not enough for claiming final
release readiness after Phase-2C.

**Fix plan**

Run before final release tag / Zenodo snapshot:

```powershell
.\.venv-modern\Scripts\python.exe -m pytest -m mvp1_regression -v
.\.venv-modern\Scripts\python.exe -m pytest tests\ -v
```

Record the results in the Phase-2C status artifact.

**Acceptance criteria**

- Full suite passes or known failures are documented with explicit acceptance.
- MVP1 regression gate remains green.

---

### P2. Untracked manuscript is present in the repository root

**Evidence**

```text
?? Manuscript.docx
```

**Why this matters**

This does not affect runtime. It can, however, accidentally be committed or
confuse repository hygiene before public release.

**Fix plan**

Choose one:

1. Move manuscript artifacts outside the repository;
2. Add an explicit ignore pattern for manuscript drafts;
3. Track it intentionally only if the paper manuscript is part of the public
   artifact plan.

**Acceptance criteria**

- `git status --short` is clean or intentionally shows only files planned for
  commit.

## Final Review Position

The core Phase-2C runtime path appears technically coherent and targeted tests
cover the important candidate-native and mask/evaluation behaviors. The main
open risks are not broken tests, but consistency and handoff risks:

1. no discoverable Phase-2C completion artifact;
2. `group_by_topology` wording implies a runnable feature that is explicitly
   not implemented;
3. candidate identity defaults are split between CLI/catalog and the domain
   builder;
4. current-state docs need a compact readiness summary.

These should be fixed before treating Phase-2C as cleanly closed.
