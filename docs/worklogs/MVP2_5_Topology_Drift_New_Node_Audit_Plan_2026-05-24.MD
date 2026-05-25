# MVP2.5 Topology Drift New-Node Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an audit that explains whether drift degradation is caused by added/removed topology nodes, changed transitions, unseen/weakly-seen activity classes, and fixed classifier limitations.

**Architecture:** Add an offline diagnostic path that consumes existing prepared data, topology projection artifacts, activity vocab, model predictions, and drift-window inference records. The audit must not change training semantics. It produces JSON/CSV/Markdown artifacts that connect version topology diffs to prediction failures.

**Tech Stack:** Python, PyTorch/PyG graph artifacts, existing `DynamicGraphBuilder` contracts, MLflow local `mlruns`, pytest, existing CLI/runbook patterns.

---

## Context

Current evidence shows that `StructXAttn + pre_norm_context` removed the Post-LN amplification bug, but did not improve drift. The likely deeper issue is that each new process version can add or remove BPMN nodes and activity classes. The current model head is a fixed parametric classifier over `activity_vocab`, while business drift changes the candidate set itself.

The audit must answer:

1. Are `v2/v3/v4` adding/removing topology nodes or activity labels compared with train versions?
2. Are strict-F1 errors concentrated around changed transitions such as `A -> B` in old topology vs `A -> C` in new topology?
3. Are model failures caused by targets that are unseen or weakly-seen in the training vocab?
4. Does `allowed_target_mask` correctly reflect the new version candidate set?
5. Does the baseline look better because it is less anchored to obsolete structural transitions?

---

## Scope

### In Scope

- Offline analysis only.
- Works on completed train/eval_drift runs.
- Uses existing `mlruns`, checkpoint/config artifacts, prepared data metadata, and graph dataset cache when available.
- Produces:
  - version topology diff summary;
  - activity label diff summary;
  - transition diff summary;
  - prediction error attribution;
  - recommendations for fixed-head vs dynamic candidate-scoring architecture.

### Out of Scope

- No model retraining.
- No new fusion mode.
- No non-parametric retrieval head implementation.
- No changes to graph builder semantics.
- No automatic dissertation claim; this audit only provides evidence.

---

## Files

- Create: `src/application/services/topology_drift_audit.py`
  - Pure analysis functions and dataclasses.
- Create: `tools/audit_topology_drift.py`
  - CLI entry point for running the audit against run IDs/configs.
- Create: `tests/application/test_topology_drift_audit.py`
  - Unit tests for topology/activity/transition diff and error attribution.
- Modify: `docs/runbooks/mvp2_5-commands.md`
  - Add command examples after implementation.
- Modify: `docs/GNN_LEARNING_STRATEGY.MD`
  - Add audit as required diagnostic before claiming zero-shot benefit.
- Modify: `docs/dissertation_changes.MD`
  - Add note about fixed classifier vs dynamic topology candidate set.
- Create after execution: `docs/worklogs/MVP2_5_Topology_Drift_New_Node_Audit_Report_YYYY-MM-DD.MD`
  - Result summary for concrete runs.

---

## Audit Outputs

The tool should write all outputs under:

```text
outputs/audits/topology_drift/<audit_id>/
```

Required files:

```text
summary.json
version_topology_diff.csv
activity_label_diff.csv
transition_diff.csv
prediction_error_attribution.csv
by_version_metrics.csv
report.md
```

### `summary.json`

Required shape:

```json
{
  "audit_id": "20260524_loan_v1_v4_XAt_TC_vs_Base",
  "dataset": "loan_v1_v4_simulated",
  "train_versions": ["v1", "v2"],
  "eval_versions": ["v1", "v2", "v3", "v4"],
  "runs": {
    "baseline": "7681dd2ef7f841e68591e0b1145296ae",
    "structural": "5af1d05da0524106b4031dd7c39767c6"
  },
  "findings": {
    "new_activity_labels_present": true,
    "removed_activity_labels_present": true,
    "changed_transition_error_concentration": 0.0,
    "unseen_target_error_rate": 0.0,
    "fixed_head_limitation_detected": false
  }
}
```

### `prediction_error_attribution.csv`

Required columns:

```text
run_id,model_label,trace_idx,step,process_version,prefix_last_activity,
target_label,pred_label,strict_correct,pred_in_mask,target_in_mask,
strict_error_but_allowed,is_oos,mask_cardinality,
target_seen_in_train,target_count_in_train,target_is_new_in_eval_version,
pred_removed_in_eval_version,prefix_transition_changed,
old_allowed_targets,new_allowed_targets,error_bucket
```

`error_bucket` values:

```text
correct
strict_error_but_allowed
oos_prediction
target_not_in_mask
unseen_target_class
removed_prediction_class
changed_transition_zone
unknown
```

---

## Task 1: Add Pure Topology Diff Service

**Files:**

- Create: `src/application/services/topology_drift_audit.py`
- Test: `tests/application/test_topology_drift_audit.py`

- [ ] **Step 1: Define dataclasses and signatures**

Add:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Sequence, Set, Tuple


Activity = str
Version = str
Transition = Tuple[Activity, Activity]


@dataclass(frozen=True)
class VersionTopologySnapshot:
    version: Version
    activities: Set[Activity]
    transitions: Set[Transition]


@dataclass(frozen=True)
class VersionTopologyDiff:
    source_version: Version
    target_version: Version
    added_activities: Set[Activity]
    removed_activities: Set[Activity]
    common_activities: Set[Activity]
    added_transitions: Set[Transition]
    removed_transitions: Set[Transition]
    changed_successors: Dict[Activity, Dict[str, Set[Activity]]]


def diff_topology_versions(
    source: VersionTopologySnapshot,
    target: VersionTopologySnapshot,
) -> VersionTopologyDiff:
    ...
```

- [ ] **Step 2: Write failing tests for activity/node diff**

Test case:

```python
def test_diff_topology_versions_reports_added_and_removed_activities():
    source = VersionTopologySnapshot(
        version="v1",
        activities={"A", "B", "D"},
        transitions={("A", "B"), ("B", "D")},
    )
    target = VersionTopologySnapshot(
        version="v2",
        activities={"A", "C", "D"},
        transitions={("A", "C"), ("C", "D")},
    )

    diff = diff_topology_versions(source, target)

    assert diff.added_activities == {"C"}
    assert diff.removed_activities == {"B"}
    assert diff.common_activities == {"A", "D"}
```

Run:

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\application\test_topology_drift_audit.py::test_diff_topology_versions_reports_added_and_removed_activities -v
```

Expected: fail until implementation exists.

- [ ] **Step 3: Implement activity/node diff**

Implementation:

```python
def diff_topology_versions(source: VersionTopologySnapshot, target: VersionTopologySnapshot) -> VersionTopologyDiff:
    added_activities = set(target.activities) - set(source.activities)
    removed_activities = set(source.activities) - set(target.activities)
    common_activities = set(source.activities) & set(target.activities)
    added_transitions = set(target.transitions) - set(source.transitions)
    removed_transitions = set(source.transitions) - set(target.transitions)
    changed_successors = _changed_successors(source.transitions, target.transitions)
    return VersionTopologyDiff(
        source_version=source.version,
        target_version=target.version,
        added_activities=added_activities,
        removed_activities=removed_activities,
        common_activities=common_activities,
        added_transitions=added_transitions,
        removed_transitions=removed_transitions,
        changed_successors=changed_successors,
    )
```

- [ ] **Step 4: Write failing tests for changed transition successors**

Test case:

```python
def test_diff_topology_versions_reports_changed_successors():
    source = VersionTopologySnapshot(
        version="v1",
        activities={"A", "B", "C"},
        transitions={("A", "B"), ("B", "C")},
    )
    target = VersionTopologySnapshot(
        version="v2",
        activities={"A", "B", "C"},
        transitions={("A", "C"), ("B", "C")},
    )

    diff = diff_topology_versions(source, target)

    assert diff.changed_successors == {
        "A": {
            "source_successors": {"B"},
            "target_successors": {"C"},
            "removed_successors": {"B"},
            "added_successors": {"C"},
        }
    }
```

- [ ] **Step 5: Implement `_changed_successors`**

Implementation:

```python
def _successors(transitions: Iterable[Transition]) -> Dict[Activity, Set[Activity]]:
    result: Dict[Activity, Set[Activity]] = {}
    for src, dst in transitions:
        result.setdefault(str(src), set()).add(str(dst))
    return result


def _changed_successors(
    source_transitions: Iterable[Transition],
    target_transitions: Iterable[Transition],
) -> Dict[Activity, Dict[str, Set[Activity]]]:
    source_successors = _successors(source_transitions)
    target_successors = _successors(target_transitions)
    changed: Dict[Activity, Dict[str, Set[Activity]]] = {}
    for activity in sorted(set(source_successors) | set(target_successors)):
        src_set = set(source_successors.get(activity, set()))
        dst_set = set(target_successors.get(activity, set()))
        if src_set == dst_set:
            continue
        changed[activity] = {
            "source_successors": src_set,
            "target_successors": dst_set,
            "removed_successors": src_set - dst_set,
            "added_successors": dst_set - src_set,
        }
    return changed
```

---

## Task 2: Add Prediction Error Attribution

**Files:**

- Modify: `src/application/services/topology_drift_audit.py`
- Test: `tests/application/test_topology_drift_audit.py`

- [ ] **Step 1: Define attribution inputs**

Add:

```python
@dataclass(frozen=True)
class PredictionRecord:
    run_id: str
    model_label: str
    trace_idx: int
    step: int
    process_version: Version
    prefix_last_activity: Activity
    target_label: Activity
    pred_label: Activity
    strict_correct: bool
    pred_in_mask: bool
    target_in_mask: bool
    strict_error_but_allowed: bool
    mask_cardinality: int


@dataclass(frozen=True)
class PredictionAttribution:
    record: PredictionRecord
    target_seen_in_train: bool
    target_count_in_train: int
    target_is_new_in_eval_version: bool
    pred_removed_in_eval_version: bool
    prefix_transition_changed: bool
    old_allowed_targets: Set[Activity]
    new_allowed_targets: Set[Activity]
    error_bucket: str
```

- [ ] **Step 2: Write failing tests for unseen target attribution**

Test case:

```python
def test_attribute_prediction_marks_unseen_target_class():
    record = PredictionRecord(
        run_id="r1",
        model_label="struct",
        trace_idx=1,
        step=10,
        process_version="v2",
        prefix_last_activity="A",
        target_label="C",
        pred_label="B",
        strict_correct=False,
        pred_in_mask=True,
        target_in_mask=True,
        strict_error_but_allowed=True,
        mask_cardinality=2,
    )
    diff = VersionTopologyDiff(
        source_version="v1",
        target_version="v2",
        added_activities={"C"},
        removed_activities=set(),
        common_activities={"A", "B"},
        added_transitions={("A", "C")},
        removed_transitions={("A", "B")},
        changed_successors={
            "A": {
                "source_successors": {"B"},
                "target_successors": {"C"},
                "removed_successors": {"B"},
                "added_successors": {"C"},
            }
        },
    )

    attr = attribute_prediction_error(
        record,
        train_activity_counts={"A": 100, "B": 100},
        topology_diff=diff,
    )

    assert attr.target_seen_in_train is False
    assert attr.target_is_new_in_eval_version is True
    assert attr.prefix_transition_changed is True
    assert attr.error_bucket == "unseen_target_class"
```

- [ ] **Step 3: Implement `attribute_prediction_error`**

Decision order:

```python
def attribute_prediction_error(
    record: PredictionRecord,
    *,
    train_activity_counts: Mapping[Activity, int],
    topology_diff: VersionTopologyDiff,
) -> PredictionAttribution:
    target_count = int(train_activity_counts.get(record.target_label, 0))
    target_seen = target_count > 0
    target_new = record.target_label in topology_diff.added_activities
    pred_removed = record.pred_label in topology_diff.removed_activities
    successor_change = topology_diff.changed_successors.get(record.prefix_last_activity, {})
    old_allowed = set(successor_change.get("source_successors", set()))
    new_allowed = set(successor_change.get("target_successors", set()))
    transition_changed = bool(old_allowed or new_allowed)

    if record.strict_correct:
        bucket = "correct"
    elif not target_seen:
        bucket = "unseen_target_class"
    elif pred_removed:
        bucket = "removed_prediction_class"
    elif not record.pred_in_mask:
        bucket = "oos_prediction"
    elif not record.target_in_mask:
        bucket = "target_not_in_mask"
    elif transition_changed:
        bucket = "changed_transition_zone"
    elif record.strict_error_but_allowed:
        bucket = "strict_error_but_allowed"
    else:
        bucket = "unknown"

    return PredictionAttribution(
        record=record,
        target_seen_in_train=target_seen,
        target_count_in_train=target_count,
        target_is_new_in_eval_version=target_new,
        pred_removed_in_eval_version=pred_removed,
        prefix_transition_changed=transition_changed,
        old_allowed_targets=old_allowed,
        new_allowed_targets=new_allowed,
        error_bucket=bucket,
    )
```

---

## Task 3: Build MLflow/Artifact Loader for Audit Inputs

**Files:**

- Create: `tools/audit_topology_drift.py`
- Test: `tests/application/test_topology_drift_audit.py`

- [ ] **Step 1: Add CLI arguments**

Required command:

```powershell
.\.venv-modern\Scripts\python.exe tools\audit_topology_drift.py `
  --mlruns-dir mlruns `
  --experiment-id 854778689611649472 `
  --baseline-run 7681dd2ef7f841e68591e0b1145296ae `
  --structural-run 5af1d05da0524106b4031dd7c39767c6 `
  --train-run 9de636ce471745d9b818df6edefd73c2 `
  --output-dir outputs\audits\topology_drift
```

- [ ] **Step 2: Implement safe local MLflow file reading**

Functions:

```python
def read_run_param(run_dir: Path, key: str) -> str | None:
    path = run_dir / "params" / key
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8").strip()


def read_metric_series(run_dir: Path, key: str) -> list[tuple[int, float]]:
    path = run_dir / "metrics" / key
    if not path.exists():
        return []
    rows: list[tuple[int, float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 3:
            rows.append((int(parts[2]), float(parts[1])))
    return rows
```

- [ ] **Step 3: Load existing structural trace fallback JSONL**

Find:

```text
mlruns/<experiment_id>/<run_id>/artifacts/structural_traces_run_<run_id>_*.jsonl
```

Parse fields:

```text
attributes.process_version
attributes.strict_correct
attributes.pred_in_mask
attributes.target_in_mask
attributes.strict_error_but_allowed
attributes.mask_cardinality
attributes.pred_index
attributes.target_index
inputs.contract.trace_idx
outputs.top_predictions
```

If trace artifacts are absent, the tool must still generate `summary.json` and mark:

```json
"prediction_error_attribution_available": false
```

---

## Task 4: Extract Version Topology Snapshots

**Files:**

- Modify: `tools/audit_topology_drift.py`
- Modify: `src/application/services/topology_drift_audit.py`
- Test: `tests/application/test_topology_drift_audit.py`

- [ ] **Step 1: Prefer graph artifact topology**

Read graph dataset cache artifacts referenced by the run config when available. For each version, collect:

```text
activities: reverse activity labels from `struct_node_to_class_index`
transitions: class-label edges from `structural_edge_index` mapped through `struct_node_to_class_index`
```

This matches the exact projected topology seen by the model.

- [ ] **Step 2: Fallback to topology repository/export**

If graph artifacts are not accessible, fallback to config/topology JSON artifacts if present. If neither exists, emit:

```json
"topology_diff_available": false
```

and continue with MLflow metric-only summary.

- [ ] **Step 3: Add duplicate label diagnostics**

For each version, calculate:

```text
struct_nodes_total
classes_total
duplicate_class_node_count
duplicate_class_labels
```

This detects whether several BPMN nodes collapse into one activity class, which can hide true new-node behavior.

---

## Task 5: Generate CSV/JSON/Markdown Reports

**Files:**

- Modify: `tools/audit_topology_drift.py`
- Test: `tests/application/test_topology_drift_audit.py`

- [ ] **Step 1: Write CSV serializers**

Use Python stdlib `csv.DictWriter`. Do not introduce pandas dependency.

Required CSVs:

```text
version_topology_diff.csv
activity_label_diff.csv
transition_diff.csv
prediction_error_attribution.csv
by_version_metrics.csv
```

- [ ] **Step 2: Write `report.md`**

Required sections:

```markdown
# Topology Drift Audit Report

## Scope
## Runs
## Version Topology Diff
## Activity Label Diff
## Transition Diff
## Prediction Error Attribution
## Fixed Classifier Head Risk
## Methodology Implications
## Recommended Next Experiments
```

- [ ] **Step 3: Add explicit interpretation rules**

Report must label evidence as:

```text
supports_fixed_head_limitation
supports_changed_transition_failure
supports_mask_alignment_failure
supports_structural_negative_transfer
insufficient_evidence
```

---

## Task 6: Add Tests for End-to-End Synthetic Audit

**Files:**

- Test: `tests/application/test_topology_drift_audit.py`

- [ ] **Step 1: Build synthetic v1/v2 scenario**

Synthetic topology:

```text
v1: A -> B -> D
v2: A -> C -> D
```

Training activity counts:

```python
{"A": 10, "B": 10, "D": 10}
```

Prediction:

```text
prefix_last=A, target=C, pred=B
```

Expected:

```text
target_is_new_in_eval_version = true
prefix_transition_changed = true
error_bucket = unseen_target_class
```

- [ ] **Step 2: Build removed-node scenario**

Synthetic topology:

```text
v1: A -> B
v2: A -> C
```

Prediction:

```text
prefix_last=A, target=C, pred=B
```

Expected:

```text
pred_removed_in_eval_version = true
error_bucket = removed_prediction_class
```

- [ ] **Step 3: Build changed-transition known-class scenario**

Training activity counts:

```python
{"A": 10, "B": 10, "C": 10}
```

Topology:

```text
v1: A -> B
v2: A -> C
```

Prediction:

```text
prefix_last=A, target=C, pred=B
```

Expected:

```text
target_seen_in_train = true
prefix_transition_changed = true
error_bucket = changed_transition_zone
```

---

## Task 7: Documentation Updates

**Files:**

- Modify: `docs/GNN_LEARNING_STRATEGY.MD`
- Modify: `docs/runbooks/mvp2_5-commands.md`
- Modify: `docs/dissertation_changes.MD`

- [ ] **Step 1: Update learning strategy docs**

Add a section:

```markdown
## Topology Drift Validity Audit

Before claiming zero-shot benefit, run the topology drift audit. The audit
separates four failure modes:

- fixed classifier cannot emit new activity classes;
- changed transitions move the correct target from old successor to new successor;
- mask/projection mismatch makes target unavailable;
- structural branch causes negative transfer despite valid candidate set.
```

- [ ] **Step 2: Update runbook**

Add command:

```powershell
.\.venv-modern\Scripts\python.exe tools\audit_topology_drift.py `
  --mlruns-dir mlruns `
  --experiment-id 854778689611649472 `
  --baseline-run <baseline_eval_drift_run_id> `
  --structural-run <structural_eval_drift_run_id> `
  --train-run <train_run_id> `
  --output-dir outputs\audits\topology_drift
```

- [ ] **Step 3: Update dissertation changes**

Add:

```markdown
## YYYY-MM-DD - Dynamic Candidate Set Limitation

The current fixed parametric classifier can only predict classes present in
`activity_vocab`. Business-valid structural drift often adds/removes nodes and
can add new activity labels. This motivates an audit of new-node/new-label
failure modes and may require a future dynamic topology-conditioned candidate
scorer or semantic-topological retrieval head.
```

---

## Task 8: Verification

- [ ] **Step 1: Run targeted tests**

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\application\test_topology_drift_audit.py -v
```

Expected: all tests pass.

- [ ] **Step 2: Run architecture guard**

```powershell
.\.venv-modern\Scripts\python.exe tools\architecture_guard.py
```

Expected:

```text
[ARCH_GUARD] OK: architecture dependency rules satisfied.
```

- [ ] **Step 3: Run relevant application tests**

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\application\test_trainer_topology_conditioned_learning.py tests\application\test_topology_drift_audit.py -v
```

Expected: all tests pass.

- [ ] **Step 4: Run full suite if implementation touched shared trainer/CLI contracts**

```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests\ -v
```

Expected: all tests pass or only known external deprecation warnings.

---

## First Concrete Audit Run

Use the currently discussed runs:

```powershell
.\.venv-modern\Scripts\python.exe tools\audit_topology_drift.py `
  --mlruns-dir mlruns `
  --experiment-id 854778689611649472 `
  --baseline-run 7681dd2ef7f841e68591e0b1145296ae `
  --structural-run 5af1d05da0524106b4031dd7c39767c6 `
  --train-run 9de636ce471745d9b818df6edefd73c2 `
  --output-dir outputs\audits\topology_drift
```

Expected research questions for this run:

1. Does `v3/v4` contain target labels unseen in train versions `v1/v2`?
2. Are structural OOS errors concentrated where old successors were removed?
3. Are strict errors mostly `changed_transition_zone` rather than generic model errors?
4. Does baseline avoid some obsolete-structure errors because it lacks structural anchoring?

---

## Decision Criteria After Audit

### Continue With Current Fixed Head If

- New activity labels are rare or absent.
- Most errors are known-class changed-transition errors.
- `allowed_target_mask` contains the target in most error windows.
- Structural model can be improved by training methodology only.

### Plan Dynamic Candidate Scorer If

- Many drift errors have `target_seen_in_train=false`.
- Many target labels exist in topology but are unavailable as classifier outputs.
- Added nodes/classes explain a large share of post-drift strict-F1 loss.

### Fix Mask/Projection First If

- `target_in_mask=false` is frequent for valid topology targets.
- `struct_node_to_class_index` collapses or drops relevant nodes incorrectly.
- Duplicate activity labels dominate error zones.

---

## Self-Review

- Spec coverage: covers topology node diff, label diff, transition diff, prediction attribution, fixed-head risk, documentation, and first concrete run.
- Placeholder scan: no `TBD` / `TODO` placeholders.
- Type consistency: dataclasses and function names are consistent across tasks.
- Scope check: implementation is limited to offline audit and documentation; no runtime training behavior changes.

