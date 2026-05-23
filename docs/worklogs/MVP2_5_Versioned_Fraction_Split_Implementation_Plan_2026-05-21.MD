# MVP2.5 Versioned Fraction and Split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:test-driven-development` or `test-driven-development` to implement this plan task-by-task. Use `superpowers:verification-before-completion` before reporting completion. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add version-aware data subsampling and train/val/test splitting so topology-conditioned experiments can train on small balanced samples from known versions without accidentally leaking future versions.

**Architecture:** Keep current `temporal` behavior as the default. Add explicit `experiment.fraction_strategy`, extend `experiment.split_strategy` with `versioned`, and add `experiment.version_scope_policy=train_cut` to prevent future-version leakage in versioned experiments. Implement shared helpers in `src/cli.py` and mirror the same behavior in `ModelTrainer` until the split pipeline is fully centralized.

**Tech Stack:** Python 3.11, PyTorch Geometric data pipeline, pytest, YAML experiment configs, `configs/ui/config_catalog.yaml`.

---

## 1. Current Problem

Current cascade behavior is:

```text
filter traces with events
-> sort by first event timestamp when split_strategy=temporal
-> macro cut by experiment.train_ratio
-> apply experiment.fraction to the selected macro slice
-> micro split by experiment.split_ratio
```

For versioned logs this can collapse selected data to only `v1`, especially with small `fraction` or low `train_ratio`. That is harmful for `training.learning_strategy=topology_conditioned`, because `wrong_version_topology` requires at least two known train versions.

Example failure mode:

```yaml
experiment.train_ratio: 0.5
experiment.fraction: 0.2
experiment.split_strategy: temporal
```

If traces are ordered by versions, the first selected macro slice can still mostly or only contain `v1`, so the training run has no useful wrong-version topology negatives.

---

## 2. Target Configuration Contract

Add these keys:

```yaml
experiment.fraction_strategy: temporal | versioned
experiment.split_strategy: temporal | none | versioned
experiment.version_scope_policy: all | train_cut
```

Defaults:

```yaml
experiment.fraction_strategy: temporal
experiment.split_strategy: temporal
experiment.version_scope_policy: all
```

Backward compatibility:

- Missing `fraction_strategy` must behave exactly like current behavior.
- Existing `split_strategy: temporal`, `none`, and legacy alias `time` must continue to work.
- `split_strategy: versioned` is new and must not change existing temporal tests.

Recommended topology-conditioned quick experiment:

```yaml
experiment.fraction_strategy: versioned
experiment.split_strategy: versioned
experiment.version_scope_policy: train_cut
experiment.train_ratio: 0.5
experiment.fraction: 0.2
```

Expected effect:

- Take a small balanced fraction from each known version.
- Use only train-cut versions for train/val.
- Keep future versions out of training.
- Preserve enough versions in train to activate `wrong_version_topology`.

---

## 3. Semantics

### 3.1 `fraction_strategy=temporal`

Current behavior:

```text
ordered_macro[:int(len(ordered_macro) * fraction)]
```

Use for strict chronological subsampling.

### 3.2 `fraction_strategy=versioned`

Take `fraction` independently inside each `process_version`.

Per version:

```text
version_traces_sorted_by_timestamp[:int(len(version_traces) * fraction)]
```

Ordering of final output:

- For `split_strategy=temporal`: merge selected traces and sort by timestamp.
- For `split_strategy=none`: preserve original stable order as much as possible.
- For `split_strategy=versioned`: keep deterministic version order, then timestamp order inside each version.

Minimum count:

- If `fraction > 0` and a version has at least one trace, keep at least one trace from that version.
- This differs from the current `_apply_cascade_prepare` behavior where `int(len(macro) * fraction)` can become zero.
- The minimum-one behavior is needed because versioned fraction is explicitly used to preserve version coverage.

### 3.3 `split_strategy=versioned`

Apply `split_ratio` independently inside each selected version:

```text
v1 -> train/val/test by split_ratio
v2 -> train/val/test by split_ratio
...
train = concat(v1.train, v2.train, ...)
val   = concat(v1.val,   v2.val,   ...)
test  = concat(v1.test,  v2.test,  ...)
```

This is a controlled supervised multi-version split.

### 3.4 `version_scope_policy=all`

All selected versions participate in versioned split.

Use case:

```text
supervised multi-version training
```

Risk:

```text
future versions can enter train
```

### 3.5 `version_scope_policy=train_cut`

Purpose: prevent future-version leakage while still allowing balanced version-aware sampling for known versions.

Algorithm:

1. Sort full trace stream by first event timestamp.
2. Compute macro train cut index:

```python
cut_idx = int(len(ordered) * experiment.train_ratio)
```

3. Determine known train versions from traces before `cut_idx`.
4. For `mode=train`:
   - keep only traces whose version is in known train versions;
   - apply `fraction_strategy`;
   - apply `split_strategy`.
5. For `mode=eval_drift` or `eval_cross_dataset`:
   - keep tail traces after `cut_idx`;
   - do not route future tail into train/val.

Important:

- `version_scope_policy=train_cut` does not mean “split by version number only”.
- It uses chronological train cut to discover known versions, which is safer for real logs.
- Future improvement debt: when process versions are non-numeric or appear through metadata timestamps, add explicit `experiment.version_train_cut` / version timeline metadata support.

---

## 4. Files

Modify:

- `src/cli.py`
  - Add shared helpers for version grouping, versioned fraction, and versioned split.
  - Extend `_apply_cascade_prepare`.
  - Extend `_strict_temporal_split` or add a version-aware sibling.
  - Update CLI diagnostics to include `fraction_strategy` and `version_scope_policy`.

- `src/application/use_cases/trainer.py`
  - Mirror the new split semantics in `_prepare_data`, `_prepare_split_data`, and `_strict_temporal_split`.
  - Keep behavior aligned with `src/cli.py`.

- `configs/ui/config_catalog.yaml`
  - Add `experiment.fraction_strategy`.
  - Extend `experiment.split_strategy` enum with `versioned`.
  - Add `experiment.version_scope_policy`.

- `tests/application/test_cascade_data_split.py`
  - Add unit tests for new split/fraction semantics.

- `tests/contracts/test_mvp1_config_contract.py`
  - Extend allowed `split_strategy` values if relevant to config contract.
  - Add optional/default compatibility checks if needed.

- `docs/EVF_MVP2_5.MD`
  - Document the evaluation meaning of version-aware fraction/split.

- `docs/GNN_LEARNING_STRATEGY.MD`
  - Add a short note that `topology_conditioned` needs at least two known train versions for wrong-version negatives.

- `docs/current/project-state.md`
  - Update current runtime behavior summary after implementation.

Optional after implementation:

- Add a short runbook note in `docs/runbooks/mvp2_5-commands.md` with recommended fast topology-conditioned training config.

---

## 5. Task Plan

### Task 1: Add Failing Tests for Versioned Fraction

**Files:**

- Modify: `tests/application/test_cascade_data_split.py`

- [ ] **Step 1: Add a test that versioned fraction keeps representatives from each version**

Add a test using three versions with five traces each:

```python
def test_cascade_prepare_versioned_fraction_keeps_each_version():
    traces = []
    for version_idx, version in enumerate(["v1", "v2", "v3"]):
        for local_idx in range(5):
            ts = float((version_idx * 10) + local_idx)
            traces.append(_build_trace(ts=ts, case_id=f"{version}_c{local_idx}", version=version))

    prepared = _apply_cascade_prepare(
        traces,
        mode="train",
        split_strategy="versioned",
        train_ratio=1.0,
        fraction=0.2,
        fraction_strategy="versioned",
        version_scope_policy="all",
    )

    assert _trace_version_counts(prepared) == {"v1": 1, "v2": 1, "v3": 1}
```

- [ ] **Step 2: Run RED test**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\application\test_cascade_data_split.py::test_cascade_prepare_versioned_fraction_keeps_each_version -v
```

Expected:

```text
FAIL: _apply_cascade_prepare() got an unexpected keyword argument 'fraction_strategy'
```

---

### Task 2: Implement Versioned Fraction Helper in `src/cli.py`

**Files:**

- Modify: `src/cli.py`

- [ ] **Step 1: Add helper functions near `_apply_fraction`**

Add:

```python
def _trace_version_label(trace: RawTrace) -> str:
    return str(getattr(trace, "process_version", "")).strip() or "__missing__"


def _first_event_ts(trace: RawTrace) -> Any:
    return trace.events[0].timestamp if trace.events else 0


def _group_traces_by_version(traces: Sequence[RawTrace]) -> Dict[str, List[RawTrace]]:
    grouped: Dict[str, List[RawTrace]] = {}
    for trace in traces:
        grouped.setdefault(_trace_version_label(trace), []).append(trace)
    return dict(sorted(grouped.items(), key=lambda item: item[0]))


def _apply_versioned_fraction(traces: Sequence[RawTrace], fraction: float) -> List[RawTrace]:
    if fraction <= 0.0 or fraction > 1.0:
        raise ValueError("experiment.fraction must be within (0.0, 1.0].")
    grouped = _group_traces_by_version([trace for trace in traces if trace.events])
    selected: List[RawTrace] = []
    for _, version_traces in grouped.items():
        ordered = sorted(version_traces, key=_first_event_ts)
        if fraction >= 1.0:
            keep_count = len(ordered)
        else:
            keep_count = max(1, int(len(ordered) * fraction))
        selected.extend(ordered[:keep_count])
    return selected
```

- [ ] **Step 2: Extend `_apply_cascade_prepare` signature**

Change:

```python
def _apply_cascade_prepare(
    traces: Sequence[RawTrace],
    *,
    mode: str,
    split_strategy: str,
    train_ratio: float,
    fraction: float,
) -> List[RawTrace]:
```

to:

```python
def _apply_cascade_prepare(
    traces: Sequence[RawTrace],
    *,
    mode: str,
    split_strategy: str,
    train_ratio: float,
    fraction: float,
    fraction_strategy: str = "temporal",
    version_scope_policy: str = "all",
) -> List[RawTrace]:
```

- [ ] **Step 3: Validate strategies**

Use:

```python
if split_strategy == "time":
    split_strategy = "temporal"
if split_strategy not in {"temporal", "none", "versioned"}:
    raise ValueError("experiment.split_strategy must be 'temporal', 'none', or 'versioned'.")
fraction_strategy = str(fraction_strategy).strip().lower()
if fraction_strategy not in {"temporal", "versioned"}:
    raise ValueError("experiment.fraction_strategy must be 'temporal' or 'versioned'.")
version_scope_policy = str(version_scope_policy).strip().lower()
if version_scope_policy not in {"all", "train_cut"}:
    raise ValueError("experiment.version_scope_policy must be 'all' or 'train_cut'.")
```

- [ ] **Step 4: Implement versioned fraction branch**

After macro selection:

```python
if fraction >= 1.0:
    filtered = macro
elif fraction_strategy == "versioned":
    filtered = _apply_versioned_fraction(macro, fraction)
else:
    keep_count = int(len(macro) * fraction)
    filtered = macro[:keep_count]

if split_strategy in {"temporal", "versioned"}:
    return sorted(filtered, key=_first_event_ts)
return filtered
```

- [ ] **Step 5: Run GREEN test**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\application\test_cascade_data_split.py::test_cascade_prepare_versioned_fraction_keeps_each_version -v
```

Expected:

```text
PASSED
```

---

### Task 3: Add Versioned Micro Split

**Files:**

- Modify: `src/cli.py`
- Modify: `src/application/use_cases/trainer.py`
- Modify: `tests/application/test_cascade_data_split.py`

- [ ] **Step 1: Add failing trainer test for versioned split**

Add:

```python
def test_trainer_versioned_micro_split_applies_split_ratio_per_version():
    traces = []
    for version_idx, version in enumerate(["v1", "v2"]):
        for local_idx in range(10):
            ts = float((version_idx * 100) + local_idx)
            traces.append(_build_trace(ts=ts, case_id=f"{version}_c{local_idx}", version=version))

    trainer = _trainer(
        experiment_config={
            "split_strategy": "versioned",
            "fraction_strategy": "versioned",
            "version_scope_policy": "all",
            "train_ratio": 1.0,
            "fraction": 1.0,
            "split_ratio": [0.5, 0.2, 0.3],
        },
        mode="train",
    )

    prepared = trainer._prepare_data(traces, mode="train")
    split = trainer._prepare_split_data(prepared)

    assert _trace_version_counts(split.train) == {"v1": 5, "v2": 5}
    assert _trace_version_counts(split.val) == {"v1": 2, "v2": 2}
    assert _trace_version_counts(split.test) == {"v1": 3, "v2": 3}
```

- [ ] **Step 2: Run RED test**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\application\test_cascade_data_split.py::test_trainer_versioned_micro_split_applies_split_ratio_per_version -v
```

Expected:

```text
FAIL: Unsupported experiment.split_strategy 'versioned'
```

- [ ] **Step 3: Add `_versioned_split` helper in trainer**

In `ModelTrainer`, add:

```python
def _versioned_split(
    self,
    traces: Sequence[RawTrace],
    split_ratio: Tuple[float, float, float],
) -> SplitData:
    grouped: Dict[str, List[RawTrace]] = {}
    for trace in traces:
        version = str(getattr(trace, "process_version", "")).strip() or "__missing__"
        grouped.setdefault(version, []).append(trace)

    train: List[RawTrace] = []
    val: List[RawTrace] = []
    test: List[RawTrace] = []
    train_ratio, val_ratio, _ = split_ratio
    for _, version_traces in sorted(grouped.items(), key=lambda item: item[0]):
        ordered = sorted([trace for trace in version_traces if trace.events], key=lambda tr: tr.events[0].timestamp)
        train_end = int(len(ordered) * train_ratio)
        val_end = train_end + int(len(ordered) * val_ratio)
        train.extend(ordered[:train_end])
        val.extend(ordered[train_end:val_end])
        test.extend(ordered[val_end:])
    return SplitData(train=train, val=val, test=test)
```

- [ ] **Step 4: Route `split_strategy=versioned` in `_strict_temporal_split`**

Change validation from:

```python
if split_strategy not in {"temporal", "none"}:
```

to:

```python
if split_strategy not in {"temporal", "none", "versioned"}:
```

And add first branch:

```python
if split_strategy == "versioned":
    return self._versioned_split(ordered, split_ratio)
```

- [ ] **Step 5: Mirror helper in `src/cli.py`**

Add a CLI-level `_versioned_split` returning tuple:

```python
def _versioned_split(
    traces: Sequence[RawTrace],
    split_ratio: Tuple[float, float, float],
) -> Tuple[List[RawTrace], List[RawTrace], List[RawTrace]]:
    grouped = _group_traces_by_version([trace for trace in traces if trace.events])
    train: List[RawTrace] = []
    val: List[RawTrace] = []
    test: List[RawTrace] = []
    train_ratio, val_ratio, _ = split_ratio
    for _, version_traces in grouped.items():
        ordered = sorted(version_traces, key=_first_event_ts)
        train_end = int(len(ordered) * train_ratio)
        val_end = train_end + int(len(ordered) * val_ratio)
        train.extend(ordered[:train_end])
        val.extend(ordered[train_end:val_end])
        test.extend(ordered[val_end:])
    return train, val, test
```

Route in CLI `_strict_temporal_split`.

- [ ] **Step 6: Run GREEN test**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\application\test_cascade_data_split.py::test_trainer_versioned_micro_split_applies_split_ratio_per_version -v
```

Expected:

```text
PASSED
```

---

### Task 4: Implement `version_scope_policy=train_cut`

**Files:**

- Modify: `src/cli.py`
- Modify: `src/application/use_cases/trainer.py`
- Modify: `tests/application/test_cascade_data_split.py`

- [ ] **Step 1: Add failing test for leakage prevention**

Add:

```python
def test_version_scope_train_cut_excludes_future_versions_from_train_prepare():
    traces = []
    for version_idx, version in enumerate(["v1", "v2", "v3", "v4"]):
        for local_idx in range(5):
            ts = float((version_idx * 10) + local_idx)
            traces.append(_build_trace(ts=ts, case_id=f"{version}_c{local_idx}", version=version))

    prepared = _apply_cascade_prepare(
        traces,
        mode="train",
        split_strategy="versioned",
        train_ratio=0.5,
        fraction=0.4,
        fraction_strategy="versioned",
        version_scope_policy="train_cut",
    )

    assert _trace_version_counts(prepared) == {"v1": 2, "v2": 2}
```

- [ ] **Step 2: Add failing test for eval tail**

Add:

```python
def test_version_scope_train_cut_eval_drift_uses_future_tail_only():
    traces = []
    for version_idx, version in enumerate(["v1", "v2", "v3", "v4"]):
        for local_idx in range(5):
            ts = float((version_idx * 10) + local_idx)
            traces.append(_build_trace(ts=ts, case_id=f"{version}_c{local_idx}", version=version))

    prepared = _apply_cascade_prepare(
        traces,
        mode="eval_drift",
        split_strategy="versioned",
        train_ratio=0.5,
        fraction=0.4,
        fraction_strategy="versioned",
        version_scope_policy="train_cut",
    )

    assert _trace_version_counts(prepared) == {"v3": 2, "v4": 2}
```

- [ ] **Step 3: Run RED tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\application\test_cascade_data_split.py::test_version_scope_train_cut_excludes_future_versions_from_train_prepare tests\application\test_cascade_data_split.py::test_version_scope_train_cut_eval_drift_uses_future_tail_only -v
```

Expected:

```text
FAIL
```

- [ ] **Step 4: Implement scope helper**

In `src/cli.py`, add:

```python
def _apply_version_scope(
    ordered: Sequence[RawTrace],
    *,
    mode: str,
    train_ratio: float,
    version_scope_policy: str,
) -> List[RawTrace]:
    split_idx = int(len(ordered) * train_ratio)
    mode_key = str(mode).strip().lower()
    if version_scope_policy == "all":
        if mode_key == "train":
            return list(ordered[:split_idx])
        if mode_key in {"eval_drift", "eval_cross_dataset"}:
            return list(ordered[split_idx:])
        return list(ordered)

    known_versions = {_trace_version_label(trace) for trace in ordered[:split_idx]}
    if mode_key == "train":
        return [trace for trace in ordered if _trace_version_label(trace) in known_versions]
    if mode_key in {"eval_drift", "eval_cross_dataset"}:
        return [trace for trace in ordered[split_idx:] if _trace_version_label(trace) not in known_versions]
    return list(ordered)
```

Use it inside `_apply_cascade_prepare` after `ordered` is created.

- [ ] **Step 5: Mirror scope helper in trainer**

Add equivalent private methods in `ModelTrainer`:

```python
@staticmethod
def _trace_version_label(trace: RawTrace) -> str:
    return str(getattr(trace, "process_version", "")).strip() or "__missing__"

def _apply_version_scope(
    self,
    ordered: Sequence[RawTrace],
    *,
    mode: str,
    train_ratio: float,
    version_scope_policy: str,
) -> List[RawTrace]:
    split_idx = int(len(ordered) * train_ratio)
    mode_key = str(mode).strip().lower()
    if version_scope_policy == "all":
        if mode_key == "train":
            return list(ordered[:split_idx])
        if mode_key in {"eval_drift", "eval_cross_dataset"}:
            return list(ordered[split_idx:])
        return list(ordered)

    known_versions = {self._trace_version_label(trace) for trace in ordered[:split_idx]}
    if mode_key == "train":
        return [trace for trace in ordered if self._trace_version_label(trace) in known_versions]
    if mode_key in {"eval_drift", "eval_cross_dataset"}:
        return [trace for trace in ordered[split_idx:] if self._trace_version_label(trace) not in known_versions]
    return list(ordered)
```

- [ ] **Step 6: Run GREEN tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\application\test_cascade_data_split.py -v
```

Expected:

```text
PASSED
```

---

### Task 5: Add UI Catalog Keys

**Files:**

- Modify: `configs/ui/config_catalog.yaml`
- Modify: `tests/application/test_experiment_ui_catalog_fields.py`

- [ ] **Step 1: Add failing UI catalog test**

Add:

```python
def test_catalog_includes_versioned_fraction_split_fields():
    catalog = _load_catalog()
    assert "experiment.fraction_strategy" in catalog["fields"]
    assert "experiment.version_scope_policy" in catalog["fields"]
    split_enum = catalog["fields"]["experiment.split_strategy"]["enum"]
    assert "versioned" in split_enum
```

Use the local catalog loading helper already present in the file.

- [ ] **Step 2: Run RED test**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\application\test_experiment_ui_catalog_fields.py::test_catalog_includes_versioned_fraction_split_fields -v
```

Expected:

```text
FAIL
```

- [ ] **Step 3: Update `configs/ui/config_catalog.yaml`**

Add field:

```yaml
  experiment.fraction_strategy:
    path: experiment.fraction_strategy
    section: experiment
    label: Fraction strategy
    description: Data fraction strategy. temporal preserves chronological sampling; versioned samples independently per process version.
    affects: Data split and experiment validity.
    default: temporal
    enum:
    - temporal
    - versioned
    required_in_modes: []
    required_when: {}
    ui:
      tab: general
      group: advanced
      priority: 100
      order: 1000
```

Extend `experiment.split_strategy.enum`:

```yaml
    enum:
    - temporal
    - none
    - versioned
```

Add:

```yaml
  experiment.version_scope_policy:
    path: experiment.version_scope_policy
    section: experiment
    label: Version scope policy
    description: Version leakage control for versioned fraction/split. all uses all selected versions; train_cut restricts train to versions known before train_ratio cut and keeps future versions for eval tail.
    affects: Data split and future-version leakage control.
    default: all
    enum:
    - all
    - train_cut
    required_in_modes: []
    required_when: {}
    ui:
      tab: general
      group: advanced
      priority: 100
      order: 1000
```

- [ ] **Step 4: Run GREEN test**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\application\test_experiment_ui_catalog_fields.py::test_catalog_includes_versioned_fraction_split_fields -v
```

Expected:

```text
PASSED
```

---

### Task 6: Update CLI Config Parsing and Logging

**Files:**

- Modify: `src/cli.py`
- Modify: `tests/application/test_cli_flatten_config.py` if config flattening expectations need explicit coverage

- [ ] **Step 1: Parse new config values in CLI main path**

Near current parsing:

```python
fraction = float(experiment_cfg.get("fraction", 1.0))
split_strategy = str(experiment_cfg.get("split_strategy", "temporal")).strip().lower()
train_ratio = float(experiment_cfg.get("train_ratio", 0.7))
```

Add:

```python
fraction_strategy = str(experiment_cfg.get("fraction_strategy", "temporal")).strip().lower()
version_scope_policy = str(experiment_cfg.get("version_scope_policy", "all")).strip().lower()
```

Pass these into `_apply_cascade_prepare`.

- [ ] **Step 2: Log diagnostics**

Update existing log:

```python
logger.info("Applied cascade filter (mode=%s, fraction=%.4f) -> traces=%d", mode, fraction, len(traces))
```

to include:

```python
logger.info(
    "Applied cascade filter (mode=%s, fraction=%.4f, fraction_strategy=%s, split_strategy=%s, version_scope_policy=%s) -> traces=%d",
    mode,
    fraction,
    fraction_strategy,
    split_strategy,
    version_scope_policy,
    len(traces),
)
```

- [ ] **Step 3: Include values in run context diagnostics**

Where split diagnostics are logged or params are built, include:

```python
"fraction_strategy": fraction_strategy,
"version_scope_policy": version_scope_policy,
```

- [ ] **Step 4: Run targeted CLI tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\application\test_cli_flatten_config.py tests\application\test_cascade_data_split.py -v
```

Expected:

```text
PASSED
```

---

### Task 7: Update Contracts and Docs

**Files:**

- Modify: `tests/contracts/test_mvp1_config_contract.py`
- Modify: `docs/EVF_MVP2_5.MD`
- Modify: `docs/GNN_LEARNING_STRATEGY.MD`
- Modify: `docs/current/project-state.md`

- [ ] **Step 1: Extend split strategy contract**

Change:

```python
assert experiment["split_strategy"] in {"temporal", "none", "time"}
```

to:

```python
assert experiment["split_strategy"] in {"temporal", "none", "time", "versioned"}
```

- [ ] **Step 2: Document versioned split in `docs/EVF_MVP2_5.MD`**

Add under experiment reproducibility or metrics notes:

```markdown
For topology-conditioned studies, `experiment.fraction_strategy=versioned`
samples a fraction per process version instead of taking only the earliest
chronological slice. `experiment.split_strategy=versioned` applies
`split_ratio` per version. Use `experiment.version_scope_policy=train_cut` when
future-version leakage must be prevented.
```

- [ ] **Step 3: Document learning strategy dependency**

Add to `docs/GNN_LEARNING_STRATEGY.MD`:

```markdown
`topology_conditioned` wrong-version negatives require at least two known train
versions. When training data is intentionally small, prefer
`fraction_strategy=versioned`, `split_strategy=versioned`, and
`version_scope_policy=train_cut`.
```

- [ ] **Step 4: Update current project state**

Add a short note that version-aware fraction/split is available and preserves future-version leakage control.

- [ ] **Step 5: Run doc-adjacent tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\contracts\test_mvp1_config_contract.py tests\application\test_experiment_ui_catalog_fields.py -v
```

Expected:

```text
PASSED
```

---

### Task 8: Full Verification

**Files:**

- No new files.

- [ ] **Step 1: Run cascade and UI tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\application\test_cascade_data_split.py tests\application\test_experiment_ui_catalog_fields.py -v
```

Expected:

```text
PASSED
```

- [ ] **Step 2: Run architecture guard**

Run:

```powershell
.\.venv\Scripts\python.exe tools\architecture_guard.py
```

Expected:

```text
[ARCH_GUARD] OK
```

- [ ] **Step 3: Run full test suite**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\ -v
```

Expected:

```text
all tests passed
```

- [ ] **Step 4: Optional smoke config**

Run a small training smoke manually through the UI/CLI with:

```yaml
experiment:
  mode: train
  fraction: 0.2
  fraction_strategy: versioned
  split_strategy: versioned
  version_scope_policy: train_cut
  train_ratio: 0.5
  split_ratio: [0.5, 0.2, 0.3]
training:
  learning_strategy: topology_conditioned
```

Expected run profile:

```text
data_train_versions includes at least v1 and v2
data_test_versions does not include future versions in train mode unless split_ratio test portion is same known versions
train_topology_conditioned_wrong_version_ce_delta is logged when more than one train version exists
```

---

## 6. Risks and Decisions

### Risk: duplicated split logic in `src/cli.py` and `ModelTrainer`

Current project already has split logic in both places. This plan mirrors behavior to keep the change scoped. Long-term debt is to centralize split/fraction policy in an application service.

Add to architecture debt if not already present:

```text
Centralize experiment data split/fraction policy to avoid divergence between CLI preparation and ModelTrainer preparation.
```

### Risk: version ordering by label

This plan groups by `process_version` label and sorts labels lexicographically. For `v1`, `v2`, `v3`, `v4` this is fine. If versions become non-numeric or metadata-based, version ordering must use explicit version timeline metadata.

Existing agreed debt:

```text
If no numeric version exists and version is described by metadata/timestamp, add version ordering through metadata timeline.
```

### Risk: small per-version sample counts

`fraction_strategy=versioned` keeps at least one trace per version. This preserves coverage but can create tiny validation/test slices when combined with `split_strategy=versioned`. Experiments must report version counts.

---

## 7. Acceptance Criteria

Implementation is complete when:

1. Existing `temporal` and `none` behavior remains unchanged.
2. `fraction_strategy=versioned` preserves representatives from each version.
3. `split_strategy=versioned` applies `split_ratio` per version.
4. `version_scope_policy=train_cut` prevents future versions from entering train preparation.
5. UI catalog exposes all new keys.
6. Full tests pass.
7. Docs explain when to use versioned fraction/split for `topology_conditioned`.

