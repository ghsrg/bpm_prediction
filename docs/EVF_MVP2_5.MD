# EVF_MVP2_5.MD

Updated: 2026-05-20
Status: ACTIVE

## 1. Purpose
Evaluation framework for MVP2.5 with offline topology + offline stats snapshots.

## 2. Scientific Validity Goals
1. prevent future leakage,
2. keep reproducible artifact timeline,
3. compare baseline vs EOPKG fairly.

## 3. Stage 3.4 Temporal Policy

### 3.1 Artifact preparation order
1. `ingest-topology` or `sync-topology`
2. `sync-stats` (optional repeated timeline snapshots)
3. train/eval/infer

### 3.2 Snapshot timestamp source
1. explicit `--as-of`: fixed historical cutoff.
2. no `--as-of`: derived snapshot cutoff `effective_as_of=max(event_ts)` after train-cut selection.

### 3.3 Runtime retrieval policy
1. `stats_time_policy = strict_asof`:
   - lookup uses prefix last-event time.
2. `stats_time_policy = latest`:
   - lookup uses latest snapshot/structure.

## 4. Recommended Research Protocol

For drift-safe studies:
1. Build periodic snapshot timeline (for example monthly).
2. Use isolated experiment namespace/storage.
3. Run train/eval with `experiment.stats_time_policy = strict_asof`.
4. Use `experiment.on_missing_asof_snapshot = raise` for dissertation-grade evidence.
5. Keep fixed seed and identical train split for baseline and EOPKG.
6. For small topology-conditioned experiments on versioned logs, use
   `experiment.fraction_strategy=versioned` to sample a fraction per process
   version instead of only the earliest chronological slice.
7. Use `experiment.split_strategy=versioned` when train/val/test must preserve
   per-version proportions.
8. Use `experiment.version_scope_policy=train_cut` when future-version leakage
   must be prevented: train uses only versions known before `train_ratio`, while
   eval tail keeps future versions for drift/zero-shot checks.

## 5. Leakage Risk Notes
1. If no eligible snapshot exists for requested `as_of`, runtime behavior is controlled by `experiment.on_missing_asof_snapshot`.
2. Fallback policies (`disable_stats`, `use_base`) must be explicitly reported in experiment notes and treated as exploratory.
3. For strict historical studies, ensure snapshots cover required timeline points.

## 6. Metrics
Mandatory:
1. Accuracy
2. F1 macro
3. ECE
4. OOS (when mask exists)

`top3_accuracy` is a diagnostic metric. For small class spaces the runtime uses
an effective `k = min(3, n_classes - 1)` when `n_classes > 2`; for binary or
single-class windows it falls back to plain accuracy. This avoids meaningless
perfect top-k scores when `k >= n_classes`.

Optional sliced metrics:
1. by process version
2. by prefix length bins

Qualitative diagnostics:

1. Selective structural prediction traces may be enabled through
   `tracking.tracing.*`.
2. Traces explain individual predictions and structural-path contribution.
3. Traces are not primary evidence for article/dissertation conclusions.
4. Primary evidence remains aggregate metrics, drift-window metrics,
   calibration/ECE, OOS/mask diagnostics, and seed variance.

## 7. Stats Enrichment Audit Metrics
Track in reports:
1. `history_coverage_percent`
2. `cleaned_instances_percent`
3. `fallback_triggered`
4. `knowledge_version`
5. effective `as_of_ts`

## 7.1 Learning Strategy Evaluation Notes

`training.learning_strategy` is part of experiment interpretation:

1. `standard` is the control strategy and must preserve previous training behavior.
2. `topology_conditioned` adds train-only topology dependency objectives while
   keeping the selected `model.fusion_mode` unchanged.
3. Compare `standard` vs `topology_conditioned` with identical data split,
   seed set, `fusion_mode`, mask setting, and stats setting.
4. Track:
   - `topology_conditioned_wrong_version_ce_delta`
   - `topology_conditioned_drop_edges_ce_delta`
   - `topology_conditioned_allowed_set_loss`
   - `topology_conditioned_retention_loss`
   - `topology_conditioned_mixed_version_batches`
   - `topology_conditioned_wrong_version_skipped_mixed_batches`
5. Positive `wrong_version_ce_delta` means the correct topology gives lower CE
   than a known wrong-version topology.
6. Mixed-version batch counters must be reported when interpreting transition
   windows.

## 8. Experiment Reproducibility Checklist
1. Save config artifact.
2. Save sync-topology summary.
3. Save sync-stats summary (with `as_of` value).
4. Log backend mode and storage path.
5. Log selected process/version scope.
6. Run non-regression tests.

## 9. Validation Commands
```bash
.\.venv\Scripts\python.exe main.py sync-topology --config <sync_topology.yaml> --out outputs/sync_topology.json
.\.venv\Scripts\python.exe main.py sync-stats --config <sync_stats.yaml> --as-of 2024-01-01T00:00:00Z --out outputs/sync_stats_asof.json
.\.venv\Scripts\python.exe main.py --config <train_or_eval.yaml>
.\.venv\Scripts\python.exe -m pytest -m mvp1_regression -v
```

## 10. Evaluation Timeline Diagram
```mermaid
flowchart LR
    T0[Build structure artifacts] --> T1[Build stats snapshots timeline]
    T1 --> T2[Train with strict_asof]
    T2 --> T3[Eval and drift windows]
    T3 --> T4[Report with snapshot metadata]
```
