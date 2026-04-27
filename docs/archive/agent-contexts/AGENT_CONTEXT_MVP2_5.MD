# AGENT_CONTEXT_MVP2_5.MD

Last updated: 2026-03-26  
Status: deprecated-for-agent-first-read; Stage 4.2 historical context.

> **Deprecated / Redirect (2026-04-27):**
> This file is historical context and is no longer part of the default
> first-read path for agents.
>
> Current agent entrypoint:
> 1. `AGENTS.MD`
> 2. `docs/current/project-state.md`
> 3. `docs/current/architecture-debt.md`
>
> Use `docs/current/project-state.md` for current MVP2.5 state and
> `docs/runbooks/mvp2_5-commands.md` for operational commands.

## 1) Current factual state

Implemented:
1. Offline structure ingestion and bulk sync (`ingest-topology`, `sync-topology`).
2. Repository-backed runtime path via `IKnowledgeGraphPort` (`file`, `in_memory`, `neo4j`).
3. Immutable snapshot policy (strict key, JSON payload, as-of lookup, no TTL).
4. Universal stats contract v1 in `sync-stats`.
5. Producer quality/alignment gates in `sync-stats`.
6. Consumer quality gate in `DynamicGraphBuilder`.
7. Config-driven stats mapping (`node_numeric`/`edge_weight`) with `encoding` (`identity|log1p|z-score`).
8. Forward-path telemetry for snapshot metadata in train/eval/inference.
9. XES stats fixes (version binding, no-data skip, auto as-of by train-cut).

## 2) Canonical runtime rules

1. Ingestion/sync and train/eval/infer are physically separated.
2. Train/eval/infer must not rebuild topology/stats from raw sources synchronously.
3. For temporal-safe studies:
   - use `experiment.stats_time_policy = strict_asof`,
   - prebuild snapshots timeline (`sync-stats-backfill` recommended).
4. If snapshot is missing/degraded, stats branch may be disabled by policy without crashing the whole run.

## 3) Open sync tasks

1. Remove stale "next step" wording in canonical docs where implementation already exists.
2. Align timestamp semantics documentation:
   - no `--as-of` means derived `effective_as_of=max(event_ts)` after train-cut, not "current UTC".
3. Align default policy documentation:
   - current runtime default for `experiment.on_missing_asof_snapshot` is `disable_stats`.
4. Keep UI positioning consistent:
   - CLI is primary, `experiment-ui` is convenience wrapper, `web-ui` is prototype.

## 4) Minimal operational playbook

1. Sync topology:
```bash
python main.py sync-topology --config <sync_topology_experiment.yaml> --out outputs/sync_topology.json
```

2. Sync stats:
```bash
python main.py sync-stats --config <sync_stats_experiment.yaml> --out outputs/sync_stats.json
python main.py sync-stats --config <sync_stats_experiment.yaml> --as-of <ISO_TS> --out outputs/sync_stats_asof.json
```

3. Train/eval:
```bash
python main.py --config <train_or_eval_experiment.yaml>
```

4. Regression gates:
```bash
.\.venv\Scripts\python.exe tools\architecture_guard.py
.\.venv\Scripts\python.exe -m pytest -m mvp1_regression -v
.\.venv\Scripts\python.exe -m pytest tests/ -v
```

## 5) Historical Context Files

Current first-read path is defined in `AGENTS.MD`.

Historical context, only if explicitly needed:

1. `docs/worklogs/Finish_MVP2_5_Plan.MD`
2. `docs/worklogs/MVP2_5_Universal_Stats_Contract_Phase1_Report.MD`
3. `docs/worklogs/MVP2_5_Universal_Stats_Contract_Phase2_Quality_Gate_Report.MD`
4. `docs/worklogs/MVP2_5_Config_Driven_Stats_Matrix_Report.MD`
5. `docs/worklogs/MVP2_5_Stage4_2_XES_Stats_Fix_Report.MD`
6. `docs/worklogs/MVP2_5_Stage4_2_GNN_Stats_Forward_Logging_Report.MD`
