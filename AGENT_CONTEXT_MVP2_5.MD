# AGENT_CONTEXT_MVP2_5.MD

Останнє оновлення: 2026-03-21  
Статус: ACTIVE, Stage 4.2 у фазі завершення за `docs/worklogs/Finish_MVP2_5_Plan.MD`.

## 1) Поточний фактичний стан (з урахуванням робіт 2026-03-20)

Виконано:
1. Offline structure ingestion (`ingest-topology`) і bulk sync (`sync-topology`).
2. Repository-backed runtime path через `IKnowledgeGraphPort` (file/in_memory/neo4j).
3. Immutable snapshot policy (strict key, JSON payload, as-of lookup, no TTL).
4. Universal stats contract v1 у `sync-stats` (`stats_contract_version=1.0`).
5. Producer quality gate у `sync-stats` (thresholds + `write_with_flag|skip_snapshot`).
6. Consumer quality gate у `DynamicGraphBuilder` (`ignore_with_warning|allow_with_warning|raise`).
7. Config-driven stats mapping (`node_numeric/edge_weight` + `encoding`: `identity|log1p|z-score`).
8. Forward-path telemetry для snapshot metadata у train/inference/eval_drift.
9. XES stats calculation fix (version binding, no-data skip policy, auto as-of by train-cut).

## 2) Канонічні runtime правила MVP2.5

1. Ingestion/sync і train/eval фізично розділені.
2. Train/eval/infer не мають синхронно перебудовувати topology/stats із raw-джерел.
3. Для temporal-safe досліджень:
   - `experiment.stats_time_policy = strict_asof`,
   - snapshots мають бути підготовлені наперед.
4. Якщо stats snapshot неякісний або відсутній:
   - stats-гілка може бути вимкнена policy-gate без падіння всього run.

## 3) Актуальні ризики/відкриті пункти перед closure MVP2.5

1. Документація не повністю синхронна з останніми Stage4.2 змінами:
   - частина canonical docs описує це як “next step”, хоча код і звіти вже містять реалізацію.
2. `Finish_MVP2_5_Plan.MD` частково застарів у секції “Ще не закрито”.
3. Потрібен єдиний release-closure звіт з фінальним acceptance checklist і переносом залишкового техборгу в MVP3 backlog.

## 4) Мінімальний operational playbook

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
py -3 tools/architecture_guard.py
.\.venv\Scripts\python.exe -m pytest -m mvp1_regression -v
.\.venv\Scripts\python.exe -m pytest tests/ -v
```

## 5) Ключові файли контексту для наступних сесій

1. `PROJECT_CONTEXT.md`
2. `docs/worklogs/Finish_MVP2_5_Plan.MD`
3. `docs/worklogs/MVP2_5_Universal_Stats_Contract_Phase1_Report.MD`
4. `docs/worklogs/MVP2_5_Universal_Stats_Contract_Phase2_Quality_Gate_Report.MD`
5. `docs/worklogs/MVP2_5_Config_Driven_Stats_Matrix_Report.MD`
6. `docs/worklogs/MVP2_5_Stage4_2_XES_Stats_Fix_Report.MD`
7. `docs/worklogs/MVP2_5_Stage4_2_GNN_Stats_Forward_Logging_Report.MD`
