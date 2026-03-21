# MVP2.5 Context & Documentation Sync Analysis

Date: 2026-03-21  
Status: Completed  
Scope: актуалізація контексту без змін у коді.

## 1. Мета

1. Відновити цілісний контекст після переходів між чатами.
2. Перевірити відповідність документації фактичному стану Stage 4.2.
3. Підготувати актуальні `AGENT_CONTEXT_*` для MVP1/MVP2/MVP2.5.

## 2. Що переглянуто

1. Базові файли: `README.MD`, `AGENT_GUIDE.MD`, `PROJECT_CONTEXT.md`.
2. Канонічні docs по MVP2/MVP2.5:
   - `ARCHITECTURE_*`, `DATA_MODEL_*`, `DATA_FLOWS_*`, `LLD_*`, `EVF_*`.
3. Worklogs Stage4.2:
   - universal contract phase1/phase2,
   - config-driven stats matrix,
   - XES stats fix,
   - forward logging,
   - execution/experiment updates.
4. Фактичні реалізації (тільки read-only аудит):
   - `tools/sync_stats.py`,
   - `src/domain/services/dynamic_graph_builder.py`,
   - `src/domain/entities/tensor_contract.py`,
   - `src/application/use_cases/trainer.py`,
   - відповідні тести.

## 3. Підтверджений фактичний стан

1. Universal stats contract v1 реалізовано у producer path (`sync-stats`).
2. Runtime quality gate реалізовано в `DynamicGraphBuilder`.
3. Config-driven mapping + `encoding` для `struct_x` реалізовано.
4. Snapshot metadata протягується у forward/logging path.
5. XES fix (version binding, no-data skip, auto as-of by train-cut) реалізовано.

## 4. Виявлені невідповідності (документація vs актуальний стан)

## 4.1 Високий пріоритет

1. `Finish_MVP2_5_Plan.MD` містить застарілий блок “Ще не закрито”:
   - пункти про universal contract, runtime quality gate і config-driven mapping вже закриті у Stage4.2 worklogs.

## 4.2 Середній пріоритет

1. `PROJECT_CONTEXT.md` (Updated: 2026-03-19) фіксує “next step” як інтеграцію stats features, але значна частина цього вже виконана 2026-03-20.
2. `LLD_MVP2_5.MD` у секції Known Limitation згадує encoding policy як “next step”, тоді як config-driven encoding вже є в коді й worklog.
3. `DATA_MODEL_MVP2_5.MD` не відображає нові runtime-поля контракту для snapshot-forward telemetry (`stats_snapshot_version_seq`, `stats_snapshot_as_of_epoch`, `stats_allowed`).

## 4.3 Низький пріоритет

1. Частина runbook-прикладів у `README.MD` ще Stage3.4-орієнтована і не включає повну Stage4.2 stat-config матрицю.
2. Текст одного зі звітів (`MVP2_5_Stage4_2_GNN_Stats_Forward_Logging_Report.MD`) використовує старі назви metadata-полів, тоді як поточний контракт у коді вже на `*_seq`/`*_epoch` форматі.

## 5. Виконані дії в межах цієї задачі

1. Створено `AGENT_CONTEXT_MVP2.MD`.
2. Створено `AGENT_CONTEXT_MVP2_5.MD`.
3. Оновлено `AGENT_CONTEXT_MVP1.MD`.
4. Код проєкту не змінювався.

## 6. Пропозиції для стабілізації контексту між чатами

1. Ввести короткий “Session Handoff” шаблон у `docs/worklogs/` після кожної суттєвої сесії:
   - `Done`,
   - `Decisions locked`,
   - `Open questions`,
   - `Next exact command/file`.
2. Підтримувати один “живий” статус-файл для активної фази:
   - для поточного етапу це `AGENT_CONTEXT_MVP2_5.MD`.
3. Додати lightweight правило до Definition of Done для задач:
   - якщо змінено поведінку runtime/contract, оновити мінімум один canonical doc + один worklog.
4. Раз на етап (або раз на тиждень) робити doc-sync check:
   - `PROJECT_CONTEXT`,
   - active `Finish_*_Plan`,
   - active `AGENT_CONTEXT_*`.

## 7. Наступний крок (для окремої задачі)

1. Провести звірку з цілями дисертації:
   - відповідність реалізованих Stage4.2 механік дослідницьким гіпотезам,
   - виділення техборгу, який блокує валідність/відтворюваність.
2. На основі цієї звірки оновити `Finish_MVP2_5_Plan.MD` до фактичного closure-плану.
