# Analyse_after_mvp2.5.md

Оновлено: 2026-03-20  
Призначення: зібраний контекст після MVP2.5 Stage 4.2, фіксація відкритих питань перед виправленням XES→Neo4j статистики, тестовим закриттям та аудитом техборгу.

## 1. Що проаналізовано

Опрацьовані ключові джерела правди:
1. `AGENT_GUIDE.MD`
2. `PROJECT_CONTEXT.md`
3. MVP2.5 canonical docs (`ARCHITECTURE_MVP2_5`, `DATA_MODEL_MVP2_5`, `DATA_FLOWS_MVP2_5`, `LLD_MVP2_5`, `EVF_MVP2_5`)
4. `README.MD` секція `MVP2.5 Stage 3.4 Runbook (Canonical)`
5. `docs/worklogs/*Stage3_4*`, `MVP2_5_Stage4_1_Report.MD`, `MVP2_5_Stage4_2_Plan.MD`, `MVP2_5_Stage4_2_Report.MD`
6. Код `tools/sync_stats.py`, `src/adapters/ingestion/xes_adapter.py`, `src/infrastructure/repositories/neo4j_knowledge_graph_repository.py`, пов’язані тести.
7. Фактичні результати в `outputs/*.json` + перевірка `StatsSnapshot` у локальному Neo4j.

## 2. Поточний стан Stage 4.2 (факт)

1. `Neo4jKnowledgeGraphRepository` реалізовано й інтегровано.
2. Snapshot-політика Stage 3.4 діє: immutable, strict composite key, JSON payload, `as_of`, без TTL.
3. `sync-stats` працює для `camunda` і `xes` режимів.
4. Проблема відтворюється: для частини XES запусків у Neo4j нові snapshots мають майже суцільні `0.0`.

## 3. Підтверджені спостереження по дефекту `XES -> Neo4j -> 0.0`

### 3.1 По summary-артефактах
1. `outputs/sync_stat_files_neo4j.json` (camunda): coverage = `100.0`, метрики ненульові.
2. `outputs/sync_files_xes_neo4j.json` (xes): для `B2BContracts_ApproveProject v5/v6` coverage = `0.0`.

### 3.2 По фактичних `StatsSnapshot` у Neo4j
1. `B2BContracts_ApproveProject v5/v6`:
   - `k000001`, `k000002`: coverage `100.0`, ненульові метрики (`~45-50%` numeric fields non-zero).
   - `k000003`, `k000004`, `k000005`: coverage `0.0`, `0/6608` і `0/9632` non-zero (повністю нульові).
2. `BPI_Challenge_2012`:
   - є snapshot з ненульовими значеннями (`k000003`) і snapshot-и з `0.0` при coverage `0.0`.

### 3.3 Ймовірний корінь проблеми в коді (підтверджено трасуванням)
1. `tools/sync_stats.py:636-665`:
   - у XES конвертері `proc_def_version` примусово ставиться в `fallback_process` (namespace процесу), а не в реальну `trace.process_version`.
2. `tools/sync_stats.py:356-379`:
   - `version/process` фільтр очікує відповідність `target version` (наприклад, `v5`) і `event.proc_def_version`.
3. Для процесів з версіями `vN` отримуємо mismatch:
   - події мають `proc_def_version = process_namespace` (напр. `B2BContracts_ApproveProject`),
   - DTO має `version = v5/v6`,
   - результат: порожні `version_events/process_events` => всі агрегати в `0.0`.

### 3.4 Додаткові фактори, що підсилюють проблему
1. `sync-stats` за замовчуванням обробляє всі процеси з repository (`list_process_names`), якщо `process_filters` пустий:
   - для процесів без відповідного XES джерела також створюються snapshots з coverage `0.0`.
2. Потенційний semantic mismatch:
   - XES `activity_id` (часто `concept:name`) може не співпадати з BPMN `node_id` у структурі Neo4j,
   - навіть при coverage > 0 частина node-метрик може лишатися нульовою.
3. При `stats_time_policy=latest` нові zero-snapshots стають “останнім” станом і можуть перекривати корисні попередні snapshots.

## 4. Початкові питання до узгодження (історія)

`[OPEN-Q1]` Яке канонічне джерело `version_key` для XES у multi-version процесах?
1. `concept:version` з XES
2. мапінг із файла/каталогу
3. окремий config override

`[OPEN-Q2]` Чи залишаємо політику “писати snapshot навіть при coverage=0”, чи переходимо на `skip`?

`[OPEN-Q3]` Для `mapping.adapter: xes` робимо `process_filters` обов’язковим, чи лишаємо optional?

`[OPEN-Q4]` Чи ізолюємо XES-experiments в окремий namespace/storage (щоб не змішувати з camunda snapshots)?

`[OPEN-Q5]` Потрібен офіційний mapping `XES activity -> BPMN node_id` для гібридного сценарію (BPMN topology + XES stats)?

`[OPEN-Q6]` Для `process_scope_policy=up_to_target_version`: як поводитись з не-ранжованими версіями (не `vN`)?

`[OPEN-Q7]` Для XES без `--as-of`: залишаємо `as_of=now UTC` чи ставимо `as_of=max(event_ts)`?

`[OPEN-Q8]` Якщо `strict_asof` не знаходить snapshot `<= as_of`, залишаємо fallback на base DTO чи робимо fail-fast в strict режимі?

`[OPEN-Q9]` Який acceptance-критерій “статистика нормальна” для XES?
1. мінімальний non-zero ratio
2. coverage threshold
3. перевірка ключових метрик (exec_count / transition_probability)

`[OPEN-Q10]` Як поводимось із вже створеними zero-snapshots (`k000003+`)?
1. лишаємо як immutable historical факт
2. вводимо runtime-ігнор по quality flag
3. перезапускаємо в новому namespace

## 5. План робіт після узгодження (черга)

`[DONE]` Крок 1. Рішення по `Q1-Q10` зафіксовано (див. секцію 8).  
`[READY]` Крок 2. Виправити XES-branch у `sync_stats` (version binding + scope filtering + no-data policy).  
`[READY]` Крок 3. Додати тести:
1. XES + multi-version `vN` (non-zero assertion)
2. no-data process handling (skip vs warning snapshot)
3. activity mapping scenario для BPMN topology
4. regression: camunda path без змін
`[READY]` Крок 4. Провести тест-гейт:
1. таргетовані unit/integration
2. `pytest -m mvp1_regression -v`
3. `pytest tests/ -v`
`[READY]` Крок 5. Техборг-аудит (мінімум):
1. quality flags для snapshots
2. розділення production/research namespaces
3. спостережність coverage/non-zero ratio в summary

## 6. Технічний борг (зафіксовано)

`[DEBT-1]` Немає policy-driven guard для запису “порожніх” snapshots при XES no-data.  
`[DEBT-2]` Немає канонічного bridge між XES activity labels і BPMN node IDs.  
`[DEBT-3]` Недостатній тестовий контур для XES multi-version (`vN`) випадків.  
`[DEBT-4]` `latest` policy вразлива до деградації через пізні zero-snapshots без quality gate.  
`[DEBT-5]` Немає явного quality маркера snapshot-а (`coverage`, `non_zero_ratio`, `is_usable_for_training`).

## 7. Підсумок (до отримання рішень)

1. Проблема `XES -> Neo4j mostly 0.0` підтверджена фактично і локалізована на стику `version binding + scope filtering + no-data snapshot policy`.
2. Критичний блокер не в Neo4j persistence як такому, а в семантиці підготовки XES подій до агрегації.
3. Для безпечного фіксу були потрібні продуктові рішення по `Q1-Q10` (отримані в секції 8).

## 8. Рішення користувача (2026-03-20)

`[RESOLVED-Q1]` Version source для XES:
1. Пріоритет `concept:version` з XES.
2. Якщо не заповнений — брати `dataset_name` з config.

`[RESOLVED-Q2]` No-data policy:
1. Переходимо на `skip`.
2. Писати warning, що snapshot не створено через відсутність даних.

`[RESOLVED-Q3]` `process_filters`:
1. Лишається optional.
2. Для Camunda також може бути пустим (наприклад перший full прогін).

`[RESOLVED-Q4]` Namespace isolation:
1. Не ізолювати XES окремо примусово.
2. Допускається mixed-source knowledge в одному Neo4j.

`[RESOLVED-Q5]` BPMN<->XES mapping:
1. Зараз не робити обов’язковий mapping.
2. Залишити можливість додати mapping у майбутньому.

`[CLARIFIED-Q6]` `process_scope_policy=up_to_target_version`:
1. Це режим, де для `scope=process` статистика для версії `vN` рахується по подіях `<= vN`.
2. Питання закривалось для узгодження поведінки не-ранжованих версій (не `vN`) — потрібне fallback правило.

`[RESOLVED-Q7]` Default `as_of`:
1. Ставимо `as_of = max(event_ts)`.
2. `max(event_ts)` має враховувати train-cut (`train_ratio`), щоб не було leakage.
3. Додати логування обраного діапазону дат статистики.

`[CLARIFIED-Q8]` Поведінка при відсутності snapshot під час runtime:
1. Для train/infer брати лише наявну статистику.
2. Якщо snapshot відсутній — повідомлення, що статистичні тензори пусті (без падіння).

`[RESOLVED-Q9]` Acceptance:
1. Орієнтир — високий % мапінгу логів на структуру.
2. Цільовий sanity-check: high non-zero ratio (орієнтовно до 95% non-zero де очікувано).

`[RESOLVED-Q10]` Runtime quality gate:
1. Ввести runtime-ігнор статистики, якщо >=95% значень `0.0` (щоб не шуміти в GNN).

## 9. Дії з рішень

`[NEXT]` Внести зміни в `sync-stats`:
1. XES version binding за правилом `concept:version -> dataset_name`.
2. `skip` при no-data + warning.
3. Default `as_of=max(event_ts)` з train-cut.
4. Видиме логування обраних timestamp boundaries.

`[NEXT]` Внести runtime guard у builder/consumption path:
1. Якщо snapshot/статистика відсутня — працювати без stats tensors + warning.
2. Якщо zero-dominant (>=95% нулів) — ігнорувати stats tensors + warning.
