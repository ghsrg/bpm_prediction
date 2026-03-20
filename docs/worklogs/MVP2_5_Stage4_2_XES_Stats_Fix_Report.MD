# MVP2.5 Stage 4.2 - XES Stats Fix Report

Date: 2026-03-20  
Status: Completed (calculation path fix)

## 1. Scope

Виконано виправлення розрахунку статистики для `sync-stats` (XES path) згідно узгоджених рішень:
1. `Q1`: version source `concept:version`, fallback на `dataset_name`.
2. `Q2`: no-data policy = `skip` + warning.
3. `Q6`: для non-`vN` версій застосовується exact-match.
4. `Q7`: якщо `--as-of` не задано, використовується `as_of=max(event_ts)` після train-cut (`split_strategy/train_ratio/fraction`).
5. Додано видиме логування обраного `effective_as_of`.

## 2. Main Changes

Updated:
1. `tools/sync_stats.py`
2. `tests/application/test_sync_stats_tool.py`

### 2.1 `tools/sync_stats.py`
1. Виправлено XES version binding у `_trace_to_process_events`:
   - тепер береться `trace.process_version` (а не примусовий process namespace).
2. Оновлено `process_scope_policy=up_to_target_version`:
   - non-`vN` / mixed cases = exact-match only.
3. Додано anti-leakage train-cut для stats refresh:
   - `split_strategy`, `train_ratio`, `fraction` з `experiment`.
   - для XES — на рівні traces.
   - для Camunda — на рівні cases/events.
4. Додано auto-`as_of`:
   - якщо не задано `--as-of`, `effective_as_of = max(event_ts)` по train-cut даних для кожного процесу.
5. Додано `skip` для no-data per version:
   - якщо у scope (version+process) немає подій до `as_of`, snapshot не пишеться.
6. Розширено summary/details:
   - `effective_as_of_ts`, `scope_events_version`, `scope_events_process`,
   - train-cut metadata (`split_strategy`, `train_ratio`, `fraction`).
7. `stats_time_policy` більше не хардкодиться як `strict_asof` у payload metadata.

### 2.2 `tests/application/test_sync_stats_tool.py`
Додано/оновлено кейси:
1. `test_sync_stats_xes_uses_trace_version_for_ranked_version_scope`
2. `test_sync_stats_xes_skips_snapshot_when_no_scope_events`
3. `test_sync_stats_xes_auto_asof_uses_train_cut`
4. Оновлено alias-case для узгодження з новою version policy.

## 3. Validation

Executed:
1. `.\.venv\Scripts\python.exe -m pytest tests\application\test_sync_stats_tool.py -v`
   - Result: `6 passed`
2. `.\.venv\Scripts\python.exe -m pytest -m mvp1_regression -v`
   - Result: `17 passed`

Note:
1. `py -3 tools/architecture_guard.py` завершується помилкою через pre-existing BOM у `src/adapters/ingestion/xes_adapter.py` (не пов’язано з поточним фіксом).

## 4. Outcome

1. Виправлено ключовий дефект розрахунку статистики XES для version-scoped сценаріїв (`vN`).
2. При відсутності релевантних подій snapshot більше не генерує “порожні” `0.0` payload-и.
3. Додано leakage-safe auto-`as_of` за train-cut вхідних даних.

## 5. Follow-up (next)

1. Runtime quality gate `>=95% zeros -> ignore stats tensors` (`Q10`) поки не внесено у `DynamicGraphBuilder`; це окремий наступний change-set.
