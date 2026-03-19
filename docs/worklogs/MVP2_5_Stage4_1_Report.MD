# MVP2.5 Stage 4.1 Report - Connection/Secrets Refactor

Date: 2026-03-18  
Status: Done

## 1. Що реалізовано

## 1.1 Централізований resolver підключень
Додано модуль:
1. `src/infrastructure/config/connection_resolver.py`

Функції:
1. `load_local_env_files()` - підвантаження `.env` / `.env.local` (без перезапису вже заданих env).
2. `resolve_mssql_connection_string(...)` - резолв MSSQL підключення через:
   - `connection_string_env` -> env,
   - legacy `connection_string` (backward compatibility),
   - profile з `configs/connections/mssql_camunda.yaml` + `user_env/password_env`.
3. `resolve_neo4j_connection_settings(...)` - підготовлено для Stage 4.2+ (Neo4j profile + env).

## 1.2 Інтеграція resolver у Camunda adapters
Оновлено:
1. `src/adapters/ingestion/camunda_runtime_adapter.py`
2. `src/adapters/ingestion/camunda_bpmn_adapter.py`

Тепер обидва адаптери використовують profile-based резолвинг замість жорстко заданого `connection_string`.

## 1.3 Нові connection profiles та шаблон секретів
Додано:
1. `configs/connections/mssql_camunda.yaml`
2. `configs/connections/neo4j.yaml`
3. `configs/connections/README.MD`
4. `.env.template`

## 1.4 Оновлено data-конфіг SQL режиму
Оновлено:
1. `configs/data/mvp2_5_stage3_1_camunda_sql.yaml`

Замість секрета в YAML:
1. `connections_file: "configs/connections/mssql_camunda.yaml"`
2. `connection_profile: "local"`
3. `mssql.connection_profile: "local"`

## 1.5 Git hygiene
Оновлено `.gitignore`:
1. додано `.env.local`
2. додано `configs/secrets/`

---

## 2. Як тепер використовувати (SQL mode)

## 2.1 Підготувати секрети локально
Створити `.env.local` на основі `.env.template`:
```env
BPM_MSSQL_USER=sa
BPM_MSSQL_PASSWORD=your_password

BPM_NEO4J_USER=neo4j
BPM_NEO4J_PASSWORD=your_password
```

## 2.2 Запуск Camunda SQL сценарію
Конфіг:
1. `configs/data/mvp2_5_stage3_1_camunda_sql.yaml`

Запуск (приклад):
```bash
python main.py --config configs/experiments/mvp2_5_stage3_1_baseline_sql.yaml
```

Адаптери візьмуть профіль MSSQL з `configs/connections/mssql_camunda.yaml`, а user/password з env.

---

## 3. Важливо для dual-mode досліджень

Phase 4.1 **не ламає** file-mode експерименти:
1. логи можуть читатись з файлів (`runtime_source: files`),
2. структура може лишатись у файлах (`knowledge_graph.backend: file`),
3. Neo4j не є обов’язковим для кожного експерименту.

Це зберігає сценарій “дослідницькі точкові запуски без засмічення БД”.

---

## 4. Тести

## 4.1 Нові/оновлені тести
1. `tests/infrastructure/test_connection_resolver.py`
2. `tests/adapters/test_camunda_bpmn_adapter.py`
3. `tests/adapters/test_camunda_runtime_adapter.py` (додано profile-based кейс)

## 4.2 Прогін тестів
Виконано:
1. таргетований набір:
```bash
.venv\Scripts\python.exe -m pytest tests/infrastructure/test_connection_resolver.py tests/adapters/test_camunda_runtime_adapter.py tests/adapters/test_camunda_bpmn_adapter.py tests/application/test_ingest_topology_camunda_bpmn.py -v
```
Результат: `15 passed`

2. регресійний щит:
```bash
.venv\Scripts\python.exe -m pytest -m mvp1_regression -v
```
Результат: `17 passed`

3. повний набір:
```bash
.venv\Scripts\python.exe -m pytest tests/ -v
```
Результат: `169 passed`

---

## 5. Підсумок

Phase 4.1 закрито:
1. secrets винесено в env,
2. profile-based підключення для MSSQL увімкнено,
3. підготовлено Neo4j profile resolver,
4. сумісність і regression підтверджені тестами.
