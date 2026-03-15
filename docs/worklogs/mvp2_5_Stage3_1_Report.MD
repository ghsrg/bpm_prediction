# MVP2.5 Stage 3.1 Report

Date: 2026-03-13  
Status: Implemented (Camunda Stage 3.1 baseline)

## 1. Що реалізовано

Реалізовано Stage 3.1 (Runtime InstanceGraph Ingestion) у вигляді ізольованого Camunda-модуля з підтримкою:
1. `runtime_source: files | mssql` через конфіг.
2. Cleanup-aware обробки (`REMOVAL_TIME_`, legacy fallback policy).
3. Canonical graph modes (`activity-centric`, `execution-centric`) з auto-degrade.
4. Hysteresis для fallback при low coverage.
5. High-MI guard (aggregate/sample strategy).
6. Fallback mode `fallback_structure_only` через `IKnowledgeGraphPort`.
7. Idempotent in-memory persistence (events/graph/projection/watermark).

## 2. Додані компоненти

## 2.1 Domain

1. `src/domain/entities/process_event.py`
2. `src/domain/entities/runtime_fetch_diagnostics.py`
3. `src/domain/ports/camunda_runtime_port.py`
4. `src/domain/ports/instance_graph_port.py`

## 2.2 Application

1. `src/application/services/instance_graph_assembler_service.py`
2. `src/application/ports/camunda_runtime_port.py` (alias)
3. `src/application/ports/instance_graph_port.py` (alias)

## 2.3 Infrastructure / Adapters

1. `src/adapters/ingestion/camunda_runtime_adapter.py`
2. `src/infrastructure/repositories/in_memory_instance_graph_repository.py`
3. SQL templates:
   - `src/adapters/ingestion/camunda/sql/runtime/historic_activity_events.sql`
   - `src/adapters/ingestion/camunda/sql/runtime/historic_tasks.sql`
   - `src/adapters/ingestion/camunda/sql/runtime/identity_links.sql`
   - `src/adapters/ingestion/camunda/sql/runtime/execution_tree.sql`
   - `src/adapters/ingestion/camunda/sql/runtime/multi_instance_variables.sql`

## 2.4 Tooling + Config

1. Runner:
   - `tools/stage3_1_runtime_refresh.py`
2. Configs:
   - `configs/experiments/mvp2_5_stage3_1_files.yaml`
   - `configs/experiments/mvp2_5_stage3_1_mssql.yaml`
3. Manual export folder + mocks:
   - `data/camunda_exports/README.MD`
   - `data/camunda_exports/mock_historic_activity_events.csv`
   - `data/camunda_exports/mock_historic_tasks.csv`
   - `data/camunda_exports/mock_identity_links.csv`
   - `data/camunda_exports/mock_execution_tree.csv`
   - `data/camunda_exports/mock_multi_instance_variables.csv`
4. `.gitignore` оновлено, щоб `data/camunda_exports/**` трекалась у git.

## 3. Тести Stage 3.1

## 3.1 Нові тести

1. `tests/adapters/test_camunda_runtime_adapter.py`
   - files source + diagnostics
   - missing `REMOVAL_TIME_` auto fallback
   - dedup
   - mssql mode without driver (safe empty response)

2. `tests/application/test_instance_graph_assembler_service.py`
   - fallback structure-only on empty history
   - execution-centric -> activity-centric degrade
   - hysteresis low-coverage logic
   - high-MI guard
   - null-safe identity link attach

3. `tests/infrastructure/test_in_memory_instance_graph_repository.py`
   - idempotent save + dedup
   - interval filtering
   - graph/projection/watermark persistence

## 3.2 Результати запусків тестів

1. Stage 3.1 тести:
```bash
.\.venv\Scripts\python.exe -m pytest tests/adapters/test_camunda_runtime_adapter.py tests/application/test_instance_graph_assembler_service.py tests/infrastructure/test_in_memory_instance_graph_repository.py -v
```
Result: `12 passed`

2. Regression Shield (MVP1):
```bash
.\.venv\Scripts\python.exe -m pytest -m mvp1_regression -v
```
Result: `16 passed, 87 deselected`

3. Повний тестовий набір:
```bash
.\.venv\Scripts\python.exe -m pytest tests/ -v
```
Result: `103 passed`

## 4. Як запускати Stage 3.1 (files / mssql)

## 4.1 Files mode (ручні CSV/XLSX вивантаження)

1. Покласти файли в папку:
   - `data/camunda_exports/`
2. Запуск:
```bash
.\.venv\Scripts\python.exe tools/stage3_1_runtime_refresh.py --config configs/experiments/mvp2_5_stage3_1_files.yaml --out outputs/stage3_1_refresh_result.json
```
3. Що отримаєш:
   - JSON summary у `outputs/stage3_1_refresh_result.json`
   - режим роботи (`canonical_*` або `fallback_structure_only`)
   - diagnostics (`coverage`, `warnings`, `fallback_triggered`)

## 4.2 MSSQL mode

1. Заповнити connection string у:
   - `configs/experiments/mvp2_5_stage3_1_mssql.yaml`
2. Встановити ODBC/driver та `pyodbc` у `.venv`.
3. Запуск:
```bash
.\.venv\Scripts\python.exe tools/stage3_1_runtime_refresh.py --config configs/experiments/mvp2_5_stage3_1_mssql.yaml --out outputs/stage3_1_refresh_result_mssql.json
```
4. Якщо драйвер/конект відсутній:
   - процес не падає,
   - повертається безпечний `fallback_structure_only`.

## 5. Формат файлів для ручних вигрузок

Підтримувані імена (CSV/XLSX):
1. `historic_activity_events.*`
2. `historic_tasks.*`
3. `identity_links.*`
4. `execution_tree.*`
5. `multi_instance_variables.*`

Для тестових/чернеткових файлів також підтримуються `mock_*.csv`.

## 6. Поточні обмеження (свідомо залишені для Stage 3.2/3.3)

1. Stage 3.1 не парсить BPMN XML (це Stage 3.2).
2. Stage 3.1 не робить multi-window decayed stats (це Stage 3.3).
3. Немає BPMS/External data dependency (ізоляція збережена).

## 7. Висновок

Stage 3.1 реалізовано відповідно до плану з робочим перемиканням `files|mssql`, fallback-resilience, тестовим покриттям і практичним runner для стенду з ручними вивантаженнями.
