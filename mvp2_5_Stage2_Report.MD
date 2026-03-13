# MVP2.5 Stage 2 Report

Date: 2026-03-13  
Status: Completed

## 1. Scope виконання

У Stage 2 реалізовано та прогнано:
1. Hardening тести для knowledge-graph port/repository.
2. Інтеграційні тести extractor->repository->builder.
3. Тести сценаріїв для майбутніх Camunda adapters.
4. Тести сценаріїв для майбутніх Neo4j graph-query backend.
5. Повний non-regression прогін (`mvp1_regression`, model unit tests, full `tests/`).

---

## 2. Що саме додано у Stage 2

## 2.1 Нові тести (Port/Repository Contract)

Файл:
- `tests/contracts/test_knowledge_graph_port_contract.py`

Покриті сценарії:
1. Процес-namespace ізоляція (`process_name + version`).
2. Безпечна поведінка для неоднозначного version-only lookup.
3. Фільтрація графа для візуалізації за частотою ребер.
4. Перевірка, що повертається detached-copy графа (без мутації сховища).
5. Camunda/BPMN-like labels (наприклад `Gateway_XOR_1`, `Task: Approve Request`).
6. Циклічні графи (майбутній Neo4j parity case).

## 2.2 Нові тести (Extractor/Builder Integration)

Файли:
- `tests/application/test_topology_extractor_multi_process.py`
- `tests/domain/test_dynamic_graph_builder_ambiguous_version.py`

Покриті сценарії:
1. Shared repository з двома process namespaces без перезапису.
2. Builder-safe behavior при неоднозначному `version` без process scope (`allowed_target_mask=None`, без падіння).

## 2.3 Розширення repository tests

Оновлено:
- `tests/infrastructure/test_in_memory_networkx_repository.py`

Додано перевірки:
1. Помилка для невідомої пари `process/version` у `get_graph_for_visualization`.
2. Валідація `min_edge_frequency >= 0`.

---

## 3. Camunda / Neo4j орієнтовані тест-сценарії

Додані тести покривають потенційні майбутні проблеми:
1. Camunda labels та BPMN-подібні вузли не ламають DTO/graph storage.
2. Циклічні маршрути коректно повертаються для graph-query layer.
3. Відсутність leakage між процесами у shared backend.
4. Ambiguous version lookup без process scope не повертає некоректний граф.
5. Граф для візуалізації повертається як копія (безпечне використання у view/plot коді).

---

## 4. Запуски тестів і результати

## 4.1 Stage 2 targeted tests
Команда:
```bash
.\.venv\Scripts\python.exe -m pytest tests/contracts/test_knowledge_graph_port_contract.py tests/infrastructure/test_in_memory_networkx_repository.py tests/application/test_topology_extractor_multi_process.py tests/domain/test_dynamic_graph_builder_ambiguous_version.py -v
```
Результат:
- `14 passed`
- `0 failed`

## 4.2 MVP1 Regression Shield
Команда:
```bash
.\.venv\Scripts\python.exe -m pytest -m mvp1_regression -v
```
Результат:
- `16 passed`
- `0 failed`

## 4.3 Model Unit Tests (Dual-Encoder)
Команда:
```bash
.\.venv\Scripts\python.exe -m pytest tests/domain/test_eopkg_models_forward.py tests/domain/test_eopkg_gatv2_dual_encoder.py -v
```
Результат:
- `9 passed`
- `0 failed`

## 4.4 Full test suite
Команда:
```bash
.\.venv\Scripts\python.exe -m pytest tests/ -v
```
Результат:
- `91 passed`
- `0 failed`
- `1 warning` (mlflow/pkg_resources deprecation)

---

## 5. Висновок Stage 2

Stage 2 завершено успішно:
1. Port/Repository контракт став суттєво більш надійним.
2. Підготовлені тестові гарантії для майбутньої інтеграції Camunda adapters.
3. Підготовлені backend-agnostic кейси для майбутнього Neo4j adapter.
4. MVP1 та MVP2 поведінка залишилась стабільною (повний зелений прогін тестів).
