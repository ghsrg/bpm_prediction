# MVP2.5 Stage 4 Plan - Neo4j Integration

Date: 2026-03-18  
Status: Planned  
Goal: додати повноцінний `neo4j` backend для Knowledge Graph, не ламаючи режими `file` та `in_memory`.

## 0. Передумови (вже виконано)

1. Neo4j уже розгорнутий локально у Docker.
2. Stage 3.3 контракт `ProcessStructureDTO` стабілізовано.
3. Ingestion/train/eval вже працюють через `IKnowledgeGraphPort`.

Тобто інфраструктурний bootstrap Neo4j **не є пунктом плану Stage 4**, а стартова умова.

---

## 1. Ключове архітектурне правило Stage 4

Розділяємо:
1. **джерело логів** (`logs datasource`) - звідки читаємо події/трейси;
2. **сховище структури** (`structure datasource`) - де лежить EOPKG/ProcessStructureDTO.

Це дозволяє:
1. запускати production-пайплайн з `structure_backend=neo4j`;
2. робити точкові наукові експерименти з `structure_backend=file`, не засмічуючи БД.

---

## 2. Цільова конфігураційна схема datasource

## 2.1 Logs datasource (як і зараз, але явніше)
В `data`/`mapping` описуємо:
1. `adapter: xes | camunda`
2. де лежать логи (`data.log_path`, `camunda_adapter.runtime.*`)

## 2.2 Structure datasource (новий акцент)
В `mapping.knowledge_graph`:
1. `backend: in_memory | file | neo4j`
2. `path` для file backend
3. `neo4j_profile` для neo4j backend
4. `strict_load`

Приклад ідеї:
```yaml
mapping:
  knowledge_graph:
    backend: "file"            # для research локально
    path: "data/knowledge_graph"
    strict_load: false
```

```yaml
mapping:
  knowledge_graph:
    backend: "neo4j"           # для стенду / production-like runs
    neo4j_profile: "local"
    strict_load: true
```

---

## 3. Секрети та підключення (обов’язковий refactor)

## 3.1 Що переносимо
1. Neo4j доступи.
2. Camunda MSSQL доступи (runtime + BPMN mssql mode).

## 3.2 Куди переносимо
Tracked (без секретів):
1. `configs/connections/neo4j.yaml`
2. `configs/connections/mssql_camunda.yaml`
3. `configs/connections/README.MD`

Secrets (не в git):
1. `.env.local`
2. CI/CD secret store

## 3.3 Політика
1. У YAML - тільки non-secret поля + `*_env` ключі.
2. Паролі/юзери - тільки через env.
3. Якщо env-секрет відсутній - fail-fast з явним повідомленням.

---

## 4. Розбиття Stage 4 на підфази

## Phase 4.1 - Connection/Secrets Refactor

### Зміни
1. Додати централізований resolver профілів підключень.
2. Прибрати `connection_string` із `configs/data/*`.
3. Підключити resolver у:
   - `CamundaRuntimeAdapter`
   - `CamundaBpmnAdapter`
   - майбутній `Neo4jKnowledgeGraphRepository`.

### DoD
1. Немає секретів у tracked YAML.
2. Всі DB-клієнти працюють через централізований resolver.

---

## Phase 4.2 - Neo4j Repository Implementation

### Новий компонент
`src/infrastructure/repositories/neo4j_knowledge_graph_repository.py`

### Методи (контракт порту)
1. `save_process_structure(...)`
2. `get_process_structure(...)`
3. `list_versions(...)`
4. `get_graph_for_visualization(...)`

### Мінімальна модель вузлів/зв’язків
1. `(:Process {name})`
2. `(:ProcessVersion {process_name, version, proc_def_id, ...})`
3. `(:Activity {process_name, version, node_id, ...})`
4. `(:Activity)-[:FOLLOWS {...}]->(:Activity)`

### DoD
1. DTO roundtrip (save/get) без втрати ключових полів.
2. Підтримка фільтра `min_edge_frequency` для visualize.

---

## Phase 4.3 - Factory + Pipeline Wiring

### Зміни
1. Розширити `knowledge_graph_repository_factory` backend `neo4j`.
2. `ingest-topology` зберігає структуру в обраному backend.
3. Train/Eval/Visualize читають структуру з обраного backend.
4. Додати модуль оновлення статистики
5. Це запускаєтсья окремо аналізує логи і оновлює ці ваги в графі знань
### DoD
1. Однаковий experiment config працює з `file` або `neo4j` (лише зміна backend профілю).

---

## Phase 4.4 - Research Dual-Mode (щоб не засмічувати БД)

### Зміни
1. Явно документувати 2 сценарії:
   - `logs -> structure=file` (локальні наукові досліди, EOPKG аналіз),
   - `logs -> structure=neo4j` (shared/stage сценарій).
2. Додати рекомендацію namespace/tagging для експериментів у Neo4j (опційно).
3. Додати CLI-перевірку, що у research mode за замовчуванням можна лишатися на file backend.

### DoD
1. Можна проводити експерименти з файлами незалежно від наявності Neo4j.
2. Немає примусу писати кожен експеримент у БД.

---

## Phase 4.5 - Migration, Parity, Tests

### Зміни
1. Додати parity-тести `file vs neo4j`:
   - `allowed_edges`
   - `call_bindings`
   - `list_versions`
2. Додати smoke інтеграцію з локальним Neo4j.
3. Нон-регресія:
   - `pytest -m mvp1_regression -v`
   - `pytest tests/ -v`

### DoD
1. Повний зелений тест-гейт.
2. Функціональна еквівалентність file і neo4j backend підтверджена тестами.

---

## 5. Ризики і контроль

1. Ризик: розходження контракту між file/neo4j.
   - Контроль: parity-тести + DTO contract tests.
2. Ризик: випадковий витік секретів у git.
   - Контроль: env-only policy + `.gitignore` + pre-commit check.
3. Ризик: перевантаження Neo4j на дослідних ітераціях.
   - Контроль: default research режим на `file`.

---

## 6. Порядок реалізації

1. Phase 4.1 (Secrets/Connections)
2. Phase 4.2 (Neo4j repository)
3. Phase 4.3 (Factory/Wiring)
4. Phase 4.4 (Research dual-mode)
5. Phase 4.5 (Parity + tests)

Цей порядок гарантує безпечний перехід і зберігає можливість локальних експериментів без запису в БД.
