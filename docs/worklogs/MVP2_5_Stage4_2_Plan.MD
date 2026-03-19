# MVP2.5 Phase 4.2 Plan - Neo4j Repository (Detailed)

Date: 2026-03-18  
Status: Planned  
Scope: детальна реалізація `Neo4jKnowledgeGraphRepository` з урахуванням:
1. version як масив на вузлах/ребрах,
2. багатої BPMN-семантики з Stage 3.3,
3. розширення під майбутні типи вузлів (resource/policy/external).

---

## 1. Цілі Phase 4.2

1. Реалізувати `neo4j` backend для `IKnowledgeGraphPort`.
2. Зберігати не лише transition graph, а структурну BPMN-семантику:
   - тип вузла,
   - тип зв’язку (`edge_type`),
   - метадані вузла/ребра.
3. Забезпечити version-aware модель:
   - `ProcessVersion` як окремий мета-вузол,
   - `versions: [ ... ]` як масив на вузлах і ребрах.
4. Залишити шлях для еволюції в `Phase 4.3` (статистика) і далі (resources/policies/external).

---

## 2. Ключове архітектурне рішення (щоб не переробляти в Stage 5)

## 2.1 Hybrid version model (обов’язково)
Використовуємо **обидва** механізми:
1. `(:ProcessVersion {version_key})` для точного фільтру “покажи граф версії vX”.
2. `versions: [v1, v2, ...]` на `ProcessNode` і relationships для швидкої агрегації та історичності.

Чому так:
1. Лише масиву версій недостатньо для чистих version-queries і керування життєвим циклом.
2. Лише окремих вузлів версії недостатньо для швидкого порівняння cross-version еволюції елементів.

---

## 3. Цільова граф-модель Neo4j

## 3.1 Core вузли
1. `(:Process {process_name})`
2. `(:ProcessVersion {process_name, version_key, proc_def_id, proc_def_key, deployment_id, created_at})`
3. `(:ProcessNode:Task|Gateway|StartEvent|EndEvent|CallActivity|Event|Other)`

Кожен `ProcessNode` має також загальний label `:ProcessNode` для універсальних запитів.

## 3.2 Core зв’язки
1. `(:Process)-[:HAS_VERSION]->(:ProcessVersion)`
2. `(:ProcessVersion)-[:CONTAINS_NODE]->(:ProcessNode)`
3. BPMN edge relations між ProcessNode з type = `edge_type`:
   - `:SEQUENCE_FLOW`
   - `:BOUNDARY_ATTACHMENT`
   - `:CANCELLATION`
   - `:FORK`
   - `:SUBPROCESS_REWIRE`
   - інші типи з DTO (`edge_type.upper()` з normalize мапінгом)

## 3.3 Властивості вузла
1. `process_name`
2. `node_id`
3. `name`
4. `bpmn_tag`
5. `camunda_type`
6. `logical_type` (Task/Gateway/StartEvent/...)
7. `versions` (масив)
8. `activity_type`
9. `scope_level`
10. `parent_scope_id`
11. `is_event_subprocess`
12. `is_multi_instance`
13. `is_sequential_mi`
14. `loop_cardinality_expr`
15. `attached_to`
16. `extensions_json` (serialized JSON)
17. `extra_json` (serialized JSON для forward-compat)

## 3.4 Властивості ребра
1. `process_name`
2. `edge_id`
3. `edge_type`
4. `versions` (масив)
5. `is_default`
6. `condition_expr`
7. `condition_complexity`
8. `scope_level`
9. `is_interrupting`
10. `stats_json` (майбутні статистики з Phase 4.3)
11. `extra_json`

---

## 4. Mapping DTO -> Neo4j (детальний контракт)

## 4.1 Mapping вузлів
`ProcessStructureDTO.nodes[*]`:
1. `id` -> `node_id`
2. `name` -> `name`
3. `bpmn_tag` -> label mapping + `bpmn_tag` property
4. `camunda_type` -> `camunda_type`
5. `type`/`activity_type` -> `logical_type` + `activity_type`
6. `extensions` -> `extensions_json`

## 4.2 Mapping ребер
`ProcessStructureDTO.edges[*]`:
1. `source`, `target` -> start/end `ProcessNode`
2. `edge_type` -> relationship type
3. `id` -> `edge_id`
4. `is_default`, `condition_expr`, `condition_complexity`, `scope_level`, `is_interrupting` -> relation properties

## 4.3 Mapping мета-блоків
1. `call_bindings` -> `ProcessVersion.call_bindings_json`
2. `graph_topology` -> `ProcessVersion.graph_topology_json`
3. `metadata` -> `ProcessVersion.metadata_json`
4. `allowed_edges` -> опціонально `ProcessVersion.allowed_edges_json` (для parity/debug)

---

## 5. Політика версіонування елементів

## 5.1 Node key (merge policy)
Базово:
1. `MERGE (n:ProcessNode {process_name, node_id})`
2. APOC-free policy: жодних `apoc.*`. Оновлення `versions` робимо тільки чистим Cypher через `CASE WHEN ... IN ...`.

## 5.2 Edge key (merge policy)
Базово:
1. relationship key: `(process_name, source.node_id, target.node_id, edge_type, edge_id)`
2. при upsert також додаємо `version_key` у relation `versions` масив.

## 5.3 Конфлікти (node_id той самий, атрибути змінились)
Політика:
1. скалярні поля оновлюємо last-write-wins,
2. критичні конфлікти логуємо у `metadata_json.conflicts`,
3. у `Phase 4.3` можна додати snapshot/history вузли, якщо потрібно повна темпоральність.

---

## 6. Розширюваність під нові типи вузлів

## 6.1 Що закладаємо вже зараз
1. Загальний label: `:DomainNode` (або `:KGNode`) для всіх майбутніх сутностей.
2. `ProcessNode` - тільки для BPMN activity-level.
3. Майбутні вузли:
   - `:Resource`
   - `:Policy`
   - `:ExternalSystem`
   - `:OrgUnit`
4. Узагальнені зв’язки:
   - `:ASSIGNED_TO`
   - `:CONSTRAINED_BY`
   - `:ENRICHED_BY`
   - `:EXECUTED_IN_SYSTEM`

## 6.2 Важливий принцип
Не змішувати runtime-статистику Phase 4.3 у core-схему вузлів/ребер зараз.  
Для статистики використовуємо окремі `*_json` поля або окремі stats-вузли (рішення в 4.3).

---

## 7. Підфази реалізації в межах 4.2

## 4.2.A - Schema & Query Design
1. Узгодити labels і relationship type mapping.
2. Узгодити індекси/constraints.
3. Зафіксувати Cypher templates.

Deliverable:
1. `docs/neo4j_schema_stage4_2.MD` (або секція в LLD)

## 4.2.B - Repository Implementation
1. Створити `neo4j_knowledge_graph_repository.py`.
2. Реалізувати всі методи `IKnowledgeGraphPort`.
3. Додати connection resolver integration (з Phase 4.1 профілями).

Deliverable:
1. Працюючий backend `neo4j` для save/get/list/visualization.

## 4.2.C - Factory Wiring
1. Розширити `knowledge_graph_repository_factory`.
2. Додати strict validation для `mapping.knowledge_graph.neo4j.*`.

Deliverable:
1. `backend=neo4j` працює у ingest/train/eval/visualize.

## 4.2.D - Validation & Tests
1. Contract parity tests: file vs neo4j.
2. Unit tests mapping + upsert + version arrays.
3. Integration smoke на локальному docker Neo4j.

Deliverable:
1. Green test gate.

---

## 8. Індекси та constraints (мінімум)

1. `CONSTRAINT process_name_unique IF NOT EXISTS FOR (p:Process) REQUIRE p.process_name IS UNIQUE`
2. `CONSTRAINT process_version_unique IF NOT EXISTS FOR (pv:ProcessVersion) REQUIRE (pv.process_name, pv.version_key) IS UNIQUE`
3. `INDEX process_node_idx IF NOT EXISTS FOR (n:ProcessNode) ON (n.process_name, n.node_id)`

---

## 9. Сумісність з visualize_topology

Перевірка (обов’язково в 4.2.D):
1. `visualize_topology.py` має працювати через `IKnowledgeGraphPort` незалежно від backend.
2. Для `backend=neo4j` метод `get_graph_for_visualization(...)` повинен повертати такий самий `nx.DiGraph` контракт, як file backend.
3. Порівняти output на одному датасеті (кількість вузлів/ребер після `min_edge_frequency`).

---

## 10. Питання, які фіксуємо як “відкладені до 4.3”

1. Time/resource statistics на ребрах.
2. Статистичні параметри вузлів (robust metrics, confidence, freshness).
3. Snapshot-історія статистики по вікнах.

Для 4.2 зберігаємо `stats_json`/`extra_json` гачки, але не ускладнюємо ingestion logic.

---

## 11. Acceptance Criteria для Phase 4.2

1. `ProcessStructureDTO` повністю мапиться в Neo4j із BPMN-семантикою.
2. На вузлах і ребрах є `versions` масив.
3. Є окремі вузли `ProcessVersion` та зв’язок `HAS_VERSION`.
4. `edge_type` з DTO мапиться на relationship type (не один `FOLLOWS`).
5. `visualize_topology` працює без зміни CLI-контракту.
6. File-backend дослідницький режим залишається доступним без змін.
