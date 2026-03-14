# MVP2.5 Stage 3.1 Report (v3)

Date: 2026-03-14
Status: Planning (single delivery)

## Decision

Так, можна робити **єдиною доробкою** (one-shot delivery), але з внутрішніми технічними хвилями і жорсткими quality gates після кожної хвилі.

Чому так:
1. Всі зміни сильно зв'язані: SQL -> DTO -> Adapter -> IG Assembler -> Tests.
2. Частковий реліз дасть тимчасово неконсистентний контракт полів.
3. Ризик знижуємо не поділом на релізи, а поділом на внутрішні кроки + повний regression після кожного кроку.

---

## Target (what we improve)

Поточний IG у Stage 3.1 відновлює лише базову послідовність по часу. Потрібно перейти до **max-link IG**:
1. Token sequence edges (execution-aware).
2. Scope/parallel edges з execution tree.
3. Multi-instance semantics (loop counters + nrOf*).
4. Call activity links parent <-> child process.
5. Розширені node/edge features для GNN.
6. Resilience при history cleanup/partial data.

---

## Scope of single delivery

### 1) SQL layer expansion (manual + runtime templates)

Оновити SQL шаблони в:
1. `src/adapters/ingestion/camunda/sql/runtime/*.sql`
2. `src/adapters/ingestion/camunda/sql/manual/*.sql`

#### 1.1 historic_activity_events.sql

Додати/гарантувати поля:
1. `H.ID_ AS act_inst_id`
2. `H.PARENT_ACT_INST_ID_ AS parent_act_inst_id`
3. `H.EXECUTION_ID_ AS execution_id`
4. `H.PROC_INST_ID_ AS case_id`
5. `H.PROC_DEF_ID_`, `PD.KEY_ AS proc_def_key`, `PD.VERSION_ AS proc_def_version_num`
6. `H.ACT_ID_ AS activity_def_id`, `H.ACT_NAME_`, `H.ACT_TYPE_`
7. `H.ACT_INST_STATE_ AS act_inst_state`
8. `H.SEQUENCE_COUNTER_ AS sequence_counter`
9. `H.TASK_ID_`
10. `H.CALL_PROC_INST_ID_`
11. `H.ROOT_PROC_INST_ID_ AS root_proc_inst_id`
12. `H.TENANT_ID_ AS tenant_id`
13. `H.START_TIME_`, `H.END_TIME_`, `H.DURATION_`
14. `H.REMOVAL_TIME_`

#### 1.2 execution_tree.sql

Додати/гарантувати поля:
1. `E.ID_`, `E.PARENT_ID_`
2. `E.PROC_INST_ID_`, `E.ROOT_PROC_INST_ID_`
3. `E.PROC_DEF_ID_`, `E.ACT_ID_`
4. `E.IS_CONCURRENT_`, `E.IS_SCOPE_`, `E.IS_EVENT_SCOPE_`
5. `E.SUPER_EXEC_`
6. `E.REV_`
7. bounded `depth` (рекурсія + depth_limit)

#### 1.3 historic_tasks.sql

Додати:
1. `TASK_DEF_KEY_`, `EXECUTION_ID_`
2. `ASSIGNEE_`, `OWNER_`, `PRIORITY_`
3. `DELETE_REASON_`, `DUE_DATE_`, `FOLLOW_UP_DATE_`
4. `REMOVAL_TIME_`

#### 1.4 identity_links.sql

Нормалізувати через `ACT_HI_TASKINST`:
1. `case_id`, `proc_def_id` через task join
2. `TASK_ID_`, `TYPE_`, `USER_ID_`, `GROUP_ID_`
3. Без залежності від неіснуючих колонок `ACT_HI_IDENTITYLINK`

#### 1.5 multi_instance_variables.sql

Покрити:
1. `loopCounter`
2. `nrOfInstances`
3. `nrOfActiveInstances`
4. `nrOfCompletedInstances`
5. `execution_id`, `proc_inst_id`, `task_id(if present)`

---

### 2) DTO contract expansion

Оновити `ProcessEventDTO`:
1. `parent_act_inst_id: Optional[str]`
2. `sequence_counter: Optional[int]`
3. `act_inst_state: Optional[str]`
4. `root_proc_inst_id: Optional[str]`
5. `tenant_id: Optional[str]`
6. `super_execution_id: Optional[str]`
7. `is_concurrent: Optional[bool]`
8. `is_scope: Optional[bool]`
9. `is_event_scope: Optional[bool]`

Backward compatibility:
1. Всі нові поля optional.
2. Стара схема CSV не ламає ingestion.

---

### 3) Adapter mapping update

Оновити `CamundaRuntimeAdapter._row_to_event(...)`:
1. мапінг нових полів з SQL/CSV.
2. `sequence_counter` / boolean flags parsing.
3. null-safe parse для mixed CSV/Excel exports.

Оновити `_event_known_fields()` відповідно до нових полів.

---

### 4) Canonical IG algorithm upgrade

Оновити `InstanceGraphAssemblerService._build_graph_payload(...)`:

#### 4.1 Edge types

Додати edge types:
1. `token_next`:
   - primary order: `(case_id, execution_id, sequence_counter)`
   - fallback order: `start_time`
2. `scope_parent_child`:
   - via `parent_act_inst_id` and/or execution tree relation
3. `parallel_sibling`:
   - sibling executions in same scope with overlapping intervals
4. `mi_iteration`:
   - links for repeated activity in MI context
5. `call_activity_link`:
   - `callActivity(act_inst)` -> child process root instance
6. Existing `sequence` залишити як compatibility alias, але новий default: `token_next`

#### 4.2 Node attributes

Розширити node payload:
1. identifiers: `act_inst_id`, `task_id`, `execution_id`, `case_id`, `proc_def_id`
2. semantic: `activity_def_id`, `activity_name`, `activity_type`, `act_inst_state`
3. temporal: `start_time`, `end_time`, `duration_ms`
4. org: `assignee`, `candidate_groups`
5. mi: `loopCounter`, `nrOfInstances`, `nrOfActiveInstances`, `nrOfCompletedInstances`

#### 4.3 Edge attributes

Додати:
1. `delta_ms`
2. `same_execution` (bool)
3. `same_scope` (bool)
4. `parallel_overlap` (bool)
5. `confidence` (simple heuristic score)

---

### 5) IG quality validation (hard checks)

Додати `validate_instance_graph(...)` у assembler layer:
1. No self-loop for `token_next`.
2. `token_next` must be acyclic within one execution chain.
3. `call_activity_link` only when `call_proc_inst_id` exists.
4. Parallel edges only when execution relation supports it.
5. MI fields consistency: `loopCounter < nrOfInstances` (when both known).
6. Diagnostics counters:
   - invalid_edges_dropped
   - token_chains_built
   - parallel_edges_built
   - call_links_built

---

### 6) Tests (mandatory)

#### 6.1 Adapter tests

1. Parse all new columns from CSV.
2. Missing optional columns does not fail.
3. Mixed naming and types (string/int/nullable) are parsed safely.

#### 6.2 Assembler tests

1. `token_next` reconstruction by `sequence_counter`.
2. Fallback to time order when `sequence_counter` absent.
3. Parallel sibling edge creation for overlapping branches.
4. MI edges and counters.
5. CallActivity parent-child link.
6. Validation drops invalid edges and reports diagnostics.

#### 6.3 Contract tests

1. DTO backward compatibility.
2. Repository roundtrip for extended graph payload.

#### 6.4 Regression tests

1. `pytest -m mvp1_regression -v`
2. `pytest tests/ -v`

---

## Delivery sequence (single release, internal waves)

1. Wave A: SQL templates + DTO + adapter mapping.
2. Wave B: assembler edge logic + validation.
3. Wave C: tests + docs + config comments update.
4. Final gate: full regression + smoke run on files mode.

No partial release between waves.

---

## Acceptance criteria

1. IG contains execution-aware links (`token_next`, `scope_parent_child`, `parallel_sibling`, `mi_iteration`, `call_activity_link`).
2. Exported/manual SQL files allow filtered extraction by process keys (`IN (...)`) without full DB scan.
3. Graph build survives partial/missing historical data.
4. All automated tests pass.
5. MVP1 regression shield stays green.

---

## Operational usage after delivery

1. Run manual SQL from `src/adapters/ingestion/camunda/sql/manual/` with your process list.
2. Export to `data/camunda_exports/` using canonical/flexible file names.
3. Run Stage 3.1 pipeline from experiment YAML with `mapping.adapter: camunda`.
4. Validate diagnostics (`history_coverage_percent`, fallback flags, edge counters).

---

## Risks and mitigations

1. Camunda schema variance across versions.
Mitigation: optional mapping + null-safe parser + compatibility aliases.

2. Execution tree explosion on large MI.
Mitigation: depth limit + max nodes per case + aggregation strategy.

3. Cleanup lag causes temporary data inconsistency.
Mitigation: lookback + hysteresis + fallback_structure_only mode.

4. Over-connecting graph with noisy edges.
Mitigation: validation guards + edge confidence + strict test fixtures.
