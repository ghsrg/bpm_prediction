# MVP2.5 Stage 3.3 - callActivity Analytics

Date: 2026-03-17  
Scope: How `callActivity` works end-to-end in current MVP2.5 architecture (Camunda logs -> structure -> inference -> training implications)

## 1. Що таке callActivity у Camunda (операційно)

`callActivity` у parent-процесі:
1. є окремою активністю в parent instance;
2. запускає child process instance;
3. має межу іншого процесу (інша дефініція, інша версія може бути).

Ключове: `callActivity` не є embedded subprocess. Це зовнішній виклик.

---

## 2. Що побачимо в логах Camunda

## 2.1 В `ACT_HI_ACTINST` (історія активностей)
Для parent case побачимо запис активності типу `callActivity`:
1. `ACT_TYPE_ = 'callActivity'`
2. `ACT_ID_` = BPMN activity id вузла виклику
3. `PROC_INST_ID_` = parent instance id
4. `CALL_PROC_INST_ID_` = id child process instance (якщо зв'язок резолвиться в історії)

## 2.2 В `ACT_HI_PROCINST` (історія інстансів процесу)
Для child інстансу:
1. `ID_` = child process instance id
2. `SUPER_PROCESS_INSTANCE_ID_` = parent process instance id

Це головний runtime-міст між parent і child.

## 2.3 Що візуально в логах
1. У parent IG буде вузол callActivity.
2. Child кроки не повинні “вливатися” в parent canonical IG (в межах поточного дизайну).
3. Child виконується як окремий процес зі своїм instance graph.

---

## 3. Що фіксуємо у структурі (ProcessStructureDTO)

Під час BPMN-ingest у `call_bindings` для кожного callActivity зберігаємо:
1. `called_element`
2. `binding_type` (`latest` / `version` / `versionTag` / `deployment`)
3. `version_tag`, `version_number`
4. `resolved_child_proc_def_id` (або `null`)
5. `status` (`resolved` / `unresolved`)
6. `reason` (`child_process_out_of_current_filter`, `child_not_found_in_catalog`, ...)
7. `requires_separate_inference = true`
8. `inference_trigger = "child_process_root"`
9. `child_process_key`, `child_version_tag`
10. `inference_fallback_strategy` (`skip | use_aggregated_stats | raise`)

Тобто структура вже містить інструкцію, що це межа іншого inference-контексту.

---

## 4. Як поєднати parent/child на інференсі (рекомендований runtime flow)

## 4.1 Оркестрація
Коли inference engine доходить до вузла з `requires_separate_inference=true`:
1. Фіксує checkpoint поточного parent inference state.
2. Визначає child target:
   - пріоритет 1: `resolved_child_proc_def_id` + runtime `call_proc_inst_id` (якщо є),
   - пріоритет 2: `child_process_key` + binding policy.
3. Запускає окремий inference для child процесу (новий контекст даних/структури/графа).
4. Після завершення child:
   - повертає child summary (confidence/outcome/ризики),
   - резюме передається в parent як feature/контекст,
   - parent inference продовжується з наступного вузла після callActivity.

## 4.2 Якщо child не резолвиться
Використати `inference_fallback_strategy`:
1. `skip` -> пропустити child inference, логнути warning.
2. `use_aggregated_stats` -> використати агреговані статистики (Stage 3.3 feature store).
3. `raise` -> зупинити inference із явною помилкою конфігурації/даних.

Default для production-safe поведінки: `use_aggregated_stats`.

---

## 5. Що треба логувати (щоб це було прозоро)

Мінімальний набір логів на кожному callActivity:
1. `parent_proc_def_id`, `parent_case_id`, `call_activity_id`
2. `called_element`, `binding_type`, `status`, `reason`
3. `resolved_child_proc_def_id`, `call_proc_inst_id`
4. `fallback_strategy`
5. `child_inference_started` / `child_inference_finished`
6. `child_inference_mode` (`resolved` / `fallback`)
7. `parent_inference_resumed`

Це дає трасованість для дисертації та дебагу orchestration.

---

## 6. Чи треба щось додаткове для навчання

Коротко: так, але поетапно.

## 6.1 Що вже достатньо зараз
1. Для поточного MVP2.5 навчання можна лишати callActivity як supernode.
2. Модель тренується на parent графі без “онлайн стрибка” в child.
3. Це стабільно і не ламає baseline.

## 6.2 Що потрібно додати для повної chained-логіки
1. Dataset builder для paired samples:
   - parent sample + optional child summary features.
2. Inference orchestrator/state machine:
   - stack або recursion guard (щоб уникати циклічних викликів).
3. Timeout/retry policy для child inference.
4. Cache для child structure/runtime fetch.
5. Метрики якості саме для callActivity-шагів:
   - resolved_call_rate,
   - child_fallback_rate,
   - chained_inference_latency.

---

## 7. Практичний висновок для поточної фази

1. У Camunda логах ми надійно бачимо call boundary через `CALL_PROC_INST_ID_` + `SUPER_PROCESS_INSTANCE_ID_`.
2. У структурі ми вже маємо повний контракт для callActivity (`call_bindings` + fallback strategy).
3. Для продовження inference після child потрібен окремий orchestration layer (це наступний інкремент, не блокер для поточних експериментів).
4. Для навчання зараз достатньо supernode-підходу; chained training можна додати окремим етапом без зламу контракту.
