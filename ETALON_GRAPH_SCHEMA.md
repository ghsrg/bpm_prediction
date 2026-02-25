# ETALON_GRAPH_SCHEMA

## 1. Тип графа
- `networkx.DiGraph`.
- Граф будується **на рівні process instance** (`root_proc_id`) для кожного документа (`doc_id`), не глобально.
- Структура зберігання: `{doc_id: {root_proc_id: DiGraph}}`.

## 2. Вузли
Вузли беруться з BPMN XML (`task`, `userTask`, `scriptTask`, `serviceTask`, `startEvent`, `endEvent`, `intermediateThrowEvent`, `boundaryEvent`, `exclusiveGateway`, `parallelGateway`, `subProcess`, `callActivity`, `intermediateCatchEvent`, `manualTask`, `businessRuleTask`, `receiveTask`, `sendTask`, `complexGateway`, `eventBasedGateway`).

### 2.1 Базові node attributes
- `type` (`str`)
- `name` (`str`)
- `attachedToRef` (`str`, тільки для `boundaryEvent`)
- `calledElement` (`str`, тільки для `callActivity`)
- `active_executions` (`int`, default `0`)
- `PROC_KEY_` (`str`) — process key
- `PROC_VER_` (`str|int`) — process version κ

### 2.2 Runtime enrichment з Camunda/BPMS
- `DURATION_`, `START_TIME_`, `END_TIME_`, `SEQUENCE_COUNTER_`, `TASK_ID_`
- `task_status`, `taskaction_code`, `user_compl_login`, `user_compl_position`, `first_view`, `overdue_work`, `duration_work`

Повторні user-task виконання (`SEQUENCE_COUNTER_` групування) створюють дублікати вузлів з id-шаблоном: `"{node_id}_{task_id}"`.

## 3. Ребра
### 3.1 Фактичні типи ребер
У etalon немає єдиного enum типів ребер. Реально зустрічаються:
1. BPMN `sequenceFlow` ребра з атрибутами `id`, `name`.
2. Boundary-link ребра (`attributes.type = 'boundaryLink'` на етапі парсингу).
3. Додатково при вставці boundary-елементів вручну додається ребро з `type='attached'`.
4. При розгортанні `callActivity` додаються службові ребра stitching (pred -> start_node, end_node -> succ) без явного edge type.

### 3.2 Edge attributes після enrichment
Для існуючих ребер додаються:
- `DURATION_E` (із source node `DURATION_`)
- `taskaction_code_E` (із source node `taskaction_code`)
- `duration_work_E` (із target node `duration_work`)

## 4. Prefix construction (process prediction)
Префікс не зберігається як окремий object у графі. Він формується під час `prepare_data`:
- беруться вузли з `SEQUENCE_COUNTER_ > 0`;
- для кожної позиції `i` формується `current_nodes = executed[:i]`;
- next event = вузол з мінімальним `SEQUENCE_COUNTER_`, що більший за максимум у `current_nodes`.

У tensor-представленні префікс кодується через `active_mask` (останній канал у `x`) + `timestamps` (для active вузлів час, інакше sentinel `1.1`).

## 5. Вплив κ (process version)
- `proc_def_version` мерджиться на рівні process instances.
- Кожному node встановлюється `PROC_VER_`.
- Далі `PROC_VER_` може потрапити у фічі тільки якщо входить у selected attrs конкретної моделі (у багатьох `*_pr` не входить напряму, але залишається в graph object).
