# ETALON_EVENT_SCHEMA

## 1. Джерела сирих подій у etalon
У `etalon` немає окремого класу `EventRecord`; фактичні події формуються з таблиць/паркетів.

### 1.1 Camunda event-level records (`act_inst`)
Фактичний набір полів завантажується вибіркою:
- `ACT_ID_` (`str`) — BPMN activity id.
- `ACT_NAME_` (`str`) — activity name.
- `ACT_TYPE_` (`str`) — activity type.
- `SEQUENCE_COUNTER_` (`int`/`float` у pandas, семантично integer counter).
- `DURATION_` (`float`/`int`, тривалість виконання).
- `ROOT_PROC_INST_ID_` (`str`) — root process instance.
- `PROC_INST_ID_` (`str`) — process instance.
- `TASK_ID_` (`str | NaN`) — task reference.
- `START_TIME_` (`datetime` у parquet/`pd.Timestamp`, далі конвертується у числовий/строковий формат).
- `END_TIME_` (`datetime` у parquet/`pd.Timestamp`, далі конвертується у числовий/строковий формат).

### 1.2 Camunda task-level records (`act_hi_taskinst`)
- `ID_` (`str`)
- `TASK_DEF_KEY_` (`str`)

### 1.3 BPMS task-level records (`bpm_tasks`)
Використовуються поля:
- `externalid` (`str`)
- `task_status` (`str`)
- `taskaction_code` (`str`/код)
- `user_compl_login` (`str`)
- `user_compl_position` (`str`)
- `first_view_compluser_date` (`datetime`)
- `overduet_work` (`float`/`int`)
- `durationt_work` (`float`/`int`)

### 1.4 BPMS document metadata (`bpm_doc_purch`)
У граф-регістрі `doc_info` зберігаються поля документа (частина datetime одразу в ISO-рядки):
- Ідентифікатори/категорії: `doc_id`, `doc_subject`, `docstate_code`, `KindPurchase`, `TypePurchase`, `ClassSSD`, `FlowType`, `CategoryL1`, `CategoryL2`, `CategoryL3`, `Company_SO` (`str`).
- Дати: `ExpectedDate`, `DateKTC`, `DateInWorkKaM`, `DateApprovalFD`, `DateApprovalStartProcurement`, `DateAppFunAss`, `DateAppCommAss`, `DateApprovalProcurementResults`, `DateAppProcCom`, `DateAppContract`, `DateSentSO`, `doc_createdate` (`datetime -> ISO str`).
- Фінансові: `PurchasingBudget`, `InitialPrice`, `FinalPrice` (`str`, пізніше парсяться в float з fallback 0.0).
- Користувачі: `responsible_user_login`, `CAM_user_login`, `CEO2_user_login`, `BudgetAnalyst_user_login`, `ContractManager_user_login`, `ManagerFunction_user_login` (`str`).

## 2. Як оброблявся process_version (κ)
Явного поля `process_version` немає. Його роль виконує `proc_def_version` із `bpm_proc_def`, яке:
1. при merge додається у grouped process instances;
2. переноситься у кожен вузол графа як `PROC_VER_`.

Тобто κ у etalon — це node-level атрибут `PROC_VER_`, а не stage-level metadata.

## 3. Неявні інваріанти для сирих записів
- Для побудови графа обов'язково потрібні: `PROC_DEF_ID_`, `ID_`, `ROOT_PROC_INST_ID_`, `bpmn_model`.
- Для event enrichment очікується відповідність `ACT_ID_` (Camunda action) до BPMN node id.
- Якщо `TASK_ID_` порожній, вузол трактують як технічний (не user-task).
- Для replay-порядку очікується валідний `SEQUENCE_COUNTER_ > 0` у виконаних вузлах.
- Для document features значення, які не конвертуються у float, примусово стають `0.0`.
