SELECT
    PD.KEY_ AS process_name,
    CONCAT('v', CAST(PD.VERSION_ AS VARCHAR(16))) AS version_key,
    H.PROC_INST_ID_ AS case_id,
    H.PROC_DEF_ID_ AS proc_def_id,
    PD.KEY_ AS proc_def_key,
    CONCAT('v', CAST(PD.VERSION_ AS VARCHAR(16))) AS proc_def_version,
    H.ACT_ID_ AS activity_def_id,
    H.ACT_NAME_ AS activity_name,
    H.ACT_TYPE_ AS activity_type,
    H.ID_ AS act_inst_id,
    H.PARENT_ACT_INST_ID_ AS parent_act_inst_id,
    H.SEQUENCE_COUNTER_ AS sequence_counter,
    H.TASK_ID_ AS task_id,
    H.EXECUTION_ID_ AS execution_id,
    RE.PARENT_ID_ AS parent_execution_id,
    RE.IS_CONCURRENT_ AS is_concurrent,
    RE.IS_SCOPE_ AS is_scope,
    RE.IS_EVENT_SCOPE_ AS is_event_scope,
    RE.REV_ AS rev,
    H.CALL_PROC_INST_ID_ AS call_proc_inst_id,
    H.START_TIME_ AS start_time,
    H.END_TIME_ AS end_time,
    H.DURATION_ AS duration_ms,
    HT.ASSIGNEE_ AS assignee,
    H.REMOVAL_TIME_ AS removal_time_
FROM bpms_camunda_mssql_tst.dbo.ACT_HI_ACTINST H
LEFT JOIN bpms_camunda_mssql_tst.dbo.ACT_RE_PROCDEF PD ON PD.ID_ = H.PROC_DEF_ID_
LEFT JOIN bpms_camunda_mssql_tst.dbo.ACT_HI_TASKINST HT ON HT.ID_ = H.TASK_ID_
LEFT JOIN bpms_camunda_mssql_tst.dbo.ACT_RU_EXECUTION RE ON RE.ID_ = H.EXECUTION_ID_
WHERE (H.REMOVAL_TIME_ IS NULL OR H.REMOVAL_TIME_ > SYSUTCDATETIME());

