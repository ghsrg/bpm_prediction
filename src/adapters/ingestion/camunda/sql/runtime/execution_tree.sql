SELECT
    E.PROC_INST_ID_ AS case_id,
    E.ID_ AS execution_id,
    E.PARENT_ID_ AS parent_execution_id,
    E.ACT_ID_ AS activity_def_id,
    E.IS_CONCURRENT_ AS is_concurrent,
    E.IS_SCOPE_ AS is_scope,
    E.IS_EVENT_SCOPE_ AS is_event_scope,
    E.REV_ AS rev,
    0 AS depth
FROM bpms_camunda_mssql_tst.dbo.ACT_RU_EXECUTION E;

