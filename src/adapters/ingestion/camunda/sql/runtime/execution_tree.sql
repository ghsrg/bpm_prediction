SELECT
    E.PROC_INST_ID_ AS case_id,
    E.ID_ AS execution_id,
    E.PARENT_ID_ AS parent_execution_id,
    E.ACT_ID_ AS activity_def_id,
    E.IS_CONCURRENT_ AS is_concurrent,
    E.IS_SCOPE_ AS is_scope,
    0 AS depth
FROM ACT_RU_EXECUTION E;

