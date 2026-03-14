SELECT
    T.PROC_INST_ID_ AS case_id,
    T.PROC_DEF_ID_ AS proc_def_id,
    T.ID_ AS task_id,
    T.TASK_DEF_KEY_ AS activity_def_id,
    T.ASSIGNEE_ AS assignee,
    T.START_TIME_ AS start_time,
    T.END_TIME_ AS end_time,
    T.DURATION_ AS duration_ms
FROM ACT_HI_TASKINST T;

