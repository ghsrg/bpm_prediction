SELECT
    H.PROC_INST_ID_ AS case_id,
    H.PROC_DEF_ID_ AS proc_def_id,
    H.ACT_ID_ AS activity_def_id,
    H.ACT_NAME_ AS activity_name,
    H.ACT_TYPE_ AS activity_type,
    H.ACT_INST_ID_ AS act_inst_id,
    H.TASK_ID_ AS task_id,
    H.CALL_PROC_INST_ID_ AS call_proc_inst_id,
    H.START_TIME_ AS start_time,
    H.END_TIME_ AS end_time,
    H.DURATION_ AS duration_ms,
    H.REMOVAL_TIME_ AS removal_time_
FROM ACT_HI_ACTINST H;

