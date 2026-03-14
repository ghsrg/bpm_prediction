SELECT
    PD.KEY_ AS process_name,
    CONCAT('v', CAST(PD.VERSION_ AS VARCHAR(16))) AS version_key,
    HP.ID_ AS case_id,
    HP.PROC_DEF_ID_ AS proc_def_id,
    PD.KEY_ AS proc_def_key,
    CONCAT('v', CAST(PD.VERSION_ AS VARCHAR(16))) AS proc_def_version,
    HP.SUPER_PROCESS_INSTANCE_ID_ AS super_proc_inst_id,
    HP.SUPER_PROCESS_INSTANCE_ID_ AS super_process_instance_id,
    HP.BUSINESS_KEY_ AS business_key,
    HP.START_TIME_ AS start_time,
    HP.END_TIME_ AS end_time,
    HP.REMOVAL_TIME_ AS removal_time_
FROM bpms_camunda_mssql_tst.dbo.ACT_HI_PROCINST HP
LEFT JOIN bpms_camunda_mssql_tst.dbo.ACT_RE_PROCDEF PD ON PD.ID_ = HP.PROC_DEF_ID_
WHERE (HP.REMOVAL_TIME_ IS NULL OR HP.REMOVAL_TIME_ > SYSUTCDATETIME());

