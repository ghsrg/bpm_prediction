/*
Manual export for: historic_tasks.*
Edit TARGET_PROC_KEYS list before running.
*/

WITH TARGET_PROC_KEYS AS (
    SELECT V.proc_key
    FROM (VALUES
        ('B2BContracts_ApproveProject'),
        ('BP_MediumRiskCheck')
    ) AS V(proc_key)
),
TARGET_PROCDEF AS (
    SELECT
        PD.ID_ AS proc_def_id,
        PD.KEY_ AS proc_def_key,
        PD.VERSION_ AS proc_def_version_num
    FROM bpms_camunda_mssql_tst.dbo.ACT_RE_PROCDEF PD
    INNER JOIN TARGET_PROC_KEYS T ON T.proc_key = PD.KEY_
)
SELECT
    PD.proc_def_key AS process_name,
    CONCAT('v', CAST(PD.proc_def_version_num AS VARCHAR(16))) AS version_key,
    T.PROC_INST_ID_ AS case_id,
    T.PROC_DEF_ID_ AS proc_def_id,
    PD.proc_def_key AS proc_def_key,
    CONCAT('v', CAST(PD.proc_def_version_num AS VARCHAR(16))) AS proc_def_version,
    T.ID_ AS task_id,
    T.TASK_DEF_KEY_ AS activity_def_id,
    T.ASSIGNEE_ AS assignee,
    T.OWNER_ AS owner_id,
    T.START_TIME_ AS start_time,
    T.END_TIME_ AS end_time,
    T.DURATION_ AS duration_ms,
    T.REMOVAL_TIME_ AS removal_time_
FROM bpms_camunda_mssql_tst.dbo.ACT_HI_TASKINST T
INNER JOIN TARGET_PROCDEF PD ON PD.proc_def_id = T.PROC_DEF_ID_
WHERE (T.REMOVAL_TIME_ IS NULL OR T.REMOVAL_TIME_ > SYSUTCDATETIME())
ORDER BY T.PROC_INST_ID_, T.START_TIME_;

