/*
Manual export for: process_instance_links.*
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
    HP.ID_ AS case_id,
    HP.PROC_DEF_ID_ AS proc_def_id,
    PD.proc_def_key AS proc_def_key,
    CONCAT('v', CAST(PD.proc_def_version_num AS VARCHAR(16))) AS proc_def_version,
    HP.SUPER_PROC_INST_ID_ AS super_proc_inst_id,
    HP.BUSINESS_KEY_ AS business_key,
    HP.START_TIME_ AS start_time,
    HP.END_TIME_ AS end_time,
    HP.REMOVAL_TIME_ AS removal_time_
FROM bpms_camunda_mssql_tst.dbo.ACT_HI_PROCINST HP
INNER JOIN TARGET_PROCDEF PD ON PD.proc_def_id = HP.PROC_DEF_ID_
WHERE (HP.REMOVAL_TIME_ IS NULL OR HP.REMOVAL_TIME_ > SYSUTCDATETIME())
ORDER BY HP.START_TIME_;

