/*
Manual export for: multi_instance_variables.*
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
    FROM ACT_RE_PROCDEF PD
    INNER JOIN TARGET_PROC_KEYS T ON T.proc_key = PD.KEY_
)
SELECT
    PD.proc_def_key AS process_name,
    CONCAT('v', CAST(PD.proc_def_version_num AS VARCHAR(16))) AS version_key,
    V.PROC_INST_ID_ AS case_id,
    V.PROC_DEF_ID_ AS proc_def_id,
    PD.proc_def_key AS proc_def_key,
    CONCAT('v', CAST(PD.proc_def_version_num AS VARCHAR(16))) AS proc_def_version,
    V.EXECUTION_ID_ AS execution_id,
    V.NAME_ AS var_name,
    V.LONG_ AS long_value,
    V.DOUBLE_ AS double_value,
    V.TEXT_ AS text_value,
    V.TEXT2_ AS text2_value,
    V.REMOVAL_TIME_ AS removal_time_
FROM ACT_HI_VARINST V
INNER JOIN TARGET_PROCDEF PD ON PD.proc_def_id = V.PROC_DEF_ID_
WHERE V.NAME_ IN ('loopCounter', 'nrOfInstances', 'nrOfActiveInstances', 'nrOfCompletedInstances')
ORDER BY V.PROC_INST_ID_, V.EXECUTION_ID_, V.NAME_;
