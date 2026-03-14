/*
Manual export for: historic_activity_events.*
Edit TARGET_PROC_KEYS list before running.
*/

WITH TARGET_PROC_KEYS AS (
    SELECT V.proc_key
    FROM (VALUES
        ('procurement'),
        ('sales')
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
    H.PROC_INST_ID_ AS case_id,
    H.PROC_DEF_ID_ AS proc_def_id,
    PD.proc_def_key AS proc_def_key,
    CONCAT('v', CAST(PD.proc_def_version_num AS VARCHAR(16))) AS proc_def_version,
    H.ACT_ID_ AS activity_def_id,
    H.ACT_NAME_ AS activity_name,
    H.ACT_TYPE_ AS activity_type,
    H.ACT_INST_ID_ AS act_inst_id,
    H.TASK_ID_ AS task_id,
    H.EXECUTION_ID_ AS execution_id,
    RE.PARENT_ID_ AS parent_execution_id,
    H.CALL_PROC_INST_ID_ AS call_proc_inst_id,
    H.START_TIME_ AS start_time,
    H.END_TIME_ AS end_time,
    H.DURATION_ AS duration_ms,
    HT.ASSIGNEE_ AS assignee,
    IL.candidate_groups AS candidate_groups,
    H.REMOVAL_TIME_ AS removal_time_
FROM ACT_HI_ACTINST H
INNER JOIN TARGET_PROCDEF PD ON PD.proc_def_id = H.PROC_DEF_ID_
LEFT JOIN ACT_HI_TASKINST HT ON HT.ID_ = H.TASK_ID_
LEFT JOIN ACT_RU_EXECUTION RE ON RE.ID_ = H.EXECUTION_ID_
LEFT JOIN (
    SELECT
        TASK_ID_,
        STRING_AGG(GROUP_ID_, ',') AS candidate_groups
    FROM ACT_HI_IDENTITYLINK
    WHERE GROUP_ID_ IS NOT NULL
    GROUP BY TASK_ID_
) IL ON IL.TASK_ID_ = H.TASK_ID_
ORDER BY H.PROC_INST_ID_, H.START_TIME_;
