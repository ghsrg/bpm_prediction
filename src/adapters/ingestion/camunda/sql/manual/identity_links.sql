/*
Manual export for: identity_links.*
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
    T.PROC_INST_ID_ AS case_id,
    T.PROC_DEF_ID_ AS proc_def_id,
    PD.proc_def_key AS proc_def_key,
    CONCAT('v', CAST(PD.proc_def_version_num AS VARCHAR(16))) AS proc_def_version,
    L.TASK_ID_ AS task_id,
    L.USER_ID_ AS candidate_user_id,
    L.GROUP_ID_ AS candidate_group_id,
    L.TYPE_ AS link_type
FROM ACT_HI_IDENTITYLINK L
LEFT JOIN ACT_HI_TASKINST T ON T.ID_ = L.TASK_ID_
INNER JOIN TARGET_PROCDEF PD ON PD.proc_def_id = T.PROC_DEF_ID_
WHERE T.PROC_INST_ID_ IS NOT NULL
ORDER BY T.PROC_INST_ID_, L.TASK_ID_;
