/*
Manual export for: execution_tree.*
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
),
ROOTS AS (
    SELECT
        E.PROC_INST_ID_ AS case_id,
        E.ID_ AS execution_id,
        E.PARENT_ID_ AS parent_execution_id,
        E.ACT_ID_ AS activity_def_id,
        E.IS_CONCURRENT_ AS is_concurrent,
        E.IS_SCOPE_ AS is_scope,
        E.PROC_DEF_ID_ AS proc_def_id,
        CAST(0 AS INT) AS depth
    FROM ACT_RU_EXECUTION E
    INNER JOIN TARGET_PROCDEF PD ON PD.proc_def_id = E.PROC_DEF_ID_
    WHERE E.PARENT_ID_ IS NULL
),
EXEC_TREE AS (
    SELECT * FROM ROOTS
    UNION ALL
    SELECT
        C.PROC_INST_ID_ AS case_id,
        C.ID_ AS execution_id,
        C.PARENT_ID_ AS parent_execution_id,
        C.ACT_ID_ AS activity_def_id,
        C.IS_CONCURRENT_ AS is_concurrent,
        C.IS_SCOPE_ AS is_scope,
        C.PROC_DEF_ID_ AS proc_def_id,
        P.depth + 1 AS depth
    FROM ACT_RU_EXECUTION C
    INNER JOIN EXEC_TREE P ON P.execution_id = C.PARENT_ID_
)
SELECT
    PD.proc_def_key AS process_name,
    CONCAT('v', CAST(PD.proc_def_version_num AS VARCHAR(16))) AS version_key,
    T.case_id AS case_id,
    T.proc_def_id AS proc_def_id,
    PD.proc_def_key AS proc_def_key,
    CONCAT('v', CAST(PD.proc_def_version_num AS VARCHAR(16))) AS proc_def_version,
    T.execution_id AS execution_id,
    T.parent_execution_id AS parent_execution_id,
    T.activity_def_id AS activity_def_id,
    T.is_concurrent AS is_concurrent,
    T.is_scope AS is_scope,
    T.depth AS depth
FROM EXEC_TREE T
INNER JOIN TARGET_PROCDEF PD ON PD.proc_def_id = T.proc_def_id
ORDER BY T.case_id, T.depth, T.execution_id
OPTION (MAXRECURSION 32767);
