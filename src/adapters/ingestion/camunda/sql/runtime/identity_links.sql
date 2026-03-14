SELECT
    T.PROC_INST_ID_ AS case_id,
    T.PROC_DEF_ID_ AS proc_def_id,
    PD.KEY_ AS proc_def_key,
    CONCAT('v', CAST(PD.VERSION_ AS VARCHAR(16))) AS proc_def_version,
    L.TASK_ID_ AS task_id,
    L.USER_ID_ AS candidate_user_id,
    L.GROUP_ID_ AS candidate_group_id,
    L.TYPE_ AS link_type
FROM ACT_HI_IDENTITYLINK L
LEFT JOIN ACT_HI_TASKINST T ON T.ID_ = L.TASK_ID_
LEFT JOIN ACT_RE_PROCDEF PD ON PD.ID_ = T.PROC_DEF_ID_;
