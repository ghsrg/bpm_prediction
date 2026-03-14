SELECT
    L.PROC_INST_ID_ AS case_id,
    L.TASK_ID_ AS task_id,
    L.USER_ID_ AS candidate_user_id,
    L.GROUP_ID_ AS candidate_group_id,
    L.TYPE_ AS link_type
FROM ACT_HI_IDENTITYLINK L;

