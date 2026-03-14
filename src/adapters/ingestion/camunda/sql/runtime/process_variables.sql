SELECT
    PD.KEY_ AS process_name,
    CONCAT('v', CAST(PD.VERSION_ AS VARCHAR(16))) AS version_key,
    V.PROC_INST_ID_ AS case_id,
    V.PROC_DEF_ID_ AS proc_def_id,
    PD.KEY_ AS proc_def_key,
    CONCAT('v', CAST(PD.VERSION_ AS VARCHAR(16))) AS proc_def_version,
    V.EXECUTION_ID_ AS execution_id,
    V.NAME_ AS var_name,
    V.LONG_ AS long_value,
    V.DOUBLE_ AS double_value,
    V.TEXT_ AS text_value,
    V.TEXT2_ AS text2_value,
    V.REMOVAL_TIME_ AS removal_time_
FROM ACT_HI_VARINST V
LEFT JOIN ACT_RE_PROCDEF PD ON PD.ID_ = V.PROC_DEF_ID_;
