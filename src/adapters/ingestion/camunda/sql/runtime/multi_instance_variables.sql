SELECT
    V.PROC_INST_ID_ AS case_id,
    V.EXECUTION_ID_ AS execution_id,
    V.NAME_ AS var_name,
    V.LONG_ AS long_value,
    V.TEXT_ AS text_value
FROM ACT_HI_VARINST V
WHERE V.NAME_ IN ('loopCounter', 'nrOfInstances', 'nrOfActiveInstances', 'nrOfCompletedInstances');

