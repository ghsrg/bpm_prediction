/* Stage 3.2 BPMN structure ingestion (MSSQL mode) */
SELECT
    PD.ID_ AS proc_def_id,
    PD.KEY_ AS proc_def_key,
    CAST(PD.VERSION_ AS VARCHAR(32)) AS version,
    PD.VERSION_TAG_ AS version_tag,
    PD.DEPLOYMENT_ID_ AS deployment_id,
    PD.RESOURCE_NAME_ AS resource_name,
    CAST(PD.SUSPENSION_STATE_ AS VARCHAR(16)) AS suspension_state
FROM bpms_camunda_mssql_tst.dbo.ACT_RE_PROCDEF PD
ORDER BY PD.KEY_, PD.VERSION_;
