SELECT TOP 1
    PD.ID_ AS proc_def_id,
    PD.KEY_ AS proc_def_key,
    CAST(PD.VERSION_ AS VARCHAR(32)) AS version,
    PD.VERSION_TAG_ AS version_tag,
    PD.DEPLOYMENT_ID_ AS deployment_id,
    PD.RESOURCE_NAME_ AS resource_name,
    BA.BYTES_ AS bpmn_xml_content
FROM bpms_camunda_mssql_tst.dbo.ACT_RE_PROCDEF PD
JOIN bpms_camunda_mssql_tst.dbo.ACT_GE_BYTEARRAY BA
    ON BA.DEPLOYMENT_ID_ = PD.DEPLOYMENT_ID_
   AND BA.NAME_ = PD.RESOURCE_NAME_
WHERE PD.ID_ = '{{PROC_DEF_ID}}';
