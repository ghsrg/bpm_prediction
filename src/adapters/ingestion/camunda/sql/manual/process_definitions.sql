/* ============================================================================
Manual export for Stage 3.2 BPMN structure ingestion.

Output target:
  process_definitions.csv

After exporting CSV:
1) Export/prepare BPMN XML files manually as:
     data/camunda_exports/bpmn/bpmn_xml/<proc_def_id>.bpmn
2) Keep CSV in:
     data/camunda_exports/bpmn/process_definitions.csv
=========================================================================== */

WITH TARGET_PROC_KEYS AS (
    SELECT PROC_DEF_KEY
    FROM (VALUES
        ('B2BContracts_ApproveProject'),
        ('BP_MediumRiskCheck')
    ) AS V(PROC_DEF_KEY)
)
SELECT
    PD.ID_ AS proc_def_id,
    PD.KEY_ AS proc_def_key,
    CAST(PD.VERSION_ AS VARCHAR(32)) AS version,
    PD.VERSION_TAG_ AS version_tag,
    PD.DEPLOYMENT_ID_ AS deployment_id,
    PD.RESOURCE_NAME_ AS resource_name,
    CONCAT('bpmn_xml/', PD.ID_, '.bpmn') AS bpmn_path
FROM bpms_camunda_mssql_tst.dbo.ACT_RE_PROCDEF PD
WHERE PD.KEY_ IN (SELECT PROC_DEF_KEY FROM TARGET_PROC_KEYS)
ORDER BY PD.KEY_, PD.VERSION_;
