$root = 'outputs/stage3_2_ingest_demo'
$kg = Join-Path $root 'knowledge'
New-Item -ItemType Directory -Force -Path $root | Out-Null
$xesPath = Join-Path $root 'tiny_ingest.xes'
$configPath = Join-Path $root 'tiny_ingest.yaml'
$summaryPath = Join-Path $root 'ingest_summary.json'
$xes = @"
<?xml version="1.0" encoding="UTF-8" ?>
<log xes.version="1.0" xes.features="nested-attributes" xmlns="http://www.xes-standard.org/">
  <trace>
    <string key="concept:name" value="case_1"/>
    <event>
      <string key="concept:name" value="A"/>
      <string key="org:resource" value="r1"/>
      <date key="time:timestamp" value="2026-03-10T10:00:00.000+00:00"/>
    </event>
    <event>
      <string key="concept:name" value="B"/>
      <string key="org:resource" value="r2"/>
      <date key="time:timestamp" value="2026-03-10T10:01:00.000+00:00"/>
    </event>
    <event>
      <string key="concept:name" value="C"/>
      <string key="org:resource" value="r3"/>
      <date key="time:timestamp" value="2026-03-10T10:02:00.000+00:00"/>
    </event>
  </trace>
</log>
"@
Set-Content -Path $xesPath -Value $xes -Encoding UTF8

$config = @"
data:
  dataset_name: "demo_xes"
  dataset_label: "demo_xes"
  log_path: "$($xesPath -replace '\\','/')"

mapping:
  adapter: "xes"
  knowledge_graph:
    backend: "file"
    path: "$($kg -replace '\\','/')"
    strict_load: false
    ingest_split: "train"
  features:
    - name: "concept:name"
      role: "activity"
      source: "event"
      dtype: "string"
      fill_na: "<UNK>"
      encoding: ["embedding"]
    - name: "org:resource"
      role: "resource"
      source: "event"
      dtype: "string"
      fill_na: "UNKNOWN"
      encoding: ["embedding"]
    - name: "duration"
      source: "event"
      dtype: "float"
      fill_na: 0.0
      encoding: ["z-score"]

experiment:
  mode: "train"
  fraction: 1.0
  split_strategy: "temporal"
  train_ratio: 1.0
  split_ratio: [1.0, 0.0, 0.0]

training:
  show_progress: false
  tqdm_disable: true
"@
Set-Content -Path $configPath -Value $config -Encoding UTF8

.venv\Scripts\python main.py ingest-topology --config $configPath --out $summaryPath

$artifact = Join-Path $kg 'demo_xes/demo_xes/process_structure.json'
Write-Output "ARTIFACT_PATH=$artifact"
Write-Output "ARTIFACT_EXISTS=$(Test-Path $artifact)"
Write-Output "SUMMARY_EXISTS=$(Test-Path $summaryPath)"
if (Test-Path $summaryPath) { Get-Content $summaryPath -TotalCount 40 }
