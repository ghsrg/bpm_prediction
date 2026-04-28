# MVP2.5 Commands Runbook

Operational command reference for MVP2.5 Stage 4.2.

---

## Metadata

- `status`: active
- `audience`: human-and-agent
- `source_of_truth`: true
- `language_policy`: keys and section headers in English, human descriptions in Ukrainian
- `last_updated`: 2026-04-27
- `primary_interface`: CLI

---

## Environment

Use the project virtual environment:

```powershell
.\.venv\Scripts\python.exe <command>
```

**Description (ukr):**

У Codex або PowerShell не треба покладатися на `py -3` чи plain `python`.
Канонічний запуск — через `.venv`.

---

## Validation

```powershell
.\.venv\Scripts\python.exe tools\architecture_guard.py
.\.venv\Scripts\python.exe -m pytest -m mvp1_regression -v
.\.venv\Scripts\python.exe -m pytest tests/ -v
```

---

## Basic Train And Eval

### train

```powershell
.\.venv\Scripts\python.exe main.py --config configs/experiments/02_train_bpi2012.yaml
```

### eval_drift

```powershell
.\.venv\Scripts\python.exe main.py --config configs/experiments/01_eval_drift_bpi2012.yaml
```

### generic_train_eval

```powershell
.\.venv\Scripts\python.exe main.py --config <train_or_eval_experiment.yaml>
```

---

## Topology Preparation

### ingest_topology_single_dataset

```powershell
.\.venv\Scripts\python.exe main.py ingest-topology --config configs/experiments/02_train_bpi2012.yaml --out outputs/bpi2012_ingest_summary.json
```

### ingest_topology_cli_keys

```text
--config PATH            YAML config path
--split train|full       optional ingestion split override
--out PATH               summary JSON output path
```

### sync_topology_bulk

```powershell
.\.venv\Scripts\python.exe main.py sync-topology --config <sync_topology_experiment.yaml> --out outputs/sync_topology.json
```

### sync_topology_config_matrix

```text
configs/experiments/mvp2_5_stage4_2_sync_camunda_files_file.yaml
configs/experiments/mvp2_5_stage4_2_sync_camunda_files_neo4j.yaml
configs/experiments/mvp2_5_stage4_2_sync_camunda_sql_file.yaml
configs/experiments/mvp2_5_stage4_2_sync_camunda_sql_neo4j.yaml
configs/experiments/mvp2_5_stage4_2_sync_xes_dir_file.yaml
configs/experiments/mvp2_5_stage4_2_sync_xes_dir_neo4j.yaml
```

---

## Stats Snapshots

### sync_stats_latest_or_auto_asof

```powershell
.\.venv\Scripts\python.exe main.py sync-stats --config <sync_stats_experiment.yaml> --out outputs/sync_stats.json
```

### sync_stats_explicit_asof

```powershell
.\.venv\Scripts\python.exe main.py sync-stats --config <sync_stats_experiment.yaml> --as-of 2024-01-01T00:00:00Z --out outputs/sync_stats_asof.json
```

### sync_stats_xes_lite

```powershell
.\.venv\Scripts\python.exe main.py sync-stats --config configs/experiments/mvp2_5_stage3_4_sync_stats_xes.yaml --out outputs/sync_stats_xes.json
.\.venv\Scripts\python.exe main.py sync-stats --config configs/experiments/mvp2_5_stage3_4_sync_stats_xes.yaml --as-of 2024-01-01T00:00:00Z --out outputs/sync_stats_xes_asof.json
```

### sync_stats_backfill

```powershell
.\.venv\Scripts\python.exe main.py sync-stats-backfill --config <sync_stats_experiment.yaml> --step weekly --out-dir outputs/sync_stats_backfill
.\.venv\Scripts\python.exe main.py sync-stats-backfill --config <sync_stats_experiment.yaml> --step monthly --from 2024-01-01T00:00:00Z --to 2024-12-31T23:59:59Z
```

`backfill_summary.json` includes:

- `runs`: per-as-of run status and summary file path
- `aggregate.runs`: total/ok/failed/planned run counts
- `aggregate.versions`: processed/skipped/usable/not-usable version counts
- `aggregate.quality.reasons`: producer quality gate reason counts
- `aggregate.alignment`: alignment ok/failed counts and minimum observed ratios
- `aggregate.skips.reasons`: skipped snapshot reason counts
- `aggregate.by_process_version`: compact per process-version rollup

### sync_stats_cli_keys

```text
--config PATH            experiment config (supports mapping.adapter=camunda|xes)
--out PATH               summary JSON output path
--as-of ISO_TS           optional historical cutoff for strict_asof policy
```

### sync_stats_backfill_cli_keys

```text
--config PATH            experiment config (camunda or xes adapter)
--out-dir PATH           directory for per-run sync summaries
--summary-out PATH       optional path for aggregated backfill summary JSON
--step VALUE             daily | weekly | monthly
--step-days N            custom step in days (overrides --step)
--from ISO_TS            optional lower bound override
--to ISO_TS              optional upper bound override
--dry-run                print planned points without executing sync-stats
```

---

## Research-Safe Workflow

```mermaid
flowchart LR
    R1[sync-topology] --> R2[sync-stats-backfill timeline]
    R2 --> R3[train/eval with strict_asof]
    R3 --> R4[report with snapshot metadata]
```

Recommended research config:

```yaml
experiment:
  stats_time_policy: strict_asof
  on_missing_asof_snapshot: raise

sync_stats:
  alignment_gate:
    enabled: true
    profile: research_strict
    min_event_match_ratio: 0.9
    min_unique_activity_coverage: 0.9
    min_node_coverage: 0.8
    on_fail: raise
```

**Description (ukr):**

Для temporal/drift досліджень треба мати timeline snapshots. Fallback режими
дозволені для exploratory запусків, але фінальні висновки мають використовувати
`strict_asof` і fail-fast policy.

---

## Visualization

### visualize_topology_repository

```powershell
.\.venv\Scripts\python.exe main.py visualize-topology --config <experiment.yaml> --version <version_key> --out outputs/topology.png
```

### visualize_topology_from_raw

```powershell
.\.venv\Scripts\python.exe main.py visualize-topology --config <experiment.yaml> --from-raw --version <version_key> --out outputs/topology_raw.png
.\.venv\Scripts\python.exe main.py visualize-topology --data "../Data/Business Process Drift/logs/cb/cb2.5k.xes" --from-raw --version <version_key> --out outputs/topology_raw_xes.png
```

### visualize_topology_cli_keys

```text
--config PATH            experiment config path
--data PATH              direct XES file path; supported only with --from-raw
--from-raw               build topology directly from raw traces
--version VALUE          process version key to render
--out PATH               optional output image path
--min-freq N             minimum DFG edge frequency
--renderer VALUE         graphviz | pm4py
--label-mode VALUE       id | name | id+name | id+name+type
--typed-colors           enable type colors
--no-typed-colors        disable type colors
```

### visualize_graph_instance

```powershell
.\.venv\Scripts\python.exe main.py visualize-graph --config <experiment.yaml> --case-id <PROC_INST_ID> --out outputs/ig_case.png
```

### visualize_graph_pick_case

```powershell
.\.venv\Scripts\python.exe main.py visualize-graph --config configs/experiments/mvp2_5_stage3_1_baseline_files.yaml --pick with-call-activity --index 0 --out outputs/ig_call_case.png
```

### visualize_graph_cli_keys

```text
--config PATH            experiment config path
--data PATH              direct XES file path
--case-id ID             exact process instance id
--pick VALUE             latest | random | longest | shortest | with-call-activity
--index N                index in ranked candidate list
--seed N                 seed for --pick random
--list-cases             print ranked candidates
--top N                  number of candidates to print
--mode VALUE             activity-centric | execution-centric
--max-nodes N            maximum events/nodes to render
--hide-loop-back         hide loop_back edges from rendered graph
--out PATH               optional output image path
--title TEXT             optional plot title
```

---

## Dataset And Simulation Tools

### mxml_to_xes

```powershell
.\.venv\Scripts\python.exe tools\mxml2xes_convertor.py --input "../Data/Business Process Drift/logs/cb/cb2.5k.mxml" --output "../Data/Business Process Drift/logs/cb/cb2.5k.xes"
```

### add_version_to_xes

```powershell
.\.venv\Scripts\python.exe main.py add-version2xes --config configs/tools/add_version2xes_re2.5.yaml
.\.venv\Scripts\python.exe main.py add-version2xes --config configs/tools/add_version2xes_re2.5.yaml --out outputs/add_version2xes_summary.json
```

### simulate_versioned_log

```powershell
.\.venv\Scripts\python.exe main.py simulate-versioned-log --config configs/tools/simulate_versioned_log_demo.yaml
.\.venv\Scripts\python.exe main.py simulate-versioned-log --config configs/tools/simulate_versioned_log_demo.yaml --out outputs/simulate_versioned_log_summary.json
```

### simulate_versioned_log_cli_keys

```text
--config PATH            simulator YAML config path
--out PATH               optional run summary path
--seed VALUE             optional random seed override
--xes-out PATH           optional generated XES path override
--summary-out PATH       optional simulator summary path override
--data-config-out PATH   optional generated data config path override
```

---

## Cache Maintenance

### cache_clean

```powershell
.\.venv\Scripts\python.exe main.py cache-clean --cache-dir .cache/graph_datasets
```

### cache_clean_dry_run

```powershell
.\.venv\Scripts\python.exe main.py cache-clean --cache-dir .cache/graph_datasets --dry-run --older-than-days 7 --keep-last 5
```

### cache_clean_size_limit

```powershell
.\.venv\Scripts\python.exe main.py cache-clean --cache-dir .cache/graph_datasets --dry-run --max-size-gb 8 --keep-last 5
.\.venv\Scripts\python.exe main.py cache-clean --cache-dir .cache/graph_datasets --max-size-gb 8 --keep-last 5
```

---

## UI Commands

### web_ui

```powershell
.\.venv\Scripts\python.exe main.py web-ui
.\.venv\Scripts\python.exe main.py web-ui --config configs/experiments/mvp2_5_stage4_2_eopkg_files_stat.yaml
```

### experiment_ui

```powershell
.\.venv\Scripts\python.exe tools\experiment_ui.py --config configs/experiments/mvp2_5_stage4_2_eopkg_files_stat.yaml
.\.venv\Scripts\python.exe main.py experiment-ui --config configs/experiments/mvp2_5_stage4_2_eopkg_files_stat.yaml
```

Operational stance:

1. CLI is primary.
2. `experiment-ui` is a convenience wrapper.
3. `web-ui` is a prototype.

---

## Key Config Attributes

### experiment

- `mode`: `train | eval_drift | eval_cross_dataset`
- `split_strategy`: `temporal | none`
- `train_ratio`, `fraction`, `split_ratio`
- `graph_dataset_cache_policy`: `off | read | write | full`
- `graph_dataset_cache_dir`
- `graph_dataset_disk_spill_enabled`: enable sharded disk spill during graph build
- `graph_dataset_shard_size`: target graphs per shard; runtime minimum is 128
- `max_ram_gb`: soft RSS limit for spill flushes; `0` disables RAM guard
- `stats_time_policy`: `latest | strict_asof`
- `on_missing_asof_snapshot`: `disable_stats | use_base | raise`

**Description (ukr):**

`graph_dataset_disk_spill_enabled=true` потребує cache write mode. Якщо policy
не дозволяє write, runtime примусово вмикає write для поточного запуску.

### mapping.knowledge_graph

- `backend`: `in_memory | file | neo4j`
- `strict_load`
- backend-specific storage/connection settings

### mapping.graph_feature_mapping

- `enabled`
- `node_numeric`
- `edge_weight`
- `encoding`: `identity | log1p | z-score`
- `stats_quality_gate`

### sync_stats.quality_gate

- `enabled`
- `zero_dominant_threshold`
- `min_non_zero_ratio_overall`
- `min_history_coverage_percent`
- `on_fail`: `write_with_flag | skip_snapshot`

### sync_stats.alignment_gate

- `enabled`
- `profile`: `legacy_exact | safe_normalized | research_strict`
- `candidate_node_fields`
- `ignore_structural_only_nodes`
- `strip_classifier_suffix`
- `normalize_case`
- `collapse_separators`
- `fail_on_ambiguity`
- `min_event_match_ratio`
- `min_unique_activity_coverage`
- `min_node_coverage`
- `on_fail`: `write_with_flag | skip_snapshot | raise`
