# MVP2.5 Stage 3.1 Report (v3)

Date: 2026-03-14
Status: Implemented and validated

## Scope delivered

Implemented Stage 3.1 extension for Camunda IG ingestion with three additions requested:

1. Explicit actor-role enrichment:
- assigned executor
- potential executors (users/groups)
- executed-by user

2. Process variables extension for future node features.

3. Mandatory `call_activity_link` persistence in canonical IG for future cascade handling.

## Code changes

### 1) DTO / Contracts

Updated `ProcessEventDTO`:
- `assigned_executor: Optional[str]`
- `executed_by: Optional[str]`
- `potential_executor_users: Optional[list[str]]`
- `potential_executor_groups: Optional[list[str]]`
- `process_variables: Optional[Dict[str, Any]]`
- `call_activity_link: Optional[Dict[str, Any]]`

Files:
- `src/domain/entities/process_event.py`
- `src/domain/ports/camunda_runtime_port.py` (new method `fetch_process_variables`)

### 2) Camunda runtime adapter

Extended `CamundaRuntimeAdapter`:
- new source table mapping: `process_variables`
- new API method: `fetch_process_variables(...)`
- richer row-to-event mapping for actor-role fields
- normalized list parsing for potential users/groups

File:
- `src/adapters/ingestion/camunda_runtime_adapter.py`

### 3) IG assembler

Extended `InstanceGraphAssemblerService.build(...)` with optional inputs:
- `task_rows`
- `process_variables_rows`

Implemented:
- actor-role enrichment from `historic_tasks` + `identity_links`
- process variable indexing (`case-level` + `execution-level`) and attachment to node payload
- mandatory `call_activity_link` object on call activity nodes
- `metadata.call_activity_links` + `metadata.num_call_activity_links`
- diagnostics counters (`task_rows`, `process_variables_rows`)

File:
- `src/application/services/instance_graph_assembler_service.py`

### 4) Stage 3.1 runner

Updated runtime refresh tool to fetch and pass all enrichments:
- `historic_tasks`
- `multi_instance_variables`
- `process_variables`
- `identity_links`

File:
- `tools/stage3_1_runtime_refresh.py`

### 5) Camunda trace adapter (training data path)

Extended trace adapter to enrich events before RawTrace conversion:
- actor-role fields propagated to `EventRecord.extra`
- `process_variables` propagated to `EventRecord.extra`
- `call_activity_link` propagated to `EventRecord.extra`

File:
- `src/adapters/ingestion/camunda_trace_adapter.py`

## SQL templates

Added new template for generic process variables:
- `src/adapters/ingestion/camunda/sql/runtime/process_variables.sql`
- `src/adapters/ingestion/camunda/sql/manual/process_variables.sql`

Updated existing templates with richer fields:
- runtime:
  - `historic_activity_events.sql`
  - `historic_tasks.sql`
  - `identity_links.sql`
  - `multi_instance_variables.sql`
- manual:
  - `historic_activity_events.sql`
  - `historic_tasks.sql`
  - `identity_links.sql`
  - `multi_instance_variables.sql`

Updated export docs:
- `data/camunda_exports/README.MD`
- `src/adapters/ingestion/camunda/sql/manual/README.MD`

## Mock exports added

Added process variable mock file:
- `data/camunda_exports/mock_process_variables.csv`

Also prepared canonical example file:
- `data/camunda_exports/process_variables.csv`

## Tests added/updated

### Updated tests
- `tests/adapters/test_camunda_runtime_adapter.py`
  - new test: `test_runtime_adapter_reads_process_variables_from_files`
- `tests/application/test_instance_graph_assembler_service.py`
  - new test: `test_assembler_preserves_call_activity_link_and_process_variables`
- `tests/adapters/test_camunda_trace_adapter.py`
  - asserts for new enriched keys in event extras
- `tests/integration/test_cli_camunda_adapter_smoke.py`
  - mock copy helper includes `mock_process_variables.csv`

## Test execution results

### Focused Stage 3.1 tests
Command:
```bash
.\.venv\Scripts\python.exe -m pytest tests/adapters/test_camunda_runtime_adapter.py tests/application/test_instance_graph_assembler_service.py tests/adapters/test_camunda_trace_adapter.py tests/integration/test_cli_camunda_adapter_smoke.py -v
```
Result:
- `15 passed`

### Full suite
Command:
```bash
.\.venv\Scripts\python.exe -m pytest tests/ -v
```
Result:
- `115 passed`

### MVP1 Regression Shield
Command:
```bash
.\.venv\Scripts\python.exe -m pytest -m mvp1_regression -v
```
Result:
- `16 passed, 99 deselected`

## How to run (files vs sql)

### A) Files mode (manual exports CSV/XLSX)

1. Place exported files in `data/camunda_exports/`:
- `historic_activity_events.*`
- `historic_tasks.*`
- `identity_links.*`
- `execution_tree.*`
- `multi_instance_variables.*`
- `process_variables.*`

2. Run refresh:
```bash
.\.venv\Scripts\python.exe tools/stage3_1_runtime_refresh.py --config configs/experiments/mvp2_5_stage3_1_baseline_files.yaml --out outputs/stage3_1_refresh_files.json
```

3. Run full training experiment (same data adapter path):
```bash
python main.py --config configs/experiments/mvp2_5_stage3_1_baseline_files.yaml
```

### B) SQL mode (MSSQL)

1. Configure `mapping.camunda_adapter.runtime.mssql.connection_string` in data config.
2. Ensure SQL templates exist under `src/adapters/ingestion/camunda/sql/runtime/` (including `process_variables.sql`).
3. Run refresh:
```bash
.\.venv\Scripts\python.exe tools/stage3_1_runtime_refresh.py --config configs/experiments/mvp2_5_stage3_1_baseline_sql.yaml --out outputs/stage3_1_refresh_sql.json
```

4. Run full training experiment (same data adapter path):
```bash
python main.py --config configs/experiments/mvp2_5_stage3_1_baseline_sql.yaml
```

## Notes

1. Backward compatibility is preserved: missing optional tables/columns degrade safely to empty enrichment without runtime crash.
2. `call_activity_link` is now retained in canonical IG metadata and node payload, ready for cascade logic in next stages.
3. Process variable enrichment is isolated and optional; it does not break MVP1 training paths.
4. `version_key` remains optional; empty value means \"all available versions\".
