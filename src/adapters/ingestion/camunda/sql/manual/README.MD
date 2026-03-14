# Camunda Manual SQL Exports

These SQL templates are intended for manual exports from Camunda 7 DB to CSV/XLSX.

## How to use

1. Open the needed SQL file in your SQL client.
2. Edit process filter in `TARGET_PROC_KEYS` CTE:
   - Default example: `('procurement'), ('sales')`
3. Run query and export result to file.
4. Put exported files into `data/camunda_exports/`.

## Expected file names in `data/camunda_exports`

Canonical names:

1. `historic_activity_events.*`
2. `historic_tasks.*`
3. `identity_links.*`
4. `execution_tree.*`
5. `multi_instance_variables.*`

Adapter also accepts:

1. `mock_<canonical_name>.*`
2. Any file name that contains canonical base name (for example `historic_activity_events_2026_03.csv`)

Supported extensions:

1. `.csv`
2. `.xlsx`
3. `.xls`
