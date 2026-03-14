# Camunda Runtime Export Folder (Stage 3.1)

Drop your manual Camunda exports here.

Supported files (CSV/XLSX):
1. `historic_activity_events.*`
2. `historic_tasks.*`
3. `identity_links.*`
4. `execution_tree.*`
5. `multi_instance_variables.*`
6. `process_variables.*`

The adapter also accepts `mock_*.csv` names for local testing.

Current files are mocks used by automated tests.
You can replace them with real exports without changing code.
