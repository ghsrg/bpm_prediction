# Connection Profiles

This directory stores non-secret connection profiles.

Rules:
1. Keep only non-secret values in YAML.
2. Keep credentials in environment variables (`.env.local` or CI secret store).
3. Reference env variable names via `*_env` keys.

## Files
1. `mssql_camunda.yaml` - MSSQL profile for Camunda adapters.
2. `neo4j.yaml` - Neo4j profile for knowledge graph backend.

## Usage in mapping config

For Camunda runtime/BPMN SQL mode:
```yaml
mapping:
  camunda_adapter:
    runtime:
      runtime_source: "mssql"
      connections_file: "configs/connections/mssql_camunda.yaml"
      connection_profile: "local"
      mssql:
        connection_profile: "local"
```

For future Neo4j backend:
```yaml
mapping:
  knowledge_graph:
    backend: "neo4j"
    neo4j:
      connections_file: "configs/connections/neo4j.yaml"
      connection_profile: "local"
```
