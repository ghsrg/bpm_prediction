# MVP2.5 Stage 3.2 Pre-Plan

## 1. Goal
Decouple topology ingestion from model training so knowledge graph preparation runs as a separate pipeline, persisted to file-backed storage, and training only reads prebuilt structures.

Target outcomes:
1. New file-backed knowledge repository (`JSON`) replacing in-memory-only flow for production-like runs.
2. Dedicated `ingest` mode (no training) that builds and persists topology artifacts.
3. Trainer/graph builder no longer runs synchronous extraction.
4. Fallback remains safe: if knowledge artifact missing, training continues in baseline-compatible mode.

## 2. Current State (Code Audit)
1. Topology is extracted synchronously in `src/cli.py` during data preparation:
   - `TopologyExtractorService(...).extract_from_logs(train_traces, ...)`
   - then `DynamicGraphBuilder(..., knowledge_port=InMemoryNetworkXRepository())`
2. `DynamicGraphBuilder` already depends on `IKnowledgeGraphPort` (good abstraction).
3. `IKnowledgeGraphPort` contract already supports save/get/list and visualization graph fetch.
4. `main.py` routes `train`, `visualize-topology`, `visualize-graph`; no formal ingestion command yet.

## 3. Target Architecture

### 3.1 Runtime Modes
1. `ingest_topology` mode:
   - Read source traces (XES or Camunda runtime exports)
   - Extract topology
   - Save `ProcessStructureDTO` to file-backed repository
   - Exit
2. `train/eval` modes:
   - No extraction
   - Load topology from repository through `IKnowledgeGraphPort`
   - Build tensors; fallback if missing topology

### 3.2 Repository Strategy
1. Keep `InMemoryNetworkXRepository` for tests/dev utilities.
2. Add `FileBasedKnowledgeGraphRepository` for persisted artifacts.
3. Use repository factory in wiring layer (`cli.py`) based on config.

## 4. Work Breakdown

## WP1: File-Backed Repository

### Scope
Implement `FileBasedKnowledgeGraphRepository` in infrastructure.

### Files
1. `src/infrastructure/repositories/file_based_knowledge_graph_repository.py` (new)
2. `src/domain/ports/knowledge_graph_port.py` (no signature change expected)
3. `tests/infrastructure/test_file_based_knowledge_graph_repository.py` (new)

### Design
1. Storage root default: `data/knowledge_graph/`.
2. Key space: `process_name/version`.
3. JSON path convention:
   - `data/knowledge_graph/<process_name>/<version>/process_structure.json`
4. Persist fields:
   - Full `ProcessStructureDTO` payload + optional metadata block (`schema_version`, `created_at`, `source`, `artifact_hash`).
5. Read behavior:
   - strict JSON parse + DTO validation
   - corrupted/missing file -> graceful `None` in `get_process_structure`
6. `get_graph_for_visualization`:
   - reconstruct `networkx.DiGraph` from persisted DTO
   - apply `min_edge_frequency` filtering

### Acceptance
1. Save->load idempotent for DTO content.
2. Process isolation guaranteed (`same version` across different process names must not collide).
3. Missing file returns `None`, no crash.

## WP2: Repository Factory and Config Wiring

### Scope
Add config-based switch between in-memory and file-backed knowledge repositories.

### Files
1. `src/cli.py`
2. `configs/data/*.yaml` templates (comments + defaults)
3. `tests/integration/test_cli_knowledge_repo_wiring.py` (new)

### Config proposal
`mapping.knowledge_graph` block:
1. `backend`: `in_memory | file`
2. `path`: `data/knowledge_graph`
3. `strict_load`: `false` (if true -> fail when topology missing)
4. `artifact_namespace`: optional (for multi-tenant/process isolation)

### Acceptance
1. CLI builds `IKnowledgeGraphPort` from config only.
2. Existing MVP1 configs still work without adding this block.

## WP3: Ingestion Pipeline (Decoupled)

### Scope
Create a dedicated ingestion entrypoint.

### Recommended approach
Add new command in `main.py`:
1. `python main.py ingest-topology --config <...>`
2. Implementation in `tools/ingest_topology.py`

Reasoning:
1. Keeps training CLI behavior stable.
2. Cleanly separates operational tasks.
3. Easy to schedule in cron/Airflow later.

### Files
1. `tools/ingest_topology.py` (new)
2. `main.py` command routing update
3. `tests/application/test_ingest_topology_tool.py` (new)

### Behavior
1. Load config + adapter (`xes` or `camunda`).
2. Read traces/events.
3. Select source subset for topology (default train split or full historical window by config).
4. Run `TopologyExtractorService.extract_from_logs(...)`.
5. Persist to file repository via `save_process_structure`.
6. Print artifact summary:
   - process, versions saved, edge counts, output paths.

### Acceptance
1. Running ingestion produces JSON artifacts on disk.
2. Command exits without invoking trainer.

## WP4: Remove Sync Extraction from Training

### Scope
Stop extracting topology in training path.

### Files
1. `src/cli.py`
2. `tests/integration/test_cli_train_uses_prebuilt_topology.py` (new)

### Refactor
1. Remove from train preparation:
   - `TopologyExtractorService(...).extract_from_logs(...)`
2. Build `DynamicGraphBuilder` with repository loaded from config.
3. If structure missing:
   - `DynamicGraphBuilder` keeps current fallback (`allowed_target_mask=None`) and training continues.
4. Optional strict mode:
   - if `knowledge_graph.strict_load=true` and mode requires topology -> fail fast with clear error.

### Acceptance
1. Train command performs zero topology extraction calls.
2. Training succeeds with prebuilt artifacts.
3. Training still runs in fallback mode if artifacts absent and strict mode disabled.

## WP5: DynamicGraphBuilder and Compatibility Guardrails

### Scope
Ensure builder behavior is deterministic with file-backed reads.

### Files
1. `src/domain/services/dynamic_graph_builder.py`
2. `tests/domain/test_dynamic_graph_builder_file_repo.py` (new)

### Checks
1. Correct version/process lookup from repository.
2. Same mask results as in-memory repo for same DTO.
3. No side effects when repository read fails.

## WP6: End-to-End Validation Flow (Required by request)

### Scenario A (XES)
1. Run `ingest-topology` with XES config.
2. Assert JSON exists in `data/knowledge_graph/...`.
3. Run training config using same process/version.
4. Verify trainer loads topology and builds masks.

### Scenario B (Camunda exports)
1. Run `ingest-topology` with `mapping.adapter=camunda` (files/sql mode).
2. Verify persisted JSON.
3. Run visualize-topology from persisted store (without re-reading source logs).
4. Run train/eval using persisted store.

### Test commands to include in report
1. `python main.py ingest-topology --config <...>`
2. `python -m pytest -m mvp1_regression -v`
3. `python -m pytest tests/domain/test_dynamic_graph_builder* -v`
4. `python -m pytest tests/application/test_ingest_topology_tool.py -v`

## WP7: Backward Compatibility and Migration

### Migration strategy
1. Phase 1: keep current defaults non-breaking (`in_memory` unless explicitly configured to `file`).
2. Phase 2: switch Stage3.x configs to file-backed by default.
3. Phase 3: deprecate synchronous extraction path fully.

### Safety requirements
1. MVP1 unaffected by default behavior.
2. Existing visualization utilities remain operational.
3. Clear logs for source of topology:
   - `loaded from file` / `fallback no topology`.

## 5. Required Config Additions

## 5.1 Data/Mapping level
`mapping.knowledge_graph`:
1. `backend: file`
2. `path: data/knowledge_graph`
3. `strict_load: false`
4. `process_name_override: optional`

## 5.2 Experiment level
1. `experiment.mode: ingest_topology | train | eval_drift | ...`
2. If keeping subcommand approach only, mode value may remain optional; command controls flow.

Recommendation:
Use subcommand (`main.py ingest-topology`) + keep experiment mode unchanged to avoid breaking trainer logic.

## 6. What Else Is Needed Before Stage 3.2 (Complex BPMN Parsing)

## 6.1 Ingestion source abstraction hardening
Add a source-agnostic port layer for topology ingestion input:
1. `ITopologyIngestionSourcePort` with methods returning normalized trace/edge stream.
2. Adapters:
   - `XESTopologySourceAdapter`
   - `CamundaTopologySourceAdapter`
   - future enterprise adapters (BPMN parser, Neo4j bootstrap, external systems).

## 6.2 Artifact schema versioning
1. Add `artifact_schema_version` in JSON.
2. Add migration loader strategy for future DTO changes.
3. Add checksum/hash for reproducibility.

## 6.3 Process/version identity model
1. Use stable compound identity in artifact metadata:
   - `process_name`
   - `version_key`
   - `source_type` (`xes|camunda|bpmn`)
   - optional `proc_def_id/deployment_id` for Camunda.
2. Prevent overwrite collisions between heterogeneous sources.

## 6.4 Operational concerns
1. File locking / atomic writes during ingestion.
2. Partial-write protection (`tmp -> rename`).
3. Optional retention policy for old topology snapshots.

## 6.5 Stage 3.2 readiness criterion
System is ready for BPMN parsing when:
1. Ingestion and training are fully decoupled.
2. Topology artifacts are read-only for trainer.
3. Adapters can produce equivalent DTO payloads regardless of source.

## 7. Implementation Order (Recommended)
1. WP1 repository
2. WP2 factory/config wiring
3. WP3 ingestion command
4. WP4 remove sync extraction from trainer path
5. WP5 builder compatibility tests
6. WP6 validation runs and report
7. WP7 migration cleanup

## 8. Definition of Done
1. `ingest-topology` creates JSON artifacts from XES/Camunda sources.
2. Training no longer performs topology extraction.
3. DynamicGraphBuilder loads from file-backed repository.
4. Missing artifacts trigger fallback (or strict failure by config).
5. MVP1 regression shield green.
6. MVP2 dynamic graph tests green.
7. Documentation/config comments updated for new mode and repository settings.
