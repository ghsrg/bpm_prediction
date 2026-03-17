# MVP2.5 Audit Report

Date: 2026-03-17  
Scope: documentation sync + technical debt scan for current MVP2.5 state (up to Stage 3.2).

Update (same day): P0/P1/P2 remediation was implemented after this audit snapshot:
1. `TARGET_ARCHITECTURE.MD` rewritten to current module layout and ports.
2. `AGENT_GUIDE.MD` document priority updated to make MVP2.5 docs primary.
3. Double trace read removed from main train path by passing preloaded artifacts to `ModelTrainer`.
4. Stage3 fail-fast policy added for missing/invalid `mapping.knowledge_graph.backend`.

## 1. What Was Updated

1. [PROJECT_CONTEXT.md](C:/Users/korsr/PycharmProjects/bpm_prediction/PROJECT_CONTEXT.md)
   - Rewritten to current MVP2.5 reality (offline ingestion, repository-backed training, Stage 4 next).
2. [DATA_FLOWS_MVP2_5.MD](C:/Users/korsr/PycharmProjects/bpm_prediction/docs/DATA_FLOWS_MVP2_5.MD)
   - Flow A now includes both branches:
     - Camunda BPMN path (`CamundaBpmnAdapter` + parser),
     - legacy logs path.
3. [ADAPTER_CAMUNDA_SQL.MD](C:/Users/korsr/PycharmProjects/bpm_prediction/docs/ADAPTER_CAMUNDA_SQL.MD)
   - Corrected DTO field mapping (`act_inst_id`, `activity_def_id`, `parent_act_inst_id`, `sequence_counter`).
   - Clarified BPMN SQL payload field usage (`ACT_GE_BYTEARRAY.BYTES_`).
   - Aligned default BPMN files path with config (`data/camunda_exports/bpmn_xml/...`).
4. [ARCHITECTURE_GUIDELINES.MD](C:/Users/korsr/PycharmProjects/bpm_prediction/docs/ARCHITECTURE_GUIDELINES.MD)
   - Fixed reference casing `ARCHITECTURE_RULES.MD`.
5. [README.MD](C:/Users/korsr/PycharmProjects/bpm_prediction/README.MD)
   - Fixed `ARCHITECTURE_RULES.MD` link label.
6. Manual SQL docs/comments synced with current folder layout:
   - [README.MD](C:/Users/korsr/PycharmProjects/bpm_prediction/src/adapters/ingestion/camunda/sql/manual/README.MD)
   - [process_definitions.sql](C:/Users/korsr/PycharmProjects/bpm_prediction/src/adapters/ingestion/camunda/sql/manual/process_definitions.sql)

## 2. Verification Runs

1. MVP1 regression shield:
   - `pytest -m mvp1_regression -v`
   - Result: `17 passed, 133 deselected`
2. Key MVP2.5 topology/repository tests:
   - `pytest tests/application/test_ingest_topology_camunda_bpmn.py tests/application/test_bpmn_structure_parser_service.py tests/infrastructure/test_file_based_knowledge_graph_repository.py tests/domain/test_dynamic_graph_builder_version_fallback.py -v`
   - Result: `13 passed`

Status: green for tested suites.

## 3. Sync Findings (Code vs Docs)

### High

1. Target architecture doc is not synchronized with current code paths.
   - Evidence:
     - [TARGET_ARCHITECTURE.MD](C:/Users/korsr/PycharmProjects/bpm_prediction/docs/TARGET_ARCHITECTURE.MD):51 references `src/core/interfaces/`
     - [TARGET_ARCHITECTURE.MD](C:/Users/korsr/PycharmProjects/bpm_prediction/docs/TARGET_ARCHITECTURE.MD):734 references `src/core/models/`
   - Current code uses `src/domain`, `src/application`, `src/infrastructure` layout.
   - Risk: wrong implementation guidance for next phases.

### Medium

1. `AGENT_GUIDE.MD` priority stack still focuses on MVP1/MVP2 list before MVP2.5 docs.
   - Evidence:
     - [AGENT_GUIDE.MD](C:/Users/korsr/PycharmProjects/bpm_prediction/AGENT_GUIDE.MD):19
     - [AGENT_GUIDE.MD](C:/Users/korsr/PycharmProjects/bpm_prediction/AGENT_GUIDE.MD):21
     - [AGENT_GUIDE.MD](C:/Users/korsr/PycharmProjects/bpm_prediction/AGENT_GUIDE.MD):26
     - [AGENT_GUIDE.MD](C:/Users/korsr/PycharmProjects/bpm_prediction/AGENT_GUIDE.MD):28
   - There is an MVP2.5 addendum, but ordering is still mixed and can cause wrong planning context.

2. Performance debt: traces are read twice in train flow (preparation + trainer run).
   - Evidence:
     - [cli.py](C:/Users/korsr/PycharmProjects/bpm_prediction/src/cli.py):320 (`adapter.read(...)`)
     - [trainer.py](C:/Users/korsr/PycharmProjects/bpm_prediction/src/application/use_cases/trainer.py):162 (`self.xes_adapter.read(...)`)
   - Impact: redundant I/O and long startup on large logs.

3. Backend default can silently fall back to in-memory repository.
   - Evidence:
     - [knowledge_graph_repository_factory.py](C:/Users/korsr/PycharmProjects/bpm_prediction/src/infrastructure/repositories/knowledge_graph_repository_factory.py) defaults `backend` to `in_memory` when config key is missing/invalid.
   - Impact: user may run ingestion/train expecting persistent file artifacts, but get transient memory behavior.

### Low

1. Two broken local links in historical worklog file.
   - Evidence:
     - [mvp2_5_Stage3_1_Report4.MD](C:/Users/korsr/PycharmProjects/bpm_prediction/docs/worklogs/mvp2_5_Stage3_1_Report4.MD)
   - Impact: documentation navigation only.

2. Worktree contains parallel config variants not yet normalized (`*_files.yaml` and `*_files2.yaml`).
   - Impact: configuration sprawl and operator confusion.

## 4. Technical Debt Backlog (Prioritized)

1. P0: Rewrite `docs/TARGET_ARCHITECTURE.MD` to true target-only view aligned with current module layout and Stage 4 trajectory.
2. P1: Refactor train orchestration to avoid double trace reading.
   - Option A: pass pre-read traces/datasets into trainer.
   - Option B: make trainer the single source of read and remove duplicate pre-read in `prepare_data`.
3. P1: Harden knowledge backend policy for MVP2.5 experiments.
   - Recommended for experiment configs: `mapping.knowledge_graph.backend: file` + `strict_load: true` where topology is mandatory.
4. P2: Normalize AGENT guide priority stack to explicitly rank MVP2.5 docs above MVP2 for current phase.
5. P3: Cleanup old/duplicate config variants and fix broken worklog links.

## 5. Conclusion

Core MVP2.5 Stage 3.2 path is operational and tested (green on selected suites + regression shield).  
Main risk now is documentation governance drift (`TARGET_ARCHITECTURE` and mixed guidance order), plus startup performance overhead from duplicated ingestion in train flow.
