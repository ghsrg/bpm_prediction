# MVP2.5 Stage 3.4 Docs Update Plan

Date: 2026-03-19
Scope: Freeze and synchronize documentation after Stage 3.4 (stats sync + snapshots + as-of retrieval), before full stats integration into train/inference/eval_drift.

## 1. Problem Summary (Current Doc Gap)

1. Stage 3.4 behavior is implemented in code, but canonical docs are only partially updated.
2. Some top-level docs still have mixed phase narrative (MVP2/MVP2.5 overlap) and unclear "what is already done" vs "next steps".
3. There is encoding/corruption noise in key human-facing files (README/AGENT_GUIDE snippets), which increases risk of wrong execution and planning.
4. New semantics are not fully centralized in one place:
   - immutable `StatsSnapshot`
   - strict composite key
   - JSON-only stats payload (no `:NodeStat/:EdgeStat` normalization)
   - `as_of` lookup behavior
   - current limitation: stats are consumed via basic mapping, not full graph feature encoder pipeline yet.

## 2. Documentation Update Objectives

1. Make MVP2.5 docs the single source of truth for Stage 3.4 runtime behavior.
2. Explicitly separate:
   - Stage 3.4 completed capabilities
   - Stage 3.5/next task: full statistical feature integration into train/inference/eval_drift.
3. Eliminate ambiguity in commands and config ownership.
4. Remove stale or conflicting descriptions across architecture/data/model/evaluation docs.

## 3. Files To Update (Mandatory)

## 3.1 P0 - Canonical MVP2.5 Pack

1. `docs/ARCHITECTURE_MVP2_5.MD`
   - Add finalized Stage 3.4 block in architecture timeline.
   - Show where `sync-stats` sits between `sync-topology` and train/eval.
   - Add explicit note that stats ingestion is offline and immutable snapshots are append-only.

2. `docs/DATA_MODEL_MVP2_5.MD`
   - Freeze `ProcessStructureDTO` stats-related sections:
     - `node_stats`, `edge_stats`, `gnn_features`, `stats_diagnostics`, `metadata.stats_index`.
   - Document snapshot identity key:
     - `tenant_id + process_name + version_key + proc_def_id + knowledge_version`.
   - Document JSON-only storage policy for Neo4j snapshots.

3. `docs/DATA_FLOWS_MVP2_5.MD`
   - Add end-to-end flow for Stage 3.4:
     - `sync-topology -> sync-stats(--as-of optional) -> repository snapshot -> train/eval builder lookup`.
   - Add as-of retrieval decision point (`strict_asof` vs `latest`).
   - Add fallback behavior when no eligible snapshot exists.

4. `docs/LLD_MVP2_5.MD`
   - Low-level behavior of repositories:
     - `save_process_structure_snapshot`, `get_process_structure_as_of`.
   - Neo4j storage semantics (APOC-free, JSON payload fields, no TTL).
   - Clarify current builder mapping behavior and known limitation (no advanced feature encoding policy yet).

5. `docs/EVF_MVP2_5.MD`
   - Add evaluation reproducibility policy with `as_of` snapshots.
   - Add anti-leakage section for stats timeline experiments.
   - Add recommended experiment protocol (monthly snapshots / temporal slices).

## 3.2 P1 - Operational Entry Points

1. `README.MD`
   - Normalize encoding to valid UTF-8.
   - Consolidate command cookbook:
     - `ingest-topology`
     - `sync-topology`
     - `sync-stats` (`--as-of` examples)
     - train/eval_drift with strict_asof.
   - Add short "What Stage 3.4 does NOT do yet" section.

2. `PROJECT_CONTEXT.md`
   - Move "current state" to include Stage 3.4 final status.
   - Add exact note on partial stats-consumption state vs next full integration task.

3. `AGENT_GUIDE.MD`
   - Keep priority stack intact, but add explicit routing for Stage 3.4 docs first when task mentions stats/snapshots/as_of.
   - Ensure guidance text is readable (fix encoding issues only, no semantic regression).

## 3.3 P2 - Adapter Specs (consistency)

1. `docs/ADAPTER_CAMUNDA_SQL.MD`
   - Add explicit interface point with Stage 3.4 stats sync input expectations.
   - Clarify that stats sync reads runtime/case events and writes snapshot enrichment; no BPMN parse here.

2. `docs/ADAPTER_XES.MD`
   - Add note for research/offline experiments with XES-backed structure + optional stats timeline.

3. `docs/TARGET_ARCHITECTURE.MD`
   - Verify references to Stage docs only; avoid phase-level details duplication.
   - Keep it as target blueprint and ensure links point to MVP2.5 canonical docs for implementation details.

## 4. Proposed Work Sequence

1. P0 pass (5 MVP2.5 canonical docs).
2. P1 pass (README + PROJECT_CONTEXT + AGENT_GUIDE).
3. P2 consistency pass (adapter docs + target architecture references).
4. Cross-linking pass:
   - verify all command names and config keys match code (`sync-stats`, `stats_time_policy`, `graph_feature_mapping`, `strict_asof`).
5. Final QA pass:
   - markdown lint/readability
   - Mermaid syntax validation (if diagrams touched)
   - no broken links.

## 5. Acceptance Criteria

1. Every Stage 3.4 capability is documented in canonical MVP2.5 docs.
2. No contradiction between README, PROJECT_CONTEXT, AGENT_GUIDE, and MVP2.5 pack.
3. Snapshot policy is explicit and identical across docs:
   - immutable
   - strict composite key
   - JSON-only payload
   - as_of lookup
   - no TTL.
4. Clear statement of current limitation:
   - advanced graph feature encoding policy for stats is next task.
5. All examples are executable with existing CLI and configs.

## 6. Risks and Mitigations

1. Risk: documentation drift due to mixed MVP2/MVP2.5 narratives.
   - Mitigation: prioritize MVP2.5 pack as canonical and downgrade MVP2 docs to historical/reference context.

2. Risk: user confusion about as_of behavior during training.
   - Mitigation: add explicit timeline examples and fallback behavior in README + EVF.

3. Risk: encoding corruption reappears in edited files.
   - Mitigation: enforce UTF-8 save and run a quick post-edit readability check.

## 7. Next Step After Docs Freeze

After this docs update is merged:
1. Start implementation plan for "Full Stats Integration" (train/inference/eval_drift):
   - graph feature encoding policies
   - fit/transform/state persistence
   - leakage-safe fitting protocol
   - integration tests for strict_asof timeline behavior.
