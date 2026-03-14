# MVP2.5 Stage 3.1 Report 4

## Goal
Fix canonical IG assembly so embedded subprocesses are not rendered as fan-out and preserve correct parallel/sequence semantics for case:
`29537244-fdf3-11f0-bc36-26cbd5bba463`.

## Root Cause Analysis
1. Subprocess container rewiring redirected parent flow directly to first internal child, so top-level branch looked like a fan.
2. Old execution-parallel refinement and new scope-sequencing logic overlapped, producing conflicting edges.
3. Sequential chain between sibling branch nodes was not removed when parallel branches were detected.
4. Event ordering for active tasks (`end_time=None`) could place unfinished tasks before gateways with same start timestamp.

## Implementation Plan (Applied)
1. Keep subprocess container nodes in parent flow; embed internal flow inside container.
2. Build sequence edges by scope (`parent_act_inst_id`) and exclude phantom inner start/end from inner chain.
3. Add container entry/exit edges (`sequence` in flattened mode; `subprocess_entry/subprocess_exit` in hierarchical mode).
4. Rewire split/join parallel pattern by scope (not by container inner-entry replacement).
5. Remove forced sequence edges between branch entry nodes after parallel detection.
6. Stabilize ordering using `sequence_counter` first; fallback on timestamp with `end_time=None -> datetime.max`.
7. Add tests for flattened/hierarchical subprocess modes, callActivity flags, and real-case regression.

## Code Changes
- Updated subprocess + parallel logic in:
  - [instance_graph_assembler_service.py](/C:/Users/korsr/PycharmProjects/bpm_prediction/src/application/services/instance_graph_assembler_service.py)
    - `_build_graph_payload` (scope sequencing, container entry/exit, parallel cleanup)
    - `_rewire_parallel_subprocess_branches` (scope-based split/join rewrite)
    - `_event_order_key` (stable ordering for active tasks)
- Added/updated tests in:
  - [test_instance_graph_assembler_service.py](/C:/Users/korsr/PycharmProjects/bpm_prediction/tests/application/test_instance_graph_assembler_service.py)
    - subprocess flattened embedding
    - hierarchical entry/exit edge types
    - callActivity unresolved binding behavior without forced fallback
    - real-case regression for `B2BContracts_ApproveProject` case-id

## Validation Results

### 1) Targeted assembler tests
Command:
```bash
.venv\Scripts\python -m pytest tests/application/test_instance_graph_assembler_service.py -v
```
Result:
- `12 passed`

### 2) visualize-graph tool smoke
Command:
```bash
.venv\Scripts\python main.py visualize-graph --config configs/experiments/mvp2_5_stage3_1_baseline_files.yaml --case-id 29537244-fdf3-11f0-bc36-26cbd5bba463 --out outputs/ig_case_after_fix_stage4.png --mode activity-centric
```
Result:
- Graph saved successfully
- Summary: `nodes=36, edges=34, mode=canonical_activity_centric`

### 3) Regression shield + EOPKG forward tests
Commands:
```bash
.venv\Scripts\python -m pytest -m mvp1_regression -v
.venv\Scripts\python -m pytest tests/domain/test_eopkg_models_forward.py -v
```
Result:
- MVP1 regression: `17 passed`
- EOPKG forward tests: `6 passed`

## Behavior Confirmed on Target Case
- `Gateway_1ag44zb -> Activity_0z3oyip` is `parallel_branch`
- `Gateway_1ag44zb -> Activity_0ztae70` is `parallel_branch`
- No forced sequential edge between those two subprocess containers
- Subprocess containers expose internal entry/exit edges, enabling embedded visualization semantics

## Notes
- If `sequence_counter` is absent in exports, ordering falls back to timestamps and deterministic tie-breakers.
- For callActivity nodes, graph keeps supernode semantics with inference flags (`requires_separate_inference`, `inference_entry_point`).
