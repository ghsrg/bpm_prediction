# Delivery Roadmap

## Active Slices

| Plan ID | Title | Status | Confidence | Sources | Acceptance IDs | Evidence IDs |
| --- | --- | --- | --- | --- | --- | --- |
| CBR-001 | CDLG benchmark sequential runner | complete | high | `docs/superpowers/specs/2026-08-29-cdlg-benchmark-runner-design.md` | CBR-001-AC01, CBR-001-AC02, CBR-001-AC03, CBR-001-AC04 | CBR-001-EV01 to CBR-001-EV08 |
| MOU-001 | Topology-mask uniform baseline | complete | high | `docs/superpowers/plans/2026-08-31-topology-mask-uniform-baseline.md`; `outputs/worklogs/2026-08-31-closure-mou-001.md` | MOU-001-AC01 to MOU-001-AC07 | MOU-001-EV01 to MOU-001-EV20 |
| MSK-001 | GATv2 mask tracking identity | implemented | medium | User-approved `OK EXECUTE MSK-001`; `docs/GNN_RUNTIME_MVP2_5.MD` GATv2 + Mask reviewer-control contract | MSK-001-AC01, MSK-001-AC02 | MSK-001-EV01 to MSK-001-EV04 |
| SAO-001 | Strict-asof process-state mask DTO lookup | complete | high | User-approved strict-asof DTO lookup fix; `docs/GNN_RUNTIME_MVP2_5.MD` process-state-aware mask contract | SAO-001-AC01, SAO-001-AC02 | SAO-001-EV01 to SAO-001-EV08 |
| PSM-001 | Lifecycle active-candidate mask leakage | complete | high | `docs/superpowers/plans/2026-09-02-psm-001-lifecycle-mask-leakage.md` | PSM-001-AC01 to PSM-001-AC04 | PSM-001-EV01 to PSM-001-EV08 |
| BRG-001 | Fixed-vocab BPMN topology bridge and mock-graph model matrix | complete | high | `docs/superpowers/plans/2026-09-02-brg-001-fixed-vocab-topology-bridge.md`; `outputs/worklogs/2026-09-03-0824-REPORT-brg-001-fixed-vocab-topology-bridge.md` | BRG-001-AC01 to BRG-001-AC05 | BRG-001-EV01 to BRG-001-EV06 |

## Scope

`CBR-001` adds a CLI helper that executes an explicit ordered CDLG preset plan,
renders runtime progress events in the console, records run artifacts, and
stops safely on failed dependencies. It does not modify model, training, or
runtime experiment semantics.

`MOU-001` adds an evaluation-only uniform baseline over
`candidate_allowed_target_mask [B, C_v]`. It does not change mask construction,
topology ingestion, EOPKG/GAT scoring, training, or `ModelTrainer`.

`MSK-001` makes the `BaselineGATv2 + Mask` reviewer-control visible in MLflow
params as `BaselineGATv2Mask` when `BaselineGATv2` runs with
`training.mask_guided_enabled=true`. It does not add a new model factory type:
the underlying neural model remains `BaselineGATv2`.

`SAO-001` restores the strict-as-of topology DTO lookup used by process-state
and topology masks. When `experiment.stats_time_policy=strict_asof`, the graph
builder resolves process structure as of the last prefix event timestamp even
when stats-backed structural features are disabled.

`PSM-001` fixes lifecycle-active candidate leakage. With
`training.process_state_mask_enabled=false`, lifecycle metadata must not expand
topology-derived masks or structural prefix state. The same lifecycle metadata
is ignored when `training.process_state_mask_include_active_candidates=false`.
The graph dataset cache schema is bumped so stale per-prefix mask cache entries
are rebuilt.

`BRG-001` restores topology projection for `fixed_vocab_bridge` when BPMN node
IDs differ from XES activity labels. It adds a compact mock-graph contract
matrix for GATv2, GATv2 + Mask, EOPKG, and MOU.
