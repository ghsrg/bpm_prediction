# Delivery Roadmap

## Active Slices

| Plan ID | Title | Status | Confidence | Sources | Acceptance IDs | Evidence IDs |
| --- | --- | --- | --- | --- | --- | --- |
| CBR-001 | CDLG benchmark sequential runner | complete | high | `docs/superpowers/specs/2026-08-29-cdlg-benchmark-runner-design.md` | CBR-001-AC01, CBR-001-AC02, CBR-001-AC03, CBR-001-AC04 | CBR-001-EV01 to CBR-001-EV08 |
| MOU-001 | Topology-mask uniform baseline | complete | high | `docs/superpowers/plans/2026-08-31-topology-mask-uniform-baseline.md`; `outputs/worklogs/2026-08-31-closure-mou-001.md` | MOU-001-AC01 to MOU-001-AC07 | MOU-001-EV01 to MOU-001-EV20 |

## Scope

`CBR-001` adds a CLI helper that executes an explicit ordered CDLG preset plan,
renders runtime progress events in the console, records run artifacts, and
stops safely on failed dependencies. It does not modify model, training, or
runtime experiment semantics.

`MOU-001` adds an evaluation-only uniform baseline over
`candidate_allowed_target_mask [B, C_v]`. It does not change mask construction,
topology ingestion, EOPKG/GAT scoring, training, or `ModelTrainer`.
