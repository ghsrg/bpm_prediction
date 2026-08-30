# Delivery Roadmap

## Active Slices

| Plan ID | Title | Status | Confidence | Sources | Acceptance IDs | Evidence IDs |
| --- | --- | --- | --- | --- | --- | --- |
| CBR-001 | CDLG benchmark sequential runner | complete | high | `docs/superpowers/specs/2026-08-29-cdlg-benchmark-runner-design.md` | CBR-001-AC01, CBR-001-AC02, CBR-001-AC03, CBR-001-AC04 | CBR-001-EV01 to CBR-001-EV08 |

## Scope

`CBR-001` adds a CLI helper that executes an explicit ordered CDLG preset plan,
renders runtime progress events in the console, records run artifacts, and
stops safely on failed dependencies. It does not modify model, training, or
runtime experiment semantics.
