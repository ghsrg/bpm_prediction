# MVP2.5 Loan Simulation Node Coverage Gap

Date: 2026-04-28
Status: documented
Scope: `loan_v1_v4_simulated`, `sync-stats-backfill`, `activity_topology_alignment`

## Context

Після виправлення preset `S_Att-MN-loan` backfill почав працювати тільки з потрібним процесом:

- `selected_processes`: `loan_v1_v4_simulated`
- `processed_versions`: 32
- `skipped_versions`: 0
- `usable_for_training`: 32
- `alignment_event_match_ratio`: 1.0
- `alignment_unique_activity_coverage`: 1.0
- `alignment_failed`: 0

Це означає, що всі activity ids, які реально присутні у XES events, успішно мапляться на BPMN topology.

## Finding

`aggregate.alignment.min_node_coverage` залишається низьким (`0.023809523809523808`), але це не означає зламаний mapping.

Причина: aggregate зараз рахує мінімум по всіх backfill snapshots. Перший snapshot (`2025-01-01T00:29:35.928886Z`) має тільки один event у version/process scope, тому покриває лише 1 BPMN task із приблизно 41-42 loggable task nodes.

Фінальний snapshot (`2025-07-07T17:17:11.801497Z`) показує реалістичне покриття:

| version | final_node_coverage | matched / loggable |
|---|---:|---:|
| `v1` | `0.9024390243902439` | `37 / 41` |
| `v2` | `0.8780487804878049` | `36 / 41` |
| `v3` | `0.9024390243902439` | `37 / 41` |
| `v4` | `0.8333333333333334` | `35 / 42` |

## Interpretation

Це не критично для поточного навчання, тому що:

- `event_match_ratio = 1.0`: усі події з XES знайшли topology match.
- `unique_activity_coverage = 1.0`: усі унікальні activities з XES покриті.
- `alignment_failed = 0`: alignment gate не відхиляє snapshots.
- `is_usable_for_training = true`: усі 32 snapshots придатні для training.

Але це важливо для якості симуляції: частина BPMN task-like nodes реально не потрапляє в XES log. Gateways/start/end вже відкидаються через `ignore_structural_only_nodes=true`, тому залишковий gap стосується саме task nodes.

## Missing Loggable Nodes

Ці BPMN task ids присутні у BPMN, але не зустрілись у XES як `sim:bpmn_element_id`.

### v1

- `t_receive_update`
- `t_reopen_case`
- `t_return_form`
- `t_revise_offer`

### v2

- `t_receive_update`
- `t_reject_loan`
- `t_reopen_case`
- `t_return_form`
- `t_revise_offer`

### v3

- `t_receive_update`
- `t_reopen_case`
- `t_return_form`
- `t_revise_offer`

### v4

- `t_book_loan`
- `t_monitor_five`
- `t_monitor_nine`
- `t_receive_update`
- `t_reopen_case`
- `t_return_form`
- `t_revise_offer`

## Follow-Up Tasks

- Update loan simulation config/probabilities so rare or currently unreachable branches are generated often enough.
- Re-run `sync-stats-backfill` for `S_Att-MN-loan`.
- Compare final snapshot `alignment_node_coverage` by version after simulation changes.
- Consider extending `backfill_summary.json` with `latest_node_coverage` and `latest_by_process_version`, because `min_node_coverage` across all timeline snapshots is misleading for sparse first snapshots.
- Consider adding `missing_loggable_nodes` diagnostics to sync stats output for faster analysis.

## Decision

No immediate blocker for continuing architecture debt work.

This item should be treated as simulation coverage debt, not as an alignment gate bug.
