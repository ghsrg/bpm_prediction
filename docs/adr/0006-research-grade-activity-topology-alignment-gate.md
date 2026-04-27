# ADR-0006: Research-Grade Activity-to-Topology Alignment Gate

Date: 2026-04-27
Status: Proposed

## Context

Structural enrichment depends on alignment between:

1. activity identifiers from event traces,
2. BPMN/topology node identifiers persisted in the knowledge repository,
3. stats indexes projected into model class space.

Stage 4.2 already has a producer-side alignment gate in `sync-stats`.
It computes metrics including:

- `event_match_ratio`
- `unique_activity_coverage`
- `node_coverage`

Current configurable behavior supports:

- `write_with_flag`
- `skip_snapshot`
- `raise`

The dissertation alignment worklog still treats research-grade alignment as
blocking debt because exploratory configurations may allow weak alignment to be
written with flags.

## Decision

Target contract:

Research-grade runs require a strict activity-to-topology alignment profile with
locked thresholds and fail-fast behavior.

Exploratory runs may keep configurable alignment fallback policies, but final
structure/drift evidence must not rely on weakly aligned snapshots.

## Consequences

Positive:

1. structural signal quality becomes auditable before train/eval,
2. silent degradation from identifier mismatch is reduced,
3. XES classifier and BPMN node-id mismatches become explicit,
4. dissertation-grade experiments can report alignment acceptance criteria.

Negative:

1. strict profiles may reject datasets that need mapping cleanup,
2. users may need dataset-specific normalization or mapping fixes before
   research runs,
3. some exploratory convenience is intentionally separated from final evidence.

## Runtime Rules

Current behavior:

1. `sync-stats` computes alignment metrics.
2. `sync-stats.alignment_gate.on_fail` controls behavior.
3. Alignment metadata is written into `metadata.stats_contract.alignment`.

Target research-grade behavior:

1. use locked thresholds for `event_match_ratio`,
   `unique_activity_coverage`, and `node_coverage`,
2. use `on_fail: raise` or an equivalent fail-fast profile for final runs,
3. report alignment metrics in experiment artifacts,
4. do not treat `write_with_flag` snapshots as final research-grade evidence.

## Affected Files

- `tools/sync_stats.py`
- `tests/application/test_sync_stats_tool.py`
- `configs/experiments/`
- `configs/ui/config_catalog.yaml`
- future experiment acceptance docs

## Related

- `docs/worklogs/MVP2_5_Dissertation_Alignment_and_Blocking_Debt_Analysis_2026-03-21.MD`
- `docs/worklogs/MVP2_5_Canonical_Doc_Sync_and_Architecture_Debt_2026-04-24.MD`
- `docs/DATA_MODEL_MVP2_5.MD`
- `docs/EVF_MVP2_5.MD`

