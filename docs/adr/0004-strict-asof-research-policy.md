# ADR-0004: Strict As-Of Research Policy

Date: 2026-04-27
Status: Accepted

## Context

Research-grade temporal experiments must avoid future leakage. Stage 4.2
introduced `strict_asof` lookup semantics and `on_missing_asof_snapshot` fallback
policy.

Worklogs identify silent fallback in strict temporal studies as blocking debt
for defensible structure/drift experiments.

## Decision

For research-grade temporal studies, use:

```yaml
experiment.stats_time_policy: strict_asof
experiment.on_missing_asof_snapshot: raise
```

Exploratory runs may use fallback policies such as `disable_stats` or `use_base`,
but those runs must not be treated as final dissertation-grade evidence.

## Consequences

Positive:

1. temporal leakage risk is explicit,
2. missing snapshot coverage is surfaced early,
3. experiment results are easier to defend scientifically,
4. snapshot timeline preparation becomes measurable.

Negative:

1. strict runs require maintained snapshot timelines,
2. first runs may fail until `sync-stats-backfill` coverage is sufficient,
3. exploratory convenience and research-grade rigor are intentionally different.

## Runtime Rules

1. Use `sync-stats-backfill` to prepare a periodic snapshot timeline when needed.
2. In `strict_asof`, runtime lookup uses the prefix last-event timestamp.
3. If no eligible snapshot exists for a research-grade run, fail fast.
4. If a fallback policy is used, report it as exploratory.
5. Forward logs must preserve snapshot metadata and missing-as-of counters.

## Affected Files

- `tools/sync_stats.py`
- `tools/sync_stats_backfill.py`
- `src/domain/services/dynamic_graph_builder.py`
- `src/application/use_cases/trainer.py`
- `configs/experiments/`

## Related

- `docs/EVF_MVP2_5.MD`
- `docs/LLD_MVP2_5.MD`
- `docs/DATA_FLOWS_MVP2_5.MD`
- `docs/worklogs/MVP2_5_Dissertation_Alignment_and_Blocking_Debt_Analysis_2026-03-21.MD`
- `docs/worklogs/MVP2_5_Canonical_Doc_Sync_and_Architecture_Debt_2026-04-24.MD`

