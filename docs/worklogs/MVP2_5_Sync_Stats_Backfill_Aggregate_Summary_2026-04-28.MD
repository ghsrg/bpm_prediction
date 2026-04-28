# MVP2.5 Sync-Stats Backfill Aggregate Summary

Date: 2026-04-28
Status: completed

## Summary

`sync-stats-backfill` now writes compact aggregate diagnostics into
`backfill_summary.json`.

The goal is to avoid manually opening hundreds of per-run
`sync_stats_*.json` files after long backfills.

## Added Block

`backfill_summary.json` now includes:

```text
aggregate
```

It summarizes:

- run status counts,
- processed/skipped versions,
- usable/not-usable producer quality counts,
- producer quality reason counts,
- alignment ok/failed counts,
- minimum observed alignment ratios,
- alignment failure reasons,
- skipped snapshot reasons,
- compact per process-version rollups.

## Example Interpretation

For a single `sync_stats_*.json`, `status: ok` only means the run completed.
It does not mean the produced snapshots are usable for training.

The useful training signal is in per-version fields such as:

- `is_usable_for_training`
- `quality_reason`
- `alignment_is_ok`
- `alignment_reason`

The aggregate block lifts those fields to the backfill level.

## Verification

```powershell
.\.venv\Scripts\python.exe -m pytest tests/application/test_sync_stats_backfill_tool.py -v
```

Result:

```text
2 passed
```
