# ADR-0008: CLI Composition Root Boundary

Date: 2026-04-27
Status: Proposed

## Context

`src/cli.py` has grown beyond a thin composition root.

It currently contains multiple responsibilities:

1. config loading and override logic,
2. trace preparation and cascade splitting,
3. graph dataset cache orchestration,
4. telemetry/profile preparation,
5. model and trainer wiring.

This is acknowledged as architecture debt in the canonical debt worklog. It is
not currently a dependency-rule violation, but it weakens maintainability and
makes runtime behavior harder to audit.

## Decision

Target boundary:

`src/cli.py` should remain the composition root and command entrypoint for
train/eval runtime wiring.

Reusable orchestration logic should move into focused application-level helpers
or services when refactoring is planned.

Do not perform opportunistic large refactors of `src/cli.py` during unrelated
feature work.

## Consequences

Positive:

1. composition responsibilities become clearer,
2. data preparation can be tested independently,
3. cache orchestration can evolve without bloating CLI,
4. future run modes become easier to add safely.

Negative:

1. refactoring must be planned and tested,
2. short-term debt remains until a dedicated task is scheduled,
3. extraction boundaries need care to avoid moving complexity without reducing
   it.

## Runtime Rules

Current behavior:

1. `src/cli.py` remains the train/eval entrypoint.
2. `main.py` routes operational subcommands to tools.

Target behavior:

1. keep `src/cli.py` as composition root,
2. move data preparation orchestration into application-level services,
3. move graph cache orchestration into focused helper/service modules,
4. move run profile and report assembly into explicit helpers,
5. preserve existing CLI behavior while refactoring.

## Affected Files

- `src/cli.py`
- future application services/helpers
- tests for train/eval wiring and cache behavior

## Related

- `docs/worklogs/MVP2_5_Canonical_Doc_Sync_and_Architecture_Debt_2026-04-24.MD`
- `docs/ARCHITECTURE_RULES.MD`
- `docs/TARGET_ARCHITECTURE.MD`

