# ADR-0001: Agent Knowledge Base Governance

Date: 2026-04-27
Status: Accepted

## Context

The repository accumulated many Markdown files across MVP1, MVP2, MVP2.5, and
Stage 4.2 worklogs. Many worklogs remain useful historical context, but they are
not equally current or equally authoritative.

Loading all documentation by default is inefficient for agents and increases the
risk of using stale guidance.

Relevant historical context:

- `docs/worklogs/MVP2_5_Context_Documentation_Sync_Analysis_2026-03-21.MD`
- `docs/worklogs/MVP2_5_Canonical_Doc_Sync_and_Architecture_Debt_2026-04-24.MD`

## Decision

Use `AGENTS.MD` as the first-read routing document for agents.

Use ADRs for durable decisions.

Use worklogs only as historical context unless explicitly referenced by:

1. `AGENTS.MD`,
2. `docs/current/project-state.md`,
3. an accepted ADR,
4. the user.

Agents must not read all Markdown files by default.

## Consequences

Positive:

1. agent context usage becomes predictable,
2. stale worklogs stop acting as hidden source of truth,
3. architectural decisions become easier to audit,
4. future sessions have a smaller and more reliable documentation entrypoint.

Negative:

1. existing documents need gradual status cleanup,
2. some old links may remain until the documentation map is migrated,
3. agents must follow routing discipline instead of broad documentation loading.

## Runtime Rules

1. Read `AGENTS.MD` first.
2. Read `docs/current/project-state.md` second.
3. Read `docs/current/architecture-debt.md` third.
4. Read only task-routed documents after that.
5. Do not treat `docs/worklogs/*` as current requirements unless explicitly routed.
6. If a worklog and an accepted ADR conflict, the ADR wins.

## Affected Files

- `AGENTS.MD`
- `docs/current/project-state.md`
- `docs/current/architecture-debt.md`
- `docs/adr/`
- `docs/worklogs/`

## Related

- `docs/worklogs/MVP2_5_Context_Documentation_Sync_Analysis_2026-03-21.MD`
- `docs/worklogs/MVP2_5_Canonical_Doc_Sync_and_Architecture_Debt_2026-04-24.MD`
