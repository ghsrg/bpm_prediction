# Architectural Decision Records

This directory contains Architecture Decision Records (ADR) for `bpm_prediction`.

ADRs are the canonical place for durable architectural decisions. They are not
worklogs, plans, or session notes.

---

## Status Values

- `Proposed`: decision is under discussion and must not be treated as final.
- `Accepted`: decision is active and must be followed.
- `Superseded`: decision was replaced by another ADR.
- `Deprecated`: decision is no longer recommended, but no direct replacement exists.

---

## Truth Rule

Accepted ADRs override historical worklogs.

If an accepted ADR conflicts with old `docs/worklogs/*` content, the ADR wins.

If an ADR conflicts with `AGENTS.MD`, stop and ask for clarification. `AGENTS.MD`
is the routing and operating guide; ADRs are decision records.

---

## Naming

Use sequential filenames:

```text
0001-short-decision-name.md
0002-short-decision-name.md
```

---

## Template

```md
# ADR-0000: Decision Title

Date: YYYY-MM-DD
Status: Proposed | Accepted | Superseded | Deprecated

## Context

Why this decision is needed.

## Decision

What is decided.

## Consequences

Positive and negative consequences.

## Runtime Rules

Concrete rules agents and developers must follow.

## Affected Files

- `path/to/file`

## Related

- `path/to/doc-or-worklog`
```

---

## Maintenance Rule

Create or update an ADR when a change affects:

1. dependency boundaries,
2. pipeline separation,
3. persistence contracts,
4. temporal or anti-leakage policy,
5. fallback policy,
6. tensor contract semantics,
7. accepted architecture debt direction.

