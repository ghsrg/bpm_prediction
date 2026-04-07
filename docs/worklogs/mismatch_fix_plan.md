# Target-Mask Mismatch Fix Plan (3 stages)

Updated: 2026-04-07
Scope: Align `allowed_target_mask` with `next complete` semantics under parallel execution.

## 1. Problem summary

Current mismatch source:
1. Target `y` is the observed next complete event.
2. `allowed_target_mask` is built mainly from structural topology.
3. In parallel branches, the next completion order is often non-deterministic.

Observed effect (RIO run):
1. Non-trivial `y not in mask` rate.
2. Mismatch concentration in parallel patterns.

## 2. Core principle (important correction)

`mask_active` is a good validity superset, but not a good ranking signal.

So:
1. Mask must answer: "What is allowed to complete next?"
2. Model logits must answer: "What is most likely to complete next?"

Implication:
1. We must not force ranking from `active` alone.
2. We use `active` primarily for OOS safety and label-mask alignment.

## 3. Stage breakdown

## Stage 1 - Main fix (XES only, minimal-risk)

Goal:
1. Remove systematic `target-mask` mismatch for XES lifecycle traces.

In-scope:
1. Lifecycle-aware active set reconstruction for XES prefixes.
2. New mask rule:
   - `allowed_target_mask = mask_active OR mask_struct`.
3. Keep current target:
   - `y = next complete` (no target definition change).
4. Keep current training loss:
   - pure cross-entropy (no new loss terms).
5. Add deterministic ordering/tie-break for equal timestamps.

Out-of-scope (for this stage):
1. Camunda generalization.
2. New loss design.
3. Ranking priors.

Deliverables:
1. Prefix-state module for XES (`active_activity_counts`, `active_instances`).
2. Mask-v2 implementation in graph builder.
3. Regression tests for parallel traces with `N > 2`.

Acceptance for Stage 1:
1. `target_in_mask_rate` is materially improved on XES parallel datasets.
2. No MVP1 baseline regression.

## Stage 2 - Validation and observability

Goal:
1. Prove semantic correctness and improve metric interpretability under parallelism.

Mandatory metrics to add and report:
1. `target_in_mask_rate = mean( I[mask[y_true]] )`
2. `pred_in_mask_rate = mean( I[mask[y_hat]] )`
3. `strict_error_but_allowed_rate = mean( I[y_hat != y_true and mask[y_hat]] )`
4. `ambiguous_prefix_rate = mean( I[mask_cardinality > 1] )`

Mandatory slicing:
1. by `mask_cardinality`:
   - `=1`, `=2`, `>=3`
2. by process version
3. by prefix-length bins

Keep existing strict metrics:
1. `accuracy`, `macro_f1`, `weighted_f1`, `precision_macro`, `recall_macro`, `ECE`
2. `OOS = mean( I[not mask[y_hat]] )`

Required interpretation rule:
1. If strict metrics drop but `pred_in_mask_rate` is high in ambiguous slices, this is parallel-order ambiguity, not structural violation.

Deliverables:
1. Audit report command/output for per-prefix mask-target checks.
2. Extended evaluation summary with new metrics and slices.
3. Documentation update for strict vs relaxed interpretation.

Acceptance for Stage 2:
1. New metrics available in eval outputs and tracked consistently.
2. Parallel ambiguity is diagnosable without manual log forensics.

## Stage 3 - Camunda generalization and advanced modeling

Goal:
1. Make the same semantics robust for Camunda runtime data and multi-instance parallelism.

In-scope:
1. Camunda lifecycle/state normalization:
   - stable activity-instance identity (`act_inst_id`, fallbacks),
   - concurrent same-activity counts (`active_count > 0`),
   - deterministic event ordering for ties.
2. Multi-instance and random completion order handling (N approvers).
3. Optional ranking improvements (without changing mask semantics):
   - time-in-state/age features,
   - duration hazard-like features,
   - optional priors in model input.
4. Optional loss extension (behind config flag):
   - hybrid `CE + set-aware` objective.

Important constraint:
1. `allowed_target_mask` remains a validity filter, not a hard ranking mechanism.

Deliverables:
1. Camunda parity tests vs synthetic XES parallel scenarios.
2. Optional experimental flags for ranking/loss enhancements.
3. Rollout plan for production presets.

Acceptance for Stage 3:
1. Stable behavior on Camunda parallel approvals with variable `N`.
2. No hidden dependence on deterministic execution order.

## 4. Why this 3-stage split is safe

1. Stage 1 fixes the core semantic bug with minimal blast radius.
2. Stage 2 prevents blind spots by adding the right validation metrics.
3. Stage 3 handles Camunda-specific complexity and optional modeling upgrades without blocking core fix.

## 5. Non-goals for Stage 1 and Stage 2

1. Full BPMN token replay engine.
2. Mandatory replacement of CE with set-loss.
3. Large architecture refactor of model families.
