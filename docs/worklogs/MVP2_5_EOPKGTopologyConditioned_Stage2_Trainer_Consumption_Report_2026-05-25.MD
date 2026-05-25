# MVP2.5 EOPKGTopologyConditioned Stage 2 Trainer Consumption Report

- `date`: 2026-05-25
- `scope`: EOPKGTopologyConditioned dynamic candidate contract
- `status`: implemented

## Summary

Stage 2 foundation previously added `EOPKGTopologyConditioned.forward_candidate()`
and `CandidatePredictionOutput`, but standard train/eval/drift still consumed
`forward()` fixed-label logits.

This change adds an opt-in trainer/evaluator path:

```yaml
training.dynamic_candidate_contract_enabled: true
```

When enabled, `ModelTrainer` calls `model.forward_candidate(contract)`, consumes
`candidate_logits [B, C_v]`, and projects them to sparse fixed-label logits
`[B, C_train]` using `candidate_class_index`.

## What Changed

1. Added `ModelTrainer._forward_model_logits(contract)`.
2. Added `ModelTrainer._candidate_output_to_fixed_logits(output)`.
3. Replaced direct `self.model(contract)` calls in train, validation, test,
   drift one-pass inference, and topology-conditioned negative-loss paths.
4. Added `training.dynamic_candidate_contract_enabled` to
   `configs/ui/config_catalog.yaml`.
5. Added tests for:
   - `_evaluate_test()` using `forward_candidate()`;
   - `_run_epoch()` using `forward_candidate()`;
   - config catalog exposure.

## Current Semantics

```text
candidate_logits [B, C_v]
candidate_class_index [C_v]
    -> sparse fixed label logits [B, C_train]
```

Classes not present in current topology candidates receive a very low logit.
Existing CE, mask-guided inference, calibration, and drift metrics remain
compatible.

## Research Interpretation

This is a compatibility bridge, not the final zero-shot candidate contract.

It is now valid to say:

```text
training loss can optimize topology-local candidate scores
```

It is not yet valid to say:

```text
the evaluator fully supports unseen activity classes outside C_train
```

The next architecture step is direct candidate-id evaluation with BPMN node ids,
duplicate target mapping, and candidate-level OOS/calibration metrics without
projection back to fixed training vocabulary.

## How To Test

Enable:

```yaml
model.type: EOPKGTopologyConditioned
training.dynamic_candidate_contract_enabled: true
```

Compare against the same preset with:

```yaml
training.dynamic_candidate_contract_enabled: false
```

Expected signal:

- if metrics are identical, candidate ranking still matches the fixed-label
  fallback and the remaining bottleneck is data/features/objective;
- if metrics diverge, inspect `candidate_dynamic_count`,
  `candidate_prediction_entropy`, `candidate_score_gap`, OOS, ECE, and NLL.

