# Stage34-C Evaluator Summary Fix Report

## Scope

Stage34-C updates only the held-out coverage evaluator reporting layer. It does not change model architecture, train logic, final logits, final predictions, shadow predictions, Stage33-F owner behavior, training data, Stage31 data, Stage34 probe examples, or checkpoint selection.

## Top-Level Diagnostics

The evaluator now computes top-level Stage33 diagnostic distributions directly from prediction rows:

- `stage33_structured_coverage_route_counts`
- `stage33_structured_coverage_reason_counts`
- `stage33_structured_coverage_rule_strength_counts`
- `stage33_conditional_action_counts`
- `stage33_conditional_override_type_counts`
- `stage33_whole_part_relation_counts`
- `stage33_whole_part_match_counts`
- `stage33_whole_part_direct_support_allowed_counts`
- `stage33_whole_part_direct_support_candidate_counts`

Held-out safety and recovery counters are also exported at top level, so report consumers no longer need to read nested `stage34_counters` to find the scalar diagnostics.

## Held-Out Summary

The evaluator now exports `heldout_group_summary` with compact per-group counts for gold labels, current predictions, shadow predictions, Stage33 routes/reasons, whole-part relations/matches, conditional actions, and conditional override types.

## Reverse Overclaim Handling

The evaluator now exports `stage34_reverse_overclaim_handling` with one of:

- `explicit_overclaim_route`
- `fallback_preserved_ne`
- `mixed`
- `unsafe`

Decision reasons now include the reverse-overclaim handling mode while preserving the existing promising/unsafe/diagnostic/memorization-risk labels.

## Markdown Summary

The Markdown report now includes aggregate metrics, support recovery, whole/part-family support recovery, reverse overclaim handling, safety counters, pattern-vs-lexicon match counts, and a caution note about fallback-preserved reverse-overclaim safety.

## Validation

Static syntax check only:

`python -m py_compile scripts/evaluate_stage34_heldout_coverage.py`

No training, evaluation, Kaggle run, smoke test, full experiment, or git command was run for this Stage34-C change.

## Remaining Risks

- Runtime values depend on the user's prediction export schema and will be validated by the user's Kaggle run.
- Reverse-overclaim handling is inferred from exported route/override metadata when explicit Stage34 route metadata is absent.
