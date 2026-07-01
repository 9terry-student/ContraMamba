# Stage35-A Adversarial Coverage Probe Report

## Scope

Stage35-A adds a diagnostic-only adversarial structured coverage probe builder and evaluator. It does not change the train script, model architecture, final logits, final predictions, shadow predictions, Stage33-F owner logic, Stage34 evaluator logic, training data, Stage31 data, Stage34 data, or checkpoint selection.

## Probe Schema

The builder writes `data/stage35a_adversarial_coverage_probe.jsonl` when executed. Defaults:

- 600 rows total
- 24 groups
- 25 examples per group
- labels: `REFUTE = 0`, `NOT_ENTITLED = 1`, `SUPPORT = 2`

Each generated row includes:

`id`, `pair_id`, `claim`, `evidence`, `final_label`, `gold_label`, `label`, `group`, `intervention_type`, `normalized_intervention`, `primary_failure_type`, `failure_type`, `source`, `split`, `stage35_family`, `stage35_relation`, `stage35_perturbation`, `stage35_expected_route`, `stage35_is_adversarial`.

## Adversarial Groups

- `adv_whole_to_part_support_verb_diverse`
- `adv_part_to_whole_not_entitled_verb_diverse`
- `adv_whole_to_part_support_fronted_modifier`
- `adv_part_to_whole_not_entitled_fronted_modifier`
- `adv_whole_to_part_support_postnominal_modifier`
- `adv_part_to_whole_not_entitled_postnominal_modifier`
- `adv_whole_to_part_support_sentence_order_flip`
- `adv_part_to_whole_not_entitled_sentence_order_flip`
- `adv_all_except_subset_not_entitled`
- `adv_all_except_subset_support_for_nonexcluded`
- `adv_no_except_subset_support`
- `adv_no_except_nonexcluded_refute`
- `adv_exactly_some_to_all_not_entitled`
- `adv_all_to_at_least_some_support`
- `adv_not_all_to_some_not_entitled`
- `adv_none_to_any_refute`
- `adv_passive_active_support`
- `adv_passive_active_reverse_not_entitled`
- `adv_coordination_support`
- `adv_coordination_distractor_not_entitled`
- `adv_numeric_subset_support`
- `adv_numeric_reverse_not_entitled`
- `adv_temporal_scope_not_entitled`
- `adv_location_scope_not_entitled`

## Evaluator Outputs

The evaluator reports current metrics, shadow metrics, delta, group metrics, family metrics, perturbation metrics, route/reason/action counts, whole-part relation/match counts, support recovery, reverse-overclaim handling, scope safety, pattern diagnostics, and safety counters.

Decision labels:

- `STAGE35A_ADVERSARIAL_GENERALIZATION_STRONG`
- `STAGE35A_TEMPLATE_GENERALIZATION_ONLY`
- `STAGE35A_ADVERSARIAL_UNSAFE`
- `STAGE35A_DIAGNOSTIC_ONLY`

Additional classifier fields:

- `stage35_reverse_overclaim_handling`: `explicit_overclaim_route`, `fallback_preserved_ne`, `mixed`, or `unsafe`
- `stage35_scope_safety`: `safe`, `unsafe`, or `mixed`

## Diagnostic-Only Policy

Stage35-A must not be used for training, calibration, threshold selection, checkpoint selection, loss, or Kaggle selection.

## Validation

No training, evaluation, Kaggle run, smoke test, full experiment, or git command was run. Static syntax checks only are intended for this implementation step.

## Remaining Risks

- The adversarial probe is synthetic and should be interpreted as a stress test, not a real-world benchmark.
- Runtime outcomes depend on the user's external prediction export fields.
