# Stage33-D Whole/Part Structured Coverage Report

## Purpose

Stage33-D adds conservative whole/part structured coverage rules while preserving the Stage33-C conditional fallback policy. The goal is to recover Stage31 `whole_to_part_support` cases without turning `part_to_whole_not_entitled` into SUPPORT.

## Stage33-C Summary

Stage33-C showed that structured coverage is safer as a local shadow correction layer than as a global replacement composer. Conditional fallback preserved dev behavior by falling back to the current final prediction for residual, unresolved, weak proxy, and blocked cases.

## Whole/Part Motivation

The remaining Stage31 gap is whole/part coverage:

- `whole_to_part_support` remained unresolved and shadowed to NOT_ENTITLED.
- `part_to_whole_not_entitled` remained safe under fallback, but should be explicitly recognized as overclaim when conservative evidence is available.

## Implemented Heuristics

Whole/part matching is deterministic and disabled by default. When enabled, it uses normalized lowercase phrase matching with light singular/plural variants for:

- `service` / `services`
- `project` / `projects`
- `employee` / `employees`

Built-in pairs:

- `employees -> engineers`
- `research projects -> biology projects`
- `services -> payment service`
- `projects -> biology projects`
- `company -> engineers at the company`
- `department -> biology projects in the department`
- `platform -> payment service on the platform`

Custom pairs can be added with `--stage33-structured-coverage-whole-part-lexicon` using `whole->part` or `whole:part`.

## Rule Behavior

When `--stage33-structured-coverage-enable-whole-part-rules` is active:

- universal whole evidence plus recognized part claim routes as `STRUCT_ENTAILMENT_PRESERVE` with reason `whole_to_part_proxy`
- recognized part evidence plus universal whole claim routes as `STRUCT_OVERCLAIM_NE` with reason `part_to_whole_proxy`
- both rules keep `rule_strength = proxy` and `confidence = 0.75`

## Direct Support Policy

By default, `whole_to_part_proxy` does not directly recover SUPPORT. It can recover through the existing conditional positive-polarity path.

Direct whole/part SUPPORT requires:

- conditional fallback enabled
- `--stage33-structured-coverage-whole-part-direct-support`
- `hard_core.pass is true`

The direct-support shadow reason is `stage33_conditional_whole_part_direct_support`.

## Exported Fields

- `stage33_structured_coverage_whole_part_enabled`
- `stage33_structured_coverage_whole_part_relation`
- `stage33_structured_coverage_whole_part_match`
- `stage33_structured_coverage_whole_part_direct_support_enabled`

Existing Stage33 structured and conditional fields remain unchanged.

## Evaluator Diagnostics

The evaluator now reports:

- `stage33_whole_part_relation_counts`
- `stage33_whole_part_match_counts`
- `stage33_whole_part_direct_support_enabled_counts`
- `whole_to_part_stage33_entailment_preserve`
- `whole_to_part_stage33_shadow_support`
- `whole_to_part_stage33_unresolved`
- `part_to_whole_stage33_overclaim_ne`
- `part_to_whole_stage33_shadow_ne`
- `part_to_whole_stage33_unresolved`
- `whole_part_shadow_overclaim_to_support`
- `whole_part_shadow_refute_to_support`

## Shadow-Only Guarantee

Stage33-D does not change final logits, final predictions, H1 final decision logic, checkpoint selection, loss, calibration, threshold selection, training data, caps, or boosts.

## Remaining Risks

- The lexicon is intentionally narrow and may miss valid whole/part paraphrases.
- Custom lexicon entries should be inspected carefully because they can create broader proxy behavior.
- Direct whole/part SUPPORT remains experimental and should stay shadow-only until validated.
