# Stage33-E Whole/Part v2 Report

## Purpose

Stage33-E broadens whole/part structured coverage while preserving Stage33-C conditional fallback safety. It remains shadow-only and does not change final logits, final predictions, H1 decision logic, checkpoint selection, loss, calibration, or training data.

## Stage33-D Summary

Stage33-D was safe but weak. It detected only a small subset of whole/part cases and recovered limited `whole_to_part_support`; `part_to_whole_not_entitled` remained safe.

## Expanded Lexicon

Stage33-E keeps the original pairs and optionally adds:

- `vehicles -> trucks`
- `members -> senior members`
- `models -> mark ii model`
- `floors -> third floor`
- `performing arts groups -> regional dance troupe`
- `financial records -> expense reports`
- `staff -> contract staff`
- `access roads -> eastern access road`
- `new hires -> new hires in operations`
- `top-ranked competitors -> top-ranked competitor from zone a`
- `subscribers -> premium subscribers`
- `nodes -> gateway nodes`
- `contracts -> service contracts`
- `registered participants -> registered participants from abroad`
- `residents in the zone -> residents in the northern sector of the zone`
- `construction projects -> residential construction projects`
- `persons -> foreign nationals`

The expanded lexicon is enabled with `--stage33-structured-coverage-whole-part-v2-use-expanded-lexicon`.

## Pattern Heuristic

When whole/part v2 is enabled, a conservative pattern heuristic can detect `all <WHOLE_NP> <suffix>` evidence paired with a more specific claim sharing substantial suffix/domain evidence. Uncertain cases remain unresolved.

## Direct Support Policy

New policy flag:

- `--stage33-structured-coverage-whole-part-v2-direct-support-policy`

Choices:

- `off`: no direct whole/part SUPPORT
- `hard_core_required`: direct SUPPORT requires `hard_core.pass is true`
- `conditional_safe`: direct SUPPORT requires conditional fallback, `whole_to_part`, and no conflicting high-precision contradiction/overclaim route

## Exported Diagnostics

- `stage33_structured_coverage_whole_part_v2_enabled`
- `stage33_structured_coverage_whole_part_v2_expanded_lexicon`
- `stage33_structured_coverage_whole_part_v2_direct_support_policy`
- `stage33_whole_part_direct_support_candidate`
- `stage33_whole_part_direct_support_allowed`
- `stage33_whole_part_direct_support_block_reason`
- `stage33_whole_part_hard_core_pass`
- `stage33_whole_part_original_current_label`
- `stage33_whole_part_conditional_action`

## Evaluator Updates

The evaluator reports v2 enabled counts, direct-support candidate/allowed/block-reason counts, relation and match counts, Stage31 whole-to-part recovery, part-to-whole safety, and Stage33-E decision labels.

## Remaining Risks

- Expanded pairs are still lexical and may miss paraphrases.
- Pattern matching is conservative by design and may under-detect.
- `conditional_safe` direct support should remain shadow-only until validation confirms safety.
