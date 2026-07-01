# Stage33-F Whole/Part Conditional-Safe Consistency Report

## Purpose

Stage33-F fixes whole/part conditional-safe action diagnostics while preserving the Stage33 shadow-only contract. Final logits, final predictions, H1 final decision logic, checkpoint selection, loss, calibration, and training data remain unchanged.

## Stage33-E Summary

Stage33-E improved whole/part detection and Stage31 external shadow performance, but exposed a diagnostic/action inconsistency: some whole-to-part rows were marked direct-support allowed while the conditional action still fell back to the current final prediction.

## Consistency Fix

For whole-to-part conditional-safe direct support:

- conservative default keeps hard-core priority
- if hard-core blocks an allowed direct-support row, the block reason is now explicit: `hard_core_priority_blocks_action`
- a new action block field distinguishes policy allowance from action priority blocking

When the override ablation is enabled and the row is allowed:

- `stage33_conditional_action = SUPPORT`
- `stage33_conditional_override_type = whole_part_conditional_safe_direct_support`
- `stage32_shadow_label = SUPPORT`
- `stage32_shadow_reason = stage33_conditional_whole_part_conditional_safe_direct_support`

## New Ablation Flag

- `--stage33-whole-part-conditional-safe-overrides-hard-core`

This flag is default off and only affects shadow-mode conditional-safe whole/part direct support.

## Pattern-Overfire Cleanup

Whole/part relation export is now suppressed when a higher-priority non-whole-part structured rule is selected, including quantifier, exclusive/additive, and contradiction rules. Those rows export:

- relation `none`
- empty match
- candidate false
- block reason `not_whole_to_part`

## Exported Diagnostics

- `stage33_whole_part_direct_support_action_block_reason`
- `stage33_whole_part_conditional_safe_override_hard_core_enabled`

Existing direct-support diagnostics remain:

- `stage33_whole_part_direct_support_candidate`
- `stage33_whole_part_direct_support_allowed`
- `stage33_whole_part_direct_support_block_reason`

## Evaluator Diagnostics

Added:

- `stage33_whole_part_action_block_reason_counts`
- `stage33_whole_part_allowed_but_fallback_count`
- `stage33_whole_part_allowed_but_fallback_rate`
- `stage33_whole_part_conditional_safe_override_hard_core_enabled_counts`
- `whole_to_part_allowed_but_fallback`
- `whole_to_part_conditional_safe_override_support`
- `whole_to_part_hard_core_false_support`
- `whole_to_part_hard_core_false_fallback`
- `whole_to_part_pattern_overfire_on_quantifier_count`

Decision labels added:

- `STAGE33F_CONDITIONAL_SAFE_CONSISTENT_PROMISING`
- `STAGE33F_CONSERVATIVE_LOCK_RECOMMENDED`
- `STAGE33F_DIAGNOSTIC_ONLY`

## Remaining Risks

- The override-hard-core ablation may improve Stage31 recovery while exposing dev or safety risk.
- Conservative mode may remain safest even if it leaves some whole-to-part examples unrecovered.
- Pattern cleanup reduces diagnostic contamination but does not broaden semantic understanding.
