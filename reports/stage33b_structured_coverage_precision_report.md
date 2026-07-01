# Stage33-B Structured Coverage Precision Report

## Purpose

Stage33-B keeps the Stage33 structured Coverage Owner shadow-only, but restricts direct SUPPORT recovery to high-precision entailment-preserve rules. Stage33-A showed strong Stage31 support recovery, while dev exports exposed proxy overreach from `specific_to_general_proxy`.

## Stage33-A Summary

- Stage31 external preserve was promising: support recovery appeared without overclaim/refute to SUPPORT errors.
- Dev preserve was unsafe: proxy lexical containment produced broad REFUTE to SUPPORT conversion.
- Interpretation: explicit directional rules are useful; proxy rules must remain diagnostic unless explicitly allowed.

## Precision Policy

Direct SUPPORT from `ENTAILMENT_PRESERVE` now requires all of:

- `--stage33-structured-coverage-preserve-can-support`
- rule reason in `--stage33-structured-coverage-direct-support-rules`
- rule strength `high_precision`
- `hard_core.pass is true`

Default direct-support rules:

- `quantifier_all_to_some`
- `only_to_base`

Proxy rules such as `specific_to_general_proxy` are blocked from direct SUPPORT by default.

## New Flags

- `--stage33-structured-coverage-direct-support-rules`
- `--stage33-structured-coverage-disable-specific-general-direct-support`
- `--stage33-structured-coverage-weak-rules-to-residual`

Comma-separated lists are trimmed and empty entries are ignored.

## Exported Fields

- `stage33_structured_coverage_original_reason`
- `stage33_structured_coverage_rule_strength`
- `stage33_structured_coverage_direct_support_allowed`
- `stage33_structured_coverage_direct_support_block_reason`

Existing Stage33-A fields remain exported when owner-state export is enabled.

## Evaluator Diagnostics

The Stage32 shadow composer evaluator now reports:

- rule-strength counts
- direct-support allowed counts
- direct-support block-reason counts
- Stage31 support recovery by structured reason
- overclaim/refute to SUPPORT safety errors by structured reason

Decision labels now distinguish:

- `STAGE33_STRUCTURED_OWNER_PROMISING`
- `STAGE33_STRUCTURED_OWNER_NEEDS_RULE_RESTRICTION`
- `STAGE33_STRUCTURED_OWNER_UNSAFE`

## Shadow-Only Guarantee

Stage33-B does not change final logits, final predictions, H1 final decision logic, checkpoint selection, training data selection, loss, calibration, thresholds, caps, or boosts.

## Known Limitations

- Whole/part support cases remain unresolved until a dedicated rule is added.
- Proxy rules are still simple lexical approximations and should be treated as diagnostic.
- Positive-polarity SUPPORT remains a separate shadow path; direct override restrictions apply only when polarity does not already predict SUPPORT.

## Next Step

Stage33-C should add a safer whole/part rule extension instead of broadening lexical proxy support.
