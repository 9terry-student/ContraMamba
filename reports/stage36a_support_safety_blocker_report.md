# Stage36-A: Support Safety Blockers (shadow-only)

## Background: Stage35-A adversarial baseline (unsafe)

Stage33-F was strong on Stage34-C held-out structured coverage (shadow macro-F1
0.9652, whole_to_part SUPPORT recovery 120/120, zero overclaim/refute leakage)
but the Stage35-A adversarial stress test exposed unsafe structured SUPPORT
overrides:

| Metric | Value |
|---|---:|
| Decision | `STAGE35A_ADVERSARIAL_UNSAFE` |
| Current macro-F1 | 0.2832 |
| Shadow macro-F1 | 0.5832 |
| Delta macro-F1 | +0.3000 |
| `adv_overclaim_to_support` | 58 |
| `adv_exception_to_support_error` | 8 |
| `adv_location_scope_to_support_error` | 25 |
| `adv_support_to_refute` | 4 |
| `adv_support_shadow_support` | 91 / 250 |
| `adv_subset_support_recovered` | 60 / 175 |
| `adv_numeric_support_recovered` | 0 / 25 |
| `stage35_reverse_overclaim_handling` | unsafe |
| `stage35_scope_safety` | unsafe |

Three concrete failure modes drove the unsafe scope/exception/quantifier
errors:

1. **Location scope mismatch** — "All users in the east district were
   reached." does not support "The verified users in the west district were
   reached." (gold `NOT_ENTITLED`, shadow was `SUPPORT` via
   `whole_part_direct_support`).
2. **Exception exclusion** — "All permits except the temporary permits were
   denied." does not support "The temporary permits were denied." (gold
   `NOT_ENTITLED`, shadow was `SUPPORT` via `entailment_positive_polarity`).
3. **Not-all existential fallacy** — "Not all permits were recalled." does not
   support "Some permits were recalled." (gold `NOT_ENTITLED`, shadow was
   `SUPPORT` via `high_precision_direct_support`).

## What Stage36-A adds

Stage36-A adds four conservative, deterministic safety blockers that run
**after** Stage33 proposes a structured/conditional shadow label and
**before** that label is treated as a safe SUPPORT override. Blockers only
ever fire on a *proposed* `SUPPORT` label — they never touch `REFUTE` or
`NOT_ENTITLED`, and never fire when the proposal is already
`fallback_current_final` non-SUPPORT.

- `--stage36-block-exception-scope`: blocks SUPPORT when evidence has an
  `all X except Y` clause and the claim targets the excluded subset `Y`
  (content-word overlap ratio >= 0.6 against the excluded phrase).
- `--stage36-block-not-all-existential`: blocks SUPPORT when evidence asserts
  `not all/every X` and the claim asserts `some X` (subject overlap ratio
  >= 0.5).
- `--stage36-block-location-scope-mismatch`: blocks SUPPORT when claim and
  evidence both carry an explicit location/scope marker (district, zone,
  region, campus, warehouse, east/west/north/south, ...) with conflicting
  qualifiers on both sides. Never fires if only one side has a location, or
  if both sides share the same qualifier.
- `--stage36-block-temporal-scope-mismatch`: blocks SUPPORT when claim and
  evidence both carry explicit temporal markers (year 2020-2029, weekday,
  quarter, or phrases like "last year"/"today") that do not overlap.

All four are gated behind `--stage36-use-support-safety-blockers` (master
switch), `--stage36-support-safety-export` (include `stage36_*` diagnostic
fields in prediction exports), and `--stage36-support-safety-shadow-mode`
(allow a fired blocker to actually replace the exported
`stage32_shadow_label` / `stage33_conditional_shadow_label`). All three
default to **off**. When off, behavior is byte-identical to Stage33-F.

When a blocker fires, `--stage36-support-blocker-action` controls the
replacement label:

- `fallback_current_final` (default): use the current final classifier label,
  or `NOT_ENTITLED` if that label is also SUPPORT.
- `force_not_entitled`: always replace with `NOT_ENTITLED`.

## No model or final-output changes

Stage36-A never modifies:

- model architecture, the Mamba backbone, or any training/loss/dataloader
  code path,
- `final_logits`, `final_probs`, or `pred_label`/`pred_final_label` (the
  classifier's real prediction),
- checkpoint selection,
- Stage31/Stage34/Stage35 data or the Stage33-F rule-extraction logic itself
  (Stage36 only adds a call site immediately before a proposed SUPPORT
  override is exported).

It only ever changes the exported `stage32_shadow_label` /
`stage33_conditional_shadow_label` diagnostic fields, and only when
`--stage36-support-safety-shadow-mode` is explicitly enabled.

## Expected impact (to be confirmed on Kaggle)

| Metric | Stage35-A baseline | Expected after Stage36-A |
|---|---:|---:|
| `adv_location_scope_to_support_error` | 25 | near 0 |
| `adv_exception_to_support_error` | 8 | near 0 |
| not-all-to-some SUPPORT errors (counted under `adv_overclaim_to_support`) | included in 58 | near 0 |
| `adv_overclaim_to_support` | 58 | lower |

The evaluator (`scripts/evaluate_stage35_adversarial_coverage.py`) now emits
paired before/after counters (`stage36_original_*` / `stage36_post_*`) and a
`stage36_decision` label
(`STAGE36A_SAFETY_BLOCKERS_EFFECTIVE` /
`STAGE36A_SAFETY_BLOCKERS_TOO_CONSERVATIVE` /
`STAGE36A_SAFETY_BLOCKERS_INEFFECTIVE` /
`STAGE36A_DIAGNOSTIC_ONLY`) whenever `stage36_*` fields are present in the
prediction export. Runs without Stage36 fields keep the original Stage35
`decision` label unchanged.

## Remaining risk

- `adv_numeric_support_recovered` (0/25) is not addressed by Stage36-A and
  remains unresolved.
- Passive/coordination SUPPORT recovery is unaffected by these blockers and
  likely still needs dedicated handling.
- Blocking is conservative but heuristic (regex + word-overlap); it is
  possible for `--stage36-support-safety-shadow-mode` to trade some SUPPORT
  recovery for safety, so `stage36_original_support_shadow_support` vs.
  `stage36_post_support_shadow_support` should be checked on every run to
  confirm recovery did not collapse (`STAGE36A_SAFETY_BLOCKERS_TOO_CONSERVATIVE`).
- All numbers above from the Stage35-A baseline section are carried over from
  the reported Kaggle run; this report does not re-run training or
  evaluation locally. Validation is expected on Kaggle per task guardrails.
