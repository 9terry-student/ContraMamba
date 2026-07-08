# Stage122-A Prefix/Core Localization Diagnostic

## Purpose

Generate paired, label-preserved variants from Stage121 rows to test whether prefix-contaminated evidence causes polarity triggers, core suppression, position sensitivity, or weak evidence-core localization.

## Hypotheses

- Prefix-only polarity trigger: the prefix alone can elicit REFUTE, especially for SUPPORT rows.
- Core-evidence suppression: the core remains valid, but prefix+core hides or deweights it.
- Prefix position/order sensitivity: moving the prefix to a suffix recovers behavior.
- Discourse-wrapper semantics: discourse and temporal prefix wording drives inversion beyond length.
- Evidence-core localization: explicit Evidence markers can help the controller find the core.

## Variant Definitions

- `original_control`: Core relation sanity baseline using the original Stage121 evidence core.
- `prefix_plus_core`: Reconstructs prefixed evidence to reproduce prefix contamination.
- `prefix_only`: Uses the prefix alone to test whether it is a learned polarity trigger.
- `core_only`: Localization control verifying that evidence core alone remains label-correct.
- `core_plus_suffix`: Moves the prefix after the core to test front-position sensitivity.
- `core_with_neutral_marker`: Adds a minimal evidence marker before the core.
- `prefix_separator_core`: Adds an explicit Evidence marker between prefix and core.

## Counts

- Decision: `STAGE122A_PREFIX_CORE_LOCALIZATION_DIAGNOSTIC_GENERATED`
- Input rows: 2700
- Valid source rows: 2700
- Selected source rows: 900
- Output rows: 6200

### Source Label Counts

| label | count |
| --- | ---: |
| NOT_ENTITLED | 675 |
| REFUTE | 117 |
| SUPPORT | 108 |

### Output Label Counts

| label | count |
| --- | ---: |
| NOT_ENTITLED | 4650 |
| REFUTE | 806 |
| SUPPORT | 744 |

### Source Stage121 Family Counts

| stage121_family | count |
| --- | ---: |
| explicit_before_temporal | 100 |
| length_matched_neutral_tokens | 100 |
| length_matched_nonsense | 100 |
| long_discourse_no_temporal | 100 |
| long_neutral_no_order | 100 |
| old_before_prefix | 100 |
| original_control | 100 |
| short_neutral_fragment | 100 |
| short_neutral_sentence | 100 |

### Variant Counts

| variant | count |
| --- | ---: |
| core_only | 900 |
| core_plus_suffix | 900 |
| core_with_neutral_marker | 900 |
| original_control | 900 |
| prefix_only | 800 |
| prefix_plus_core | 900 |
| prefix_separator_core | 900 |

## Diagnostic Interpretation Guide

- `prefix_only` predicts REFUTE for SUPPORT families: prefix text itself is a learned negative polarity trigger.
- `core_only` recovers SUPPORT while `prefix_plus_core` fails: core relation is intact but prefix suppresses/localizes incorrectly.
- `core_plus_suffix` recovers relative to `prefix_plus_core`: leading prefix position is the main contaminant.
- `prefix_separator_core` recovers: explicit evidence-core marker helps localization.
- `length_matched_nonsense` `prefix_plus_core` still NE, not REFUTE: pure length closes gates rather than flipping polarity.
- Discourse prefixes produce REFUTE in `prefix_plus_core`: discourse semantics drive polarity inversion.

## Next-Stage Recommendation

Stage122-B evaluate prefix/core localization diagnostic with Stage118 generic diagnostic path.
