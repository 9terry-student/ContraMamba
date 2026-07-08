# Stage121-A Prefix Length/Discourse Ablation

## Purpose

Generate label-preserved diagnostic rows that isolate whether Stage120 prefix failures come from prefix length, sentence/discourse wrapper form, or temporal/discourse lexical cues.

## Hypothesis

If collapse tracks only token count, length-matched nonsense and weak neutral tokens should behave like long discourse prefixes. If collapse tracks discourse or temporal wording, the long discourse and before-prefix families should be worse than length-only controls.

## Prefix Families

- `original_control` (none): ``
- `short_neutral_fragment` (short_fragment): `Additional context:`
- `short_neutral_sentence` (short_sentence): `Context is provided.`
- `length_matched_nonsense` (length_only): `alpha beta gamma delta epsilon zeta eta theta`
- `length_matched_neutral_tokens` (weak_neutral_tokens): `context note information detail passage text content statement`
- `long_neutral_no_order` (long_neutral_sentence): `This text contains background information and relevant evidence.`
- `long_discourse_no_temporal` (long_discourse_sentence): `The passage contains context and the relevant statement.`
- `old_before_prefix` (temporal_discourse): `The following passage provides broader context before the relevant statement.`
- `explicit_before_temporal` (temporal_discourse): `Before the relevant statement, the passage provides broader context.`

## Safety Constraints

- Claims are never rewritten.
- Gold labels are normalized and preserved.
- No external data is read or used.
- Generated rows are diagnostic-only.

## Summary Tables

- Decision: `STAGE121A_PREFIX_LENGTH_DISCOURSE_ABLATION_GENERATED`
- Input rows: 3600
- Valid source rows: 3600
- Selected source rows: 300
- Output rows: 2700

### Source Label Counts

| label | count |
| --- | ---: |
| NOT_ENTITLED | 225 |
| REFUTE | 38 |
| SUPPORT | 37 |

### Output Label Counts

| label | count |
| --- | ---: |
| NOT_ENTITLED | 2025 |
| REFUTE | 342 |
| SUPPORT | 333 |

### Family Counts

| family | count |
| --- | ---: |
| explicit_before_temporal | 300 |
| length_matched_neutral_tokens | 300 |
| length_matched_nonsense | 300 |
| long_discourse_no_temporal | 300 |
| long_neutral_no_order | 300 |
| old_before_prefix | 300 |
| original_control | 300 |
| short_neutral_fragment | 300 |
| short_neutral_sentence | 300 |

### Prefix Category Counts

| prefix_category | count |
| --- | ---: |
| length_only | 300 |
| long_discourse_sentence | 300 |
| long_neutral_sentence | 300 |
| none | 300 |
| short_fragment | 300 |
| short_sentence | 300 |
| temporal_discourse | 600 |
| weak_neutral_tokens | 300 |

### Trigger Counts By Family

| family | contains_before | contains_after | contains_following | contains_temporal_trigger | contains_discourse_marker | is_sentence_prefix |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| explicit_before_temporal | 300 | 0 | 0 | 300 | 300 | 300 |
| length_matched_neutral_tokens | 0 | 0 | 0 | 224 | 300 | 0 |
| length_matched_nonsense | 0 | 0 | 0 | 224 | 30 | 0 |
| long_discourse_no_temporal | 0 | 0 | 0 | 224 | 300 | 300 |
| long_neutral_no_order | 0 | 0 | 0 | 224 | 300 | 300 |
| old_before_prefix | 300 | 0 | 300 | 300 | 300 | 300 |
| original_control | 0 | 0 | 0 | 224 | 30 | 0 |
| short_neutral_fragment | 0 | 0 | 0 | 224 | 300 | 0 |
| short_neutral_sentence | 0 | 0 | 0 | 224 | 300 | 300 |

## Next-Stage Recommendation

Stage121-B evaluate prefix length/discourse ablation with Stage118 generic diagnostic path.
