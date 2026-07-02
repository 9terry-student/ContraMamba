# Stage56 — Non-Leaking External Transfer Bridge Design

## Decision

`STAGE56_NONLEAKING_EXTERNAL_TRANSFER_BRIDGE_DESIGN_READY`

## Problem statement

Stage53A/55 showed VitaminC external failure dominated by NOT_ENTITLED overprediction. The bridge objective is to teach entitlement under open-domain-like lexical, numeric, temporal, and entity-attribute forms without using VitaminC data for training, calibration, checkpoint selection, or threshold tuning.

## Design principle

- Frame first: Preserve ContraMamba ordering: Frame first, Predicate second, Polarity third, Uncertainty/NE last.
- Not composer tuning: Do not solve the failure by changing composer thresholds on VitaminC.
- Not external supervision: Do not use VitaminC labels/text/examples for bridge generation.
- Controlled synthetic only: Generate new controlled bridge examples from templates and synthetic entities/values.

## Summary

| decision                                                 | problem                                 | solution_type                       |   num_bridge_families | vitaminc_text_used   | vitaminc_labels_used   | next_stage                                      |
|:---------------------------------------------------------|:----------------------------------------|:------------------------------------|----------------------:|:---------------------|:-----------------------|:------------------------------------------------|
| STAGE56_NONLEAKING_EXTERNAL_TRANSFER_BRIDGE_DESIGN_READY | VitaminC external NOT_ENTITLED collapse | non-leaking synthetic bridge design |                     5 | False                | False                  | Stage57 bridge dataset generator implementation |

## Bridge families

| family                     | purpose                                                                                                | examples_to_generate                                                                                                                                                                                                | target_failure_from_stage55                                                        | labels                                | source_policy                                                                                |
|:---------------------------|:-------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|:--------------------------------------|:---------------------------------------------------------------------------------------------|
| entity_attribute_bridge    | Teach entitlement for ordinary entity-attribute facts without requiring synthetic frame identity cues. | ['X was born before/after YEAR.', 'X is a singer/actor/album/film/company/city.', 'X was founded in YEAR.', 'X is located in COUNTRY/CITY.']                                                                        | SUPPORT/REFUTE examples about ordinary entities collapse into NOT_ENTITLED.        | ['SUPPORT', 'REFUTE', 'NOT_ENTITLED'] | synthetic templates only; do not copy VitaminC claims/evidence/text.                         |
| numeric_comparison_bridge  | Teach entitlement for numeric thresholds and approximate comparisons under open-domain wording.        | ['less than / more than / at least / fewer than / exactly', 'population, cases, gross revenue, scores, dates, counts', 'claim number differs from evidence number by small or large margin']                        | COVID/count/revenue examples collapse into NOT_ENTITLED despite explicit evidence. | ['SUPPORT', 'REFUTE', 'NOT_ENTITLED'] | synthetic numbers and controlled arithmetic only; no external dataset text.                  |
| temporal_comparison_bridge | Teach before/after/as-of relations without using time_swap as main classifier data.                    | ['before YEAR / after YEAR / as of DATE', 'event happened in YEAR vs claim threshold YEAR', 'founded/released/born/died before-or-after claim']                                                                     | Temporal entitlement cases collapse into NOT_ENTITLED.                             | ['SUPPORT', 'REFUTE', 'NOT_ENTITLED'] | new synthetic temporal templates; do not reintroduce corrupted time_swap into main training. |
| lexical_paraphrase_bridge  | Reduce domain shift from exact controlled wording to open-domain paraphrase wording.                   | ['was sentenced to four years -> had a four-year sentence', 'is the seventh studio album -> is an album', 'was founded in 1988 -> was founded before 1990', 'is a television competition show -> is a competition'] | Obvious paraphrastic SUPPORT cases collapse into NOT_ENTITLED.                     | ['SUPPORT', 'REFUTE', 'NOT_ENTITLED'] | manually specified synthetic paraphrase patterns, not mined from VitaminC.                   |
| distractor_evidence_bridge | Preserve NOT_ENTITLED when evidence is related but insufficient, without overgeneralizing NE.          | ['same entity but missing predicate', 'same domain but wrong subject', 'evidence mentions only partial role/list/count', 'evidence provides related context but not claim-verifying relation']                      | Need distinguish true insufficiency from open-domain paraphrastic sufficiency.     | ['NOT_ENTITLED']                      | synthetic distractors; no external dataset examples.                                         |

## Risk controls

| risk                                           | control                                                                                                                                            |
|:-----------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------|
| VitaminC leakage through design                | Use Stage55 only for failure category names, not for copying rows, labels, strings, or thresholds tuned on external performance.                   |
| Overcorrecting NE collapse into unsafe SUPPORT | Every bridge family must include REFUTE and NOT_ENTITLED counterexamples, not only SUPPORT recovery examples.                                      |
| Reintroducing time_swap corruption             | Do not use data/controlled_v5_v3.jsonl time_swap rows in main classifier training. Generate new temporal bridge rows with explicit audit metadata. |
| Composer threshold tuning on external set      | Do not alter composer thresholds using VitaminC metrics. Bridge is a training-data/diagnostic design, not an external-set calibration.             |
| Internal clean-dev degradation                 | Next training stage must report clean-dev, pairwise controlled diagnostics, and external diagnostic separately.                                    |

## Leakage policy

- VitaminC text used for generation: `False`
- VitaminC labels used for generation: `False`
- VitaminC metrics used for threshold tuning: `False`
- Stage55 used for failure taxonomy only: `True`

## Recommended next stage

Stage57: `bridge dataset generator implementation`

Expected outputs:
- `data/stage57_nonleaking_external_bridge.jsonl`
- `results/stage57_nonleaking_external_bridge_audit.json`
- `results/stage57_nonleaking_external_bridge_audit.md`
