# ContraMamba Gen4 — Prospective Forced-Decisive Pre-emission Stage A Design

## Status

This document preregisters a **new prospective experiment** after closure of the original finite-grammar greedy Stage A.

It is not a rescue, continuation, reinterpretation, or replacement of the frozen all-abstain execution.

No generation from this new protocol has been executed or inspected at the time of this design freeze.

## 1. Prior closed experiment

The original Stage A is permanently closed by:

- raw evidence freeze commit:
  `a58b1ebef0957a8437e0255fb4069d2fe9cf8449`;
- static closure commit:
  `130aa76cf3bb0816505a45a18df8334445e6f0ce`.

Its bounded closure is:

`FINITE_GRAMMAR_GREEDY_DECODING_ALL_ABSTAINED`

`PRIMARY_STAGE_A_TEMPORAL_PRECEDENCE_NOT_ESTIMABLE`

The original execution generated `NOT_ENTITLED` for all 462 examples and therefore created neither a supported-decisive nor an unsupported-decisive primary comparison.

That result remains valid evidence about the original three-class finite grammar and must not be overwritten.

## 2. New scientific question

The new experiment asks a different and narrower question:

> When the model is prospectively required to make a decisive `REFUTE` versus `SUPPORT` commitment, does the already-frozen Mamba-370M block-35 P3 signal differ before an unsupported versus supported forced-decisive commitment is emitted?

This changes the estimand from spontaneous three-way commitment with abstention available to **forced decisive commitment conditional on a two-way finite grammar**.

It does not estimate spontaneous abstention behavior and does not support a broad hallucination claim.

## 3. Frozen objects retained unchanged

The following remain frozen and are not reselected:

- model: `state-spaces/mamba-370m-hf`;
- revision:
  `589179554943157be31701edd8b4558889276674`;
- frozen backbone canonical SHA256:
  `2f1bb0820efd7376103541cd6de7b2893456083359708ac78f3fd30f8d1d56ac`;
- tied LM-head/input-embedding SHA256:
  `a473cbd256224b82e4ce004797629aee049c99db483bb5dd05fe7dcc67bdddeb`;
- tokenizer bytes and tokenizers version from the frozen causal-LM identity/grammar gate;
- population:
  the frozen 462-row AVeriTeC-compatible cohort;
- early causal site:
  block 35;
- selected plane:
  `P3`;
- response-blind control plane:
  `P5`;
- late diagnostic/readout site:
  post-block 47 followed by the frozen terminal norm and tied LM head;
- no P3 projection at block 47;
- no layer scan;
- no P3 reselection;
- no training;
- no newly trained hallucination classifier.

## 4. Population and partition

Use the exact same frozen 462-row AVeriTeC-compatible population.

Reuse the already-frozen deterministic response-blind partition:

- calibration: 231;
- confirmatory: 231;
- partition SHA256:
  `871fb5c1e2c62f247c284ceae409ef9ec19acc75f63ffc832829c2664db46311`.

No repartitioning is allowed.

The prior all-abstain outcome must not be used to select items, strata, thresholds, surfaces, layers, offsets, or coordinates.

## 5. Forced-decisive grammar

The new grammar contains exactly two alternatives, both already frozen by the prior tokenizer gate:

- `REFUTE`:
  `Based on the evidence, the verdict is REFUTE.`
- `SUPPORT`:
  `Based on the evidence, the verdict is SUPPORT.`

Exact frozen token sequences:

- `REFUTE`:
  `[15545, 327, 253, 1941, 13, 253, 11844, 310, 5689, 39, 23638, 15]`
- `SUPPORT`:
  `[15545, 327, 253, 1941, 13, 253, 11844, 310, 9242, 27425, 15]`

The two alternatives share exactly the same first 8 generated tokens.

The decisive branch token index is therefore prospectively fixed as:

`t* = 8`

using zero-based generated-token indexing.

`NOT_ENTITLED` is not an available output in this new protocol.

This omission is part of the new forced-decisive estimand and must not be described as a correction to the prior protocol.

## 6. Decoding rule

Use deterministic manual full-prefix finite-grammar greedy decoding.

At each generated step:

1. construct the exact prompt plus all emitted generated-prefix tokens;
2. run a fresh full-prefix causal-LM forward with `use_cache=False`;
3. restrict candidate tokens to those allowed by the frozen two-class grammar trie;
4. choose the allowed token with the highest raw next-token logit;
5. on an exact logit tie, choose the lower token ID.

Forbidden:

- sampling;
- temperature;
- top-k/top-p;
- beam search;
- repetition penalty;
- logit bias;
- thresholding;
- post-hoc surface changes;
- generation-response-dependent decoding changes.

The Hugging Face `.generate()` API is not used.

## 7. Forced-decisive outcome definition

Every generated sequence must terminate as exactly one of:

- `REFUTE`;
- `SUPPORT`.

An emitted forced-decisive commitment is **supported** when:

- gold = `REFUTE` and emitted = `REFUTE`; or
- gold = `SUPPORT` and emitted = `SUPPORT`.

An emitted forced-decisive commitment is **unsupported** when:

- gold = `NOT_ENTITLED` and emitted is either `REFUTE` or `SUPPORT`; or
- gold = `REFUTE` and emitted = `SUPPORT`; or
- gold = `SUPPORT` and emitted = `REFUTE`.

Thus every row belongs to exactly one forced-decisive outcome class unless execution fails closed.

Gold `NOT_ENTITLED` rows are retained. Under a forced-decisive grammar, either available decisive commitment is unsupported for those rows by the already-frozen evidence-grounded event definition.

## 8. Pre-emission observation window

Use the same relative offsets as the closed design:

- `t*-4`
- `t*-3`
- `t*-2`
- `t*-1`

Because `t*=8`, the corresponding generated-prefix lengths before the next-token forward are:

- 5;
- 6;
- 7;
- 8.

Every feature at offset `t` may use only:

- the fixed claim/evidence prompt;
- emitted generated tokens through that prefix.

No future emitted token may enter the feature calculation.

## 9. Early P3 measurement

At block 35, reuse the exact frozen intervention-space representation and strong-channel mask.

For the last prefix token at each preregistered offset, record:

- `P3_A`;
- `P3_B`;
- `P3_COMPONENT_L2`.

Primary Stage A signal:

`P3_COMPONENT_L2`

Also record the response-blind P5 analogues as diagnostics/control only.

P5 does not enter the primary inferential family.

## 10. Late readout diagnostic

Capture the full post-block-47 hidden sequence.

Replay:

`full post_block_47 sequence -> frozen terminal norm -> last token -> tied LM head`

Do not project P3 onto block 47.

Record the frozen next-token logits for the two branch-start tokens:

- `REFUTE`;
- `SUPPORT`.

A simple late diagnostic margin may be stored as:

`M47 = logit(SUPPORT) - logit(REFUTE)`

This is diagnostic and is not part of the primary Stage A inferential family.

The replay must reproduce the model's actual last-token logits within the already-frozen implementation tolerance:

`1e-5`

or execution fails closed.

## 11. Primary Stage A inferential family

Primary partition:

`confirmatory`

Primary comparison:

- unsupported forced-decisive commitments
  versus
- supported forced-decisive commitments.

Primary signal:

`P3_COMPONENT_L2`

Offsets:

- `t*-4`
- `t*-3`
- `t*-2`
- `t*-1`

For each offset, use one two-sided Welch t-test.

Multiplicity:

Holm correction across exactly 4 p-values.

Familywise alpha:

`0.05`

No additional primary p-values are allowed.

The raw GPU generation run must not calculate these p-values. It writes only raw rows and provenance. Statistical inference is performed only after the raw artifact is collected, imported, validated, and frozen.

## 12. Estimability rule

Before any Stage A p-value is calculated, verify on the frozen raw confirmatory artifact that:

- supported forced-decisive count > 0; and
- unsupported forced-decisive count > 0.

If either group is empty:

- execute zero Stage A p-values;
- add zero p-values;
- close as:
  `FORCED_DECISIVE_STAGE_A_NOT_ESTIMABLE`;
- do not modify the decoder, population, split, grammar, layer, plane, or offsets within this experiment.

No rescue is permitted.

## 13. Stage A interpretation

If the primary family is estimable, report effect direction, group means, Welch statistics, raw p-values, and Holm-adjusted p-values for all four preregistered offsets.

The bounded positive Stage A label is allowed only if at least one preregistered offset has Holm-adjusted `p < 0.05`:

`FORCED_DECISIVE_PRE_EMISSION_TEMPORAL_PRECEDENCE_OBSERVED`

If the family is estimable and no offset passes Holm:

`FORCED_DECISIVE_PRE_EMISSION_TEMPORAL_PRECEDENCE_NOT_SUPPORTED`

If the family is not estimable:

`FORCED_DECISIVE_STAGE_A_NOT_ESTIMABLE`

These labels apply only to the forced-decisive two-class regime.

## 14. Claim boundary

Even a positive result does not establish:

- a spontaneous hallucination precursor;
- behavior when abstention is available;
- free-form hallucination prediction;
- causal prevention;
- Stage B prospective prediction;
- Stage C intervention efficacy;
- architecture-independent mechanism;
- itemwise invariance.

The strongest Stage A-only statement would be:

> Under the prospectively frozen forced-decisive two-class finite grammar, the frozen block-35 P3 signal showed pre-emission temporal separation between later unsupported and supported decisive commitments.

No broader wording is licensed.

## 15. Progression rule

Do not run Stage B or Stage C from the original closed experiment.

For this new forced-decisive experiment:

1. freeze this design;
2. implement and statically validate the two-class raw generator;
3. freeze implementation;
4. execute one pinned raw GPU run;
5. collect/import/validate/freeze the raw artifact;
6. only then run the preregistered static Stage A family if estimable;
7. freeze and interpret Stage A before any decision about a separate Stage B.

No generation/evaluation is authorized by this design document alone.
