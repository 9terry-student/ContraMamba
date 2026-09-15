# Gen4 × K Generator-Family Prevalence Transportability Prospective Design Candidate

## Status

**PROSPECTIVE DESIGN CANDIDATE — NEW OUTCOMES NOT YET OBSERVED**

This stage begins only after XG1 is closed and frozen at:

- closure commit: `61d9efb0f1aa7afc8d00ab1453b6ff42e766c402`
- XG1 result freeze commit: `189f3e6a060859b05d78898b6129844111378aa2`

The completed same-generator-family prospective result remains frozen and unchanged.

This study does not rescue XG1 and does not test intervention response.

No new generator implementation, tokenizer execution, model execution, CUDA execution, or statistical analysis is authorized by this candidate until the design itself is reviewed and frozen.

## Scientific question

The frozen adverse-response discriminator is:

`alignment_shift_abs = abs(reference_C - target_C)`

with frozen threshold:

`T = 0.11228626366380845`

The new question is:

> Across multiple independently constructed synthetic generator families, how frequently does the already-frozen discriminator produce enough LARGE-regime pairs to support the previously fixed 300-pair confirmatory replication design?

This is a baseline-only prevalence-transportability study.

It does not ask whether `R_ALIGN` is adverse in any new family.

## Motivation

The frozen same-generator-family prospective holdout produced:

- `n_LARGE = 81`
- `n_SMALL = 219`
- 300 pairs total

The frozen XG1 population produced:

- `n_LARGE = 19`
- `n_SMALL = 281`
- 300 pairs total

XG1 therefore failed the pre-specified minimum group-size gate of 30 LARGE and 30 SMALL before any intervention response was observed.

These historical values motivate the present question but are not outcomes of this new prospective study.

## Frozen scientific quantities

The following are carried forward without modification:

- discriminator:
  `alignment_shift_abs = abs(reference_C - target_C)`
- threshold:
  `T = 0.11228626366380845`
- LARGE iff:
  `alignment_shift_abs >= T`
- SMALL iff:
  `alignment_shift_abs < T`
- target pair:
  `C2_NAME - C0_SHAM`
- reference pair:
  `C5_TITLE_NAME - C1_TITLE`
- layer roles:
  `15 → 16 → 17`
- representative checkpoint:
  seed180 `G3-GROUP-D-HALF`
- checkpoint SHA256:
  `1ff3fcf2ebd754ab6f9483d6a9982b9b04b9a4eb3357f9f8cdbe2b30399e7d2f`
- tokenizer revision:
  `40e5d2bd7452abb3ca8fadbafe9131ee0e2c2f37`
- minimum viable group size for a 300-pair response-replication design:
  `30`

No threshold, layer, endpoint, checkpoint, token offset, or discriminator re-estimation is allowed.

## New generator families

Construct exactly three new independent deterministic generator families:

- `XG2`
- `XG3`
- `XG4`

Each family contains exactly 300 source pairs and exactly six materialized cells per pair.

Family IDs:

- `xg2_fact_001` through `xg2_fact_300`
- `xg3_fact_001` through `xg3_fact_300`
- `xg4_fact_001` through `xg4_fact_300`

Each family must have its own:

- hard-coded lexical inventories;
- deterministic index schedule;
- sentence renderer;
- source-value inventory;
- rendered strings.

No new family may import or call the original `build_controlled_v5` source-generation functions or the XG1 source generator.

No new family may reuse source inventory values from:

- the original discovery population;
- the original same-family prospective population;
- XG1;
- either of the other two new families.

Case-folded exact matches count as reuse and are forbidden.

The abstract six-cell causal slot semantics may remain shared.

## Structural independence gate

Before tokenizer or model execution, each of XG2, XG3, and XG4 must independently pass:

1. exactly 300 unique source-pair IDs;
2. exactly six complete cells per pair;
3. exactly 1800 rows;
4. zero duplicate row IDs;
5. exact frozen six-cell masks;
6. no labels, logits, model geometry, response, or intervention fields;
7. deterministic byte-identical regeneration;
8. zero exact rendered claim-string overlap with all earlier frozen populations;
9. zero exact rendered evidence-string overlap with all earlier frozen populations;
10. zero case-folded source-inventory value overlap with all earlier frozen populations;
11. zero cross-family source-inventory overlap among XG2, XG3, and XG4;
12. zero cross-family rendered-string overlap among XG2, XG3, and XG4.

Failure closes that family before tokenizer/model execution.

No replacement pairs may be added after seeing tokenizer or model geometry.

## Tokenizer / anchor eligibility gate

For each family, the complete fixed 300-pair cohort must pass the already-frozen active-token and anchor contract.

Required properties include:

- required `A_IDENTITY` events;
- required `A_NAME` events;
- target branches satisfy `A_IDENTITY == A_NAME`;
- all required rows satisfy POST4 eligibility.

No pair may be dropped.

If any of 300 pairs in a family fails, that family is recorded as:

`BLOCKED_TOKENIZER_ANCHOR_INELIGIBILITY`

and no model forward is run for that family.

Renderer repair after tokenizer observation is not allowed inside this study.

## Backend gate

Use the already-validated frozen CUDA backend identities and tolerances.

Before full baseline execution of each new family, run one bounded outcome-blind equivalence pair:

- XG2: `xg2_fact_001`
- XG3: `xg3_fact_001`
- XG4: `xg4_fact_001`

Compare CPU-slow and CUDA-fast baseline geometry under the existing frozen equivalence tolerances.

These bounded checks are not scientific outcomes.

Failure blocks the corresponding family.

## Scientific execution

For every structurally and tokenizer-eligible family, execute baseline cells only:

- C0 baseline
- C1 baseline
- C2 baseline
- C5 baseline

For 300 pairs:

`300 × 4 = 1200` scientific model forwards per family.

Maximum baseline scientific budget across three families:

`3600` model forwards.

Do not execute:

- alignment intervention;
- magnitude intervention;
- `R_ALIGN`;
- response endpoints;
- task heads;
- logits;
- training;
- backward.

For every pair compute only the already-frozen baseline discriminator:

`alignment_shift_abs = abs(reference_C - target_C)`

Then classify with the fixed threshold.

## Primary family-level endpoint

For each new generator family `g`, define:

- `n_LARGE_g`
- `n_SMALL_g`
- `p_LARGE_g = n_LARGE_g / 300`

Define the already-frozen design-viability condition:

`VIABLE_g = (n_LARGE_g >= 30) and (n_SMALL_g >= 30)`

This criterion is not newly tuned. It is inherited directly from the XG1 prospective replication design's fixed minimum group-size gate.

## Primary prospective decision rule

After XG2, XG3, and XG4 are all frozen:

### `PREVALENCE_TRANSPORTABILITY_ROBUST`

if:

- XG2, XG3, and XG4 all pass structural/tokenizer/backend/provenance gates; and
- all three satisfy `VIABLE_g = true`.

### `PREVALENCE_TRANSPORTABILITY_MIXED`

if:

- all three families pass structural/tokenizer/backend/provenance gates; and
- exactly one or two families satisfy `VIABLE_g = true`.

### `PREVALENCE_TRANSPORTABILITY_SYSTEMATICALLY_LOW`

if:

- all three families pass structural/tokenizer/backend/provenance gates; and
- none satisfies `VIABLE_g = true`.

If any family is blocked before baseline geometry is validly produced, the cross-family decision is:

`PREVALENCE_TRANSPORTABILITY_INCOMPLETE`

The blocked family must not be repaired or replaced inside this study.

## Descriptive quantities

For each new family, report without adding hypothesis tests:

- `n_LARGE`;
- `n_SMALL`;
- `p_LARGE`;
- exact 95% binomial confidence interval for `p_LARGE`;
- mean, median, SD of `alignment_shift_abs`;
- fixed quantiles:
  10%, 25%, 50%, 75%, 90%;
- minimum and maximum `alignment_shift_abs`;
- count and fraction above the frozen threshold.

Also report the frozen historical same-family and XG1 prevalence values as labeled historical references only.

No p-value is required for the primary study.

## Why there is no intervention-response test

This study deliberately stops at the pre-intervention discriminator.

Its purpose is to determine whether the fixed 300-pair response-replication design is broadly viable across independent generator families.

Running intervention responses here would confound prevalence mapping with response-family selection and would turn this stage into a post-XG1 rescue.

Therefore no XG2/XG3/XG4 `R_ALIGN` value may be observed in this study.

## Interpretation boundaries

`PREVALENCE_TRANSPORTABILITY_ROBUST` would mean:

> The frozen discriminator produces prospectively adequate LARGE and SMALL group sizes in all three newly constructed independent generator families under the inherited 300-pair design.

It would not establish an adverse intervention response in those families.

`PREVALENCE_TRANSPORTABILITY_MIXED` would mean:

> Design viability under the frozen discriminator depends on generator family.

`PREVALENCE_TRANSPORTABILITY_SYSTEMATICALLY_LOW` would mean:

> Across all three new generator families, the frozen discriminator fails to produce enough LARGE cases for the inherited 300-pair confirmatory response design.

It would not prove that the discriminator is mechanistically invalid.

`PREVALENCE_TRANSPORTABILITY_INCOMPLETE` is a provenance/eligibility incompleteness state, not a scientific transportability conclusion.

## Historical evidence boundary

The previously observed same-family and XG1 values are frozen context.

They may be displayed next to XG2-XG4 results, but:

- they are not reclassified;
- their thresholds are not recomputed;
- their populations are not altered;
- no new historical p-values are introduced;
- no combined threshold is fit;
- no pooled discriminator is trained.

## No rescue rule

This study forbids:

- threshold scans;
- threshold re-estimation;
- changing the 300-pair family size after baseline geometry is observed;
- pair replacement after tokenizer/model observation;
- selective enrichment for LARGE-like examples;
- choosing renderer variants based on model geometry;
- merging generator families to satisfy `n_LARGE >= 30`;
- intervention execution on a family because its prevalence looked favorable.

## Future-stage boundary

This study does not itself authorize a new response-replication experiment.

If prevalence transportability is robust or mixed, any later causal-response study must be separately designed and frozen before intervention response is observed.

To avoid selecting a response cohort because of favorable observed baseline prevalence, a later causal-response replication should use a fresh generator family not contained in XG2-XG4, unless a separate authority explicitly justifies another design.

A real-world external bridge remains unauthorized unless separately justified by later evidence.

## Immediate phase boundary

The immediate phase authorized by a future freeze of this design is limited to:

1. implement XG2, XG3, and XG4 deterministic source generators and renderers;
2. materialize the exact three 300-pair / 1800-row six-cell cohorts;
3. run static structural-independence validation only;
4. freeze generators, materialized cohorts, and structural manifests.

No tokenizer execution, model execution, CUDA execution, or baseline geometry observation is authorized in that immediate phase.
