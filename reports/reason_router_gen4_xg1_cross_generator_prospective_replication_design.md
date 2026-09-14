# Gen4 × K XG1 Cross-Generator Prospective Replication Design

## Status

**PROSPECTIVE DESIGN — OUTCOME BLIND**

This study begins only after the completed and frozen same-generator-family prospective validation:

- result freeze commit: `a4d661a04b8547f4955c1ecf28e21698c53ec74e`
- frozen conclusion: `PROSPECTIVE_ADVERSE_REGIME_SUPPORTED`
- frozen threshold: `T = 0.11228626366380845`

The completed result remains limited to an independent synthetic holdout from the same `build_controlled_v5` generator family.

This XG1 study asks whether that pre-specified conditional causal effect generalizes to a second, independently constructed synthetic generator family.

No XG1 model geometry, intervention response, `R_ALIGN`, or inferential result has been observed when this design is frozen.

---

## Scientific question

Does the already-frozen **large pre-intervention directional-correction-demand regime** prospectively identify an adverse alignment-intervention response in an independently constructed synthetic generator family?

The exact target claim is:

> In XG1, pairs with `alignment_shift_abs >= 0.11228626366380845` have negative mean `R_ALIGN` and more adverse mean `R_ALIGN` than pairs below that threshold, under the already-frozen Gen4 × K directional-alignment intervention.

This is a cross-generator replication of the conditional regime effect, not a re-estimation exercise.

---

## What is frozen from the completed study

The following are carried forward unchanged:

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
- reference geometry pair:
  `C5_TITLE_NAME - C1_TITLE`
- layer roles:
  `15 → 16 → 17`
- intervention coordinate:
  own-branch `A_IDENTITY + 2`
- target-branch constraint:
  `A_IDENTITY == A_NAME`
- intervention site:
  layer-17 x branch, post-`in_proj`, pre-depthwise-conv
- gate branch:
  untouched
- strong channel set:
  fixed 395 channels
- endpoint:
  layer-17 `POST4_PATH_EFFICIENCY`
- representative checkpoint:
  seed180 `G3-GROUP-D-HALF`
- checkpoint SHA256:
  `1ff3fcf2ebd754ab6f9483d6a9982b9b04b9a4eb3357f9f8cdbe2b30399e7d2f`
- tokenizer revision:
  `40e5d2bd7452abb3ca8fadbafe9131ee0e2c2f37`
- fast-CUDA runtime/kernel identities:
  unchanged from the validated prospective backend
- family alpha:
  `0.05`
- minimum group size:
  `30`

No threshold, layer, endpoint, token offset, channel, checkpoint, or intervention scan is allowed.

---

## XG1 generator-family independence contract

XG1 must be implemented as a new deterministic source generator.

The XG1 production generator must **not import or call**:

- `scripts.build_controlled_v5.FACT_TEMPLATES`
- `scripts.build_controlled_v5.fact_templates_for_count`
- `scripts.build_controlled_v5._generated_fact_template`
- `scripts.build_controlled_v5._statement`
- `scripts.build_controlled_v5._paraphrase`

It must not reuse the original generator's lexical inventories as a source of XG1 values.

The experimental six-cell mask semantics may be reused because those masks define the causal manipulation, not the source generator family.

The intended separation is:

- frozen causal design: shared;
- source entity/value inventory: independent;
- source-pair construction rule: independent;
- sentence renderer: independent;
- rendered strings: independent.

---

## XG1 structured source schema

Each source pair retains the abstract fields needed by the already-frozen six-cell manipulation:

- `pair_id`
- `title`
- `name`
- `role`
- `predicate`
- `object`
- `time`
- `location`
- `alternate_title`
- `alternate_name`
- `alternate_role`
- `alternate_predicate`

Object/time/location alternates are not needed by the six-cell Gen4 manipulation and must not be introduced into the inferential family.

For every pair:

- original title != alternate title
- original name != alternate name
- original role != alternate role
- original predicate != alternate predicate

---

## XG1 source population

Exactly 300 source pairs are frozen:

`xg1_fact_001` through `xg1_fact_300`

The source generator must be deterministic and index-addressable.

The implementation must use new hard-coded lexical inventories and a new deterministic index schedule rather than sampling from the original generator.

At minimum the XG1 inventories must contain:

- 12 or more person names;
- 6 or more titles;
- 6 or more roles;
- 7 or more transitive predicate pairs;
- 12 or more objects;
- 12 or more locations;
- 12 or more time expressions.

All XG1 inventory values must pass a static exact-string non-overlap audit against the source values used by the frozen original generator population covering discovery and prior prospective validation.

Case-folded exact matches also count as overlap and are forbidden.

This restriction is intentionally stronger than merely requiring new pair IDs.

---

## XG1 renderer

The exact XG1 positive-statement renderer is frozen as:

`During {time}, records from {location} identify {title} {name} as {role}; this person {predicate} {object}.`

All six cells for a source pair use this renderer with only the cell-authorized slot substitutions.

This renderer deliberately changes the surface ordering relative to the original generator family while keeping the same semantic slots identifiable.

No alternate renderer is tested in this study.

No paraphrase ensemble is used.

---

## Six-cell materialization

For each XG1 source pair, materialize exactly:

- `C0_SHAM`
- `C1_TITLE`
- `C2_NAME`
- `C3_ROLE`
- `C4_PREDICATE`
- `C5_TITLE_NAME`

with the same frozen axis masks as the completed Gen4 six-cell design.

Required scientific pairs remain:

- target:
  `C2_NAME - C0_SHAM`
- reference:
  `C5_TITLE_NAME - C1_TITLE`

The other two cells remain structural completeness controls and are not added to the inferential family.

Total frozen materialized population:

- 300 source pairs
- 6 rows per pair
- 1800 rows

---

## Structural independence gate

Before tokenizer or model execution, XG1 must pass all of the following:

1. exactly 300 unique source pair IDs;
2. exact IDs `xg1_fact_001` through `xg1_fact_300`;
3. exactly six complete cells per pair;
4. exactly 1800 rows;
5. zero duplicate row IDs;
6. exact frozen cell masks;
7. no labels, logits, predictions, model geometry, endpoint values, or response fields;
8. zero exact rendered claim-string overlap with both:
   - original discovery population;
   - original same-family prospective holdout;
9. zero exact rendered evidence-string overlap with those populations;
10. zero case-folded XG1 source-inventory value overlap with the original generator source-value inventory used by those populations;
11. deterministic byte-identical regeneration from the frozen XG1 generator.

Failure of any condition blocks tokenizer/model execution.

---

## Tokenizer / anchor gate

After structural freeze, use the exact frozen tokenizer revision.

Require the complete fixed XG1 cohort to satisfy the same anchor contract:

- required `A_IDENTITY` events;
- required `A_NAME` events;
- target branches satisfy `A_IDENTITY == A_NAME`;
- all required rows satisfy POST4 eligibility.

No pair may be dropped because of tokenizer geometry, baseline geometry, or intervention response.

If all 300 pairs do not satisfy the fixed anchor contract:

`XG1_REPLICATION = BLOCKED_TOKENIZER_ANCHOR_INELIGIBILITY`

The study stops before any model forward.

A renderer redesign is allowed only after this blocked study is explicitly closed and before any XG1 model outcome has been produced.

---

## Bounded backend equivalence gate

Before scientific execution, run one outcome-blind fixed XG1 pair:

`xg1_fact_001`

Compare:

- CPU-slow: 6 forwards
- CUDA-fast: 6 forwards

with the already-frozen prospective equivalence tolerances:

- state atol/rtol: `1e-4`
- geometry atol/rtol: `1e-4`
- path-efficiency / `R_ALIGN` atol: `1e-4`

This bounded equivalence run is not part of the XG1 scientific forward budget and must not be interpreted scientifically.

---

## Pre-intervention regime freeze

The XG1 full scientific run must preserve the same two-phase ordering:

### Phase 1 — baseline only

Run all 300 pairs through:

- C0 baseline
- C1 baseline
- C2 baseline
- C5 baseline

Total:

`300 × 4 = 1200` scientific forwards.

Compute only baseline geometry and baseline endpoint quantities.

For every pair compute:

`alignment_shift_abs = abs(reference_C - target_C)`

Then classify with the already-frozen threshold:

- LARGE iff `alignment_shift_abs >= 0.11228626366380845`
- SMALL otherwise

Persist the complete 300-pair regime manifest before any alignment response exists.

The regime-freeze artifact must record:

- threshold;
- threshold source = original frozen discovery q75;
- `n_LARGE`;
- `n_SMALL`;
- baseline forward count = 1200;
- alignment forward count at freeze = 0;
- response fields observed at freeze = false.

---

## Minimum group-size stop condition

After the XG1 regime freeze and before any intervention response:

- require `n_LARGE >= 30`
- require `n_SMALL >= 30`

Otherwise:

`XG1_PROSPECTIVE_REGIME_TEST = BLOCKED_INSUFFICIENT_FIXED_GROUP_SIZE`

Do not alter the threshold or pair population to repair group size.

The blocked result is itself informative about cross-generator transport of the original discriminator.

---

## Alignment-intervention phase

Only after group-size PASS:

- alignment C0
- alignment C2

for all 300 pairs.

Total:

`300 × 2 = 600` additional scientific forwards.

Full successful scientific budget:

`1200 + 600 = 1800` GPU forwards.

Magnitude intervention remains outside this study.

No response-dependent inclusion/exclusion is permitted.

---

## XG1 inferential family

Exactly the same two one-sided confirmatory tests are used.

### H1-XG1

`mean(R_ALIGN | LARGE) < 0`

Test:

one-sample Student t-test, one-sided less-than-zero.

### H2-XG1

`mean(R_ALIGN | LARGE) < mean(R_ALIGN | SMALL)`

Test:

independent two-sample Welch t-test, one-sided LARGE < SMALL.

Multiplicity:

- Holm correction over exactly these two tests;
- family alpha = `0.05`.

No additional p-values, subgroup tests, threshold scans, renderer comparisons, or alternative endpoint tests are allowed.

---

## Replication decision rule

`CROSS_GENERATOR_ADVERSE_REGIME_REPLICATED`

requires all of:

1. structural independence gate PASS;
2. tokenizer/anchor gate PASS;
3. bounded CPU-slow ↔ CUDA-fast equivalence gate PASS;
4. provenance/runtime/manipulation gates PASS;
5. fixed group-size gates PASS;
6. H1-XG1 direction condition PASS;
7. H1-XG1 Holm rejection;
8. H2-XG1 direction condition PASS;
9. H2-XG1 Holm rejection.

If either confirmatory hypothesis fails:

`CROSS_GENERATOR_ADVERSE_REGIME_NOT_ESTABLISHED`

No rescue analysis changes this decision.

---

## Descriptive quantities

The study may report without adding tests:

- `n_LARGE`, `n_SMALL`;
- LARGE/SMALL/overall mean, median, SD of `R_ALIGN`;
- negative/positive/zero response fractions;
- fixed-threshold baseline geometry summaries.

These are descriptive only.

---

## Falsification / interpretation

A negative XG1 result is scientifically meaningful.

The current same-family result would remain valid, but the evidence would favor generator-family dependence of the adverse regime.

A positive XG1 result would support:

> The pre-specified large directional-correction-demand regime predicts an adverse causal response to the frozen alignment intervention across two independently constructed synthetic generator families sharing the same abstract slot manipulation.

Even a positive XG1 result would **not** establish:

- behavioral mediation;
- hallucination causation;
- a state-of-the-art modeling result;
- general Mamba instability;
- real-world external validity;
- validity outside the matched slot-manipulation setting.

A positive XG1 result authorizes the next scientific stage:

**real-world external bridge / mechanism-transfer study**.

---

## Immediate phase boundary

The immediate phase authorizes only:

1. implement the deterministic XG1 source generator and renderer;
2. materialize the exact 300-pair / 1800-row six-cell cohort;
3. run structural-independence validation;
4. freeze generator + materialized cohort + structural manifest.

No tokenizer execution, model execution, CUDA execution, intervention, or inferential testing is authorized in this immediate phase.
