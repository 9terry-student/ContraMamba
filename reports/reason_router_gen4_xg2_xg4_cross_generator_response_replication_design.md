# Gen4 × K XG2/XG4 Fresh-Holdout Cross-Generator Response Replication Design

## Status

**PROSPECTIVE RESPONSE DESIGN — RESPONSE OUTCOME BLIND**

Scientific parent:

`8b03eb0814c187ff37e037d5356827d302d33ee1`

Frozen prevalence evidence:

`97ab2a7e6182645c6fb234928bf1e1e0aa0e8146`

The completed generator-family prevalence stage observed only baseline geometry.
No XG2 or XG4 alignment-intervention response, `R_ALIGN`, H1/H2 result,
task-head output, logit output, training result, or backward result has been
observed.

This design does not reopen XG1 and does not repair XG3.

---

## Scientific question

Among the generator families that independently passed the frozen prevalence
support criterion in the calibration stage, does the already-frozen LARGE
directional-correction-demand regime predict an adverse causal response in a
**new disjoint prospective cohort from the same family**?

XG2 and XG4 are separate prospective replications.

The causal cohorts used here are not the 300-pair prevalence cohorts.

---

## Why a fresh disjoint cohort is mandatory

The prevalence stage already observed:

- XG2 calibration cohort: `94 LARGE / 206 SMALL`
- XG4 calibration cohort: `56 LARGE / 244 SMALL`

Those observations were used to establish that these families have sufficient
support to motivate a later causal-response study.

Therefore the original prevalence pairs must not subsequently be reused for
the confirmatory `R_ALIGN` tests.

The calibration cohorts are used only to qualify the generator families.

The causal-response study uses completely new pair IDs and rendered examples.

This prevents outcome-blind prevalence-based family qualification from becoming
same-item adaptive causal testing.

---

## Frozen calibration evidence

### XG2 calibration cohort

Pair IDs:

`xg2_fact_001` through `xg2_fact_300`

Frozen artifact root:

`reports/reason_router_gen4_generator_family_prevalence_full_baseline_f18152e_r2/xg2/`

Frozen result:

`n_LARGE = 94`

`n_SMALL = 206`

`VIABLE = true`

`baseline_items.jsonl` SHA256:

`d52df4ca14a406f9a81cceac6b81c2b31183a119130a20cea450c26152960f06`

`prevalence_summary.json` SHA256:

`a84b2a71a52ce9edd67ce6de260bb1b945574369dad289f34f9cd0231564b888`

### XG4 calibration cohort

Pair IDs:

`xg4_fact_001` through `xg4_fact_300`

Frozen artifact root:

`reports/reason_router_gen4_generator_family_prevalence_full_baseline_f18152e_r1/xg4/`

Frozen result:

`n_LARGE = 56`

`n_SMALL = 244`

`VIABLE = true`

`baseline_items.jsonl` SHA256:

`a764787bbb3e00f8511acb2a4251d54069d1f6b2eadc8fa178336eaba412b6ac`

`prevalence_summary.json` SHA256:

`378b7e658d6029cccf65455924ceddd30081b5ac59e6570cce9f3b07026e0fc9`

These calibration artifacts must never be read as causal-response observations.

---

## Fresh prospective causal cohorts

Exactly two new fixed cohorts are defined.

### XG2 causal holdout

`xg2_fact_301` through `xg2_fact_600`

Exactly:

- 300 source pairs
- 6 cells per pair
- 1800 materialized rows

### XG4 causal holdout

`xg4_fact_301` through `xg4_fact_600`

Exactly:

- 300 source pairs
- 6 cells per pair
- 1800 materialized rows

These IDs are disjoint from the calibration cohorts.

No prevalence result from these fresh cohorts may be observed before their
source populations and structural artifacts are frozen.

---

## Generator-extension rule

The XG2 and XG4 generator families themselves are not redesigned.

The already-frozen family definitions remain unchanged:

- lexical inventories;
- renderer;
- deterministic schedule;
- axis masks;
- six-cell manipulation;
- generator-family identity.

The existing generator currently materializes indices 1–300 only.

Implementation may make the minimum mechanical extension required to generate
absolute source indices 301–600.

For fresh pair number `j` in `301..600`, the generator must apply the existing
deterministic schedule using the corresponding absolute zero-based index
`j - 1`.

It must not restart the schedule at zero for the fresh holdout.

Object/index suffixes must likewise use the absolute source index, yielding
fresh identifiers such as the existing family-specific packet numbers 301–600.

No new vocabulary, renderer, schedule, sampling process, or family-specific
tuning is permitted.

The original 1–300 cohort must regenerate byte-identically after this
implementation extension.

---

## Structural holdout gate

Before tokenizer or model execution, each fresh family cohort must satisfy:

- exactly 300 unique source pair IDs;
- exact IDs 301–600 for that family;
- exactly six complete cells per pair;
- exactly 1800 rows;
- exact frozen cell masks;
- zero duplicate row IDs;
- deterministic byte-identical regeneration;
- zero pair-ID overlap with that family's calibration cohort;
- zero exact rendered claim-string overlap with that family's calibration cohort;
- zero exact rendered evidence-string overlap with that family's calibration cohort;
- no labels, logits, predictions, baseline geometry, endpoint values,
  intervention responses, or other model outcomes.

Failure blocks that family before tokenizer/model execution.

No pair replacement is allowed.

---

## Tokenizer / anchor gate

After structural freeze, use the exact frozen tokenizer revision and anchor
contract.

For all 300 fresh pairs require:

- required `A_IDENTITY` events;
- required `A_NAME` events;
- target branches satisfy `A_IDENTITY == A_NAME`;
- required rows satisfy POST4 eligibility.

All 300 fresh pairs must be retained.

No pair may be removed because of tokenization, geometry, expected prevalence,
or later response.

If the complete fixed holdout fails this contract:

`<FAMILY>_REPLICATION = BLOCKED_TOKENIZER_ANCHOR_INELIGIBILITY`

and that family stops before any model forward.

No renderer rescue is permitted inside this frozen study.

---

## Frozen discriminator

The discriminator remains:

`alignment_shift_abs = abs(reference_C - target_C)`

Frozen threshold:

`T = 0.11228626366380845`

Fresh-holdout classification:

`LARGE iff alignment_shift_abs >= T`

`SMALL iff alignment_shift_abs < T`

The threshold must not be re-estimated, optimized, quantile-matched, or tuned.

Calibration prevalence does not determine fresh-holdout regime membership.

Each fresh holdout obtains its own prospective LARGE/SMALL assignments solely
from its own baseline geometry under the unchanged frozen threshold.

---

## Frozen causal mapping

The completed same-family prospective causal design is inherited unchanged.

Target pair:

`C2_NAME - C0_SHAM`

Reference geometry pair:

`C5_TITLE_NAME - C1_TITLE`

Layer roles:

`15 -> 16 -> 17`

Intervention coordinate:

own-branch `A_IDENTITY + 2`

Target-branch constraint:

`A_IDENTITY == A_NAME`

Intervention site:

layer-17 x branch, post-`in_proj`, pre-depthwise-conv

Gate branch:

untouched

Strong channel set:

fixed 395 channels

Endpoint:

layer-17 `POST4_PATH_EFFICIENCY`

Representative checkpoint:

seed180 `G3-GROUP-D-HALF`

Checkpoint SHA256:

`1ff3fcf2ebd754ab6f9483d6a9982b9b04b9a4eb3357f9f8cdbe2b30399e7d2f`

No threshold, layer, endpoint, token-offset, channel, checkpoint, renderer,
family, or intervention scan is allowed.

Magnitude intervention is outside this study.

---

## Two-phase prospective execution

Each family is executed independently.

### Phase 1 — fresh baseline regime freeze

For all 300 fresh pairs run:

- C0 baseline
- C1 baseline
- C2 baseline
- C5 baseline

Budget:

`300 x 4 = 1200 scientific forwards`

Compute:

`alignment_shift_abs = abs(reference_C - target_C)`

Then assign LARGE/SMALL using only the frozen threshold.

Before any intervention response exists, persist a complete regime-freeze
artifact containing:

- all 300 pair IDs;
- `alignment_shift_abs`;
- fixed threshold;
- frozen regime label;
- `n_LARGE`;
- `n_SMALL`;
- baseline forward count = 1200;
- alignment forward count = 0;
- response fields observed = false.

---

## Fresh-holdout minimum-support gate

After Phase 1 and before any alignment intervention:

require:

`n_LARGE >= 30`

and:

`n_SMALL >= 30`

If either fails:

`<FAMILY>_PROSPECTIVE_REGIME_TEST = BLOCKED_INSUFFICIENT_FIXED_GROUP_SIZE`

That family stops after exactly 1200 baseline forwards.

This is not a failed causal replication.

No H1/H2 test is performed.

No `R_ALIGN` is observed.

The threshold must not be changed.

The fresh cohort must not be enlarged, repaired, enriched, or selectively
resampled.

The other independently frozen family may proceed according to its own gate.

---

## Phase 2 — alignment response

Only after that family's fresh-holdout group-size gate passes:

run:

- alignment C0
- alignment C2

for all 300 pairs.

Additional budget:

`300 x 2 = 600 scientific forwards`

Maximum successful scientific budget per family:

`1800 forwards`

Maximum successful XG2 + XG4 scientific budget:

`3600 forwards`

No response-dependent pair inclusion or exclusion is permitted.

---

## Bounded response-backend equivalence gate

The existing generator-family r5 equivalence validates only the baseline
geometry path.

Before full causal execution, the complete causal-response path must be
validated separately for each fresh holdout family.

Fixed gate pairs:

XG2:

`xg2_fact_301`

XG4:

`xg4_fact_301`

Per family compare:

CPU-slow: `6 forwards`

CUDA-fast: `6 forwards`

Total:

`12 bounded forwards`

These forwards are outside the scientific budget and have no scientific
interpretation.

Inherited tolerances remain:

state atol/rtol:

`1e-4`

geometry atol/rtol:

`1e-4`

path-efficiency / `R_ALIGN` atol:

`1e-4`

A full family run is blocked unless its response-equivalence gate passes and
the resulting gate artifact is frozen.

---

## Confirmatory inferential family

XG2 and XG4 are analyzed independently.

For each family exactly two one-sided tests are permitted.

### H1

`mean(R_ALIGN | LARGE) < 0`

Test:

one-sample Student t-test, one-sided less-than-zero.

### H2

`mean(R_ALIGN | LARGE) < mean(R_ALIGN | SMALL)`

Test:

independent two-sample Welch t-test, one-sided LARGE < SMALL.

Multiplicity within each family:

Holm correction over exactly H1 and H2.

Family alpha:

`0.05`

No additional p-values, subgroup tests, threshold scans, renderer comparisons,
family pooling, or alternative endpoint tests are allowed.

---

## Family-level decision rules

A family result:

`<FAMILY>_CROSS_GENERATOR_ADVERSE_REGIME_REPLICATED`

requires:

- structural holdout gate PASS;
- tokenizer/anchor gate PASS;
- bounded response-equivalence gate PASS;
- runtime/provenance/manipulation gates PASS;
- fresh fixed group-size gate PASS;
- H1 direction PASS;
- H1 Holm rejection;
- H2 direction PASS;
- H2 Holm rejection.

If the group-size gate passes and a valid full response run completes but
either confirmatory hypothesis fails:

`<FAMILY>_CROSS_GENERATOR_ADVERSE_REGIME_NOT_ESTABLISHED`

If the fresh group-size gate fails, neither of those response conclusions is
assigned because causal replication was not tested.

---

## Joint selected-family statement

The XG2 and XG4 family-level results remain separate.

A conjunction statement:

`SELECTED_ELIGIBLE_FAMILY_RESPONSE_REPLICATION_SUPPORTED`

requires both XG2 and XG4 to independently achieve their complete
`...REPLICATED` result.

No pooled response sample and no pooled p-value are used.

If only one family replicates, report the two family-level outcomes separately
and do not assign the conjunction statement.

This claim applies only to the two prevalence-qualified generator families and
must not be described as universal generator-family transportability.

---

## XG1 / XG3 boundary

XG1 remains closed.

Its frozen result was a prospective insufficient-group-size block, not a
causal-response replication failure.

No XG1 response rescue is allowed.

XG3 remains structurally/tokenizer-anchor ineligible under its frozen design.

No XG3 model forward or renderer repair is allowed in this study.

---

## Prohibited operations

Threshold re-estimation or sweep: NO

Reuse of XG2/XG4 calibration pairs for causal response: NO

Pair replacement: NO

LARGE enrichment: NO

Fresh-cohort enlargement after baseline: NO

Family merge: NO

XG1 rescue: NO

XG3 rescue: NO

Magnitude intervention: NO

Task-head execution: NO

Logit readout: NO

Training: NO

Backward: NO

---

## Immediate phase boundary

After this design is frozen, the immediate authorized phase is limited to:

1. minimally extend the existing deterministic XG2/XG4 generator indexing so
   that absolute pair indices 301–600 can be produced without changing the
   family definitions;
2. materialize the exact fresh XG2/XG4 300-pair / 1800-row cohorts;
3. prove byte-identical regeneration of the existing 1–300 calibration cohorts;
4. run structural disjointness validation;
5. freeze the implementation, fresh cohorts, and structural validation
   artifacts.

Tokenizer execution, model execution, CUDA execution, response-equivalence
execution, full baseline execution, alignment intervention, `R_ALIGN`, and
inferential testing remain outside this immediate phase until the fresh
structural holdouts are frozen.

No additional generator family is introduced by this design.
