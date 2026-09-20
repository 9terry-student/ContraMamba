# ContraMamba Gen4 — Precursor v2 Dynamic Causal Susceptibility Stage A Design

## Status

`PROSPECTIVE_PRECURSOR_V2_DYNAMIC_CAUSAL_SUSCEPTIBILITY_DESIGN`

This document freezes a new prospective Stage A experiment before any Precursor v2
generation response is produced or inspected.

It is downstream of two completed static failure-anatomy analyses:

- Experiment 3 steering failure anatomy, frozen at
  `180d03a0929d003b0934eb4a63e630eb8e349525`;
- signed-P3 precursor failure anatomy, frozen at
  `e2a51cc5dabd6ac6d97cae8ee030ac17358799e4`.

Those anatomy results are hypothesis-generating context only.

They are **not** used to select a coordinate, temporal offset, layer, sign, subgroup,
threshold, or test statistic in this design.

The prior forced-decisive precursor result remains frozen:

`FORCED_DECISIVE_PRE_EMISSION_TEMPORAL_PRECEDENCE_NOT_SUPPORTED`

Precursor v2 is a new prospective experiment, not a rescue or reinterpretation of that
null.

This design authorizes no scientific execution by itself.

---

## 1. Scientific question

Primary Stage A question:

> Before a future forced-decisive supported versus unsupported commitment is emitted,
> does the local causal susceptibility of the frozen block-35 P3 plane, relative to the
> frozen response-blind P5 plane, differ on prefix-only states?

The intended object is not static radial occupancy.

It is a local differential response:

`current prefix state`
→ `small frozen-plane perturbation at block 35`
→ `change in a frozen downstream decisive readout at block 47`.

The experiment therefore asks whether a **dynamic causal susceptibility** precedes a
future wrong decisive commitment.

It does not ask whether one signed P3 coordinate or one temporal offset is best.

---

## 2. Frozen prior null and failure-anatomy boundary

The original forced-decisive Stage A tested the radial observable

`P3_COMPONENT_L2 = sqrt(a^2 + b^2)`

at four offsets:

- `t* - 4`
- `t* - 3`
- `t* - 2`
- `t* - 1`

and did not support temporal precedence.

The later static anatomy reported the already stored signed coordinates and their
increments without new p-values.

That anatomy must not be converted into a feature-selection step.

Therefore Precursor v2 prohibits:

- selecting `a`;
- selecting `b`;
- selecting `theta`;
- selecting `delta_a`;
- selecting `delta_b`;
- selecting `delta_theta`;
- selecting a single best offset;
- selecting an offset subset;
- choosing a sign from the old response;
- fitting a linear combination to the old precursor data.

The new observable is defined independently of those descriptive patterns.

---

## 3. Model identity

Primary model only:

- model family: Mamba;
- scale: 370M;
- Hugging Face repository:
  `state-spaces/mamba-370m-hf`;
- pinned revision:
  `589179554943157be31701edd8b4558889276674`;
- frozen Mamba backbone canonical SHA256:
  `2f1bb0820efd7376103541cd6de7b2893456083359708ac78f3fd30f8d1d56ac`;
- frozen tied LM-head/input-embedding SHA256:
  `a473cbd256224b82e4ce004797629aee049c99db483bb5dd05fe7dcc67bdddeb`.

Generation and readout reconstruction must follow the already frozen forced-decisive
causal-LM protocol.

No training, fine-tuning, weight update, learned probe, or new classifier is allowed.

---

## 4. Frozen causal geometry

Use the already frozen Mamba-370M geometry bundle:

`reports/reason_router_gen4_mamba370m_geometry_preparation_runs/g4k-mamba370-geometry-xg2xg4-2gpu-d8e71ad-retry2`

Geometry execution HEAD:

`d8e71addfea4354a947393abd4b9e76ab769f3f3`

Intervention site:

`block 35`

Strong-channel dimension:

`650`

Strong-index SHA256:

`6e2fb851713a15d29756e2a2b8220d97a521bc9648819bffa1955e1c1ad44776`

Selected causal plane:

`P3`

Response-blind control plane:

`P5`

P3 frozen basis identities:

- plus:
  `5d404f8f402639ab101413a0237fa458246a155855507600256fa3dd250adb7f`;
- minus:
  `4a48563f5e419916a32f4b3a4106bea1056abafd898ef12496ea5bd079bbd0fb`.

P5 frozen basis identities:

- plus:
  `1be1bb5eaee1cdf583c0c4a249ac6dd624e1dd0194bd81c08d34e50cce7a8ed3`;
- minus:
  `7af4a78366e9258d2833bced5a5c824990d7c062a970d3c44b3adc386db3a0dc`.

No plane reconstruction, reselection, rotation, Procrustes fit, coordinate fit, or
response-dependent reorientation is permitted.

---

## 5. Population

### 5.1 Source

Use the pinned AVeriTeC train source already frozen in the repository workflow:

- upstream repository:
  `MichSchli/AVeriTeC`;
- upstream commit:
  `7c62d1ec8df3fb560d6efe2b85fa191135636f81`;
- train path:
  `data/train.json`;
- train git blob SHA1:
  `0f190e115cf2ee23416e8a539c8d6ac043d7cc83`;
- train SHA256:
  `ae5eda7c42ddf1695ef185a7ba1bc716928f5adf57103e4f78aae5f9afe00f9c`.

Start from the already frozen response-free, dev-deduplicated, token-gated train-derived
cohort:

`data/reason_router_gen4_averitec_train_fresh_steering_boundary_v1/fresh_compatible_cohort.jsonl`

Its frozen population is:

- total: `2799`;
- Refuted: `1727`;
- Supported: `805`;
- Not Enough Evidence: `267`.

### 5.2 Precursor-v2 eligibility

Precursor v2 Stage A includes only decisive-gold rows:

- source label `Refuted`;
- source label `Supported`.

`Not Enough Evidence` is excluded prospectively.

Reason:

The forced-decisive grammar has exactly two output classes. Restricting to decisive
gold labels makes the future outcome definition exactly:

- supported = emitted decisive label equals gold;
- unsupported = emitted decisive label is the opposite decisive label.

This prevents `NOT_ENTITLED` from being structurally forced into the unsupported group.

### 5.3 Prospective fixed cohort size

Stage A target:

`N = 800`

with exact gold-label balance:

- `400` Refuted;
- `400` Supported.

Selection is performed separately within the two gold labels.

For each eligible row define:

`rank_key = SHA256("CONTRAMAMBA_PRECURSOR_V2_DCS_STAGE_A_V1|" + example_id)`

using the exact UTF-8 bytes of the literal namespace above followed by the repository
`example_id`.

Within each gold label:

1. sort ascending by `rank_key`;
2. break an exact hash tie by ascending `example_id`;
3. retain the first 400.

No model output, steering output, old precursor value, native margin, confidence,
truncation severity, generation response, P3/P5 value, or future Stage A outcome may
enter selection.

The resulting 800-row list must be materialized and frozen before scientific
generation.

### 5.4 Freshness statement

The selected rows are:

`FRESH_FOR_PRECURSOR_V2_GENERATION_RESPONSE`

They are not claimed to be globally unseen by every earlier ContraMamba experiment:
the train-derived source population was used by Experiment 3.

However:

- these rows are disjoint from the prior dev-based precursor cohort;
- no Precursor v2 generation outcome exists at design freeze;
- row selection is independent of all Experiment 3 response fields;
- Experiment 3 response fields are prohibited inputs to the Stage A builder,
  generation runner, and analyzer.

No outcome-dependent exclusion is allowed after materialization.

---

## 6. Forced-decisive grammar

Reuse exactly the already frozen two-way grammar.

REFUTE surface:

`Based on the evidence, the verdict is REFUTE.`

Frozen token sequence:

`[15545, 327, 253, 1941, 13, 253, 11844, 310, 5689, 39, 23638, 15]`

SUPPORT surface:

`Based on the evidence, the verdict is SUPPORT.`

Frozen token sequence:

`[15545, 327, 253, 1941, 13, 253, 11844, 310, 9242, 27425, 15]`

The two alternatives share their first 8 generated tokens.

The decisive branch token remains:

`t* = 8`

under zero-based generated-token indexing.

Every eligible generation must terminate as exactly REFUTE or SUPPORT.

---

## 7. Native decoding

Reuse deterministic manual full-prefix finite-grammar greedy decoding.

At every generated step:

1. construct the exact prompt plus already emitted generated-prefix tokens;
2. run a fresh full-prefix causal-LM forward;
3. use `use_cache=False`;
4. restrict candidates to the frozen grammar;
5. select the allowed token with highest raw next-token logit;
6. on an exact tie, select the lower token ID.

Forbidden:

- sampling;
- temperature;
- top-k;
- top-p;
- beam search;
- logit bias;
- response-dependent decoding changes.

Scientific outcome labels are determined from the **unperturbed native generation
only**.

Susceptibility probe forwards must never feed their perturbed states or tokens back into
the native generation path.

---

## 8. Prefix-only temporal window

Use exactly the same four pre-emission offsets:

- `t* - 4`;
- `t* - 3`;
- `t* - 2`;
- `t* - 1`.

No other offset is tested.

All four offsets are mandatory for every scientific row.

At each offset, the forward input may contain only:

- the frozen prompt;
- tokens natively generated up to that prefix time.

No decisive token or later future token may enter the susceptibility forward.

If any row cannot provide all four exact prefix states under the fixed grammar, the run
fails closed rather than dropping the row.

---

## 9. Frozen downstream scalar F

At every prefix time `t`, define a block-47-native decisive readout.

Capture:

`post_block_47`

for the current prefix.

Apply only the frozen terminal normalization and tied LM head already used by the
forced-decisive precursor machinery.

Because the v2 population contains only decisive gold labels, define the gold-aligned
decisive logit margin:

For gold SUPPORT:

`F_t = logit(SUPPORT_branch_token) - logit(REFUTE_branch_token)`

For gold REFUTE:

`F_t = logit(REFUTE_branch_token) - logit(SUPPORT_branch_token)`

`F_t` is a frozen readout diagnostic.

P3 is **not** projected onto block 47.

No block-47 direction is fit.

---

## 10. Dynamic causal susceptibility

### 10.1 Perturbation scale

Use exactly:

`epsilon = 0.025`

No epsilon sweep is allowed in Stage A.

### 10.2 Basis-direction derivatives

For plane `Pk`, where `Pk` is P3 or P5, let its two frozen orthonormal intervention-space
basis vectors be:

`u_k,+`

and

`u_k,-`.

At prefix time `t`, intervene only on the block-35 strong-channel intervention vector at
the current target token.

For each basis direction `s in {+, -}` define:

`r_k,s,t = [F_t(h_t + epsilon*u_k,s) - F_t(h_t - epsilon*u_k,s)] / (2*epsilon)`

Each perturbed run:

- uses the identical current prefix;
- changes only the frozen strong-channel block-35 intervention coordinate;
- performs no generation step;
- does not update the native generation trajectory.

### 10.3 Plane susceptibility magnitude

Define:

`chi_k,t = sqrt(r_k,+,t^2 + r_k,-,t^2)`

for `k in {P3, P5}`.

This is invariant to sign flips or orthonormal basis rotation within the same frozen
2D plane.

It does not select `a`, `b`, or `theta`.

### 10.4 P3-specific susceptibility contrast

Define:

`D_t = chi_P3,t - chi_P5,t`

The P5 subtraction is the frozen response-blind geometric control.

No additional control plane is tested.

### 10.5 Single per-item Stage A endpoint

For each generated example `i`, define:

`Z_i = (D_i,t*-4 + D_i,t*-3 + D_i,t*-2 + D_i,t*-1) / 4`

This equal-weight mean is frozen prospectively.

No temporal weight is fit.

No max, min, slope, last-offset-only value, area-under-curve variant, or best-offset
variant may replace it after response inspection.

Per-offset values are retained only as descriptive diagnostics.

---

## 11. Future outcome

For the native unperturbed forced-decisive generation:

Supported:

`emitted decisive label == gold decisive label`

Unsupported:

`emitted decisive label != gold decisive label`

There are no abstention outcomes in this Stage A estimand.

No subgroup redefinition after generation is allowed.

---

## 12. Primary inference

The confirmatory Stage A inferential family contains exactly one p-value.

Primary groups:

- future unsupported forced-decisive commitments;
- future supported forced-decisive commitments.

Primary scalar:

`Z_i`

defined above.

Primary test:

two-sided Welch independent-samples t-test:

`H0: E[Z | unsupported] = E[Z | supported]`

`H1: E[Z | unsupported] != E[Z | supported]`

Alpha:

`0.05`

Primary p-value count:

`1`

Multiplicity correction:

`none_single_test_family`

Minimum estimability requirement:

- at least 30 unsupported rows;
- at least 30 supported rows.

If either group has fewer than 30 rows:

`PRECURSOR_V2_STAGE_A_NOT_ESTIMABLE`

No subgroup, threshold, label-specific test, alternative statistic, or enlarged cohort
may rescue that run.

---

## 13. Success rule and allowed claim

Stage A is supported only if:

1. all 800 rows satisfy provenance and prefix-window requirements;
2. both future-outcome groups satisfy the minimum estimability rule;
3. the single primary two-sided Welch test has `p < 0.05`.

If supported, the strongest allowed Stage A wording is:

`DYNAMIC_CAUSAL_SUSCEPTIBILITY_TEMPORAL_PRECEDENCE_OBSERVED`

This means only:

> The prospectively frozen P3-versus-P5 local susceptibility contrast differed before
> future supported versus unsupported forced-decisive commitment on the frozen fresh
> Precursor-v2 generation cohort.

It does **not** establish:

- useful prediction;
- calibrated early warning;
- causal prevention;
- free-form hallucination detection;
- a universal precursor;
- feature-level superiority over the failed radial observable.

If not supported:

`DYNAMIC_CAUSAL_SUSCEPTIBILITY_TEMPORAL_PRECEDENCE_NOT_SUPPORTED`

and Stage A closes without rescue.

---

## 14. Descriptive diagnostics permitted after the primary result

The following may be reported without additional p-values:

- group counts;
- mean, SD, median, quartiles, min, max of `Z`;
- Cohen-style standardized mean difference as a descriptive effect size;
- per-offset mean `D_t` by outcome group;
- per-offset `chi_P3,t`;
- per-offset `chi_P5,t`;
- the four individual central-difference components;
- gold-label-stratified descriptive summaries;
- native decisive margins;
- numerical symmetry and intervention audit residuals.

These are descriptive only.

They cannot alter the Stage A verdict.

---

## 15. Scientific forward accounting

Each of the two frozen grammar surfaces contains 12 generated tokens.

Native unperturbed generation therefore requires exactly:

`12 full-prefix forwards / row`

For each of four precursor offsets:

- P3 plus direction: `+epsilon`, `-epsilon`;
- P3 minus direction: `+epsilon`, `-epsilon`;
- P5 plus direction: `+epsilon`, `-epsilon`;
- P5 minus direction: `+epsilon`, `-epsilon`.

Therefore susceptibility measurement requires:

`16 full-prefix probe forwards / row`

Total scientific forwards per row:

`28`

For `N = 800`:

`22400`

planned scientific full-model forwards.

No backward pass is required.

No JVP/autograd implementation is required for Stage A; central differences are the
frozen operational definition.

---

## 16. Backend equivalence requirement

This is a new comparable causal-response workload.

Before the 22,400-forward scientific execution, implementation must pass the project
runtime rule:

1. exact runtime/kernel provenance gate;
2. bounded CPU-slow versus CUDA-fast equivalence;
3. fixed tolerance:
   `atol = 1e-4`, `rtol = 1e-4`;
4. only then scientific execution.

The equivalence row must come from already used historical precursor data, not from the
new 800-row Stage A cohort.

The bounded equivalence gate may inspect:

- branch token logits;
- `F_t`;
- the eight perturbed values needed for one offset;
- the derived `chi_P3`, `chi_P5`, and `D_t`.

It must not perform scientific inference.

No tolerance relaxation is allowed.

---

## 17. Required static cohort preparation before execution

Before model execution, materialize a response-free Precursor-v2 cohort artifact that
contains exactly the 800 selected rows and records:

- source train index;
- example ID;
- gold label;
- exact claim;
- exact frozen gold-evidence serialization;
- tokenizer/truncation provenance;
- deterministic rank key;
- cohort-selection namespace;
- no generation fields;
- no P3/P5 response fields;
- no future supported/unsupported field;
- no Experiment 3 response field.

The cohort artifact must verify:

- 400 Refuted;
- 400 Supported;
- zero overlap with the old dev-based precursor cohort by normalized claim and exact
  nonempty source URL;
- deterministic regeneration;
- exact tokenizer bytes;
- all rows are valid for the frozen prompt/generation serialization.

Static cohort preparation performs:

`0 model forwards`

and:

`0 p-values`.

---

## 18. Prohibited moves

Before Stage A is frozen, do not:

- inspect Precursor-v2 generation response before design and cohort freeze;
- select a best signed coordinate from the old anatomy;
- select a best old temporal offset;
- scan layers;
- change block 35;
- project P3 directly at block 47;
- replace P3;
- replace P5;
- add P1/P2/P4 controls;
- rotate or fit a plane;
- fit temporal weights;
- change epsilon;
- run an epsilon sweep;
- use autograd/JVP as an alternate primary measurement;
- enlarge N after observing group counts or p-value;
- add `NOT_ENTITLED`;
- use Experiment 3 margins or steering response to select rows;
- create label-specific inferential tests;
- add a one-sided test after seeing the direction;
- convert per-offset descriptives into additional confirmatory tests;
- tune decoding;
- claim rescue of the original radial Stage A.

---

## 19. Relationship to later stages

If Stage A is supported, the next scientific question is prospective prediction.

Stage B must be frozen separately after Stage A closure.

Stage B may ask whether prefix-only dynamic susceptibility predicts a future unsupported
commitment on a new prospective cohort.

Stage C causal prevention is not authorized by this design.

The evidence ladder remains:

`precedes`
→ `predicts`
→ `causally modulates`

No later stage may be inferred from Stage A alone.

---

## 20. Relationship to cross-layer transport/readout research

The proposed cross-layer causal transport/readout-alignment project remains separate.

It is intentionally deferred until after Precursor-v2 Stage A closure.

No cross-layer transport result may be used to change:

- P3/P5;
- block 35;
- block 47;
- temporal offsets;
- susceptibility definition;
- cohort;
- epsilon;
- primary test

in this frozen Stage A.

This separation preserves prospectivity.

---

## 21. Immediate next phase

After this design is frozen, the next authorized phase is:

`STATIC PRECURSOR_V2 COHORT MATERIALIZATION AND IMPLEMENTATION PREPARATION`

That phase may:

- build and freeze the exact 800-row response-free cohort;
- implement the susceptibility runner;
- implement static tests;
- implement the bounded CPU/CUDA equivalence gate.

It may not yet run the 22,400-forward Stage A scientific experiment until the required
implementation and equivalence gates are frozen PASS.
