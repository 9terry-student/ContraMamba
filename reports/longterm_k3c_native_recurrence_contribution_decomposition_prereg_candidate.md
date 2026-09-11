# K3C Native Recurrence Contribution Decomposition Preregistration Candidate

**Status:** K3C CONFIRMATORY CAUSAL-MECHANISM SCIENTIFIC DESIGN / PREREGISTRATION CANDIDATE.

**Authority boundary:** once committed, this document authorizes bounded K3C implementation, focused tests, state-blind population materialization, tokenizer-only feasibility validation, and non-scientific synthetic/replay instrumentation validation only. It does **not** authorize K3C scientific recurrent-state execution on the frozen K3C population. A separate K3C scientific execution authority must be frozen only after the implementation passes the exact gates defined here.

K3C is a new prospective experiment.

K3C is not a rescue or reinterpretation of K3.

K3 remains closed with:

`LAYER23_NATIVE_RECURRENCE_COMPONENT_SPECIALIZATION_CONTRADICTED`

K3C does not authorize K4.

## 1. Governing evidence and theory authority

Immediate theory authority:

`a8101ce9baf340f35c0c6b135f7312681805693c`

This commit freezes:

`reports/longterm_k3_successor_write_injection_retained_carry_hypothesis_report_candidate.md`

with SHA256:

`0040d3d3573faad4dfacb0b617b4626b5e1143a9aa946bd4d270fdf80c44523e`

The predecessor K3 contradiction archive is:

`99988e8d2a4c47d6bcaec1bf487aac777076fe2f`

K3 established that the preregistered local-W / net-G coefficient-specialization hypothesis was contradicted.

The post-K3 read-only audit then generated, but did not confirm, the successor hypothesis that:

1. branch-specific direct write increments `W_t` are the primary source of new correction/control trajectory divergence;
2. the full retained contribution `H_t = G_t ⊙ S_(t-1)` transports, preserves, or reshapes divergence after it has entered the recurrent state;
3. equalizing the coefficient `G_t` alone is not equivalent to equalizing the full retained contribution `H_t`.

K3C prospectively tests that successor hypothesis on a new claim-disjoint population.

## 2. K3C scientific question

K3C asks:

**Within the frozen layer-23 Mamba recurrence, is pair-specific trajectory geometry causally sourced primarily by direct write injection `W_t`, while a full history-dependent retained contribution `H_t = G_t ⊙ S_(t-1)` can carry W-seeded divergence through the remainder of the event window?**

This is a contribution-level causal question.

It is not a coefficient-specialization question.

## 3. Claim boundary

A full positive K3C result may support only a claim of the form:

`LAYER23_WRITE_INJECTION_WITH_RETAINED_CARRY_CAUSALLY_SUPPORTED`

meaning that, under the exact frozen K3C contribution interventions on a new prospective population:

- eliminating branch-specific W contributions attenuates the replicated pair-specific geometry more strongly than eliminating branch-specific H contributions; and
- divergence seeded by W at the first branch-divergence token remains pair-specifically expressed when later branch-specific W differences are removed and only history-dependent retained contribution can propagate the seed.

Even full K3C support does not establish:

- causal control of final task decisions;
- causal control of authorization/entitlement;
- global necessity or sufficiency of native state;
- confident-error prediction;
- detector utility;
- natural-corpus generalization;
- external-distribution generalization;
- uniqueness of this mechanism outside the frozen layer/window/intervention.

## 4. Prospectively frozen population

K3C uses a new deterministic claim-disjoint slice from the already-frozen controlled generator.

Generator:

`scripts/build_controlled_v5.py`

Required SHA256:

`4e9798591fbfffb6d15ea9b2f8cf5cd804a9e76713b9f6b13b354fe6ce93aa5c`

Required Git blob:

`baee23a9f71333125f4a8735c2c92d20cab7eb4f`

Global template slice:

`[600:900]`

Required first pair ID:

`generated_fact_601`

Required last pair ID:

`generated_fact_900`

Required item count:

`300`

Required source-row count:

`3900`

Required generated-source canonical SHA256:

`33bff5a0b657d1ceb38ae9c651e1cadfc8308286398cc1b8c4245c47f1c42000`

Required K3C candidate schema:

`k3c-independent-population-v1`

Required stable-ID namespace:

`k3c-v1:`

Required candidate-pool canonical SHA256:

`9603f6b20ba870807c151bb70df4c42b0957ded578729b133a37fee8aa1da83e`

Required correction-source balance:

- `none = 150`
- `polarity_flip = 150`

No item filtering, replacement, weighting, resampling, or post-hoc rematerialization is allowed.

## 5. Candidate recipe

For each generated pair, preserve the K2R/K3 controlled construction:

Prefix:

`Claim: <claim>`

`Evidence: <evidence_truncation evidence>`

`Additional evidence:`

Correction continuation:

- the unique REFUTE continuation selected by the deterministic K2R rule;
- `polarity_flip` when the unique REFUTE polarity-flip row exists;
- otherwise the unique REFUTE `none` row.

Control continuation:

- the unique `entity_swap` row.

Required semantic contracts:

- truncation row = NOT_ENTITLED / sufficiency failure / sufficiency_label 0;
- control row = NOT_ENTITLED / frame failure / polarity NONE;
- correction row = REFUTE / polarity REFUTE;
- all three rows share the identical base claim.

The candidate recipe is frozen before any K3C recurrent-state read.

## 6. Claim-disjointness

The state-blind prospective audit established zero overlap with both prior controlled population families.

Against K2W:

- pair-ID overlap = 0;
- base-claim-SHA overlap = 0;
- claim-text overlap = 0.

Against K2R/K3:

- pair-ID overlap = 0;
- base-claim-SHA overlap = 0;
- claim-text overlap = 0.

These zero-overlap conditions are required implementation gates.

Any nonzero overlap blocks K3C scientific execution.

## 7. Reciprocal block mapping

Sort K3C candidates by ascending `stable_item_id`.

For item index `i`, donor index is:

`j = i XOR 1`

giving exactly:

`150`

reciprocal blocks.

Required reciprocal mapping canonical SHA256:

`4fbc0f6642db3b2c3cca148fdc03cdc738dd6e8718cffb3fcdee73cd5b7f9acc`

No alternate pairing is permitted.

## 8. State-blind tokenizer feasibility

The frozen state-blind audit used no model forward and no recurrent-state read.

Required tokenizer/runtime identity:

HF model:

`state-spaces/mamba-130m-hf`

HF revision:

`5708daa364c50b880e7bd92eab456e0d34492ee9`

Transformers:

`5.12.1`

Tokenizer class:

`GPTNeoXTokenizer`

Fast tokenizer:

`true`

Required feasibility summary:

- N_total = 300
- N_blocks = 150
- N_matched_valid = 300
- N_swapped_valid = 300
- matched d-p = {2: 150, 3: 150}
- swapped d-p = {2: 150, 3: 150}
- matched correction available continuation min/max = [23, 28]
- matched control available continuation min/max = [20, 27]
- swapped correction available continuation min/max = [23, 28]
- swapped control available continuation min/max = [20, 27]
- prefix marginal preserved = true
- correction marginal preserved = true
- control marginal preserved = true

These values must be reproduced before implementation can be frozen.

## 9. Frozen model / checkpoint identity

Use the same authenticated seed180 A0 encoder realization used by K2S, K2R, and K3.

Expected handoff ZIP SHA256:

`96859bad3e400613b4c981990e56aaf35b1d92baaaa930eb11448cf38a63b861`

Expected checkpoint SHA256:

`4f7ad019bddb988a534c477b58b36bdabe2775d6c9748331e8311653c07c864c`

Encoder canonical digest:

`48a7e9ac9dfa6c8c292090ee0fcb606bd4c85d13706bfc8a3e371af77c440597`

Encoder raw-concatenation digest:

`968c12c095a6aab883db5984f4c02ad5e893a5ff140781ffbda41b97970401ae`

Encoder structure:

- 242 tensors;
- 129135360 total numel;
- 516541440 raw bytes;
- float32.

No training or fine-tuning is authorized.

## 10. Frozen recurrence runtime

Required Transformers version:

`5.12.1`

Required sequential Mamba source SHA256:

`23c7b410e204b5da01732566de10c94b70a8418ecb608e409754b00332eb2a41`

Required recurrence:

`MambaMixer.slow_forward`

Required state timing:

`post_consumption_s_t`

Required scientific device:

`cpu`

GPU / fast-kernel execution is not authorized because K3C requires the same exact sequential recurrence and replay semantics validated in K3.

## 11. Frozen layer and event window

Primary and only causal layer:

`23`

Primary event window:

`W = 8`

Recipient prefix endpoint:

`p = len(tokens(prefix)) - 1`

First correction/control token divergence:

`d`

Required:

`d - p in {2,3}`

for every matched and swapped pair.

Scientific replay range:

`t = d, ..., p+8`

No layer search, best-time search, window search, onset search, or endpoint selection is permitted.

## 12. Structural contribution decomposition

At frozen layer 23 define:

`G_t = discrete_A_t`

`W_t = deltaB_u_t`

and the full retained contribution:

`H_t = G_t ⊙ S_(t-1)`.

The recurrence is:

`S_t = H_t + W_t`.

Terminology:

- `W_t`: direct input-driven write contribution;
- `H_t`: history-dependent retained contribution;
- `G_t`: retention coefficient internal to H.

K3C does not treat G and H as synonyms.

K3C tests W versus H.

## 13. Natural capture contract

For every natural scientific branch, capture:

- natural `G_t`;
- natural `W_t`;
- natural pre-update `S_(t-1)`;
- natural post-update `S_t`.

Natural H is derived as:

`H_t = G_t ⊙ S_(t-1)`.

Capture must be observational and non-mutating.

Stored tensors must be cloned and non-aliased.

No K3C scientific-population capture is authorized before a separate execution-authority document is frozen.

Implementation preflight may use synthetic text only.

## 14. Exact natural structural replay

For every branch, natural replay must use:

`H_t^replay = G_t ⊙ S_(t-1)^replay`

`S_t^replay = H_t^replay + W_t`.

Using the exact shared pre-divergence initial state, natural replay must reproduce captured native states bit-for-bit at every required coordinate.

Required:

`NATURAL_STRUCTURAL_REPLAY = PASS_EXACT`

A sham replay assigning unchanged natural G and W must also be bit-exact:

`SHAM_REPLAY = PASS_EXACT`

Tolerance-based equivalence is insufficient.

## 15. Paired intervention unit

For every recipient i, preserve separately:

Matched assignment:

- `M_corr(i)`
- `M_ctrl(i)`

Swapped assignment:

- `S_corr(i)`
- `S_ctrl(i)`

Within each assignment pair, correction and control share the exact same state through `d-1`.

All contribution interventions are symmetric within each correction/control pair.

Matched and swapped pairs are never mixed when constructing intervention midpoints.

## 16. BASE condition

For every t:

`H_t^corr = G_t^corr ⊙ S_(t-1)^corr`

`H_t^ctrl = G_t^ctrl ⊙ S_(t-1)^ctrl`

and:

`S_t^corr = H_t^corr + W_t^corr`

`S_t^ctrl = H_t^ctrl + W_t^ctrl`.

BASE is the exact natural structural replay.

## 17. W_EQ condition

For each pair and every:

`t = d, ..., p+8`

define the frozen natural-write midpoint:

`Wbar_t = 0.5 * (W_t^corr + W_t^ctrl)`.

Replay retained contributions dynamically from each branch's current replay state:

`H_t^corr = G_t^corr ⊙ S_(t-1)^corr,replay`

`H_t^ctrl = G_t^ctrl ⊙ S_(t-1)^ctrl,replay`.

Then set:

`S_t^corr = H_t^corr + Wbar_t`

`S_t^ctrl = H_t^ctrl + Wbar_t`.

Thus W_EQ removes branch-specific direct-write differences while preserving the branch-specific retained-contribution dynamics generated by natural G acting on the replay histories.

Before d, replay is natural.

No normalization, rescaling, clipping, or alternate midpoint is permitted.

## 18. H_EQ condition

For each pair and every:

`t = d, ..., p+8`

first compute the branch-specific retained contributions from the current replay states:

`H_t^corr = G_t^corr ⊙ S_(t-1)^corr,replay`

`H_t^ctrl = G_t^ctrl ⊙ S_(t-1)^ctrl,replay`.

Define:

`Hbar_t = 0.5 * (H_t^corr + H_t^ctrl)`.

Then set:

`S_t^corr = Hbar_t + W_t^corr`

`S_t^ctrl = Hbar_t + W_t^ctrl`.

Thus H_EQ removes branch-specific differences in the **full retained contribution**, not merely differences in G.

Natural branch-specific W is retained.

Before d, replay is natural.

No frozen-natural-H substitution is allowed.

H must be recomputed from each condition's current replay states before midpoint equalization.

## 19. W_SEED_H_CARRY condition

This condition directly tests the successor source/carry hypothesis.

At the first divergence token:

`t = d`

compute:

`H_d^corr = G_d^corr ⊙ S_(d-1)`

`H_d^ctrl = G_d^ctrl ⊙ S_(d-1)`.

Because the pre-divergence state is shared, both use the exact common `S_(d-1)`.

Define:

`Hbar_d = 0.5 * (H_d^corr + H_d^ctrl)`.

At d set:

`S_d^corr = Hbar_d + W_d^corr`

`S_d^ctrl = Hbar_d + W_d^ctrl`.

Therefore the state difference introduced at d can arise only from branch-specific W.

For every later token:

`t = d+1, ..., p+8`

define:

`Wbar_t = 0.5 * (W_t^corr + W_t^ctrl)`.

Compute dynamic retained contributions from each replay history:

`H_t^corr = G_t^corr ⊙ S_(t-1)^corr,replay`

`H_t^ctrl = G_t^ctrl ⊙ S_(t-1)^ctrl,replay`.

Then set:

`S_t^corr = H_t^corr + Wbar_t`

`S_t^ctrl = H_t^ctrl + Wbar_t`.

Therefore no new branch-specific W difference is injected after d.

Any correction/control divergence persisting after d is propagated or reshaped through the retained-contribution path.

This condition is abbreviated:

`W_SEED_H_CARRY`.

## 20. WH_EQ integrity control

For every:

`t = d, ..., p+8`

compute dynamic H contributions from the current replay states and their midpoint:

`Hbar_t = 0.5 * (H_t^corr + H_t^ctrl)`.

Also define:

`Wbar_t = 0.5 * (W_t^corr + W_t^ctrl)`.

Set both branches:

`S_t = Hbar_t + Wbar_t`.

Required exact control:

`WH_EQ_PAIR_STATE_COLLAPSE = PASS_EXACT`

for every matched and swapped pair and every replayed token from d onward.

WH_EQ is an integrity control only.

It is not a scientific endpoint.

## 21. Required synthetic intervention controls

Before any K3C scientific execution authority can be frozen, implementation must prove on synthetic non-scientific recurrence terms:

1. natural structural replay exactness;
2. sham exactness;
3. W_EQ semantics;
4. H_EQ dynamic contribution semantics;
5. W_SEED_H_CARRY semantics;
6. WH_EQ exact pair collapse;
7. exact common d-1 state binding;
8. finite float32 tensors;
9. no state/component aliasing;
10. no scientific-population recurrent-state read during preflight.

The synthetic source/carry control must include a known-term example where:

- d-step H is equalized;
- d-step state difference is induced only by W;
- all later W is equalized;
- retained dynamics carry a nonzero state difference forward.

## 22. Frozen kinematic endpoints

For each scientific condition and branch, compute exactly the K2R/K3 layer-23 trajectory metrics:

### R

Mean speed over k=1..8.

### D

Mean valid turning over k=1..8 with the frozen epsilon policy.

### DISP

`||S_(p+8) - S_p||_F`

### P

`DISP / (path_length + 1e-12)`.

No new confirmatory trajectory metric is introduced.

## 23. Recipient pair-specificity

For condition:

`c in {BASE, W_EQ, H_EQ, W_SEED_H_CARRY}`

and metric:

`q in {R, D, DISP, P}`

define:

`Delta_q_matched(c) = q(M_corr,c) - q(M_ctrl,c)`

`Delta_q_swapped(c) = q(S_corr,c) - q(S_ctrl,c)`.

Then:

`X_q,i(c) = abs(Delta_q_matched(c)) - abs(Delta_q_swapped(c))`.

For reciprocal block b containing recipients a and b:

`B_q,b(c) = (X_q,a(c) + X_q,b(c)) / 2`.

This preserves the same matched-vs-swapped pair-specificity geometry used in K2R and K3.

## 24. Frozen direction alignment

Use the previously replicated K2R directions:

- `E_R = +1`
- `E_D = +1`
- `E_DISP = -1`
- `E_P = -1`.

Define:

`Z_q,b(c) = E_q * B_q,b(c)`.

Positive Z means that condition c expresses the previously replicated pair-specific direction.

These signs are frozen before K3C outcomes.

## 25. Mandatory BASE replication gate

K3C mechanism inference is valid only if the new claim-disjoint BASE population first reproduces the frozen K2R trajectory phenomenon.

For each q test:

`B_q(BASE)`

with the frozen expected direction `E_q`.

Use:

- exactly 150 reciprocal blocks;
- `n_valid >= 120`;
- `n_eff >= 30`;
- two-sided exact sign test;
- Holm correction with `m = 4`;
- alpha = 0.05;
- tie order `R < D < DISP < P`;
- rank-biserial sign effect in the frozen expected direction.

BASE gate verdicts:

If all four endpoints direction-match:

`K3C_BASE_TRAJECTORY_REPLICATION = PASS`

If any endpoint is Holm-significant in the opposite direction:

`K3C_BASE_TRAJECTORY_REPLICATION = CONTRADICTED`

Otherwise:

`K3C_BASE_TRAJECTORY_REPLICATION = NOT_ESTABLISHED`

If the BASE gate is not PASS, no K3C mechanistic support or contradiction verdict is authorized.

The run is then scientifically:

`INCONCLUSIVE_DUE_TO_BASELINE_REPLICATION_FAILURE`.

No population rescue is allowed.

## 26. Write-vs-retained attenuation contrast

For every q and block b define:

`ATT_W(q,b) = Z_q,b(BASE) - Z_q,b(W_EQ)`

`ATT_H(q,b) = Z_q,b(BASE) - Z_q,b(H_EQ)`.

Define the source-dominance contrast:

`DOM_q,b = ATT_W(q,b) - ATT_H(q,b)`.

Equivalently:

`DOM_q,b = Z_q,b(H_EQ) - Z_q,b(W_EQ)`.

Positive DOM means removing branch-specific direct-write differences attenuates the replicated geometry more strongly than removing branch-specific full-retained-contribution differences.

K3C predicts positive DOM for all four metric families.

## 27. Retained-carry contrast

For every q and block b define:

`CARRY_q,b = Z_q,b(W_SEED_H_CARRY)`.

Positive CARRY means that a divergence seeded at d through W alone remains expressed in the previously replicated pair-specific direction when all later branch-specific W differences are removed.

K3C predicts positive CARRY for all four metric families.

This is a causal source/carry test.

It is not a test that H alone generates divergence from an identical state.

## 28. Eight K3C confirmatory mechanism tests

Conditional on a PASS BASE gate, K3C has exactly eight confirmatory mechanism tests in this frozen order:

1. `R_DOM`
2. `R_CARRY`
3. `D_DOM`
4. `D_CARRY`
5. `DISP_DOM`
6. `DISP_CARRY`
7. `P_DOM`
8. `P_CARRY`

For each test use the 150 reciprocal block values.

Promotion floors:

- `n_valid >= 120`
- `n_eff >= 30`

Primary test:

- two-sided exact sign test;
- zero values excluded from n_eff;
- undefined values excluded from n_valid and counted explicitly;
- rank-biserial sign effect `(positive-negative)/n_eff`.

Multiple-testing correction:

- Holm;
- `m = 8`;
- alpha = 0.05;
- exact tie order equal to the eight-test order above.

Direction match requires:

- promotion floor PASS;
- Holm reject = true;
- rank-biserial sign effect > 0.

Directional contradiction requires:

- promotion floor PASS;
- Holm reject = true;
- rank-biserial sign effect < 0.

Raw p-values do not authorize promotion.

## 29. K3C full-support rule

Full K3C success requires:

1. BASE replication gate = PASS; and
2. all eight mechanism tests direction-match.

Only then:

`K3C_SCIENTIFIC_VERDICT = LAYER23_WRITE_INJECTION_WITH_RETAINED_CARRY_CAUSALLY_SUPPORTED`

Seven of eight is not full support.

Positive uncorrected trends are not full support.

## 30. K3C contradiction rule

Conditional on BASE gate = PASS:

if any of the eight mechanism tests is Holm-significant in the negative direction:

`K3C_SCIENTIFIC_VERDICT = LAYER23_WRITE_INJECTION_WITH_RETAINED_CARRY_CONTRADICTED`

A significant negative DOM test contradicts write-source dominance for that metric.

A significant negative CARRY test contradicts retained carry in the frozen pair-specific direction for that metric.

## 31. K3C not-established rule

Conditional on BASE gate = PASS:

if full support is false and there is no Holm-significant negative mechanism test:

`K3C_SCIENTIFIC_VERDICT = LAYER23_WRITE_INJECTION_WITH_RETAINED_CARRY_NOT_ESTABLISHED`

No secondary rescue can alter this primary verdict.

## 32. Anti-rescue rules

After K3C scientific execution begins, do not:

- reuse K2R/K3 items;
- replace any generated_fact_601..900 item;
- change stable-ID ordering;
- change reciprocal pairing;
- change layer 23;
- change W=8;
- choose a different d definition;
- choose a favorable time step;
- drop DISP or P;
- drop zero blocks;
- alter n_valid or n_eff floors;
- redefine H;
- substitute frozen-natural H for dynamic H;
- redefine W;
- switch from arithmetic midpoint;
- use zeroing or clipping;
- change BASE replication family;
- change the eight-test mechanism family;
- reinterpret raw p-values as corrected success;
- treat exploratory K3 p-values as K3C evidence;
- add an unregistered endpoint to rescue the claim.

A materially different intervention requires a new preregistration.

## 33. Secondary diagnostics

Secondary diagnostics may be emitted only if predeclared in implementation and must remain non-promotional.

Allowed descriptive diagnostics include:

- per-condition median absolute B and Z;
- residual-to-BASE ratios;
- sign preservation counts;
- path length as a decomposition diagnostic;
- exact-zero counts;
- component tensor norms;
- d-p subgroup summaries for d-p=2 versus d-p=3.

No secondary diagnostic can alter the K3C primary verdict.

No favorable subgroup can replace the full 300-item result.

## 34. Implementation authority

Once this preregistration is committed, it authorizes bounded implementation of:

- deterministic K3C population materialization and exact provenance checks;
- reciprocal mapping verification;
- tokenizer-only feasibility verification;
- component capture machinery;
- BASE structural replay;
- W_EQ;
- dynamic H_EQ;
- W_SEED_H_CARRY;
- WH_EQ integrity control;
- exact K2R/K3 trajectory metric reuse;
- BASE gate statistics;
- eight mechanism statistics;
- manifest and artifact schemas;
- focused tests;
- synthetic non-scientific replay preflight.

It does not authorize scientific recurrent-state execution on generated_fact_601..900.

## 35. Required implementation gates before scientific execution authority

A separate K3C execution authority may be drafted only after all of the following pass:

1. exact prereg SHA and commit binding;
2. exact generator SHA and Git blob;
3. exact generated-source canonical SHA;
4. exact candidate-pool canonical SHA;
5. exact reciprocal-mapping canonical SHA;
6. zero overlap against K2W and K2R/K3 claim identities;
7. exact tokenizer feasibility summary;
8. exact model/checkpoint/encoder fingerprints;
9. exact HF revision;
10. Transformers 5.12.1;
11. exact recurrence-source SHA;
12. CPU sequential recurrence;
13. component capture noninterference;
14. exact G/W/pre-state/post-state capture contract;
15. NATURAL_STRUCTURAL_REPLAY = PASS_EXACT;
16. SHAM_REPLAY = PASS_EXACT;
17. W_EQ synthetic semantics PASS;
18. H_EQ dynamic semantics PASS;
19. W_SEED_H_CARRY synthetic semantics PASS;
20. WH_EQ_PAIR_STATE_COLLAPSE = PASS_EXACT;
21. synthetic known-term source/carry replay PASS;
22. focused test suite PASS;
23. no K3C scientific-population recurrent-state read;
24. no K3C mechanism outcome inspected.

Gate PASS authorizes only drafting a separate execution authority.

It does not itself authorize scientific execution.

## 36. Scientific execution count

A future K3C execution authority, if frozen, may authorize at most one scientific execution on the exact frozen 300-item K3C population.

No sweep or repeated run is authorized.

A technical failure must enter explicit failure-recovery handling and may not be silently replaced by a second attempt.

## 37. Artifacts required from a future authorized execution

The future scientific harness must emit and cryptographically bind at least:

- generated source;
- candidate pool;
- reciprocal mapping;
- item-level condition metrics;
- block-level condition metrics;
- BASE replication statistics;
- K3C mechanism primary statistics;
- integrity report;
- manifest;
- scientific report;
- SHA256SUMS.

The manifest must bind:

- prereg commit/SHA;
- implementation commit/SHA;
- execution authority commit/SHA;
- population identities;
- reciprocal mapping;
- checkpoint/encoder identities;
- HF/runtime identities;
- recurrence source;
- all intervention definitions;
- exact command;
- primary statistics;
- integrity results;
- artifact hashes.

## 38. Interpretation boundary

If K3C is fully supported, the strongest justified interpretation is:

**Within the frozen layer-23 W=8 structural replay and the prospectively frozen controlled population, pair-specific trajectory geometry depends more strongly on branch-specific direct-write injection than on branch-specific full retained-contribution differences, while retained dynamics can propagate a divergence seeded through W after later branch-specific W differences are removed.**

This still does not prove:

- W is globally necessary in all Mamba layers/tasks;
- H is merely passive;
- G is irrelevant;
- final task decisions are caused by this geometry;
- authorization semantics are caused by this geometry;
- K4 claims.

## 39. K4 boundary

`K4_EXECUTION_AUTHORIZED = NO`

K3C does not reopen or automatically authorize K4.

Any future K4 study requires a separate scientific rationale and preregistration.

## 40. Frozen authority state

`K3_CLOSED = YES`

`K3C_POPULATION_STATE_BLIND_AUDIT = PASS`

`K3C_POPULATION_COUNT = 300`

`K3C_RECIPROCAL_BLOCK_COUNT = 150`

`K3C_GENERATED_SOURCE_SHA256 = 33bff5a0b657d1ceb38ae9c651e1cadfc8308286398cc1b8c4245c47f1c42000`

`K3C_CANDIDATE_POOL_SHA256 = 9603f6b20ba870807c151bb70df4c42b0957ded578729b133a37fee8aa1da83e`

`K3C_RECIPROCAL_MAPPING_SHA256 = 4fbc0f6642db3b2c3cca148fdc03cdc738dd6e8718cffb3fcdee73cd5b7f9acc`

`K3C_IMPLEMENTATION_AUTHORIZED = YES_AFTER_PREREG_FREEZE`

`K3C_SCIENTIFIC_EXECUTION_AUTHORIZED = NO`

`K4_EXECUTION_AUTHORIZED = NO`

Once committed, this preregistration freezes the K3C scientific design and authorizes only bounded implementation plus non-scientific validation.
