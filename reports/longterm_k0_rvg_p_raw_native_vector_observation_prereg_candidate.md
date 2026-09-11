# K0-RVG-P Prospective Raw Native Vector Observation Preregistration Candidate

**Status:** prospective scientific-design preregistration candidate only.

**Parent validation/readiness commit:**

`48f3309b89de15881c6ccee8743e27a33a8516a9`

**Validated observer implementation commit:**

`fcfe161c12f4ed8ef37aff435554cc0660e477af`

**Validated observer SHA256:**

`12542e32d49b368e727de782b0cb833991464503b1fab56d375f15ec58649c25`

**Validated observer test SHA256:**

`54543c91b9c919ab6fa9cf8e0b5e37380681d776f81e70a06057ed69e7c95af4`

This document defines a fresh prospective raw-vector observation study.

It does not authorize model forward, recurrent-state read, scientific execution, causal intervention, or K4.

## 1. Scientific role

K0-RVG-P is a **pre-branch observational study**.

It does not activate:

- Branch A, the parked scalar D / displacement / P replication program;
- Branch B, the parked confident-wrong versus confident-correct precursor program.

The purpose is narrower:

**test whether a fresh, claim-disjoint controlled correction population exhibits preregistered directional organization in the raw native Mamba recurrent-state trajectory, before deciding which successor branch is scientifically justified.**

Current branch state remains:

`K_SUCCESSOR_BRANCH_SELECTION = DEFERRED`

`BRANCH_A_STATUS = PARKED_NOT_CLOSED`

`BRANCH_B_STATUS = PARKED_NOT_ACTIVE`

## 2. Scientific object

The primary scientific object is the raw native selective-SSM recurrence state at frozen layer 23:

`S_t ∈ R^(1536 × 16)`

under the validated post-consumption timing.

Raw velocity:

`V_t = S_t - S_(t-1)`

Correction-control response velocity:

`R_t = V_t^(corr) - V_t^(ctrl)`

Correction-control state separation:

`DeltaCC_t = S_t^(corr) - S_t^(ctrl)`

No learned metric, whitening, PCA projection, probe, semantic projection, or outcome-selected subspace is permitted.

All primary direction calculations use the raw Frobenius inner product:

`<X,Y>_F`

and Frobenius cosine:

`cos_F(X,Y) = <X,Y>_F / (||X||_F ||Y||_F)`

computed from the frozen float32 snapshots using float64 accumulation.

## 3. Why layer 23 is frozen

Primary inference is restricted to layer 23.

This is not a fresh search over layers.

Layer 23 is retained as the historical late-layer measurement coordinate used by the existing prospective K-series line.

The scientific claim scope is therefore explicitly:

**frozen model, frozen layer 23, frozen native recurrent coordinates.**

No architecture-wide or cross-layer intrinsic-geometry claim is allowed.

The validated observer necessarily traverses all 24 Mamba layers.

For this study:

- only layer-23 vector metrics may enter confirmatory inference;
- nonprimary-layer recurrent tensors may not be persisted as scientific metrics;
- no layer may replace layer 23 after outcome inspection;
- no all-layer search may rescue a failed primary result.

## 4. Fresh population source

Generator:

`scripts/build_controlled_v5.py`

Frozen generator SHA256:

`4e9798591fbfffb6d15ea9b2f8cf5cd804a9e76713b9f6b13b354fe6ce93aa5c`

Frozen generator Git blob:

`baee23a9f71333125f4a8735c2c92d20cab7eb4f`

The fresh template slice is exactly:

`fact_templates_for_count(1572)[1236:1572]`

Global zero-based template indices:

`1236..1571`

Pair IDs:

`generated_fact_1237..generated_fact_1572`

Population size:

`336`

This range is strictly after the K3T range:

`[900:1236]`

and therefore has no pair-ID overlap with K3T.

State-blind provisioning must also prove zero overlap in:

- pair ID;
- exact claim text;
- canonical claim SHA256;

against K2W, K2R/K3, K3C, and K3T archived candidate pools.

Any overlap is a provisioning failure.

No replacement item may be selected after a failure.

## 5. Generator phase balance

The deterministic generated lexical cycle has period:

`168`

with phase:

`phase(g) = (g - 30) mod 168`

for generated-region global template index `g`.

The 336-item slice contains exactly two copies of each of the 168 phase classes.

For every phase:

- cycle 0 contains one template;
- cycle 1 contains one template;
- the frozen lexical-signature fields must match across the two cycles;
- one cycle must yield REFUTE correction source `polarity_flip`;
- the other cycle must yield REFUTE correction source `none`.

Required correction-source totals:

`polarity_flip = 168`

`none = 168`

Any phase imbalance blocks the population.

## 6. Phase-paired block definition

For local template index:

`i ∈ {0,...,167}`

define its phase mate:

`j = i + 168`

The two items share the same frozen generator lexical phase.

They form one phase-paired block.

Number of blocks:

`168`

This phase-paired block definition is frozen before tokenization and before any recurrent-state observation.

It replaces any outcome-dependent or hash-order pairing.

## 7. Controlled evidence recipe

For every item, source records must contain exactly:

1. one `evidence_truncation` row satisfying the frozen insufficiency contract;
2. one REFUTE correction row from exactly one of:
   - `polarity_flip`, or
   - `none`;
3. one `entity_swap` row satisfying the frozen NOT_ENTITLED/frame control contract.

The item prefix is:

```text
Claim: <claim>
Evidence: <truncation evidence>
Additional evidence:
```

The matched branches are:

`M_corr = prefix_i + correction_i`

`M_ctrl = prefix_i + control_i`

For the phase mate donor `j`, the phase-swapped branches are:

`S_corr = prefix_i + correction_j`

`S_ctrl = prefix_i + control_j`

The reciprocal direction is also constructed:

`prefix_j` with donor branches from item `i`.

Thus each phase block contributes two reciprocal prefix-centered contrasts.

The swapped branches are negative controls for correction/control continuation content that is not specific to the current claim/prefix.

## 8. State-blind token contract

All token validation occurs before any model forward.

Tokenizer:

`GPTNeoXTokenizer`

Model tokenizer source:

`state-spaces/mamba-130m-hf`

HF revision:

`5708daa364c50b880e7bd92eab456e0d34492ee9`

For each matched or swapped correction-control pair:

1. tokenize the complete branch text with `add_special_tokens=False`;
2. prove exact token identity over the frozen prefix;
3. define `t_e` as the first token index at or after the prefix boundary where correction and control token IDs differ;
4. require such divergence within the first 8 continuation tokens;
5. require at least 8 aligned token positions beginning at `t_e` in both branches;
6. require `t_e >= 1` so an incoming velocity exists.

The population is fail-closed.

No item may be dropped and replaced after token-state feasibility is inspected.

If any required item fails the token contract, K0-RVG-P execution remains unauthorized and a new preregistration is required.

## 9. Frozen event window

Post-divergence window:

`W = 8`

For a branch pair with divergence anchor `t_e`, define:

incoming velocity:

`U = V_(t_e-1)`

post-event branch velocities:

`V_tau^(corr) = V_(t_e+tau)^(corr)`

`V_tau^(ctrl) = V_(t_e+tau)^(ctrl)`

for:

`tau = 0,...,7`

Response velocity:

`R_tau = V_tau^(corr) - V_tau^(ctrl)`

No other window length is confirmatory.

No time-window search is allowed.

## 10. Zero-vector handling

All Frobenius dot products and norms are accumulated in float64 from the frozen float32 snapshots.

A cosine is defined only when both participating Frobenius norms are strictly greater than zero.

No epsilon is added to a scientific cosine denominator.

An item endpoint is invalid if any cosine required for that endpoint is undefined.

A phase block is invalid for an endpoint if either reciprocal item is invalid.

If more than 5% of the 168 phase blocks are invalid for a primary endpoint:

`n_valid_blocks < 160`

that endpoint is:

`NOT_EVALUABLE_DUE_TO_VECTOR_NORM_SUPPORT_FAILURE`

It may not be rescued by changing the window, layer, metric, or zero policy.

## 11. Primary endpoint P1 — pair-specific correction-induced turning

For one correction-control pair define incoming-direction alignment:

`A_corr = (1/8) * sum_(tau=0..7) cos_F(V_tau^(corr), U)`

`A_ctrl = (1/8) * sum_(tau=0..7) cos_F(V_tau^(ctrl), U)`

Define correction-induced turning contrast:

`T = A_ctrl - A_corr`

Positive `T` means the correction branch is less aligned with the incoming raw trajectory than the control branch.

For item `i`:

matched turning:

`T_M(i)`

phase-swapped turning:

`T_S(i)`

Primary item score:

`X_turn(i) = T_M(i) - T_S(i)`

For phase block `(i,j=i+168)`:

`B_turn = (X_turn(i) + X_turn(j)) / 2`

Scientific interpretation of positive `B_turn`:

the matched corrective evidence produces more claim-specific turning away from the incoming raw trajectory than its phase-matched swapped continuation control.

## 12. Primary endpoint P2 — pair-specific response-direction coherence

For one correction-control pair define response velocities:

`R_tau = V_tau^(corr) - V_tau^(ctrl)`

for `tau=0,...,7`.

Define temporal response-direction coherence:

`C = (1/7) * sum_(tau=1..7) cos_F(R_tau, R_(tau-1))`

For item `i`:

matched response coherence:

`C_M(i)`

phase-swapped response coherence:

`C_S(i)`

Primary item score:

`X_coh(i) = C_M(i) - C_S(i)`

For phase block `(i,j=i+168)`:

`B_coh = (X_coh(i) + X_coh(j)) / 2`

Scientific interpretation of positive `B_coh`:

the matched correction-control response uses a more temporally coherent raw vector direction than the phase-matched swapped continuation response.

## 13. Primary statistical tests

There are exactly two primary hypotheses:

`P1 = B_turn`

`P2 = B_coh`

For each endpoint independently:

- positive block: `B > 0`;
- negative block: `B < 0`;
- zero block: `B == 0`;
- effective n: positives + negatives.

Use an exact two-sided sign test.

Family-wise alpha:

`0.05`

Multiplicity:

Holm correction across exactly two primary p-values.

For each endpoint report:

- positive count;
- negative count;
- zero count;
- valid block count;
- effective n;
- raw exact sign-test p;
- Holm-adjusted p;
- sign effect:
  `(positive - negative) / effective_n`.

No mean-based parametric test is primary.

## 14. Endpoint verdicts

For each primary endpoint:

If Holm-adjusted p < 0.05 and sign effect > 0:

`<ENDPOINT>_POSITIVE_DIRECTIONAL_SIGNAL`

If Holm-adjusted p < 0.05 and sign effect < 0:

`<ENDPOINT>_REVERSED_DIRECTIONAL_SIGNAL`

Otherwise:

`<ENDPOINT>_DIRECTIONAL_SIGNAL_NOT_ESTABLISHED`

where `<ENDPOINT>` is:

- `TURNING`;
- `RESPONSE_COHERENCE`.

## 15. Overall raw-vector verdict

If both primary endpoints are positive-directional:

`RAW_NATIVE_VECTOR_ORGANIZATION_CONVERGENT`

If exactly one endpoint is positive-directional and the other is not established:

`RAW_NATIVE_VECTOR_ORGANIZATION_ENDPOINT_SPECIFIC`

If neither endpoint is significant:

`RAW_NATIVE_VECTOR_ORGANIZATION_NOT_ESTABLISHED`

If any endpoint is reversed-directional while the other is not positive-directional:

`RAW_NATIVE_VECTOR_ORGANIZATION_DIRECTIONALLY_CONTRADICTED_OR_MIXED`

If one endpoint is positive-directional and the other is reversed-directional:

`RAW_NATIVE_VECTOR_ORGANIZATION_MIXED_NOT_PROMOTABLE`

If either endpoint is not evaluable due to vector-norm support failure, no convergent verdict is allowed.

## 16. Scalar trajectory summaries are secondary diagnostics only

The following may be computed at layer 23 over the same frozen windows:

- raw speed;
- displacement norm;
- path length;
- path efficiency;
- correction-control separation magnitude;
- response magnitude.

Historical D / displacement / P definitions may be reproduced for context where algebraically applicable.

However:

`SCALAR_ENDPOINTS_PRIMARY = NO`

No scalar result can rescue failed P1/P2.

No scalar p-value enters the two-test Holm family.

No scalar result may be called the primary K0-RVG-P finding.

## 17. Recurrence-exact carry/write diagnostics

For every branch and token:

`V_t = V_t^(carry-change) + V_t^(write)`

with:

`V_t^(carry-change) = (G_t - 1) ⊙ S_(t-1)`

`V_t^(write) = W_t`

For correction-control response define:

`R_tau^(carry) = V_tau^(carry-change,corr) - V_tau^(carry-change,ctrl)`

`R_tau^(write) = V_tau^(write,corr) - V_tau^(write,ctrl)`

and:

`R_tau = R_tau^(carry) + R_tau^(write)`

The following are descriptive diagnostics only:

- `cos_F(R_write, R_total)`;
- `cos_F(R_carry, R_total)`;
- `cos_F(R_write, R_carry)`;
- Frobenius norms of each component;
- exact source-recurrence reconstruction status;
- velocity-rearrangement residual.

No carry/write diagnostic is a causal endpoint.

No causal specialization or intervention claim is authorized.

## 18. Singular-spectrum descriptive audit

For each matched correction-control response window, construct:

`M = [vec(R_0)^T; ...; vec(R_7)^T]`

and record its singular values.

Permitted descriptive outputs:

- normalized singular spectrum;
- fraction of squared Frobenius energy in singular values 1..8;
- numerical rank under a separately frozen machine-precision reporting rule.

Forbidden:

- using a singular vector as a confirmatory projection;
- selecting rank to maximize group separation;
- using singular spectrum to replace failed P1/P2;
- fitting a same-data low-dimensional geometry.

## 19. Fresh-data integrity

The new population is not allowed to be inspected through recurrent states before all of the following are frozen in code and artifacts:

- exact candidate pool;
- exact phase-pair mapping;
- exact token contracts;
- exact layer;
- exact W=8 window;
- exact P1/P2 formulas;
- exact zero-vector policy;
- exact Holm procedure;
- exact output schema;
- exact implementation/runtime identities.

Any scientific recurrent-state read before these are frozen invalidates confirmatory use of that population.

## 20. No outcome-selected rescue

After scientific execution begins, the following are forbidden rescue operations:

- another layer;
- another window;
- another event anchor;
- another cosine/reference direction;
- another pair mapping;
- another zero threshold;
- PCA/subspace projection;
- whitening;
- Mahalanobis geometry;
- channel selection;
- state-mode selection;
- alternate population slice;
- deleting unfavorable blocks;
- scalar-only promotion.

A failed result remains a failed result for K0-RVG-P.

## 21. Claim boundaries

A positive K0-RVG-P result may establish only:

**fresh same-generator evidence for pair-specific directional organization of raw native recurrent-state motion at frozen layer 23 under controlled correction/control continuations.**

It does not establish:

- confident-error prediction;
- pre-evidence epistemic failure detection;
- Branch B;
- stable D/DISP/P transportability;
- Branch A;
- semantic meaning of a vector direction;
- an intrinsic coordinate-free Mamba manifold;
- carry/write causality;
- decision-space authorization or polarity;
- external-distribution generalization;
- K4.

## 22. Relationship to Branch A

Branch A remains:

`PARKED_NOT_CLOSED`

K0-RVG-P may produce scalar D / displacement / P diagnostics, but these are not a Branch-A confirmatory replication because the study's primary hypotheses and multiplicity family are vector-directional.

A later Branch-A study would require its own explicit activation and preregistration.

## 23. Relationship to Branch B

Branch B remains:

`PARKED_NOT_ACTIVE`

K0-RVG-P does not compare confident-wrong versus confident-correct examples.

Therefore even a strong positive result does not establish the original K0 confident-error precursor.

A later Branch-B study still requires a genuinely new support design and independent confirmatory population.

## 24. Scientific decision after K0-RVG-P

After a valid execution is independently validated:

- `RAW_NATIVE_VECTOR_ORGANIZATION_CONVERGENT` provides strong motivation to design a new branch-specific study around raw vector objects;
- `...ENDPOINT_SPECIFIC` provides limited motivation and requires endpoint-specific interpretation;
- `...NOT_ESTABLISHED` argues against treating raw direction as a stable phenomenon in this controlled line;
- mixed/reversed results require interpretation before either branch can inherit the result.

No branch becomes active automatically from the execution verdict.

A separate synthesis/decision artifact is required.

## 25. State-blind next stage

After this preregistration is frozen, the only authorized next stage is:

`K0-RVG-P0 — State-Blind Population / Token Contract Provisioning Specification`

K0-RVG-P0 may define implementation for:

- deterministic population materialization;
- prior-pool overlap checks;
- phase-pair mapping;
- tokenizer identity;
- token-prefix identity;
- divergence-anchor validation;
- W=8 availability;
- candidate/mapping/token-contract hashes.

It may not run the Mamba model.

It may not import the validated observer to capture recurrent state.

It may not inspect logits or scientific outcomes.

## 26. Authority state

`K0_RVG_P_PREREGISTRATION_DEFINED = YES`

`K_SUCCESSOR_BRANCH_SELECTION = DEFERRED`

`BRANCH_A_STATUS = PARKED_NOT_CLOSED`

`BRANCH_B_STATUS = PARKED_NOT_ACTIVE`

`FRESH_TEMPLATE_RANGE = [1236,1572)`

`FRESH_POPULATION_N = 336`

`PHASE_BLOCKS = 168`

`PRIMARY_LAYER = 23`

`PRIMARY_WINDOW = 8`

`PRIMARY_VECTOR_ENDPOINT_COUNT = 2`

`SCALAR_ENDPOINTS_PRIMARY = NO`

`LEARNED_OR_TUNED_GEOMETRY_AUTHORIZED = NO`

`SCIENTIFIC_MODEL_FORWARD_AUTHORIZED = NO`

`SCIENTIFIC_RECURRENT_STATE_READ_AUTHORIZED = NO`

`SCIENTIFIC_POPULATION_STATE_ACCESS_AUTHORIZED = NO`

`CAUSAL_INTERVENTION_AUTHORIZED = NO`

`K4_EXECUTION_AUTHORIZED = NO`

`K0_RVG_P0_STATE_BLIND_PROVISIONING_SPEC_AUTHORIZED_AFTER_FREEZE = YES`

This document authorizes no scientific execution.
