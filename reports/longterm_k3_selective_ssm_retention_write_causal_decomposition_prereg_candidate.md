# K3 Selective-SSM Retention-vs-Write Causal Decomposition Preregistration Candidate

**Status:** K3 CAUSAL-MECHANISM SCIENTIFIC DESIGN / PREREGISTRATION CANDIDATE.

**Authority boundary:** once committed, this document authorizes bounded K3 implementation and non-scientific instrumentation/replay validation only. It does **not** authorize K3 scientific intervention execution. A separate K3 execution authority must be frozen after the implementation passes the exact equivalence and intervention-integrity gates defined here.

K3 does not reopen K2S or K2R.

K3 does not authorize K4 decision-space linkage.

## 1. Governing evidence state

Current authority anchor before K3:

`ba0ad9052a3b8a5eb80fef46dea461f372f95ee8`

This commit archives the validated K2R full-replication result.

K2R implementation/runtime commit:

`52bd363bb3690b4c85dbdb6add686ecb31088627`

K2R preregistration:

`54b1a9a2188e3e678e8378b43faa5d527aca1457`

K2R frozen verdict:

`LOCAL_VS_NET_TRAJECTORY_DISSOCIATION_REPLICATED`

K2R established, within the same deterministic controlled-generator family and on a claim-disjoint prospective population, that:

- R mean-speed pair-specificity replicated in the positive matched-semantic direction;
- D mean-turning pair-specificity replicated in the positive matched-semantic direction;
- DISP net-displacement pair-specificity replicated in the negative direction;
- P trajectory-efficiency pair-specificity replicated in the negative direction.

K2R did **not** establish:

- a causal selective-SSM mechanism;
- native-state necessity;
- native-state sufficiency;
- task-decision causality;
- an epistemic-state ontology;
- external-distribution generalization.

K0 explicitly parked selective recurrent-update decomposition for a later positive-result mechanistic stage and required perturbational/interventional evidence before causal/mechanistic language.

K3 is that bounded mechanistic stage.

## 2. K3 scientific question

K3 asks:

**Within the frozen layer-23 native Mamba recurrence, is the replicated local-vs-net trajectory dissociation differentially caused by branch-specific input-driven write terms versus branch-specific retained-state gating?**

K3 tests a specific component-specialization hypothesis:

- local movement/directional dynamics, R and D, are predicted to depend more strongly on branch-specific input-driven write;
- net displacement/path-efficiency dynamics, DISP and P, are predicted to depend more strongly on branch-specific retained-state gating.

The opposite pattern does not count as confirmatory success.

A partial pattern does not count as full success.

## 3. Claim boundary

A K3 positive result may support only a claim of the form:

`LAYER23_NATIVE_RECURRENCE_COMPONENT_SPECIALIZATION_CAUSALLY_SUPPORTED`

meaning that, under the exact frozen replay interventions, branch-specific write versus retention-gate terms make different causal contributions to the replicated layer-23 trajectory geometry.

Even a K3 positive result does not establish:

- causal control of the final task decision;
- causal control of authorization/entitlement semantics;
- a confident-error precursor;
- a deployable detector;
- natural-corpus generalization;
- external-distribution generalization;
- a unique biological/physical ontology of the state coordinates.

Those remain outside K3.

## 4. Frozen population

Use exactly the archived K2R candidate population:

`reports/longterm_k2r_claim_disjoint_replication_52bd363_v1/candidate_pool.jsonl`

Expected SHA256:

`00bbdc9977679aa0561be413e7cef8e710dc7987618fe9593e8bf0852a5a7de4`

Expected item count:

`300`

Use the exact K2R reciprocal item order and pairing derived from ascending `stable_item_id`:

`j = i XOR 1`

giving exactly:

`150`

reciprocal blocks.

No K3 item filtering, replacement, weighting, or resampling is allowed.

The K2S population is not part of K3 primary inference.

## 5. Frozen model / checkpoint / tokenizer identity

Use the same authenticated seed180 encoder realization used in K2S/K2R.

Handoff ZIP expected SHA256:

`96859bad3e400613b4c981990e56aaf35b1d92baaaa930eb11448cf38a63b861`

Selected checkpoint expected SHA256:

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

Use:

`state-spaces/mamba-130m-hf`

at exact HF revision:

`5708daa364c50b880e7bd92eab456e0d34492ee9`

Tokenizer semantics remain:

- fast tokenizer;
- `trust_remote_code = false`;
- `add_special_tokens = false`;
- no padding;
- no truncation.

## 6. Frozen runtime recurrence source

K3 is bound to the same CPU sequential Mamba recurrence source observed in K2R.

Required Transformers version:

`5.12.1`

Required `transformers.models.mamba.modeling_mamba.py` SHA256:

`23c7b410e204b5da01732566de10c94b70a8418ecb608e409754b00332eb2a41`

Required native-state timing:

`post_consumption_s_t`

Required scientific device:

`cpu`

GPU fast-kernel execution is not authorized for K3 because it does not expose the same validated recurrence/replay semantics.

Any recurrence-source or runtime-path drift blocks scientific execution.

## 7. Frozen layer and event window

Primary/only causal layer:

`layer 23`

This is retained because layer 23 was prospectively frozen as the K2S and K2R primary layer before their scientific outcomes.

No intermediate-layer causal search is allowed.

Primary event geometry remains:

`W = 8`

with recipient prefix end:

`p = len(tokens(P_i)) - 1`

and post-prefix state window:

`S_(p+1)` through `S_(p+8)`.

For each correction/control branch pair, define:

`d`

as the first token index after the common prefix at which correction and control token IDs differ.

The K2R construction freezes:

`d - p in {2,3}`

for every matched and swapped pair.

K3 intervention begins at `d`, not at a post-hoc state-dynamic time point.

K3 intervention ends at `p+8`.

No best-time, k=7, onset, peak, or latency selection is allowed.

## 8. Selective recurrence structural decomposition

For the frozen sequential recurrence, define at layer 23:

`G_t = discrete_A_t`

and:

`W_t = deltaB_u_t`

so the native recurrent-state structural update is:

`S_t = G_t ⊙ S_(t-1) + W_t`

where `⊙` is elementwise multiplication.

Terminology:

- `G_t` is the **retention coefficient** or **retention-gate term**;
- `W_t` is the **input-driven write increment**.

This notation intentionally avoids using `A` for the K3 causal endpoint because `A` is reserved elsewhere for authorization/entitlement decision-space notation.

K3 does not claim that G and W are statistically independent.

In Mamba, selectivity parameters can share upstream dependencies.

K3 tests their distinct structural positions in the recurrence equation.

## 9. Required baseline component capture

For every scientific branch, K3 implementation must capture at layer 23, through the required window:

- natural `G_t`;
- natural `W_t`;
- natural post-consumption recurrent state `S_t`.

The capture must be observational/non-mutating.

Before any intervention can be authorized, the implementation must prove:

1. trace/capture noninterference on the ordinary frozen model path;
2. exact source binding to the frozen recurrence equation;
3. exact token-coordinate binding;
4. exact layer binding;
5. finite float32 tensors;
6. no aliasing between stored component snapshots and live tensors.

## 10. Exact structural replay requirement

K3 intervention is performed by offline structural replay of the frozen layer-23 recurrence, not by changing tokens or lower-layer hidden states.

For every baseline branch, reconstruct:

`S_t^replay = G_t ⊙ S_(t-1)^replay + W_t`

using the captured natural terms and the exact natural initial state required by the recurrence.

Before scientific execution is authorized, natural replay must reproduce the trace-captured native recurrent state **bit-for-bit** at every replayed token coordinate required for K3.

A tolerance-based baseline equivalence is insufficient.

Required result:

`NATURAL_STRUCTURAL_REPLAY = PASS_EXACT`

A sham replay that explicitly reassigns each branch its own unchanged G and W must also be bit-exact:

`SHAM_REPLAY = PASS_EXACT`

Any failure blocks K3 execution.

## 11. Paired intervention unit

For each recipient i, K3 preserves the two K2R assignment families separately:

### Matched pair

- correction branch: `M_corr(i)`
- control branch: `M_ctrl(i)`

### Swapped pair

- correction branch: `S_corr(i)`
- control branch: `S_ctrl(i)`

Within each assignment family, correction and control have an exact common state history through `d-1`.

Interventions are applied symmetrically to the correction/control pair.

Matched and swapped assignments are never mixed when constructing intervention midpoints.

## 12. W-equalization intervention

For an assignment pair and every token:

`t = d, ..., p+8`

let the natural write terms be:

`W_t^corr`

and:

`W_t^ctrl`.

Define the exact arithmetic midpoint:

`Wbar_t = 0.5 * (W_t^corr + W_t^ctrl)`.

Under condition:

`W_EQ`

set:

`W_t^corr := Wbar_t`

and:

`W_t^ctrl := Wbar_t`

for replay only.

Keep each branch's natural retention coefficient:

`G_t^corr`

and:

`G_t^ctrl`.

Before d, all natural terms are retained unchanged.

No normalization or rescaling of Wbar is permitted.

## 13. G-equalization intervention

For every token:

`t = d, ..., p+8`

let the natural retention coefficients be:

`G_t^corr`

and:

`G_t^ctrl`.

Define the exact arithmetic midpoint:

`Gbar_t = 0.5 * (G_t^corr + G_t^ctrl)`.

Under condition:

`G_EQ`

set:

`G_t^corr := Gbar_t`

and:

`G_t^ctrl := Gbar_t`

for replay only.

Keep each branch's natural write term:

`W_t^corr`

and:

`W_t^ctrl`.

Before d, all natural terms are retained unchanged.

No log-space averaging, renormalization, clipping, or alternate midpoint is permitted.

## 14. GW-equalization instrumentation control

Define:

`GW_EQ`

by applying both G_EQ and W_EQ over:

`t = d, ..., p+8`.

Because the paired correction/control branches share the same state at d-1 and receive identical G and W thereafter, their replayed native states must be exactly identical from d through p+8.

Required implementation control:

`GW_EQ_PAIR_STATE_COLLAPSE = PASS_EXACT`

for every matched and swapped recipient pair.

GW_EQ is an implementation/integrity control only.

It is not a scientific primary endpoint and cannot be used for promotion.

## 15. No lower-layer or token intervention

K3 does not alter:

- input text;
- token IDs;
- tokenizer;
- embeddings;
- convolutional inputs;
- layers 0 through 22;
- task heads;
- attention masks;
- A0 predictions;
- confidence;
- branch assignment.

The natural layer-23 input-derived G/W tensors are captured from the unmodified frozen branches and then used as exogenous replay inputs.

K3 therefore tests the layer-23 recurrence structural equation only.

## 16. Frozen kinematic endpoints

For each replay condition and branch, compute exactly the K2R primary state metrics:

### R

Mean speed over k=1..8.

### D

Mean valid turning over k=1..8 using the frozen epsilon policy.

### DISP

Net displacement:

`||S_(p+8) - S_p||_F`

### P

Trajectory efficiency:

`DISP / (path_length + 1e-12)`.

The metric implementation must be byte-identical in semantics to K2R.

No new primary state metric is introduced.

## 17. Recipient pair-specificity under replay condition

For condition:

`c in {BASE, W_EQ, G_EQ}`

and metric q:

`q in {R, D, DISP, P}`

define within each assignment family:

`Delta_q_matched(c) = q(M_corr,c) - q(M_ctrl,c)`

`Delta_q_swapped(c) = q(S_corr,c) - q(S_ctrl,c)`.

Then:

`X_q,i(c) = abs(Delta_q_matched(c)) - abs(Delta_q_swapped(c))`.

For reciprocal block b containing recipients a and b:

`B_q,b(c) = (X_q,a(c) + X_q,b(c)) / 2`.

This exactly preserves the K2R pair-specificity geometry.

## 18. Direction alignment

Use the frozen K2R replicated directions:

- `E_R = +1`
- `E_D = +1`
- `E_DISP = -1`
- `E_P = -1`.

Define aligned block signal:

`Z_q,b(c) = E_q * B_q,b(c)`.

Thus positive Z always means the replay condition expresses the previously replicated K2R direction.

This sign alignment is fixed before K3 intervention outcomes are observed.

## 19. Prospectively frozen component-specialization hypothesis

Predicted dominant recurrence component:

### Local family

For:

- R
- D

predicted dominant component:

`W`

The scientific prediction is that equalizing branch-specific write terms attenuates the replicated aligned effect more strongly than equalizing branch-specific retention coefficients.

### Net family

For:

- DISP
- P

predicted dominant component:

`G`

The scientific prediction is that equalizing branch-specific retention coefficients attenuates the replicated aligned effect more strongly than equalizing branch-specific write terms.

This local-W / net-G specialization is the only confirmatory K3 mechanism hypothesis.

The opposite specialization does not count as success.

## 20. Causal attenuation quantities

For every endpoint q and block b, define:

`ATT_W(q,b) = Z_q,b(BASE) - Z_q,b(W_EQ)`

and:

`ATT_G(q,b) = Z_q,b(BASE) - Z_q,b(G_EQ)`.

Positive attenuation means the intervention weakens the previously replicated directional effect.

Define the predicted dominant attenuation:

For q in {R,D}:

`ATT_DOM = ATT_W`

`ATT_OTHER = ATT_G`.

For q in {DISP,P}:

`ATT_DOM = ATT_G`

`ATT_OTHER = ATT_W`.

Define the selectivity contrast:

`SEL_q,b = ATT_DOM(q,b) - ATT_OTHER(q,b)`.

Positive SEL means the preregistered dominant component intervention attenuates more strongly than the alternative component intervention.

## 21. Eight confirmatory tests

K3 has exactly eight confirmatory block-level tests.

For each q in ordered metric family:

`R, D, DISP, P`

test:

1. `ATT_q`: predicted dominant attenuation `ATT_DOM > 0`;
2. `SEL_q`: component selectivity `SEL > 0`.

Exact primary order:

1. `R_ATT`
2. `R_SEL`
3. `D_ATT`
4. `D_SEL`
5. `DISP_ATT`
6. `DISP_SEL`
7. `P_ATT`
8. `P_SEL`

No other statistic enters the K3 confirmatory family.

## 22. Statistical procedure

Primary inference unit:

`150 reciprocal blocks`.

For every one of the eight tests:

- report positive count;
- negative count;
- exact-zero count;
- undefined count;
- n_valid;
- n_eff;
- rank-biserial sign effect.

Exact zeros are retained as valid observations and excluded only from n_eff.

Support floor:

- n_valid >= 120;
- n_eff >= 30.

If the support floor fails:

raw p = 1.

Otherwise use a two-sided exact binomial sign test under p=0.5.

Apply Holm familywise correction across exactly:

`m = 8`

tests at:

`alpha = 0.05`.

Deterministic Holm tie order is the exact primary order listed above.

Confirmatory directional match requires:

- support floor PASS;
- Holm reject;
- rank-biserial sign effect > 0.

## 23. Full K3 success criterion

K3 achieves:

`LAYER23_NATIVE_RECURRENCE_COMPONENT_SPECIALIZATION_CAUSALLY_SUPPORTED`

if and only if all eight confirmatory tests simultaneously achieve directional match.

That requires:

- R dominant W equalization causally attenuates the replicated R effect;
- R W attenuation exceeds G attenuation;
- D dominant W equalization causally attenuates the replicated D effect;
- D W attenuation exceeds G attenuation;
- DISP dominant G equalization causally attenuates the replicated DISP effect;
- DISP G attenuation exceeds W attenuation;
- P dominant G equalization causally attenuates the replicated P effect;
- P G attenuation exceeds W attenuation.

Eight-of-eight is required.

Seven-of-eight is not full support.

## 24. Contradiction and non-establishment verdicts

If one or more confirmatory tests are Holm-significant with rank-biserial sign effect < 0, use:

`LAYER23_NATIVE_RECURRENCE_COMPONENT_SPECIALIZATION_CONTRADICTED`

unless all eight positive success conditions are met, which by definition cannot coexist with such a contradiction.

If valid execution completes but full support is absent and there is no significant directional contradiction, use:

`LAYER23_NATIVE_RECURRENCE_COMPONENT_SPECIALIZATION_NOT_ESTABLISHED`.

Partial endpoint patterns may be reported descriptively but cannot be promoted as the frozen K3 mechanism.

## 25. Baseline reproduction gate during scientific execution

Even after implementation preflight, a K3 scientific execution must reproduce the archived natural K2R primary block values from the same exact K2R population before intervention statistics are accepted.

The archived K2R block artifact is:

`reports/longterm_k2r_claim_disjoint_replication_52bd363_v1/block_metrics.jsonl`

Expected SHA256:

`e4982e8a57e15863227d17f080e7a7a699fb14100387fc36c7c47ab68962c8de`

K3 natural replay must reproduce the archived block-level R/D/DISP/P values according to an exact frozen serialization/recomputation policy established during implementation validation.

Scientific execution fails closed if baseline reproduction fails.

## 26. Implementation equivalence gates required before execution authority

K3 implementation must pass all of the following before a scientific execution authority can be frozen:

1. exact K2R population SHA;
2. exact K2R reciprocal mapping;
3. exact checkpoint and encoder fingerprints;
4. exact HF model/tokenizer revision;
5. exact Transformers 5.12.1 recurrence-source SHA;
6. CPU sequential recurrence proof;
7. observational component capture noninterference;
8. exact G/W coordinate and shape contract;
9. `NATURAL_STRUCTURAL_REPLAY = PASS_EXACT`;
10. `SHAM_REPLAY = PASS_EXACT`;
11. `GW_EQ_PAIR_STATE_COLLAPSE = PASS_EXACT`;
12. synthetic known-term replay tests;
13. focused unit tests;
14. no scientific K2R/K3 intervention outcome inspected during these gates.

Only after all gates pass may a separate K3 execution authority be written.

## 27. Required implementation tests

At minimum, focused tests must cover:

- exact source/provenance constants;
- G/W extraction binding;
- recurrence AST/source proof;
- pair divergence index;
- intervention-span boundaries;
- arithmetic-midpoint definition;
- W_EQ semantics;
- G_EQ semantics;
- GW_EQ semantics;
- natural replay exact equality;
- sham exact equality;
- GW pair collapse;
- metric equivalence with K2R;
- X/B/Z definitions;
- attenuation definitions;
- SEL definitions;
- exact sign test;
- Holm m=8 and tie order;
- verdict logic;
- output-directory fail-closed behavior;
- dirty-worktree contract.

Synthetic tests must not use scientific K2R native-state outcomes.

## 28. Scientific artifact requirements

A later authorized K3 execution must emit and cryptographically bind at minimum:

- exact K2R population identity;
- exact reciprocal mapping;
- exact token arrays;
- d and p per recipient;
- runtime/package/source identities;
- G/W capture provenance;
- natural replay equivalence evidence;
- sham equivalence evidence;
- GW_EQ collapse evidence;
- recipient BASE/W_EQ/G_EQ metrics;
- recipient X values by condition;
- block B values by condition;
- aligned Z values;
- ATT_W;
- ATT_G;
- ATT_DOM;
- ATT_OTHER;
- SEL;
- complete eight-test primary statistics;
- scientific verdict;
- artifact SHA256s.

Raw full recurrent-state tensors need not be serialized if exact hashes and all required reduced scientific quantities are bound.

## 29. Prespecified descriptive outputs

Permitted descriptive outputs include:

- G norm summaries;
- W norm summaries;
- retained contribution norm `||G_t ⊙ S_(t-1)||`;
- write contribution norm `||W_t||`;
- contribution angle/alignment;
- per-step intervention attenuation;
- matched versus swapped decomposition;
- correction-source category;
- exact-zero block patterns.

These are not confirmatory endpoints.

They cannot replace failed ATT/SEL tests.

## 30. Explicit anti-rescue rules

K3 may not:

- choose another layer;
- choose another W;
- choose k=7 or another favorable time;
- shift intervention onset after observing outcomes;
- intervene only on favorable recipients;
- drop zero blocks;
- drop unfavorable blocks;
- replace arithmetic midpoint with another intervention after outcomes;
- switch from G coefficient equalization to full retained-contribution equalization after outcomes;
- zero or clamp a component as a rescue;
- change absolute pair-specificity;
- remove an endpoint;
- change the eight-test family;
- change Holm m=8;
- pool item-level observations as independent replicates;
- use A0 confidence/prediction for filtering;
- use task-head outcomes for K3 promotion;
- promote a partial result to the full causal-specialization label.

Any such change is a new experiment.

## 31. K4 boundary

K3 is causal characterization of native-state recurrence geometry only.

K3 does not test whether the intervention causally changes:

- final task class;
- task confidence;
- entitlement coordinate;
- polarity coordinate;
- decision margin.

Those belong to a separately preregistered K4 decision-space linkage study.

A positive K3 result may motivate K4 preregistration.

It does not authorize K4 execution.

## 32. Execution-compute policy

K3 implementation/preflight and any later scientific execution must preserve the exact CPU sequential recurrence semantics.

Kaggle GPU execution is not authorized under this design.

A remote CPU environment could be used only if all package/source/checkpoint/provenance identities and replay-equivalence gates are reproduced exactly.

The default environment is the already validated local CPU runtime.

## 33. Current authority state

After this preregistration is committed:

`K3_DESIGN_FROZEN = YES`

`K3_IMPLEMENTATION_AUTHORIZED = YES`

`K3_SYNTHETIC_REPLAY_VALIDATION_AUTHORIZED = YES`

`K3_SCIENTIFIC_INTERVENTION_EXECUTION_AUTHORIZED = NO`

`K4_AUTHORIZED = NO`

No model training or fine-tuning is authorized.

## 34. Final preregistration verdict

`PASS_READY_TO_FREEZE_K3_CAUSAL_DECOMPOSITION_PREREG`

This is a causal-design authority candidate, not scientific evidence and not execution authority.
