# K3C Native Recurrence Contribution Scientific Closure Report Candidate

**Status:** RESULT / INTERPRETATION / CLOSURE REPORT ONLY.

**Authority status:** NOT EXECUTION AUTHORITY.

**Preregistration status:** NOT A K3C AMENDMENT.

This report closes the one authorized K3C scientific recurrent-state execution.

No new scientific execution, rerun, rescue, K3C population replacement, or K4 execution is authorized by this report.

## 1. Execution identity

K3C preregistration commit:

`b85272d88d0bb57db45fdc963d313714529e7975`

K3C preregistration SHA256:

`0a9de28237e107ce3a62d4fa9e0bb7f230d2e0289e019d3864edf491dd271436`

K3C replay implementation commit:

`e91116b4b5837a16de1d6eabfa2503be5bfe1d3d`

K3C replay module SHA256:

`ab07c6052a07af354043e0aa38bf24fd73c2fde53fccec125c25f7db2718f233`

K3C scientific execution harness commit:

`6232a601ad762d20abe67473c0f2cf2e510f9849`

Harness SHA256:

`fefef63a6b49d39ec3dbd34cb5b7413d54e15e6616992a2972e0c343334454b2`

Harness-test SHA256:

`36b9ab89dee06835dd5a9765eb6cf31d5d9ed810005fc36f5c01bc3c1012f53a`

One-run scientific execution authority commit:

`db75edfbf34bb6efacf94982b18406fd4284de56`

Authority SHA256:

`8d4b1c104ded779f0461a05ffd40ef3b10182de2c67ae4bd969e453e86697746`

Scientific output directory:

`C:\Users\Home1\Desktop\ContraMamba-K3C-Runs\k3c-contribution-db75edfbf34b`

Independent post-execution artifact/statistics validation:

`PASS`

## 2. Frozen population identity

The execution used the prospectively frozen claim-disjoint K3C population.

Global generated-template slice:

`[600:900]`

Pair IDs:

`generated_fact_601` through `generated_fact_900`

Generated source row count:

`3900`

Generated-source canonical SHA256:

`33bff5a0b657d1ceb38ae9c651e1cadfc8308286398cc1b8c4245c47f1c42000`

Candidate count:

`300`

Candidate-pool canonical SHA256:

`9603f6b20ba870807c151bb70df4c42b0957ded578729b133a37fee8aa1da83e`

Reciprocal block count:

`150`

Reciprocal-mapping canonical SHA256:

`4fbc0f6642db3b2c3cca148fdc03cdc738dd6e8718cffb3fcdee73cd5b7f9acc`

Prior-population pair/claim overlap with K2W and K2R/K3:

`0`

No population filtering, replacement, resampling, or rescue occurred.

## 3. Runtime identity

Model:

`state-spaces/mamba-130m-hf`

HF revision:

`5708daa364c50b880e7bd92eab456e0d34492ee9`

Transformers version:

`5.12.1`

Recurrence:

`MambaMixer.slow_forward`

Mamba source SHA256:

`23c7b410e204b5da01732566de10c94b70a8418ecb608e409754b00332eb2a41`

Primary causal layer:

`23`

Window:

`W = 8`

Device:

`cpu`

State timing:

`post_consumption_s_t`

Handoff ZIP SHA256:

`96859bad3e400613b4c981990e56aaf35b1d92baaaa930eb11448cf38a63b861`

Checkpoint SHA256:

`4f7ad019bddb988a534c477b58b36bdabe2775d6c9748331e8311653c07c864c`

Encoder canonical digest:

`48a7e9ac9dfa6c8c292090ee0fcb606bd4c85d13706bfc8a3e371af77c440597`

Encoder raw-concatenation digest:

`968c12c095a6aab883db5984f4c02ad5e893a5ff140781ffbda41b97970401ae`

## 4. Scientific integrity

All required structural replay and intervention integrity gates passed.

Natural H+W identity:

`PASS_EXACT`

Natural structural replay:

`PASS_EXACT`

Natural exact branch count:

`1200`

Sham replay:

`PASS_EXACT`

Sham exact branch count:

`1200`

W_SEED_H_CARRY d-step semantics:

`PASS_EXACT`

W-seed pair exact count:

`600`

WH_EQ pair-state collapse:

`PASS_EXACT`

WH pair exact count:

`600`

Synthetic replay preflight:

`PASS_K3C_REPLAY_PREFLIGHT`

Synthetic preflight scientific-population recurrent-state read:

`false`

Synthetic preflight scientific-population intervention executed:

`false`

Therefore the K3C outcome is not attributed to provenance drift, recurrence drift, replay mismatch, failed H/W decomposition, failed W-seed semantics, or failed WH collapse.

## 5. Mandatory BASE replication gate

K3C preregistered that mechanistic promotion was permitted only if all four previously replicated trajectory endpoints reproduced on the new claim-disjoint population.

The four frozen BASE endpoints were:

- R;
- D;
- DISP;
- P.

The frozen rule required all four endpoint direction matches after Holm correction with m=4.

### R

Expected direction:

`positive`

Counts:

- positive = 78
- negative = 68
- zero = 4
- undefined = 0
- n_valid = 150
- n_eff = 146

Rank-biserial sign effect:

`+0.0684931506849315`

Raw p:

`0.45648223496512574`

Holm-adjusted p:

`0.45648223496512574`

Direction match:

`NO`

Directional contradiction:

`NO`

Interpretation:

R did not reproduce confirmatorily on the new claim-disjoint K3C population.

The observed effect remained slightly positive, but it was weak and non-significant.

This is a replication failure for R, not an opposite-direction contradiction.

### D

Expected direction:

`positive`

Counts:

- positive = 94
- negative = 52
- zero = 4
- undefined = 0
- n_valid = 150
- n_eff = 146

Rank-biserial sign effect:

`+0.2876712328767123`

Raw p:

`0.0006398230663331769`

Holm-adjusted p:

`0.0012796461326663538`

Direction match:

`YES`

D reproduced.

### DISP

Expected direction:

`negative`

Counts:

- positive = 28
- negative = 118
- zero = 4
- undefined = 0
- n_valid = 150
- n_eff = 146

Rank-biserial sign effect:

`-0.6164383561643836`

Raw p:

`2.4024726949774936e-14`

Holm-adjusted p:

`7.20741808493248e-14`

Direction match:

`YES`

DISP reproduced strongly.

### P

Expected direction:

`negative`

Counts:

- positive = 15
- negative = 131
- zero = 4
- undefined = 0
- n_valid = 150
- n_eff = 146

Rank-biserial sign effect:

`-0.7945205479452054`

Raw p:

`2.679769781117354e-24`

Holm-adjusted p:

`1.0719079124469416e-23`

Direction match:

`YES`

P reproduced very strongly.

## 6. Frozen BASE verdict

Direction-matched endpoints:

`["D", "DISP", "P"]`

Directional-contradiction endpoints:

`[]`

Frozen BASE replication verdict:

`NOT_ESTABLISHED`

The complete four-endpoint K2R trajectory signature did not prospectively reproduce on the new claim-disjoint K3C population because R failed the preregistered replication criterion.

The appropriate interpretation is not that the trajectory phenomenon disappeared entirely.

Three of four endpoints reproduced, including the strongest net-geometry endpoints DISP and P.

However, the preregistered four-endpoint joint replication condition failed.

## 7. Consequence for K3C mechanism inference

The K3C preregistration explicitly required:

`BASE replication gate = PASS`

before any mechanistic promotion.

Because:

`BASE replication gate = NOT_ESTABLISHED`

the frozen promotion state is:

`mechanism_authorized_for_promotion = false`

Therefore the final K3C scientific verdict is mechanically:

`INCONCLUSIVE_DUE_TO_BASELINE_REPLICATION_FAILURE`

This verdict dominates all downstream mechanism statistics.

The K3C W-vs-H successor hypothesis is neither confirmatorily supported nor confirmatorily contradicted by this execution.

## 8. Non-promotional mechanism statistics

The frozen execution still computed the preregistered mechanism tests, but these values are scientifically gated and may be used only as descriptive/hypothesis-generating information.

They must not be promoted to a K3C causal verdict.

### R_DOM

- positive = 77
- negative = 69
- zero = 4
- rank-biserial effect = +0.0547945205479452
- raw p = 0.5625209423048506
- Holm p = 0.5625209423048506
- nominal direction match = NO
- nominal contradiction = NO

### R_CARRY

- positive = 43
- negative = 103
- zero = 4
- rank-biserial effect = -0.410958904109589
- raw p = 7.396940204558083e-07
- Holm p = 3.698470102279042e-06
- nominal direction match = NO
- nominal contradiction = YES

### D_DOM

- positive = 95
- negative = 51
- zero = 4
- rank-biserial effect = +0.3013698630136986
- raw p = 0.0003394484300037497
- Holm p = 0.0013577937200149987
- nominal direction match = YES

### D_CARRY

- positive = 63
- negative = 83
- zero = 4
- rank-biserial effect = -0.136986301369863
- raw p = 0.11553971025192666
- Holm p = 0.23107942050385333
- nominal direction match = NO
- nominal contradiction = NO

### DISP_DOM

- positive = 118
- negative = 28
- zero = 4
- rank-biserial effect = +0.6164383561643836
- raw p = 2.4024726949774936e-14
- Holm p = 1.6817308864842456e-13
- nominal direction match = YES

### DISP_CARRY

- positive = 40
- negative = 106
- zero = 4
- rank-biserial effect = -0.4520547945205479
- raw p = 4.448054793188397e-08
- Holm p = 2.6688328759130383e-07
- nominal direction match = NO
- nominal contradiction = YES

### P_DOM

- positive = 131
- negative = 15
- zero = 4
- rank-biserial effect = +0.7945205479452054
- raw p = 2.679769781117354e-24
- Holm p = 2.143815824893883e-23
- nominal direction match = YES

### P_CARRY

- positive = 94
- negative = 52
- zero = 4
- rank-biserial effect = +0.2876712328767123
- raw p = 0.0006398230663331769
- Holm p = 0.0019194691989995306
- nominal direction match = YES

Nominal direction-matched mechanism tests:

`["D_DOM", "DISP_DOM", "P_DOM", "P_CARRY"]`

Nominal directional-contradiction mechanism tests:

`["R_CARRY", "DISP_CARRY"]`

These lists are not authorized as confirmatory causal conclusions because the BASE gate failed.

## 9. Scientific interpretation

K3C establishes four distinct facts that must remain separate.

### 9.1 Code and execution correctness

The frozen harness executed successfully.

All required runtime, population, model, recurrence, and intervention integrity gates passed.

### 9.2 Artifact and provenance validity

Independent post-run validation recomputed the BASE and mechanism statistics from block-level artifacts, reconstructed the final verdict, and validated artifact hash bindings.

Artifact/provenance state:

`VALID`

### 9.3 New-population observational replication

The previously replicated K2R trajectory signature did not fully generalize to the new claim-disjoint K3C population.

D, DISP, and P replicated.

R did not.

Therefore:

`FULL_FOUR_ENDPOINT_TRAJECTORY_REPLICATION = NO`

This weakens the breadth of the K2R observational claim.

It does not invalidate K2R on its original prospective replication population.

The correct scope is now:

- D/DISP/P show stronger cross-claim stability within this controlled-generator family;
- R does not presently show the same cross-claim stability;
- the full R/D/DISP/P signature is not established as generator-family-wide.

### 9.4 W-vs-H mechanism

Because the observational BASE signature required for causal interpretation failed prospectively, K3C does not establish whether direct W injection is the dominant source of the intended four-endpoint geometry, nor whether H carries W-seeded divergence in the preregistered manner.

The nominal mechanism pattern is heterogeneous:

- D, DISP, and P show positive DOM;
- R does not;
- P shows positive CARRY;
- R and DISP show significant negative CARRY;
- D CARRY is not established.

This heterogeneity is scientifically interesting but remains hypothesis-generating under the frozen K3C authority.

## 10. Relation to K3

K3 remains closed with:

`LAYER23_NATIVE_RECURRENCE_COMPONENT_SPECIALIZATION_CONTRADICTED`

K3C was designed as a new successor experiment rather than a K3 rescue.

K3C does not reverse the K3 contradiction.

K3C also does not confirm the successor W-injection / retained-carry hypothesis because the required new-population BASE replication gate failed.

Therefore the mechanistic state after K3C is:

- K3 W-local / G-net coefficient specialization: contradicted;
- K3C W-vs-H source/carry successor hypothesis: inconclusive;
- no validated replacement mechanism has yet been established.

## 11. Anti-rescue closure

The one authorized K3C scientific execution has been consumed.

No K3C rerun is authorized.

Do not rescue the result by:

- dropping R;
- redefining the BASE gate as three-of-four;
- promoting D/DISP/P mechanism tests despite the gate;
- dropping negative CARRY tests;
- changing the population;
- selecting a different generated-template slice after outcome inspection;
- changing layer 23;
- changing W=8;
- changing d;
- changing H;
- changing the midpoint intervention;
- altering Holm families;
- interpreting raw p-values as corrected success;
- treating nominal mechanism contradiction as the official K3C verdict.

A future experiment may use the present outcome to generate a new question, but it must be separately preregistered and use prospectively justified evidence.

## 12. Artifact identities

`SHA256SUMS.txt`

`914a83c21a781f3d119559a53fcd2c4fb6225f70db8ecf517bd30a821a271063`

`base_replication_stats.json`

`3b6e19f0cb297153f61cf34d49753c6477ab9261f68366725f5175f3ff2c4584`

`block_metrics.jsonl`

`6d01a0dbe275668c096b65fc8875f7a00f37a1ced32fe1a35d3446fa83a6ea97`

`candidate_pool.jsonl`

`9603f6b20ba870807c151bb70df4c42b0957ded578729b133a37fee8aa1da83e`

`generated_source.jsonl`

`33bff5a0b657d1ceb38ae9c651e1cadfc8308286398cc1b8c4245c47f1c42000`

`integrity.json`

`0bdf360ea1114ef518cef32b14a5a018077a43d583c889af10691558aca3745d`

`item_metrics.jsonl`

`c9c227bd3a7e208cce2d5ab0edc0c1eaf37ddcd1053a5f228ffb823ae4aab50e`

`manifest.json`

`1ef874e8bb868666c1ac888337b3ed200260133cc555c9b5a001e3896f460c0a`

`primary_stats.json`

`07d29ba87dd4aacdbf6cabadbabc202540e670da4d889a37e8c6e8cfb69460fe`

`reciprocal_mapping.json`

`943154f5973b496e3eaa6d10bb6569568ce21d407d410d6bf1a8ba48eb24bfdb`

`report.md`

`a9700f6302989ac5b386cec13cdf86cb5e15bb08356c00baee88dbc7b104ea8e`

## 13. Closure state

`K3C_VALID_EXECUTION = YES`

`K3C_ARTIFACT_PROVENANCE_VALID = YES`

`K3C_SCIENTIFIC_INTEGRITY = PASS_EXACT`

`K3C_BASE_R_REPLICATED = NO`

`K3C_BASE_D_REPLICATED = YES`

`K3C_BASE_DISP_REPLICATED = YES`

`K3C_BASE_P_REPLICATED = YES`

`K3C_BASE_FULL_FOUR_ENDPOINT_REPLICATION = NO`

`K3C_BASE_REPLICATION_VERDICT = NOT_ESTABLISHED`

`K3C_MECHANISM_AUTHORIZED_FOR_PROMOTION = NO`

`K3C_FULL_SUPPORT = NO`

`K3C_SCIENTIFIC_VERDICT = INCONCLUSIVE_DUE_TO_BASELINE_REPLICATION_FAILURE`

`K3C_SUCCESSOR_MECHANISM_SUPPORTED = NO`

`K3C_SUCCESSOR_MECHANISM_CONTRADICTED = NO`

`K3C_RERUN_AUTHORIZED = NO`

`K4_EXECUTION_AUTHORIZED = NO`

K3C scientific execution is closed.

Any next scientific step must be separately justified and must preserve this baseline-replication failure rather than rescue it.
