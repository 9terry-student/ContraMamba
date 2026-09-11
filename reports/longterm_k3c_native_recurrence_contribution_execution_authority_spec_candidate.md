# K3C Native Recurrence Contribution Scientific Execution Authority Specification Candidate

**Status:** ONE-RUN SCIENTIFIC EXECUTION AUTHORITY CANDIDATE.

**Scope:** authorizes exactly one K3C scientific recurrent-state execution under the frozen preregistration, replay implementation, execution harness, population, runtime, intervention, and statistics contracts stated below.

This document does not amend K3C design.

This document does not reopen K3.

This document does not authorize K4.

## 1. Machine-readable authority markers

`K3C_EXECUTION_AUTHORITY_SCHEMA=k3c-scientific-execution-authority-v1`

`K3C_SCIENTIFIC_RECURRENT_STATE_EXECUTION_AUTHORIZED=YES`

`K3C_ONE_SCIENTIFIC_EXECUTION=YES`

`K3C_EXECUTION_IMPLEMENTATION_COMMIT=6232a601ad762d20abe67473c0f2cf2e510f9849`

`K3C_EXECUTION_RUNNER_SHA256=fefef63a6b49d39ec3dbd34cb5b7413d54e15e6616992a2972e0c343334454b2`

`K3C_EXECUTION_TEST_SHA256=36b9ab89dee06835dd5a9765eb6cf31d5d9ed810005fc36f5c01bc3c1012f53a`

`K3C_REPLAY_IMPLEMENTATION_COMMIT=e91116b4b5837a16de1d6eabfa2503be5bfe1d3d`

`K3C_REPLAY_MODULE_SHA256=ab07c6052a07af354043e0aa38bf24fd73c2fde53fccec125c25f7db2718f233`

`K3C_PREREG_COMMIT=b85272d88d0bb57db45fdc963d313714529e7975`

`K3C_PREREG_SHA256=0a9de28237e107ce3a62d4fa9e0bb7f230d2e0289e019d3864edf491dd271436`

These ten markers are the complete machine-readable authority marker set expected by the frozen K3C execution harness.

## 2. Governing preregistration

Governing K3C preregistration:

`reports/longterm_k3c_native_recurrence_contribution_decomposition_prereg_candidate.md`

Frozen commit:

`b85272d88d0bb57db45fdc963d313714529e7975`

Frozen SHA256:

`0a9de28237e107ce3a62d4fa9e0bb7f230d2e0289e019d3864edf491dd271436`

The preregistration freezes:

- a new claim-disjoint K3C population;
- layer 23;
- W = 8;
- direct write contribution `W_t`;
- full retained contribution `H_t = G_t ⊙ S_(t-1)`;
- BASE;
- W_EQ;
- H_EQ;
- W_SEED_H_CARRY;
- WH_EQ integrity control;
- BASE replication gate with Holm m=4;
- eight mechanism tests with Holm m=8;
- anti-rescue rules;
- no K4 authorization.

This authority does not modify any of those terms.

## 3. Frozen replay implementation

Frozen K3C replay implementation commit:

`e91116b4b5837a16de1d6eabfa2503be5bfe1d3d`

Replay module:

`scripts/longterm_k3c_native_recurrence_contribution_decomposition.py`

Replay module SHA256:

`ab07c6052a07af354043e0aa38bf24fd73c2fde53fccec125c25f7db2718f233`

Replay module Git blob:

`633ca87b365f50ac27dbec5a8595dbdad42a7dfe`

Replay focused test:

`tests/test_longterm_k3c_native_recurrence_contribution_decomposition.py`

Replay focused-test SHA256:

`a1e1eafe80806d62103e2a1445b4264451785099a6ac27385ea95afdb3a4058d`

Replay focused-test Git blob:

`6734cbccf34107cba160ac1da05137deb55bcc52`

The replay implementation passed the combined K3/K3C validation suite and synthetic/state-blind K3C preflight before being frozen.

## 4. Frozen scientific execution harness

Frozen K3C scientific execution harness commit:

`6232a601ad762d20abe67473c0f2cf2e510f9849`

Harness:

`scripts/longterm_k3c_native_recurrence_contribution_execution.py`

Harness SHA256:

`fefef63a6b49d39ec3dbd34cb5b7413d54e15e6616992a2972e0c343334454b2`

Harness test:

`tests/test_longterm_k3c_native_recurrence_contribution_execution.py`

Harness-test SHA256:

`36b9ab89dee06835dd5a9765eb6cf31d5d9ed810005fc36f5c01bc3c1012f53a`

The harness passed the frozen combined suite:

`78 passed`

and demonstrated:

`NO_AUTHORITY_FAIL_CLOSED = PASS`

before this authority was created.

Therefore the scientific path is permitted only when this authority is frozen under the exact commit topology required below.

## 5. Required authority commit topology

This authority file must be added in exactly one commit.

Required path:

`reports/longterm_k3c_native_recurrence_contribution_execution_authority_spec_candidate.md`

The authority commit must:

1. have parent exactly `6232a601ad762d20abe67473c0f2cf2e510f9849`;
2. change exactly this one authority file;
3. leave the frozen harness and tests unchanged;
4. become the exact runtime HEAD for the one authorized scientific execution.

No additional commit may be created after the authority commit and before the scientific execution.

If another commit is created first, this authority/harness topology is invalid and the harness must fail closed.

## 6. Runtime repository state

Required branch:

`longterm-k-series-native-state-kinematics`

At runtime, the only permitted dirty entries are the two historical K1 untracked files:

`scripts/longterm_k1_native_state_kinematics.py`

`tests/test_longterm_k1_native_state_kinematics.py`

No K3C implementation, authority, result, scratch, or unrelated tracked/untracked modification is permitted in the repository at scientific execution time.

## 7. Prospectively frozen K3C population

Generator:

`scripts/build_controlled_v5.py`

Generator SHA256:

`4e9798591fbfffb6d15ea9b2f8cf5cd804a9e76713b9f6b13b354fe6ce93aa5c`

Generator Git blob:

`baee23a9f71333125f4a8735c2c92d20cab7eb4f`

Global deterministic template slice:

`[600:900]`

Pair IDs:

`generated_fact_601` through `generated_fact_900`

Required source-row count:

`3900`

Generated-source canonical SHA256:

`33bff5a0b657d1ceb38ae9c651e1cadfc8308286398cc1b8c4245c47f1c42000`

Candidate count:

`300`

Candidate-pool canonical SHA256:

`9603f6b20ba870807c151bb70df4c42b0957ded578729b133a37fee8aa1da83e`

Reciprocal blocks:

`150`

Reciprocal-mapping canonical SHA256:

`4fbc0f6642db3b2c3cca148fdc03cdc738dd6e8718cffb3fcdee73cd5b7f9acc`

Required correction-source balance:

- none: 150;
- polarity_flip: 150.

Required prior-population overlap:

- K2W pair-ID overlap: 0;
- K2W claim-SHA overlap: 0;
- K2W claim-text overlap: 0;
- K2R/K3 pair-ID overlap: 0;
- K2R/K3 claim-SHA overlap: 0;
- K2R/K3 claim-text overlap: 0.

No filtering, replacement, resampling, weighting, or population rescue is authorized.

## 8. Frozen tokenizer feasibility

Required:

- N_total = 300;
- N_blocks = 150;
- N_matched_valid = 300;
- N_swapped_valid = 300;
- matched d-p counts = {2:150, 3:150};
- swapped d-p counts = {2:150, 3:150};
- matched correction continuation min/max = [23,28];
- matched control continuation min/max = [20,27];
- swapped correction continuation min/max = [23,28];
- swapped control continuation min/max = [20,27];
- prefix marginal preserved;
- correction marginal preserved;
- control marginal preserved.

Any mismatch blocks scientific execution.

## 9. Frozen model and checkpoint identity

Handoff ZIP SHA256:

`96859bad3e400613b4c981990e56aaf35b1d92baaaa930eb11448cf38a63b861`

Checkpoint SHA256:

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

## 10. Frozen Hugging Face and recurrence runtime

HF model:

`state-spaces/mamba-130m-hf`

Exact HF revision:

`5708daa364c50b880e7bd92eab456e0d34492ee9`

Transformers:

`5.12.1`

Required recurrence:

`MambaMixer.slow_forward`

Required Mamba source SHA256:

`23c7b410e204b5da01732566de10c94b70a8418ecb608e409754b00332eb2a41`

Required recurrence update line:

`350`

Required post-update line:

`351`

Required `discrete_A` line:

`322`

Required `deltaB_u` line:

`324`

Scientific device:

`cpu`

State timing:

`post_consumption_s_t`

Fast-kernel/GPU execution is not authorized.

The sequential-fallback warning is expected and does not constitute an execution defect.

## 11. Pre-scientific gates

Before any K3C scientific-population recurrent-state read, the frozen harness must successfully re-establish:

1. exact authority topology;
2. exact authority markers;
3. exact prereg bytes;
4. exact replay implementation bytes/blob;
5. exact replay-test bytes/blob;
6. exact generated-source identity;
7. exact candidate-pool identity;
8. exact reciprocal mapping;
9. exact tokenizer feasibility;
10. exact checkpoint and encoder identity;
11. exact HF revision;
12. Transformers 5.12.1;
13. exact recurrence-source SHA;
14. CPU sequential recurrence;
15. synthetic component-capture noninterference;
16. natural H+W identity;
17. synthetic natural structural replay exactness;
18. synthetic sham replay exactness;
19. W_EQ synthetic semantics;
20. dynamic H_EQ synthetic semantics;
21. W_SEED_H_CARRY synthetic semantics;
22. WH_EQ exact pair-state collapse;
23. known-term source/carry replay exactness;
24. proof that the preflight itself has not read K3C scientific-population recurrent states;
25. proof that the preflight itself has not executed K3C scientific-population interventions.

Any failed gate blocks the scientific run.

## 12. Authorized scientific conditions

Exactly four scientific conditions are authorized:

- BASE;
- W_EQ;
- H_EQ;
- W_SEED_H_CARRY.

Exactly one integrity-only condition is authorized:

- WH_EQ.

No G_EQ scientific endpoint is part of K3C.

No alternate intervention is authorized.

## 13. Structural definitions

At layer 23:

`G_t = discrete_A_t`

`W_t = deltaB_u_t`

`H_t = G_t ⊙ S_(t-1)`

`S_t = H_t + W_t`

K3C distinguishes the retention coefficient G from the full retained contribution H.

## 14. BASE condition

BASE uses natural branch-specific G and W with exact structural replay.

Every scientific natural replay must reproduce captured post-consumption native state bit-for-bit.

Every scientific sham replay must reproduce captured native state bit-for-bit.

## 15. W_EQ condition

For every scientific correction/control pair and every token from d through p+8:

`Wbar_t = 0.5 * (W_t^corr + W_t^ctrl)`

Each branch's retained contribution is dynamically recomputed from its current replay state and natural branch-specific G.

W_EQ removes branch-specific direct-write differences while preserving branch-specific retained dynamics.

## 16. H_EQ condition

For every token from d through p+8:

`H_t^corr = G_t^corr ⊙ S_(t-1)^corr,replay`

`H_t^ctrl = G_t^ctrl ⊙ S_(t-1)^ctrl,replay`

`Hbar_t = 0.5 * (H_t^corr + H_t^ctrl)`

Each branch then adds its own natural W.

H must be recomputed dynamically from current replay state.

Frozen-natural-H substitution is forbidden.

## 17. W_SEED_H_CARRY condition

At d, H is midpoint-equalized and natural branch-specific W remains.

Therefore d-step divergence can be introduced only through W.

From d+1 through p+8, W is midpoint-equalized while each branch dynamically computes H from its current replay state and natural G.

Therefore later branch-specific direct-write differences are removed, and persistence of the W-seeded difference is attributable to the retained-contribution path under this intervention.

## 18. WH_EQ integrity control

From d through p+8, both H and W are pair-midpoint equalized.

Correction and control states must collapse exactly from d onward.

Required scientific count:

`600`

paired WH collapse checks.

Failure blocks validity.

## 19. Required scientific integrity counts

Across 300 items, matched and swapped assignments, and correction/control branches:

Natural exact branch replays:

`1200`

Sham exact branch replays:

`1200`

WH pair-collapse checks:

`600`

W-seed pair semantic checks:

`600`

Any mismatch blocks scientific validity.

## 20. Frozen trajectory endpoints

Exactly four trajectory endpoints are authorized:

- R: mean speed;
- D: mean valid turning;
- DISP: endpoint displacement;
- P: trajectory efficiency.

No new confirmatory endpoint may be added after execution.

## 21. Pair-specificity geometry

For each condition and endpoint, matched and swapped correction/control differences are converted to recipient X and reciprocal-block B using the frozen K2R/K3 geometry.

Expected alignment signs are frozen as:

- R: +1;
- D: +1;
- DISP: -1;
- P: -1.

No direction may be changed after outcome inspection.

## 22. Mandatory new-population BASE replication gate

Before any mechanistic promotion is permitted, the new claim-disjoint K3C BASE population must independently reproduce all four frozen trajectory directions.

Exactly four BASE tests are authorized:

- R;
- D;
- DISP;
- P.

Use:

- 150 reciprocal blocks;
- n_valid >= 120;
- n_eff >= 30;
- two-sided exact sign test;
- Holm m=4;
- alpha=0.05;
- tie order R, D, DISP, P;
- rank-biserial sign effect in the frozen expected direction.

BASE gate PASS requires all four direction matches.

A Holm-significant opposite-direction endpoint makes the BASE gate CONTRADICTED.

Otherwise the BASE gate is NOT_ESTABLISHED.

If BASE is not PASS, final scientific verdict is:

`INCONCLUSIVE_DUE_TO_BASELINE_REPLICATION_FAILURE`

even if downstream mechanism statistics look favorable.

No rescue population or endpoint removal is authorized.

## 23. Mechanism contrasts

For each endpoint q:

`ATT_W = Z_BASE - Z_W_EQ`

`ATT_H = Z_BASE - Z_H_EQ`

`DOM = ATT_W - ATT_H`

Positive DOM means removing branch-specific W differences attenuates aligned geometry more strongly than removing branch-specific full H differences.

Retained-carry statistic:

`CARRY = Z_W_SEED_H_CARRY`

Positive CARRY means W-seeded divergence remains expressed in the frozen replicated direction after all later branch-specific W differences are removed.

## 24. Eight confirmatory mechanism tests

Conditional on BASE gate PASS, exactly eight tests are authorized in this order:

1. R_DOM
2. R_CARRY
3. D_DOM
4. D_CARRY
5. DISP_DOM
6. DISP_CARRY
7. P_DOM
8. P_CARRY

For every test:

- 150 reciprocal blocks;
- n_valid >= 120;
- n_eff >= 30;
- two-sided exact sign test;
- zero values excluded from n_eff;
- undefined values excluded from n_valid and explicitly counted;
- Holm m=8;
- alpha=0.05;
- exact tie order as above;
- positive rank-biserial effect required for direction match.

Raw p-values do not authorize promotion.

## 25. Scientific verdict rule

If BASE gate PASS and all eight mechanism tests direction-match:

`LAYER23_WRITE_INJECTION_WITH_RETAINED_CARRY_CAUSALLY_SUPPORTED`

If BASE gate PASS and any mechanism test is Holm-significant in the negative direction:

`LAYER23_WRITE_INJECTION_WITH_RETAINED_CARRY_CONTRADICTED`

If BASE gate PASS, full support is false, and no mechanism test is significantly negative:

`LAYER23_WRITE_INJECTION_WITH_RETAINED_CARRY_NOT_ESTABLISHED`

If BASE gate is not PASS:

`INCONCLUSIVE_DUE_TO_BASELINE_REPLICATION_FAILURE`

These rules are frozen before outcome inspection.

## 26. Authorized execution count

Exactly one K3C scientific execution is authorized.

The authority is consumed when the authorized harness begins a valid scientific execution attempt under the frozen authority HEAD.

An unfavorable outcome does not authorize a rerun.

A successful run does not authorize a replication run.

A technical failure must be handled under explicit failure-recovery rules; no silent second attempt is permitted.

## 27. Output contract

The output directory must be outside the repository, must not already exist, and must have the exact basename:

`k3c-contribution-<authority-commit-first-12-hex>`

The frozen harness must emit and bind at least:

- generated_source.jsonl;
- candidate_pool.jsonl;
- reciprocal_mapping.json;
- item_metrics.jsonl;
- block_metrics.jsonl;
- base_replication_stats.json;
- primary_stats.json;
- integrity.json;
- report.md;
- manifest.json;
- SHA256SUMS.txt.

The output artifacts must be validated before scientific interpretation or repository archival.

## 28. No-rescue restrictions

This authority does not permit:

- rerunning because the result is unfavorable;
- changing layer 23;
- changing W=8;
- changing d;
- choosing a favorable token coordinate;
- changing the population;
- dropping items;
- dropping zeros;
- dropping DISP or P;
- changing expected signs;
- changing H definition;
- substituting G for H;
- replacing dynamic H with frozen-natural H;
- changing arithmetic midpoint;
- adding G_EQ as a scientific endpoint;
- zeroing/clamping/rescaling;
- changing the BASE family;
- changing the eight-test family;
- changing Holm families;
- promoting raw p-values;
- using secondary diagnostics as rescue evidence;
- training;
- fine-tuning;
- learned probes;
- hyperparameter sweeps;
- K4 execution.

## 29. Scientific claim boundary

Even a fully supported K3C result establishes only a bounded layer-23 structural-replay causal claim under the frozen controlled population and intervention definitions.

It does not establish:

- final task-decision causality;
- authorization/entitlement causality;
- global Mamba mechanism;
- external-distribution generalization;
- natural-corpus generalization;
- confident-error prediction;
- detector utility;
- K4 claims.

## 30. K4 boundary

K4 remains unauthorized.

`K4_EXECUTION_AUTHORIZED = NO`

No K4 implementation or scientific execution may be inferred from K3C execution authority or from any K3C result.

## 31. Current execution state

The frozen K3C preregistration exists.

The frozen K3C replay implementation exists.

The frozen K3C execution harness exists and is fail-closed without this authority.

After this authority is committed under the exact topology above, exactly one K3C scientific recurrent-state execution is authorized.

No other scientific execution is authorized.
