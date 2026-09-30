# ContraMamba Gen5 Phase 1B — R22 Restoration Confirmation Implementation Authority

## 0. Status

PHASE =
`GEN5_PHASE1B_R22_RESTORATION_CONFIRMATION_IMPLEMENTATION`

STATUS =
`FROZEN_ON_COMMIT`

IMPLEMENTATION_ALLOWED_AFTER_FREEZE =
`YES_BOUNDED`

SCIENTIFIC_EXECUTION_ALLOWED =
`NO`

KAGGLE_ALLOWED =
`NO`

MODEL_FORWARD_ALLOWED =
`NO`

TRAINING_ALLOWED =
`NO`

BACKWARD_ALLOWED =
`NO`

README_UPDATE_REQUIRED =
`NO`

FUTURE_EXECUTION_TOPOLOGY =
`EXACT_TWO_T4_INDEPENDENT_SHARDS`

This authority permits only implementation and zero-forward verification of the
already-frozen Phase 1B restoration-confirmation protocol.

It does not authorize restoration execution.

---

## 1. Parent scientific design

PHASE1B_DESIGN_COMMIT =
`c8dc7a4bb69dd4e86f4bbbcf48b88940bc007cd8`

PHASE1B_DESIGN_PATH =
`reports/reason_router_gen5_phase1b_native_update_role_bridge_spec_candidate.md`

The implementation must preserve Sections 16-18 of that design exactly.

The restoration question is already frozen and is not redesigned here.

---

## 2. Necessity prerequisite

NECESSITY_EVIDENCE_FREEZE_COMMIT =
`4f7dd3a9e0ca2606a7aee8e204e416f32adc884e`

NECESSITY_ARTIFACT_ROOT =
`reports/reason_router_gen5_phase1b_r22_local_necessity_full_cuda_437187c_retry2`

NECESSITY_RESULT =
`PASS_GEN5_PHASE1B_R22_LOCAL_NECESSITY_CONFIRMATION`

NECESSITY_DECISION =
`GEN5_R22_LOCAL_NECESSITY_OVER_MATCHED_C22_CONTROL_SUPPORTED`

NECESSITY_SUMMARY_SHA256 =
`053405f6770a67e7f312fa5a192783bdfdc8bbde332d85491367d670054b6d0d`

NECESSITY_ITEMS_SHA256 =
`a62e4fd8030a86c06f93268dcc862525bb188ed6b1276e39665e362bec557ea9`

NECESSITY_MANIFEST_SHA256 =
`5f55cd3cc4cf898a62a91fe93e908631971827c4b9e4ae163d128b6dac173dd9`

The implementation must fail closed unless this exact positive necessity
evidence is present.

Restoration is not a rescue path for failed necessity.

---

## 3. Frozen restoration population

DATA_ROOT =
`data/reason_router_gen5_phase1b_xg1_restoration_confirmation_v1`

PAIR_RANGE =
`xg1_fact_8401..xg1_fact_8700`

PAIR_COUNT =
`300`

CHECKSUMS_SHA256 =
`ca8dd49b71235e3c876168b46ec97b7071fddabeb1125fd2d833901f8c5c40dd`

STRUCTURED_SOURCE_FACTS_SHA256 =
`a2681a1fe7a76ffa7ba42c08bbe9809bc8e751d282e4909fb84ee65c49bb0245`

SIX_CELL_ROWS_SHA256 =
`8d601c44a25f7733d3613e2ce361fe1add467ae84bd7db7ce41c511802590de2`

STRUCTURAL_MANIFEST_SHA256 =
`a443fc4a04f05bb7ce1630b6b6a56d08126d86ffcc6d236d941e0af1a31dfeb4`

TOKENIZER_ANCHOR_MANIFEST_SHA256 =
`c1529631d7c82d88815a3d402858aa3d439ecea6860a523c6844c1ef38008a7d`

TOKENIZER_ELIGIBILITY_SUMMARY_SHA256 =
`22618f92edbba8fe138b3f8aa7c6ed66430bc15a8871547044af66989c0594f1`

No pair may be removed, reordered, replaced, filtered, or selected from
responses.

---

## 4. Frozen R22/C22 objects

R22_PATH =
`reports/reason_router_gen5_phase1b_r22_c22_construction_c9eca38_v1/r22_basis.f64le`

R22_SHA256 =
`a69232900e8b5a91ec5248e36facee4d421eabd502829e6abf3d2719fdb02214`

C22_PATH =
`reports/reason_router_gen5_phase1b_r22_c22_construction_c9eca38_v1/c22_basis.f64le`

C22_SHA256 =
`c692d39a7387e32e9bf76fe4db7ce2d30d5f7d22a4389e8af363a80b394155c4`

RANK =
`2`

STATE_WIDTH =
`24576`

No reconstruction, refit, reranking, sign change, rank change, or alternate
basis selection is permitted.

---

## 5. Frozen upstream PP3 semantics

PP3_PLUS_SHA256 =
`66ad0cd0f931b0aff88bfc8afc4e9ffa9e355d32f054d376acebb97b469a3cff`

PP3_MINUS_SHA256 =
`ea2c997de7dbaea4edad8b1f30cdac31b4a0c76db0f0df2983900f28a2529ce7`

FROZEN_PP3_SEMANTIC_SOURCE =
`scripts/reason_router_gen4_pp3_necessity_fast_cuda.py`

FROZEN_PP3_SEMANTIC_SOURCE_BLOB =
`26ca67ad8603799a849c151a39728368227326df`

For each layer-17 target coordinate, PP3 coefficients are computed from the
native pre-intervention strong-channel vector.

The frozen PP3 neutralization correction is constructed before the signed
XG2/XG4 probe is added.

The implementation must reuse this ordering exactly.

---

## 6. Frozen model and tokenizer identity

REPRESENTATIVE_CHECKPOINT_SHA256 =
`1ff3fcf2ebd754ab6f9483d6a9982b9b04b9a4eb3357f9f8cdbe2b30399e7d2f`

NATIVE_BACKBONE_SIGNATURE_SHA256 =
`81cd368d8a94932561e0ccd50f45a7db1f27941c00b3a08c8b816badaf25f415`

CANONICAL_MODEL_TOKENIZER_REVISION =
`40e5d2bd7452abb3ca8fadbafe9131ee0e2c2f37`

MODEL_CONFIG_SHA256 =
`784825b6b6cdde47a1602278db0e66d6764169af4a8e662404701bc636a2686a`

TOKENIZER_JSON_SHA256 =
`b074ad869d4f45d1265ca5c9814f78604f3d7e187acc063b15dd232b27585fcf`

TOKENIZER_CONFIG_SHA256 =
`9d7016c33747c6309346e59bd7bf63bfc33c9d9366ecb7e514b3b84dc6b46acb`

SPECIAL_TOKENS_MAP_SHA256 =
`57491904f8680d4b52ed440f1f7ba48cad1c31ecf3eb453b03484e6ff4723ae8`

No substitution is permitted.

---

## 7. Qualified layer-22 backend

QUALIFIED_BACKEND =
`FROZEN_MAMBA_SSM_KERNEL_CAPTURE_PLUS_LAYER22_STATE_REPLAY`

QUALIFIED_BACKEND_SOURCE =
`scripts/reason_router_gen5_phase1b_q22_cuda_equivalence.py`

QUALIFIED_BACKEND_SOURCE_BLOB =
`421d30f00cf71690ed41c983ccf0540808e1de1c`

CUDA_EQUIVALENCE_ARTIFACT_FREEZE_COMMIT =
`96f8a9a8385d71175db6c0d52a86f16c5ea75040`

MAMBA_BINARY_SHA256 =
`dc4d76a6323b510e77cfb66b5aa7bb0086c8f5cba238002b9c20bc31ea706587`

CAUSAL_CONV1D_BINARY_SHA256 =
`6b013d7b9a033bb9b0a2a714b26470e1aaba4af9bf1b3ec7442c2a53afb6b7b6`

The implementation must import and reuse the qualified backend.

No hand-written recurrence replacement is permitted.

---

## 8. Restoration semantics

For every item, direction, orientation, and TP/TM branch, obtain a paired
native donor under the same input and signed XG2/XG4 probe.

The donor differs from the restoration background only in that upstream PP3
neutralization is absent.

Let:

`w22_native = native donor WRITE22`

`a_native = R22^T w22_native`

Let:

`w22_B = WRITE22 under PP3_NEUTRALIZED_AT_LAYER17`

The three restoration conditions are:

`B  = w22_B`

`RR = w22_B + R22 a_native`

`RC = w22_B + C22 a_native`

The exact same donor coefficient vector `a_native` must be used for RR and RC.

Mandatory matched-addition identity:

`||R22 a_native||_2 = ||C22 a_native||_2`

within the frozen numerical tolerance.

No coefficient fitting is permitted on the restoration population.

---

## 9. Frozen endpoint

For each item compute:

`Q_B`

`Q_RR`

`Q_RC`

Then:

`S_R = Q_RR - Q_B`

`S_C = Q_RC - Q_B`

`D_SUF22 = Q_RR - Q_RC`

The broad endpoint remains:

`Q = E_XG2_22 - E_XG4_22`

using exactly five XG2 and five XG4 directions, epsilon `0.025`, both
orientations, and TP/TM branches.

No alternative endpoint is permitted.

---

## 10. Confirmatory rule

Exactly one future confirmatory p-value is authorized.

Positive restoration requires all of:

1. all provenance and manipulation gates pass;
2. `mean(Q_RR) > 0`;
3. `mean(S_R) > 0`;
4. `mean(D_SUF22) > 0`;
5. one-sided one-sample Student t-test on `D_SUF22` gives `p < 0.05`.

Positive label:

`GEN5_R22_RESTORATION_OVER_MATCHED_C22_REPLACEMENT_SUPPORTED`

Otherwise:

`GEN5_R22_RESTORATION_NOT_ESTABLISHED`

No subgroup confirmatory test, alternative endpoint, alternative threshold, or
additional p-value is permitted.

During implementation and verification, real restoration-population p-values
are forbidden.

---

## 11. Exact future two-GPU topology

Future scientific execution must use exactly two Tesla T4 workers.

The coordinator must not modify the frozen backend's logical `cuda:0`
assumption.

Instead:

- worker 0 executes with physical GPU 0 exposed as its sole logical `cuda:0`;
- worker 1 executes with physical GPU 1 exposed as its sole logical `cuda:0`;
- workers are independent processes;
- no DDP gradient synchronization;
- no model-parameter sharing;
- no scientific cross-device reduction during model execution.

Deterministic shard assignment:

`GPU0 = xg1_fact_8401..xg1_fact_8550`

`GPU1 = xg1_fact_8551..xg1_fact_8700`

Each shard contains exactly 150 pairs.

The coordinator must merge scalar item outputs back into canonical order
`xg1_fact_8401..xg1_fact_8700`.

No confirmatory decision or p-value may be computed per shard.

---

## 12. Exact future forward budget

Per pair:

- native donor: 40 model forwards;
- B: 40 model forwards;
- RR: 40 model forwards;
- RC: 40 model forwards.

Therefore:

`FORWARDS_PER_PAIR = 160`

`PAIR_COUNT = 300`

`FULL_MODEL_FORWARD_BUDGET = 48000`

`GPU0_MODEL_FORWARD_BUDGET = 24000`

`GPU1_MODEL_FORWARD_BUDGET = 24000`

`CPU_SCIENTIFIC_MODEL_FORWARD_BUDGET = 0`

A future confirmatory run is valid only if both shards finish and the merged
budget is exactly 48,000 forwards.

Partial-shard results must not receive a confirmatory p-value.

---

## 13. Mandatory implementation gates

The implementation must fail closed on at least:

- branch/head/authority mismatch;
- positive necessity evidence mismatch;
- restoration cohort identity mismatch;
- checkpoint/model/tokenizer mismatch;
- PP3 basis mismatch;
- R22/C22 mismatch;
- qualified backend source mismatch;
- target-token mismatch;
- donor/background input-coordinate mismatch;
- donor direction/orientation/branch mismatch;
- PP3 neutralization ordering mismatch;
- donor coefficient mismatch across RR and RC;
- matched-addition norm inequality;
- layer-22 site mismatch;
- nonfinite Q/J/F/PE values;
- duplicate or missing pair IDs;
- shard overlap;
- shard order mismatch;
- per-shard or global forward-budget mismatch;
- parameter mutation;
- training/backward/task-head optimization;
- logits use;
- response-guided row dropping.

---

## 14. Artifact boundary

The future completed run may persist only:

`r22_restoration_items.jsonl`

`r22_restoration_summary.json`

`artifact_manifest.json`

`SHA256SUMS.txt`

Raw full WRITE22 or POST_STATE22 vectors must not be persisted.

Scalar hashes, norms, residuals, direction-level scalar diagnostics, shard
identities, and pair-level endpoints are allowed.

---

## 15. Allowed implementation files

Exactly these new files may be created:

`scripts/reason_router_gen5_phase1b_r22_restoration_confirmation.py`

`scripts/reason_router_gen5_phase1b_r22_restoration_fast_cuda_2gpu.py`

`scripts/verify_reason_router_gen5_phase1b_r22_restoration_confirmation.py`

`tests/test_reason_router_gen5_phase1b_r22_restoration_confirmation.py`

Existing production files must not be modified.

README must not be modified.

---

## 16. Required zero-forward verification

Before implementation freeze:

1. `git diff --check`;
2. narrow pytest;
3. independent static verifier;
4. exact parent-design identity;
5. exact necessity-evidence identity;
6. exact restoration-cohort identity;
7. exact PP3/R22/C22 identities;
8. synthetic donor/RR/RC algebra tests;
9. synthetic two-shard partition/merge tests;
10. synthetic exact 48,000-forward accounting test;
11. model loaded = false;
12. checkpoint loaded = false;
13. model forward count = 0;
14. CUDA scientific execution = false;
15. scientific p-value count = 0.

Successful verification establishes only:

`PASS_READY_FOR_GEN5_PHASE1B_RESTORATION_EXECUTION_AUTHORITY`

It does not authorize execution.

---

## 17. Future execution gate

After implementation freeze, a separate minimal execution authority must
authorize:

1. a non-confirmatory two-GPU topology/equivalence gate on construction
   population pairs only;
2. only if that gate passes, one full restoration-confirmation run on
   `xg1_fact_8401..8700`;
3. collect/import before scientific interpretation.

The restoration population must remain scientifically unread until that future
execution authority is frozen.

---

## 18. Advancement

If restoration is validated positive, and the already-frozen necessity result
remains positive, the Phase 1B bridge label is:

`GEN5_LAYER17_CAUSAL_ROLE_TO_LAYER22_NATIVE_WRITE_REALIZATION_BRIDGE_SUPPORTED`

Only then may Phase 2 state-update ownership implementation design begin.

If restoration is negative:

`GEN5_R22_NECESSITY_SUPPORTED_BUT_MEDIATION_RESTORATION_NOT_ESTABLISHED`

and Phase 2 ownership implementation is not authorized.

STATUS =
`FROZEN_ON_COMMIT`