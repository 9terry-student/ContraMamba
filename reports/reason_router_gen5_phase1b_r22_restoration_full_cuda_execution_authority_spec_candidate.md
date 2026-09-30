# ContraMamba Gen5 Phase 1B — R22 Restoration Confirmation Full CUDA Execution Authority

## 0. Status

PHASE =
`GEN5_PHASE1B_R22_RESTORATION_CONFIRMATION_EXECUTION`

STATUS =
`FROZEN_ON_COMMIT`

IMPLEMENTATION_ALLOWED =
`NO`

SCIENTIFIC_EXECUTION_ALLOWED_AFTER_FREEZE =
`YES_EXACTLY_ONE_CONFIRMATORY_RESTORATION_RUN`

KAGGLE_ALLOWED_AFTER_FREEZE =
`YES_EXACTLY_FOR_THIS_RUN`

TRAINING_ALLOWED =
`NO`

BACKWARD_ALLOWED =
`NO`

TASK_EVALUATION_ALLOWED =
`NO`

README_UPDATE_REQUIRED =
`NO`

This authority permits exactly one prospective restoration-confirmation run
using the already frozen Phase 1B restoration implementation and the already
qualified 2×T4 execution topology.

It does not authorize any implementation modification, cohort replacement,
parameter search, retry-by-reuse, or post-hoc rescue.

---

## 1. Immediate prerequisites

RESTORATION_IMPLEMENTATION_FREEZE_COMMIT =
`d8d87aa8891ae1f2bef16ed3f1b174e58f87b978`

TOPOLOGY_GATE_IMPLEMENTATION_FREEZE_COMMIT =
`a54fcaa84159336bd79e6b01397a58dd4be1ac54`

TOPOLOGY_GATE_EXECUTION_AUTHORITY_COMMIT =
`759ea7fbca9c8736806d3ad5f9d989ad09532dbb`

TOPOLOGY_QUALIFICATION_EVIDENCE_FREEZE_COMMIT =
`d8ad2bed114ca22ea0262210e8cfc42d80e6bdcb`

TOPOLOGY_QUALIFICATION_RESULT =
`PASS_GEN5_PHASE1B_RESTORATION_2GPU_TOPOLOGY_EQUIVALENCE`

TOPOLOGY_QUALIFICATION_INTERPRETATION =
`TWO_T4_RESTORATION_EXECUTION_TOPOLOGY_QUALIFIED`

TOPOLOGY_SUMMARY =
`reports/reason_router_gen5_phase1b_r22_restoration_2gpu_topology_equivalence_759ea7f/topology_equivalence_summary.json`

TOPOLOGY_SUMMARY_SHA256 =
`b8a27b82c1ef26e846137dde79646a810890ba8bc3d596890b635472647077d9`

TOPOLOGY_ITEMS_SHA256 =
`7b382786a1fe96e52ddba696d0176940884a172a2a6094161ff602385d5e6323`

TOPOLOGY_MANIFEST_SHA256 =
`4e42782a4e315c960d4e59ad1bdae80912c5f03a60cb16ab75d35308ef8a969e`

Observed topology comparison:

- all discrete equivalence checks PASS;
- all floating equivalence checks PASS;
- max floating absolute difference = 0;
- max floating bound usage ratio = 0;
- confirmatory p-value count = 0;
- restoration-confirmation population loaded = false;
- training/backward = false.

The topology gate itself is not restoration evidence.

---

## 2. Frozen scientific parent

PHASE1B_DESIGN_COMMIT =
`c8dc7a4bb69dd4e86f4bbbcf48b88940bc007cd8`

PHASE1B_DESIGN =
`reports/reason_router_gen5_phase1b_native_update_role_bridge_spec_candidate.md`

The scientific question is whether the already validated upstream layer-17 PP3
causal role can be restored through the prospectively constructed layer-22
native-write realization R22 more strongly than through the matched response-
blind C22 replacement control.

No scientific semantics may be changed by this authority.

---

## 3. Necessity prerequisite

NECESSITY_EVIDENCE_FREEZE_COMMIT =
`4f7dd3a9e0ca2606a7aee8e204e416f32adc884e`

NECESSITY_RESULT =
`PASS_GEN5_PHASE1B_R22_LOCAL_NECESSITY_CONFIRMATION`

NECESSITY_SCIENTIFIC_CONCLUSION =
`GEN5_R22_LOCAL_NECESSITY_OVER_MATCHED_C22_CONTROL_SUPPORTED`

NECESSITY_PAIR_RANGE =
`xg1_fact_8101..xg1_fact_8400`

NECESSITY_CUDA_SCIENTIFIC_MODEL_FORWARD_COUNT =
`36000`

NECESSITY_CONFIRMATORY_P_VALUE_COUNT =
`1`

The restoration run is authorized only because the frozen necessity stage is
already positive.

---

## 4. Frozen implementation identities

RESTORATION_SEMANTICS =
`scripts/reason_router_gen5_phase1b_r22_restoration_confirmation.py`

RESTORATION_SEMANTICS_GIT_BLOB =
`786f186ab89ffc44bb5373b70df7a1041165b7a5`

RESTORATION_CUDA_RUNNER =
`scripts/reason_router_gen5_phase1b_r22_restoration_fast_cuda_2gpu.py`

RESTORATION_CUDA_RUNNER_GIT_BLOB =
`ed73904fdf9a555e11d30e3b9068ac12941225cd`

RESTORATION_ZERO_FORWARD_VERIFIER =
`scripts/verify_reason_router_gen5_phase1b_r22_restoration_confirmation.py`

RESTORATION_ZERO_FORWARD_VERIFIER_GIT_BLOB =
`e50888814d0b489ffb37f0e97c89a102e4709f41`

QUALIFIED_Q22_BACKEND =
`scripts/reason_router_gen5_phase1b_q22_cuda_equivalence.py`

QUALIFIED_Q22_BACKEND_GIT_BLOB =
`421d30f00cf71690ed41c983ccf0540808e1de1c`

PP3_CUDA_SOURCE =
`scripts/reason_router_gen4_pp3_necessity_fast_cuda.py`

PP3_CUDA_SOURCE_GIT_BLOB =
`26ca67ad8603799a849c151a39728368227326df`

No frozen implementation source may be modified before or during execution.

EXECUTION_HEAD =
`THIS_EXECUTION_AUTHORITY_FREEZE_COMMIT`

No descendant commit may substitute for the execution head without a new
execution authority.

---

## 5. Frozen restoration population

DATA_ROOT =
`data/reason_router_gen5_phase1b_xg1_restoration_confirmation_v1`

PAIR_FIRST =
`xg1_fact_8401`

PAIR_LAST =
`xg1_fact_8700`

PAIR_COUNT =
`300`

ROWS_PER_PAIR =
`6`

ROW_COUNT =
`1800`

No alternate pair, extra pair, omitted pair, replacement pair, filtering, or
reordering is authorized.

Frozen static identities:

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

---

## 6. Frozen checkpoint and external snapshot

REPRESENTATIVE_CHECKPOINT =
`reports/reason_router_gen3_grouped_factorial_runs/seed180/G3-GROUP-D-HALF/selected_checkpoint.pt`

REPRESENTATIVE_CHECKPOINT_SHA256 =
`1ff3fcf2ebd754ab6f9483d6a9982b9b04b9a4eb3357f9f8cdbe2b30399e7d2f`

REPRESENTATIVE_CHECKPOINT_BYTES =
`518270455`

NATIVE_BACKBONE_SIGNATURE_SHA256 =
`81cd368d8a94932561e0ccd50f45a7db1f27941c00b3a08c8b816badaf25f415`

HF_MODEL_FAMILY =
`state-spaces/mamba-130m-hf`

FROZEN_MODEL_TOKENIZER_REVISION =
`40e5d2bd7452abb3ca8fadbafe9131ee0e2c2f37`

MODEL_CONFIG_SHA256 =
`784825b6b6cdde47a1602278db0e66d6764169af4a8e662404701bc636a2686a`

TOKENIZER_JSON_SHA256 =
`b074ad869d4f45d1265ca5c9814f78604f3d7e187acc063b15dd232b27585fcf`

TOKENIZER_CONFIG_SHA256 =
`9d7016c33747c6309346e59bd7bf63bfc33c9d9366ecb7e514b3b84dc6b46acb`

SPECIAL_TOKENS_MAP_SHA256 =
`57491904f8680d4b52ed440f1f7ba48cad1c31ecf3eb453b03484e6ff4723ae8`

The same previously validated frozen release assets may be provisioned again
outside the repository worktree after Kaggle bootstrap.

No alternate checkpoint or mutable latest snapshot is permitted.

---

## 7. Frozen R22/C22 identities

R22_C22_FREEZE_COMMIT =
`1d3542013934870aa9181d1bbaf565ff4724112c`

R22_SHA256 =
`a69232900e8b5a91ec5248e36facee4d421eabd502829e6abf3d2719fdb02214`

C22_SHA256 =
`c692d39a7387e32e9bf76fe4db7ce2d30d5f7d22a4389e8af363a80b394155c4`

RANK_R22 =
`2`

RANK_C22 =
`2`

R22 and C22 may not be reconstructed, refit, rotated, replaced, or selected
using restoration responses.

---

## 8. Frozen restoration semantics

All items begin from:

`PP3_NEUTRALIZED_AT_LAYER17`

For each same-item/same-branch native donor:

`a_native = R22^T w22_native`

Background:

`B = w22_B`

R22 restoration:

`RR = w22_B + R22 a_native`

Matched C22 replacement:

`RC = w22_B + C22 a_native`

The exact same R22-derived coefficient vector `a_native` is used for RR and RC.

RR/RC addition norms must match within the frozen implementation tolerance.

No coefficient fitting on the restoration cohort is allowed.

---

## 9. Frozen scientific endpoint

For each item:

`Q_B`

`Q_RR`

`Q_RC`

with unchanged:

`Q = E_XG2_22 - E_XG4_22`

Derived endpoints:

`S_R = Q_RR - Q_B`

`S_C = Q_RC - Q_B`

Primary confirmatory contrast:

`D_SUF22 = Q_RR - Q_RC`

Exactly one inferential test is authorized:

`one-sided one-sample Student t-test(D_SUF22, alternative > 0)`

No other confirmatory p-value is authorized.

---

## 10. Positive restoration rule

Positive restoration requires all implementation/provenance/manipulation gates
plus all four numerical conditions:

1. `mean(Q_RR) > 0`
2. `mean(S_R) > 0`
3. `mean(D_SUF22) > 0`
4. `p_one_sided_greater < 0.05`

Positive label:

`GEN5_R22_RESTORATION_OVER_MATCHED_C22_REPLACEMENT_SUPPORTED`

Otherwise:

`GEN5_R22_RESTORATION_NOT_ESTABLISHED`

The result bundle itself must report:

`PASS_GEN5_PHASE1B_R22_RESTORATION_CONFIRMATION`

for a completed valid confirmatory execution regardless of whether the scientific
label is supported or not established.

---

## 11. Bridge interpretation rule

If restoration is positive, combine it only with the already frozen positive
necessity result to establish the bounded Phase 1B conclusion:

`GEN5_LAYER17_CAUSAL_ROLE_TO_LAYER22_NATIVE_WRITE_REALIZATION_BRIDGE_SUPPORTED`

Only that B3 result permits the next design phase:

`GEN5_PHASE2_STATE_UPDATE_OWNERSHIP_IMPLEMENTATION_DESIGN`

If restoration is not established while necessity remains positive, the bounded
conclusion is:

`GEN5_R22_NECESSITY_SUPPORTED_BUT_MEDIATION_RESTORATION_NOT_ESTABLISHED`

and Phase 2 state-update ownership implementation is not authorized.

No higher-rank rescue, cohort substitution, or parameter tuning is permitted
under this authority.

---

## 12. Exact 2×T4 topology

This run inherits the already validated topology.

Physical GPU 0 worker:

- pairs `xg1_fact_8401..xg1_fact_8550`
- 150 pairs
- logical device `cuda:0` under `CUDA_VISIBLE_DEVICES=0`
- exact 24,000 scientific model forwards

Physical GPU 1 worker:

- pairs `xg1_fact_8551..xg1_fact_8700`
- 150 pairs
- logical device `cuda:0` under `CUDA_VISIBLE_DEVICES=1`
- exact 24,000 scientific model forwards

Coordinator:

- exactly two physical Tesla T4 devices
- launches both workers
- merges canonical pair order `8401..8700`
- performs the single confirmatory decision after both workers complete

No DDP, gradient synchronization, shared live model object, one-GPU fallback, or
CPU scientific fallback is authorized.

---

## 13. Exact forward accounting

FORWARDS_PER_PAIR =
`160`

GPU0_FORWARD_BUDGET =
`24000`

GPU1_FORWARD_BUDGET =
`24000`

FULL_CUDA_SCIENTIFIC_MODEL_FORWARD_BUDGET =
`48000`

CPU_SCIENTIFIC_MODEL_FORWARD_BUDGET =
`0`

CHECKPOINT_LOAD_COUNT =
`2`

Any over-budget, under-budget, partial, interrupted, or reordered run is not a
valid confirmatory result.

---

## 14. Frozen CUDA runtime

Expected runtime:

- Python `3.12.13`
- NumPy `2.0.2`
- Torch `2.10.0+cu128`
- Transformers `5.0.0`
- tokenizers `0.22.2`
- kernels `0.10.2`
- CUDA `12.8`
- exactly two `Tesla T4`
- compute capability `(7,5)` on both devices

Frozen kernel scientific revisions:

MAMBA_KERNEL_REVISION =
`c8ffc584c147878a6eb978ae0e8db4d116c93a8c`

CAUSAL_CONV1D_KERNEL_REVISION =
`f2651e776f66069cdcf842840db637583def1223`

Frozen loaded binary identities:

MAMBA_BINARY_SHA256 =
`dc4d76a6323b510e77cfb66b5aa7bb0086c8f5cba238002b9c20bc31ea706587`

CAUSAL_CONV1D_BINARY_SHA256 =
`6b013d7b9a033bb9b0a2a714b26470e1aaba4af9bf1b3ec7442c2a53afb6b7b6`

Runtime or binary mismatch blocks execution.

---

## 15. Artifact boundary

The exact output directory must contain only:

`r22_restoration_items.jsonl`

`r22_restoration_summary.json`

`artifact_manifest.json`

`SHA256SUMS.txt`

Raw full WRITE22 vectors must not be persisted.

Raw full POST_STATE22 vectors must not be persisted.

The item artifact may contain frozen scalar diagnostics and tensor hashes already
defined by the implementation.

After execution, `cm collect` and local `cm import` are mandatory.

Execution success without validated import is insufficient for scientific
interpretation.

---

## 16. Run identity

RUN_NAME =
`gen5-phase1b-r22-restoration-confirmation`

OUTPUT_DIR =
`reports/reason_router_gen5_phase1b_r22_restoration_full_cuda_<EXECUTION_HEAD_SHORT>`

`<EXECUTION_HEAD_SHORT>` is the first seven hexadecimal characters of the commit
that freezes this execution authority.

The run name must not be reused for a different execution head.

If execution fails before a valid artifact bundle is created, any retry requires
a new run-name suffix such as `-retry1`; it must still use an explicitly
authorized exact head and exact scientific command.

---

## 17. Execution command contract

After this authority is frozen and the exact execution HEAD is known, the top-
level scientific command must invoke:

`python -u -m scripts.reason_router_gen5_phase1b_r22_restoration_fast_cuda_2gpu`

with exactly these semantic arguments:

- `--expected-head <THIS_EXECUTION_AUTHORITY_FREEZE_COMMIT>`
- `--model-snapshot <EXACT_PROVISIONED_FROZEN_MODEL_DIRECTORY>`
- `--tokenizer-snapshot <SAME_EXACT_PROVISIONED_FROZEN_MODEL_DIRECTORY>`
- `--checkpoint <EXACT_PROVISIONED_REPRESENTATIVE_CHECKPOINT>`
- `--output-dir <FROZEN_OUTPUT_DIR>`

The top-level command must not specify `--worker` or `--shard-id`.

The frozen coordinator alone controls process sharding and
`CUDA_VISIBLE_DEVICES`.

The exact command must be registered with:

`cm run save gen5-phase1b-r22-restoration-confirmation`

and executed only through the corresponding pinned `cm run`.

---

## 18. Stop conditions

Stop rather than bypass if:

- execution HEAD differs from the authority-freeze commit;
- worktree is dirty;
- the topology qualification evidence is missing or altered;
- necessity prerequisite evidence is missing or altered;
- restoration static-input hashes mismatch;
- checkpoint or tokenizer/model snapshot hashes mismatch;
- R22/C22 identities mismatch;
- PP3/PP5 identity checks fail;
- runtime differs from the frozen runtime;
- exact kernel binary hashes differ;
- exactly two Tesla T4 devices are unavailable;
- output directory already exists;
- any worker fails;
- any item is dropped;
- any forward count mismatches;
- any parameter signature mutates;
- more or fewer than one confirmatory p-value is computed;
- training, backward, task-head execution, or logits access occurs;
- collect/import provenance or artifact hashes mismatch.

No fallback or repair is authorized inside the scientific run.

---

## 19. Success boundary

A valid imported artifact bundle establishes one of two scientific outcomes.

### Positive restoration

`GEN5_R22_RESTORATION_OVER_MATCHED_C22_REPLACEMENT_SUPPORTED`

Together with the already frozen positive necessity result, this yields:

`GEN5_LAYER17_CAUSAL_ROLE_TO_LAYER22_NATIVE_WRITE_REALIZATION_BRIDGE_SUPPORTED`

and permits a subsequent Phase 2 state-update ownership implementation design
authority.

### Restoration not established

`GEN5_R22_RESTORATION_NOT_ESTABLISHED`

Together with the already frozen positive necessity result, this yields:

`GEN5_R22_NECESSITY_SUPPORTED_BUT_MEDIATION_RESTORATION_NOT_ESTABLISHED`

and does not permit Phase 2 ownership implementation.

No conclusion is valid until collect/import validation succeeds.

STATUS =
`FROZEN_ON_COMMIT`
