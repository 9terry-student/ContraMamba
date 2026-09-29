# ContraMamba Gen5 Phase 1B — R22/C22 Construction Execution Authority

## 0. Status

PHASE = `GEN5_PHASE1B_R22_C22_CONSTRUCTION_EXECUTION_AUTHORITY`

STATUS = `FROZEN_ON_COMMIT`

SCIENTIFIC_EXECUTION_ALLOWED_AFTER_FREEZE = `YES_CONSTRUCTION_ONLY`

TRAINING = `FORBIDDEN`

BACKWARD = `FORBIDDEN`

TASK_EVALUATION = `FORBIDDEN`

Q_ENDPOINT_COMPUTATION = `FORBIDDEN`

PRIMARY_INFERENCE = `FORBIDDEN`

CUDA_MODEL_EXECUTION = `FORBIDDEN`

README_UPDATE_REQUIRED = `NO`

This authority permits exactly one bounded Gen5 Phase 1B R22/C22 construction
run after this document is frozen. It does not authorize necessity or restoration
confirmation.

---

## 1. Binding lineage

PHASE1B_DESIGN_COMMIT =
`c8dc7a4bb69dd4e86f4bbbcf48b88940bc007cd8`

STATIC_PREPARATION_FREEZE_COMMIT =
`5c0d91959af1f502b667ed6ab815c949c1043cbf`

CONSTRUCTION_COORDINATE_CORRECTION_COMMIT =
`0eb93f49cd33553ba244bb943364ea97125a2e23`

RUNNER_IMPLEMENTATION_COMMIT =
`3362010184de69c2faceb69c0a49c7ca60c67bcd`

RUNNER_PATH =
`scripts/reason_router_gen5_phase1b_r22_c22_construction.py`

RUNNER_SHA256 =
`d92f829a4351b631d213c86755c754e7e5fd5f09240be4e1be6382bbc0421236`

VERIFIER_PATH =
`scripts/verify_reason_router_gen5_phase1b_r22_c22_construction.py`

VERIFIER_SHA256 =
`1fa7bed44effaebaf9f6923fa0a61ff6716de1e4b7493e92994e31ecf444ebb6`

TEST_PATH =
`tests/test_reason_router_gen5_phase1b_r22_c22_construction.py`

TEST_SHA256 =
`119c63cb899a973f59d7e88701073941ea5ac531ee994f69eca80d57326d70ba`

IMPLEMENTATION_VERIFICATION =
`PASS_GEN5_PHASE1B_R22_C22_IMPLEMENTATION_VERIFICATION`

IMPLEMENTATION_TEST_COUNT =
`6`

No implementation file may change between this authority freeze and execution.

---

## 2. Scientific purpose

The only permitted scientific purpose is:

Construct the frozen rank-2 layer-22 native-write role realization `R22` and the
rank-matched response-blind control `C22` from the prospective construction
population.

The run may not test:

- local necessity;
- restoration sufficiency;
- bridge success;
- ownership benefit;
- task behavior;
- downstream logits;
- any training objective.

---

## 3. Frozen construction population

AUTHORIZED_DATA_ROOT =
`data/reason_router_gen5_phase1b_xg1_construction_v1`

PAIR_RANGE =
`xg1_fact_7801..xg1_fact_8100`

PAIR_COUNT =
`300`

ROW_COUNT =
`1800`

CONSTRUCTION_SOURCE_SHA256 =
`3f8eac771794e1d022bee9f669f315c3ac05feeaa9c2195095b8a892b4269ea1`

CONSTRUCTION_ROWS_SHA256 =
`5a82508e54cd6097aeee5afe10d7c416357302701d558cfdc8740382168feca4`

CONSTRUCTION_STRUCTURAL_MANIFEST_SHA256 =
`d68d812ca43a6c651d284f0aa2889f87adb7e030c599af37cb49deaa9d6d8105`

CONSTRUCTION_ANCHOR_SHA256 =
`1dbdd3e072245072f197d87443cc8cf81cbbcd2e61437c99b26e0d82815e33b4`

CONSTRUCTION_ELIGIBILITY_SHA256 =
`7e304fb1d6f2ecb9ed6fb87911623c47d0b6ae9c2eb9bf9b8f45f70ae1fcb06a`

Cohort replacement, row filtering, pair deletion, and response-guided exclusion
are forbidden.

---

## 4. Confirmation-data firewall

The run must not read scientific rows, claims, evidence, anchors, or model
responses from:

`data/reason_router_gen5_phase1b_xg1_necessity_confirmation_v1`

or:

`data/reason_router_gen5_phase1b_xg1_restoration_confirmation_v1`

The runner implementation contains no path to either confirmation dataset.

NECESSITY_CONFIRMATION_DATA_LOADED =
`False`

RESTORATION_CONFIRMATION_DATA_LOADED =
`False`

must be persisted.

---

## 5. Frozen model identity

MODEL_FAMILY =
`state-spaces/mamba-130m-hf`

REPRESENTATIVE_SEED =
`180`

REPRESENTATIVE_ARM =
`G3-GROUP-D-HALF`

REPRESENTATIVE_CHECKPOINT_RUNTIME_PATH =
`reports/reason_router_gen3_grouped_factorial_runs/seed180/G3-GROUP-D-HALF/selected_checkpoint.pt`

REPRESENTATIVE_CHECKPOINT_SHA256 =
`1ff3fcf2ebd754ab6f9483d6a9982b9b04b9a4eb3357f9f8cdbe2b30399e7d2f`

NATIVE_BACKBONE_SIGNATURE_SHA256 =
`81cd368d8a94932561e0ccd50f45a7db1f27941c00b3a08c8b816badaf25f415`

MODEL_REPLICATION_COUNT =
`1`

CHECKPOINT_LOAD_COUNT =
`1`

No checkpoint substitution or checkpoint sweep is authorized.

---

## 6. Frozen PP3/PP5 geometry

PP3_PLUS_SHA256 =
`66ad0cd0f931b0aff88bfc8afc4e9ffa9e355d32f054d376acebb97b469a3cff`

PP3_MINUS_SHA256 =
`ea2c997de7dbaea4edad8b1f30cdac31b4a0c76db0f0df2983900f28a2529ce7`

PP5_PLUS_SHA256 =
`7eb8154a10f647a4b732f7a7b7e34087840a513b88177da06633d8f7b28a4df2`

PP5_MINUS_SHA256 =
`311a41c37b9586206ea9bfc9da390688d9a6bb5672a5f63bb589c9d019478855`

No plane recomputation, refit, sign selection, or alternative control search is
authorized.

---

## 7. Frozen intervention and observation semantics

CONSTRUCTION_PROBE =
`NONE`

CONDITIONS =
`NATIVE / PP3_NEUTRALIZED / PP5_COEFFICIENT_CONTROL`

BRANCHES =
`TARGET_PLUS / TARGET_MINUS`

BRANCH_COORDINATE =
`TARGET_PLUS_MINUS_TARGET_MINUS`

PROPAGATED_WRITE_CONTRAST =
`PP5_CONTROL_MINUS_PP3_NEUTRALIZED`

INTERVENTION_LAYER =
`17`

NATIVE_WRITE_LAYER =
`22`

NATIVE_WRITE_OBJECT =
`deltaB_u[:, :, target_token, :]`

POST_STATE_OBJECT =
`ssm_state after recurrent update and before recurrent readout`

STATE_SHAPE =
`(1,1536,16)`

FLATTENED_WIDTH =
`24576`

No channel selection, top-k filtering, layer search, token search, rank search,
or response-guided basis construction is authorized.

---

## 8. Frozen construction budget

FORWARDS_PER_CONDITION =
`2`

CONDITIONS_PER_PAIR =
`3`

FORWARDS_PER_PAIR =
`6`

PAIR_COUNT =
`300`

EXPECTED_BACKBONE_FORWARD_COUNT =
`1800`

FORWARD_BATCH_SIZE =
`1`

EXTRA_SCIENTIFIC_FORWARD =
`FORBIDDEN`

The forward budget must be exact. A run using fewer or more than 1800 model
forwards is not a valid construction run.

---

## 9. R22/C22 construction contract

R22_RANK =
`2`

R22_SOURCE =
`CENTERED_dW`

C22_RANK =
`2`

C22_SOURCE =
`CENTERED_NATIVE_BRANCH_WRITE_CONTRAST_AFTER_R22_RESIDUALIZATION`

BASIS_ORTHOGONALITY_ATOL =
`1e-10`

RECURRENCE_RECONSTRUCTION_REL_TOL =
`5e-6`

R22 rank-2 uniqueness must pass the frozen second-versus-third singular-value
gate.

C22 rank-2 uniqueness must independently pass the same gate.

If either gate fails, the run is blocked and no alternative rank or control may
be tried under this authority.

---

## 10. Runtime authority

EXECUTION_VENUE =
`KAGGLE_NOTEBOOK_CPU_ONLY`

PYTHON_VERSION =
`3.12.13`

NUMPY_VERSION =
`2.0.2`

TORCH_VERSION =
`2.10.0+cpu`

TRANSFORMERS_VERSION =
`5.0.0`

TOKENIZERS_VERSION =
`0.22.2`

RUNTIME_VERSION_MATCHING =
`EXACT`

PACKAGE_INSTALLATION_DURING_SCIENTIFIC_RUN =
`FORBIDDEN`

PACKAGE_UPGRADE_DURING_SCIENTIFIC_RUN =
`FORBIDDEN`

PACKAGE_DOWNGRADE_DURING_SCIENTIFIC_RUN =
`FORBIDDEN`

Runtime validation must pass before checkpoint deserialization or model
construction.

---

## 11. CPU-only invariant

ACCELERATOR =
`NONE`

GPU =
`OFF`

CUDA_VISIBLE_DEVICES =
`EMPTY`

NVIDIA_VISIBLE_DEVICES =
`void`

TORCH_CUDA_IS_AVAILABLE =
`False`

TORCH_CUDA_DEVICE_COUNT =
`0`

SCIENTIFIC_MODEL_DEVICE =
`cpu`

CUDA_MODEL_EXECUTION =
`FORBIDDEN`

The outer execution wrapper must set:

`CUDA_VISIBLE_DEVICES=""`

and:

`NVIDIA_VISIBLE_DEVICES=void`

before importing the scientific runner.

The exact CPU-only gate must pass before checkpoint deserialization.

---

## 12. Frozen Transformers source identity

TRANSFORMERS_MAMBA_SOURCE_SHA256 =
`4c972b30f3c2cca977824fcc6891f956cd4387b6383aa7336848fbc5f2db1d83`

TRANSFORMERS_MAMBA_SOURCE_BYTES =
`39500`

TRANSFORMERS_CACHE_SOURCE_SHA256 =
`6c123bbe3d23500462f0b617119a8231aa054b6d5475b295443a45f34466e6bc`

TRANSFORMERS_CACHE_SOURCE_BYTES =
`60432`

Source-hash or source-byte-count relaxation is forbidden.

---

## 13. Historical backend continuity evidence

HISTORICAL_EQUIVALENCE_RESULT =
`PASS_XG1_FAST_CUDA_ONE_PAIR_EQUIVALENCE`

HISTORICAL_EQUIVALENCE_PAIR =
`xg1_fact_001`

HISTORICAL_EQUIVALENCE_CHECKPOINT_SHA256 =
`1ff3fcf2ebd754ab6f9483d6a9982b9b04b9a4eb3357f9f8cdbe2b30399e7d2f`

HISTORICAL_STATE_ATOL =
`1e-4`

HISTORICAL_STATE_RTOL =
`1e-4`

HISTORICAL_MAX_STATE_ABS_DIFF =
`5.520135164260864e-05`

This historical result supports backend continuity only. It does not replace the
native-write source-role gate and it does not constitute a Gen5 scientific
result.

---

## 14. Frozen model/tokenizer provisioning

PROVISIONING_SOURCE =
`FROZEN_GITHUB_RELEASE_READ_ONLY`

RELEASE_REPOSITORY =
`9terry-student/ContraMamba`

RELEASE_TAG =
`gen4-r5-evaluator-checkpoints-cf08261`

RELEASE_TARGET_COMMIT =
`cf0826174c2ab1b2203f68afbdeed9da3ff64aa2`

CHECKPOINT_RELEASE_ASSET =
`seed180__G3-GROUP-D-HALF__selected_checkpoint.pt`

CHECKPOINT_RELEASE_ASSET_BYTES =
`518270455`

CHECKPOINT_RELEASE_ASSET_SHA256 =
`1ff3fcf2ebd754ab6f9483d6a9982b9b04b9a4eb3357f9f8cdbe2b30399e7d2f`

MODEL_SNAPSHOT_CONFIG_ASSET =
`model_snapshot_A__config.json`

MODEL_SNAPSHOT_CONFIG_SHA256 =
`784825b6b6cdde47a1602278db0e66d6764169af4a8e662404701bc636a2686a`

MODEL_SNAPSHOT_SPECIAL_TOKENS_ASSET =
`model_snapshot_A__special_tokens_map.json`

MODEL_SNAPSHOT_SPECIAL_TOKENS_SHA256 =
`57491904f8680d4b52ed440f1f7ba48cad1c31ecf3eb453b03484e6ff4723ae8`

MODEL_SNAPSHOT_TOKENIZER_ASSET =
`model_snapshot_A__tokenizer.json`

MODEL_SNAPSHOT_TOKENIZER_SHA256 =
`b074ad869d4f45d1265ca5c9814f78604f3d7e187acc063b15dd232b27585fcf`

MODEL_SNAPSHOT_TOKENIZER_CONFIG_ASSET =
`model_snapshot_A__tokenizer_config.json`

MODEL_SNAPSHOT_TOKENIZER_CONFIG_SHA256 =
`9d7016c33747c6309346e59bd7bf63bfc33c9d9366ecb7e514b3b84dc6b46acb`

MODEL_SNAPSHOT_LOCATION =
`OUTSIDE_GIT_WORKTREE`

TOKENIZER_SNAPSHOT_LOCATION =
`OUTSIDE_GIT_WORKTREE`

HUGGINGFACE_NETWORK_DOWNLOAD =
`FORBIDDEN`

ARBITRARY_MODEL_SUBSTITUTION =
`FORBIDDEN`

---

## 15. Execution binding

The construction run must execute from a clean checkout of the commit that
freezes this authority candidate.

EXPECTED_EXECUTION_HEAD =
`THE_COMMIT_THAT_FREEZES_THIS_AUTHORITY`

EXECUTION_AUTHORITY_COMMIT =
`THE_SAME_COMMIT`

The implementation commit
`3362010184de69c2faceb69c0a49c7ca60c67bcd`
must be an ancestor of the execution authority commit.

The worktree must be clean immediately before scientific execution.

Exactly one bounded construction run identity is authorized under this authority.

A failed run identity is single-use and must not be silently rerun. A retry
requires preserved failure provenance and a new run identity.

---

## 16. Checkpoint provisioning rule

The checkpoint may be provisioned temporarily at the exact historical path only
after byte-count and SHA256 verification.

Temporary provisioning must:

- use a regular file, not a symlink;
- preserve prior `.git/info/exclude` bytes;
- temporarily ignore only the exact checkpoint destination if required;
- leave `git status --porcelain` empty before scientific execution;
- restore `.git/info/exclude` on success or failure;
- remove temporary provisioning inputs when the run wrapper finishes.

Provisioning failure or cleanup failure is a blocker.

---

## 17. Output contract

OUTPUT_DIRECTORY_PATTERN =
`reports/reason_router_gen5_phase1b_r22_c22_construction_<execution-head-short>_v1`

OUTPUT_BUNDLE_FILE_COUNT =
`6`

OUTPUT_FILES =

- `r22_basis.f64le`
- `c22_basis.f64le`
- `construction_item_audit.jsonl`
- `construction_summary.json`
- `artifact_manifest.json`
- `SHA256SUMS.txt`

R22_BASIS_BYTES =
`393216`

C22_BASIS_BYTES =
`393216`

BASIS_DTYPE =
`little-endian float64`

RAW_NATIVE_VECTORS_PERSISTED =
`NO`

PARTIAL_OUTPUT_AS_SCIENTIFIC_EVIDENCE =
`FORBIDDEN`

Output collision is a blocker.

---

## 18. Required negative attestations

A successful summary must persist:

`necessity_confirmation_data_loaded = false`

`restoration_confirmation_data_loaded = false`

`Q_computed = false`

`primary_inference_executed = false`

`multiplicity_correction_executed = false`

`training_executed = false`

`backward_executed = false`

`task_heads_executed = false`

`logits_read = false`

`cuda_executed = false`

`raw_native_vectors_persisted = false`

`scientific_conclusion = null`

---

## 19. Success boundary

The strongest successful execution verdict is:

`PASS_GEN5_PHASE1B_R22_C22_CONSTRUCTION`

A PASS establishes only:

1. the authorized 300-pair construction population executed successfully;
2. the exact 1800-forward budget was consumed;
3. the frozen layer-17 conditions were applied;
4. layer-22 native write/post-state capture passed recurrence checks;
5. the pre-specified rank-2 R22 and rank-2 C22 construction gates passed;
6. six collision-protected artifacts were emitted.

A PASS does not establish:

- R22 local necessity;
- R22 restoration sufficiency;
- layer17→layer22 bridge support;
- state-ownership benefit;
- behavioral or task improvement.

---

## 20. Post-run handling

CM_COLLECT_AFTER_SUCCESSFUL_EXECUTION =
`AUTHORIZED`

Canonical handoff:

`cm collect <run-name>`

then execute the generated collector in Kaggle, download the ZIP, and locally:

`cm import <handoff.zip>`

Only imported, provenance-valid artifacts may be frozen or interpreted.

After successful import the next authorized step is:

`R22_C22_ARTIFACT_PROVENANCE_VALIDATION_AND_FREEZE`

Necessity confirmation implementation is not authorized until R22/C22 artifacts
are frozen.

---

## 21. Summary

PHASE =
`GEN5_PHASE1B_R22_C22_CONSTRUCTION_EXECUTION_AUTHORITY`

AUTHORIZED_POPULATION =
`xg1_fact_7801..xg1_fact_8100`

AUTHORIZED_FORWARD_COUNT =
`1800`

AUTHORIZED_DEVICE =
`CPU_ONLY`

AUTHORIZED_OUTPUT =
`R22_C22_CONSTRUCTION_ARTIFACTS_ONLY`

Q =
`FORBIDDEN`

CONFIRMATION =
`FORBIDDEN`

TRAINING =
`FORBIDDEN`

README_UPDATE_REQUIRED =
`NO`

STATUS =
`FROZEN_ON_COMMIT`
