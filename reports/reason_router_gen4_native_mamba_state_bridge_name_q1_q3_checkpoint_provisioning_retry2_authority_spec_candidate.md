# ContraMamba Gen4 Native Mamba State Bridge
# NAME Q1/Q3 Representative Checkpoint Provisioning
# Retry2 Failure Recovery Authority
# Candidate

STATUS =
CANDIDATE

AUTHORITY_ID =
GEN4_NATIVE_MAMBA_STATE_BRIDGE_NAME_Q1_Q3_CHECKPOINT_PROVISIONING_RETRY2_AUTHORITY

PHASE =
NAME_Q1_Q3_SCIENTIFIC_EXTRACTION_FAILURE_RECOVERY

SCIENTIFIC_CONCLUSION =
NONE


## 1. Authority lineage

PARENT_SCIENTIFIC_EXECUTION_AUTHORITY_COMMIT =
a01c625b18c58fea90246755476ac0df2ff331e6

PARENT_SNAPSHOT_RECOVERY_AUTHORITY_COMMIT =
07ec52d584793d14e7d9d9a1db45fc8529d76c69

Q1_Q3_EXECUTION_DRIVER_IMPLEMENTATION_COMMIT =
131f304901719547e2dca046909861a65e34ac59

Q1_Q3_EXECUTION_DRIVER_SHA256 =
1d932c22779cf55cbe81a76250d61ea62ff654077aec585cf85b94d2d2516a06

Q1_Q3_PRIMITIVE_IMPLEMENTATION_COMMIT =
d5317b2c5be09464c9196325429479c6bff25efd


## 2. Retry1 failed-run provenance

FAILED_RETRY_RUN_NAME =
gen4-name-q1q3-scientific-extraction-retry1

FAILED_RETRY_HEAD =
07ec52d584793d14e7d9d9a1db45fc8529d76c69

FAILED_RETRY_COMMAND_SHA256 =
aa6c460645fc87774b1ba2f1c91795c170b1514e86379afd3b9071457e512b95

FAILED_RETRY_EXIT_CODE =
1

FAILED_RETRY_STARTED_UTC =
2026-09-13T15:32:49Z

FAILED_RETRY_FINISHED_UTC =
2026-09-13T15:33:16Z

FAILED_RETRY_RUN_LOG_SHA256 =
4ac71853a7a9f5fc35db0f92aa7d3adcec775262f9f86ea2511f1fd01d5741cf

FAILED_RETRY_RUN_META_SHA256 =
61bfc18b7d9b7739ddd356d47924524cffa01cf5cc64e2629cd0a613192ea1b2

FAILED_RETRY_HANDOFF_ZIP_SHA256 =
d0c97771905578efda2e792c835c70f7f07255033a03a88333c59d4281e990e8

FAILED_RETRY_IMPORT =
PASS

FAILED_RETRY_VALIDATED_ARTIFACTS =
0

FAILED_RETRY_COPIED_ARTIFACTS =
0

FAILED_RETRY_RUN_IDENTITY_REUSE =
PROHIBITED


## 3. Failure classification

FAILURE_CLASSIFICATION =
REPRESENTATIVE_CHECKPOINT_NOT_PROVISIONED_AT_CANONICAL_PATH

IMPLEMENTATION_DEFECT_ESTABLISHED =
NO

SCIENTIFIC_RESULT_ESTABLISHED =
NO

SCIENTIFIC_ARTIFACT_BUNDLE_VALIDATED =
NO

The recovery target is infrastructure provisioning only.

No checkpoint may be regenerated, retrained, reselected, converted,
rewritten, or substituted.


## 4. Frozen representative checkpoint identity

REPRESENTATIVE_SEED =
180

REPRESENTATIVE_ARM =
G3-GROUP-D-HALF

CANONICAL_CHECKPOINT_PATH =
reports/reason_router_gen3_grouped_factorial_runs/seed180/G3-GROUP-D-HALF/selected_checkpoint.pt

CHECKPOINT_BYTES =
518270455

CHECKPOINT_SHA256 =
1ff3fcf2ebd754ab6f9483d6a9982b9b04b9a4eb3357f9f8cdbe2b30399e7d2f

CHECKPOINT_PAYLOAD_SCHEMA =
stage176a0_selected_checkpoint_v1

HISTORICAL_SELECTED_EPOCH =
15

HISTORICAL_STRICT_LOAD_VALIDATION =
PASS

CHECKPOINT_RESELECTION =
PROHIBITED

CHECKPOINT_REGENERATION =
PROHIBITED

CHECKPOINT_RETRAINING =
PROHIBITED

CHECKPOINT_STATE_DICT_REWRITE =
PROHIBITED


## 5. Frozen release provisioning source

RELEASE_TAG =
gen4-r5-evaluator-checkpoints-cf08261

RELEASE_TARGET_COMMIT =
cf0826174c2ab1b2203f68afbdeed9da3ff64aa2

RELEASE_ID =
387669220

RELEASE_KIND =
PRERELEASE

RELEASE_PURPOSE =
EXECUTION_ONLY_FROZEN_GEN3_EVALUATOR_CHECKPOINT_PROVISIONING

CHECKPOINT_RELEASE_ASSET_ID =
559765805

CHECKPOINT_RELEASE_ASSET_NAME =
seed180__G3-GROUP-D-HALF__selected_checkpoint.pt

CHECKPOINT_RELEASE_ASSET_BYTES =
518270455

CHECKPOINT_RELEASE_ASSET_SHA256 =
1ff3fcf2ebd754ab6f9483d6a9982b9b04b9a4eb3357f9f8cdbe2b30399e7d2f

ALTERNATE_CHECKPOINT_SOURCE =
PROHIBITED

LATEST_ASSET_SELECTION =
PROHIBITED

UNPINNED_CHECKPOINT_DOWNLOAD =
PROHIBITED


## 6. Provisioning contract

NETWORK_RETRIEVAL =
YES_BOUNDED

Only the single frozen release asset identified above may be retrieved.

The checkpoint must first be downloaded outside the repository worktree.

Before publication into the canonical checkpoint path:

1. downloaded byte count must equal 518270455;
2. downloaded SHA256 must equal
   1ff3fcf2ebd754ab6f9483d6a9982b9b04b9a4eb3357f9f8cdbe2b30399e7d2f;
3. any mismatch must fail closed;
4. no model construction or model forward may occur.

PARTIAL_CHECKPOINT_PUBLICATION =
PROHIBITED

MISMATCHED_CHECKPOINT_PUBLICATION =
PROHIBITED


## 7. Clean-worktree preservation

The frozen Q1/Q3 execution driver requires a clean git worktree before
scientific execution.

The canonical checkpoint is a large execution artifact and is not tracked
in the scientific execution commit.

TRACKED_GITIGNORE_MODIFICATION =
PROHIBITED

TRACKED_REPOSITORY_MODIFICATION =
PROHIBITED

TEMPORARY_LOCAL_GIT_EXCLUDE =
YES_BOUNDED

EXACT_TEMPORARY_EXCLUDE_PATH =
/reports/reason_router_gen3_grouped_factorial_runs/seed180/G3-GROUP-D-HALF/selected_checkpoint.pt

Only this exact path may be added temporarily to the repository-local
git info/exclude file.

Before changing the local exclude file:

- the original git info/exclude bytes must be backed up outside the repo;
- the repository must otherwise be clean.

After the exact authenticated checkpoint is placed at the canonical path:

- git status --porcelain must still be empty;
- the checkpoint byte count and SHA256 must be re-authenticated at the
  canonical path;
- only then may the frozen execution driver start.

OTHER_EXCLUDE_PATTERN_ADDITION =
PROHIBITED

GLOBAL_GITIGNORE_MODIFICATION =
PROHIBITED

GIT_CONFIG_CLEAN_GATE_BYPASS =
PROHIBITED

STATUS_SHOW_UNTRACKED_FILES_DISABLE =
PROHIBITED


## 8. Mandatory cleanup

The recovery command must install an exit trap before checkpoint publication.

On both success and failure it must:

1. remove only the provisioned canonical checkpoint;
2. remove newly created empty checkpoint parent directories only when safe;
3. restore the exact original git info/exclude bytes;
4. preserve scientific output artifacts and run provenance;
5. preserve the driver's actual exit code.

DESTRUCTIVE_REPOSITORY_CLEAN =
PROHIBITED

GIT_RESET =
PROHIBITED

GIT_CLEAN =
PROHIBITED


## 9. Snapshot recovery inheritance

The exact model/tokenizer snapshot contract from
07ec52d584793d14e7d9d9a1db45fc8529d76c69 remains unchanged.

HF_REPOSITORY =
state-spaces/mamba-130m-hf

HF_REVISION =
40e5d2bd7452abb3ca8fadbafe9131ee0e2c2f37

TOKENIZER_JSON_SHA256 =
b074ad869d4f45d1265ca5c9814f78604f3d7e187acc063b15dd232b27585fcf

TOKENIZER_CONFIG_SHA256 =
9d7016c33747c6309346e59bd7bf63bfc33c9d9366ecb7e514b3b84dc6b46acb

SPECIAL_TOKENS_MAP_SHA256 =
57491904f8680d4b52ed440f1f7ba48cad1c31ecf3eb453b03484e6ff4723ae8

MODEL_CONFIG_BYTES =
895

MODEL_CONFIG_SHA256 =
784825b6b6cdde47a1602278db0e66d6764169af4a8e662404701bc636a2686a

SNAPSHOT_SEMANTICS_CHANGE =
NO


## 10. Retry2 scientific execution boundary

RETRY_RUN_NAME =
gen4-name-q1q3-scientific-extraction-retry2

RETRY_REUSES_V1_NAME =
NO

RETRY_REUSES_RETRY1_NAME =
NO

DEVICE =
CPU_ONLY

GPU =
OFF

EXPECTED_BACKBONE_FORWARD_COUNT =
600

EXPECTED_SUPPORT_STATE_ROWS =
7200

EXPECTED_ENDPOINT_ROWS =
1200

EXPECTED_OUTPUT_DIRECTORY =
reports/reason_router_gen4_native_mamba_state_bridge_name_q1_q3_scientific_extraction_v1

OUTPUT_COLLISION_BEHAVIOR =
FAIL_CLOSED

SCIENTIFIC_EXECUTION_ALLOWED =
YES_BOUNDED_INHERITED

STATISTICAL_TESTING_ALLOWED =
NO

TRAINING_ALLOWED =
NO

BACKWARD_ALLOWED =
NO

GRADIENT_UPDATE_ALLOWED =
NO


## 11. Required execution order

The future pinned retry2 command must:

1. require the exact retry2 authority commit as HEAD;
2. require a clean Kaggle repository;
3. provision and authenticate the frozen four-file model/tokenizer snapshot;
4. download the single frozen representative checkpoint to external staging;
5. authenticate checkpoint bytes and SHA256 in staging;
6. back up repository-local git info/exclude;
7. add only the exact canonical checkpoint path to local exclude;
8. publish the authenticated checkpoint at the canonical path;
9. require git status --porcelain to remain empty;
10. re-authenticate checkpoint bytes and SHA256 at canonical path;
11. invoke the unchanged frozen Q1/Q3 execution driver;
12. preserve the driver's exit code;
13. remove the temporary canonical checkpoint;
14. restore the exact original git info/exclude bytes.

No scientific forward may occur before the driver independently passes:

- execution authority validation;
- frozen dependency validation;
- full 1800-row R2 coordinate authentication;
- representative checkpoint authentication.


## 12. Scientific interpretation boundary

Successful retry2 execution establishes only execution success and candidate
Q1/Q3 extraction artifacts.

It does not establish:

- statistical support;
- a significant Q1-versus-midpoint difference;
- a significant Q3-versus-midpoint difference;
- depth selectivity;
- mediation;
- causality;
- overall adaptive-program FWER.

SCIENTIFIC_CONCLUSION_FROM_THIS_AUTHORITY =
NONE


## 13. Next transition

NEXT_ACTION_AFTER_THIS_AUTHORITY_FREEZE =
KAGGLE_FRESH_BOOTSTRAP_AND_PINNED_RETRY2

NEXT_ACTION_AFTER_SUCCESSFUL_RETRY2_IMPORT =
Q1_Q3_EXTRACTION_ARTIFACT_PROVENANCE_VALIDATION

STATISTICAL_EXECUTION_AUTHORITY =
NOT_YET_AUTHORIZED
