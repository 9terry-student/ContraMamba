# ContraMamba Gen4 Native Mamba State Bridge
# NAME Q1/Q3 Snapshot Provisioning
# Failure Recovery Authority
# Candidate

STATUS =
CANDIDATE

AUTHORITY_ID =
GEN4_NATIVE_MAMBA_STATE_BRIDGE_NAME_Q1_Q3_SNAPSHOT_PROVISIONING_FAILURE_RECOVERY_AUTHORITY

PHASE =
NAME_Q1_Q3_SCIENTIFIC_EXTRACTION_FAILURE_RECOVERY

SCIENTIFIC_CONCLUSION =
NONE


## 1. Parent execution authority

PARENT_EXECUTION_AUTHORITY_COMMIT =
a01c625b18c58fea90246755476ac0df2ff331e6

Q1_Q3_EXECUTION_DRIVER_IMPLEMENTATION_COMMIT =
131f304901719547e2dca046909861a65e34ac59

Q1_Q3_EXECUTION_DRIVER_SHA256 =
1d932c22779cf55cbe81a76250d61ea62ff654077aec585cf85b94d2d2516a06

Q1_Q3_PRIMITIVE_IMPLEMENTATION_COMMIT =
d5317b2c5be09464c9196325429479c6bff25efd


## 2. Failed run provenance

FAILED_RUN_NAME =
gen4-name-q1q3-scientific-extraction-v1

FAILED_RUN_HEAD =
a01c625b18c58fea90246755476ac0df2ff331e6

FAILED_COMMAND_SHA256 =
b46f905983e5f9351cfc6ba809ef72cac0ed65bc663ad23cb5e7c0390201f35f

FAILED_EXIT_CODE =
1

FAILED_RUN_STARTED_UTC =
2026-09-13T15:21:29Z

FAILED_RUN_FINISHED_UTC =
2026-09-13T15:21:58Z

FAILED_RUN_LOG_SHA256 =
1a7ede8178c724004ad12ef80cc2626c4b924675c52031360498be749f064dbc

FAILED_RUN_META_SHA256 =
5d9b69019ea24286ed38d47bb317caa48ad490cf8e35cd7f4d45edb051e7593a

FAILED_HANDOFF_ZIP_SHA256 =
170a748a3d15be9023e8a652878619b35671f9ce60bceede1f7ddfeb548de708

FAILED_HANDOFF_FILES_COLLECTED =
0

FAILED_HANDOFF_IMPORT =
PASS

FAILED_HANDOFF_VALIDATED_ARTIFACTS =
0

FAILED_HANDOFF_COPIED_ARTIFACTS =
0

FAILED_RUN_IDENTITY_REUSE =
PROHIBITED


## 3. Failure classification

FAILURE_STAGE =
TOKENIZER_SNAPSHOT_AUTHENTICATION

FAILURE_MESSAGE =
tokenizer file missing: tokenizer.json

MODEL_CONSTRUCTION_REACHED =
NO

CHECKPOINT_DESERIALIZATION_REACHED =
NO

SCIENTIFIC_MODEL_FORWARD_REACHED =
NO

SCIENTIFIC_BACKBONE_FORWARD_COUNT =
0

SCIENTIFIC_ARTIFACTS_CREATED =
0

SCIENTIFIC_EVIDENCE_CREATED =
NO

CLASSIFICATION =
EXECUTION_ENVIRONMENT_SNAPSHOT_NOT_PROVISIONED

IMPLEMENTATION_DEFECT_ESTABLISHED =
NO


## 4. Recovery scope

This recovery does not modify scientific code.

TRACKED_IMPLEMENTATION_MODIFICATION =
PROHIBITED

Q1_Q3_DRIVER_MODIFICATION =
PROHIBITED

Q1_Q3_PRIMITIVE_MODIFICATION =
PROHIBITED

PRIMARY_PHASE_D_MODIFICATION =
PROHIBITED

SCIENTIFIC_SPECIFICATION_MODIFICATION =
PROHIBITED

The recovery is limited to provisioning and authenticating the exact frozen
external model/tokenizer snapshot before retrying the already-authorized
scientific extraction.


## 5. Frozen external snapshot

HF_REPOSITORY =
state-spaces/mamba-130m-hf

HF_REVISION =
40e5d2bd7452abb3ca8fadbafe9131ee0e2c2f37

TARGET_SNAPSHOT_DIRECTORY =
~/.cache/huggingface/hub/models--state-spaces--mamba-130m-hf/snapshots/40e5d2bd7452abb3ca8fadbafe9131ee0e2c2f37

Only the exact revision above is authorized.

LATEST_REVISION =
PROHIBITED

BRANCH_REVISION =
PROHIBITED

UNPINNED_DOWNLOAD =
PROHIBITED


## 6. Required tokenizer files

TOKENIZER_JSON =
tokenizer.json

TOKENIZER_JSON_SHA256 =
b074ad869d4f45d1265ca5c9814f78604f3d7e187acc063b15dd232b27585fcf

TOKENIZER_CONFIG =
tokenizer_config.json

TOKENIZER_CONFIG_SHA256 =
9d7016c33747c6309346e59bd7bf63bfc33c9d9366ecb7e514b3b84dc6b46acb

SPECIAL_TOKENS_MAP =
special_tokens_map.json

SPECIAL_TOKENS_MAP_SHA256 =
57491904f8680d4b52ed440f1f7ba48cad1c31ecf3eb453b03484e6ff4723ae8


## 7. Required model configuration

MODEL_CONFIG =
config.json

MODEL_CONFIG_BYTES =
895

MODEL_CONFIG_SHA256 =
784825b6b6cdde47a1602278db0e66d6764169af4a8e662404701bc636a2686a

Provisioning only the tokenizer files is insufficient.

The same exact frozen snapshot directory must contain all four required files
before retry execution begins.


## 8. Provisioning policy

NETWORK_RETRIEVAL =
YES_BOUNDED

Authorized retrieval source is only the pinned HF repository/revision above.

Each downloaded file must be written to a temporary staging location first.

Before publication into the snapshot directory:

- every required file must exist;
- every tokenizer file must match its exact SHA256;
- config.json must match exact byte count 895;
- config.json must match its exact SHA256.

MISMATCH_BEHAVIOR =
FAIL_CLOSED

PARTIAL_SNAPSHOT_PUBLICATION =
PROHIBITED

MISMATCHED_PREEXISTING_FILE_OVERWRITE =
PROHIBITED

REPOSITORY_WORKTREE_MODIFICATION =
PROHIBITED


## 9. Retry execution

RETRY_RUN_NAME =
gen4-name-q1q3-scientific-extraction-retry1

RETRY_REUSES_FAILED_RUN_NAME =
NO

The retry may proceed only after exact snapshot provisioning succeeds.

The retry must use:

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

STATISTICAL_TESTING_ALLOWED =
NO

TRAINING_ALLOWED =
NO

BACKWARD_ALLOWED =
NO

GRADIENT_UPDATE_ALLOWED =
NO


## 10. Retry ordering

The pinned recovery command must perform the following order:

1. verify exact execution HEAD and clean repository;
2. stage/download the exact four frozen snapshot files;
3. verify all frozen file SHA256 values;
4. verify config.json byte count;
5. publish the authenticated snapshot outside the repository;
6. invoke the frozen Q1/Q3 execution driver;
7. driver re-authenticates tokenizer snapshot;
8. driver reconstructs and authenticates the complete 1800-row R2 coordinate;
9. driver selects the 600-row C0_SHAM/C2_NAME subset;
10. only then construct/load the representative model;
11. execute exactly 600 CPU backbone forwards;
12. produce the frozen five-artifact bundle.

MODEL_FORWARD_BEFORE_SNAPSHOT_AUTHENTICATION =
PROHIBITED

MODEL_FORWARD_BEFORE_R2_COORDINATE_GATE =
PROHIBITED


## 11. Scientific interpretation boundary

A successful retry establishes only:

- execution success;
- candidate Q1/Q3 measurement artifacts;
- artifact provenance subject to later collect/import validation.

It does not establish:

- statistical support;
- depth selectivity;
- cross-layer difference;
- mediation;
- causality;
- overall adaptive-program FWER.

STATISTICAL_TESTING =
NOT_AUTHORIZED

SCIENTIFIC_CONCLUSION_FROM_RECOVERY =
NONE


## 12. Next phase

NEXT_ACTION_AFTER_THIS_AUTHORITY_FREEZE =
KAGGLE_FRESH_BOOTSTRAP_AND_PINNED_RETRY1

NEXT_ACTION_AFTER_SUCCESSFUL_RETRY_IMPORT =
Q1_Q3_EXTRACTION_ARTIFACT_PROVENANCE_VALIDATION

STATISTICAL_EXECUTION_AUTHORITY =
NOT_YET_AUTHORIZED
