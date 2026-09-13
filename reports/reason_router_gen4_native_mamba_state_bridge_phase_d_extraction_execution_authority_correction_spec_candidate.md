# ContraMamba Gen4 Native Mamba State Bridge
# Corrected Phase D Native-State Extraction Execution Authority
# Candidate

STATUS =
CANDIDATE

AUTHORITY_ID =
GEN4_NATIVE_MAMBA_STATE_BRIDGE_PHASE_D_EXTRACTION_EXECUTION_AUTHORITY

SCIENTIFIC_EXECUTION_ALLOWED =
YES_BOUNDED

## Corrected implementation binding

PHASE_D_RUNNER_IMPLEMENTATION_COMMIT =
3e5a79593427707c1a66bc3fa6e21878af2a4d68

PHASE_D_RUNNER_SHA256 =
3edc4de69433ededd9e3ad811a716c0f37d00e19ab896473b9e56877edb737b2

PHASE_D_RUNNER_PATH =
scripts/reason_router_gen4_native_mamba_state_extraction.py

CORRECTED_MEASUREMENT_SHA256 =
e31d82af229c79b7de3d55f3419a1ae1cc9da0e1c0af5bfd84ba4cec85a36a82

CORRECTED_MEASUREMENT_PATH =
scripts/reason_router_gen4_native_mamba_state_measurement.py

SOURCE_ROLE_VALIDATOR_CORRECTION_AUTHORITY =
5ee0826c629978a72240bb88f732aacf6f357c4b

CORRECTED_IMPLEMENTATION_COMMIT =
3e5a79593427707c1a66bc3fa6e21878af2a4d68

CORRECTED_IMPLEMENTATION_VALIDATION =
PASS_64_SYNTHETIC_STATIC_TESTS

## Superseded execution authority

PRIOR_PHASE_D_EXECUTION_AUTHORITY_COMMIT =
8b73320c88362f6caed44877971cfc81f33e0bbf

PRIOR_PHASE_D_EXECUTION_AUTHORITY_STATUS =
SUPERSEDED_FOR_CORRECTED_IMPLEMENTATION_BYTES

The prior execution authority does not authorize execution of the corrected
measurement/runner bytes.

The failed v1/v2/v3 attempts under the prior authority remain provenance
records and must not be reused as scientific evidence.

PRIOR_FAILED_RUNS_SCIENTIFIC_FORWARD =
ZERO

PRIOR_FAILED_RUNS_NATIVE_STATE_EXTRACTION =
ZERO

## Scientific execution scope

EXECUTION_PURPOSE =
Produce the frozen Gen4 native-Mamba recurrent-state measurement bundle for the
already-prespecified mechanistic bridge.

MODEL_REPLICATION_COUNT =
1

REPRESENTATIVE_SEED =
180

REPRESENTATIVE_ARM =
G3-GROUP-D-HALF

REPRESENTATIVE_CHECKPOINT_SHA256 =
1ff3fcf2ebd754ab6f9483d6a9982b9b04b9a4eb3357f9f8cdbe2b30399e7d2f

PRIMARY_LAYER =
11

SCIENTIFIC_FORWARD_BATCH_SIZE =
1

CANONICAL_GEN4_ROW_COUNT =
1800

EXPECTED_BACKBONE_FORWARD_COUNT =
1800

EXPECTED_SUPPORT_STATE_VECTOR_COUNT =
14717

EXPECTED_ENDPOINT_ROW_COUNT =
3600

MODEL_FORWARD_MODE =
BACKBONE_ONLY_MODEL_MAMBA

DOWNSTREAM_HEAD_FORWARD =
FORBIDDEN

Q_AUTHORIZED_FORWARD =
FORBIDDEN

TRAINING =
FORBIDDEN

BACKWARD =
FORBIDDEN

OPTIMIZER_STEP =
FORBIDDEN

PARAMETER_MUTATION =
FORBIDDEN

STATISTICAL_TESTING =
FORBIDDEN

STRUCTURAL_CONTRAST_TESTING =
FORBIDDEN

SCIENTIFIC_CONCLUSION_AT_PHASE_D =
NONE

## Native-state measurement contract

STATE_OBJECT =
native selective-SSM recurrent state s_t

STATE_TIMING =
post-consumption state after recurrent update and before C readout

DELTA_B_U_LINE =
397

RECURRENT_UPDATE_LINE =
409

CAPTURE_LINE =
410

FINAL_CACHE_PERSISTENCE_LINE =
417

STATE_SHAPE =
[1,1536,16]

FLATTENED_STATE_SIZE =
24576

SUPPORT_WINDOW =
[a-1,a+4]

PREFIX_ELIGIBILITY =
a+4<=terminal_index-1

WINDOW_SHORTENING =
FORBIDDEN

EPSILON_IMPUTATION =
FORBIDDEN

NONFINITE_IMPUTATION =
FORBIDDEN

ZERO_TRANSITION_TURNING =
BLOCKED_UNDEFINED_ZERO_TRANSITION_NORM

ZERO_PATH_LENGTH =
BLOCKED_UNDEFINED_ZERO_PATH_LENGTH

AUTHORIZED_ENDPOINTS =
POST4_SPEED
POST4_TURNING
POST4_PATH_EFFICIENCY

## Exact runtime

EXECUTION_VENUE =
KAGGLE_NOTEBOOK_CPU_ONLY

ACCELERATOR =
NONE

GPU =
OFF

PYTHON_VERSION =
3.12.13

NUMPY_VERSION =
2.0.2

TORCH_VERSION =
2.10.0+cpu

TRANSFORMERS_VERSION =
5.0.0

PACKAGE_MUTATION =
FORBIDDEN

TRANSFORMERS_MAMBA_SOURCE_SHA256 =
4c972b30f3c2cca977824fcc6891f956cd4387b6383aa7336848fbc5f2db1d83

TRANSFORMERS_MAMBA_SOURCE_BYTES =
39500

TRANSFORMERS_CACHE_SOURCE_SHA256 =
6c123bbe3d23500462f0b617119a8231aa054b6d5475b295443a45f34466e6bc

TRANSFORMERS_CACHE_SOURCE_BYTES =
60432

The corrected runtime gate must pass before checkpoint deserialization, model
construction, or scientific forward.

No runtime/source/version gate may be bypassed.

## Frozen scientific population

CANONICAL_GEN4_ARTIFACT =
reports/reason_router_gen4_six_cell_masked_slot_substitution_materialization_fbc780ce12cbfbcf4e20a3cb9d1f099553045fc0/gen4_six_cell_masked_slot_substitution.jsonl

CANONICAL_GEN4_ARTIFACT_SHA256 =
b9c54604863ed15c237fa17c7890a20f3f5ec062a7429638b39e8b468be050a7

CANONICAL_GEN4_ARTIFACT_BYTES =
1465573

SOURCE_PAIR_COUNT =
300

CELL_COUNT =
6

EVENT_ANCHOR_MANIFEST_SHA256 =
70c84c68b36751bb7c7145b33ccb71ab91bc8ee9e6cc5f2c7a0d4e925f36581f

EVENT_ANCHOR_MANIFEST_BYTES =
2268260

ACTIVE_ENCODING_SHA256 =
d3cf61e55bb04e6dbebe44be433597cce958cb15989412dcf6efc15a329c576a

## Frozen model/tokenizer provisioning

PROVISIONING_SOURCE =
PUBLIC_GITHUB_PRERELEASE_READ_ONLY

RELEASE_REPOSITORY =
9terry-student/ContraMamba

RELEASE_TAG =
gen4-r5-evaluator-checkpoints-cf08261

RELEASE_TARGET_COMMIT =
cf0826174c2ab1b2203f68afbdeed9da3ff64aa2

RELEASE_MUTATION =
FORBIDDEN

ARBITRARY_NETWORK_DOWNLOAD =
FORBIDDEN

HUGGINGFACE_NETWORK_DOWNLOAD =
FORBIDDEN

The execution may retrieve only the exact required frozen release assets.

REQUIRED_RELEASE_ASSET =
seed180__G3-GROUP-D-HALF__selected_checkpoint.pt

REQUIRED_RELEASE_ASSET_BYTES =
518270455

REQUIRED_RELEASE_ASSET_SHA256 =
1ff3fcf2ebd754ab6f9483d6a9982b9b04b9a4eb3357f9f8cdbe2b30399e7d2f

MODEL_SNAPSHOT_CONFIG_ASSET =
model_snapshot_A__config.json

MODEL_SNAPSHOT_CONFIG_BYTES =
895

MODEL_SNAPSHOT_CONFIG_SHA256 =
784825b6b6cdde47a1602278db0e66d6764169af4a8e662404701bc636a2686a

MODEL_SNAPSHOT_SPECIAL_TOKENS_ASSET =
model_snapshot_A__special_tokens_map.json

MODEL_SNAPSHOT_SPECIAL_TOKENS_BYTES =
473

MODEL_SNAPSHOT_SPECIAL_TOKENS_SHA256 =
57491904f8680d4b52ed440f1f7ba48cad1c31ecf3eb453b03484e6ff4723ae8

MODEL_SNAPSHOT_TOKENIZER_ASSET =
model_snapshot_A__tokenizer.json

MODEL_SNAPSHOT_TOKENIZER_BYTES =
2113837

MODEL_SNAPSHOT_TOKENIZER_SHA256 =
b074ad869d4f45d1265ca5c9814f78604f3d7e187acc063b15dd232b27585fcf

MODEL_SNAPSHOT_TOKENIZER_CONFIG_ASSET =
model_snapshot_A__tokenizer_config.json

MODEL_SNAPSHOT_TOKENIZER_CONFIG_BYTES =
4793

MODEL_SNAPSHOT_TOKENIZER_CONFIG_SHA256 =
9d7016c33747c6309346e59bd7bf63bfc33c9d9366ecb7e514b3b84dc6b46acb

PROVISION_MANIFEST_ASSET =
provision_manifest.json

PROVISION_MANIFEST_BYTES =
6548

PROVISION_MANIFEST_SHA256 =
5d4927ff4396ce16c2e270b5041301bfd963b03eaec7647298432fae65d9df80

PRIVATE_KAGGLE_DATASET_SCIENTIFIC_USE =
NO_SUPERSEDED_BY_FROZEN_GITHUB_RELEASE

## Runtime provisioning rules

MODEL_SNAPSHOT_LOCATION =
OUTSIDE_GIT_WORKTREE

TOKENIZER_SNAPSHOT_LOCATION =
OUTSIDE_GIT_WORKTREE

REPRESENTATIVE_CHECKPOINT_RUNTIME_PATH =
reports/reason_router_gen3_grouped_factorial_runs/seed180/G3-GROUP-D-HALF/selected_checkpoint.pt

The representative checkpoint may be copied temporarily to the exact historical
runner path only after its byte count and SHA256 pass.

Temporary checkpoint provisioning must:

- use a regular file, not a symlink;
- temporarily ignore only the exact checkpoint destination;
- preserve the prior .git/info/exclude bytes;
- leave git status --porcelain empty before scientific execution;
- restore the prior .git/info/exclude bytes on success or failure;
- remove the temporary checkpoint on success or failure;
- remove provisioning input directories on success or failure.

Provisioning cleanup failure is a blocker.

## Execution binding

The future execution must run from a clean checkout of the commit that freezes
this authority candidate.

At execution time:

EXPECTED_EXECUTION_HEAD =
THE_COMMIT_THAT_FREEZES_THIS_AUTHORITY

EXECUTION_AUTHORITY_COMMIT =
THE_SAME_COMMIT

RUNNER_IMPLEMENTATION_COMMIT =
3e5a79593427707c1a66bc3fa6e21878af2a4d68

RUNNER_SHA256 =
3edc4de69433ededd9e3ad811a716c0f37d00e19ab896473b9e56877edb737b2

The runner implementation commit must be an ancestor of the execution authority
commit.

The worktree must be clean before scientific execution.

## Output contract

OUTPUT_BUNDLE_FILE_COUNT =
5

OUTPUT_FILES =
manifest.json
support_state_rows.jsonl
support_states.npy
kinematic_endpoints.jsonl
SHA256SUMS.txt

SUPPORT_STATES_DTYPE =
little-endian float32

SUPPORT_STATES_VECTOR_WIDTH =
24576

SUPPORT_STATES_ORDER =
deterministic support-union order

OUTPUT_PUBLICATION =
collision-protected staging followed by atomic publish

PARTIAL_OUTPUT_AS_SCIENTIFIC_EVIDENCE =
FORBIDDEN

## Success boundary

STRONGEST_PHASE_D_SUCCESS_VERDICT =
PASS_READY_FOR_PHASE_E_MEASUREMENT_ARTIFACT_FREEZE

A successful Phase D run establishes execution success and produces measurement
artifacts.

It does not establish:

- statistical significance;
- a structural contrast result;
- a mechanistic causal claim;
- broader generalization;
- training benefit;
- task improvement.

Phase E artifact/provenance validation is required before scientific
interpretation.

Phase F statistical analysis requires separate authority.

## Collection

CM_COLLECT_AFTER_SUCCESSFUL_EXECUTION =
AUTHORIZED

Manual copying of individual scientific result files is not the canonical
handoff.

The canonical handoff is:

cm collect
-> Kaggle collector
-> handoff ZIP
-> cm import <handoff.zip>

Provenance/hash/commit mismatch is a blocker.
