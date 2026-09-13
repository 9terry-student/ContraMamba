# ContraMamba Gen4 Native Mamba State Bridge
# Cache/Recurrent-Validator-Corrected Phase D Extraction Execution Authority
# Candidate

STATUS =
CANDIDATE

AUTHORITY_ID =
GEN4_NATIVE_MAMBA_STATE_BRIDGE_PHASE_D_EXTRACTION_EXECUTION_AUTHORITY

SCIENTIFIC_EXECUTION_ALLOWED =
YES_BOUNDED

## Corrected implementation binding

PHASE_D_RUNNER_IMPLEMENTATION_COMMIT =
c7ae7fac4c64e9bd64adcae819a3da3dd46f17f7

PHASE_D_RUNNER_SHA256 =
653f96713d8bdc776cdf03733ae230784e413f960f58a9c120f00cb8c6a6d3eb

PHASE_D_RUNNER_PATH =
scripts/reason_router_gen4_native_mamba_state_extraction.py

CORRECTED_MEASUREMENT_SHA256 =
7729424f03058b86b4f120dc0e6da573d6c996b0877858f2d6d38aa94dac268c

CORRECTED_MEASUREMENT_PATH =
scripts/reason_router_gen4_native_mamba_state_measurement.py

CACHE_RECURRENT_VALIDATOR_CORRECTION_AUTHORITY =
c4e143e9593503bc9520c864e2a9afa32e266e29

CORRECTED_IMPLEMENTATION_COMMIT =
c7ae7fac4c64e9bd64adcae819a3da3dd46f17f7

CORRECTED_IMPLEMENTATION_VALIDATION =
PASS_75_SYNTHETIC_STATIC_TESTS

MEASUREMENT_TESTS =
43_PASS

EXTRACTION_TESTS =
32_PASS

PY_COMPILE =
PASS

GIT_DIFF_CHECK =
PASS

## Parent execution authority

PARENT_PHASE_D_EXECUTION_AUTHORITY_COMMIT =
75b56b5b9390c0d1dcbbc38330d5dfdba8f44d75

PARENT_PHASE_D_EXECUTION_AUTHORITY_STATUS =
SUPERSEDED_FOR_CACHE_RECURRENT_VALIDATOR_CORRECTED_IMPLEMENTATION_BYTES

The parent authority remains the source of the unchanged Phase D scientific
design, population, checkpoint, tokenizer, runtime, support window, endpoint,
and output contracts.

The only scientific-execution-relevant implementation change since that
authority is the correction of the cache/recurrent runtime validator and its
required dependency/provenance rebind.

## Failed execution evidence motivating correction

FAILED_RUN =
gen4-native-mamba-phase-d-extraction-75b56b5-v2

FAILED_RUN_COMMAND_SHA256 =
d6ca4b3feb4c106b31ea1c04d86c6a7a11fa6c847a8ffbb4764cbe132a1ce7bb

FAILED_RUN_FIRST_FAILURE =
scripts.reason_router_gen4_native_mamba_state_measurement.ContractError: cache/recurrent ambiguity

FAILED_RUN_PHASE_D_RUNNER_ENTERED =
YES

FAILED_RUN_CHECKPOINT_DESERIALIZATION =
NO

FAILED_RUN_MODEL_CONSTRUCTION =
NO

FAILED_RUN_SCIENTIFIC_FORWARD_COUNT =
0

FAILED_RUN_NATIVE_STATE_EXTRACTION =
NO

FAILED_RUN_SCIENTIFIC_CONCLUSION =
NONE

FAILED_RUN_REUSE =
FORBIDDEN_SINGLE_USE_IDENTITY

## Corrected validator result

DEFECT_CORRECTED =
YES

DEFECT_CLASS =
FALSE_REJECTION_DUE_TO_CACHE_RECURRENT_SOURCE_ROLE_MISBINDING

The exact frozen Transformers v5.0.0 source identities remain required.

The corrected validator establishes Mamba cache/recurrent roles from the
exact-hash-bound Mamba source rather than requiring Mamba-specific state names
to occur in transformers/cache_utils.py.

The corrected validator retains:

- exact runtime version matching;
- exact Mamba source SHA256 and byte count;
- exact cache_utils source SHA256 and byte count;
- import/distribution-root validation;
- slow_forward code identity;
- recurrent update/readout/persistence source-role validation;
- CPU slow-path backend-dispatch validation;
- fail-closed cache/recurrent-role validation.

SOURCE_HASH_RELAXATION =
NONE

RUNTIME_VERSION_RELAXATION =
NONE

BACKEND_RELAXATION =
NONE

SCIENTIFIC_LOGIC_CHANGE =
NONE

## Exact runtime binding

EXECUTION_VENUE =
KAGGLE_NOTEBOOK_CPU_ONLY

PYTHON_VERSION =
3.12.13

NUMPY_VERSION =
2.0.2

TORCH_VERSION =
2.10.0+cpu

TRANSFORMERS_VERSION =
5.0.0

RUNTIME_VERSION_MATCHING =
EXACT_DICTIONARY_EQUALITY

ACCELERATOR =
NONE

GPU =
OFF

CUDA_VISIBLE_DEVICES =
EMPTY

NVIDIA_VISIBLE_DEVICES =
void

TORCH_CUDA_IS_AVAILABLE =
False

TORCH_CUDA_DEVICE_COUNT =
0

SCIENTIFIC_MODEL_DEVICE =
cpu

PACKAGE_INSTALLATION =
FORBIDDEN

PACKAGE_UPGRADE =
FORBIDDEN

PACKAGE_DOWNGRADE =
FORBIDDEN

## Frozen source identities

TRANSFORMERS_MAMBA_SOURCE_SHA256 =
4c972b30f3c2cca977824fcc6891f956cd4387b6383aa7336848fbc5f2db1d83

TRANSFORMERS_MAMBA_SOURCE_BYTES =
39500

TRANSFORMERS_CACHE_SOURCE_SHA256 =
6c123bbe3d23500462f0b617119a8231aa054b6d5475b295443a45f34466e6bc

TRANSFORMERS_CACHE_SOURCE_BYTES =
60432

## Scientific execution scope

EXECUTION_PURPOSE =
Produce the frozen Gen4 native-Mamba recurrent-state measurement bundle for
the prespecified mechanistic bridge.

REPRESENTATIVE_SEED =
180

REPRESENTATIVE_ARM =
G3-GROUP-D-HALF

REPRESENTATIVE_CHECKPOINT_SHA256 =
1ff3fcf2ebd754ab6f9483d6a9982b9b04b9a4eb3357f9f8cdbe2b30399e7d2f

PRIMARY_LAYER =
11

CANONICAL_GEN4_ROW_COUNT =
1800

SOURCE_PAIR_COUNT =
300

CELL_COUNT =
6

EXPECTED_BACKBONE_FORWARD_COUNT =
1800

EXPECTED_SUPPORT_STATE_VECTOR_COUNT =
14717

EXPECTED_ENDPOINT_ROW_COUNT =
3600

MODEL_REPLICATION_COUNT =
1

MODEL_FORWARD_MODE =
BACKBONE_ONLY_MODEL_MAMBA

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

STATE_SHAPE =
[1,1536,16]

FLATTENED_STATE_SIZE =
24576

PRIMARY_LAYER =
11

SUPPORT_WINDOW =
[a-1,a+4]

AUTHORIZED_ENDPOINTS =
POST4_SPEED
POST4_TURNING
POST4_PATH_EFFICIENCY

## Required successful output bundle

OUTPUT_FILE_COUNT =
5

OUTPUT_FILES =
manifest.json
support_state_rows.jsonl
support_states.npy
kinematic_endpoints.jsonl
SHA256SUMS.txt

PARTIAL_OUTPUT =
NOT_EVIDENCE

## Execution boundary

ONE_NEW_PHASE_D_EXECUTION =
AUTHORIZED

PRIOR_FAILED_RUN_REUSE =
FORBIDDEN

NEW_RUN_IDENTITY =
REQUIRED

GPU_ENABLEMENT =
FORBIDDEN

SCIENTIFIC_EXECUTION_BEFORE_THIS_AUTHORITY_FREEZE =
FORBIDDEN

## Strongest permitted Phase D verdict

PASS_READY_FOR_PHASE_E_MEASUREMENT_ARTIFACT_FREEZE

A successful execution establishes execution success and a valid Phase D
measurement bundle only.

It does not establish a statistical, causal, or mechanistic scientific claim.
