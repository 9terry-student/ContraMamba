# ContraMamba Gen4 Native Mamba State Bridge
# Phase D Kaggle Torch Build-Tag Correction
# Implementation Authority Specification - Candidate

STATUS =
CANDIDATE

AUTHORITY_ID =
GEN4_NATIVE_MAMBA_STATE_BRIDGE_PHASE_D_KAGGLE_TORCH_BUILD_TAG_CORRECTION_IMPLEMENTATION_AUTHORITY

PARENT_EXECUTION_AUTHORITY_COMMIT =
8bb4422d5183d1431e07c83d96cf00f33b740be2

FAILED_RUN =
gen4-native-mamba-phase-d-extraction-8bb4422-v1

FAILED_RUN_COMMAND_SHA256 =
c07af65b4e53ed39b0c3315f09b72d43e5005a14aa71eb644219b0fc86773674

FAILED_RUN_RESULT =
BLOCKED_BEFORE_SCIENTIFIC_MODEL_WORK

FAILED_RUN_BLOCKER =
scripts.reason_router_gen4_native_mamba_state_measurement.ContractError: runtime version

FAILED_RUN_PROVISIONING =
PASS_EXACT_BYTES

FAILED_RUN_PROVISIONING_CLEANUP =
PASS

FAILED_RUN_SCIENTIFIC_FORWARD_COUNT =
0

FAILED_RUN_CHECKPOINT_DESERIALIZATION =
NO

FAILED_RUN_MODEL_CONSTRUCTION =
NO

FAILED_RUN_NATIVE_STATE_EXTRACTION =
NO

SCIENTIFIC_CONCLUSION =
NONE

## Runtime diagnostic evidence

AUTHORITY_EXPECTED_TORCH_VERSION =
2.10.0+cpu

KAGGLE_OBSERVED_TORCH_VERSION =
2.10.0+cu128

PYTHON_VERSION =
3.12.13

NUMPY_VERSION =
2.0.2

TRANSFORMERS_VERSION =
5.0.0

TORCH_VERSION_CUDA =
12.8

CUDA_VISIBLE_DEVICES =
EMPTY

NVIDIA_VISIBLE_DEVICES =
void

TORCH_CUDA_IS_AVAILABLE =
False

TORCH_CUDA_DEVICE_COUNT =
0

EFFECTIVE_CPU_ONLY =
PASS

AVAILABLE_PYTHON_INTERPRETERS =
/usr/bin/python3
/usr/local/bin/python

EXACT_TORCH_2_10_0_CPU_RUNTIME_DISCOVERED =
NO

KAGGLE_RUNTIME_CANDIDATE =
Python 3.12.13 / NumPy 2.0.2 / torch 2.10.0+cu128 / Transformers 5.0.0

## Frozen Transformers source identity under candidate runtime

MAMBA_SOURCE_BYTES =
39500

MAMBA_SOURCE_SHA256 =
4c972b30f3c2cca977824fcc6891f956cd4387b6383aa7336848fbc5f2db1d83

MAMBA_SOURCE_IDENTITY =
PASS_EXACT_FROZEN_BYTES

CACHE_SOURCE_BYTES =
60432

CACHE_SOURCE_SHA256 =
6c123bbe3d23500462f0b617119a8231aa054b6d5475b295443a45f34466e6bc

CACHE_SOURCE_IDENTITY =
PASS_EXACT_FROZEN_BYTES

SOURCE_DRIFT =
NO

TRANSFORMERS_DRIFT =
NO

PYTHON_DRIFT =
NO

NUMPY_DRIFT =
NO

TORCH_MAJOR_MINOR_PATCH_DRIFT =
NO

TORCH_BUILD_TAG_DRIFT =
YES_EXACTLY_CPU_TO_CU128

SCIENTIFIC_FAILURE =
NO

## Root cause

DEFECT_CLASS =
FROZEN_RUNTIME_BUILD_TAG_NO_LONGER_AVAILABLE_IN_CURRENT_KAGGLE_IMAGE

ROOT_CAUSE =
The Phase D runtime validator correctly fail-closed because the frozen authority
requires the exact torch version string 2.10.0+cpu, while the current Kaggle
runtime exposes only torch 2.10.0+cu128 through all discovered Python
interpreters.

The candidate runtime was separately audited with CUDA hidden and unavailable.
The exact frozen Transformers Mamba and cache source bytes remain identical.

This evidence does not authorize generic CUDA builds, generic torch version
ranges, suffix-insensitive comparison, or runtime-version relaxation.

## Exact authorized implementation scope

AUTHORIZED_FILES =
scripts/reason_router_gen4_native_mamba_state_measurement.py
tests/test_reason_router_gen4_native_mamba_state_measurement.py
scripts/reason_router_gen4_native_mamba_state_extraction.py
tests/test_reason_router_gen4_native_mamba_state_extraction.py

NO_OTHER_IMPLEMENTATION_FILE_CHANGES =
REQUIRED

## Measurement correction contract

MEASUREMENT_CORRECTION =
EXACT_TORCH_BUILD_TAG_REBIND_ONLY

The measurement runtime contract may change exactly:

OLD_EXPECTED_TORCH_VERSION =
2.10.0+cpu

NEW_EXPECTED_TORCH_VERSION =
2.10.0+cu128

The corrected runtime validator must continue to use exact dictionary equality
for runtime versions.

GENERIC_VERSION_RANGE =
FORBIDDEN

TORCH_LOCAL_VERSION_SUFFIX_IGNORING =
FORBIDDEN

PREFIX_MATCHING =
FORBIDDEN

SEMVER_COMPATIBILITY_MATCHING =
FORBIDDEN

MULTIPLE_ALLOWED_TORCH_VERSIONS =
FORBIDDEN

SOURCE_HASH_RELAXATION =
FORBIDDEN

SOURCE_BYTE_COUNT_RELAXATION =
FORBIDDEN

TRANSFORMERS_VERSION_RELAXATION =
FORBIDDEN

PYTHON_VERSION_RELAXATION =
FORBIDDEN

NUMPY_VERSION_RELAXATION =
FORBIDDEN

MAMBA_SOURCE_ROLE_CHANGE =
FORBIDDEN

OBSERVER_SEMANTICS_CHANGE =
FORBIDDEN

STATE_TIMING_CHANGE =
FORBIDDEN

KINEMATIC_DEFINITION_CHANGE =
FORBIDDEN

PRIMARY_LAYER_CHANGE =
FORBIDDEN

## CPU-only execution invariant

The cu128 build tag does not authorize CUDA execution.

Future scientific execution remains CPU-only and must preserve:

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

GPU_EXECUTION =
FORBIDDEN

ACCELERATOR =
NONE

No code change may introduce CUDA device selection or automatic accelerator
selection.

## Test correction contract

Required positive/static coverage:

- EXPECTED_VERSIONS["torch"] is exactly "2.10.0+cu128";
- exact candidate runtime dictionary is accepted;
- existing frozen Python/NumPy/Transformers values remain unchanged;
- exact Mamba/cache source identity checks remain unchanged;
- source-role validator tests remain unchanged in meaning;
- observer/kinematic tests remain unchanged in meaning.

Required negative coverage:

- torch 2.10.0+cpu is rejected after correction;
- any other torch build tag is rejected;
- wrong Python version remains rejected;
- wrong NumPy version remains rejected;
- wrong Transformers version remains rejected;
- wrong Mamba/cache source hashes remain rejected.

## Phase D runner correction

The measurement implementation SHA256 will change.

Therefore the Phase D extraction runner must update the frozen measurement SHA
pin in the same bounded implementation stage.

RUNNER_SCIENTIFIC_EXTRACTION_LOGIC_CHANGE =
FORBIDDEN

RUNNER_SUPPORT_PLAN_CHANGE =
FORBIDDEN

RUNNER_ENDPOINT_DEFINITION_CHANGE =
FORBIDDEN

RUNNER_FORWARD_COUNT_CHANGE =
FORBIDDEN

RUNNER_MODEL_SELECTION_CHANGE =
FORBIDDEN

RUNNER_PRIMARY_LAYER_CHANGE =
FORBIDDEN

RUNNER_DEVICE_SELECTION_CHANGE =
FORBIDDEN

Authorized runner deltas are limited to:

- corrected measurement SHA256 pin;
- this runtime-correction authority provenance binding;
- corresponding manifest provenance field if required;
- corrected future execution-authority path;
- tests for those dependency/provenance bindings.

FUTURE_EXECUTION_AUTHORITY_PATH =
reports/reason_router_gen4_native_mamba_state_bridge_phase_d_extraction_execution_authority_runtime_correction_spec_candidate.md

The execution authority frozen at
8bb4422d5183d1431e07c83d96cf00f33b740be2
does not authorize corrected runtime-binding implementation bytes.

## Validation boundary

VALIDATION_REQUIRED =
YES

Required implementation validation:

- full measurement synthetic/static tests;
- full extraction synthetic/static tests;
- exact runtime-version acceptance/rejection tests;
- exact frozen Transformers source-role regression;
- git diff --check;
- exact intended-file scope review.

CANONICAL_GEN4_MODEL_FORWARD_ALLOWED =
NO

CHECKPOINT_DESERIALIZATION_ALLOWED =
NO

MODEL_CONSTRUCTION_ALLOWED =
NO

NATIVE_STATE_EXTRACTION_ALLOWED =
NO

PRIMARY_KINEMATIC_ARTIFACT_COMPUTATION_ALLOWED =
NO

STATISTICAL_TESTING_ALLOWED =
NO

TRAINING_ALLOWED =
NO

EVALUATION_ALLOWED =
NO

KAGGLE_SCIENTIFIC_EXECUTION_ALLOWED =
NO

PACKAGE_INSTALLATION_ALLOWED =
NO

PACKAGE_DOWNGRADE_ALLOWED =
NO

PACKAGE_UPGRADE_ALLOWED =
NO

CODEX_ALLOWED =
NO

## Strongest permitted implementation verdict

PASS_READY_FOR_SEPARATE_RUNTIME_CORRECTED_PHASE_D_EXECUTION_AUTHORITY

This verdict establishes only that the exact current Kaggle torch build tag has
been rebound under a CPU-only execution contract.

It is not a Phase D execution result and is not a scientific conclusion.
