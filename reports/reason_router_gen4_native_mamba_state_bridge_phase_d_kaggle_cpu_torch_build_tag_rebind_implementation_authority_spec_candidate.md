# ContraMamba Gen4 Native Mamba State Bridge
# Phase D Kaggle CPU Torch Build-Tag Rebind
# Implementation Authority Specification - Candidate

STATUS =
CANDIDATE

AUTHORITY_ID =
GEN4_NATIVE_MAMBA_STATE_BRIDGE_PHASE_D_KAGGLE_CPU_TORCH_BUILD_TAG_REBIND_IMPLEMENTATION_AUTHORITY

PARENT_EXECUTION_AUTHORITY_COMMIT =
60d72527667214a5ba80bb8291661b8396e29143

CURRENT_IMPLEMENTATION_COMMIT =
ff9fe86c3ea13366a129ab1e7f6c2b13686f631b

PRIOR_TORCH_BUILD_TAG_CORRECTION_AUTHORITY =
9608e80efced2f6029bb4265eabf8e46df8d638c

BACKEND_DISPATCH_VALIDATOR_CORRECTION_AUTHORITY =
94c23155b8422d23ae01c43c0b63c5f696bd52d1

## Failed execution evidence

FAILED_RUN =
gen4-native-mamba-phase-d-extraction-60d7252-v3

FAILED_RUN_COMMAND_SHA256 =
4ca5267531e61221c61e27ddb3bce803e09a72a1d8a37d80575eb5805480ab81

FAILED_RUN_RESULT =
BLOCKED_AT_EXACT_RUNTIME_VERSION_GATE_BEFORE_SCIENTIFIC_MODEL_WORK

FAILED_RUN_FIRST_BLOCKER =
torch mismatch: 2.10.0+cpu

FAILED_RUN_RELEASE_METADATA_GATE =
PASS

FAILED_RUN_RELEASE_ASSET_GATE =
PASS

FAILED_RUN_PROVISIONING_GATE =
PASS_REPO_CLEAN

FAILED_RUN_PROVISIONING_CLEANUP =
PASS

FAILED_RUN_CHECKPOINT_DESERIALIZATION =
NO

FAILED_RUN_MODEL_CONSTRUCTION =
NO

FAILED_RUN_SCIENTIFIC_FORWARD_COUNT =
0

FAILED_RUN_NATIVE_STATE_EXTRACTION =
NO

FAILED_RUN_REUSE =
FORBIDDEN_SINGLE_USE_IDENTITY

SCIENTIFIC_CONCLUSION =
NONE

Earlier v1 and v2 run identities failed at transport/provenance-wrapper
boundaries before scientific execution and remain single-use provenance only.

## Current read-only Kaggle runtime evidence

RUNTIME_DIAGNOSTIC_EXECUTION_HEAD =
60d72527667214a5ba80bb8291661b8396e29143

RUNTIME_DIAGNOSTIC_WORKTREE =
CLEAN

PYTHON_VERSION =
3.12.13

NUMPY_VERSION =
2.0.2

OBSERVED_TORCH_VERSION =
2.10.0+cpu

TRANSFORMERS_VERSION =
5.0.0

TORCH_CUDA_IS_AVAILABLE =
False

TORCH_CUDA_DEVICE_COUNT =
0

MAMBA_SOURCE_BYTES =
39500

MAMBA_SOURCE_SHA256 =
4c972b30f3c2cca977824fcc6891f956cd4387b6383aa7336848fbc5f2db1d83

CACHE_SOURCE_BYTES =
60432

CACHE_SOURCE_SHA256 =
6c123bbe3d23500462f0b617119a8231aa054b6d5475b295443a45f34466e6bc

MAMBA_SOURCE_IDENTITY =
PASS_EXACT_FROZEN_BYTES

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
YES_EXACTLY_CU128_TO_CPU

SCIENTIFIC_FAILURE =
NO

## Root cause

DEFECT_CLASS =
FROZEN_RUNTIME_BUILD_TAG_NO_LONGER_MATCHES_CURRENT_KAGGLE_CPU_IMAGE

ROOT_CAUSE =
The frozen Phase D implementation requires the exact torch version string
2.10.0+cu128, while the current Kaggle CPU runtime now exposes exactly
2.10.0+cpu.

Python, NumPy, Transformers, the frozen Mamba source bytes, and the frozen
cache source bytes remain identical to the established scientific runtime
contract.

The observed runtime is CPU-only:
torch.cuda.is_available() is False and torch.cuda.device_count() is 0.

This authority does not authorize generic torch compatibility, multiple
accepted builds, suffix-insensitive matching, CUDA execution, package
installation, or any scientific-contract relaxation.

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

OLD_EXPECTED_TORCH_VERSION =
2.10.0+cu128

NEW_EXPECTED_TORCH_VERSION =
2.10.0+cpu

The corrected validator must continue to use exact runtime-version equality.

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

PYTHON_VERSION_RELAXATION =
FORBIDDEN

NUMPY_VERSION_RELAXATION =
FORBIDDEN

TRANSFORMERS_VERSION_RELAXATION =
FORBIDDEN

SOURCE_HASH_RELAXATION =
FORBIDDEN

SOURCE_BYTE_COUNT_RELAXATION =
FORBIDDEN

BACKEND_DISPATCH_VALIDATOR_CHANGE =
FORBIDDEN

BACKEND_DISPATCH_VALIDATOR =
AST_STRUCTURAL

AST_SOURCE_NORMALIZATION =
TEXTWRAP_DEDENT

MAMBA_SOURCE_ROLE_CHANGE =
FORBIDDEN

OBSERVER_SEMANTICS_CHANGE =
FORBIDDEN

STATE_TIMING_CHANGE =
FORBIDDEN

PRIMARY_LAYER_CHANGE =
FORBIDDEN

KINEMATIC_DEFINITION_CHANGE =
FORBIDDEN

SUPPORT_WINDOW_CHANGE =
FORBIDDEN

## CPU-only invariant

ACCELERATOR =
NONE

GPU_EXECUTION =
FORBIDDEN

SCIENTIFIC_MODEL_DEVICE =
cpu

CUDA_MODEL_EXECUTION =
FORBIDDEN

AUTOMATIC_ACCELERATOR_SELECTION =
FORBIDDEN

Future Phase D execution wrapper must explicitly set:

CUDA_VISIBLE_DEVICES =
EMPTY

NVIDIA_VISIBLE_DEVICES =
void

and must fail closed unless:

TORCH_CUDA_IS_AVAILABLE =
False

TORCH_CUDA_DEVICE_COUNT =
0

The new +cpu build binding does not authorize any change to scientific model
device selection.

## Required test correction

POSITIVE_RUNTIME =
Python 3.12.13 / NumPy 2.0.2 / torch 2.10.0+cpu / Transformers 5.0.0

Required positive/static coverage:

- EXPECTED_VERSIONS["torch"] is exactly "2.10.0+cpu";
- the exact current runtime dictionary is accepted;
- Python/NumPy/Transformers values remain unchanged;
- exact Mamba/cache hashes and byte counts remain unchanged;
- AST backend-dispatch validation remains unchanged;
- source-role, observer, state-timing, layer, and kinematic tests remain unchanged
  in meaning.

Required negative coverage:

- torch 2.10.0+cu128 is rejected after correction;
- any other torch build tag is rejected;
- wrong Python is rejected;
- wrong NumPy is rejected;
- wrong Transformers is rejected;
- wrong Mamba source identity is rejected;
- wrong cache source identity is rejected;
- malformed backend dispatch remains rejected.

## Extraction runner correction contract

The measurement implementation SHA256 will change.

The extraction runner must therefore update only the bounded dependency and
provenance bindings required by this correction.

AUTHORIZED_RUNNER_DELTAS =

- corrected measurement SHA256 pin;
- new CPU build-tag rebind authority provenance binding;
- corresponding manifest provenance field;
- corrected future execution-authority path;
- tests for those exact bindings.

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

FUTURE_EXECUTION_AUTHORITY_PATH =
reports/reason_router_gen4_native_mamba_state_bridge_phase_d_extraction_execution_authority_cpu_runtime_rebind_spec_candidate.md

The execution authority frozen at
60d72527667214a5ba80bb8291661b8396e29143
does not authorize implementation bytes modified under this authority.

## Validation boundary

VALIDATION_REQUIRED =
YES

Required validation:

- python py_compile on all four authorized files;
- full native-Mamba measurement synthetic/static tests;
- full Phase D extraction synthetic/static tests;
- exact +cpu positive runtime test;
- exact +cu128 rejection regression;
- alternate torch build rejection;
- frozen source identity regression;
- backend-dispatch regression;
- git diff --check;
- exact four-file implementation scope.

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

PACKAGE_UPGRADE_ALLOWED =
NO

PACKAGE_DOWNGRADE_ALLOWED =
NO

CODEX_ALLOWED =
NO

## Strongest permitted implementation verdict

PASS_READY_FOR_SEPARATE_CPU_RUNTIME_REBOUND_PHASE_D_EXECUTION_AUTHORITY

This verdict establishes only that the exact current Kaggle CPU torch build
has been rebound while preserving all frozen scientific and source contracts.

It is not a Phase D execution result, artifact validation result, or scientific
conclusion.
