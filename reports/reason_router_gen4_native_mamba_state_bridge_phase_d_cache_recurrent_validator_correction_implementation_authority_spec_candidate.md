# ContraMamba Gen4 Native Mamba State Bridge
# Phase D Cache/Recurrent Validator Correction
# Implementation Authority Specification - Candidate

STATUS =
CANDIDATE

AUTHORITY_ID =
GEN4_NATIVE_MAMBA_STATE_BRIDGE_PHASE_D_CACHE_RECURRENT_VALIDATOR_CORRECTION_IMPLEMENTATION_AUTHORITY

PARENT_EXECUTION_AUTHORITY_COMMIT =
75b56b5b9390c0d1dcbbc38330d5dfdba8f44d75

PARENT_EXECUTION_AUTHORITY =
reports/reason_router_gen4_native_mamba_state_bridge_phase_d_extraction_execution_authority_cpu_runtime_rebind_spec_candidate.md

## Failed execution evidence

FAILED_RUN =
gen4-native-mamba-phase-d-extraction-75b56b5-v2

FAILED_RUN_COMMAND_SHA256 =
d6ca4b3feb4c106b31ea1c04d86c6a7a11fa6c847a8ffbb4764cbe132a1ce7bb

FAILED_RUN_RESULT =
BLOCKED_AT_CACHE_RECURRENT_RUNTIME_VALIDATOR_BEFORE_SCIENTIFIC_MODEL_WORK

FAILED_RUN_FIRST_FAILURE =
scripts.reason_router_gen4_native_mamba_state_measurement.ContractError: cache/recurrent ambiguity

FAILED_RUN_RELEASE_METADATA_GATE =
PASS

FAILED_RUN_RELEASE_ASSET_GATE =
PASS

FAILED_RUN_PROVISIONING_GATE =
PASS_REPO_CLEAN

FAILED_RUN_RUNTIME_CPU_ONLY_GATE =
PASS

FAILED_RUN_RUNTIME =
Python 3.12.13 / NumPy 2.0.2 / torch 2.10.0+cpu / Transformers 5.0.0

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

FAILED_RUN_PROVISIONING_CLEANUP =
PASS

FAILED_RUN_SCIENTIFIC_CONCLUSION =
NONE

FAILED_RUN_REUSE =
FORBIDDEN_SINGLE_USE_IDENTITY

## Frozen source identity

TRANSFORMERS_VERSION =
5.0.0

MAMBA_SOURCE_SHA256 =
4c972b30f3c2cca977824fcc6891f956cd4387b6383aa7336848fbc5f2db1d83

MAMBA_SOURCE_BYTES =
39500

CACHE_SOURCE_SHA256 =
6c123bbe3d23500462f0b617119a8231aa054b6d5475b295443a45f34466e6bc

CACHE_SOURCE_BYTES =
60432

SOURCE_IDENTITY_DRIFT =
NO

SOURCE_HASH_RELAXATION =
FORBIDDEN

SOURCE_BYTE_COUNT_RELAXATION =
FORBIDDEN

## Defect

DEFECT_CLASS =
FALSE_REJECTION_DUE_TO_CACHE_RECURRENT_SOURCE_ROLE_MISBINDING

The current runtime validator first establishes the exact frozen
Transformers v5.0.0 Mamba and cache_utils byte identities.

After those exact identity gates pass, it additionally requires the text of
transformers/cache_utils.py to contain both:

- "conv_states"
- "ssm_states"

and rejects the runtime as "cache/recurrent ambiguity" otherwise.

That source-role assumption is incorrect for the exact frozen Mamba runtime.

In Transformers v5.0.0, the Mamba convolution and selective-SSM recurrent
state objects are defined and used by the Mamba implementation in
transformers/models/mamba/modeling_mamba.py.

The exact frozen Mamba implementation distinguishes:

- convolution cache state: conv_states
- selective-SSM recurrent state: ssm_states

and slow_forward persists the recurrent state through
cache_params.ssm_states.

Therefore absence of those two literals from cache_utils.py is not evidence
of cache/recurrent ambiguity.

## Test-fixture defect

The synthetic runtime fixture currently writes:

conv_states = []
ssm_states = []

into its synthetic cache_utils.py.

This artificial placement makes the defective production validator pass and
therefore masks the real frozen-runtime failure.

The corrected fixture must represent the actual source-role placement rather
than inserting Mamba state names into cache_utils.py solely to satisfy the
validator.

## Authorized implementation scope

AUTHORIZED_FILES =
scripts/reason_router_gen4_native_mamba_state_measurement.py
tests/test_reason_router_gen4_native_mamba_state_measurement.py
scripts/reason_router_gen4_native_mamba_state_extraction.py
tests/test_reason_router_gen4_native_mamba_state_extraction.py

NO_OTHER_IMPLEMENTATION_FILE_CHANGES =
REQUIRED

## Measurement correction contract

MEASUREMENT_CORRECTION =
CACHE_RECURRENT_ROLE_VALIDATOR_ONLY

The correction must retain all existing exact runtime-version, distribution
root, import root, Mamba source hash/byte-count, cache_utils source
hash/byte-count, slow-forward identity, source-role, backend-dispatch, and
CPU-only checks.

The correction must remove the invalid requirement that cache_utils.py itself
contain both conv_states and ssm_states.

It must replace that requirement with fail-closed validation of the exact
frozen Mamba source roles.

The corrected validation must establish that the frozen Mamba implementation
keeps convolution-cache state and selective-SSM recurrent state distinct.

At minimum it must establish from the already exact-hash-bound Mamba source
that:

1. conv_states and ssm_states are distinct Mamba cache roles;
2. slow_forward uses the selective-SSM recurrent role through ssm_states;
3. convolution-state use is not accepted as the recurrent-state capture role;
4. the frozen recurrent update / capture / final persistence contract remains
   unchanged.

AST or narrowly exact source-structure validation is permitted.

The correction must not weaken or remove exact source-identity validation.

## Forbidden scientific changes

RUNTIME_VERSION_RELAXATION =
FORBIDDEN

TORCH_VERSION_CHANGE =
FORBIDDEN

TORCH_EXPECTED_VERSION =
2.10.0+cpu

MAMBA_SOURCE_CHANGE =
FORBIDDEN

CACHE_UTILS_SOURCE_CHANGE =
FORBIDDEN

BACKEND_DISPATCH_CHANGE =
FORBIDDEN

SOURCE_ROLE_SEMANTICS_CHANGE =
FORBIDDEN

OBSERVER_SEMANTICS_CHANGE =
FORBIDDEN

STATE_OBJECT_CHANGE =
FORBIDDEN

STATE_TIMING_CHANGE =
FORBIDDEN

CAPTURE_LINE_CHANGE =
FORBIDDEN

PRIMARY_LAYER_CHANGE =
FORBIDDEN

SUPPORT_WINDOW_CHANGE =
FORBIDDEN

KINEMATIC_ENDPOINT_CHANGE =
FORBIDDEN

SCIENTIFIC_POPULATION_CHANGE =
FORBIDDEN

CHECKPOINT_CHANGE =
FORBIDDEN

TOKENIZER_CHANGE =
FORBIDDEN

MODEL_DEVICE_CHANGE =
FORBIDDEN

MODEL_DEVICE =
cpu

TRAINING =
FORBIDDEN

EVALUATION =
FORBIDDEN

STATISTICAL_TESTING =
FORBIDDEN

KAGGLE_SCIENTIFIC_EXECUTION =
FORBIDDEN

## Required regression-test correction

The synthetic runtime fixture must no longer depend on placing conv_states and
ssm_states in synthetic cache_utils.py.

Required positive coverage:

- frozen-style Mamba source with distinct conv_states and ssm_states roles is
  accepted;
- exact CPU runtime binding remains accepted;
- existing slow_forward source-role validation remains accepted;
- exact Mamba and cache_utils hashes remain enforced.

Required negative coverage:

- missing selective-SSM recurrent-state role is rejected;
- convolution cache cannot be accepted as the recurrent-state role;
- broken recurrent persistence through ssm_states is rejected;
- wrong Mamba source identity remains rejected;
- wrong cache_utils source identity remains rejected;
- wrong runtime versions remain rejected;
- unsupported backend dispatch remains rejected.

## Extraction runner dependency rebind

Changing the measurement implementation changes its SHA256.

The extraction runner may therefore change only the dependency/provenance
bindings required by the corrected measurement identity.

Authorized extraction-runner deltas are limited to:

- corrected measurement SHA256 pin;
- this cache/recurrent validator correction authority provenance binding;
- corresponding manifest provenance field;
- future corrected Phase D execution-authority path;
- tests for those exact dependency/provenance bindings.

RUNNER_SCIENTIFIC_LOGIC_CHANGE =
FORBIDDEN

RUNNER_MODEL_LOADING_CHANGE =
FORBIDDEN

RUNNER_MODEL_DEVICE_CHANGE =
FORBIDDEN

RUNNER_SUPPORT_PLAN_CHANGE =
FORBIDDEN

RUNNER_FORWARD_COUNT_CHANGE =
FORBIDDEN

RUNNER_ENDPOINT_COMPUTATION_CHANGE =
FORBIDDEN

FUTURE_EXECUTION_AUTHORITY_PATH =
reports/reason_router_gen4_native_mamba_state_bridge_phase_d_extraction_execution_authority_cache_recurrent_validator_correction_spec_candidate.md

## Validation boundary

VALIDATION_REQUIRED =
YES

Required validation:

- py_compile of the four authorized implementation/test files;
- full measurement synthetic/static test suite;
- full extraction synthetic/static test suite;
- cache/recurrent positive and negative regression coverage;
- exact runtime/source identity regression coverage;
- git diff --check;
- exact intended four-file implementation scope.

CANONICAL_MODEL_CONSTRUCTION_ALLOWED =
NO

CHECKPOINT_DESERIALIZATION_ALLOWED =
NO

CANONICAL_SCIENTIFIC_FORWARD_ALLOWED =
NO

NATIVE_STATE_EXTRACTION_ALLOWED =
NO

PRIMARY_KINEMATIC_ARTIFACT_COMPUTATION_ALLOWED =
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

PASS_READY_FOR_SEPARATE_CACHE_RECURRENT_CORRECTED_PHASE_D_EXECUTION_AUTHORITY

This verdict establishes only that the frozen-runtime validator correctly
distinguishes the Mamba convolution cache from the selective-SSM recurrent
state under the exact frozen Transformers v5.0.0 source identity.

It is not Phase D execution success and is not scientific evidence.
