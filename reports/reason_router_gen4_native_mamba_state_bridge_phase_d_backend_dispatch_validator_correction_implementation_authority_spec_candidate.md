# ContraMamba Gen4 Native Mamba State Bridge
# Phase D Backend-Dispatch Validator Correction
# Implementation Authority Specification - Candidate

STATUS =
CANDIDATE

AUTHORITY_ID =
GEN4_NATIVE_MAMBA_STATE_BRIDGE_PHASE_D_BACKEND_DISPATCH_VALIDATOR_CORRECTION_IMPLEMENTATION_AUTHORITY

PARENT_EXECUTION_AUTHORITY_COMMIT =
7a00ce1a5d08be755a89581b4724ad7fb6dd0543

RUNTIME_BUILD_TAG_CORRECTION_AUTHORITY =
9608e80efced2f6029bb4265eabf8e46df8d638c

RUNTIME_REBOUND_IMPLEMENTATION_COMMIT =
3341c1002c14b7df09923491b6e51873398d4c0c

## Failed execution evidence

FAILED_RUN =
gen4-native-mamba-phase-d-extraction-7a00ce1-v1

FAILED_RUN_COMMAND_SHA256 =
daee0b46e2c78768ed56ef270b24741545f9e879f3b161fb2bd2f1767f373c7d

FAILED_RUN_RESULT =
BLOCKED_BEFORE_SCIENTIFIC_MODEL_WORK

FAILED_RUN_FIRST_FAILURE =
scripts.reason_router_gen4_native_mamba_state_measurement.ContractError: unsupported backend

FAILED_RUN_PROVISIONING_GATE =
PASS_REPO_CLEAN

FAILED_RUN_RUNTIME_CPU_ONLY_GATE =
PASS

FAILED_RUN_RUNTIME =
Python 3.12.13 / NumPy 2.0.2 / torch 2.10.0+cu128 / Transformers 5.0.0

FAILED_RUN_CUDA_AVAILABLE =
False

FAILED_RUN_CUDA_DEVICE_COUNT =
0

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
FALSE_REJECTION_OF_FROZEN_TRANSFORMERS_V5_CPU_SLOW_PATH_DISPATCH

CURRENT_DEFECTIVE_CONTRACT =
The validator joins the source text of MambaMixer.forward and requires both:
1. "slow_forward" is present; and
2. "mamba_inner_fn" is absent.

This second condition is invalid for the exact frozen Transformers v5.0.0 source.

The frozen MambaMixer.forward legitimately mentions mamba_inner_fn while
computing fast-path availability.

Its dispatch semantics are:

- compute is_fast_path_available using the optional fast-path functions,
  including mamba_inner_fn;
- enter cuda_kernels_forward only when is_fast_path_available is true,
  the mixer weight device type contains "cuda", and torchdynamo compilation is
  not active;
- otherwise return slow_forward.

Therefore textual presence of mamba_inner_fn is not evidence that the CUDA
backend will execute.

The current CPU-only Phase D contract places the scientific model on CPU.
Under the exact frozen forward dispatch, CPU execution resolves to slow_forward.

## Test-fixture defect

The synthetic runtime fixture currently defines a simplified forward equivalent
to returning self.slow_forward directly.

That fixture omits the real frozen v5.0.0 fast-path conditional and therefore
failed to expose the false-rejection defect.

The corrected tests must represent the real dispatch structure rather than the
oversimplified source shape.

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
BACKEND_DISPATCH_VALIDATOR_ONLY

The exact frozen runtime/source checks must remain in force.

The corrected validator must no longer treat the mere textual presence of
mamba_inner_fn in MambaMixer.forward as an unsupported backend.

Instead it must validate the frozen dispatch structure fail-closed.

At minimum, the validator must establish that:

1. MambaMixer.forward retains the already-frozen module/code identity;
2. the exact frozen Mamba source byte-count and SHA256 remain required;
3. the forward dispatch contains a slow_forward fallback;
4. the CUDA fast path is gated by fast-path availability;
5. the CUDA fast path is gated by a CUDA device-type condition;
6. the CUDA fast path remains distinct from the slow_forward fallback;
7. the CPU path cannot select cuda_kernels_forward under the validated frozen
   dispatch structure.

Preferred implementation:
AST-based structural validation of MambaMixer.forward.

A narrowly exact source-structure validation is also permitted because the
entire module is already bound to exact frozen bytes.

The correction must not simply delete backend validation.

## Forbidden implementation changes

MAMBA_SOURCE_HASH_RELAXATION =
FORBIDDEN

CACHE_SOURCE_HASH_RELAXATION =
FORBIDDEN

RUNTIME_VERSION_RELAXATION =
FORBIDDEN

TORCH_VERSION_CHANGE =
FORBIDDEN

TORCH_EXPECTED_VERSION =
2.10.0+cu128

CUDA_EXECUTION_AUTHORIZATION =
FORBIDDEN

FAST_PATH_EXECUTION_AUTHORIZATION =
FORBIDDEN

MODEL_DEVICE_CHANGE =
FORBIDDEN

MODEL_DEVICE =
cpu

SOURCE_ROLE_CHANGE =
FORBIDDEN

SLOW_FORWARD_CAPTURE_CHANGE =
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

TRAINING =
FORBIDDEN

EVALUATION =
FORBIDDEN

STATISTICAL_TESTING =
FORBIDDEN

## Required test correction

The synthetic runtime fixture must exercise a forward dispatch structurally
equivalent to the frozen Transformers v5.0.0 dispatch:

- optional fast-path availability may mention mamba_inner_fn;
- CUDA fast path requires a CUDA device-type condition;
- CPU/default fallback is slow_forward.

Required positive coverage:

- exact frozen-style dispatch containing mamba_inner_fn is accepted;
- exact candidate runtime remains accepted;
- slow_forward/source-role validation remains accepted;
- exact Mamba/cache source identities remain enforced.

Required negative coverage:

- dispatch with no slow_forward fallback is rejected;
- dispatch that can enter the fast path without a CUDA device guard is rejected;
- dispatch whose CPU/default branch does not resolve to slow_forward is rejected;
- wrong Mamba source hash remains rejected;
- wrong cache source hash remains rejected;
- wrong runtime version remains rejected;
- old torch 2.10.0+cpu remains rejected;
- alternate torch build tags remain rejected.

## Extraction runner correction

Changing the measurement implementation changes its SHA256.

The extraction runner must therefore update only the dependency/provenance
bindings required by that new measurement identity.

Authorized runner deltas are limited to:

- corrected measurement SHA256 pin;
- this backend-dispatch correction authority provenance binding;
- corresponding manifest provenance field if required;
- corrected future execution-authority path;
- tests for those dependency/provenance bindings.

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
reports/reason_router_gen4_native_mamba_state_bridge_phase_d_extraction_execution_authority_backend_dispatch_correction_spec_candidate.md

## Validation boundary

VALIDATION_REQUIRED =
YES

Required validation:

- py_compile of all four authorized files;
- full measurement synthetic/static test suite;
- full extraction synthetic/static test suite;
- corrected backend-dispatch positive and negative tests;
- exact runtime acceptance/rejection regression tests;
- exact frozen source-role regression tests;
- git diff --check;
- exact intended four-file change scope.

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

PASS_READY_FOR_SEPARATE_BACKEND_DISPATCH_CORRECTED_PHASE_D_EXECUTION_AUTHORITY

This verdict establishes only that the runtime/backend validator correctly
recognizes the exact frozen Transformers v5.0.0 CPU slow-path dispatch.

It is not Phase D execution success and is not scientific evidence.
