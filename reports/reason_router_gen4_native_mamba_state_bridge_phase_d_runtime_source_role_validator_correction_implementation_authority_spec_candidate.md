# ContraMamba Gen4 Native Mamba State Bridge
# Phase D Runtime Source-Role Validator Correction
# Implementation Authority Specification - Candidate

STATUS =
CANDIDATE

AUTHORITY_ID =
GEN4_NATIVE_MAMBA_STATE_BRIDGE_PHASE_D_RUNTIME_SOURCE_ROLE_VALIDATOR_CORRECTION_IMPLEMENTATION_AUTHORITY

PARENT_EXECUTION_AUTHORITY_COMMIT =
8b73320c88362f6caed44877971cfc81f33e0bbf

FAILED_RUN =
gen4-native-mamba-phase-d-extraction-8b73320-v3

FAILED_RUN_COMMAND_SHA256 =
aab923a02da5bd01d7a56fa92a2b79a399f7140e0d507260eeccd8d837f08160

FAILED_RUN_RESULT =
BLOCKED_BEFORE_SCIENTIFIC_MODEL_WORK

FAILED_RUN_BLOCKER =
scripts.reason_router_gen4_native_mamba_state_measurement.ContractError: source update role

FAILED_RUN_PROVISIONING =
PASS_EXACT_BYTES

FAILED_RUN_REPO_CLEAN_GATE =
PASS

FAILED_RUN_PROVISIONING_CLEANUP =
PASS

SCIENTIFIC_FORWARD_PERFORMED =
NO

NATIVE_STATE_EXTRACTION_PERFORMED =
NO

SCIENTIFIC_ARTIFACT_PUBLISHED =
NO

SCIENTIFIC_CONCLUSION =
NONE

## Root cause

FROZEN_RUNTIME =
Python 3.12.13
NumPy 2.0.2
torch 2.10.0+cpu
Transformers 5.0.0

MAMBA_SOURCE_SHA256 =
4c972b30f3c2cca977824fcc6891f956cd4387b6383aa7336848fbc5f2db1d83

MAMBA_SOURCE_BYTES =
39500

DELTA_B_U_LINE =
397

RECURRENT_UPDATE_LINE =
409

READOUT_LINE =
410

FINAL_CACHE_PERSISTENCE_LINE =
417

ACTUAL_LINE_397_ROLE =
deltaB_u = discrete_B * hidden_states[:, :, :, None].float()

ACTUAL_LINE_409_ROLE =
ssm_state = discrete_A[:, :, i, :] * ssm_state + deltaB_u[:, :, i, :]

DEFECT =
The frozen source-role validator requires the line-409 update expression itself
to contain discrete_B and hidden_states. The exact frozen Transformers 5.0.0
source instead materializes their product as deltaB_u at line 397 and consumes
deltaB_u at line 409.

DEFECT_CLASS =
IMPLEMENTATION_VALIDATOR_FALSE_REJECTION

SOURCE_DRIFT =
NO

RUNTIME_DRIFT =
NO

SCIENTIFIC_FAILURE =
NO

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
NARROW_SOURCE_ROLE_DATAFLOW_CORRECTION_ONLY

The corrected validator must prove the exact frozen two-stage dataflow:

1. line 397 contains exactly one relevant assignment;
2. its target is deltaB_u;
3. its RHS depends on discrete_B and hidden_states;
4. line 409 contains exactly one recurrent-state assignment;
5. its target is ssm_state;
6. its RHS is an additive recurrence using discrete_A, ssm_state, and deltaB_u;
7. line 410 remains the post-update readout binding;
8. line 417 remains recurrent-cache persistence of ssm_state.

The correction must not accept arbitrary transitive aliases, arbitrary source
lines, or generic expression searching.

The following checks remain fail-closed and must not be weakened:

- exact runtime versions;
- exact Transformers import/distribution root;
- exact modeling_mamba byte count and SHA256;
- exact cache_utils byte count and SHA256;
- MambaMixer.slow_forward identity;
- capture line 410 binding;
- line-410 readout role;
- line-417 recurrent-cache role;
- slow-forward backend dispatch;
- native-state shape/dtype/finite checks;
- primary layer 11;
- observer non-interference;
- zero-transition and zero-path blockers.

SOURCE_HASH_RELAXATION =
FORBIDDEN

SOURCE_VERSION_RELAXATION =
FORBIDDEN

LINE_NUMBER_RELAXATION =
FORBIDDEN

GENERIC_ALIAS_ACCEPTANCE =
FORBIDDEN

OBSERVER_SEMANTICS_CHANGE =
FORBIDDEN

KINEMATIC_DEFINITION_CHANGE =
FORBIDDEN

## Test correction

The synthetic runtime fixture must mirror the frozen two-stage dataflow.

Required positive coverage:

- line-397 deltaB_u role;
- line-409 recurrent update using deltaB_u;
- line-410 readout;
- line-417 cache persistence;
- complete synthetic runtime gate PASS.

Required negative coverage:

- wrong or missing deltaB_u definition;
- deltaB_u missing discrete_B dependency;
- deltaB_u missing hidden_states dependency;
- recurrent update missing deltaB_u;
- wrong recurrent-state target;
- wrong readout role;
- wrong cache-persistence role.

## Phase D runner correction

Because the measurement SHA256 changes, the Phase D runner frozen dependency
pin must be corrected in the same bounded implementation stage.

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

Authorized runner deltas are limited to:

- corrected measurement SHA256 pin;
- correction-authority provenance binding;
- corresponding manifest provenance field if required;
- corrected execution-authority document path;
- tests for those provenance/dependency bindings.

FUTURE_EXECUTION_AUTHORITY_PATH =
reports/reason_router_gen4_native_mamba_state_bridge_phase_d_extraction_execution_authority_correction_spec_candidate.md

The existing execution authority at
8b73320c88362f6caed44877971cfc81f33e0bbf does not authorize corrected
implementation bytes.

## Validation boundary

VALIDATION_REQUIRED =
YES

Required implementation validation:

- full measurement synthetic tests;
- full extraction synthetic/static tests;
- exact frozen Transformers 5.0.0 source-role regression;
- git diff --check;
- exact intended-file scope review.

CANONICAL_GEN4_MODEL_FORWARD_ALLOWED =
NO

CHECKPOINT_DESERIALIZATION_ALLOWED =
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

CODEX_ALLOWED =
NO

## Strongest permitted verdict

PASS_READY_FOR_SEPARATE_CORRECTED_PHASE_D_EXECUTION_AUTHORITY

This verdict establishes implementation correctness only.
It is not a Phase D execution result and is not a scientific conclusion.