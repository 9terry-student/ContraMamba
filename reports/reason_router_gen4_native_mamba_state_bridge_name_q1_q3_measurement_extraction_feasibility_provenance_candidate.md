# ContraMamba Gen4 Native Mamba State Bridge
# NAME Q1/Q3 Measurement / Extraction Feasibility and Provenance
# Candidate

STATUS =
CANDIDATE

AUTHORITY_ID =
GEN4_NATIVE_MAMBA_STATE_BRIDGE_NAME_Q1_Q3_MEASUREMENT_EXTRACTION_FEASIBILITY_PROVENANCE

THIS_DOCUMENT_CREATES_NEW_SCIENTIFIC_EVIDENCE =
NO

IMPLEMENTATION_ALLOWED =
NO

SCIENTIFIC_EXTRACTION_ALLOWED =
NO

MODEL_FORWARD_ALLOWED =
NO

CHECKPOINT_LOADING_ALLOWED =
NO

TOKENIZER_EXECUTION_ALLOWED =
NO

STATISTICAL_TESTING_ALLOWED =
NO

TRAINING_ALLOWED =
NO

KAGGLE_ALLOWED =
NO

GPU_ALLOWED =
NO

## 1. Frozen lineage

MECHANISTIC_BRIDGE_SPECIFICATION =
a2617aa037d1a9834003535b62ac81770a5b96aa

PHASE_F_VALIDATED_STATISTICAL_RESULT_FREEZE =
ab3428e7be08af26fa1fdafd1483a34e48fbcf8c

DEPTH_SELECTIVITY_INTERPRETATION_CORRECTION =
2e076cbd8e9633c3ab7abb222a05e409366539a7

Q1_Q3_ARCHITECTURE_DEPTH_INDEX_SPECIFICATION =
ecbf6ba720c0e173ac7089a00bb5b783aa16fa6c

Q1_Q3_SECONDARY_LOCALIZATION_SCIENTIFIC_SPECIFICATION =
01801ad1617b2ebc3ffa859ba440636d4755a55c

PHASE_D_CORRECTED_IMPLEMENTATION =
c7ae7fac4c64e9bd64adcae819a3da3dd46f17f7

## 2. Frozen implementation identities inspected

MEASUREMENT_PATH =
scripts/reason_router_gen4_native_mamba_state_measurement.py

MEASUREMENT_SHA256 =
7729424f03058b86b4f120dc0e6da573d6c996b0877858f2d6d38aa94dac268c

PHASE_D_RUNNER_PATH =
scripts/reason_router_gen4_native_mamba_state_extraction.py

PHASE_D_RUNNER_SHA256 =
653f96713d8bdc776cdf03733ae230784e413f960f58a9c120f00cb8c6a6d3eb

The current files are inspected only as frozen implementation evidence.

This document does not modify them.

## 3. Runtime/source provenance that remains relevant

The frozen measurement implementation binds the native state source to the
Transformers Mamba slow-path recurrent update.

STATE_SOURCE =
native_selective_ssm_recurrent_state

STATE_TIMING =
post_consumption_s_t

NATIVE_MAMBA_LAYER_COUNT =
24

FROZEN_TRANSFORMERS_VERSION =
5.0.0

FROZEN_TORCH_VERSION =
2.10.0+cpu

FROZEN_NUMPY_VERSION =
2.0.2

FROZEN_PYTHON_VERSION =
3.12.13

MAMBA_SOURCE_SHA256 =
4c972b30f3c2cca977824fcc6891f956cd4387b6383aa7336848fbc5f2db1d83

MAMBA_SOURCE_BYTES =
39500

CACHE_UTILS_SHA256 =
6c123bbe3d23500462f0b617119a8231aa054b6d5475b295443a45f34466e6bc

CACHE_UTILS_BYTES =
60432

The recurrent-state source role is defined by the same MambaMixer slow_forward
implementation used by all recurrent layers in this frozen backbone.

However, reuse of this static runtime/source provenance does not itself
authorize layer-5 or layer-17 scientific capture.

RUNTIME_SOURCE_PROVENANCE_REUSABLE_AS_STATIC_INPUT =
YES

Q1_Q3_RUNTIME_MUST_STILL_PASS_GATE_BEFORE_FUTURE_EXECUTION =
YES

## 4. Existing scientific observer boundary

The current scientific observer is intentionally primary-layer-only.

The frozen measurement implementation contains:

validate_primary_layer_registration(...)

which requires exactly one registered scientific layer and requires:

layer_index =
11

NativeStateObserver calls that validation before creating the scientific
collector.

The frozen capture-coordinate validator also requires:

layer_index =
PRIMARY_LAYER_INDEX

and constructs the expected coordinate sequence only for layer 11.

Therefore:

CURRENT_NATIVE_STATE_OBSERVER_SUPPORTS_LAYER_5_SCIENTIFIC_CAPTURE =
NO

CURRENT_NATIVE_STATE_OBSERVER_SUPPORTS_LAYER_17_SCIENTIFIC_CAPTURE =
NO

CURRENT_CAPTURE_COORDINATE_VALIDATOR_SUPPORTS_Q1_Q3 =
NO

Attempting to use the current scientific observer for layer 5 or 17 must fail
closed.

## 5. Low-level collector boundary

The low-level trace collector accepts a mapping of registered mixer identities
to descriptors containing integer layer indices.

Its internal coordinate key is:

(forward_id, layer_index, token_index)

and it requires unique layer identities.

This shows that the low-level instrumentation architecture is not intrinsically
encoded around layer 11.

However:

LOW_LEVEL_MULTI_LAYER_SCIENTIFIC_CAPTURE_ALREADY_VALIDATED =
NO

The existing scientific policy and validators remain primary-layer-only.

No Q1/Q3 execution may bypass those scientific policy layers by directly using
the low-level collector.

DIRECT_LOW_LEVEL_COLLECTOR_BYPASS =
PROHIBITED

## 6. Existing Phase D runner boundary

The frozen Phase D extraction runner is also primary-layer-specific.

It binds:

PRIMARY_LAYER =
measurement.PRIMARY_LAYER_INDEX

and records that layer in:

- support-state metadata;
- endpoint rows;
- support-plan validation;
- endpoint-row validation.

Its row extraction function selects:

layers[PRIMARY_LAYER].mixer

and constructs the scientific observer with:

layer_index =
PRIMARY_LAYER

Therefore:

CURRENT_PHASE_D_RUNNER_SUPPORTS_Q1_LAYER_5 =
NO

CURRENT_PHASE_D_RUNNER_SUPPORTS_Q3_LAYER_17 =
NO

CURRENT_PHASE_D_RUNNER_VALID_FOR_Q1_Q3_EXECUTION =
NO

A future Q1/Q3 extraction must use a separately implemented and validated
runner under new authority.

## 7. Existing Phase D artifact boundary

The frozen Phase D native-state tensor represents layer 11.

It cannot be relabeled as layer 5 or layer 17.

EXISTING_PHASE_D_SUPPORT_STATES_REUSABLE_AS_Q1_STATE_EVIDENCE =
NO

EXISTING_PHASE_D_SUPPORT_STATES_REUSABLE_AS_Q3_STATE_EVIDENCE =
NO

The intentionally untracked local file:

reports/reason_router_gen4_native_mamba_state_extraction_bff0a75_v1/support_states.npy

must not be staged, modified, transformed, or reused as Q1/Q3 scientific
evidence.

## 8. Token/event-coordinate provenance

The Q1/Q3 scientific specification preserves the existing active token
coordinate and A_NAME semantic anchor.

The frozen event manifest is:

reports/reason_router_gen4_six_cell_native_mamba_state_bridge_feasibility_audit_a2617aa/event_anchor_prefix_manifest_candidate.jsonl

EVENT_MANIFEST_SHA256 =
70c84c68b36751bb7c7145b33ccb71ab91bc8ee9e6cc5f2c7a0d4e925f36581f

EVENT_MANIFEST_BYTES =
2268260

The existing event-manifest contract includes:

A_NAME / C0_SHAM
A_NAME / C2_NAME

for all:

SOURCE_PAIR_COUNT =
300

The semantic/token coordinate is layer-independent.

Therefore:

A_NAME_EVENT_COORDINATE_REQUIRES_NEW_TOKENIZER_EXECUTION =
NO

A_NAME_EVENT_MANIFEST_MAY_BE_REUSED_AS_FROZEN_COORDINATE_PROVENANCE =
YES

Reuse is permitted only if its exact frozen byte identity and canonical row
identity are revalidated by the future implementation/preflight.

No retokenization is permitted.

## 9. Required scientific extraction population

The frozen research population remains the complete 300 source-pair
population.

The Q1/Q3 structural contrast requires only:

C0_SHAM
C2_NAME

for each of the 300 source pairs.

REQUIRED_Q1_Q3_INPUT_ROWS =
600

REQUIRED_SOURCE_PAIRS =
300

REQUIRED_CELLS_PER_PAIR =
2

REQUIRED_CELL_SET =
C0_SHAM
C2_NAME

This does not redefine the underlying six-cell structural population.

It only restricts new state extraction to the cells required for the frozen
DELTA_NAME secondary estimand.

No result-dependent row selection is permitted.

## 10. Required future layer set

Q1_LAYER_INDEX =
5

Q3_LAYER_INDEX =
17

ALLOWED_Q1_Q3_LAYER_SET =
{5,17}

MIDPOINT_LAYER_11_NEW_EXTRACTION =
NOT_REQUIRED

OTHER_LAYER_EXTRACTION =
PROHIBITED

BEST_LAYER_SCAN =
PROHIBITED

FALLBACK_LAYER_SELECTION =
PROHIBITED

## 11. Required future state representation

Each future secondary-layer capture must preserve:

STATE_SOURCE =
native_selective_ssm_recurrent_state

STATE_TIMING =
post_consumption_s_t

EXPECTED_STATE_DTYPE =
float32_cpu

The current primary-layer scientific state shape is:

PER_EXAMPLE_STATE_SHAPE =
(1536,16)

FLATTENED_STATE_SIZE =
24576

For layers 5 and 17:

Q1_Q3_STATE_SHAPE_ASSUMED_FROM_PRIMARY_WITHOUT_VALIDATION =
PROHIBITED

A later implementation/synthetic/runtime validation must establish that the
captured Q1/Q3 native state has the exact scientifically required rank, shape,
dtype, device, finiteness, clone/non-alias semantics, and coordinate
completeness before scientific execution.

## 12. Kinematic computation reuse boundary

The frozen numerical definitions remain:

POST4_SPEED
POST4_TURNING
POST4_PATH_EFFICIENCY

Their mathematical computation operates on a same-layer state sequence.

The kinematic formulas do not require a new definition merely because the
layer index changes.

KINEMATIC_FORMULAS_REQUIRE_SCIENTIFIC_REDEFINITION =
NO

However, the future implementation must prove that the input states supplied to
those formulas come exclusively from the correct frozen layer and coordinate.

No mixed-layer trajectory may be passed to a kinematic endpoint.

MIXED_LAYER_KINEMATIC_TRAJECTORY =
PROHIBITED

## 13. Implementation delta required before execution

A bounded future implementation must add Q1/Q3 scientific policy without
weakening the frozen Phase D primary-layer contract.

The required delta includes at minimum:

1. a secondary-layer scientific registration policy restricted exactly to
   layer 5 and layer 17;

2. capture-coordinate validation that binds each captured state to the exact
   requested secondary layer and token index;

3. extraction metadata that records the exact layer for every support state
   and endpoint row;

4. validators that reject layer 11, any layer outside {5,17}, missing layers,
   duplicate coordinates, mixed-layer support references, and layer-label
   mismatches within the Q1/Q3 artifact;

5. exact C0_SHAM/C2_NAME and A_NAME-only extraction scope;

6. exact 300-source-pair completeness;

7. reuse of the frozen tokenizer/event coordinate without retokenization;

8. unchanged post4 mathematical definitions and blockers;

9. deterministic artifact serialization and provenance binding;

10. no statistical testing in the extraction implementation.

The implementation must not relax the original primary-layer observer to accept
arbitrary layers globally.

ARBITRARY_LAYER_PARAMETERIZATION =
PROHIBITED

The scientific policy must remain explicitly bounded to the frozen Q1/Q3 set.

## 14. Validation requirements

Before any scientific extraction is authorized, synthetic/static validation
must establish at minimum:

- exact layer-5 capture labeling;
- exact layer-17 capture labeling;
- rejection of layer 11 in the Q1/Q3 path;
- rejection of every layer outside {5,17};
- complete ordered token-coordinate capture;
- no aliasing of recurrent-state snapshots;
- finite float32 CPU state validation;
- correct scientific state shape;
- observer noninterference;
- exact event-coordinate reuse;
- exact 600-row C0/C2 population;
- exact 300 source-pair completeness;
- no training/backward/optimizer path;
- no downstream statistical fields;
- fail-closed runtime/source provenance.

MULTI_LAYER_OR_DUAL_LAYER_CAPTURE_IF_IMPLEMENTED =
MUST_BE_EXPLICITLY_TESTED

A sequential one-layer-at-a-time design remains scientifically possible if
separately specified and validated.

This feasibility document does not choose between simultaneous and sequential
capture.

## 15. Feasibility verdict

STATIC_FEASIBILITY_VERDICT =
FEASIBLE_FOR_BOUNDED_IMPLEMENTATION

CURRENT_SCIENTIFIC_IMPLEMENTATION_EXECUTABLE_FOR_Q1_Q3 =
NO

REASON =
CURRENT_OBSERVER_AND_RUNNER_ARE_FAIL_CLOSED_TO_PRIMARY_LAYER_11

NEW_MODEL_ARCHITECTURE_REQUIRED =
NO

NEW_TOKENIZATION_REQUIRED =
NO

NEW_STRUCTURAL_DATA_GENERATION_REQUIRED =
NO

NEW_NATIVE_STATE_EXTRACTION_IMPLEMENTATION_REQUIRED =
YES

NEW_Q1_Q3_NATIVE_STATE_ARTIFACT_REQUIRED =
YES

SCIENTIFIC_EXECUTION_CURRENTLY_AUTHORIZED =
NO

## 16. Provenance conclusion

The following can be reused as frozen upstream provenance subject to exact
identity checks:

- Gen4 structural population;
- active token coordinate;
- A_NAME event anchor manifest;
- representative model/checkpoint lineage;
- Mamba runtime/source-role provenance;
- native recurrent-state semantic definition;
- post4 kinematic mathematical definitions.

The following cannot be reused as Q1/Q3 scientific evidence:

- layer-11 support state tensor;
- layer-11 endpoint rows;
- layer-11 scientific observer policy;
- layer-11-only extraction runner.

## 17. Next authority

NEXT_PHASE =
NAME_Q1_Q3_MEASUREMENT_EXTRACTION_IMPLEMENTATION_AUTHORITY

The next authority may permit bounded implementation and synthetic/static
validation only.

It must not authorize:

- scientific model forward;
- Q1/Q3 native-state extraction;
- statistical testing;
- training;
- Kaggle execution.

NEXT_EXECUTION =
NOT_AUTHORIZED
