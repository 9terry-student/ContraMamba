# ContraMamba Gen4 Native Mamba State Bridge
# NAME Q1/Q3 Measurement / Extraction
# Implementation Authority Specification
# Candidate

STATUS =
CANDIDATE

AUTHORITY_ID =
GEN4_NATIVE_MAMBA_STATE_BRIDGE_NAME_Q1_Q3_MEASUREMENT_EXTRACTION_IMPLEMENTATION_AUTHORITY

PHASE =
NAME_Q1_Q3_MEASUREMENT_EXTRACTION_IMPLEMENTATION

SCIENTIFIC_CONCLUSION =
NONE

IMPLEMENTATION_ALLOWED =
YES_BOUNDED

STATIC_VALIDATION_ALLOWED =
YES

SYNTHETIC_VALIDATION_ALLOWED =
YES

CANONICAL_SCIENTIFIC_MODEL_FORWARD_ALLOWED =
NO

SCIENTIFIC_CHECKPOINT_FORWARD_ALLOWED =
NO

Q1_Q3_SCIENTIFIC_EXTRACTION_ALLOWED =
NO

STATISTICAL_TESTING_ALLOWED =
NO

TRAINING_ALLOWED =
NO

EVALUATION_ALLOWED =
NO

KAGGLE_EXECUTION_ALLOWED =
NO

GPU_EXECUTION_ALLOWED =
NO

## 1. Purpose

This authority permits only bounded implementation and static/synthetic
validation of the NAME Q1/Q3 native-state measurement and extraction path.

It creates no Q1/Q3 scientific state evidence.

It performs no canonical scientific model forward.

It performs no statistical test.

A later separately frozen execution authority is required before any canonical
Q1/Q3 model forward or native-state extraction.

## 2. Frozen authority chain

MECHANISTIC_BRIDGE_SPECIFICATION =
a2617aa037d1a9834003535b62ac81770a5b96aa

PHASE_F_VALIDATED_RESULT_FREEZE =
ab3428e7be08af26fa1fdafd1483a34e48fbcf8c

DEPTH_SELECTIVITY_INTERPRETATION_CORRECTION =
2e076cbd8e9633c3ab7abb222a05e409366539a7

Q1_Q3_ARCHITECTURE_DEPTH_INDEX_SPECIFICATION =
ecbf6ba720c0e173ac7089a00bb5b783aa16fa6c

Q1_Q3_SECONDARY_LOCALIZATION_SCIENTIFIC_SPECIFICATION =
01801ad1617b2ebc3ffa859ba440636d4755a55c

Q1_Q3_EXTRACTION_FEASIBILITY_PROVENANCE =
c7d841a920a0a6d075f7f3804da9212ce4706673

Q1_Q3_INPUT_COORDINATE_RECONSTRUCTION_CORRECTION =
b9d26005b2e475d9e2645c507eedcbd0c79bbac3

R2_TOKENIZER_CONFORMANCE_VALIDATED_RESULT =
17f1ddfc8286796f27c4a61716a21e14126bb836

## 3. Frozen upstream implementation identities

PRIMARY_MEASUREMENT_PATH =
scripts/reason_router_gen4_native_mamba_state_measurement.py

PRIMARY_MEASUREMENT_SHA256 =
7729424f03058b86b4f120dc0e6da573d6c996b0877858f2d6d38aa94dac268c

PRIMARY_EXTRACTION_RUNNER_PATH =
scripts/reason_router_gen4_native_mamba_state_extraction.py

PRIMARY_EXTRACTION_RUNNER_SHA256 =
653f96713d8bdc776cdf03733ae230784e413f960f58a9c120f00cb8c6a6d3eb

The frozen layer-11 implementation is upstream provenance only.

It must not be modified to make it accept arbitrary layers.

PRIMARY_LAYER_11_IMPLEMENTATION_MODIFICATION =
PROHIBITED

## 4. Exact implementation file scope

After this authority is frozen, implementation may create exactly four new
files:

scripts/reason_router_gen4_native_mamba_state_q1_q3_measurement.py

tests/test_reason_router_gen4_native_mamba_state_q1_q3_measurement.py

scripts/reason_router_gen4_native_mamba_state_q1_q3_extraction.py

tests/test_reason_router_gen4_native_mamba_state_q1_q3_extraction.py

No other file may be created or modified by the implementation task.

EXISTING_FILE_MODIFICATION =
PROHIBITED

If implementation requires an additional file or modification of a frozen
existing file:

IMPLEMENTATION =
BLOCKED_SCOPE_REVIEW_REQUIRED

## 5. Measurement implementation strategy

The new Q1/Q3 measurement module must reuse the frozen low-level native-state
observation semantics from:

scripts/reason_router_gen4_native_mamba_state_measurement.py

It must not copy or reimplement the Mamba recurrent-state source-role logic.

It may import the frozen module's low-level collector/runtime validation
internals only while the exact frozen measurement SHA256 matches:

7729424f03058b86b4f120dc0e6da573d6c996b0877858f2d6d38aa94dac268c

PRIMARY_MEASUREMENT_BYTE_IDENTITY_MISMATCH =
BLOCK

STATE_SOURCE =
native_selective_ssm_recurrent_state

STATE_TIMING =
post_consumption_s_t

No hidden-state, residual-stream, router-logit, q_authorized, task-logit, or
learned representation may replace the native recurrent state.

## 6. Exact secondary layer policy

NATIVE_MAMBA_LAYER_COUNT =
24

Q1_LAYER_INDEX =
5

Q3_LAYER_INDEX =
17

ALLOWED_SECONDARY_LAYER_SET =
{5,17}

The new scientific observer must register exactly the two frozen layer mixer
identities corresponding to layer 5 and layer 17.

REGISTERED_SCIENTIFIC_LAYER_COUNT =
2

REGISTERED_SCIENTIFIC_LAYER_SET =
{5,17}

Layer 11 is invalid in the Q1/Q3 path.

Any layer outside {5,17} is invalid.

Missing layer 5 is invalid.

Missing layer 17 is invalid.

Duplicate mixer identity is invalid.

Duplicate layer index is invalid.

ARBITRARY_LAYER_PARAMETERIZATION =
PROHIBITED

BEST_LAYER_SCAN =
PROHIBITED

FALLBACK_LAYER_SELECTION =
PROHIBITED

## 7. Dual-layer simultaneous observation

The implementation strategy is:

DUAL_LAYER_CAPTURE_MODE =
SIMULTANEOUS_SINGLE_FORWARD

For one future canonical input row, one backbone forward must capture both:

layer 5 native recurrent state
and
layer 17 native recurrent state

from the same forward.

SEQUENTIAL_DUPLICATE_FORWARD_PER_LAYER =
PROHIBITED_FOR_CANONICAL_EXECUTION

The low-level coordinate remains:

(forward_id, layer_index, token_index)

The observer must prove complete, nonduplicated coordinate coverage separately
for each frozen layer.

No state from one layer may be relabeled as the other layer.

No mixed-layer state sequence may enter one kinematic endpoint.

## 8. Frozen state representation contract

EXPECTED_STATE_DTYPE =
float32

EXPECTED_STATE_DEVICE =
cpu

PRIMARY_REFERENCE_PER_EXAMPLE_STATE_SHAPE =
(1536,16)

PRIMARY_REFERENCE_FLATTENED_STATE_SIZE =
24576

The Q1/Q3 implementation must validate the actual captured state rank and shape
for both layers.

It may not merely assume shape compatibility because layer 11 used that shape.

Q1_Q3_SHAPE_ASSUMPTION_WITHOUT_VALIDATION =
PROHIBITED

For each captured state, validation must require:

- CPU;
- float32;
- finite values;
- exact validated scientific shape;
- detached/clone snapshot semantics;
- no aliasing with later recurrent-state mutation.

Layer 5 and layer 17 must independently satisfy the same validated state-shape
contract before future scientific execution can be authorized.

## 9. Observer noninterference

The observer must be disabled by default.

When disabled, it must not alter model behavior.

When enabled, capture must observe only the post-update native recurrent state.

Synthetic validation must prove:

OBSERVER_DEFAULT_DISABLED =
YES

OBSERVER_NONINTERFERENCE =
REQUIRED

CAPTURE_ALIASING =
PROHIBITED

CAPTURE_COORDINATE_DUPLICATION =
PROHIBITED

CAPTURE_COORDINATE_OMISSION =
PROHIBITED

## 10. Frozen input-coordinate reconstruction

The implementation must support deterministic reconstruction of the already
frozen R2 active coordinate.

R2_CANONICAL_ROWS =
1800

R2_CANONICAL_SOURCE_PAIRS =
300

R2_ENCODED_COORDINATE_SHA256 =
d3cf61e55bb04e6dbebe44be433597cce958cb15989412dcf6efc15a329c576a

TOKENIZER_REVISION_REFERENCE =
40e5d2bd7452abb3ca8fadbafe9131ee0e2c2f37

TOKENIZER_JSON_SHA256 =
b074ad869d4f45d1265ca5c9814f78604f3d7e187acc063b15dd232b27585fcf

TOKENIZER_CONFIG_JSON_SHA256 =
9d7016c33747c6309346e59bd7bf63bfc33c9d9366ecb7e514b3b84dc6b46acb

SPECIAL_TOKENS_MAP_JSON_SHA256 =
57491904f8680d4b52ed440f1f7ba48cad1c31ecf3eb453b03484e6ff4723ae8

CLAIM_BUDGET =
63

EOS_SEPARATOR_COUNT =
1

EVIDENCE_BUDGET =
64

MAX_LENGTH =
128

ADD_SPECIAL_TOKENS =
FALSE

The future runner must authenticate tokenizer bytes and reconstruct all 1800
canonical rows before selecting the Q1/Q3 subset.

FULL_1800_COORDINATE_VALIDATION_BEFORE_SUBSET =
REQUIRED

COORDINATE_HASH_MISMATCH_BEHAVIOR =
BLOCK_BEFORE_MODEL_CONSTRUCTION_OR_FORWARD

ALTERNATE_RETOKENIZATION =
PROHIBITED

## 11. Implementation-phase tokenizer boundary

This implementation authority does not authorize canonical 1800-row tokenizer
execution as scientific or provenance execution.

CANONICAL_1800_TOKENIZER_EXECUTION_DURING_IMPLEMENTATION =
NO

The reconstruction logic may be validated with deterministic synthetic/mock
tokenizer fixtures.

Tests must prove that:

- correct reconstruction reaches a supplied expected synthetic hash;
- a hash mismatch fails closed;
- subset selection cannot occur before full-coordinate validation;
- model-construction/forward callbacks are unreachable after hash failure.

Actual canonical 1800-row frozen-coordinate reconstruction is reserved for a
later pre-execution/execution authority.

## 12. Frozen structural input

CANONICAL_GEN4_ARTIFACT =
reports/reason_router_gen4_six_cell_masked_slot_substitution_materialization_fbc780ce12cbfbcf4e20a3cb9d1f099553045fc0/gen4_six_cell_masked_slot_substitution.jsonl

CANONICAL_GEN4_ARTIFACT_SHA256 =
b9c54604863ed15c237fa17c7890a20f3f5ec062a7429638b39e8b468be050a7

CANONICAL_GEN4_ARTIFACT_BYTES =
1465573

CANONICAL_GEN4_ROWS =
1800

CANONICAL_SOURCE_PAIR_COUNT =
300

The future runner must authenticate this artifact before coordinate
reconstruction.

## 13. Q1/Q3 scientific subset

Only after the full frozen R2 coordinate has passed may the runner select:

C0_SHAM
C2_NAME

for every source pair.

Q1_Q3_SOURCE_PAIR_COUNT =
300

Q1_Q3_CELLS_PER_SOURCE_PAIR =
2

Q1_Q3_MODEL_INPUT_ROWS =
600

Q1_Q3_CELL_SET =
C0_SHAM
C2_NAME

STRUCTURAL_ESTIMAND =
DELTA_NAME_ONLY

No row may be selected or dropped using state outcomes.

## 14. Frozen event-coordinate provenance

EVENT_ANCHOR_PREFIX_MANIFEST =
reports/reason_router_gen4_six_cell_native_mamba_state_bridge_feasibility_audit_a2617aa/event_anchor_prefix_manifest_candidate.jsonl

EVENT_ANCHOR_PREFIX_MANIFEST_SHA256 =
70c84c68b36751bb7c7145b33ccb71ab91bc8ee9e6cc5f2c7a0d4e925f36581f

EVENT_ANCHOR_PREFIX_MANIFEST_BYTES =
2268260

SEMANTIC_ANCHOR =
A_NAME

Required scientific combinations are exactly:

A_NAME / C0_SHAM
A_NAME / C2_NAME

for all 300 source pairs.

The implementation must consume the frozen manifest.

It must not rediscover A_NAME from rendered text, state behavior, logits, or
tokenizer output heuristics.

The runner must cross-check event-manifest row identity against the
authenticated reconstructed active coordinate.

## 15. Support-state window

For NAME anchor a at one layer, the exact support required by the three frozen
kinematic endpoints is:

s_(a-1)
s_a
s_(a+1)
s_(a+2)
s_(a+3)
s_(a+4)

SUPPORT_STATE_WINDOW =
[a-1,a+4]

The a-1 state is required for the first turning term.

No shortened support window is allowed.

For exactly one NAME anchor per Q1/Q3 row, one layer contributes exactly six
support states.

FUTURE_SUPPORT_STATE_ROW_COUNT =
7200

because:

600 input rows
x
2 layers
x
6 support states
=
7200

The future flattened support-state tensor therefore has target logical shape:

(7200,24576)

subject to runtime validation of the scientific state shape.

## 16. Frozen local kinematic endpoints

Exactly three endpoints are permitted:

POST4_SPEED
POST4_TURNING
POST4_PATH_EFFICIENCY

The new measurement implementation must reuse the frozen numerical definitions
from the primary measurement module rather than inventing alternative formulas.

KINEMATIC_FORMULA_CHANGE =
PROHIBITED

Each endpoint must be computed entirely from states belonging to one layer.

EXPECTED_FUTURE_ENDPOINT_ROWS =
1200

because:

600 input rows
x
2 layers
=
1200

Every endpoint row must include the exact layer index.

No statistical contrast or p-value is computed by the extraction runner.

## 17. Future canonical forward policy encoded by the runner

The future scientific execution contract implemented now is:

FUTURE_SCIENTIFIC_FORWARD_BATCH_SIZE =
1

FUTURE_CANONICAL_INPUT_ROWS =
600

FUTURE_EXPECTED_BACKBONE_FORWARD_COUNT =
600

FUTURE_LAYERS_CAPTURED_PER_FORWARD =
2

For each future input row:

1. use the frozen reconstructed active input_ids;
2. bind exactly layer 5 and layer 17 mixers;
3. instantiate one fresh dual-layer observer;
4. run exactly one model.mamba forward;
5. capture complete token coordinates required by the observer contract;
6. use only native recurrent states;
7. discard Mamba last_hidden_state as a scientific variable;
8. release observer/capture state before the next row.

No observer state or recurrent cache may be reused across rows.

The downstream ContraMamba heads must not be scientifically forwarded.

## 18. Representative model/checkpoint lineage

The future execution path remains bound to the frozen representative:

REPRESENTATIVE_SEED =
180

REPRESENTATIVE_ARM =
G3-GROUP-D-HALF

REPRESENTATIVE_CHECKPOINT =
reports/reason_router_gen3_grouped_factorial_runs/seed180/G3-GROUP-D-HALF/selected_checkpoint.pt

REPRESENTATIVE_CHECKPOINT_SHA256 =
1ff3fcf2ebd754ab6f9483d6a9982b9b04b9a4eb3357f9f8cdbe2b30399e7d2f

Checkpoint authentication must occur before deserialization.

Future deserialization remains:

torch.load(
    path,
    map_location="cpu",
    weights_only=True
)

No unsafe fallback is permitted.

The implementation may encode this future path but must not load the
scientific checkpoint during implementation validation.

SCIENTIFIC_CHECKPOINT_LOAD_DURING_IMPLEMENTATION =
NO

## 19. Frozen runtime/source contract

Future scientific runtime remains:

Python =
3.12.13

NumPy =
2.0.2

torch =
2.10.0+cpu

Transformers =
5.0.0

DEVICE =
cpu

GPU =
OFF

MAMBA_SOURCE_SHA256 =
4c972b30f3c2cca977824fcc6891f956cd4387b6383aa7336848fbc5f2db1d83

MAMBA_SOURCE_BYTES =
39500

CACHE_SOURCE_SHA256 =
6c123bbe3d23500462f0b617119a8231aa054b6d5475b295443a45f34466e6bc

CACHE_SOURCE_BYTES =
60432

The Q1/Q3 observer must reuse the frozen runtime/source gate.

No version-only bypass is allowed.

No package mutation is authorized.

## 20. Future extraction artifact contract

The runner implementation must be capable of producing exactly these five
future files in a caller-supplied output directory:

manifest.json

support_state_rows.jsonl

support_states.npy

kinematic_endpoints.jsonl

SHA256SUMS.txt

The future output directory must be absent before canonical execution.

The runner must refuse overwrite.

The execution authority will later freeze the exact canonical output directory
and run identity.

No canonical output artifact is produced under this implementation authority.

## 21. Support-state artifact semantics

support_state_rows.jsonl must bind every tensor row to:

- source_pair_id;
- row_id;
- contrast_cell_id;
- semantic anchor A_NAME;
- anchor token index;
- layer index;
- support token index;
- tensor row index.

support_states.npy must contain only the corresponding flattened validated
native recurrent-state vectors in deterministic metadata order.

Layer index must be one of:

5
17

No layer-11 state may appear.

No full 128-token trajectory may be published merely for convenience.

## 22. Endpoint artifact semantics

kinematic_endpoints.jsonl must contain exactly one row for every:

source input row
x
secondary layer

therefore target:

1200 rows

Each row must bind:

- source_pair_id;
- row_id;
- contrast_cell_id;
- semantic anchor A_NAME;
- anchor token index;
- layer index;
- POST4_SPEED;
- POST4_TURNING;
- POST4_PATH_EFFICIENCY;
- references to the required support-state tensor rows.

No statistical-test output fields are allowed.

No behavioral output/logit fields are allowed.

## 23. Manifest provenance requirements

The future manifest implementation must record at minimum:

- implementation commit;
- measurement implementation SHA256;
- extraction runner SHA256;
- frozen primary measurement SHA256;
- authority commits;
- structural artifact SHA256;
- event manifest SHA256;
- R2 encoded-coordinate SHA256;
- tokenizer revision and tokenizer file SHA256s;
- runtime versions;
- Mamba source SHA256 and bytes;
- cache source SHA256 and bytes;
- representative checkpoint SHA256;
- layer set {5,17};
- semantic anchor A_NAME;
- source-pair count 300;
- model-input row count 600;
- backbone forward count 600;
- support-state row count 7200;
- endpoint row count 1200;
- output peer-artifact SHA256s.

The implementation must not manufacture a manifest self-hash dependency.

## 24. Required measurement tests

The Q1/Q3 measurement test file must synthetically verify at minimum:

- exact accepted layer set {5,17};
- simultaneous registration of both layers;
- rejection of layer 11;
- rejection of every out-of-set layer;
- rejection of missing layer 5;
- rejection of missing layer 17;
- rejection of duplicate mixer identity;
- rejection of duplicate layer identity;
- deterministic coordinate ordering by forward/layer/token;
- complete coordinates for both layers;
- layer-label correctness;
- captured snapshot clone/non-alias semantics;
- float32 CPU finite-state enforcement;
- shape enforcement;
- observer disabled by default;
- observer noninterference;
- same-layer-only kinematic computation;
- unchanged post4 endpoint formulas;
- zero-transition blocker;
- zero-path blocker.

No real canonical checkpoint/model forward is permitted in these tests.

## 25. Required extraction-runner tests

The Q1/Q3 extraction test file must synthetically/staticly verify at minimum:

- exact structural artifact identity checks;
- full-coordinate reconstruction must precede subset selection;
- expected coordinate hash match allows continuation;
- coordinate hash mismatch blocks before model-construction callback;
- exact C0/C2-only subset;
- exact 300-pair completeness;
- exact 600-row subset;
- exact A_NAME event-manifest binding;
- rejection of incorrect/missing event rows;
- exact layer set {5,17};
- exactly one future forward per input row;
- both layers represented per forward;
- support-state metadata ordering;
- target 7200 support-state rows;
- target 1200 endpoint rows;
- no layer-11 artifact row;
- no mixed-layer support references;
- output overwrite refusal;
- exact five-file output contract;
- manifest peer-hash binding;
- no statistical fields;
- no downstream behavioral/logit fields;
- no training/backward/optimizer path.

Canonical 1800-row tokenizer execution, scientific checkpoint load, and
scientific model forward are prohibited during this implementation test phase.

## 26. Validation command boundary

After implementation, validation must include at minimum:

python -m py_compile scripts/reason_router_gen4_native_mamba_state_q1_q3_measurement.py scripts/reason_router_gen4_native_mamba_state_q1_q3_extraction.py tests/test_reason_router_gen4_native_mamba_state_q1_q3_measurement.py tests/test_reason_router_gen4_native_mamba_state_q1_q3_extraction.py

and the exact two dedicated pytest files.

The implementation result is not freeze-ready unless all dedicated tests pass.

No canonical scientific execution is part of implementation validation.

## 27. Preserve the primary Phase D implementation

These frozen files must remain byte-identical throughout implementation:

scripts/reason_router_gen4_native_mamba_state_measurement.py

scripts/reason_router_gen4_native_mamba_state_extraction.py

tests/test_reason_router_gen4_native_mamba_state_measurement.py

tests/test_reason_router_gen4_native_mamba_state_extraction.py

The implementation must not loosen the layer-11 fail-closed contract.

PRIMARY_PHASE_D_REGRESSION =
BLOCK

## 28. Scientific and inferential prohibitions

This authority does not permit:

- interpreting any Q1/Q3 state outcome;
- testing the six secondary hypotheses;
- comparing Q1/Q3 to midpoint;
- claiming depth selectivity;
- scanning additional layers;
- changing endpoints;
- changing the NAME estimand;
- outcome-dependent row removal;
- output-sign matching;
- causal mediation claims.

DEPTH_SELECTIVITY =
NOT_ESTABLISHED

OVERALL_ADAPTIVE_PROGRAM_FWER =
NOT_CLAIMED

## 29. Implementation completion state

A successful implementation phase may conclude only:

Q1_Q3_IMPLEMENTATION =
PASS_READY_FOR_INDEPENDENT_VERIFICATION

It may not conclude:

Q1_Q3_SCIENTIFIC_EXTRACTION =
PASS

or:

SECONDARY_LAYER_NAME_LOCALIZATION =
SUPPORTED

## 30. Next phase

After implementation and independent verification, the implementation files
must be frozen by exact byte identity before execution authority.

NEXT_PHASE_ON_IMPLEMENTATION_PASS =
NAME_Q1_Q3_IMPLEMENTATION_VALIDATION_AND_FREEZE

SCIENTIFIC_EXECUTION_AFTER_IMPLEMENTATION_FREEZE =
REQUIRES_SEPARATE_AUTHORITY

NEXT_EXECUTION =
NOT_AUTHORIZED
