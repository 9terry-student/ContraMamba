# ContraMamba Gen4 Native Mamba State Bridge
# Phase D Extraction Runner Implementation Authority Candidate

STATUS =
CANDIDATE

AUTHORITY_ID =
GEN4_NATIVE_MAMBA_STATE_BRIDGE_PHASE_D_EXTRACTION_RUNNER_IMPLEMENTATION_AUTHORITY

PHASE =
GEN4_NATIVE_MAMBA_STATE_BRIDGE_PHASE_D_RUNNER_IMPLEMENTATION

SCIENTIFIC_CONCLUSION =
NONE

IMPLEMENTATION_ALLOWED =
YES_BOUNDED

SYNTHETIC_VALIDATION_ALLOWED =
YES

CANONICAL_GEN4_MODEL_FORWARD_ALLOWED =
NO

SCIENTIFIC_CHECKPOINT_FORWARD_ALLOWED =
NO

NATIVE_STATE_SCIENTIFIC_EXTRACTION_ALLOWED =
NO

PRIMARY_STATISTICAL_TESTING_ALLOWED =
NO

TRAINING_ALLOWED =
NO

EVALUATION_ALLOWED =
NO

KAGGLE_EXECUTION_ALLOWED =
NO


## 1. Purpose

This authority permits only implementation and synthetic validation of the
runner that a later, separately frozen Phase D execution authority may use to
extract the prespecified Gen4 native-Mamba recurrent-state measurements.

This authority itself performs no scientific model forward.

It creates no native-state scientific evidence.

The later execution authority must bind the exact frozen runner commit,
runner SHA256, measurement-library SHA256, runtime identity, input artifacts,
representative checkpoint, output directory, run name, and command bytes.


## 2. Frozen authority chain

PHASE_C_MEASUREMENT_IMPLEMENTATION_FREEZE =
e3c870e7f24e183b0046b568e1de3b71446c182d

PHASE_C_IMPLEMENTATION_AUTHORITY =
480ff74aebf5ef942aa9f47fa78c06612a8f97a4

PHASE_AB_FEASIBILITY_FREEZE =
26fd55803acd05febefc8bd031f2fc23c17b0ef4

MECHANISTIC_BRIDGE_SPECIFICATION =
a2617aa037d1a9834003535b62ac81770a5b96aa

PHASE_C_SUCCESS =
PASS_READY_FOR_SEPARATE_PHASE_D_NATIVE_STATE_EXTRACTION_AUTHORITY

SCIENTIFIC_CONCLUSION_AT_PHASE_C =
NONE


## 3. Frozen Phase C implementation identity

MEASUREMENT_IMPLEMENTATION_PATH =
scripts/reason_router_gen4_native_mamba_state_measurement.py

MEASUREMENT_IMPLEMENTATION_SHA256 =
a90aea3e8a60a305ac70b34866647f8c4ff2c6d02a092d93896ac5dc8a258086

MEASUREMENT_TEST_PATH =
tests/test_reason_router_gen4_native_mamba_state_measurement.py

MEASUREMENT_TEST_SHA256 =
b1d4cd0f3917a7b7f5df21967c6daf9744176400b4c6b301c11fff60a9246515

PHASE_C_SYNTHETIC_TEST_RESULT =
PASS_25_OF_25

PRIMARY_LAYER =
11

STATE_VECTOR_SIZE =
24576

STATE_TIMING =
post_consumption_s_t


## 4. Exact implementation scope

After this authority is frozen, only these two new files may be created or
modified by the Phase D runner implementation task:

- scripts/reason_router_gen4_native_mamba_state_extraction.py
- tests/test_reason_router_gen4_native_mamba_state_extraction.py

The frozen Phase C measurement implementation must be imported and reused.

It must not be copied, forked, or modified.

No existing Gen4 artifact, evaluator, model snapshot, checkpoint, tokenizer,
Transformers installation, O0c file, report, or authority file may be modified.

A need for another implementation file is:

BLOCKED_SCOPE_REVIEW_REQUIRED


## 5. Future scientific runtime contract

The future scientific execution target remains:

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

DTYPE =
float32

GPU =
OFF

KAGGLE_ACCELERATOR =
None

MAMBA_SOURCE_SHA256 =
4c972b30f3c2cca977824fcc6891f956cd4387b6383aa7336848fbc5f2db1d83

MAMBA_SOURCE_BYTES =
39500

CACHE_SOURCE_SHA256 =
6c123bbe3d23500462f0b617119a8231aa054b6d5475b295443a45f34466e6bc

CACHE_SOURCE_BYTES =
60432

The future runner must call the frozen Phase C runtime gate before scientific
checkpoint/model execution.

No version-only bypass is allowed.

No package install, uninstall, upgrade, downgrade, vendoring, site-packages
editing, optional-kernel installation, CUDA activation, or source patching is
permitted by the future scientific runner.

If the runtime does not already satisfy the contract, execution must block.


## 6. Frozen scientific population

CANONICAL_GEN4_ARTIFACT =
reports/reason_router_gen4_six_cell_masked_slot_substitution_materialization_fbc780ce12cbfbcf4e20a3cb9d1f099553045fc0/gen4_six_cell_masked_slot_substitution.jsonl

CANONICAL_GEN4_ARTIFACT_SHA256 =
b9c54604863ed15c237fa17c7890a20f3f5ec062a7429638b39e8b468be050a7

CANONICAL_GEN4_ARTIFACT_BYTES =
1465573

CANONICAL_GEN4_ROWS =
1800

SOURCE_PAIR_COUNT =
300

CELLS_PER_SOURCE_PAIR =
6

CANONICAL_CELLS =
C0_SHAM
C1_TITLE
C2_NAME
C3_ROLE
C4_PREDICATE
C5_TITLE_NAME

No row, source pair, or cell may be added, removed, substituted, reordered
based on state outcomes, or rescued after a measurement failure.


## 7. Frozen event-anchor provenance

EVENT_ANCHOR_PREFIX_MANIFEST =
reports/reason_router_gen4_six_cell_native_mamba_state_bridge_feasibility_audit_a2617aa/event_anchor_prefix_manifest_candidate.jsonl

EVENT_ANCHOR_PREFIX_MANIFEST_SHA256 =
70c84c68b36751bb7c7145b33ccb71ab91bc8ee9e6cc5f2c7a0d4e925f36581f

EVENT_ANCHOR_PREFIX_MANIFEST_BYTES =
2268260

REQUIRED_ANCHOR_CELL_ROWS =
3600

PREFIX_FEASIBILITY =
PASS_300_OF_300

Required combinations remain exactly:

A_TITLE:
C0_SHAM
C1_TITLE

A_NAME:
C0_SHAM
C2_NAME

A_ROLE:
C0_SHAM
C3_ROLE

A_PREDICATE:
C0_SHAM
C4_PREDICATE

A_IDENTITY:
C0_SHAM
C1_TITLE
C2_NAME
C5_TITLE_NAME

The runner must consume the frozen manifest.

It must not rediscover anchors from rendered text, hidden state behavior,
kinematic outcomes, logits, q_authorized, or other model outputs.

The implementation must statically bind the actual frozen manifest field
schema. No heuristic fallback field names are permitted.


## 8. Frozen active encoding identity

TOKENIZER_REVISION_REFERENCE =
40e5d2bd7452abb3ca8fadbafe9131ee0e2c2f37

TOKENIZER_JSON_SHA256 =
b074ad869d4f45d1265ca5c9814f78604f3d7e187acc063b15dd232b27585fcf

TOKENIZER_CONFIG_SHA256 =
9d7016c33747c6309346e59bd7bf63bfc33c9d9366ecb7e514b3b84dc6b46acb

SPECIAL_TOKENS_MAP_SHA256 =
57491904f8680d4b52ed440f1f7ba48cad1c31ecf3eb453b03484e6ff4723ae8

R2_ENCODED_COORDINATE_SHA256 =
d3cf61e55bb04e6dbebe44be433597cce958cb15989412dcf6efc15a329c576a

CLAIM_BUDGET =
63

EOS_TOKEN_ID =
0

EVIDENCE_BUDGET =
64

MAX_MODEL_SEQUENCE_LENGTH =
128

ADD_SPECIAL_TOKENS =
false

The future runner must reproduce the already frozen Gen4 active encoding
identity before any model forward.

Padding after the active terminal may be consumed by the frozen Mamba forward,
but no measurement coordinate after the frozen active terminal may enter a
scientific endpoint.

Causality of the recurrent forward ensures later padding must not be used to
alter earlier support coordinates.


## 9. Representative native backbone

Phase A established:

NATIVE_MAMBA_BACKBONE_IDENTITY =
IDENTICAL

CHECKPOINTS_AUTHENTICATED =
18_OF_18

NATIVE_STATE_MODEL_REPLICATION_COUNT =
1

Therefore the Phase D scientific extraction must use exactly one
prespecified representative checkpoint, not 18 repeated scientific forwards.

REPRESENTATIVE_SEED =
180

REPRESENTATIVE_ARM =
G3-GROUP-D-HALF

REPRESENTATIVE_CHECKPOINT =
reports/reason_router_gen3_grouped_factorial_runs/seed180/G3-GROUP-D-HALF/selected_checkpoint.pt

REPRESENTATIVE_CHECKPOINT_SHA256 =
1ff3fcf2ebd754ab6f9483d6a9982b9b04b9a4eb3357f9f8cdbe2b30399e7d2f

Checkpoint authentication must occur before deserialization.

Deserialization must be:

torch.load(
    path,
    map_location="cpu",
    weights_only=True
)

No unsafe fallback is permitted.

The run must not execute any other Gen3 checkpoint merely to manufacture
replication.


## 10. Frozen model construction lineage

MODEL_CONFIG_REVISION =
40e5d2bd7452abb3ca8fadbafe9131ee0e2c2f37

MODEL_CONFIG_SHA256 =
784825b6b6cdde47a1602278db0e66d6764169af4a8e662404701bc636a2686a

MODEL_CONFIG_BYTES =
895

HISTORICAL_MODEL_SOURCE_COMMIT =
3e0e9a435068c552abf20f3a74e0c3eccca344a3

HISTORICAL_MODEL_SNAPSHOT =
src/contramamba/modeling_v6b_minimal_gen3_grouped_snapshot.py

HISTORICAL_MODEL_SNAPSHOT_SHA256 =
8c365bfa857157d91f363358d5db3abaab425dec3e0d7c62683b4207a589b6a5

R5_ADAPTER =
scripts/reason_router_gen4_six_cell_tier2_inference_adapter.py

R5_ADAPTER_SHA256 =
83177c351f82a781586c63bd8d4ef1b40e759b5a94858dc1837d65502cbff6e5

R5_SCIENTIFIC_INFERENCE =
scripts/reason_router_gen4_six_cell_tier2_scientific_inference.py

R5_SCIENTIFIC_INFERENCE_SHA256 =
468a758a7d20d048c75a0ca7e298b73a65f538527df55d3ecad3c7ff1760cf4d

The runner must reuse the frozen R5 checkpoint-authentication and historical
model-construction semantics rather than inventing a new model architecture.

Required future construction:

1. authenticate the frozen local model config;
2. construct the local Mamba backbone from that config;
3. build the historical Gen3 grouped wrapper for the representative arm;
4. authenticate the representative checkpoint before load;
5. strict-load the checkpoint;
6. set eval mode;
7. require the Mamba parameters to remain frozen;
8. move/use the model on CPU only;
9. execute only model.mamba for scientific state extraction.

The downstream ContraMamba heads must not be scientifically forwarded.

q_authorized, entitlement probability, final logits, external prediction,
router outputs, or downstream head activations must not be recorded or
inspected during Phase D state extraction.


## 11. Exact scientific forward policy for later execution

Future scientific execution must process the 1800 canonical rows exactly once
each.

SCIENTIFIC_FORWARD_BATCH_SIZE =
1

EXPECTED_BACKBONE_FORWARD_COUNT =
1800

For each row:

1. use the frozen active input_ids of length 128;
2. instantiate a fresh enabled native-state observer bound only to layer 11;
3. execute exactly one `model.mamba(input_ids=...)` forward under:
   - model.mamba.eval()
   - torch.inference_mode()
   - CPU
   - float32
   - autocast disabled;
4. require complete captured token coordinates 0..127;
5. discard Mamba last_hidden_state as a scientific variable;
6. use only the captured native recurrent state.

No recurrent cache or observer snapshot may be reused across rows.


## 12. Required support-state coordinate

For every required cell-anchor measurement with anchor a, the full support
needed by the frozen three-endpoint family is:

s_(a-1)
s_a
s_(a+1)
s_(a+2)
s_(a+3)
s_(a+4)

The a-1 state is required because:

TURNING_(a+1)

depends on:

v_(a+1)
and
v_a

and:

v_a = s_a - s_(a-1)

Therefore:

SUPPORT_STATE_WINDOW =
[a-1, a+4]

No implementation may store only [a, a+4] and reconstruct s_(a-1).

For one Gen4 row, the artifact stores the sorted union of all support-state
indices required by that row's frozen anchors.

Duplicate support coordinates are stored once.

Full 128-state scientific trajectories are not published merely for
convenience.


## 13. Prefix and numerical fail-closed policy

Before endpoint computation for every required row:

a + 4 <= active_terminal_index - 1

must hold using the frozen active terminal from Phase A+B provenance.

No padded terminal may replace the active terminal.

No shortened window is permitted.

The Phase C numerical policy remains unchanged:

- zero speed by itself is valid;
- zero norm required by turning:
  BLOCKED_UNDEFINED_ZERO_TRANSITION_NORM
- zero POST4 path length:
  BLOCKED_UNDEFINED_ZERO_PATH_LENGTH
- any nonfinite state/transition/norm/cosine/endpoint:
  BLOCKED_NONFINITE_MEASUREMENT

No epsilon.

No imputation.

No row dropping.

No contrast-specific pair removal.

Any one required scientific measurement failure blocks the entire canonical
Phase D run.


## 14. Frozen endpoint rows

The runner may compute only these deterministic measurement endpoints:

POST4_SPEED

POST4_TURNING

POST4_PATH_EFFICIENCY

For each of the 3600 frozen required cell-anchor rows.

The output remains cell-level measurement data.

Phase D must not compute:

Delta_title

Delta_name

Delta_role

Delta_predicate

Interaction_title_name

title-minus-name

p-values

confidence intervals

effect sizes

multiplicity correction

support/non-support verdicts

or any other statistical aggregate over the 300 source pairs.

Those remain later statistical-analysis authority.


## 15. Frozen future artifact bundle

A successful future Phase D extraction run must create one collision-protected
output directory containing exactly:

1. manifest.json
2. support_state_rows.jsonl
3. support_states.npy
4. kinematic_endpoints.jsonl
5. SHA256SUMS.txt

No alternate CSV, pickle, Parquet, HDF5, second vector representation, or
ad-hoc debug artifact is scientific output.

### support_states.npy

Exactly one C-contiguous two-dimensional array.

dtype =
little-endian float32

shape =
[R, 24576]

R is determined before scientific forward from the frozen union of required
support coordinates.

No object dtype.

No pickle.

All values finite.

### support_state_rows.jsonl

Exactly one row per vector, in deterministic order:

canonical source-pair order
then canonical cell order
then absolute token index ascending.

Each row must bind at minimum:

- schema_version
- source_pair_id
- row_id
- contrast_cell_id
- absolute_token_index
- active_terminal_index
- layer_index
- state_source
- state_timing
- tensor_shape
- flattened_size
- vector_index

Fixed:

layer_index = 11
state_source = native_selective_ssm_recurrent_state
state_timing = post_consumption_s_t
tensor_shape = [1536,16]
flattened_size = 24576

### kinematic_endpoints.jsonl

Exactly 3600 deterministic rows.

Order:

canonical source-pair order
then anchor order:
A_TITLE
A_NAME
A_ROLE
A_PREDICATE
A_IDENTITY
then canonical cell order, retaining only required combinations.

Each row must bind at minimum:

- schema_version
- source_pair_id
- row_id
- contrast_cell_id
- anchor_name
- anchor_token_index
- active_terminal_index
- layer_index
- support_absolute_token_indices
- support_vector_indices
- POST4_SPEED
- POST4_TURNING
- POST4_PATH_EFFICIENCY

Every endpoint must reconstruct exactly from the referenced support vectors.

No structural contrast may appear in this artifact.

### manifest.json

Must record at minimum:

- Phase D execution authority commit;
- Phase D runner implementation commit;
- runner path/SHA256/bytes;
- Phase C measurement implementation commit/path/SHA256;
- bridge and feasibility authority identities;
- runtime versions;
- Mamba/cache source hashes and byte counts;
- representative checkpoint path/SHA256;
- model config identity;
- tokenizer identity;
- canonical Gen4 artifact identity;
- event-anchor manifest identity;
- encoded-coordinate identity;
- source-pair/row/cell counts;
- support-state vector count;
- endpoint row count;
- layer index;
- scientific forward count;
- model replication count = 1;
- model forward mode = backbone_only;
- downstream forward count = 0;
- training = false;
- backward = false;
- statistical_testing = false;
- scientific_conclusion = NONE;
- blocker = null on success.

### SHA256SUMS.txt

Must hash the preceding four artifacts in fixed order.

The checksum file must not hash itself.


## 16. Publication transaction

The future runner must:

1. reject an existing final output directory;
2. reject an existing staging directory;
3. build all artifacts under one deterministic staging directory;
4. flush/close support_states.npy;
5. validate all schemas, counts, coordinates, reconstruction, finiteness, and
   hashes;
6. atomically publish only after full validation.

Overwrite and merge are prohibited.

A failed execution must not leave a valid-looking final scientific directory.


## 17. Phase D runner implementation validation

The implementation task authorized by this document may use synthetic data,
fake tensors, fake model objects, temporary directories, and static repository
reads only.

It must not load the representative checkpoint for model forward.

It must not execute the canonical Gen4 Mamba backbone.

Required tests include at minimum:

1. frozen parent and dependency identities;
2. canonical constants and representative checkpoint identity;
3. runner refuses scientific execution under wrong HEAD;
4. runner calls Phase C runtime gate before scientific model execution;
5. checkpoint authentication precedes deserialization;
6. only the representative checkpoint is accepted;
7. only primary layer 11 may be registered;
8. batch size is exactly 1;
9. exactly one synthetic backbone forward per synthetic row;
10. no downstream scientific forward;
11. support window is exactly [a-1,a+4];
12. per-row support coordinates are sorted unique union;
13. endpoint row ordering is deterministic;
14. endpoint reconstruction matches referenced synthetic vectors;
15. zero-transition and zero-path blockers propagate unchanged;
16. nonfinite measurement fails closed;
17. active-terminal prefix rule is enforced independently of padded length 128;
18. no statistical contrast/test is produced;
19. deterministic NPY shape/dtype/order validation;
20. bundle checksum and collision tests;
21. public CLI cannot silently alter scientific population, layer, checkpoint,
    endpoint set, or runtime contract.

No canonical scientific forward is permitted in these tests.


## 18. Expected narrow validation

The implementation validation command is expected to be:

python -m pytest \
  -p no:cacheprovider \
  tests/test_reason_router_gen4_native_mamba_state_extraction.py \
  -q

The exact command may be frozen with the implementation if the test filename
remains as authorized.

A test PASS establishes runner correctness only.

It establishes no native-state scientific result.


## 19. Strongest implementation verdict

The strongest successful verdict after implementation and synthetic
validation is:

PASS_READY_FOR_SEPARATE_PHASE_D_EXECUTION_AUTHORITY

This requires:

- only the two authorized implementation files changed;
- all required synthetic tests pass;
- no canonical Gen4 model forward occurred;
- no representative checkpoint scientific forward occurred;
- no native-state scientific artifact was produced;
- no endpoint outcome was inspected;
- no statistical analysis occurred.

PASS does not authorize scientific execution automatically.


## 20. Future execution environment disposition

The validated O0c exact-runtime precedent establishes that the target runtime
has previously existed in a Kaggle CPU-only environment.

Therefore the default future Phase D execution venue is:

KAGGLE_CPU_ONLY

with:

Accelerator = None
GPU = OFF

However this implementation authority does not authorize `cm kaggle`, Kaggle
execution, package mutation, or scientific model forward.

The later execution authority must exact-bind the frozen runner commit and
then provide the exact `cm kaggle` / run / collect workflow.


## 21. Scientific boundary

A successful future Phase D extraction will establish only that a valid,
provenance-bound native-state measurement artifact was produced.

Execution success alone will not establish any semantic native-state effect.

Artifact/provenance validity must be established before statistical
interpretation.

The 15 confirmatory hypotheses remain untouched until a separately frozen
statistical-analysis authority.
