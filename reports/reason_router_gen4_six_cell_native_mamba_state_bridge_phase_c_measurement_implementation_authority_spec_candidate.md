# ContraMamba Gen4 Six-Cell Native Mamba State Bridge
# Phase C Measurement Implementation and Synthetic Validation Authority Candidate

STATUS =
CANDIDATE

AUTHORITY_ID =
GEN4_NATIVE_MAMBA_STATE_BRIDGE_PHASE_C_MEASUREMENT_IMPLEMENTATION_AUTHORITY

PHASE =
GEN4_NATIVE_MAMBA_STATE_BRIDGE_PHASE_C

SCIENTIFIC_CONCLUSION =
NONE

MODEL_FORWARD_ALLOWED =
SYNTHETIC_ONLY

CANONICAL_GEN4_MODEL_FORWARD_ALLOWED =
NO

CHECKPOINT_SCIENTIFIC_FORWARD_ALLOWED =
NO

NATIVE_STATE_EXTRACTION_ALLOWED =
NO

CANONICAL_KINEMATIC_ENDPOINT_COMPUTATION_ALLOWED =
NO

PRIMARY_STATISTICAL_TESTING_ALLOWED =
NO

TRAINING_ALLOWED =
NO

EVALUATION_ALLOWED =
NO

KAGGLE_ALLOWED =
NO

COMMIT_PUSH_DURING_IMPLEMENTATION =
NO


## 1. Purpose

This authority permits one bounded Phase C implementation and validation step
for the frozen Gen4 native-Mamba-state bridge.

It authorizes:

1. implementation of the native selective-SSM recurrent-state capture library;
2. implementation of the frozen local kinematic measurement definitions;
3. synthetic-only validation of source binding, capture semantics,
   non-interference, coordinate completeness, numerical definitions, and
   fail-closed behavior.

It does not authorize scientific native-state extraction from the 1800-row
Gen4 population.

It does not authorize computation of the 15 primary Gen4 native-state
hypotheses.

A successful Phase C implementation is only preparation for a later,
separately frozen Phase D native-state extraction authority.


## 2. Frozen parent authority and evidence

PHASE_AB_FEASIBILITY_FREEZE =
26fd55803acd05febefc8bd031f2fc23c17b0ef4

PHASE_AB_FEASIBILITY_AUTHORITY =
7337efa05f7230a6e0c700a8ee99579c337c4243

MECHANISTIC_BRIDGE_SPECIFICATION =
a2617aa037d1a9834003535b62ac81770a5b96aa

PHASE_AB_VERDICT =
PASS_READY_FOR_SEPARATE_PHASE_C_AUTHORITY

NATIVE_TENSOR_SOURCE =
BOUND

NATIVE_MAMBA_BACKBONE_IDENTITY =
IDENTICAL

NATIVE_STATE_MODEL_REPLICATION_COUNT =
1

NATIVE_MAMBA_LAYER_COUNT =
24

PRIMARY_LAYER =
11

PRIMARY_COMPLETE_PAIR_PREFIX_FEASIBILITY =
PASS_300_OF_300

O0C_REUSE_DISPOSITION =
BOUNDED_PHASE_C_INSTRUMENTATION_DELTA_REQUIRED


## 3. Historical O0c implementation lineage

The following O0c lineage is precedent only and is not new Gen4 evidence.

O0C_SCIENTIFIC_DESIGN_AUTHORITY =
242ad9ed70fc995ebda560911a7d0dfd2f18f9b3

O0C_IMPLEMENTATION_AUTHORITY =
6eca52722aaffa214e8546c6b616e1f670aecf77

O0C_INSTRUMENTATION_IMPLEMENTATION =
f724b81b9b69c842652a556f850ddebf53c11987

The following O0c principles are reused:

- exact runtime/source provenance before scientific capture;
- process-local default-disabled line tracing;
- direct observation of local native `ssm_state`;
- post-update/pre-readout capture;
- observer-owned `detach().clone()` snapshots;
- no mutation of model computation;
- prior trace restoration in `finally`;
- exact synthetic capture-disabled versus capture-enabled non-interference;
- fail-closed rejection of ambiguous source roles or coordinates.

Historical O0c datasets, comparisons, anchors, summary metrics, and scientific
claims are not inherited.


## 4. Exact future implementation scope

After this authority is frozen, only these two files may be created or
modified under Phase C implementation authority:

- scripts/reason_router_gen4_native_mamba_state_measurement.py
- tests/test_reason_router_gen4_native_mamba_state_measurement.py

No existing Gen4 evaluator, checkpoint, dataset, artifact, installed package,
Transformers source file, O0c implementation file, or repository authority
file may be modified by the implementation task.

A need for another implementation path is:

BLOCKED_SCOPE_REVIEW_REQUIRED


## 5. Scientific runtime source target

The future scientific capture path is pinned to the already validated
Transformers 5.0.0 sequential Mamba source identity used by the frozen
Gen4/O0c lineage.

TRANSFORMERS_VERSION =
5.0.0

MAMBA_SOURCE_MODULE =
transformers.models.mamba.modeling_mamba

MAMBA_SOURCE_SHA256 =
4c972b30f3c2cca977824fcc6891f956cd4387b6383aa7336848fbc5f2db1d83

MAMBA_SOURCE_BYTES =
39500

CACHE_SOURCE_MODULE =
transformers.cache_utils

CACHE_SOURCE_SHA256 =
6c123bbe3d23500462f0b617119a8231aa054b6d5475b295443a45f34466e6bc

CACHE_SOURCE_BYTES =
60432

CAPTURE_QUALNAME =
MambaMixer.slow_forward

RECURRENT_UPDATE_LINE =
409

CAPTURE_LINE =
410

FINAL_CACHE_PERSISTENCE_LINE =
417

The runtime gate must validate source root, source bytes, SHA256, code object,
source roles, line binding, and CPU sequential dispatch before any future
scientific capture.

A version string alone is insufficient.

The currently installed local Transformers 5.12.1 source is not authorized as
the future scientific capture source merely because it has analogous
recurrence semantics.

No package install, downgrade, mutation, vendoring, or patching is authorized
by Phase C.


## 6. Exact native-state object and timing

The sole scientific state object supported by this implementation is:

STATE_SOURCE =
native_selective_ssm_recurrent_state

IMPLEMENTATION_OBJECT =
MambaMixer.slow_forward local ssm_state

STATE_TIMING =
post_consumption_s_t

For consumed token x_t:

s_t = native recurrent state immediately after consuming x_t.

Capture occurs after the recurrence update and before the C readout consumes
the updated state.

The implementation must not substitute:

- hidden states;
- last_hidden_state;
- residual stream;
- convolution state;
- final cache entry;
- scan output;
- reconstructed state;
- router/logit/output quantities.

STATE_SHAPE =
[batch, 1536, 16]

PER_EXAMPLE_STATE_SHAPE =
[1536, 16]

FLATTENED_STATE_SIZE =
24576

VECTORIZATION =
contiguous per-example reshape(-1)

PRIMARY_LAYER_INDEX =
11

Phase C implementation may support generic layer registration internally for
testability, but the Gen4 scientific interface must fail closed unless the
requested primary scientific layer is exactly layer 11.


## 7. Observer mechanism

The bounded implementation mechanism is a default-disabled process-local
Python frame-line observer.

When disabled:

- no trace is installed;
- no scientific snapshot collection exists;
- model computation is untouched.

When enabled on a validated future runtime:

- bind only to the validated `MambaMixer.slow_forward` code object;
- respond only to the validated capture line;
- accept only a registered eligible mixer;
- read the zero-based token loop index;
- read local `ssm_state`;
- create `ssm_state.detach().clone()`;
- store the observer-owned snapshot by deterministic coordinate;
- never write to frame locals, cache, module, input, output, parameter,
  buffer, or autograd path;
- restore the previous trace in `finally`.

Duplicate coordinates, missing coordinates, wrong layer identity, ambiguous
token index, aliasing, nonfinite values, wrong tensor rank, or unsupported
source/backend identity fail closed.


## 8. Gen4 coordinate contract

The implementation inherits the frozen active Gen4 serialization:

claim[:63]
+
EOS token id 0
+
evidence[:64]

with:

add_special_tokens = false
maximum active serialized length = 128

Event coordinates are not rediscovered from model outcomes.

They are supplied by the frozen Phase A+B event-anchor provenance using
generator-declared spans and the authenticated tokenizer coordinate.

Required anchors:

- A_TITLE
- A_NAME
- A_ROLE
- A_PREDICATE
- A_IDENTITY

Required POST4 eligibility remains:

a + 4 <= terminal_index - 1

SHORTENED_POST_WINDOW =
PROHIBITED

The complete target remains 300/300 source pairs.

Phase C must not recompute or alter the frozen semantic population.


## 9. Frozen local kinematic implementation

For primary-layer state vector s_t:

v_t =
s_t - s_(t-1)

SPEED_t =
||v_t||_2

TURNING_t =
1 - cos(v_t, v_(t-1))

For anchor a:

POST4_SPEED =
mean(
    SPEED_(a+1),
    SPEED_(a+2),
    SPEED_(a+3),
    SPEED_(a+4)
)

POST4_TURNING =
mean(
    TURNING_(a+1),
    TURNING_(a+2),
    TURNING_(a+3),
    TURNING_(a+4)
)

POST4_PATH_EFFICIENCY =
||s_(a+4) - s_a||_2
/
sum_(t=a+1 to a+4) ||v_t||_2

Only ordinary Euclidean geometry is permitted.

No whitening, Mahalanobis metric, PCA, UMAP, t-SNE, learned embedding,
learned probe, layer pooling, Jacobian, state patching, or steering is
authorized.


## 10. Zero-denominator and numerical policy

This policy is frozen before scientific native-state observation.

A zero SPEED value is valid by itself.

For TURNING_t:

if ||v_t||_2 == 0
or ||v_(t-1)||_2 == 0,

the cosine is undefined.

Required behavior:

BLOCKED_UNDEFINED_ZERO_TRANSITION_NORM

For POST4_PATH_EFFICIENCY:

if

sum_(t=a+1 to a+4) ||v_t||_2 == 0,

the endpoint is undefined.

Required behavior:

BLOCKED_UNDEFINED_ZERO_PATH_LENGTH

No epsilon denominator is permitted.

No value imputation is permitted.

No row dropping is permitted.

No shortened window is permitted.

No contrast-specific pair removal is permitted.

Any nonfinite state, transition, norm, cosine, or endpoint is also a blocking
error.

The later Phase D scientific extraction must fail closed rather than silently
reduce the frozen 300-pair complete design.


## 11. Synthetic-only Phase C validation

Phase C implementation validation may execute synthetic tensors and synthetic
functions only.

It must not instantiate the canonical Gen4 model or load a scientific
checkpoint for forward execution.

Required tests include at minimum:

1. import safety:
   no model forward or runtime capture occurs at module import;

2. disabled observer:
   installs no trace and records no snapshots;

3. post-update capture:
   a synthetic recurrence proves the observer records the updated state at
   the intended token coordinate;

4. source binding negatives:
   wrong version/source bytes/hash/code object/source role/capture line/backend
   are rejected;

5. layer binding:
   Gen4 scientific primary layer is exactly 11;

6. coordinate completeness:
   duplicate, missing, reordered, or out-of-range token coordinates are
   rejected;

7. clone ownership:
   recorded state cannot alias the live synthetic source tensor;

8. trace restoration:
   prior tracing state is restored after success and exception;

9. exact non-interference:
   otherwise-identical synthetic capture-disabled and capture-enabled
   forwards have exactly equal output structure, tensors, metadata,
   parameters, requires_grad state, buffers, and allowed cache effects;

10. deterministic kinematic definitions:
    synthetic known trajectories reproduce exact expected speed, turning,
    and path-efficiency values;

11. zero-norm turning:
    blocks with BLOCKED_UNDEFINED_ZERO_TRANSITION_NORM;

12. zero path length:
    blocks with BLOCKED_UNDEFINED_ZERO_PATH_LENGTH;

13. nonfinite inputs/intermediates:
    fail closed;

14. prohibited scientific execution:
    Phase C public entry points cannot run the 1800-row Gen4 scientific
    extraction.

Exact equality in the non-interference test may not be weakened to tolerance
without new authority.


## 12. Validation command

After implementation, the narrow validation command is expected to be:

python -m pytest tests/test_reason_router_gen4_native_mamba_state_measurement.py -q

No full training/evaluation suite is required unless an independently observed
dependency defect makes that necessary.

A test PASS proves implementation correctness only.

It does not prove scientific native-state results.


## 13. Phase C success gate

The strongest successful Phase C implementation verdict is:

PASS_READY_FOR_SEPARATE_PHASE_D_NATIVE_STATE_EXTRACTION_AUTHORITY

This requires all of the following:

- implementation scope contains only the two authorized files;
- runtime/source target constants are exact;
- source-role validation is fail-closed;
- post-update capture semantics are proven synthetically;
- primary-layer binding is 11;
- observer-disabled behavior is inert;
- exact synthetic non-interference passes;
- deterministic kinematic definitions pass;
- zero-denominator blockers pass;
- no canonical Gen4 model forward occurred;
- no scientific checkpoint forward occurred;
- no native-state scientific artifact was produced;
- no primary kinematic outcome was inspected.

Phase C PASS does not authorize Phase D automatically.


## 14. Explicit prohibitions

Phase C does not authorize:

- loading one of the 18 Gen4 checkpoints for model forward;
- using the 18 identical native backbones as 18 replications;
- extraction of scientific s_t trajectories;
- computation of canonical Gen4 POST4 outcomes;
- inspection of the 15 primary native-state hypotheses;
- statistical testing;
- outcome-dependent layer selection;
- outcome-dependent anchor selection;
- alternate tokenizer semantics;
- semantic population changes;
- GPU/Kaggle execution;
- package mutation;
- training;
- evaluation;
- scientific interpretation.


## 15. Stop conditions

Stop and report BLOCKED if:

- HEAD is not the frozen Phase C authority parent when implementation begins;
- implementation requires files outside the authorized two-file scope;
- the exact 5.0.0 runtime/source contract cannot be represented fail-closed;
- the native state source becomes ambiguous;
- exact synthetic non-interference fails;
- zero-denominator handling would require an epsilon, imputation, row drop,
  or shortened window;
- any test requires canonical scientific model forward;
- any observed requirement conflicts with the frozen Gen4 bridge
  specification or Phase A+B feasibility freeze.


## 16. Final authority boundary

This document is an implementation and synthetic-validation authority only.

It creates no native-state scientific evidence.

It does not authorize scientific extraction.

It does not authorize the 15-test primary statistical family.

The next scientific execution boundary, if Phase C later passes and is frozen,
is a separate Phase D native-state extraction authority.
