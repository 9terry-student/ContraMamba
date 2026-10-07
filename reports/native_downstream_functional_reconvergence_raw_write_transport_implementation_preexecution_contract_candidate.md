# Native Downstream Functional Reconvergence
# Fixed Raw-Write Transport Implementation and Pre-Execution Contract

STATUS =
AUTHORIZED_FOR_BOUNDED_IMPLEMENTATION_AND_POST_FREEZE_EXECUTION_IF_ALL_GATES_PASS

BASE_DESIGN_COMMIT =
c9459d6788abdf4d62cfb38c5b2c2c2075dc9638

BASE_DESIGN =
reports/native_downstream_functional_reconvergence_raw_write_transport_design_candidate.md

PHASE =
IMPLEMENTATION_PLUS_PREEXECUTION_CONTRACT

TRAINING_ALLOWED =
NO

MODEL_PARAMETER_UPDATE_ALLOWED =
NO

OPTIMIZER_ALLOWED =
NO

PARAMETER_GRADIENT_ACCUMULATION_ALLOWED =
NO

ANALYSIS_AUTOGRAD_ALLOWED =
YES_DETACHED_RAW_WRITE_LEAF_ONLY

BACKWARD_METHOD_ALLOWED =
TORCH_AUTOGRAD_GRAD_ONLY

CHECKPOINT_MUTATION_ALLOWED =
NO

CONFIRMATORY_9601_9900_ALLOWED =
NO

NEW_DATA_ALLOWED =
NO

NEW_CHECKPOINTS_ALLOWED =
NO

VITAMINC_ROUTE_REOPEN_ALLOWED =
NO

COMMIT_PUSH_ALLOWED =
MANUAL_ONLY

# ---------------------------------------------------------------------------
# 1. Scientific target
# ---------------------------------------------------------------------------

SCIENTIFIC_TARGET =

Transport one source-local raw-write task-visible/complement decomposition
through the exact downstream native correction path without refitting the
decomposition at any later stage.

PRIMARY_QUESTION =

Does the large fixed raw-write complement physically contract relative to the
fixed task-visible component, and if so at which ordered downstream boundaries;
or does it remain large while staying functionally low-gain at the final task
readout?

THIS_STAGE_IS_NOT =

a new A-init ownership study
a coordinate-alignment study
a confident-error study
a VitaminC study
a training study
a new task-visible-basis search
a best-stage search
a best-k search
a threshold search

FORMAL_MECHANISM_CLASSIFICATION_THRESHOLDS =
NONE

PRIMARY_RESULT_FORM =

A preregistered descriptive stagewise transport profile on the already
consumed Phase3A P0 development population.

The labels

RECURRENT_SELECTIVE_FILTERING
READOUT_OR_GATE_SELECTIVE_FILTERING
DISTRIBUTED_GRADUAL_RECONVERGENCE
PERSISTENT_LOW_GAIN_COMPLEMENT

may be used only as bounded descriptive interpretations of the frozen profile.
No categorical promotion threshold may be invented after viewing outputs.

# ---------------------------------------------------------------------------
# 2. Frozen source evidence and identities
# ---------------------------------------------------------------------------

SOURCE_FORWARD_JACOBIAN_RECOVERY_EVIDENCE_FREEZE_COMMIT =
a4e78ee84bbfcc64d859ccc6af3efc0f987ee07e

SOURCE_RESIDUAL_LOCALIZATION_EVIDENCE_FREEZE_COMMIT =
3c0a3d8a67e9910f91de2354ba29a5c4b3b28942

SOURCE_INTERNAL_PRECURSOR_AUTHORITY_COMMIT =
e7eba19b102016131e4990f724c825cbec49ec5c

SOURCE_INTERNAL_PRECURSOR_AUTHORITY_CORRECTION_COMMIT =
e2c563188e9534e898c6ea944c5a05f4056b2fe3

SOURCE_INTERNAL_PRECURSOR_RUN =
gen5-ainit-internal-precursor-e2c5631-r1

SOURCE_INTERNAL_PRECURSOR_SUMMARY_SHA256 =
25c3e86d8d4ebbd19f7945755b5ca8033d5b05b0a6dd2a6804709b61c34293d2

SOURCE_INTERNAL_PRECURSOR_METRICS_SHA256 =
d00aedc95873ab47716d96773fdb81ee9a3b81c80479b06de38959829e9afa9d

SOURCE_INTERNAL_PRECURSOR_PROVENANCE_SHA256 =
a1408d07b6e8e3ed4dddfcab38466acfc2a915a2bb32195806122f102ca7c8e5

PARENT_CHECKPOINT_SHA256 =
1ff3fcf2ebd754ab6f9483d6a9982b9b04b9a4eb3357f9f8cdbe2b30399e7d2f

DEV_ORDER_SHA256 =
b42f64ec4961907fb59eb5fdf9e2e1714649b7952e551c4e9abf7e99c1456e25

DEV_ENCODING_SHA256 =
e3162804bfd184907ee1b22b3f4b4cf3ecee1069fe55661a1a4dbaeecb2cca51

DEV_ROWS =
840

VALID_TOKEN_REFERENCE =
60094

ARM =
G5-C0

PRESSURE =
P0

SPLIT_SEED =
16384

# ---------------------------------------------------------------------------
# 3. Frozen 3x3 checkpoint grid
# ---------------------------------------------------------------------------

CHECKPOINT_GRID =

A6201-R6201 157ca1c945c7f70b03638ef7e06750a4f4105e5504904272529ecd82bfba5ddf
A6201-R6202 ef03fbbedf3fab8efb255f2ccb6ec33cdb65ee6881a92e6b4d43fe1e719e40d4
A6201-R6203 15582eda034befb1c8d202f04c494f7fd5882bf9fd60ddc963232761057b3df7
A6202-R6201 3c217b43eb164583980bee91c39d53a7a3dc20d181e0341db35ffe8ece162ed3
A6202-R6202 1aa16196aa5aa338c30ac71dd20c7a6fb36a62eafdbe52bcf5696cfc3b87c214
A6202-R6203 d1c478b448c5f53e2a97552196a454f63f9d9081de614a6d9a5cd4e389e30ee2
A6203-R6201 d9e1062baf554867b212da57c9d30fe8efeccafc77255ba869affac691606359
A6203-R6202 32943df3558ef72eb7f7b7bfb70c6a1a03185a1ea6ca4dd83b9677cd59d13698
A6203-R6203 c536464dd8541423d18a2bfbfee40211f068885baa37730315dbf7cf0a784770

CHECKPOINT_DISCOVERY_RULE =

Paths are convenience only.
Every loaded cell must match the exact SHA256 above before model use.
No checkpoint selection, regeneration, fallback, or nearest-match behavior is
authorized.

PRIMARY_PAIR_CLASS =

same training RNG
different A-init

PRIMARY_UNORDERED_PAIR_COUNT =
9

CONTROL_PAIR_CLASS =

same A-init
different training RNG

CONTROL_UNORDERED_PAIR_COUNT =
9

BOTH_ORIENTATIONS_REQUIRED =
YES

PRIMARY_ORIENTATIONS =
18

CONTROL_ORIENTATIONS =
18

TOTAL_ORIENTATIONS =
36

# ---------------------------------------------------------------------------
# 4. Authorized implementation surface
# ---------------------------------------------------------------------------

AUTHORIZED_NEW_FILES =

scripts/audit_native_downstream_functional_reconvergence_raw_write_transport.py
tests/test_native_downstream_functional_reconvergence_raw_write_transport.py

AUTHORIZED_EXISTING_FILE_MODIFICATION =
NONE

IMPLEMENTATION_MAY_READ =

the frozen design
the frozen source evidence/artifacts
the exact 3x3 checkpoint grid
existing model/evaluation source required to reproduce the validated Gen5
forward and correction replay semantics

IMPLEMENTATION_MUST_NOT =

modify trainer semantics
modify model source
modify dataset/split/label semantics
change gradient-ownership production behavior
alter checkpoint contents
change existing frozen reports/artifacts
introduce architecture changes
introduce parameter training
introduce result-dependent selectors
introduce stage-specific projector fitting

# ---------------------------------------------------------------------------
# 5. Exact raw-write projector
# ---------------------------------------------------------------------------

For one oriented pair source i -> target j and one example:

delta_raw =
raw_write_target - raw_write_source

The task-visible decomposition is source-local and defined ONCE at raw_write.

TASK_COORDINATES =

m_refute = logit_refute - logit_not_entitled
m_support = logit_support - logit_not_entitled

TRUE_FORWARD_GRADIENT_SEMANTICS =
joint

At the source raw-write boundary:

1. preserve the exact frozen raw-write forward value;
2. detach raw_write;
3. reintroduce it as an equal-valued analysis leaf requiring gradient;
4. resume the exact frozen downstream forward map;
5. obtain the two task-margin gradient rows with torch.autograd.grad only.

No model parameter may require or accumulate an analysis gradient.

For the valid-token flattened raw-write tensor:

J_raw =
[g_refute ; g_support]

P_raw delta_raw =
J_raw^T (J_raw J_raw^T)^+ J_raw delta_raw

PROJECTOR_IMPLEMENTATION =

2x2 Gram only
do not materialize a dense projector

PROJECTOR_GRAM_DTYPE =
float64_cpu

PROJECTOR_PINV_HERMITIAN =
true

PROJECTOR_PINV_RTOL =
1e-12

PROJECTOR_PINV_ATOL =
0.0

delta_visible_raw =
P_raw delta_raw

delta_complement_raw =
delta_raw - delta_visible_raw

CRITICAL_ANTI_CIRCULARITY =

The projector is frozen at raw_write.
No downstream stage may recompute, refit, rotate, optimize, replace, or
stage-condition the visible/complement decomposition.

# ---------------------------------------------------------------------------
# 6. Oriented downstream-map semantics
# ---------------------------------------------------------------------------

Each scientific work item is an ORIENTED pair source i -> target j.

All hybrid trajectories for that orientation use the SOURCE checkpoint's
downstream continuation semantics from raw_write.

The four trajectories are:

SOURCE =
raw_write_source

VISIBLE_ONLY =
raw_write_source + delta_visible_raw

COMPLEMENT_ONLY =
raw_write_source + delta_complement_raw

FULL_TARGET_CHORD =
raw_write_source + delta_raw

The reverse target -> source orientation is a distinct required work item.

SOURCE_REPLAY_AUTHENTICATION =

An unperturbed source replay must reproduce the real source endpoint within the
frozen tolerance.

TARGET_REPLAY_AUTHENTICATION =

A separate unperturbed target replay under the target checkpoint semantics must
reproduce the real target endpoint within the frozen tolerance.

FULL_TARGET_CHORD_STATUS =

counterfactual source-map hybrid

FULL_TARGET_CHORD_REAL_TARGET_EQUALITY_REQUIRED =
NO

FULL_TARGET_CHORD_TARGET_DIAGNOSTIC =

Record its final-logit and margin discrepancy from the real target endpoint,
but do not use that discrepancy as a pass/fail criterion or to alter the
decomposition.

If implementation inspection proves that the post-raw-write continuation is
bitwise/shared-parameter identical across this frozen grid, record that fact
and the corresponding endpoint discrepancy; do not silently change the
scientific contract.

# ---------------------------------------------------------------------------
# 7. Ordered stages and masking
# ---------------------------------------------------------------------------

STAGE_ORDER =

S0 raw_write
S1 recurrent_state
S2 c_readout_pre_gate
S3 gated_scan
S4 layer22_out_proj

No stage may be added, skipped, reordered, or substituted.

VALID_TOKEN_MASK =

Use the frozen attention/valid-token mask from the authenticated dev encoding.

For every example and stage:

1. exclude padded token positions;
2. keep all valid token positions;
3. flatten all non-batch dimensions after masking, including every stage
   feature/state axis;
4. compute squared Euclidean/Frobenius energy in float64 accumulation.

No averaging over feature dimensions is permitted before energy computation.

# ---------------------------------------------------------------------------
# 8. Primary energy observables
# ---------------------------------------------------------------------------

For example e at stage j:

D_full_e(j) =
|| state_full_e(j) - state_source_e(j) ||_2^2

D_visible_e(j) =
|| state_visible_e(j) - state_source_e(j) ||_2^2

D_complement_e(j) =
|| state_complement_e(j) - state_source_e(j) ||_2^2

ABS_INTERACTION_e(j) =
||
  (state_visible_e(j) - state_source_e(j))
  +
  (state_complement_e(j) - state_source_e(j))
  -
  (state_full_e(j) - state_source_e(j))
||_2^2

ENERGY_EPSILON =
1e-30

The implementation must always persist both raw numerator and denominator
sufficient statistics.

For one orientation o, aggregate examples by RATIO OF SUMMED ENERGIES:

RET_VISIBLE_o(j) =
sum_e D_visible_e(j)
/
max(sum_e D_visible_e(0), ENERGY_EPSILON)

RET_COMPLEMENT_o(j) =
sum_e D_complement_e(j)
/
max(sum_e D_complement_e(0), ENERGY_EPSILON)

SELECTIVE_RETENTION_o(j) =
RET_COMPLEMENT_o(j)
/
max(RET_VISIBLE_o(j), ENERGY_EPSILON)

I_ABS_o(j) =
sum_e ABS_INTERACTION_e(j)

I_REL_o(j) =
sum_e ABS_INTERACTION_e(j)
/
max(sum_e D_full_e(j), ENERGY_EPSILON)

MANDATORY_INTERACTION_OUTPUTS =

I_ABS
I_REL
sum_D_full

A small D_full denominator may never be hidden by reporting I_REL alone.

By construction, report the observed numerical deviation of
SELECTIVE_RETENTION_o(0) from 1.

# ---------------------------------------------------------------------------
# 9. Pair and group aggregation
# ---------------------------------------------------------------------------

The scientific sampling hierarchy is:

example
-> oriented checkpoint pair
-> unordered checkpoint pair
-> pair class

For each unordered pair, both directions are mandatory.

For every strictly positive orientation-level SELECTIVE_RETENTION value, define:

LOG_SELECTIVE_RETENTION_o(j) =
ln(SELECTIVE_RETENTION_o(j))

The symmetric unordered-pair summary is:

PAIR_LOG_SELECTIVE_RETENTION(j) =
mean of the two orientation LOG_SELECTIVE_RETENTION values

Equivalent displayed pair retention may be obtained by exponentiating this
mean.

PRIMARY_GROUP_SUMMARY =

For each stage report across the 9 primary unordered pairs:

all 9 pair values
median
minimum
maximum
IQR

CONTROL_GROUP_SUMMARY =

The same summaries across the 9 control unordered pairs.

SOURCE_CELL_MATCHED_CONTROL =

For each of the 9 source cells:

- average LOG_SELECTIVE_RETENTION across its two primary outgoing orientations;
- average LOG_SELECTIVE_RETENTION across its two control outgoing orientations;
- report primary minus control.

Report all 9 source-cell contrasts plus median/min/max/IQR.

UNCERTAINTY_METHOD =

DESCRIPTIVE_PAIR_DISTRIBUTION_ONLY_NO_CI_NO_PVALUE

RATIONALE =

This is a bounded mechanism-discovery assay on an already consumed development
population with a small, dependent checkpoint grid. No population-level p-value
or pseudo-independent example-level confidence interval is authorized.

# ---------------------------------------------------------------------------
# 10. Final task-effect diagnostics
# ---------------------------------------------------------------------------

At the final task readout, for both:

centered three-class logits
two-margin vector

accumulate source-relative finite effects for:

FULL_TARGET_CHORD
VISIBLE_ONLY
COMPLEMENT_ONLY

and the finite decomposition interaction:

Delta_full - Delta_visible - Delta_complement

Report raw effect energies and normalized ratios.

The raw-write task-visible decomposition must retain the previously validated
scientific meaning that the visible component carries most measurable task
effect while the complement is low-gain.

These endpoint diagnostics are authentication/context for the transport assay;
they do not permit projector refitting.

# ---------------------------------------------------------------------------
# 11. Required pre-scientific authentication
# ---------------------------------------------------------------------------

No transport interpretation is allowed until all required authentication
checks pass.

REQUIRED_RUNTIME_IDENTITY =

same frozen model family and Gen5 G5-C0 semantics
exact parent checkpoint SHA256
exact 3x3 cell checkpoint SHA256 grid
exact dev order SHA256
exact dev encoding SHA256
840 rows
confirmatory population absent

REQUIRED_FORWARD_AUTHENTICATION =

historical edge-specific forward logits versus frozen functional fingerprint
joint analysis-mode forward logits versus historical forward logits

JOINT_EDGE_FORWARD_ATOL =
5e-6

SOURCE_TARGET_REPLAY_ATOL =
5e-5

RESIDUAL_CHAIN_ATOL =
5e-4

STREAMING_SEMANTIC_ATOL =
2e-4

RAW_WRITE_DECOMPOSITION_RECON_ATOL =
5e-5

Required frozen grouped residual-chain targets for
same-training-RNG / different-A checkpoints:

raw_write =
0.459225933132

recurrent_state =
0.235282854833

c_readout_pre_gate =
0.201189474462

gated_scan =
0.139629099045

layer22_out_proj =
0.155009066845

RAW_WRITE_PROJECTOR_AUTHENTICATION_TARGET =

same-training-RNG / different-A grouped local task-row-space squared-energy
fraction approximately:

0.000323695863574

RAW_WRITE_PROJECTOR_AUTHENTICATION_ATOL =
5e-5

RAW_WRITE_FINITE_MARGIN_AUTHENTICATION_TARGETS =

R_visible = 0.894958000002
R_complement = 0.00440660190005
R_interaction = 0.00843951843385

RAW_WRITE_FINITE_MARGIN_AUTHENTICATION_ATOL =
0.002

These targets authenticate reuse of the validated raw-write task-visible
semantics. They are not mechanism-selection thresholds.

# ---------------------------------------------------------------------------
# 12. GPU and sharding contract
# ---------------------------------------------------------------------------

SCIENTIFIC_GPU_TOPOLOGY =
TWO_INDEPENDENT_SINGLE_GPU_WORKERS_NO_DDP

PAIR_ORIENTATION_ATOMIC =
YES

CROSS_GPU_SCIENTIFIC_TENSOR_REDUCTION =
NO

Use the previously validated deterministic source-cell partition:

GPU0 source cells:

A6201-R6201
A6201-R6203
A6202-R6202
A6203-R6201
A6203-R6203

GPU1 source cells:

A6201-R6202
A6202-R6201
A6202-R6203
A6203-R6202

Each source cell owns all four outgoing scientific orientations:

two primary same-R/different-A targets
two control same-A/different-R targets

Therefore expected orientation counts are:

GPU0 = 20
GPU1 = 16
TOTAL = 36

All five stages for one orientation stay on one worker.

The implementation must print before scientific execution:

worker source cells
worker orientation identities
orientation counts
model-forward/replay counts by category
total expected forward/replay counts
target batch
artifact root

The exact observed preflight counts must be bound into
CONTRAMAMBA_RUN_EXPECTED_FORWARD_COUNTS before the scientific run is saved.

# ---------------------------------------------------------------------------
# 13. Persistence and outputs
# ---------------------------------------------------------------------------

OUTPUT_ROOT_RULE =

reports/native_downstream_functional_reconvergence_raw_write_transport_runs/<run-name>/

RUN_NAMING_RULE =

gen5-native-reconvergence-raw-write-transport-<implementation-shortcommit>-r1

REQUIRED_OUTPUT_FILES =

transport_summary.json
pair_stage_transport_metrics.jsonl
shard_manifest.json
run_provenance.json

Do not persist:

full hidden-state tensors
full recurrent-state tensors
full per-example Jacobian rows
full per-example gradient tensors
checkpoint copies

Persist only sufficient statistics, pair/stage summaries, authentication
diagnostics, exact identities, and provenance necessary to reproduce and audit
the interpretation.

Each worker must persist completed orientation chunks atomically enough to
permit merge-only recovery if final parent merge fails.

Every persisted worker chunk must bind:

run identity
implementation commit
source cell
orientation identity
input coverage
completion state
SHA256 after completion

# ---------------------------------------------------------------------------
# 14. Required implementation CLI
# ---------------------------------------------------------------------------

The new script must provide at least:

--help

--static-contract-check

--preflight

--run-worker

--merge-only

`--static-contract-check` must perform no model forward and no CUDA work.

`--preflight` may authenticate model/runtime/CUDA and compute execution counts
but must not emit scientific transport conclusions.

`--merge-only` must perform no GPU model computation and must fail closed on
missing/duplicate/hash-invalid/mixed-identity chunks.

No automatic batch fallback is authorized during scientific execution.

# ---------------------------------------------------------------------------
# 15. Implementation validation before execution
# ---------------------------------------------------------------------------

Before any scientific execution, the implementation commit must pass:

1. python syntax:

python -m py_compile scripts/audit_native_downstream_functional_reconvergence_raw_write_transport.py

2. narrow tests:

python -m pytest -q tests/test_native_downstream_functional_reconvergence_raw_write_transport.py

3. direct non-scientific CLI:

python scripts/audit_native_downstream_functional_reconvergence_raw_write_transport.py --help

4. direct static contract validation:

python scripts/audit_native_downstream_functional_reconvergence_raw_write_transport.py --static-contract-check

5. git diff integrity:

git diff --check

The unit tests must cover at minimum:

exact checkpoint-grid identities
pair enumeration: 18 primary + 18 control orientations
both orientations
deterministic 20/16 worker partition
valid-token masking
stage flattening and float64 energy accumulation
2x2 Gram projector semantics
visible + complement reconstruction
source-map orientation semantics
FULL_TARGET_CHORD not treated as target-endpoint authentication
ratio-of-sums aggregation
log-symmetric pair aggregation
I_ABS/I_REL denominator reporting
forbidden parameter gradients
forbidden backward()
forbidden optimizer/training paths
output collision protection
chunk identity/hash validation
merge-only missing/duplicate coverage rejection
confirmatory-population embargo

Scientific execution is blocked until this validation is reviewed and the
implementation files are manually committed and pushed.

# ---------------------------------------------------------------------------
# 16. Post-freeze execution authorization
# ---------------------------------------------------------------------------

POST_IMPLEMENTATION_SCIENTIFIC_EXECUTION_ALLOWED =
YES_IF_AND_ONLY_IF_ALL_PRECEDING_GATES_PASS

Execution must be bound to the exact pushed implementation commit.

Before run:

cm kaggle

then CPU-only/runtime provisioning and authentication.

GPU is enabled only for CUDA preflight and scientific worker execution.

The exact execution command must be generated only after the implementation
commit exists and the preflight confirms runtime identity, target batch,
worker partition, and expected forward/replay counts.

The run must use:

cm run save <run-name>
cm run <run-name>

with a dedicated empty repo-relative artifact root.

No current or historical artifact root may be reused.

GPU must be turned off immediately after GPU-dependent scientific work
completes.

Collection/import must use the current v4 explicit-artifact-root contract.

No result is scientific evidence until:

run completes
collect succeeds
import succeeds
artifact SHA/provenance validation succeeds
expected versus observed worker/orientation coverage matches

# ---------------------------------------------------------------------------
# 17. Hard stop conditions
# ---------------------------------------------------------------------------

STOP_IF =

base design identity mismatch
parent checkpoint SHA mismatch
any grid checkpoint SHA mismatch
dev order mismatch
dev encoding mismatch
row count mismatch
valid-token contract mismatch
3x3 grid incomplete
primary/control pair enumeration mismatch
orientation count mismatch
worker source-cell partition mismatch
source/target replay authentication failure
joint-vs-edge forward authentication failure
residual-chain authentication failure
raw-write projector authentication failure
raw-write finite-margin authentication failure
visible/complement reconstruction failure
stage order mismatch
padded tokens included in scientific energy
parameter gradient requested or accumulated
.backward() called
optimizer constructed
training path entered
checkpoint mutation
confirmatory 9601..9900 access
VitaminC access
stage-specific projector refit
best-stage search
best-k search
result-dependent threshold or selector creation
pair orientation split across GPUs
cross-GPU scientific tensor reduction
automatic batch fallback
output collision
stale artifact root
missing/duplicate orientation coverage
mixed commit/run chunk identity
chunk SHA failure
scientific output schema mismatch

On a stop condition, preserve existing valid chunks/logs and do not rerun GPU
work until the failure class is understood.

# ---------------------------------------------------------------------------
# 18. Scientific interpretation boundary
# ---------------------------------------------------------------------------

This assay may establish only the bounded transport profile of the fixed
source-local raw-write task-visible/complement decomposition through the five
frozen downstream internal boundaries on the Phase3A P0 development
population.

It may distinguish descriptively between:

early preferential complement contraction
later readout/gate/out-projection filtering
distributed multi-stage attenuation
persistent hidden complement with final task insensitivity

It does not establish:

independent population replication
a universal Mamba law
a global null manifold
formal gauge symmetry
behavior outside the frozen Gen5 system
a confident-error precursor
Native Q1 support or rejection
Transformer inferiority or Mamba superiority
a new training mechanism

If the transport profile is mechanistically clear, any later replication must
use a separately justified prospective holdout and a separately frozen
confirmatory interpretation rule.

END_OF_NATIVE_DOWNSTREAM_FUNCTIONAL_RECONVERGENCE_FIXED_RAW_WRITE_TRANSPORT_CONTRACT
