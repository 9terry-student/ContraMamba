# Native Downstream Functional Reconvergence
# Raw-Write Fixed-Decomposition Transport Design Candidate

STATUS =
SCIENTIFIC_DESIGN_ONLY_NO_IMPLEMENTATION_OR_EXECUTION_AUTHORITY

BASE_HEAD =
40fffff224ef4de42a8fee1ad8ab1db65ef15a1b

TRAINING_ALLOWED =
NO

MODEL_PARAMETER_UPDATE_ALLOWED =
NO

KAGGLE_EXECUTION_ALLOWED =
NO

SCIENTIFIC_EXECUTION_ALLOWED =
NO

IMPLEMENTATION_AUTHORIZED =
NO

# ---------------------------------------------------------------------------
# 1. Scientific motivation
# ---------------------------------------------------------------------------

SOURCE_M8_CLASSIFICATION =
NONSEPARABLE_PATH_DEPENDENT_REPRESENTATIVES_WITH_FUNCTIONAL_EQUIVALENCE

SOURCE_UNRESOLVED_BOUNDARY =
UNRESOLVED_NATIVE_DOWNSTREAM_FUNCTIONAL_RECONVERGENCE_MECHANISM

ESTABLISHED_GEN5_FACTS =

1. Different A-init conditions produce strongly different layer-22
   internal representatives.

2. Those representatives are nearly functionally equivalent on the frozen
   Phase3A task distribution.

3. Most A-init-induced hidden residual energy lies in a downstream
   effectively-null or very-low-gain complement.

4. A much smaller task-visible component carries essentially all measurable
   endpoint functional difference.

5. The visible/low-gain decomposition is already present at the learned
   raw-write boundary.

6. Existing internal localization recomputed a task-visible decomposition
   separately at each internal stage.

UNRESOLVED_QUESTION =

The existing evidence does not establish how one fixed raw-write
visible/complement decomposition is transported through the native downstream
path.

In particular, it remains unresolved whether functional reconvergence is
produced primarily by:

- selective contraction in recurrent-state evolution;
- C/readout filtering;
- gating;
- out-projection;
- distributed gradual attenuation;
- or persistence of a large complement that simply remains low-gain to the
  final task readout.

# ---------------------------------------------------------------------------
# 2. Primary scientific question
# ---------------------------------------------------------------------------

PRIMARY_QUESTION =

If the A-init endpoint residual is decomposed ONCE at raw_write into a frozen
task-visible component and its orthogonal complement, how are those exact two
components transported through the downstream native Mamba computation?

PRIMARY_MECHANISTIC_TARGET =

LOCALIZE_WHERE_FIXED_RAW_WRITE_LOW_GAIN_VARIATION_IS_SELECTIVELY_ATTENUATED
RELATIVE_TO_FIXED_RAW_WRITE_TASK_VISIBLE_VARIATION

THIS_IS_NOT =

a confident-error study
a correctness study
a hallucination study
a VitaminC study
an A-init coordinate-alignment study
a new training study
a search for a better task-visible basis

# ---------------------------------------------------------------------------
# 3. Population boundary
# ---------------------------------------------------------------------------

DISCOVERY_POPULATION =

Frozen Phase3A P0 dev population already used by the Gen5 mechanistic program.

EXPECTED_DEV_ROWS =
840

PRIMARY_CHECKPOINT_PAIR_CLASS =

same training RNG
different A-init

EXPECTED_PRIMARY_UNORDERED_PAIRS =
9

CONTROL_CHECKPOINT_PAIR_CLASS =

same A-init
different training RNG

EXPECTED_CONTROL_UNORDERED_PAIRS =
9

CONFIRMATORY_9601_9900_ALLOWED =
NO

REASON =

The xg1_fact_9601..xg1_fact_9900 population has already been consumed by the
one-shot visible-vs-complement confirmatory replication and may not be reused
for threshold tuning, basis fitting, or rescue.

NEW_PROSPECTIVE_HOLDOUT_REQUIRED_FOR_LATER_REPLICATION =
YES_IF_DISCOVERY_MECHANISM_IS_CLEAR

# ---------------------------------------------------------------------------
# 4. Frozen raw-write decomposition
# ---------------------------------------------------------------------------

For each frozen ordered source/target checkpoint orientation and aligned dev
example, let:

delta_raw =
raw_write_target - raw_write_source

ORIENTATION_SEMANTICS =

The ordered pair (source, target) defines one counterfactual transport assay.
The reverse (target, source) is a distinct required orientation rather than an
implicit symmetry assumption.

Define the source-local raw-write two-margin Jacobian under the SOURCE
checkpoint downstream map:

J_raw =
d(two_margin_source_downstream) / d(raw_write_source)

Define P_raw as the orthogonal projector onto row(J_raw):

P_raw =
ORTHOGONAL_PROJECTOR_ONTO_ROW_SPACE(J_raw)

The later implementation/preexecution contract must freeze one numerically
exact realization of this projector, including dtype, rank/pseudoinverse or
SVD tolerance, and equality/authentication tolerances. Those numerical choices
must be fixed before viewing transport outputs and may not be selected to
improve downstream separation.

Then freeze exactly once:

delta_visible_raw =
P_raw delta_raw

delta_complement_raw =
(I - P_raw) delta_raw

The decomposition is defined at raw_write only and is source-local for that
ordered orientation.

CRITICAL_LOCK =

NO stage downstream of raw_write may refit, recompute, rotate, optimize, or
replace this visible/complement decomposition using that stage's outcomes.

The scientific object is transport of the SAME raw-write perturbation
components under a fixed source-checkpoint downstream map.

# ---------------------------------------------------------------------------
# 5. Finite hybrid trajectories
# ---------------------------------------------------------------------------

Let F_source^j(z) denote the stage-j downstream response obtained by injecting
raw-write value z into the SOURCE checkpoint while keeping all downstream
source-checkpoint parameters fixed.

Construct four primary forward trajectories, ALL under that same source
checkpoint downstream map:

SOURCE =
F_source(raw_write_source)

VISIBLE_ONLY =
F_source(raw_write_source + delta_visible_raw)

COMPLEMENT_ONLY =
F_source(raw_write_source + delta_complement_raw)

FULL_TARGET_CHORD =
F_source(raw_write_source + delta_raw)
=
F_source(raw_write_target)

SOURCE_MAP_LOCK =

VISIBLE_ONLY, COMPLEMENT_ONLY, and FULL_TARGET_CHORD MUST NOT switch to target
checkpoint downstream parameters. They are oriented counterfactual
interventions on the source model.

REAL_ENDPOINT_AUTHENTICATION =

1. An unperturbed source replay under the source checkpoint must reproduce the
   real source endpoint within a frozen tolerance.
2. A separate unperturbed target replay under the target checkpoint must
   reproduce the real target endpoint within a frozen tolerance.
3. FULL_TARGET_CHORD under the source downstream map is NOT required or
   expected to equal the real target endpoint, because the downstream
   checkpoint parameters differ.

The difference between FULL_TARGET_CHORD under the source map and the real
target endpoint may be reported only as a separately named
DOWNSTREAM_PARAMETER_MISMATCH_DIAGNOSTIC. It is not an authentication failure
and is not part of the primary retention statistic.

Both ordered orientations of every primary and control checkpoint pair are
required.

No learned interpolation, new optimization, parameter mixing, or state fitting
is allowed.

# ---------------------------------------------------------------------------
# 6. Ordered internal boundaries
# ---------------------------------------------------------------------------

Use the already validated Gen5 internal-stage semantics and exact order:

S0 =
raw_write

S1 =
recurrent_state

S2 =
c_readout_pre_gate

S3 =
gated_scan

S4 =
layer22_out_proj

Final task diagnostics may additionally include:

centered three-class logits
two-margin vector

but they are downstream diagnostics, not replacement internal stages.

NO_STAGE_SCAN_OUTSIDE_THIS_ORDER =
YES

# ---------------------------------------------------------------------------
# 7. Primary transport observables
# ---------------------------------------------------------------------------

All primary energies are computed PER EXAMPLE before any aggregation.

STAGE_TENSOR_REDUCTION_SEMANTICS =

- if a stage tensor contains the canonical sequence/token axis, exclude padded
  positions using the same frozen valid-token mask used by the authenticated
  forward path;
- if a stage tensor has no sequence/token axis, no token mask is introduced;
- after masking, flatten all remaining non-batch coordinates for that example;
- use squared Euclidean/Frobenius energy on that flattened tensor;
- never average examples, tokens, states, channels, or pair orientations before
  computing the squared response energy;
- exact tensor axis identities and mask broadcasting rules must be frozen in
  the implementation/preexecution contract and authenticated on a replay
  slice before scientific execution.

At each internal stage S_j, define source-relative squared response energies:

D_full(j) =
|| state_full(j) - state_source(j) ||_2^2

D_visible(j) =
|| state_visible(j) - state_source(j) ||_2^2

D_complement(j) =
|| state_complement(j) - state_source(j) ||_2^2

Define the absolute finite nonlinear decomposition residual energy:

I_ABS(j) =
||
  (state_visible(j) - state_source(j))
  +
  (state_complement(j) - state_source(j))
  -
  (state_full(j) - state_source(j))
||_2^2

Define its normalized diagnostic:

I_REL(j) =
I_ABS(j)
/
max(D_full(j), epsilon)

with one frozen numerical epsilon specified before execution.

MANDATORY_INTERACTION_REPORTING =

I_ABS(j), I_REL(j), and D_full(j) must be reported together. A large I_REL(j)
when D_full(j) is near the numerical floor may not be interpreted as strong
nonlinearity without the corresponding absolute residual energy.

Define component retention relative to raw_write:

RET_VISIBLE(j) =
D_visible(j) / max(D_visible(0), epsilon)

RET_COMPLEMENT(j) =
D_complement(j) / max(D_complement(0), epsilon)

Primary relative-retention statistic:

SELECTIVE_RETENTION(j) =
RET_COMPLEMENT(j)
/
max(RET_VISIBLE(j), epsilon)

By definition:

SELECTIVE_RETENTION(0) approximately 1

Interpretation:

SELECTIVE_RETENTION(j) < 1
means the raw-write complement has been attenuated more strongly than the
raw-write visible component by stage j.

SELECTIVE_RETENTION(j) approximately 1
means no preferential state-space attenuation has occurred up to that stage.

SELECTIVE_RETENTION(j) > 1
means the complement has been retained or amplified relative to the visible
component.

No claim threshold is frozen by this design document.

# ---------------------------------------------------------------------------
# 8. Primary localization logic
# ---------------------------------------------------------------------------

The primary descriptive sequence is:

SELECTIVE_RETENTION(recurrent_state)
SELECTIVE_RETENTION(c_readout_pre_gate)
SELECTIVE_RETENTION(gated_scan)
SELECTIVE_RETENTION(layer22_out_proj)

The scientific audit must distinguish at least four mechanisms:

RECURRENT_SELECTIVE_FILTERING =

Strong preferential complement attenuation is already present between
raw_write and recurrent_state and persists downstream.

READOUT_OR_GATE_SELECTIVE_FILTERING =

Visible/complement retention remains similar through recurrent_state, then
separates materially at C readout, gating, or out-projection.

DISTRIBUTED_GRADUAL_RECONVERGENCE =

No single boundary dominates, but complement retention decreases progressively
across multiple ordered stages.

PERSISTENT_LOW_GAIN_COMPLEMENT =

The complement remains large in internal-state norm through layer22_out_proj,
yet its final logit/margin effect remains extremely small.

The last mechanism would imply that functional reconvergence is primarily
downstream insensitivity rather than physical contraction of the hidden
difference.

# ---------------------------------------------------------------------------
# 9. Controls
# ---------------------------------------------------------------------------

Required controls:

1. same-A-init / different-training-RNG checkpoint pairs;
2. both ordered pair orientations;
3. source replay authentication under source checkpoint downstream parameters;
4. target replay authentication under target checkpoint downstream parameters;
5. explicit confirmation that FULL_TARGET_CHORD is replayed under SOURCE
   downstream parameters and is not authenticated against the real target
   endpoint;
6. source-local projector authentication against the frozen two-margin
   forward-Jacobian semantics;
7. raw_write decomposition reconstruction:
   delta_visible_raw + delta_complement_raw = delta_raw
   within a frozen numerical tolerance;
8. stage tensor/mask/reduction authentication on a fixed replay slice;
9. absolute plus normalized nonlinear-interaction diagnostics;
10. no parameter gradients;
11. no checkpoint mutation;
12. no training.

The primary A-init mechanism must be reported alongside the RNG-control
transport behavior.

# ---------------------------------------------------------------------------
# 10. Anti-circularity
# ---------------------------------------------------------------------------

The raw-write projector may depend only on the source-local frozen
two-margin forward Jacobian under the already validated forward semantics.

It may not depend on:

- recurrent_state outcomes;
- C-readout outcomes;
- gated-scan outcomes;
- out-projection outcomes;
- final selective-retention values;
- a best-stage search;
- a best-k search;
- correctness;
- confidence;
- native-state outcome labels.

No stage-specific task-visible basis may be fit for the primary assay.

# ---------------------------------------------------------------------------
# 11. Statistical / inferential boundary
# ---------------------------------------------------------------------------

This first assay is a bounded mechanism-discovery study on an already consumed
development population.

It must report:

- per-orientation, per-pair, and aggregate stagewise retention;
- dispersion across examples;
- dispersion across the 9 primary checkpoint pairs;
- paired primary/control contrasts;
- orientation consistency and orientation asymmetry;
- I_ABS, I_REL, and D_full together at every stage;
- the separately named downstream-parameter-mismatch diagnostic if it is
  retained at all.

It must not claim independent population replication.

Before scientific execution, a later implementation/execution specification
must freeze:

- ordered source/target orientation identity for every assay;
- source-checkpoint downstream-map injection semantics;
- exact P_raw numerical realization, dtype, rank/pseudoinverse or SVD tolerance,
  and projector authentication tolerance;
- stage tensor axis identities, valid-token masking, flattening, and squared
  energy reduction semantics;
- aggregation unit and aggregation order;
- numerical epsilon;
- exact effect summaries;
- mandatory joint reporting of I_ABS, I_REL, and D_full;
- uncertainty method;
- any formal mechanism-classification thresholds;
- exact implementation files;
- exact runtime/provenance contract.

No thresholds may be chosen after viewing the transport outputs.

# ---------------------------------------------------------------------------
# 12. Relation to the failed Native Q1 VitaminC route
# ---------------------------------------------------------------------------

VITAMINC_NATIVE_Q1_ROUTE =
CLOSED

The VitaminC closure established sampling-frame insufficiency for the
predeclared confident-correct versus confident-wrong confirmatory cohort.

It did NOT test the native-state precursor hypothesis.

The present assay does not reopen that route.

It asks a separate Gen5 mechanism question using the already validated
A-init representational-equivalence system.

CONFIDENCE_THRESHOLD_SEARCH =
PROHIBITED

VITAMINC_SELECTOR_SEARCH =
PROHIBITED

VITAMINC_NATIVE_STATE_EXTRACTION =
PROHIBITED

# ---------------------------------------------------------------------------
# 13. Scientific value of possible outcomes
# ---------------------------------------------------------------------------

If RECURRENT_SELECTIVE_FILTERING is observed:

The strongest next hypothesis is that native recurrence actively contracts
representational freedom while preserving the small task-visible component.

If READOUT_OR_GATE_SELECTIVE_FILTERING is observed:

The reconvergence mechanism is localized later than recurrent-state evolution.

If DISTRIBUTED_GRADUAL_RECONVERGENCE is observed:

The mechanism is a multi-stage anisotropic transport process rather than one
discrete bottleneck.

If PERSISTENT_LOW_GAIN_COMPLEMENT is observed:

Large internal representative differences survive physically but are ignored
by downstream task-sensitive directions; functional equivalence is then better
described as readout insensitivity than hidden-state convergence.

All four outcomes are scientifically discriminating.

# ---------------------------------------------------------------------------
# 14. Scope exclusions
# ---------------------------------------------------------------------------

This design does not authorize:

training
Kaggle execution
GPU execution
new model seeds
new checkpoints
new dataset generation
new holdout consumption
VitaminC execution
native confident-error analysis
tau_e mapping
architecture modification
nonlinear detector training
best-stage search
best-layer search
basis refitting
threshold tuning

# ---------------------------------------------------------------------------
# 15. Current disposition
# ---------------------------------------------------------------------------

DESIGN_VERDICT =
STATIC_REVIEW_CORRECTIONS_INTEGRATED_READY_FOR_FREEZE_REVIEW

NEXT_ACTION_IF_FROZEN =
BOUNDED_IMPLEMENTATION_AND_PREEXECUTION_CONTRACT_FOR_FIXED_RAW_WRITE_TRANSPORT

NATIVE_Q1_STATUS =
UNRESOLVED_BUT_CURRENT_VITAMINC_ROUTE_CLOSED

GEN5_AINIT_STATUS =
CLOSED_AS_PRIMARY_EXPLANATORY_AXIS

ACTIVE_MECHANISTIC_TARGET =
NATIVE_DOWNSTREAM_FUNCTIONAL_RECONVERGENCE

END_OF_NATIVE_DOWNSTREAM_FUNCTIONAL_RECONVERGENCE_RAW_WRITE_TRANSPORT_DESIGN
