# Generation-3 Grouped-Factorial Scientific Execution Authority

STATUS = CANDIDATE_ONLY_AT_AUTHORING

PHASE = GEN3_GROUPED_FACTORIAL_EXECUTION_AUTHORITY

GROUPED_IMPLEMENTATION_COMMIT = 3e0e9a435068c552abf20f3a74e0c3eccca344a3

GROUPED_IMPLEMENTATION_AUTHORITY = cc515b6ef5d0b5762ab06d5bbe916bc46ff4a5ef

GROUPED_STATIC_DESIGN = 65fdee2d5c4e60f8eadeb3d39ab4e601156d8bb4

FROZEN51_EVIDENCE_AUTHORITY = 792b7a7e9bdb7fa957fa3ecda6ee9c6e37ce0f96

PAIRWISE_VALIDATED_EVIDENCE = 4594f2d58e073610e63e679c97f4f19aea1db92e

TRAINER_CANONICAL_SHA256 = 45ea7785648171ae61232b7f505bdce002195f39081e5d3fb8fa9980a46d3aba

CONTRACT_TEST_CANONICAL_SHA256 = dfdf8a3d99678fc10454ef4883293bce327e285923f979d683944ea369b70053

GROUPED_EXECUTION_ALLOWED_AFTER_EXACT_FREEZE = YES

GROUPED_RUN_COUNT = 18

LAMBDA_SWEEP = FORBIDDEN

GROUP_SELECTION_CHANGES = FORBIDDEN

IMPLEMENTATION_CHANGE = FORBIDDEN

K_SERIES_MIXING = FORBIDDEN

## 1. Purpose

Generation-3 evidence currently supports:

PRIMARY_GEN3_INTERPRETATION = DISTRIBUTED_OWNERSHIP_DEPENDENCE

PAIRWISE_COMPONENT = SUPPORTED

REPRODUCIBLE_PAIRWISE_NONADDITIVITY = G7+G10 AND G5+G10

PAIRWISE_EXPLANATION_OF_GLOBAL_HALF = INCOMPLETE

HIGHER_ORDER_OR_GLOBAL_CUMULATIVE_COMPONENT = REMAINS_UNRESOLVED

The frozen FROZEN51 evidence further establishes a bounded descriptive
GLOBAL_CUMULATIVE_AUTHORIZATION_ASSOCIATED_RESIDUAL signature.

The frozen grouped-factorial static design therefore prespecifies the complete
three-factor U/Q/D decomposition.

The trainer implementation necessary to express the six missing proper-subset
conditions is now frozen at:

3e0e9a435068c552abf20f3a74e0c3eccca344a3

and was independently verified before manual freeze.

This authority candidate defines one bounded scientific execution matrix for
the six proper-subset grouped conditions across the three frozen training
seeds.

It does not authorize any new architecture, map search, lambda sweep, model
change, or post-hoc group selection.

## 2. Activation rule

At authoring time this document is CANDIDATE_ONLY.

Scientific grouped execution becomes authorized if and only if these exact
authority bytes are:

1. independently statically verified;
2. SHA256-verified locally;
3. staged as the sole intended tracked delta;
4. committed;
5. pushed to the active branch; and
6. remotely authenticated at the resulting commit.

No grouped scientific run is authorized before that freeze.

No separate activation commit is required.

## 3. Exact scientific source identity

Every grouped scientific run must execute from exactly:

3e0e9a435068c552abf20f3a74e0c3eccca344a3

The corresponding canonical committed implementation identities are:

scripts/train_controlled_v6b_minimal.py
SHA256 =
45ea7785648171ae61232b7f505bdce002195f39081e5d3fb8fa9980a46d3aba

tests/test_reason_router_p2_contract.py
SHA256 =
dfdf8a3d99678fc10454ef4883293bce327e285923f979d683944ea369b70053

No newer implementation commit may be substituted without a new execution
authority.

Each scientific run provenance must record source commit
3e0e9a435068c552abf20f3a74e0c3eccca344a3 and a clean source worktree.

A run originating from any different commit is inadmissible to this matrix.

## 4. Frozen grouped topology

The macro-groups remain exactly:

U:
G1 = F_TO_P
G2 = F_TO_S
G3 = P_TO_S

Q:
G4 = F_TO_Q
G5 = P_TO_Q
G6 = S_TO_Q

D:
G7 = F_TO_D
G8 = P_TO_D
G9 = S_TO_D
G10 = Q_TO_D

The partition is exhaustive and disjoint.

It is topology-prespecified rather than selected from measured arm performance.

## 5. Exact authorized grouped arms

The only new scientific arms authorized by this matrix are:

G3-GROUP-U-HALF

G3-GROUP-Q-HALF

G3-GROUP-D-HALF

G3-GROUP-U-Q-HALF

G3-GROUP-U-D-HALF

G3-GROUP-Q-D-HALF

No seventh grouped arm is authorized.

No U+Q+D arm is newly authorized because the historical D1 / GLOBAL-HALF
condition already supplies the all-group endpoint.

No arbitrary N-edge arm is authorized.

## 6. Exact arm maps

G3-GROUP-U-HALF:

G1 = 0.5
G2 = 0.5
G3 = 0.5
G4 = 1.0
G5 = 1.0
G6 = 1.0
G7 = 1.0
G8 = 1.0
G9 = 1.0
G10 = 1.0

G3-GROUP-Q-HALF:

G1 = 1.0
G2 = 1.0
G3 = 1.0
G4 = 0.5
G5 = 0.5
G6 = 0.5
G7 = 1.0
G8 = 1.0
G9 = 1.0
G10 = 1.0

G3-GROUP-D-HALF:

G1 = 1.0
G2 = 1.0
G3 = 1.0
G4 = 1.0
G5 = 1.0
G6 = 1.0
G7 = 0.5
G8 = 0.5
G9 = 0.5
G10 = 0.5

G3-GROUP-U-Q-HALF:

G1 = 0.5
G2 = 0.5
G3 = 0.5
G4 = 0.5
G5 = 0.5
G6 = 0.5
G7 = 1.0
G8 = 1.0
G9 = 1.0
G10 = 1.0

G3-GROUP-U-D-HALF:

G1 = 0.5
G2 = 0.5
G3 = 0.5
G4 = 1.0
G5 = 1.0
G6 = 1.0
G7 = 0.5
G8 = 0.5
G9 = 0.5
G10 = 0.5

G3-GROUP-Q-D-HALF:

G1 = 1.0
G2 = 1.0
G3 = 1.0
G4 = 0.5
G5 = 0.5
G6 = 0.5
G7 = 0.5
G8 = 0.5
G9 = 0.5
G10 = 0.5

For every authorized arm, the edge map must match the identity exactly.

## 7. Exact execution coordinate

Every grouped run is frozen to:

reason_router_mode = explicit_product

gradient_ownership_mode = edge_specific

reason_loss_weight = 0

split_seed = 8192

training_seeds = 180, 181, 182

freeze_encoder = true

frame_downstream_gradient_mode = joint

selected edge lambda = 0.5

non-selected edge lambda = 1.0

same dataset identities as frozen Gen3

same train/dev split identities as frozen Gen3

same primary reason order =
FRAME > PREDICATE > SUFFICIENCY > AUTHORIZED

secondary reasons = diagnostic only

same frozen three-way final-task semantics

No lambda sweep is authorized.

No conditional-first-blocker substitution is authorized.

No reason-loss restoration is authorized.

No dataset, split, seed, encoder, or model change is authorized.

## 8. Exact execution matrix

The authorized matrix is:

6 grouped arms
times
3 frozen training seeds

for exactly:

18 scientific runs.

Each arm must execute once at seed 180, once at seed 181, and once at seed 182.

No additional seed is authorized.

No arm may be omitted from the intended complete matrix unless a later explicit
authority closes the matrix with a documented missing run.

## 9. Historical reference reuse

Do not rerun:

canonical A0 / NONE / JOINT

historical D1 / GLOBAL-HALF / U+Q+D

the ten Gen3 single-edge-half arms

the twelve frozen pairwise arms

The analysis must reuse their already-frozen validated evidence.

The grouped matrix is specifically the six missing proper-subset states.

Historical endpoint provenance mismatch is a blocker rather than authorization
to silently retrain or substitute an endpoint.

## 10. Operational execution unit

The preferred execution unit is one Kaggle scientific session per seed:

seed180:
all six grouped arms

seed181:
all six grouped arms

seed182:
all six grouped arms

Every arm retains an independent run name, output directory, provenance record,
and collection identity.

The user controls GPU ON/OFF and Kaggle session termination.

Bootstrap, source verification, environment preparation, and preflight work
must use GPU OFF.

GPU may be enabled only for the actual authorized grouped training/evaluation
runs.

GPU should be disabled when the active seed batch finishes.

A failed run may be retried only with the same implementation commit, same
arm/map, same training seed, same split seed, and same scientific coordinate.

Retries must use a distinct retry identity and may not silently overwrite the
original run identity.

## 11. Canonical run naming

Use:

gen3-grouped-s<seed>-<group>-half

Canonical base names are therefore:

gen3-grouped-s180-u-half
gen3-grouped-s180-q-half
gen3-grouped-s180-d-half
gen3-grouped-s180-u-q-half
gen3-grouped-s180-u-d-half
gen3-grouped-s180-q-d-half

gen3-grouped-s181-u-half
gen3-grouped-s181-q-half
gen3-grouped-s181-d-half
gen3-grouped-s181-u-q-half
gen3-grouped-s181-u-d-half
gen3-grouped-s181-q-d-half

gen3-grouped-s182-u-half
gen3-grouped-s182-q-half
gen3-grouped-s182-d-half
gen3-grouped-s182-u-q-half
gen3-grouped-s182-u-d-half
gen3-grouped-s182-q-d-half

Retries append an explicit retry suffix such as:

-r1

A retry must never replace the original run identity silently.

## 12. Required run provenance

Every run must record at minimum:

source_commit

source_worktree_clean

reason_router_arm

training_seed

split_seed

reason_router_mode

gradient_ownership_mode

edge_gradient_lambdas

reason_loss_weight

freeze_encoder

frame_downstream_gradient_mode

Admission requires:

source_commit =
3e0e9a435068c552abf20f3a74e0c3eccca344a3

source_worktree_clean = true

training_seed = one of 180, 181, 182 matching the run identity

split_seed = 8192

reason_router_mode = explicit_product

gradient_ownership_mode = edge_specific

reason_loss_weight = 0

freeze_encoder = true

frame_downstream_gradient_mode = joint

and the exact canonical ten-entry edge map bound to the arm identity.

Any mismatch blocks scientific admission.

## 13. Required per-run artifacts

Each completed scientific run must retain:

clean_dev_predictions.json

run_provenance.json

training_report.json

training_report_predictions.jsonl

The existing workflow may also create selected_checkpoint.pt.

Checkpoint bytes are not a substitute for missing prediction, report, or
provenance artifacts.

Checkpoint bytes are not automatically part of the final scientific evidence
freeze scope.

## 14. Collection and import boundary

After an authorized run completes, it must be registered and saved using its
exact independent run identity.

Collection must preserve:

source commit

arm identity

training seed

split seed

output namespace

artifact hashes

The normal cm collect and cm import provenance paths must be used.

A provenance, source-commit, run-identity, or artifact-hash mismatch blocks
scientific admission.

Do not manually copy outputs between run namespaces to bypass collector checks.

Do not reuse a collected run from another commit.

## 15. Primary grouped scientific outcome

Aggregate accuracy is not the primary scientific endpoint.

For each seed and grouped condition X define:

A0_BREAK_X =
rows correct under matched historical A0 and wrong under X.

The primary residual outcome is:

A0_BREAK_X intersect FROZEN51.

The exact stable IDs and seed-specific occurrences must be retained.

Cross-seed recurrence means the same stable ID occurs in at least two of the
three frozen training seeds.

A scientific conclusion may not be based solely on aggregate break count.

## 16. Two-group nonadditivity outcome

For X and Y in U, Q, D, define:

NEW_FROZEN51_XY =
(A0_BREAK_XY intersect FROZEN51)
minus
(A0_BREAK_X union A0_BREAK_Y).

The exact authorized two-group tests are:

U+Q relative to U union Q

U+D relative to U union D

Q+D relative to Q union D

A cross-group cumulative interaction requires recurrent FROZEN51 stable IDs
created by the two-group condition beyond the constituent single-group union.

This is a descriptive intervention result and does not establish a unique
parameter-level causal mechanism.

## 17. Required aggregate measurements after validated import

For every grouped arm and seed, analysis must report at minimum:

accuracy

macro-F1

NOT_ENTITLED F1

REFUTE F1

SUPPORT F1

A0-correct break count and exact stable IDs

A0-wrong repair count and exact stable IDs

FROZEN51 overlap count and exact stable IDs

overlap with historical D1 A0-correct breaks

cross-seed recurrence

For U+Q, U+D, and Q+D, analysis must additionally report the exact
NEW_FROZEN51_XY set.

No conclusion may rely only on a three-seed mean.

## 18. Required residual geometry

For every grouped-condition occurrence overlapping FROZEN51, analysis must
retain matched historical A0 and grouped-condition values for at least:

frame probability

predicate coverage probability

sufficiency probability

q_FRAME

q_PREDICATE

q_SUFFICIENCY

q_AUTHORIZED

entitlement probability

polarity margin

SUPPORT-minus-NOT_ENTITLED final margin

final prediction

The historically observed common FROZEN51 signature is:

authorization-side increase
plus
final SUPPORT-versus-NOT_ENTITLED boundary movement.

Future grouped evidence must test whether that association is reproduced.

A uniform polarity direction is not required.

The analysis must not convert these associations into necessity or causal
sufficiency claims.

## 19. Frozen interpretation categories

The validated grouped evidence may support the following bounded descriptive
categories.

MACRO_GROUP_SUFFICIENT:

At least one of U, Q, or D alone reproducibly recovers recurrent FROZEN51
stable IDs together with the relevant authorization/final-boundary geometry.

This is an intervention-level sufficiency description only and does not prove
necessity or a unique mechanism.

CROSS_GROUP_CUMULATIVE_INTERACTION:

At least one two-group condition yields recurrent NEW_FROZEN51_XY stable IDs
outside the constituent single-group break union.

GLOBAL_ONLY_CUMULATIVE_THRESHOLD:

No proper grouped subset reproducibly recovers the recurrent FROZEN51 residual,
while historical U+Q+D / D1 retains it.

MIXED_GROUP_AND_GLOBAL_COMPONENT:

Some recurrent FROZEN51 behavior is recovered by proper grouped subsets, while
another recurrent component remains restricted to historical all-group D1.

These categories must remain distinguishable.

No category establishes a native Mamba recurrent-state mechanism.

## 20. Scientific claim boundary

This grouped experiment may test whether multi-group attenuation is associated
with the frozen authorization/final-boundary residual.

It does not by itself establish:

unique edge causation

unique pair causation

necessity of any macro-group

causal sufficiency of any macro-group

parameter-level ownership

gradient orthogonality

native Mamba recurrent-state mechanism

polarity irrelevance

optimal lambda

production readiness

The fixed 0.5 intervention is a prespecified probe, not an optimized value.

## 21. Explicit exclusions

This authority does not permit:

any arm outside the six grouped proper subsets

U+Q+D retraining

A0 retraining

D1 retraining

any lambda other than the frozen selected-edge 0.5 and non-selected 1.0 values

lambda sweeps

arbitrary N-edge maps

adaptive or learned ownership

seed-specific map changes

row-specific map changes

FROZEN51-conditioned training

post-hoc group selection

architecture changes

trainer changes

model changes

loss changes

dataset changes

split changes

additional training seeds

Gen4 execution

K-series execution

native-state claims

production-readiness claims

If execution requires a code change, stop and return to a new implementation
authority rather than patching the scientific source.

## 22. Execution success and scientific evidence are separate

Keep these states separate:

CODE_CORRECTNESS

EXECUTION_SUCCESS

ARTIFACT_AND_PROVENANCE_VALIDITY

SCIENTIFIC_CONCLUSION

The frozen implementation and its contract tests establish code correctness.

A successful Kaggle command establishes only execution success.

A run enters the scientific grouped matrix only after collection, local import,
and provenance validation.

Scientific interpretation must be deferred until all intended admitted runs
are present, unless a later explicit authority closes an incomplete matrix.

## 23. Stop conditions

Stop scientific execution and report a blocker if:

HEAD or Kaggle source commit differs from
3e0e9a435068c552abf20f3a74e0c3eccca344a3

source worktree is dirty

an unauthorized grouped arm is requested

arm/map identity mismatch occurs

training seed differs from run identity

split_seed differs from 8192

reason_router_mode differs from explicit_product

gradient_ownership_mode differs from edge_specific

reason_loss_weight is nonzero

encoder is not frozen

frame_downstream_gradient_mode differs from joint

dataset or split identity differs from frozen Gen3

provenance is incomplete

collector reports a source/hash/run mismatch

import reports a source/hash/run mismatch

a code modification appears necessary

a lambda sweep appears necessary

group selection changes appear necessary

a historical endpoint rerun appears necessary

## 24. Decision

AUTHORIZED_AFTER_EXACT_FREEZE
    = 6 GROUPED PROPER-SUBSET ARMS
    x SEEDS 180, 181, 182
    = 18 SCIENTIFIC RUNS

SCIENTIFIC_SOURCE_COMMIT
    = 3e0e9a435068c552abf20f3a74e0c3eccca344a3

HISTORICAL_REFERENCE_RERUN
    = NO

GROUPED_EXECUTION
    = YES, ONLY AFTER THIS AUTHORITY IS EXACTLY FROZEN

KAGGLE
    = AUTHORIZED ONLY FOR THE EXACT FROZEN 18-RUN MATRIX

IMPLEMENTATION_CHANGE
    = NO

LAMBDA_SWEEP
    = NO

SCIENTIFIC_CONCLUSION
    = DEFERRED UNTIL VALIDATED COLLECTION, IMPORT, AND ANALYSIS

PASS_READY_FOR_INDEPENDENT_GEN3_GROUPED_EXECUTION_AUTHORITY_VERIFICATION