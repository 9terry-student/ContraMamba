# Generation-3 pairwise interaction execution authority

```text
STATUS = CANDIDATE_ONLY_AT_AUTHORING
PHASE = GEN3_PAIRWISE_INTERACTION_EXECUTION_AUTHORITY
PAIRWISE_IMPLEMENTATION_COMMIT = d62c375e2d730f040717422d3951199b71dc688e
IMPLEMENTATION_AUTHORITY = a3716d6431b268a33b1adf46f7af581f1a78c72e
CAPABILITY_VERIFICATION = 3e1ce61af8b17b03e6f8c387606b182ae1490117
PAIRWISE_SELECTION = c50816540c2a4556fd45a56326fa9609616fa3a9
GEN3_FIRST_PASS_EVIDENCE = 28d7fa2fcc0d286a0cd16723c302a45f214b2901
PAIRWISE_EXECUTION_ALLOWED_AFTER_EXACT_FREEZE = YES
PAIRWISE_RUN_COUNT = 36
LAMBDA_SWEEP = FORBIDDEN
PAIR_SELECTION_CHANGES = FORBIDDEN
K_SERIES_MIXING = FORBIDDEN
```

## 1. Purpose

Generation-3 first-pass evidence established:

```text
PRIMARY = DISTRIBUTED_OWNERSHIP_DEPENDENCE
SECONDARY = GLOBAL_CUMULATIVE_OR_INTERACTION_COMPONENT_REMAINS_UNRESOLVED
LOCALIZED_DOWNSTREAM_ADAPTATION_DEPENDENCE = NOT_SUPPORTED
```

The frozen pairwise-selection audit then defined exactly 12 downstream
topology-driven pair candidates without using post-hoc "worst arm" ranking.

The trainer implementation required to express those exact pairwise arms is now
frozen at:

`d62c375e2d730f040717422d3951199b71dc688e`

and was independently verified before manual freeze.

This authority permits one bounded scientific execution matrix whose purpose is
to determine whether the unresolved GLOBAL-HALF component contains reproducible
pairwise non-additivity, pairwise cumulative-only behavior, null/tolerant
pairwise behavior, or antagonistic/repairing behavior under the already-frozen
edge-specific intervention semantics.

## 2. Activation rule

At authoring time this file is `CANDIDATE_ONLY`.

If and only if these exact report bytes are:

1. SHA256-verified locally;
2. staged as the only intended tracked delta;
3. committed;
4. pushed to the active branch; and
5. remotely authenticated at the resulting commit,

then that frozen commit becomes the execution authority for the matrix defined
below.

No scientific pairwise run is authorized before that freeze.

## 3. Exact implementation identity

All scientific pairwise runs must execute from exactly:

`d62c375e2d730f040717422d3951199b71dc688e`

No newer implementation commit may be substituted without a new authority.

Each run provenance must report the same source commit and a clean source
worktree.

No run from a different commit may be imported into this matrix.

## 4. Exact pairwise arm matrix

The only authorized arms are:

```text
G3-G4-G5-HALF
G3-G4-G6-HALF
G3-G5-G6-HALF
G3-G7-G8-HALF
G3-G7-G9-HALF
G3-G7-G10-HALF
G3-G8-G9-HALF
G3-G8-G10-HALF
G3-G9-G10-HALF
G3-G4-G10-HALF
G3-G5-G10-HALF
G3-G6-G10-HALF
```

Frozen conceptual mapping:

```text
G4  = F_TO_Q
G5  = P_TO_Q
G6  = S_TO_Q
G7  = F_TO_D
G8  = P_TO_D
G9  = S_TO_D
G10 = Q_TO_D
```

For each arm, exactly the two identity-named edges must equal `.5`; the other
eight canonical edges must equal `1.0`.

No additional pair is authorized.

## 5. Exact execution coordinate

For every arm, the execution coordinate is frozen as:

```text
reason_router_mode = explicit_product
gradient_ownership_mode = edge_specific
reason_loss_weight = 0
split_seed = 8192
training_seeds = 180, 181, 182
freeze_encoder = true
frame_downstream_gradient_mode = joint
lambda_probe = 0.5
same dataset identities as Gen3 first pass
same train/dev split identities as Gen3 first pass
same primary reason order:
    FRAME > PREDICATE > SUFFICIENCY > AUTHORIZED
secondary reasons = diagnostic only
final task semantics = frozen three-way task
```

This yields:

```text
12 pairwise arms × 3 seeds = 36 scientific runs
```

No lambda sweep, adaptive ownership, conditional-first-blocker arm, loss
restoration, dataset change, split change, or encoder change is authorized.

## 6. Historical references

Do not rerun:

- A0 / JOINT
- D1 / GLOBAL-HALF
- the ten Gen3 single-edge-half arms

Use the already-frozen validated evidence as historical references.

Pairwise analysis must compare each pair against:

1. matched historical A0 for that seed;
2. matched historical D1 for that seed;
3. both constituent Gen3 single-edge arms for that seed.

A historical reference provenance mismatch is a blocker, not a reason to
silently substitute or rerun a different reference.

## 7. Operational execution unit

The preferred operational unit is one Kaggle session per seed:

```text
seed180: all 12 pairwise arms
seed181: all 12 pairwise arms
seed182: all 12 pairwise arms
```

Within a seed session, each arm must retain an independent exact run identity
and independent output namespace.

The user controls GPU ON/OFF and Kaggle session termination.

Bootstrap/preflight work must use GPU OFF.

GPU may be turned ON only for the actual authorized training runs and should be
turned OFF when the seed batch finishes.

A failed arm may be retried only under the same source commit, same exact
configuration, and a clearly distinct retry run identity. The failed and retry
identities must not be conflated.

## 8. Required run provenance

Every authorized run must record at minimum:

```text
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
```

Admission requirements:

```text
source_commit = d62c375e2d730f040717422d3951199b71dc688e
source_worktree_clean = true
split_seed = 8192
reason_router_mode = explicit_product
gradient_ownership_mode = edge_specific
reason_loss_weight = 0
freeze_encoder = true
frame_downstream_gradient_mode = joint
```

and the ten-entry edge map must exactly match the arm identity.

Any provenance mismatch blocks scientific admission.

## 9. Required per-run artifacts

Each completed run must retain the same non-checkpoint evidence classes used by
Gen3 first-pass analysis:

```text
clean_dev_predictions.json
run_provenance.json
training_report.json
training_report_predictions.jsonl
```

A selected checkpoint may be created by the frozen training workflow if that is
part of the existing run contract, but checkpoint bytes are not part of the
intended scientific evidence freeze scope unless a later authority explicitly
says otherwise.

No checkpoint may be used to replace a missing prediction/provenance artifact.

## 10. Collection and import boundary

After an authorized seed batch completes:

1. each arm must be saved/registered with its independent run identity;
2. collection must preserve exact source commit, run identity, arm identity,
   seed, and output namespace;
3. the handoff ZIP must be imported through the normal `cm import` path;
4. provenance/hash/commit mismatch is a blocker.

Do not manually copy selected scientific outputs into a different namespace to
bypass collector checks.

Do not reuse a run collected from another commit.

## 11. Scientific analysis requirements after import

Execution success alone is not a scientific conclusion.

After all admitted pairwise evidence is imported and validated, the analysis
must compute at minimum, per pair and seed:

1. accuracy;
2. macro-F1;
3. NOT_ENTITLED / REFUTE / SUPPORT F1;
4. exact A0-correct break set;
5. exact A0-wrong repair set;
6. overlap with the matched D1 A0-correct break set;
7. overlap with the frozen D1 breaks absent from every single-edge arm;
8. union of the two constituent single-edge break sets;
9. pair-specific D1-direction breaks beyond that constituent union;
10. cross-seed recurrence of those pair-specific rows.

No pairwise conclusion may be based only on an aggregate mean, one seed, one
class, or one stable ID.

## 12. Frozen interpretation categories

The post-import analysis may assign the following descriptive categories.

### `PAIRWISE_NONADDITIVE_SIGNAL`

A pair reproducibly creates D1-direction breakage beyond the union of its two
constituent single-edge break sets, with consistent task/class degradation
direction across seeds.

This supports a controlled pairwise non-additive signal under the frozen
intervention. It does not establish a unique parameter-level causal mechanism.

### `PAIRWISE_CUMULATIVE_ONLY`

The pair is more harmful than either constituent single edge, but its break set
is largely explained by the union of constituent single-edge breaks.

### `PAIRWISE_NULL_OR_TOLERANT`

The pair adds little interpretable degradation relative to its constituents.

### `PAIRWISE_ANTAGONISTIC_OR_REPAIRING`

The pair reduces damage or repairs rows relative to one or both constituents.

The analysis must not relabel antagonistic behavior as "optimal ownership."

## 13. Explicit exclusions

This authority does not permit:

- any pair outside the 12 frozen arms;
- any λ other than `.5`;
- lambda sweeps;
- three-edge or higher-order attenuation;
- adaptive or learned edge lambdas;
- pair selection changes after observing results;
- reranking candidates during execution;
- conditional-first-blocker experiments;
- reason-loss restoration;
- architecture changes;
- model or trainer code changes;
- data or split changes;
- Gen4 execution;
- K-series execution;
- native-state mechanism claims;
- production-readiness claims.

If the frozen implementation cannot execute the matrix as specified, stop and
return to implementation authority rather than patching code during execution.

## 14. Run naming

Use one stable run namespace per arm and seed.

Canonical run-name form:

```text
gen3-pairwise-s<seed>-g<a>-g<b>-half
```

Examples:

```text
gen3-pairwise-s180-g4-g5-half
gen3-pairwise-s181-g8-g10-half
gen3-pairwise-s182-g6-g10-half
```

Retries, if needed, must append an explicit retry suffix and must never replace
the original run identity silently.

## 15. Execution success vs evidence validity

Keep these decisions separate:

```text
CODE_CORRECTNESS
EXECUTION_SUCCESS
ARTIFACT_AND_PROVENANCE_VALIDITY
SCIENTIFIC_INTERPRETATION
```

A completed Kaggle command establishes only execution success.

A run enters the scientific matrix only after local import and provenance
validation.

No scientific interpretation should be frozen until all intended admitted runs
are present or a separate authority explicitly closes the matrix with missing
runs.

## 16. Stop conditions

Stop execution and report a blocker if:

- HEAD differs from the exact implementation commit;
- local/Kaggle source worktree is dirty;
- any pair arm/map mismatch appears;
- a run uses a different split seed;
- a run uses a different training seed than its run identity;
- reason loss is nonzero;
- encoder is not frozen;
- ownership mode is not `edge_specific`;
- any non-authorized pair is requested;
- provenance is incomplete or inconsistent;
- collector/import reports commit/hash mismatch;
- code modification appears necessary;
- a lambda sweep or pair-selection change appears necessary.

## 17. Decision

```text
AUTHORIZED_AFTER_EXACT_FREEZE
    = 12 PAIRWISE ARMS
    × SEEDS 180, 181, 182
    = 36 RUNS

SOURCE_COMMIT
    = d62c375e2d730f040717422d3951199b71dc688e

HISTORICAL_REFERENCE_RERUN
    = NO

PAIRWISE_EXECUTION
    = YES, ONLY AFTER THIS AUTHORITY IS FROZEN

KAGGLE
    = AUTHORIZED ONLY FOR THE FROZEN 36-RUN MATRIX

IMPLEMENTATION_CHANGE
    = NO

SCIENTIFIC_CONCLUSION
    = DEFERRED UNTIL VALIDATED IMPORT AND ANALYSIS
```
