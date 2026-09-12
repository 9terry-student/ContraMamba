# Generation-3 grouped-factorial trainer narrow implementation authority

STATUS = CANDIDATE_ONLY_AT_AUTHORING

PHASE = GEN3_GROUPED_FACTORIAL_TRAINER_NARROW_IMPLEMENTATION_AUTHORITY

AUTHORITY_INPUT = 65fdee2d5c4e60f8eadeb3d39ab4e601156d8bb4

GROUPED_FACTORIAL_STATIC_DESIGN = 65fdee2d5c4e60f8eadeb3d39ab4e601156d8bb4

FROZEN51_EVIDENCE_AUTHORITY = 792b7a7e9bdb7fa957fa3ecda6ee9c6e37ce0f96

FROZEN_SCIENTIFIC_IMPLEMENTATION = d62c375e2d730f040717422d3951199b71dc688e

TRAINING_EVALUATION_ALLOWED = NO

KAGGLE_ALLOWED = NO

IMPLEMENTATION_ALLOWED_AFTER_EXACT_FREEZE = YES

IMPLEMENTATION_COMMIT_PUSH_BEFORE_INDEPENDENT_VERIFICATION = NO

SCIENTIFIC_GROUPED_EXECUTION = NOT_AUTHORIZED

## 1. Purpose

The frozen grouped-factorial static design defines a complete topology-
prespecified 2-by-2-by-2 decomposition over three disjoint macro-groups:

U = G1, G2, G3

Q = G4, G5, G6

D = G7, G8, G9, G10

The existing A0 and D1 / GLOBAL-HALF endpoints are historical frozen evidence.

The six missing proper-subset scientific conditions are:

U

Q

D

U+Q

U+D

Q+D

The current frozen trainer does not admit these grouped maps.

Its accepted Gen3 namespace contains the ten frozen single-edge arms and the
twelve frozen pairwise arms.

The existing fail-closed arm-map validator rejects ownership maps outside those
exact arm identities.

The model-level edge-specific operator and the ten-edge topology must remain
unchanged.

Therefore the only authorized implementation objective is to extend the
trainer/config contract with exactly six grouped arm identities and the
associated exact ten-edge maps, plus narrow contract tests.

No model refactor is authorized.

## 2. Activation rule

At authoring time this file is CANDIDATE_ONLY.

If and only if these exact authority bytes are:

1. SHA256-verified locally;
2. independently statically verified;
3. staged as the only intended tracked delta;
4. committed;
5. pushed to the active branch; and
6. remotely authenticated at the resulting commit,

then that frozen commit becomes the implementation authority for the exact
delta below.

No separate activation commit is required.

This authority never authorizes grouped scientific execution.

It authorizes only the bounded implementation and verification work defined
here.

## 3. Exact implementation scope

Only these two existing files may be modified:

scripts/train_controlled_v6b_minimal.py

tests/test_reason_router_p2_contract.py

No other tracked file may change during implementation.

Modification is explicitly forbidden to:

src/contramamba/modeling_v6b_minimal.py

src/contramamba/heads/*

cm.ps1

any dataset or sidecar

any existing report

any imported scientific evidence

any checkpoint

any Kaggle tooling

Existing unrelated untracked files must not be deleted, cleaned, rewritten, or
staged.

## 4. Canonical ten-edge topology

The existing Gen3 edge identities remain exactly:

G1 = F_TO_P

G2 = F_TO_S

G3 = P_TO_S

G4 = F_TO_Q

G5 = P_TO_Q

G6 = S_TO_Q

G7 = F_TO_D

G8 = P_TO_D

G9 = S_TO_D

G10 = Q_TO_D

No edge may be added, removed, renamed, redirected, or reinterpreted.

## 5. Exact authorized grouped arm identities

The trainer may add exactly these six new arm identities and no other grouped
or higher-order identities:

G3-GROUP-U-HALF

G3-GROUP-Q-HALF

G3-GROUP-D-HALF

G3-GROUP-U-Q-HALF

G3-GROUP-U-D-HALF

G3-GROUP-Q-D-HALF

Existing single-edge and pairwise arm IDs must remain unchanged.

The new grouped namespace must not broaden the semantic meaning of any existing
arm ID.

## 6. Exact grouped maps

Every grouped arm must use:

reason_router_mode = explicit_product

gradient_ownership_mode = edge_specific

Every selected macro-group edge must equal exactly 0.5.

Every non-selected edge must equal exactly 1.0.

The full canonical ten-entry edge map must be resolved and serialized.

### 6.1 G3-GROUP-U-HALF

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

### 6.2 G3-GROUP-Q-HALF

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

### 6.3 G3-GROUP-D-HALF

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

### 6.4 G3-GROUP-U-Q-HALF

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

### 6.5 G3-GROUP-U-D-HALF

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

### 6.6 G3-GROUP-Q-D-HALF

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

No grouped arm may infer its map from performance, stable IDs, class outcomes,
or seed-specific evidence.

## 7. Historical endpoints are not new grouped arm implementations

The implementation must not introduce new replacements for:

NONE / A0

U+Q+D / D1 / GLOBAL-HALF

Those endpoints remain historical frozen controls.

The six new arm IDs represent only the six missing proper subsets.

The implementation must not retrain, rename, alias, or reinterpret A0 or D1.

## 8. Trainer delta requirements

Changes to scripts/train_controlled_v6b_minimal.py must be minimal and reuse
the existing Gen3 edge-specific machinery.

### 8.1 Grouped arm namespace

Define a frozen grouped-arm collection containing exactly the six IDs in
Section 5.

The existing ten single-edge IDs and twelve pairwise IDs must remain
independent namespaces or otherwise preserve their exact frozen semantics.

No wildcard or arbitrary multi-edge arm mechanism is authorized.

### 8.2 Arm contract registration

Each grouped arm must resolve to:

router = explicit_product

ownership = edge_specific

Parser choices may be extended only enough to accept the six grouped IDs.

No existing A0, A1, A2, A3, D1, single-edge, or pairwise parser choice may be
removed, renamed, or reinterpreted.

### 8.3 Exact map validation

A grouped arm succeeds only when the supplied canonical ten-edge map is
exactly the map frozen in Section 6 for that arm.

Fail-closed validation must reject at minimum:

- the wrong grouped map;
- any selected edge value other than exactly 0.5;
- any non-selected edge value other than exactly 1.0;
- a one-edge map supplied to a grouped arm;
- a two-edge pairwise map supplied to a grouped arm;
- a wrong three-edge map;
- a wrong four-edge map;
- a wrong six-edge map;
- a wrong seven-edge map;
- a ten-half map supplied to a proper-subset grouped arm;
- missing canonical keys;
- extra keys;
- duplicate JSON keys;
- NaN;
- infinity;
- any value outside [0,1];
- simultaneous global gradient_ownership_lambda;
- legacy or non-edge-specific fallback.

Validation must remain exact-arm-identity based.

The implementation must not generalize the CLI into an unrestricted arbitrary
multi-edge map interface.

### 8.4 Provenance reuse

The existing resolved configuration and provenance path must be reused.

Every grouped condition must preserve:

reason_router_arm = exact grouped arm ID

edge_gradient_lambdas = exact canonical ten-entry map

No new provenance schema is authorized unless the existing fields demonstrably
cannot serialize the already-defined grouped identity and map.

If a new schema appears necessary, implementation must stop and report the
blocker rather than expanding scope.

## 9. Model and gradient semantics are frozen

The implementation must not alter:

_partial_grad

the model-level ten-edge topology

any edge recipient location

forward routing

gradient-ownership operator semantics

lambda semantics

reason-router mathematics

loss functions

loss weights

dataset behavior

split behavior

training seed behavior

encoder freezing

final three-way class semantics

The grouped implementation is a trainer/config namespace extension only.

## 10. Required tests

tests/test_reason_router_p2_contract.py may be changed only to cover the
authorized grouped contract and regressions.

The minimum test suite must establish all of the following.

### 10.1 Namespace and exact map contract

1. all six grouped arm IDs are accepted;
2. all six resolve to their exact intended canonical ten-edge maps;
3. all six resolve to explicit_product / edge_specific;
4. no seventh grouped identity, wildcard grouped identity, or arbitrary
   multi-edge identity is accepted;
5. every grouped arm rejects an identity/map mismatch;
6. selected edge values other than exactly 0.5 are rejected;
7. non-selected edge values other than exactly 1.0 are rejected;
8. one-edge and two-edge maps are rejected for grouped identities;
9. wrong 3-, 4-, 6-, and 7-edge maps are rejected;
10. an all-ten-half map is rejected for every proper-subset grouped arm;
11. malformed, duplicate-key, non-finite, out-of-range, missing-key, and
    extra-key maps remain fail-closed;
12. simultaneous global gradient_ownership_lambda remains rejected.

### 10.2 All-six provenance and metadata contract

For each of the six exact grouped arm identities separately, tests must assert:

13. reason_router_arm equals the exact grouped arm identity;
14. edge_gradient_lambdas contains the exact canonical ten-entry map for that
    identity;
15. the resolved configuration / checkpoint-metadata construction path preserves
    the exact arm/map pair without aliasing, normalization to another arm, or
    omission of any edge;
16. no new provenance schema is required.

These assertions must cover all six exact arm/map pairs, not one generic or
representative grouped condition.

Scientific checkpoint loading is not required or authorized by these tests.

### 10.3 All-six forward invariance

For each of the six grouped maps separately:

17. forward outputs must be identical to the corresponding all-ones
    edge-specific forward under the same model parameters and inputs;
18. final predictions must remain identical;
19. all relevant returned forward diagnostics used by existing edge-specific
    contract tests must remain identical within the existing numerical
    tolerances.

Forward invariance must therefore be tested for:

- G3-GROUP-U-HALF
- G3-GROUP-Q-HALF
- G3-GROUP-D-HALF
- G3-GROUP-U-Q-HALF
- G3-GROUP-U-D-HALF
- G3-GROUP-Q-D-HALF

Representative-only forward testing is insufficient.

### 10.4 Gradient attenuation and preservation across all six maps

For each of the six grouped maps separately, the tests must establish:

20. every selected recipient-edge contribution tested for that arm is attenuated
    to exactly 0.5 relative to the corresponding all-ones edge-specific
    reference;
21. every non-selected sibling recipient-edge contribution tested for that arm
    remains exactly at 1.0 relative to the same reference;
22. no selected edge silently remains at 1.0;
23. no non-selected edge is silently attenuated.

Across the complete six-arm test set, every canonical edge G1 through G10 must
be exercised at least once in a selected grouped condition and at least once in
a non-selected grouped condition wherever such a non-selected condition exists
within the six proper-subset maps.

The coverage must explicitly include:

- Q-only runtime behavior;
- Q+D runtime behavior;
- at least one 3-edge grouped map;
- the 4-edge D map;
- the 6-edge U+Q map;
- both 7-edge grouped maps U+D and Q+D.

Testing only one of the two seven-edge maps is insufficient.

### 10.5 Owner-local gradient preservation

24. owner-local gradients must remain preserved under grouped attenuation;
25. this preservation must be demonstrated using at least one selected edge
    originating from macro-group U;
26. at least one selected edge originating from macro-group Q;
27. at least one selected edge originating from macro-group D.

These owner-local checks must use the existing model/operator semantics and may
not require model-source modification.

### 10.6 Frozen-regression obligations

28. all existing ten single-edge Gen3 arm contracts remain unchanged;
29. all existing twelve pairwise arm contracts remain unchanged;
30. JOINT endpoint-equivalence behavior remains unchanged;
31. GLOBAL-HALF / D1 endpoint-equivalence behavior remains unchanged;
32. A0 behavior remains unchanged;
33. A1 behavior remains unchanged;
34. A2 behavior remains unchanged;
35. A3 behavior remains unchanged;
36. D1 behavior remains unchanged.

Existing tests must not be weakened, deleted, skipped, xfailed, or relaxed to
make grouped tests pass.

The test implementation may factor common helpers to avoid duplicated test
code, but such factoring must not reduce the obligation to exercise and assert
runtime, gradient, and provenance semantics for all six grouped identities.

## 11. Explicitly forbidden implementation changes

The implementation must not:

- modify model source;
- alter _partial_grad;
- alter the ten-edge topology;
- add conceptual edges;
- change recipient locations;
- add arbitrary user-defined grouped maps;
- add arbitrary N-edge arm construction;
- add wildcard arm parsing;
- add lambda sweeps;
- add adaptive or learned ownership;
- add seed-dependent maps;
- add row-dependent maps;
- select groups dynamically;
- use FROZEN51 stable IDs during training;
- use validation outcomes to choose maps;
- introduce U+Q+D as a new retrained grouped arm;
- replace historical A0 or D1;
- alter loss semantics;
- alter datasets or sidecars;
- alter seeds or splits;
- alter encoder freezing;
- alter existing single-edge behavior;
- alter existing pairwise behavior;
- load scientific checkpoints;
- run scientific training;
- run scientific evaluation;
- start Kaggle;
- touch K-series work;
- clean/reset unrelated local artifacts.

## 12. Implementation validation

Only non-scientific code/contract validation is authorized.

Minimum required validation:

python -m pytest tests/test_reason_router_p2_contract.py -q

The implementer must also report:

- HEAD before changes;
- exact authority commit;
- exact modified files;
- exact test count and result;
- git diff --check;
- git status --short;
- confirmation that model source is unchanged;
- confirmation that no training/evaluation/inference/checkpoint loading ran.

If the contract test unexpectedly requires scientific training, evaluation, or
checkpoint loading, stop and report the blocker.

## 13. Independent verification requirement

This change is high-risk because it controls scientific gradient configuration
and provenance.

After implementation, an independent verifier must inspect the exact diff and
verify:

- exact two-file scope;
- exactly six new grouped IDs;
- exact six arm-to-map bindings;
- absence of unrestricted arbitrary N-edge interfaces;
- fail-closed negative cases;
- trainer-only implementation delta;
- model source unchanged;
- existing single-edge semantics unchanged;
- existing pairwise semantics unchanged;
- A0/A1/A2/A3/D1 unchanged;
- JOINT and GLOBAL-HALF behavior unchanged;
- exact grouped provenance identity;
- forward identity;
- exact selected-edge attenuation;
- exact non-selected-edge preservation;
- owner-local gradient preservation;
- tests passing;
- no scientific execution.

The verifier must return one of:

PASS_READY_FOR_MANUAL_FREEZE

BLOCKED

Implementation commit/push is forbidden before verifier PASS.

## 14. Commit/push boundary

For the implementation phase:

Commit/Push = NO

The implementer and verifier must not stage, commit, or push implementation
changes.

After independent verification PASS, the research controller may direct the
user to run cm ship and manually freeze the exact implementation delta.

Even after implementation is frozen, grouped scientific execution remains
unauthorized until a separate grouped execution authority is frozen.

## 15. Stop conditions

Stop implementation and report a blocker if:

- model source appears necessary to change;
- any third tracked implementation file appears necessary;
- provenance requires a new schema;
- unrestricted arbitrary multi-edge parsing appears necessary;
- existing single-edge or pairwise behavior must be weakened;
- an existing legacy test must be deleted or relaxed;
- grouped configuration cannot be represented with the existing edge-specific
  model operator;
- scientific training/evaluation would be required for validation;
- repository HEAD does not match the frozen implementation authority;
- authority lineage cannot be reconciled.

## 16. Required implementation report

The implementer must return:

Overall:
HEAD:
Authority:
Modified files:
Grouped arms implemented:
Exact grouped maps:
Model source changed: YES/NO
Arbitrary N-edge interface added: YES/NO
Legacy single-edge behavior changed: YES/NO
Legacy pairwise behavior changed: YES/NO
Negative validation:
Provenance behavior:
Validation commands:
Validation results:
git diff --check:
git status --short:
Training/Evaluation performed: YES/NO
Checkpoint loading performed: YES/NO
Commit/Push performed: YES/NO
Blockers:

## 17. Decision

AUTHORIZED_DELTA_AFTER_FREEZE
    = TRAINER_GROUPED_ARM_NAMESPACE_AND_EXACT_MAP_CONTRACT
    + GROUPED_CONTRACT_TESTS

AUTHORIZED_NEW_ARM_COUNT
    = 6

AUTHORIZED_FILES
    = scripts/train_controlled_v6b_minimal.py
    + tests/test_reason_router_p2_contract.py

MODEL_CHANGE
    = FORBIDDEN

ARBITRARY_MULTI_EDGE_INTERFACE
    = FORBIDDEN

GROUPED_SCIENTIFIC_EXECUTION
    = NOT_AUTHORIZED

INDEPENDENT_IMPLEMENTATION_VERIFICATION
    = REQUIRED

IMPLEMENTATION_COMMIT_PUSH
    = FORBIDDEN_UNTIL_VERIFIER_PASS

PASS_READY_FOR_INDEPENDENT_GROUPED_IMPLEMENTATION_AUTHORITY_VERIFICATION