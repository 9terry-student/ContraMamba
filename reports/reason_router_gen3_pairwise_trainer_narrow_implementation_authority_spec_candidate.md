# Generation-3 pairwise trainer narrow implementation authority

```text
STATUS = CANDIDATE_ONLY_AT_AUTHORING
PHASE = GEN3_PAIRWISE_TRAINER_NARROW_IMPLEMENTATION_AUTHORITY
AUTHORITY_INPUT = 3e1ce61af8b17b03e6f8c387606b182ae1490117
PAIRWISE_SELECTION_INPUT = c50816540c2a4556fd45a56326fa9609616fa3a9
GEN3_FIRST_PASS_EVIDENCE = 28d7fa2fcc0d286a0cd16723c302a45f214b2901
FROZEN_GEN3_IMPLEMENTATION = 7e3b50b411a2e6556d0b81435cfe3b1a46236339
TRAINING_EVALUATION_ALLOWED = NO
KAGGLE_ALLOWED = NO
IMPLEMENTATION_ALLOWED_AFTER_EXACT_FREEZE = YES
IMPLEMENTATION_COMMIT_PUSH_BEFORE_INDEPENDENT_VERIFICATION = NO
SCIENTIFIC_PAIRWISE_EXECUTION = NOT_AUTHORIZED
```

## 1. Purpose

The frozen Gen3 first-pass result supports distributed ownership dependence
with an unresolved cumulative/non-additive component. The pairwise-selection
audit freezes exactly 12 downstream topology-driven pair candidates. The
subsequent read-only capability verification establishes:

- the model-level `edge_specific` operator already accepts arbitrary canonical
  ten-edge maps, including two `.5` values and eight `1.0` values;
- the training CLI/arm contract intentionally accepts only the ten single-edge
  Gen3 arms and rejects a two-edge map;
- provenance machinery already carries the resolved full edge map, but there
  is no valid pairwise arm identity.

Therefore the only authorized implementation objective is to add the missing
pairwise trainer/contract namespace and tests. No model refactor is justified.

## 2. Activation rule

At authoring time this file is `CANDIDATE_ONLY`.

If and only if these exact report bytes are:

1. SHA256-verified locally;
2. staged as the only intended tracked delta;
3. committed;
4. pushed to the active branch; and
5. remotely authenticated at the resulting commit,

then that frozen commit becomes the implementation authority for the exact
delta below.

No separate activation commit is required.

This authority never authorizes scientific pairwise execution. It authorizes
only the bounded implementation and verification work defined here.

## 3. Exact implementation scope

Only these existing files may be modified:

```text
scripts/train_controlled_v6b_minimal.py
tests/test_reason_router_p2_contract.py
```

No other tracked file may change during implementation.

In particular, modification is forbidden to:

```text
src/contramamba/modeling_v6b_minimal.py
src/contramamba/heads/*
cm.ps1
any dataset or sidecar
any existing report
any imported scientific evidence
any checkpoint
any Kaggle tooling
```

Existing unrelated untracked files must not be deleted, cleaned, staged, or
rewritten.

## 4. Exact authorized pairwise arm identities

The trainer may add exactly these 12 pairwise arm identities and no others:

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

Their edge identities are frozen as:

```text
G4  = F_TO_Q
G5  = P_TO_Q
G6  = S_TO_Q
G7  = F_TO_D
G8  = P_TO_D
G9  = S_TO_D
G10 = Q_TO_D
```

For every pairwise arm:

- `reason_router_mode = explicit_product`
- `gradient_ownership_mode = edge_specific`
- the two arm-named edges must equal exactly `0.5`
- all remaining eight canonical edges must equal exactly `1.0`
- `gradient_ownership_lambda` must remain absent/forbidden
- the full canonical ten-entry map must be resolved and serialized

No pair arm may infer its configuration from numerical performance.

## 5. Trainer delta requirements

The implementation in `scripts/train_controlled_v6b_minimal.py` must be
minimal and reuse existing Gen3 machinery.

Permitted changes are limited to the following functional requirements.

### 5.1 Pairwise arm namespace

Define a frozen pairwise arm collection containing exactly the 12 IDs above.

The existing single-edge `G3_ARM_IDS` semantics must remain unchanged. The
pairwise namespace may be separate; it must not silently broaden the meaning of
existing single-edge IDs.

### 5.2 Arm contract registration

Each pairwise arm must resolve to:

```text
router = explicit_product
ownership = edge_specific
```

Parser choices must accept the 12 pairwise IDs.

No existing A0/A1/A2/A3/D1/Gen3-single-edge parser choice may be removed,
renamed, or reinterpreted.

### 5.3 Exact pairwise map validation

Add or extend validation so that a pairwise arm succeeds only when the supplied
canonical map is exactly the map implied by its arm identity.

The validator must reject at least:

- a one-half map;
- the wrong two-edge pair;
- a three-half map;
- a pair value other than exactly `.5`;
- any non-pair edge value other than exactly `1.0`;
- missing canonical keys;
- extra keys;
- duplicate JSON keys;
- NaN or infinity;
- any value outside `[0,1]`;
- simultaneous global `gradient_ownership_lambda`;
- legacy/non-edge-specific fallback.

Fail-closed behavior is required.

### 5.4 Provenance reuse

The implementation must reuse the existing resolved configuration and
metadata/provenance path.

The resulting pairwise run identity must preserve both:

```text
reason_router_arm = exact pairwise arm ID
edge_gradient_lambdas = exact canonical ten-entry map
```

No new provenance schema is required unless an existing path demonstrably
cannot serialize those already-supported fields. If such a limitation is
found, implementation must stop rather than expanding scope.

## 6. Test requirements

`tests/test_reason_router_p2_contract.py` may be changed only to cover the
authorized pairwise contract and regressions.

The tests must establish at minimum:

1. all 12 pairwise IDs are accepted;
2. all 12 resolve to their exact intended ten-edge maps;
3. resolved mode is exactly `explicit_product / edge_specific`;
4. checkpoint/config metadata preserves the exact pairwise arm/map identity;
5. wrong pair is rejected;
6. one-half map is rejected;
7. three-half map is rejected;
8. non-.5 pair value is rejected;
9. malformed, duplicate-key, non-finite, missing-key and extra-key maps remain
   fail-closed;
10. a representative pair has unchanged forward outputs relative to the
    corresponding all-ones edge-specific forward;
11. both intended recipient gradient contributions are attenuated to `.5`;
12. at least one non-pair sibling recipient contribution is unchanged;
13. owner-local gradients remain preserved;
14. all existing ten single-edge Gen3 arm contracts remain unchanged;
15. JOINT and GLOBAL-HALF endpoint-equivalence tests remain unchanged;
16. A0/A1/A2/A3/D1 legacy behavior remains unchanged.

The implementation must not weaken or delete an existing Gen3 test merely to
make the pairwise tests pass.

## 7. Explicitly forbidden changes

The implementation must not:

- alter `_partial_grad`;
- alter the model-level ten-edge topology;
- add new conceptual edges;
- alter any edge recipient occurrence;
- modify model forward routing;
- alter lambda from `.5`;
- add a lambda sweep;
- add pair selection logic;
- select pairs dynamically;
- add three-edge or higher-order arms;
- add adaptive/learned ownership;
- alter loss terms or loss weights;
- alter dataset/split/seed behavior;
- alter encoder freezing;
- alter reason semantics;
- alter D1 or Gen3-single-edge behavior;
- load or inspect scientific checkpoints;
- run training or evaluation;
- start Kaggle;
- touch K-series work;
- clean or reset unrelated local artifacts.

## 8. Implementation validation

The implementer must run only non-scientific code/contract validation.

Minimum validation:

```text
python -m pytest tests/test_reason_router_p2_contract.py -q
```

If the environment requires a different Python launcher, the equivalent
interpreter may be used without changing test scope.

The implementer must also report:

- current HEAD before changes;
- exact modified files;
- exact test count and result;
- `git diff --check`;
- `git status --short`;
- confirmation that no training/evaluation/inference/checkpoint loading ran.

If the full contract test file triggers scientific training/evaluation or
checkpoint loading unexpectedly, stop and report the blocker instead of
running it.

## 9. Independent verification requirement

This is a high-risk scientific configuration/provenance change.

After implementation, an independent verifier must inspect the actual diff and
verify:

- exact two-file scope;
- all 12 arm identities;
- exact arm-to-map binding;
- fail-closed negative cases;
- model source unchanged;
- single-edge Gen3 semantics unchanged;
- legacy JOINT/explicit-local/D1 semantics unchanged;
- provenance identity;
- forward identity;
- two intended attenuations and third-edge preservation;
- owner-local gradient preservation;
- tests passing;
- no scientific execution.

The verifier must return one of:

```text
PASS_READY_FOR_MANUAL_FREEZE
BLOCKED
```

Implementation commit/push is not authorized before
`PASS_READY_FOR_MANUAL_FREEZE`.

## 10. Commit/push boundary

For the implementation phase:

```text
Commit/Push = NO
```

The implementer or verifier must not stage, commit, or push implementation
changes.

After independent verification PASS, the research controller may direct the
user to run `cm ship`, review the exact two-file staged scope, and manually
commit/push.

Even after implementation is frozen, pairwise scientific execution remains
unauthorized until a separate execution-authority report is frozen.

## 11. Stop conditions

Stop implementation and report a blocker if any of the following occurs:

- model source appears necessary to change;
- a third tracked implementation file appears necessary;
- provenance requires a new schema rather than existing field reuse;
- existing single-edge behavior must be weakened or redefined;
- an existing legacy test must be deleted or relaxed;
- pairwise configuration cannot be represented by the current model operator;
- scientific training/evaluation would be needed to validate correctness;
- repository HEAD or authority lineage does not match the frozen authority.

## 12. Required implementation report

The implementer must return:

```text
Overall:
HEAD:
Authority:
Modified files:
Behavior added:
Model source changed: YES/NO
Pairwise arms implemented:
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
```

## 13. Decision

```text
AUTHORIZED_DELTA_AFTER_FREEZE
    = TRAINER_PAIRWISE_ARM_NAMESPACE_AND_EXACT_MAP_CONTRACT
    + CONTRACT_TESTS

AUTHORIZED_FILES
    = scripts/train_controlled_v6b_minimal.py
    + tests/test_reason_router_p2_contract.py

MODEL_CHANGE
    = FORBIDDEN

PAIRWISE_SCIENTIFIC_EXECUTION
    = NOT_AUTHORIZED

INDEPENDENT_VERIFICATION
    = REQUIRED

IMPLEMENTATION_COMMIT_PUSH
    = FORBIDDEN_UNTIL_VERIFIER_PASS
```
