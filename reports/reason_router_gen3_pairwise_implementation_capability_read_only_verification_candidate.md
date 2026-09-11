# Generation-3 pairwise implementation capability read-only verification

```text
VERDICT = BLOCKED_REQUIRES_NARROW_TRAINER_DELTA
PHASE = GEN3_PAIRWISE_IMPLEMENTATION_CAPABILITY_READ_ONLY_VERIFICATION
AUTHORITY = c50816540c2a4556fd45a56326fa9609616fa3a9
FROZEN_GEN3_IMPLEMENTATION = 7e3b50b411a2e6556d0b81435cfe3b1a46236339
MODEL_EDGE_MAP_CAPABILITY = PASS_STATIC
TRAINER_PAIRWISE_ARM_CAPABILITY = FAIL_CLOSED
PAIRWISE_PROVENANCE_ARM_ID_CAPABILITY = NOT_PRESENT
IMPLEMENTATION_CHANGE = NOT_PERFORMED
TRAINING_EVALUATION_INFERENCE_CHECKPOINT_LOADING = NOT_PERFORMED
SCIENTIFIC_EXECUTION = NOT_AUTHORIZED
```

## 1. Scope and authority

This is a read-only static verification required by the frozen Generation-3
pairwise interaction selection audit at
`c50816540c2a4556fd45a56326fa9609616fa3a9`.

The question is narrowly:

> Does the currently frozen Gen3 implementation already support an exact
> pairwise arm with two named edge lambdas at `.5`, the remaining eight at
> `1.0`, and an unambiguous pairwise arm/provenance identity?

No file was modified. No test was run by this report, and no training,
evaluation, inference, checkpoint loading, Kaggle work, staging, commit, or
push was performed.

The remote comparison from frozen Gen3 implementation
`7e3b50b411a2e6556d0b81435cfe3b1a46236339` through current authority
`c50816540c2a4556fd45a56326fa9609616fa3a9` shows only Gen3 evidence/report
additions and the pairwise-selection report; no source or test file changed.
Therefore the implementation capability examined here is exactly the frozen
Gen3 implementation lineage.

## 2. Model-level edge-map capability

`src/contramamba/modeling_v6b_minimal.py` defines the ten canonical edge keys:

- `F_TO_P`
- `F_TO_S`
- `P_TO_S`
- `F_TO_Q`
- `P_TO_Q`
- `S_TO_Q`
- `F_TO_D`
- `P_TO_D`
- `S_TO_D`
- `Q_TO_D`

`_validate_edge_gradient_lambdas(...)` requires exactly those ten keys and
validates each value independently as finite and within `[0,1]`. It does not
enforce a one-half-only cardinality constraint.

In `edge_specific` mode, recipient-specific aliases use the map entry for the
named edge:

```text
partial_grad(z, lambda_e) = z.detach() + lambda_e * (z - z.detach())
```

The forward path separately addresses all ten conceptual direct ingresses.
Thus a Python-level edge map containing exactly two `.5` entries and eight
`1.0` entries is representable by the model and will be consumed edge by edge.

The existing Gen3 contract tests also establish:

- edge-specific forward identity against JOINT for valid edge maps;
- all-ones edge-specific endpoint equivalence to JOINT;
- all-.5 endpoint equivalence to legacy GLOBAL-HALF/partial;
- actual single-edge recipient isolation for each of the ten edges;
- sibling-path preservation;
- recipient-alias independence;
- owner-local-gradient preservation.

### Model-level verdict

```text
MODEL_ACCEPTS_TWO_HALF_VALUES = YES
MODEL_FORWARD_IDENTITY_MECHANISM = GENERIC
MODEL_RECIPIENT_ALIASING = PER_EDGE
MODEL_REFACTOR_REQUIRED_FOR_PAIRWISE = NO
```

This is a static capability finding. It does not itself authorize pairwise
execution.

## 3. Trainer/CLI arm-contract blocker

The training script is intentionally stricter than the model.

The frozen Gen3 implementation defines only:

```text
G3_ARM_IDS = G3-G1-HALF ... G3-G10-HALF
```

and extends the reason-router arm choices only with those ten single-edge
identities.

For an `edge_specific` arm,
`_p2_resolve_arm_contract(...)` resolves the full ten-edge map and then calls
the Gen3 arm-map validator.

That validator constructs an expected map with:

- exactly one `.5`, selected from the numeric `G3-Gk-HALF` arm ID; and
- the other nine values exactly `1.0`.

Any other map is rejected with:

```text
G3_ARM_EDGE_MAP_MISMATCH
```

The existing tests explicitly enforce this fail-closed behavior by passing a
map that does not match the single-edge arm identity and requiring the
`G3_ARM_EDGE_MAP_MISMATCH` failure.

There is currently no pairwise arm ID among parser choices. `arm=none` cannot
be used as an escape hatch because supplying edge-specific configuration to a
non-Gen3 arm is also fail-closed.

Therefore the current training command cannot legitimately express any of the
12 frozen pairwise candidates.

### Trainer verdict

```text
PAIRWISE_CLI_ARM_ID = ABSENT
PAIRWISE_ARM_TO_MAP_VALIDATION = ABSENT
CURRENT_SINGLE_EDGE_VALIDATOR_REJECTS_PAIRWISE_MAP = YES
BYPASS_USING_NONE_OR_LEGACY_ARM = FORBIDDEN
END_TO_END_PAIRWISE_EXECUTION_READY = NO
```

This is the decisive blocker.

## 4. Provenance implication

The existing trainer provenance/checkpoint path correctly serializes the
resolved full `edge_gradient_lambdas` map for Gen3 single-edge arms.

However, because there is no authorized pairwise arm identity and the
single-edge arm validator rejects a two-half map, current provenance cannot
produce an unambiguous valid pairwise record of the form:

```text
reason_router_arm = <frozen pairwise arm id>
gradient_ownership_mode = edge_specific
edge_gradient_lambdas = <exact two-half/eight-one map>
```

The map serialization machinery can be reused. The missing capability is the
pairwise arm namespace and exact arm-to-map contract, not a new provenance
format.

## 5. Required implementation delta

A future implementation authority should permit only the narrow missing
trainer/test capability.

### `scripts/train_controlled_v6b_minimal.py`

Permit the exact 12 frozen pairwise identities from the selection audit:

- `G3-G4-G5-HALF`
- `G3-G4-G6-HALF`
- `G3-G5-G6-HALF`
- `G3-G7-G8-HALF`
- `G3-G7-G9-HALF`
- `G3-G7-G10-HALF`
- `G3-G8-G9-HALF`
- `G3-G8-G10-HALF`
- `G3-G9-G10-HALF`
- `G3-G4-G10-HALF`
- `G3-G5-G10-HALF`
- `G3-G6-G10-HALF`

For those identities only:

- resolve `explicit_product / edge_specific`;
- require exactly the canonical ten edge keys;
- require the two identity-named edges to equal `.5`;
- require all other eight edges to equal `1.0`;
- reject any mismatch, extra attenuation, missing attenuation, global lambda,
  malformed/duplicate key, or legacy-mode fallback;
- serialize the full ten-entry map and exact pairwise arm ID through the
  existing metadata/provenance paths.

### `tests/test_reason_router_p2_contract.py`

Add narrow tests that prove:

1. all 12 frozen pairwise arm IDs resolve to the exact expected maps;
2. a pairwise arm rejects a wrong pair, one-half map, three-half map, and
   non-.5 pair value;
3. pairwise resolved configuration/checkpoint metadata contains the exact
   arm/map;
4. at model level, at least one representative pair demonstrates unchanged
   forward outputs, both intended recipient-gradient contributions attenuated,
   and a non-pair sibling unchanged;
5. no legacy JOINT, explicit-local, D1 partial, or Gen3 single-edge contract
   changes.

No model source change is justified by this verification unless implementation
evidence later proves the existing model genericity insufficient.

## 6. Explicit non-delta

The verification does not justify changes to:

- `src/contramamba/modeling_v6b_minimal.py` by default;
- edge topology;
- lambda value;
- loss composition;
- data or split semantics;
- frozen encoder policy;
- A0/D1/Gen3 single-edge semantics;
- checkpoint format;
- `cm.ps1`;
- Kaggle tooling;
- analysis categories;
- K-series.

No refactor is authorized merely to make pairwise naming more elegant.

## 7. Verification requirements after the narrow delta

Because the change touches scientific configuration/provenance but does not
change the underlying edge-gradient operator, the implementation must still be
verified before any pairwise execution.

Required verification:

1. exact intended file scope;
2. all 12 arm-to-map contracts;
3. representative two-edge forward identity;
4. exact two-edge gradient attenuation and third-edge preservation;
5. owner-local-gradient preservation;
6. pairwise provenance identity;
7. unchanged single-edge Gen3 tests;
8. unchanged JOINT and GLOBAL-HALF endpoint tests;
9. full relevant contract suite PASS;
10. no training/evaluation/inference/checkpoint loading.

An independent verifier is required because gradient/provenance semantics are
high-risk scientific controls.

## 8. Decision

```text
CURRENT_MODEL_OPERATOR = SUFFICIENT
CURRENT_TRAINER_PAIRWISE_CONTRACT = INSUFFICIENT_BY_DESIGN
PAIRWISE_EXECUTION = BLOCKED
REQUIRED_NEXT_ACTION = AUTHOR_NARROW_PAIRWISE_TRAINER_IMPLEMENTATION_AUTHORITY
MODEL_CHANGE_EXPECTED = NO
TRAINER_CHANGE_EXPECTED = YES
TEST_CHANGE_EXPECTED = YES
SCIENTIFIC_EXECUTION_AFTER_IMPLEMENTATION = STILL_REQUIRES_SEPARATE_GATE
```

The correct next phase is not Kaggle and not pairwise execution. It is a
report-only implementation authority that freezes this narrow trainer/test
delta, followed by implementation and independent verification.
