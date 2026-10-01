# ContraMamba Gen5 Phase 3A
## Causal-Role Contention Stressor Implementation Authority

### Status

PHASE =
`GEN5_PHASE3A_CONTENTION_QUALIFICATION`

AUTHORITY =
`IMPLEMENTATION_ONLY`

TRAINING =
`NOT_AUTHORIZED`

SCIENTIFIC_EXECUTION =
`NOT_AUTHORIZED`

CUDA_SCIENTIFIC_EXECUTION =
`NOT_AUTHORIZED`

PHASE3B =
`NOT_AUTHORIZED`

---

## 1. Scientific parent

Phase 3 static-design lineage:

- Phase 3 design:
  `ae52dbf98cce21226036ae38121edcdfa6f79d7b`
- training-label amendment:
  `14d2bd7db70fc6000b480221a17fa8475dfefbe5`
- stressor-domain amendment:
  `67e97cae7f6fd20072a9422d0ac40bc6f781ed13`
- static preparer:
  `c77f4adb98a949391b44e498488778e722d65eee`
- static preparation freeze:
  `654992ab9e2f9b77fd270ee6bf889c7dcddedd51`

Frozen 17-file static tree SHA256:

`f244b4b614a6632bb3c7a9dd98c3213e357e8212cc445cad85a062e9d01b6053`

Implementation must consume the frozen static artifacts exactly.

---

## 2. Phase 3A purpose

Phase 3A asks only whether the frozen upstream PP3 stressor creates genuine
optimization-time contention for the already authenticated R22 realization.

Phase 3A does not test the C1/M1 ownership hypothesis.

Only unrestricted:

`G5-C0`

is in Phase 3A.

Pressure conditions:

- `P0` = native
- `PR` = PP3 causal-role stress
- `PC` = PP5 matched-control stress

Seeds:

- `6201`
- `6202`
- `6203`

The eventual Phase 3A matrix therefore contains exactly nine cells.

This implementation authority does not authorize those nine training runs.

---

## 3. Reuse of Phase 2 WRITE22 implementation

The existing Phase 2 correction implementation must be reused without
scientific-semantic modification:

`src/contramamba/gen5_phase2_state_update_ownership.py`

The Phase 3 implementation must continue to use:

- target layer 22;
- rank 2;
- A_theta: 768 -> 2;
- B_theta: 2 -> 24576;
- no biases;
- exact zero initialization of B_theta;
- only A_theta/B_theta trainable;
- frozen parent;
- final 3-way CE only;
- exact R22/C22 objects.

R22 SHA256:

`a69232900e8b5a91ec5248e36facee4d421eabd502829e6abf3d2719fdb02214`

C22 SHA256:

`c692d39a7387e32e9bf76fe4db7ce2d30d5f7d22a4389e8af363a80b394155c4`

No Phase 2 implementation file may be edited merely to implement Phase 3A.

---

## 4. Frozen layer-17 stressor objects

Reuse the exact frozen PP3 and PP5 vectors.

PP3 plus SHA256:

`66ad0cd0f931b0aff88bfc8afc4e9ffa9e355d32f054d376acebb97b469a3cff`

PP3 minus SHA256:

`ea2c997de7dbaea4edad8b1f30cdac31b4a0c76db0f0df2983900f28a2529ce7`

PP5 plus SHA256:

`7eb8154a10f647a4b732f7a7b7e34087840a513b88177da06633d8f7b28a4df2`

PP5 minus SHA256:

`311a41c37b9586206ea9bfc9da390688d9a6bb5672a5f63bb589c9d019478855`

The implementation must verify:

- unit norm of each vector;
- PP3 plus/minus orthogonality;
- PP5 plus/minus orthogonality;
- PP3/PP5 cross-orthogonality;
- dimension = 395.

No vector may be refit or recomputed.

---

## 5. Frozen intervention support

The stressor may act only on:

- `C0_SHAM`
- `C1_TITLE`
- `C2_NAME`
- `C5_TITLE_NAME`

It must never act on:

- `C3_ROLE`
- `C4_PREDICATE`
- `C6_EXPLICIT_DENIAL`

The exact per-row coordinates come only from the frozen:

`stressor_target_manifest.jsonl`

Target contract:

- layer = 17
- anchor = `A_IDENTITY`
- offset = `+2`
- target token =
  `absolute_anchor_token_index + 2`

The runtime must not re-discover or choose target tokens.

---

## 6. Strong-channel geometry

The intervention is applied to the exact frozen layer-17 strong-channel
partition used by the PP3 causal lineage.

Expected strong-channel count:

`395`

The implementation must derive the strong mask from the frozen native layer-17
mixer and validate the inherited partition identity before use.

Only the first half of the layer-17 `in_proj` output corresponding to the
hidden/SSM branch may be changed.

The gate half must remain byte-identical.

Non-strong channels must remain byte-identical.

All non-target tokens must remain byte-identical.

---

## 7. PR forward intervention

Let the native strong-channel vector at the frozen target token be:

`h in R^395`

and frozen PP3 unit vectors be:

`p+`
`p-`

Compute native coefficients from the pre-intervention activation:

`a = <h,p+>`

`b = <h,p->`

For PR:

`delta_PR = -a p+ - b p-`

and:

`h_PR = h + delta_PR`

The post-intervention PP3 residual coordinates must satisfy the inherited
numerical tolerance.

The coefficients must be computed from the native pre-intervention activation,
not from a previously modified activation.

---

## 8. PC matched-control intervention

PC uses the same native PP3 coefficients:

`a = <h,p+>`

`b = <h,p->`

but writes into the frozen PP5 plane:

`delta_PC = -a q+ - b q-`

where `q+`, `q-` are the frozen PP5 vectors.

Therefore PR and PC use exactly matched coefficient magnitudes derived from the
same native PP3 coordinates.

No PP5-derived coefficient fitting is allowed.

No response-dependent rescaling is allowed.

---

## 9. Gradient semantics

The layer-17 stressor is a frozen external intervention.

Its coefficients and delta are not trainable.

The parent remains frozen.

The implementation must reproduce the existing causal-intervention forward
semantics: stressor coefficients are calculated from the native activation
without creating trainable stressor parameters or new gradient-owned objects.

Only layer-22 correction A_theta/B_theta may receive optimizer gradients.

The stressor must not introduce:

- trainable PP vectors;
- trainable layer-17 parameters;
- auxiliary losses;
- R22 losses;
- PP3 losses;
- additional gradient ownership.

---

## 10. Batch-aware streamed implementation

Phase 3 training uses:

`3360`

training rows.

Backbone execution remains memory-bounded and streamed.

The Phase 3 implementation must support a batch/chunk containing a mixture of:

- stressor-domain rows;
- non-stressor rows.

For each chunk it must consume frozen per-row metadata:

- source pair identity;
- contrast cell identity;
- stressor active flag;
- target token index.

The batch implementation must modify each active row only at its own frozen
target coordinate.

Inactive rows must be exactly native.

Hook/intervention state must not leak between chunks or checkpoint
recomputation.

---

## 11. Checkpoint-recomputation semantics

The Phase 2 streamed-backbone/checkpoint strategy may be reused.

The layer-17 stressor must be installed inside the chunk forward scope so
checkpoint recomputation executes the same deterministic forward
intervention.

Requirements:

- no hook accumulation;
- no stale row plan;
- no cross-chunk target leakage;
- identical stressor semantics during recomputation;
- RNG semantics unchanged from the inherited streamed backbone contract.

---

## 12. Phase 3A training data

Frozen training artifact:

`data/reason_router_gen5_phase3_xg1_contention_training_v1`

Frozen cardinalities:

- source pairs = 600
- labeled rows = 4200
- train pairs = 480
- dev pairs = 120
- train rows = 3360
- dev rows = 840
- split seed = 16384

Training-label counts:

- REFUTE = 480
- NOT_ENTITLED = 2400
- SUPPORT = 480

for the train split.

The implementation must not resplit or reorder by response.

---

## 13. Objective and optimization contract

Implementation must preserve the frozen envelope:

- final 3-way CE only;
- AdamW;
- learning rate = 0.001;
- weight decay = 0.0001;
- no scheduler;
- gradient clip norm = 5.0;
- 20 epochs;
- exactly 20 optimizer steps;
- final fixed-step checkpoint only;
- no early stopping;
- no dev checkpoint selection.

No training is authorized by this implementation authority.

---

## 14. Phase 3A contention diagnostics

The implementation may provide pure geometry functions for the frozen
Phase 3A gate.

For final unrestricted correction map:

`M = B_theta A_theta`

define:

`F_R = ||R22^T M||_F^2 / ||M||_F^2`

`F_C = ||C22^T M||_F^2 / ||M||_F^2`

These functions:

- require no model forward;
- compute no p-value;
- may be unit tested using synthetic tensors;
- must not change the frozen thresholds.

Frozen eventual PR gate:

1. `F_R(PR) >= 0.005`
2. `F_R(PR) >= 10 * F_R(P0)`
3. `F_R(PR) >= 4 * F_R(PC)`
4. `F_R(PR) >= 4 * F_C(PR)`
5. loss decreases from step 0;
6. finite correction parameters;
7. parent remains immutable.

Interpretation of this gate is outside the present implementation authority.

---

## 15. Authorized implementation files

Implementation scope is limited to new files:

`src/contramamba/gen5_phase3_causal_role_contention.py`

`scripts/train_reason_router_gen5_phase3a_contention.py`

`tests/test_reason_router_gen5_phase3_causal_role_contention.py`

`tests/test_train_reason_router_gen5_phase3a_contention.py`

No other scientific source file may be modified without a new explicit
correction.

In particular, do not modify:

- Phase 2 correction implementation;
- frozen XG1 generator;
- static Phase 3 artifacts;
- R22/C22 artifacts;
- PP3/PP5 artifacts;
- historical model snapshot.

---

## 16. Required implementation modes

The Phase 3A runner must initially support:

### `--static-verify-only`

Allowed:

- authenticate repository;
- authenticate static artifacts;
- authenticate PP3/PP5/R22/C22 bytes;
- validate pair split and row order;
- validate stressor plan;
- validate pure tensor geometry helpers.

Forbidden:

- model construction;
- checkpoint load;
- CUDA;
- model forward;
- backward;
- optimizer step;
- training;
- task evaluation;
- p-value.

### Later CUDA preflight mode

May be implemented under this authority but must not be executed until
separately authorized.

### Later Phase 3A run mode

May be implemented under this authority but must not be executed until
separately authorized.

---

## 17. Required implementation verification

Before implementation freeze, local CPU tests must establish at least:

1. frozen vector authentication;
2. PP3/PP5 geometry;
3. PR exact neutralization semantics;
4. PC coefficient-transfer semantics;
5. PR/PC matched correction norm;
6. P0 identity;
7. exact four-cell stressor-domain selection;
8. non-stressor rows unchanged;
9. per-row target-coordinate correctness;
10. gate-half unchanged;
11. non-strong channels unchanged;
12. non-target tokens unchanged;
13. mixed-batch correctness against rowwise reference;
14. pure F_R/F_C computation correctness;
15. label/split/static-artifact authentication;
16. no scientific execution in static verification.

No CUDA result is required for implementation freeze.

---

## 18. Stop conditions

Implementation must stop without training if any of the following occurs:

- static tree identity mismatch;
- PP3/PP5 SHA mismatch;
- R22/C22 SHA mismatch;
- stressor-target mismatch;
- unsupported cell receives stressor;
- target coordinate out of range;
- strong partition mismatch;
- PR residual qualification failure;
- batch-vs-reference mismatch;
- parent/trainable ownership violation;
- implementation requires modification of a frozen scientific artifact.

---

## 19. Explicit exclusions

This authority does not authorize:

- Phase 3A training;
- Phase 3A CUDA execution;
- Kaggle;
- Phase 3B implementation;
- C1/M1 Phase 3 training;
- confirmatory assay execution;
- new p-values;
- new stressor strengths;
- PP plane search;
- layer search;
- token search;
- rank search;
- new dataset construction.

---

## 20. Next gate

After implementation and CPU tests:

`PHASE3A_IMPLEMENTATION_FREEZE`

Then, and only then:

`PHASE3A_CUDA_RUNTIME_PREFLIGHT_AUTHORITY`

Training remains blocked until that later preflight passes.

STATUS =
`READY_FOR_BOUNDED_IMPLEMENTATION`
