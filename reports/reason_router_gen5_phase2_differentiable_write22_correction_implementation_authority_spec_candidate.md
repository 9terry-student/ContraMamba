# ContraMamba Gen5 — Phase 2 Differentiable WRITE22 Correction Implementation Authority

## 0. Status

PHASE =
`GEN5_PHASE2_DIFFERENTIABLE_WRITE_CORRECTION_IMPLEMENTATION_AUTHORITY`

STATUS =
`CANDIDATE_FOR_FREEZE`

IMPLEMENTATION =
`AUTHORIZED_ONLY_AFTER_THIS_FILE_IS_EXACTLY_FROZEN`

TRAINING =
`NOT_AUTHORIZED`

SCIENTIFIC_EXECUTION =
`NOT_AUTHORIZED`

TASK_EVALUATION =
`NOT_AUTHORIZED`

KAGGLE =
`NOT_AUTHORIZED`

This authority permits one bounded implementation of the already-frozen Gen5
Phase 2 state-update ownership design.

It does not authorize Phase 2 training, causal assay execution, task
evaluation, or GPU scientific execution.

---

## 1. Frozen parents

PHASE2_DESIGN_COMMIT =
`d9b84bca8871464807d2dccf6380a6e911d6dbf8`

PHASE2_DESIGN =
`reports/reason_router_gen5_phase2_state_update_ownership_implementation_design_candidate.md`

PHASE2_STATIC_PREPARATION_COMMIT =
`44e8853d08355641735b006459cd341ddb1dd631`

PHASE2_STATIC_TRAINING_PROVENANCE =
`reports/reason_router_gen5_phase2_static_preparation_d9b84bc_v1/training_provenance_summary.json`

PHASE2_PRIMARY_ASSAY_DATA =
`data/reason_router_gen5_phase2_xg1_ownership_assay_v1`

PHASE1B_FINAL_EVIDENCE_FREEZE =
`47154c02b6e691bbd2e577d934b0ef0145a04607`

PHASE1B_BRIDGE_CONCLUSION =
`GEN5_LAYER17_CAUSAL_ROLE_TO_LAYER22_NATIVE_WRITE_REALIZATION_BRIDGE_SUPPORTED`

REPRESENTATIVE_CHECKPOINT_SHA256 =
`1ff3fcf2ebd754ab6f9483d6a9982b9b04b9a4eb3357f9f8cdbe2b30399e7d2f`

R22_SHA256 =
`a69232900e8b5a91ec5248e36facee4d421eabd502829e6abf3d2719fdb02214`

C22_SHA256 =
`c692d39a7387e32e9bf76fe4db7ce2d30d5f7d22a4389e8af363a80b394155c4`

---

## 2. Implementation objective

Implement exactly three Phase 2 ownership arms:

- `G5-C0`
- `G5-C1`
- `G5-M1`

with one shared rank-2, bias-free learned correction architecture and one
difference only:

`G5-C0`:
`DeltaW_eff = DeltaW_theta`

`G5-C1`:
`DeltaW_eff = (I - P_C22) DeltaW_theta`

`G5-M1`:
`DeltaW_eff = (I - P_R22) DeltaW_theta`

The implementation must preserve:

`S22_t = Carry22_t + Write22_native_t + DeltaW_eff_t`

The historical native write itself must remain byte/operation identical to the
unmodified frozen parent path.

---

## 3. Exact target site

TARGET_LAYER_ZERO_BASED =
`22`

TARGET_OBJECT =
`model.mamba.layers[22].mixer`

CORRECTION_INPUT =
the exact normalized hidden-state tensor passed into the layer-22 Mamba mixer
before `mixer.in_proj`.

This is the `hidden_states` argument received by the layer-22 mixer after the
containing Mamba block normalization.

The implementation must not use:

- layer-21 output before layer-22 normalization;
- layer-22 mixer output;
- final token states;
- frame/predicate/sufficiency representations;
- task logits;
- Phase 1B endpoint values.

Padding/inactive sequence positions must receive zero correction.

---

## 4. Minimal correction module

Implement:

`A_theta: Linear(d_model, 2, bias=False)`

`B_theta: Linear(2, 24576, bias=False)`

with:

`DeltaW_theta_t = B_theta(A_theta(x22_t))`

The result is reshaped to the native layer-22 recurrent-state shape.

Expected frozen dimensions:

`d_model = 768`

`intermediate_size = 1536`

`state_size = 16`

`STATE_WIDTH = 1536 * 16 = 24576`

`CORRECTION_RANK = 2`

Expected trainable parameter count:

`2 * 768 + 24576 * 2 = 50688`

No other trainable parameter is permitted.

The implementation must fail closed if the observed frozen model dimensions do
not match these values.

---

## 5. Initialization

For a Phase 2 training seed `s`:

1. create a dedicated CPU `torch.Generator`;
2. seed it with exactly `s`;
3. initialize `A_theta.weight` by
   `torch.nn.init.kaiming_uniform_(..., a=sqrt(5), generator=g)`;
4. initialize `B_theta.weight` exactly to zero;
5. move the resulting parameters to the execution device without reinitializing.

For a fixed seed, C0/C1/M1 must receive byte-identical initial `A_theta` and
`B_theta`.

At initialization:

`DeltaW_theta == 0`

for every input.

No arm-specific initialization is allowed.

---

## 6. Frozen owner/control bases

R22 and C22 must be loaded from the already-frozen Phase 1B basis artifact.

They are non-trainable buffers.

They must not be reconstructed from model responses.

They must not be re-orthogonalized in a response-dependent manner.

The implementation must verify before use:

- shape `(24576, 2)`;
- rank `2`;
- expected SHA256 identity;
- orthonormal columns;
- cross-orthogonality between R22 and C22.

The runtime projector is:

`P_B(v) = B (B^T v)`

with basis `B` equal to R22 or C22.

Projector computation must remain differentiable with respect to `v`.

The basis tensor must have `requires_grad=False`.

---

## 7. Native-recurrence preservation by superposition

The Phase 2 implementation must not rewrite, replace, or approximate the
historical native layer-22 write.

Let the unmodified native recurrence be:

`H_native_t = Abar_t ⊙ H_native_(t-1) + Write_native_t`

and let the correction recurrence be:

`H_corr_t = Abar_t ⊙ H_corr_(t-1) + DeltaW_eff_t`

with:

`H_corr_-1 = 0`

Then:

`H_total_t = H_native_t + H_corr_t`

is exactly the recurrence obtained by adding `DeltaW_eff_t` to the native write
at every active step.

Therefore the implementation must preserve the frozen native scan and add only
the correction-state contribution to the layer-22 scan output.

The same frozen input-dependent decay `Abar_t`, the same C projection, and the
same downstream gate semantics must be used for the correction-state
contribution.

This superposition identity is the authorized implementation mechanism.

It is not permission to alter the native B/C/delta/A/D/gate parameters.

---

## 8. Reference backend

A small-batch autograd-preserving reference backend must be implemented first.

The reference backend must explicitly realize the correction recurrence token
by token without any detach on the correction path.

It may be slow.

It is verification-only.

It must support:

- CPU;
- CUDA when available;
- C0/C1/M1;
- forward;
- backward;
- zero-output initialization;
- diagnostic capture of pre-project and post-project correction vectors.

It must not be used for full scientific training unless a later execution
authority explicitly allows it.

---

## 9. Accelerated backend boundary

An accelerated Phase 2 backend may be implemented only if it is mathematically
the same correction recurrence defined above.

Permitted acceleration includes:

- fused implementation;
- recomputation/checkpointing;
- mathematically exact gradient accumulation internal to the correction
  recurrence;
- an equivalent custom autograd function.

Forbidden acceleration includes:

- moving the correction to mixer output;
- moving the correction to block residual output;
- treating the correction as a head residual;
- changing native B, C, delta, A, D, or gate values;
- approximating the recurrent correction as a same-token output correction;
- detaching correction state between tokens;
- truncating correction recurrence;
- token subsampling;
- owner/control-dependent batch changes.

If an accelerated backend cannot be made equivalent to the reference backend
within prospectively frozen verification tolerances, implementation stops.

Training is not authorized merely because the reference backend works.

---

## 10. Parent-model immutability

The representative historical model must be loaded before the Phase 2
correction wrapper is attached.

Every historical parent parameter must then be frozen.

The new correction must be attached without changing historical parameter
values.

Allowed runtime transformation:

replace only:

`model.mamba.layers[22].mixer`

with a wrapper that contains:

- the exact already-loaded frozen native mixer;
- the new correction module;
- frozen R22/C22 buffers;
- Phase 2 arm identity.

No other Mamba layer may be replaced.

No historical head may be replaced.

No historical parameter may be copied into a new trainable parameter.

The implementation must expose an audit mapping proving which wrapper parameter
names correspond to historical parent parameters versus new correction
parameters.

---

## 11. Historical wrapper source

The existing:

`src/contramamba/modeling_v6b_minimal.py`

must remain unchanged unless implementation proves that the runtime layer-22
wrapper is impossible without a narrow compatibility hook.

Default authorized scope is:

`NO CHANGE` to `src/contramamba/modeling_v6b_minimal.py`.

If modifying that file becomes necessary, stop.

A separate scope amendment is required before editing it.

This fail-closed rule prevents Phase 2 from becoming a refactor of the
historical Reason-Router model.

---

## 12. Trainable-parameter contract

After wrapper installation:

TRAINABLE =
exactly:

- `correction.A_theta.weight`
- `correction.B_theta.weight`

EXPECTED_TRAINABLE_TENSOR_COUNT =
`2`

EXPECTED_TRAINABLE_NUMEL =
`50688`

Every historical model parameter must have:

`requires_grad=False`

R22 and C22 buffers must have:

`requires_grad=False`

The future optimizer must receive an explicit list of the two correction
parameters.

No `model.parameters()` blanket optimizer construction is allowed.

---

## 13. Objective implementation boundary

The implementation may expose a training loss helper for:

`FINAL_3WAY_CROSS_ENTROPY_ONLY`

using only the frozen historical final logits and final labels.

It must not add:

- frame BCE;
- predicate BCE;
- sufficiency BCE;
- polarity CE;
- reason loss;
- ranking/intervention loss;
- R22/C22 loss;
- Q22 loss;
- D_NEC22 loss;
- D_SUF22 loss;
- regularization other than the frozen hard projector.

The historical final label class order remains:

`REFUTE, NOT_ENTITLED, SUPPORT`

The implementation must not use Phase 2 fresh assay rows for training.

---

## 14. Historical training envelope encoded but not executed

The training harness may encode the already-frozen Phase 2 training contract:

- train rows = `2880`;
- dev rows = `720`;
- split seed = `8192`;
- optimizer = `torch.optim.AdamW`;
- learning rate = `0.001`;
- weight decay = `0.0001`;
- scheduler = none;
- epochs = `20`;
- optimizer steps = `20`;
- gradient clip norm = `5.0`;
- training seeds = `5201, 5202, 5203`;
- checkpoint state = final fixed step only.

However this authority does not permit executing those 20-step trainings.

If engineering constraints require changing the frozen physical full-objective
training semantics, stop before scientific training authority.

No silent mini-batch substitution is permitted.

---

## 15. Required new files

Implementation scope is limited to exactly these new files:

1.
`src/contramamba/gen5_phase2_state_update_ownership.py`

2.
`scripts/train_reason_router_gen5_phase2_state_update_ownership.py`

3.
`tests/test_reason_router_gen5_phase2_state_update_ownership.py`

4.
`scripts/verify_reason_router_gen5_phase2_state_update_ownership.py`

No existing tracked file may be modified.

No additional file may be created without reporting a scope blocker.

Generated temporary test artifacts must remain outside the repository or under
existing ignored temporary paths.

---

## 16. Core module responsibilities

`src/contramamba/gen5_phase2_state_update_ownership.py` must contain only the
bounded Phase 2 mechanics:

- arm enum / exact arm validation;
- basis identity and geometry validation;
- rank-2 correction module;
- deterministic initialization;
- projector;
- reference correction recurrence;
- layer-22 native-mixer wrapper;
- wrapper installation/removal helpers;
- parent/correction parameter audit helpers;
- correction diagnostics needed by later verification.

It must not:

- load datasets;
- choose cohorts;
- compute scientific p-values;
- perform model selection;
- run training automatically;
- read Phase 2 primary assay outputs.

---

## 17. Training-script responsibilities

`scripts/train_reason_router_gen5_phase2_state_update_ownership.py` may
implement the future bounded training harness but must fail closed unless an
explicit future execution authority commit is supplied.

Without that future authority, invocation must stop before:

- model forward for training;
- backward;
- optimizer step;
- task evaluation.

The script may perform static argument validation and provenance validation
without execution.

The eventual scientific training path must load the representative checkpoint,
attach the correction wrapper, freeze the historical parent, and use only the
two correction parameters.

---

## 18. Verification-script responsibilities

`scripts/verify_reason_router_gen5_phase2_state_update_ownership.py` is a
verification tool, not a scientific runner.

It may run bounded synthetic/reference forward and backward checks.

It must not:

- load the fresh XG1 `8701..9000` assay population;
- compute Q22 scientific outcomes;
- compute a scientific p-value;
- run the 20-step Phase 2 training schedule;
- write a scientific conclusion.

Its output must clearly separate:

`CODE_CORRECTNESS`

from:

`SCIENTIFIC_EXECUTION`

---

## 19. Required unit tests

The dedicated test file must cover at minimum:

### Construction

- only C0/C1/M1 accepted;
- frozen dimensions required;
- exact trainable tensor count = 2;
- exact trainable numel = 50688;
- R22/C22 are buffers, not parameters.

### Initialization

- same seed gives byte-identical A/B;
- different seeds may change A;
- B is exactly zero;
- initial correction output is exactly zero;
- C0/C1/M1 have identical zero output at initialization.

### Projector

- C1 removes C22 component;
- M1 removes R22 component;
- projector preserves autograd to correction input;
- basis buffers receive no gradient;
- C0 does not project.

### Recurrence

On a tiny deterministic synthetic recurrence:

- explicit `native + correction` recurrence equals superposed
  `native recurrence + correction recurrence`;
- zero correction equals native recurrence;
- correction persists through later recurrent steps;
- inactive positions receive zero new correction.

### Wrapper

- only layer 22 is wrapped;
- native mixer object identity is retained inside wrapper;
- parent historical parameter values are unchanged;
- wrapper removal restores original mixer object;
- no other layer changes.

### Backward

- final scalar loss reaches A/B;
- gradients are finite;
- historical parent gradients remain absent;
- basis gradients remain absent;
- one test optimizer step changes correction parameters only.

### Objective

- Phase 2 helper equals plain three-way cross entropy;
- no auxiliary historical loss is added.

---

## 20. Independent verification gates

After implementation, before any implementation freeze, run an independent
verification that establishes:

### Gate A — static scope

- exactly four new tracked files;
- no existing file modified;
- no training/evaluation artifact created.

### Gate B — zero-correction parent equivalence

With the same representative parent, input, mode, and RNG state:

- unwrapped parent;
- wrapped C0 at zero initialization;
- wrapped C1 at zero initialization;
- wrapped M1 at zero initialization;

must produce identical final logits within a prospectively declared tolerance.

No best-arm comparison is allowed.

### Gate C — recurrence semantics

Reference backend must match direct explicit state-write addition on a bounded
synthetic case for:

- forward correction state;
- resulting scan contribution;
- gradients with respect to A/B.

### Gate D — projector semantics

For nonzero synthetic correction:

- M1 R22 residual is numerically zero within tolerance;
- C1 C22 residual is numerically zero within tolerance;
- C0 is unchanged;
- all three arms have equal parameter count.

### Gate E — parent immutability

After backward and one bounded optimizer step:

- parent parameter fingerprint unchanged;
- R22 fingerprint unchanged;
- C22 fingerprint unchanged;
- correction fingerprint changed;
- optimizer parameter ownership = correction only.

### Gate F — accelerated backend

If an accelerated backend exists:

- forward agrees with reference;
- correction-state semantics agree;
- A gradient agrees;
- B gradient agrees.

Failure of Gate F blocks training but does not invalidate a correct reference
implementation.

---

## 21. Scientific data firewall

Implementation and verification must not inspect scientific responses from:

`data/reason_router_gen5_phase2_xg1_ownership_assay_v1`

The fresh XG1 files may be referenced only to assert that they are forbidden
for implementation tests.

No model forward on `xg1_fact_8701..9000` is authorized by this document.

No Phase 2 primary endpoint may be computed.

---

## 22. Forbidden implementation changes

Forbidden:

- editing historical Phase 1B evidence;
- editing R22/C22 basis artifacts;
- editing the frozen Phase 2 cohort;
- modifying the representative checkpoint;
- modifying historical dataset/split artifacts;
- changing correction rank;
- adding bias;
- adding nonlinearity;
- adding normalization;
- adding a learned gate;
- adding a second correction module;
- moving correction away from layer 22 recurrent state;
- changing native layer-22 write;
- changing read access;
- adding detach to the correction recurrence;
- unfreezing parent parameters;
- modifying `modeling_v6b_minimal.py` under this authority;
- changing the primary assay cohort;
- implementing a hyperparameter sweep.

---

## 23. Stop conditions

Stop implementation and report a blocker if:

- layer-22 runtime dimensions differ from the frozen dimensions;
- R22/C22 identity or geometry fails;
- exact native-write superposition cannot be implemented without modifying
  historical native write semantics;
- `modeling_v6b_minimal.py` modification appears necessary;
- an existing tracked file outside the four-file scope appears necessary;
- accelerated training requires truncating/detaching the correction recurrence;
- correction-only optimizer ownership cannot be guaranteed;
- full Phase 2 training would require silently changing the frozen training
  semantics;
- fresh assay responses would need to be inspected;
- scientific training/evaluation appears necessary to debug implementation.

---

## 24. Validation command after implementation

The implementation must pass, at minimum:

`python -m pytest -q tests/test_reason_router_gen5_phase2_state_update_ownership.py`

and:

`python -m scripts.verify_reason_router_gen5_phase2_state_update_ownership`

The verifier must remain bounded and non-scientific.

No Kaggle run is required by this implementation authority.

---

## 25. Commit / push boundary

Implementation commit/push is not automatic.

After implementation and verification:

1. inspect exact repository status;
2. run the dedicated tests;
3. run the independent verifier;
4. run `cm ship`;
5. review exact staged files;
6. only then manually commit/push the four implementation files.

No training authority is created by that implementation commit.

---

## 26. Next stage after verified implementation freeze

If the implementation and independent forward/backward verification both pass:

NEXT_STAGE =
`GEN5_PHASE2_TRAINING_EXECUTION_AUTHORITY`

If reference implementation passes but accelerated backend equivalence fails:

NEXT_STAGE =
`GEN5_PHASE2_ACCELERATED_CORRECTION_BACKEND_BLOCKED`

and scientific training remains forbidden.

---

## 27. Decision

AUTHORIZED_AFTER_EXACT_FREEZE =
`BOUNDED_FOUR_FILE_PHASE2_WRITE22_IMPLEMENTATION`

SCIENTIFIC_TRAINING =
`NO`

TASK_EVALUATION =
`NO`

PRIMARY_CAUSAL_ASSAY =
`NO`

KAGGLE =
`NO`

MODEL_PARAMETER_UNFREEZE =
`NO`

HISTORICAL_MODEL_FILE_EDIT =
`NO`

FRESH_XG1_RESPONSE_ACCESS =
`NO`

STATUS =
`PASS_READY_FOR_MANUAL_IMPLEMENTATION_AUTHORITY_FREEZE`
