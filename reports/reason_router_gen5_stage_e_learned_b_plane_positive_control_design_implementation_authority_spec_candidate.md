# Gen5 Stage E Learned-B-Plane Positive Control
# Design and Implementation Authority

## Status

DESIGN_FROZEN=YES

IMPLEMENTATION_ALLOWED=YES_BOUNDED_POSITIVE_CONTROL

SCIENTIFIC_EXECUTION_ALLOWED=NO

CUDA_ALLOWED=NO

TRAINING_ALLOWED=NO

EVALUATION_ALLOWED=NO

BACKWARD_ALLOWED=NO

OPTIMIZER_STEP_ALLOWED=NO

CONFIRMATORY_9601_9900_ALLOWED=NO

## Parent evidence

STAGE_E_EVIDENCE_FREEZE_COMMIT=382a4961ee701ddf38b39ef9fa75aa932ce34f39

STAGE_E_EXECUTION_COMMIT=fdbecf39c8ccf757c4e3afbd5df62c22c74e7f1c

STAGE_E_IMPLEMENTATION_FREEZE_COMMIT=88625179cd63d4e61e8719045b9a58b611f9825e

PHASE3A_SOURCE_EXECUTION_COMMIT=d58e89477fe43d0e5fa6aaaa7cec31d8c78cda4e

## Scientific motivation

Stage E constrained the layer-22 state-write correction to either R22 or C22.

Observed mean recovery relative to the frozen unrestricted P0 gain:

- E-R22: 0.00213934506552454
- E-C22: 0.00186063347357193

Observed mean R22-minus-C22 recovery difference:

- 0.000278711591952613

All six Stage E cells retained:

- effective output rank 2;
- effective operator rank 2;
- fixed-plane residual approximately numerical zero;
- cross-plane leakage approximately numerical zero;
- unchanged parent fingerprint;
- exactly 20 optimizer steps;
- no confirmatory population.

Therefore R22 and C22 both recovered only approximately 0.2 percent of the
seed-matched unrestricted P0 gain.

This does not yet distinguish:

1. R22/C22 are poor task-adaptation output planes;

from:

2. the fixed-plane QMA parameterization or its frozen 20-step optimization
   contract is itself unable to relearn even inside a known-successful output
   plane.

The present positive control is designed only to resolve this ambiguity.

## Exact scientific question

For each seed, if the output plane is fixed to the span of the already-trained
unrestricted P0 final B matrix from that same seed, can the Stage E QMA
parameterization relearn substantial task correction under the exact same
20-step training contract?

This is a diagnostic positive control.

The learned plane was obtained using the same task objective and therefore is
not an independent scientific treatment and must not be presented as
confirmatory evidence of a general mechanism.

## Exact source checkpoints

Only the existing Phase3A P0 unrestricted final corrections are allowed.

### Seed 6201

Path:

`reports/reason_router_gen5_phase3a_training_runs/gen5-phase3a-contention-qualification-9cell-d58e894-retry3/cells/seed6201/P0/final_correction.pt`

SHA256:

`157ca1c945c7f70b03638ef7e06750a4f4105e5504904272529ecd82bfba5ddf`

### Seed 6202

Path:

`reports/reason_router_gen5_phase3a_training_runs/gen5-phase3a-contention-qualification-9cell-d58e894-retry3/cells/seed6202/P0/final_correction.pt`

SHA256:

`1aa16196aa5aa338c30ac71dd20c7a6fb36a62eafdbe52bcf5696cfc3b87c214`

### Seed 6203

Path:

`reports/reason_router_gen5_phase3a_training_runs/gen5-phase3a-contention-qualification-9cell-d58e894-retry3/cells/seed6203/P0/final_correction.pt`

SHA256:

`c536464dd8541423d18a2bfbfee40211f068885baa37730315dbf7cf0a784770`

Each source checkpoint must authenticate:

- schema `GEN5_PHASE3A_FINAL_CORRECTION_V1`;
- matching seed;
- arm `G5-C0`;
- pressure `P0`;
- parent checkpoint identity;
- exact checkpoint SHA256;
- `B_theta.weight` shape `[24576, 2]`;
- effective B rank exactly 2.

No PR or PC checkpoint may be used.

## Learned-plane derivation

For each seed s, let the authenticated unrestricted final output matrix be:

`B_free,s = B_theta.weight`

with shape `[24576, 2]`.

Derive the fixed positive-control basis only from this matrix.

Use float64 thin QR:

`B_free,s = Q_s R_s`

with:

`Q_s = torch.linalg.qr(B_free,s.float64(), mode="reduced").Q`

and corresponding `R_s`.

Canonicalize QR signs deterministically so that each diagonal entry of R is
nonnegative.

If a diagonal entry is negative, multiply the corresponding column of Q and
row of R by -1.

Require:

- Q shape `[24576, 2]`;
- Q finite;
- Q rank exactly 2;
- `Q.T @ Q` equal to identity within frozen numerical tolerance;
- projector residual of B_free outside span(Q) approximately zero;
- reconstruction `Q @ R` matches B_free within frozen tolerance.

Record exact Q tensor SHA256 for each seed.

No SVD alternative, random rotation, basis search, interpolation, or plane
optimization is allowed.

## Positive-control arm

Single arm:

`E-BFREE`

Exactly three cells:

- seed6201 / E-BFREE
- seed6202 / E-BFREE
- seed6203 / E-BFREE

No R22/C22 rerun.

No free-B rerun.

No PR/PC.

No additional seed.

## Parameterization

For each seed-specific frozen Q_s:

`B_eff = Q_s M`

with:

- Q_s `[24576, 2]`, frozen;
- M `[2, 2]`, trainable;
- A `[2, 768]`, trainable;
- correction operator `Q_s M A`;
- no bias.

Initialization must exactly match Stage E:

- M exactly zero initialized;
- A initialized from the same seed-specific deterministic initialization used
  by Stage E;
- seed 6201 positive control must therefore share the Stage E seed6201 A
  initialization convention, and analogously for 6202 and 6203.

Exactly two trainable tensors:

- `A_theta.weight`
- `M_theta.weight`

Expected trainable parameters:

`1540`

## Frozen training and evaluation contract

Use exactly the Stage E contract:

- Phase3A train population;
- Phase3A dev population;
- train rows 3360;
- dev rows 840;
- split seed 16384;
- pressure P0;
- parent checkpoint unchanged;
- AdamW;
- learning rate 0.001;
- weight decay 0.0001;
- 20 optimizer steps;
- gradient clip norm 5.0;
- no scheduler;
- final 3-way cross entropy only;
- no early stopping;
- no checkpoint selection;
- final fixed step only.

Parent checkpoint SHA256:

`1ff3fcf2ebd754ab6f9483d6a9982b9b04b9a4eb3357f9f8cdbe2b30399e7d2f`

Confirmatory population `9601..9900` remains forbidden.

## Frozen unrestricted reference

Use existing seed-matched unrestricted P0 gains only.

- seed6201: 0.4984860420227051
- seed6202: 0.5020102858543396
- seed6203: 0.4994615912437439

Do not retrain unrestricted P0.

## Primary quantities

For each positive-control cell:

`gain_BFREE = CE_ZERO - CE_BFREE`

`recovery_BFREE = gain_BFREE / gain_FREE_P0_seed`

Compare descriptively with the already-frozen Stage E R22 and C22 recoveries.

No scientific p-values.

No post-hoc numerical success threshold.

## Interpretation contract

This positive control answers only whether the frozen Stage E optimization
scheme can relearn useful correction when given a seed-matched output plane
already known to have supported a successful unrestricted solution.

### Outcome A — substantial BFREE recovery

If BFREE recovery is qualitatively far larger than the approximately 0.2
percent R22/C22 recoveries across the seed-matched cells, then the Stage E
failure cannot reasonably be attributed simply to fixed-plane QMA
parameterization incapacity.

This would strengthen the bounded interpretation that output-plane orientation
matters and that the native-causal R22 plane is not an optimizer-preferred
task-adaptation plane under the tested objective and training contract.

### Outcome B — BFREE also shows negligible recovery

Then the Stage E R22/C22 result must not be interpreted as evidence that their
orientation itself caused the failure.

The fixed-plane reoptimization scheme would be implicated, and the causal-plane
orientation claim would remain unresolved.

### Mixed outcome

If BFREE recovery varies materially across seeds, do not average away the
heterogeneity.

Treat fixed-plane learnability as seed-dependent and preserve the ambiguity.

## Scope lock

Not authorized:

- random-plane controls;
- R22/C22 reruns;
- plane sweep;
- seed sweep;
- layer sweep;
- rank sweep;
- learning-rate sweep;
- optimizer sweep;
- step-count sweep;
- pressure sweep;
- token search;
- new dataset;
- confirmatory evaluation;
- unrestricted retraining;
- result-dependent implementation changes.

## Authorized implementation paths

Exactly these new paths may be created:

1. `src/contramamba/gen5_stage_e_learned_b_plane_positive_control.py`
2. `scripts/train_reason_router_gen5_stage_e_learned_b_plane_positive_control.py`
3. `tests/test_reason_router_gen5_stage_e_learned_b_plane_positive_control.py`

No existing source, script, test, data, report, artifact, or authority file may
be modified during implementation.

The implementation should reuse frozen Stage E primitives where safe rather
than copy or alter the frozen Stage E implementation.

## Required implementation validation

Implementation phase may run only CPU/static validation.

Required checks:

1. exact source checkpoint SHA authentication;
2. checkpoint schema/seed/arm/pressure authentication;
3. deterministic QR/sign canonicalization;
4. exact learned-plane reconstruction checks;
5. exact three-cell matrix contract;
6. exactly 1540 trainable parameters in the future runtime object;
7. M zero initialization;
8. seed-matched A initialization;
9. static authentication of train/dev identities;
10. runtime modes fail closed without a later execution authority;
11. no checkpoint model instantiation in static verify;
12. no CUDA;
13. no forward;
14. no backward;
15. no optimizer construction/step;
16. no training;
17. no evaluation;
18. no confirmatory loading.

## Execution boundary

This authority does not authorize the three-cell scientific execution.

After implementation is independently validated and frozen at a clean pushed
commit, a separate minimal execution authority is required before CUDA,
backward, optimizer construction, training, or dev evaluation.
