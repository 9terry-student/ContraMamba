# Gen5 Stage E AFIX-MONLY SCALEMATCH Validated Evidence Report Candidate

## Status

VALIDATED_EVIDENCE_AND_STATIC_INTERPRETATION_CANDIDATE

## Evidence identity

Execution commit:

`a894416889c4333c591efa987239915ac8160335`

Run:

`gen5-stagee-bfree-afix-monly-scalematch-three-cell-a894416-r1`

Seeds:

- 6201
- 6202
- 6203

Pressure:

`P0`

Trainable parameterization:

- `Q_B`: fixed
- `A_free`: fixed
- `M`: trainable
- trainable parameter count per cell: `4`

Optimization contract:

- AdamW
- 20 optimizer steps per seed
- SCALEMATCH learning rate: `0.11085125168440814`
- no confirmatory 9601–9900 evaluation

Artifact/provenance validation:

`GEN5_STAGE_E_BFREE_AFIX_MONLY_SCALEMATCH_ARTIFACT_PROVENANCE_PASS`

Files validated:

`11`

Static decomposition validation:

`GEN5_STAGE_E_BFREE_AFIX_MONLY_SCALEMATCH_STATIC_CORE_DECOMPOSITION_PASS`

No CUDA, training, backward, optimizer construction, or task evaluation was performed during post-import static decomposition.

## Scientific question

The preceding AFIX-MONLY experiment established that a fixed oracle learned-B plane `Q_B`, exact same-seed `A_free`, and a trainable 2x2 core `M` had exact representation capacity for the unrestricted source operator,

`B_free A_free = Q_B R_B A_free`,

but recovered only approximately 6.30 percent of unrestricted task gain under the original learning-rate/horizon contract.

Static decomposition then identified a parameterization-dependent Adam movement-scale confound. Under the same learning rate and 20-step horizon, the four-coordinate M parameterization received a nominal movement budget approximately

`sqrt(49152 / 4) = 110.85125168440814`

times smaller than unrestricted B in Frobenius-scale terms.

The prospective SCALEMATCH diagnostic changed exactly this scale factor.

The present question is:

> Does dimension-normalized optimizer scale recover the unrestricted task benefit, and if so, does that recovery occur by reconstructing the exact source core `R_B` and its rank-2 operator geometry?

## Task recovery

Mean SCALEMATCH recovery:

`0.990747485451451`

Mean task gain:

`0.49535350004832`

Frozen AFIX-MONLY mean recovery:

`0.0629884423403136`

Mean recovery increase:

`0.927759043111137`

Seedwise recovery:

- seed6201: `0.994719609986981`
- seed6202: `0.984798752364991`
- seed6203: `0.992724094002381`

All three SCALEMATCH cells reached dev accuracy:

`0.714285714285714`

Thus dimension-normalized scale matching changed the fixed-plane four-parameter control from approximately 6.30 percent mean recovery to approximately 99.07 percent mean recovery.

## Source-core reconstruction

Despite near-complete task-gain recovery, final M did not reconstruct the exact source core `R_B`.

Mean core metrics:

- `M_REL_ERR = 0.499879735101667`
- `M_COS = 0.86890271026218`
- `M_NORM_RATIO = 0.809150777488996`
- scalar projection coefficient `ALPHA = 0.702151811358037`
- scalar-residual fraction `0.399334270931708`

Therefore the SCALEMATCH solution is not well described as exact recovery of `R_B`, nor as a simple scalar rescaling of `R_B`.

## Effective-operator reconstruction

Because `Q_B` is fixed and orthonormal, comparison of

`M A_free`

against

`R_B A_free`

directly measures mismatch of the effective operator within the fixed output plane without materializing the full ambient operator.

Mean effective-operator metrics:

- relative error: `0.431380325372166`
- cosine: `0.90586999140295`
- norm ratio: `0.843116205090085`

Near-complete task recovery therefore does not require near-exact reconstruction of the unrestricted same-seed effective operator.

## Singular-mode structure

The unrestricted source core remains clearly rank 2.

Mean source singular-value ratio:

`R_sv2 / R_sv1 = 0.419079628207683`

The SCALEMATCH M remains numerically close to rank 1.

Mean learned singular-value ratio:

`M_sv2 / M_sv1 = 0.00316262212683144`

Mean source-mode acquisition:

- mode 1: `0.828662412297384`
- mode 2: `0.00285246677950669`

Mean off-diagonal magnitude relative to source scale:

`0.268934331630214`

Seedwise M condition numbers were approximately:

- seed6201: `3973.63`
- seed6202: `111.73`
- seed6203: `3495.64`

By contrast, source `R_B` condition numbers were approximately 2–3.

Thus SCALEMATCH strongly acquires a dominant task-useful direction while leaving the second source mode almost entirely absent.

## Interpretation

The experiment supports the following bounded conclusions.

`GEN5_STAGE_E_PARAMETERIZATION_DEPENDENT_OPTIMIZER_SCALE_WAS_A_MAJOR_CAUSE_OF_THE_PREVIOUS_AFIX_MONLY_ABSOLUTE_RECOVERY_FAILURE`

The increase from approximately 6.30 percent to approximately 99.07 percent recovery after the single prospectively derived scale correction establishes that the previous absolute fixed-plane failure cannot be interpreted as evidence of insufficient four-parameter representation capacity.

`GEN5_STAGE_E_NEAR_FULL_TASK_RECOVERY_DOES_NOT_REQUIRE_SOURCE_CORE_RECOVERY`

SCALEMATCH recovers almost all unrestricted task gain while remaining substantially different from the exact same-seed `R_B` core and its effective operator.

`GEN5_STAGE_E_SOURCE_SECOND_MODE_IS_NOT_REQUIRED_FOR_NEAR_FULL_FROZEN_DEV_TASK_GAIN_UNDER_THE_TESTED_CONTRACT`

The source core has a substantial second singular mode, whereas SCALEMATCH M remains almost rank 1 and acquires essentially none of the corresponding second source mode.

`GEN5_STAGE_E_FIXED_ORACLE_SUBSPACES_ADMIT_FUNCTIONALLY_SUBSTITUTABLE_CORE_SOLUTIONS`

Within the provided same-seed `Q_B` output plane and exact `A_free` read-side geometry, materially different 2x2 cores can produce nearly the same frozen Phase3A dev task benefit.

## What is not established

This experiment does not establish that four trainable parameters are sufficient for learning the task from scratch.

The experiment is oracle-conditioned:

- `Q_B` was extracted from the same-seed unrestricted solution;
- `A_free` was extracted from the same-seed unrestricted solution.

Therefore the difficult problem of discovering useful input/output subspaces remains outside this control.

The present evidence also does not establish:

- that all unrestricted parameters are unnecessary;
- that high-dimensional parameterization exists only for subspace search;
- a universal intrinsic task dimension;
- cross-task or cross-model generalization;
- a LoRA or parameter-efficient fine-tuning mechanism;
- a universal optimizer scaling law;
- a confirmatory statistical claim.

The strongest current distinction is:

`representation capacity once useful subspaces are known`

versus

`optimization/search freedom required to discover useful subspaces`.

The former is shown to be extremely small in this oracle-conditioned Stage E setting.

The latter remains unresolved.

## Next scientific action

Do not run another training experiment yet.

Perform a read-only cross-seed subspace-discovery audit using the already frozen unrestricted seed6201, seed6202, and seed6203 solutions.

The audit should separately analyze:

1. output-plane geometry from each same-seed `Q_B`;
2. read-side row-space geometry from each same-seed `A_free`;
3. pairwise principal-angle / singular-value relationships;
4. pooled subspace spectrum;
5. leave-one-seed-out projection of the held-out seed onto the span learned from the other two seeds.

Simple membership in a pooled six-dimensional span is not itself informative because that span is constructed from the same observations.

The discriminating quantity is held-out predictability.

The next question is:

> Are the seed-dependent unrestricted rank-2 solutions samples from a substantially smaller shared task meta-subspace, or do different seeds discover genuinely different useful planes in the high-dimensional ambient space?

No CUDA, training, task evaluation, new seeds, rank sweep, learning-rate sweep, or architecture change is authorized by this report.
