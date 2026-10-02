# Gen5 Stage E Learned-B-Plane A-Fixed M-Only Diagnostic
# Validated Evidence Report Candidate

## Evidence identity

EXECUTION_HEAD=094fecbcf63c9dfece06df3c6c7a818abe1f9172

IMPLEMENTATION_FREEZE_COMMIT=b1ab97600d47e0f86ca3a427befffb777a10829c

EXECUTION_AUTHORITY_COMMIT=094fecbcf63c9dfece06df3c6c7a818abe1f9172

RUN_NAME=gen5-stagee-bfree-afix-monly-three-cell-094fecb-r1

RUN_COMMAND_SHA256=4bdbe4b3ab010df1400480e648642c4807244dc60ebb632106b34e2ba9337238

HANDOFF_ZIP_SHA256=cebad05e9d73d66819aba5366372a023c9d50c9b263faea5bd8281b9d4ee4a13

RUN_LOG_SHA256=eb0e64365d23405e7873a2b74f3ee3091efa1332fa81e958f2a119277d0a39c7

RUN_META_SHA256=00b9461c5577bc500f3be4d583f2550f755c0c41b4a85b647d8d0e152a2796ee

IMPORTED_FILE_COUNT=11

ARTIFACT_VALIDATION=GEN5_STAGE_E_BFREE_AFIX_MONLY_ARTIFACT_PROVENANCE_PASS

## Scientific question

The preceding Stage E evidence established:

1. unrestricted optimization recovers substantial task benefit while largely
   bypassing R22;
2. fixed learned-B output-plane orientation improves short-horizon
   accessibility relative to R22/C22 but still recovers only a small fraction
   of unrestricted gain;
3. initializing A from the same seed unrestricted A_free increases recovery
   relative to random-A BFREE;
4. static BFREE/AINIT decomposition localizes the remaining mismatch primarily
   to failure to acquire the source-equivalent 2x2 M core rather than final A
   row-space mismatch.

The present AFIX-MONLY diagnostic asks whether joint A/M optimization itself
interferes with M-core acquisition.

Relative to E-BFREE-AINIT, exactly one factor changes:

- A is initialized to the same seed-matched A_free;
- A remains frozen for the entire run;
- only M_theta.weight is trainable.

If joint A drift/coupling were a major bottleneck, freezing A should improve
recovery relative to AINIT.

## Frozen execution contract

Arm:

`E-BFREE-AFIX-MONLY`

Seeds:

`6201,6202,6203`

Pressure:

`P0`

Trainable tensor:

`M_theta.weight`

Trainable parameter count:

`4`

A initialization:

same-seed unrestricted `A_free`

A trainability:

`False`

M initialization:

exact zero

Optimizer:

AdamW

Learning rate:

`0.001`

Weight decay:

`0.0001`

Gradient clip norm:

`5.0`

Steps:

`20` per seed, `60` total

GPU topology:

`TWO_INDEPENDENT_SINGLE_GPU_WORKERS_NO_DDP`

Confirmatory population `9601..9900`:

not loaded

## Runtime validation

The recovery CUDA preflight passed with:

- seed6201;
- exact frozen Q identity;
- exact A_free identity;
- A frozen;
- `A.grad is None`;
- finite M gradient;
- optimizer not constructed;
- optimizer step count 0;
- training false;
- task evaluation false;
- confirmatory data not loaded.

The main matrix then completed successfully with all three cells.

## Artifact and provenance validation

Independent post-import validation passed:

`GEN5_STAGE_E_BFREE_AFIX_MONLY_ARTIFACT_PROVENANCE_PASS`

Validated files:

`11`

For every seed:

- execution/implementation/authority identities matched;
- source checkpoint, A_free, B_free, and Q identities matched;
- A remained byte-identical to A_free;
- `A_grad_present_count = 0`;
- trainable tensor count = 1;
- trainable parameter count = 4;
- optimizer steps = 20;
- parent fingerprint was unchanged;
- checkpoint/report/provenance hashes were mutually consistent;
- train/dev row and encoding identities matched;
- confirmatory data were not loaded;
- scientific conclusion remained null inside execution artifacts.

## Seedwise results

### seed6201

DEV_CE=1.3095064163208

DEV_ACCURACY=0.269047619047619

GAIN_VS_ZERO=0.0314590930938721

RECOVERY_VS_FREE_P0=0.063109275770733

FROZEN_AINIT_RECOVERY=0.135931570756102

DELTA_AFIX_MONLY_MINUS_AINIT=-0.072822294985369

### seed6202

DEV_CE=1.31032621860504

DEV_ACCURACY=0.269047619047619

GAIN_VS_ZERO=0.0306392908096313

RECOVERY_VS_FREE_P0=0.0610331932890345

FROZEN_AINIT_RECOVERY=0.124898380318522

DELTA_AFIX_MONLY_MINUS_AINIT=-0.0638651870294875

### seed6203

DEV_CE=1.30858898162842

DEV_ACCURACY=0.269047619047619

GAIN_VS_ZERO=0.0323765277862549

RECOVERY_VS_FREE_P0=0.0648228579611734

FROZEN_AINIT_RECOVERY=0.139898741881301

DELTA_AFIX_MONLY_MINUS_AINIT=-0.0750758839201276

## Aggregate result

Mean AFIX-MONLY recovery:

`0.0629884423403136`

Mean AINIT recovery:

`0.133576230985308`

Mean AFIX-MONLY minus AINIT:

`-0.0705877886449947`

Mean BFREE recovery:

`0.07142508872336398`

Mean AFIX-MONLY minus BFREE:

`-0.008436646383050375`

AFIX-MONLY / AINIT mean recovery ratio:

`0.47155427186174825`

AFIX-MONLY / BFREE mean recovery ratio:

`0.8818811914154382`

Unrecovered unrestricted gain under AFIX-MONLY:

`0.9370115576596864`

## Scientific interpretation

The prospective "A drift/coupling interference" explanation is not supported.

Freezing A to the exact same-seed unrestricted A_free did not improve recovery.
It reduced recovery relative to E-BFREE-AINIT in every seed.

The effect is seed-consistent:

- seed6201 delta vs AINIT: `-0.072822294985369`
- seed6202 delta vs AINIT: `-0.0638651870294875`
- seed6203 delta vs AINIT: `-0.0750758839201276`

Therefore, within this frozen 20-step QMA contract, A trainability is
beneficial rather than a harmful source of interference.

This result sharpens the preceding static decomposition.

The AINIT improvement should not be interpreted as successful preservation or
reacquisition of the source A row-space. The preceding decomposition showed
that final A row-space moved farther from A_free while task recovery improved.
The AFIX-MONLY result now shows that keeping A exactly at A_free removes that
benefit.

The combined evidence supports the interpretation that A movement provides
useful co-adaptation with M, even though that movement does not recover the
source A geometry.

At the same time, AFIX-MONLY recovers only approximately 6.30 percent of
unrestricted gain, leaving approximately 93.70 percent unrecovered.

The residual fixed-plane failure therefore remains concentrated in the
short-horizon acquisition dynamics of the QMA factorization, especially the
2x2 M core, but the present result does not yet distinguish:

1. M-core scale/conditioning;
2. short optimization horizon;
3. useful A/M co-adaptive reparameterization effects beyond simple A
   row-space geometry.

No learning-rate, optimizer, horizon, rank, pressure, layer, or plane sweep is
justified yet.

## Supported conclusion

`GEN5_STAGE_E_AFIX_MONLY_A_DRIFT_INTERFERENCE_NOT_SUPPORTED`

`GEN5_STAGE_E_TRAINABLE_A_COADAPTATION_IMPROVES_SHORT_HORIZON_FIXED_PLANE_ACCESSIBILITY`

These conclusions are restricted to the frozen Stage E learned-B-plane,
P0, 20-step optimization contract.

## Next scientific action

Before any new GPU experiment, perform a read-only static decomposition of the
AFIX-MONLY final M tensors against the exact same-seed source QR core R_B.

Because Q and A are both fixed in AFIX-MONLY, the source-equivalent target is
unambiguous:

`M_exact = R_B`

This removes the gauge ambiguity that affected direct M comparisons in the
trainable-A runs.

The next static analysis should report, per seed:

- final M SHA256;
- exact R_B SHA256;
- `||M_final - R_B||_F / ||R_B||_F`;
- cosine between flattened M_final and R_B;
- `||M_final||_F / ||R_B||_F`;
- singular values of M_final and R_B;
- determinant and condition number where finite;
- the exact effective-operator error induced by M mismatch under frozen
  A_free;
- comparison to the corresponding AINIT and BFREE static decomposition values
  only as frozen references.

This is the smallest discriminating next step.

Only after that decomposition should a new execution intervention be chosen.
