# Gen5 A-init Representation Freedom Temporal Birth — Phase A Evidence Report

SOURCE_EXECUTION_HEAD=d940e19d497f85315155e52818c4a78a1c1c38e0
REPLAY_AUTHENTICATION_CORRECTION_COMMIT=341e2e59668ea9b575007ddf591424b831170577
RUN_NAME=gen5-ainit-temporal-birth-phase-a-numerical-auth-d940e19-r1
SUMMARY_SHA256=37cab4866c729ad03262df562762ac8b6b91e4e24e2498223483848871ad1df6
TRAJECTORY_SHA256=0f7cd4248faa92223597e0816597b59e426f08829dadd366f9603f56a8de809e

STATUS=PHASE_A_EVIDENCE_READY_TO_FREEZE
PHASE_A_EXECUTION=PASS
PHASE_A_ARTIFACT_PROVENANCE=PASS
PHASE_B_EXECUTED=NO
CONFIRMATORY_9601_9900_LOADED=NO

## 1. Scope

This report freezes the imported Gen5 temporal-birth Phase A evidence only.

It does not execute or interpret the Phase B raw-write task-visible / low-gain
functional scan.

The Phase A evidence answers only trajectory-level questions about the learned
rank-2 write operator and its optimization birth mechanism:

1. whether the operator is structurally zero at `t=0`;
2. whether A-init-specific operator geometry appears immediately after the
   first optimizer update;
3. whether the earliest gradient and operator separation is driven primarily by
   A-init rather than training RNG;
4. whether the authenticated 20-step replay preserves the frozen historical
   trajectory within the pre-registered numerical replay gate.

## 2. Imported artifact authentication

The imported Phase A run passed all provenance and replay-authentication gates.

- cells: `9`
- optimizer steps per cell: `20`
- total optimizer steps: `180`
- historical scalar traces authenticated: `YES`
- final A/B parameters numerically authenticated: `YES`
- final learned operators numerically authenticated: `YES`
- task evaluation executed: `NO`
- Phase B executed: `NO`
- confirmatory `9601..9900` loaded: `NO`

### Replay-authentication margins

Worst observed scalar/tensor/operator deviations remained comfortably inside
the pre-registered bounds.

- maximum loss error / tolerance ratio:
  `0.046875`
  at `A6202-R6202`, step `12`
- maximum total-gradient error / tolerance ratio:
  `0.3203125`
  at `A6203-R6201`, step `14`
- maximum A parameter absolute difference:
  `6.92903995514e-07`
- maximum A relative L2 difference:
  `8.3335261706e-07`
- maximum B parameter absolute difference:
  `2.86498107016e-06`
- maximum B relative L2 difference:
  `3.8076482687e-06`
- maximum learned-operator relative Frobenius difference:
  `3.6981548456e-06`

Against the frozen replay bounds:

- parameter max-abs bound: `1e-5`
- parameter relative-L2 bound: `1e-4`
- operator relative-Frobenius bound: `1e-4`

No replay-authentication rescue or tolerance widening was used.

## 3. Structural zero state at t=0

For all nine A-init x training-RNG cells at `t=0`:

- `A_init_displacement = 0`
- `B_theta_norm = 0`
- `||BA||_F = 0`

Therefore the learned correction operator is exactly zero before the first
optimizer update.

Together with the authenticated runtime semantic result:

- `grad_A_0 = 0` exactly
- `grad_B_0` is finite and nonzero

the first nonzero correction cannot be caused by an ordinary task gradient into
A at step 0.

## 4. Step-0 grad_B factor structure

The exact persisted step-0 `grad_B` tensors show overwhelming A-init dependence.

Grouped squared-distance diagnostic:

- same training RNG / different A-init:
  `0.0841019747344`
- same A-init / different training RNG:
  `1.83561375983e-06`
- A-over-R pair-distance ratio:
  `45816.81428562779`

Full 3x3 factorial decomposition:

- A main fraction:
  `0.999978174169`
- training-RNG main fraction:
  `9.93161383787e-06`
- A-by-R interaction fraction:
  `1.18942176604e-05`
- A-over-R factorial energy ratio:
  `100686.37287888146`

Interpretation:

The step-0 gradient that writes into `B_theta` is already almost entirely
determined by the frozen A initialization. This occurs while `grad_A_0` remains
exactly zero.

Therefore the earliest optimization channel for A-init-specific
representation is:

`A_init -> step-0 grad_B -> first B update -> nonzero BA`

rather than:

`task gradient -> A update -> later representational divergence`.

The approximately `1e-7` step-1 A displacement is consistent with the frozen
AdamW weight-decay path and is not needed to explain the first large B update.

## 5. Learned-operator birth

The grouped `BA` operator distance is exactly zero at `t=0` and strictly
positive at `t=1`.

At `t=1`:

- same RNG / different A mean squared operator distance:
  `0.0320121797348`
- same A / different RNG mean squared operator distance:
  `0.000252991806163`
- A-over-R pair-distance ratio:
  `126.53445271729703`

Factorial fractions at `t=1`:

- A:
  `0.992117783694611`
- R:
  `0.0026280735982473604`
- A-by-R:
  `0.005254142707141562`
- A-over-R factorial energy ratio:
  `377.5076102724999`

Thus the Phase A learned-operator diagnostic birth step is:

`BA_OPERATOR_GEOMETRIC_BIRTH_DIAGNOSTIC_T=1`

The separation is already overwhelmingly A-init driven at its first observed
post-update state.

## 6. Temporal strengthening

A-init dominance strengthens rather than appearing only late.

Selected trajectory points:

### t=1

- A fraction: `0.992117783694611`
- R fraction: `0.0026280735982473604`
- AR fraction: `0.005254142707141562`
- pair-distance A/R ratio: `126.53445271729703`

### t=10

- A fraction: `0.9996029537143251`
- R fraction: `0.00013517340534920745`
- AR fraction: `0.0002618728803259452`
- pair-distance A/R ratio: `2518.2576003564304`

### t=20

- A fraction: `0.9997591943046895`
- R fraction: `9.757307576063599e-05`
- AR fraction: `0.00014323261954969917`
- pair-distance A/R ratio: `4152.320507351518`

Endpoint same-RNG/different-A operator separation is therefore not a late
training-RNG artifact. It is present at the first update and becomes even more
A-dominated across the frozen 20-step optimization trajectory.

## 7. Step-1 parameter geometry

At `t=1`, all cells have:

- A-init displacement approximately `9.66e-08` to `9.87e-08`
- B norm approximately `0.2203`
- `BA` Frobenius norm approximately `0.1252` to `0.1280`

This asymmetry is mechanistically important:

- A has essentially not moved in task-gradient terms;
- B has already acquired a substantial nonzero update;
- the resulting learned write operator is already nonzero and A-init specific.

This is consistent with the direct structural prediction from zero-initialized
B:

- `B_0 = 0`
- `grad_A_0 = 0`
- `grad_B_0` depends on the frozen A-initialized features.

## 8. Bounded Phase A conclusion

The validated Phase A evidence supports the following bounded conclusion:

**A-init-specific learned write-operator geometry is born at the first
optimizer update. Its earliest measurable source is the A-init-dependent
step-0 gradient into the zero-initialized B factor, not an earlier task-gradient
update of A. Training-RNG variation is already much smaller than A-init
variation at this first birth and becomes still less important over the
20-step trajectory.**

This establishes an optimization-time precursor for the previously frozen
endpoint representational non-identifiability.

It does NOT yet establish:

- the Phase B population-level raw-write geometric birth step;
- task-visible / low-gain functional-freedom birth;
- centered-logit or margin gate passage at `t=1`;
- arbitrary perturbation invariance;
- an exact gauge symmetry or global gauge group;
- a null manifold;
- universal behavior across Mamba models, tasks, layers, or distributions;
- performance superiority.

## 9. Phase B implication

Phase A now satisfies the prerequisite for Phase B:

- successful authenticated execution;
- successful collect/import;
- provenance/hash validation;
- stable temporal trajectory evidence.

The next authorized scientific action is the frozen raw-write Phase B scan on
the imported `t=0..20` snapshots.

Phase B must preserve the original fixed sequential rule:

1. authenticate `t=0` raw-write residual as exact zero;
2. locate the first post-update raw-write geometric birth;
3. evaluate the fixed task-visible / low-gain gate from that step forward in
   strictly increasing integer order;
4. stop at the first complete pass;
5. do not tune thresholds, pairs, orientations, populations, or controls after
   seeing the trajectory.
