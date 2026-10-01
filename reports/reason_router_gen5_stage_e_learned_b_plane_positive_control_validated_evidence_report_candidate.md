# Gen5 Stage E Learned-B-Plane Positive-Control Validated Evidence Report

## Status

VALIDATED_EVIDENCE

This report records the validated learned-B-plane positive-control result for
Gen5 Stage E.

It is an evidence interpretation report, not a new execution authority.

## Execution identity

Execution commit:

`467148d074f7cbd91a5e107a520df21ed4fe508c`

Implementation freeze:

`b8fa20dc3fb058412610375c255d8c25e93e14bf`

Run:

`gen5-stagee-bfree-three-cell-467148d-r1`

Imported handoff ZIP SHA256:

`c56bff23c15585642b716de6c696c4e01dfc60d19e0a01c1aa59dc0e0ce055e3`

Command SHA256:

`25c10d4269ebeaa2387531fbbb60fc600bdca3a11c735cefc6897b7e1c75a13b`

The collector reported 11 files and EXIT_CODE=0.

`cm import` validated and copied all 11 files.

## Frozen execution contract

Arm:

`E-BFREE`

Pressure:

`P0`

Seeds:

- 6201
- 6202
- 6203

GPU topology:

`TWO_INDEPENDENT_SINGLE_GPU_WORKERS_NO_DDP`

Worker allocation:

- GPU worker 0: seeds 6201 and 6203
- GPU worker 1: seed 6202

Each cell used exactly 20 AdamW optimizer steps.

Total optimizer steps:

`60`

Trainable parameter count per cell:

`1540`

The fixed output plane for each seed was the span of that seed's already
trained unrestricted Phase3A P0 final B matrix.

The plane was derived by the frozen deterministic float64 thin-QR procedure
under the corrected single-thread CPU linear-algebra execution contract.

Confirmatory IDs 9601--9900 were not loaded.

No scientific p-values were computed.

## Provenance and execution validation

The run passed:

- exact execution HEAD authentication;
- exact parent checkpoint SHA256 authentication;
- exact model/tokenizer snapshot authentication;
- exact frozen CUDA-kernel binary authentication;
- corrected seed-specific BFREE Q identity authentication;
- two Tesla T4 topology authentication;
- matrix execution;
- artifact inventory validation;
- collector validation;
- local import validation.

All three cells reported:

- effective output rank = 2;
- effective operator rank = 2;
- parent parameter fingerprint unchanged;
- fixed-plane residual approximately zero;
- exactly 20 optimizer steps;
- task evaluation executed;
- confirmatory set not loaded.

Therefore code/runtime execution success and artifact/provenance validity are
established for this matrix.

## Primary result

Frozen Stage E comparison means:

| Arm | Mean recovery vs unrestricted seed-matched P0 gain |
| --- | ---: |
| E-R22 | 0.00213934506552454 |
| E-C22 | 0.00186063347357193 |
| E-BFREE | 0.07142508872336398 |

Thus:

- BFREE / R22 = 33.3864274045251
- BFREE / C22 = 38.3875114244003
- BFREE - R22 = 0.0692857436578394
- BFREE - C22 = 0.0695644552497920

The BFREE recovery advantage over R22 and C22 was directionally consistent
for every seed.

### seed6201

- BFREE recovery = 0.0690419274517625
- dev CE = 1.30654907227
- dev accuracy = 0.269047619048
- BFREE / mean R22 recovery = 32.2724597188039
- BFREE / mean C22 recovery = 37.1066781461370

### seed6202

- BFREE recovery = 0.0737218360466545
- dev CE = 1.30395638943
- dev accuracy = 0.270238095238
- BFREE / mean R22 recovery = 34.4600023786153
- BFREE / mean C22 recovery = 39.6219014081951

### seed6203

- BFREE recovery = 0.0715115026716749
- dev CE = 1.30524826050
- dev accuracy = 0.270238095238
- BFREE / mean R22 recovery = 33.4268201161560
- BFREE / mean C22 recovery = 38.4339547188687

Mean BFREE gain vs ZERO was:

`0.0357142686843872`

Mean BFREE dev accuracy was:

`0.269841269841270`

## Interpretation

The learned-B-plane positive control resolves only part of the ambiguity left
by the original Stage E experiment.

Under the same fixed-plane QMA parameterization, optimizer, 20-step horizon,
P0 data, and seed-matched execution contract, E-BFREE recovered substantially
more task benefit than either E-R22 or E-C22.

Mean recovery was:

- E-R22: 0.00213934506552454
- E-C22: 0.00186063347357193
- E-BFREE: 0.07142508872336398

Thus E-BFREE exceeded the R22 and C22 recovery levels by approximately
33.39x and 38.39x respectively, with the same direction in all three seeds.

This establishes that output-plane orientation materially affects short-horizon
optimization accessibility.

However, the absolute E-BFREE recovery remained only approximately 7.14% of
the unrestricted seed-matched P0 gain.

Therefore this positive control does not establish that fixing the learned-B
output span is sufficient to reproduce the unrestricted learned correction.

Approximately 92.9% of unrestricted gain remained unrecovered.

The remaining gap cannot yet be attributed uniquely to any one of:

1. relearning the input/read-side A geometry from its restart initialization;
2. gauge or scale conditioning introduced by the orthonormal QMA
   parameterization;
3. short-horizon 20-step restart optimization dynamics.

Accordingly, the proposition that R22/C22 fail primarily because their output
orientation is intrinsically task-unusable is not established.

The supported result is narrower:

`LEARNED_B_OUTPUT_SPAN_IS_SUBSTANTIALLY_MORE_OPTIMIZER_ACCESSIBLE_THAN_R22_OR_C22_UNDER_THE_MATCHED_STAGE_E_RESTART_CONTRACT`

but:

`OUTPUT_SPAN_ALONE_DOES_NOT_EXPLAIN_THE_UNRESTRICTED_SOLUTION`

No inferential claim is made from the three-seed descriptive matrix.

## Stage E status

Stage E remains scientifically open.

No random-plane, rank, layer, learning-rate, optimizer, or training-horizon
sweep is warranted.

One additional bounded diagnostic is required.

## Required next diagnostic: E-BFREE-AINIT

The next experiment must preserve the existing E-BFREE output-plane contract
and change only the initialization of A.

For each seed:

- Q_B is the frozen seed-matched unrestricted Phase3A P0 final-B span;
- M is a trainable 2x2 matrix initialized to zero;
- A is initialized from the same seed's unrestricted Phase3A P0 final
  A_theta.weight;
- A and M remain trainable;
- total trainable parameter count remains 1540;
- optimizer, learning rate, weight decay, data, split, pressure, and 20-step
  horizon remain identical to E-BFREE;
- confirmatory IDs 9601--9900 remain forbidden.

Because M is initialized to zero, the initial correction output remains
exactly zero. Initializing A from the unrestricted source therefore does not
inject the unrestricted task correction at step 0.

Before execution, a static representability check must verify that if

B_free = Q_B R_B,

then the frozen parameterization with

M_exact = R_B

and

A = A_free

reconstructs B_free A_free to numerical precision.

This establishes that the parameterization can represent the unrestricted
final operator exactly before testing whether the 20-step optimizer can
recover it from M=0.

## Interpretation of the required diagnostic

If E-BFREE-AINIT materially exceeds the existing E-BFREE recovery across the
three matched seeds, then relearning A/read-side geometry is a major source of
the remaining fixed-plane optimization bottleneck.

If E-BFREE-AINIT remains near the existing E-BFREE recovery, then A
reinitialization is not the main explanation and the remaining ambiguity
shifts toward QMA scale/gauge conditioning or short-horizon optimization
dynamics.

If outcomes are materially seed-dependent, the ambiguity remains seed
dependent.

No post-hoc success threshold or p-value is authorized.

## Gen5 status

The broader Gen5 synthesis remains supported:

`NATIVE_CAUSAL_IMPORTANCE_DOES_NOT_IMPLY_OPTIMIZATION_PRIVILEGE`

but Gen5 should not yet be declared experimentally closed.

The immediate next step is the single E-BFREE-AINIT diagnostic above.

Architectural Gen6 work remains premature until this remaining Stage E
optimization-accessibility ambiguity is resolved.

This report does not authorize E-BFREE-AINIT execution.
