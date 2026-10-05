# Gen5 A-init Temporal Birth Replay Authentication Correction

BASE_TEMPORAL_AUTHORITY_COMMIT=f87d492d8de0e881c39b35beb3316b6226243e29
ORIGINAL_TEMPORAL_AUTHORITY_COMMIT=20ae761dbff10ad70853b10910cbe12e51e0666a
SOURCE_RECOVERY_IMPLEMENTATION_COMMIT=f87d492d8de0e881c39b35beb3316b6226243e29

STATUS=READY_FOR_REPLAY_AUTHENTICATION_CORRECTION_IMPLEMENTATION
AUTHORITY_CORRECTION=YES
SCIENTIFIC_EXECUTION_ALLOWED=NO_UNTIL_CORRECTION_IMPLEMENTATION_VALIDATED_AND_FROZEN
TRAINING_ALLOWED=NO_UNTIL_CORRECTION_IMPLEMENTATION_VALIDATED_AND_FROZEN
COMMIT_PUSH_ALLOWED=MANUAL_ONLY

## 1. Scope

This correction changes only the Phase A historical replay authentication rule.

It does not change:

- the scientific question;
- the frozen 3x3 A-init x training-RNG population;
- train/dev rows or split identity;
- model/tokenizer snapshot identity;
- parent checkpoint identity;
- optimizer type;
- learning rate;
- weight decay;
- gradient clipping;
- objective;
- step count;
- checkpoint-selection semantics;
- GPU topology;
- prohibition on DDP and cross-GPU scientific reduction;
- confirmatory-population prohibition;
- Phase A before Phase B ordering;
- Phase B task-visible / low-gain gates;
- collection/provenance rules.

## 2. Recovery evidence requiring correction

A clean, pinned recovery run at implementation freeze
`f87d492d8de0e881c39b35beb3316b6226243e29` authenticated the same model,
data, optimizer recipe, CUDA runtime family, and 2x Tesla T4 topology, but
independent CUDA replay was not bitwise identical to the frozen historical
trajectory.

The first observed scalar divergences were:

- `A6201-R6202`, pre-update step 1:
  - historical loss `1.2595824003219604`
  - replay loss `1.25958251953125`
  - absolute difference `1.1920928955078125e-07`
  - historical total preclip grad norm `0.1692083477973938`
  - replay total preclip grad norm `0.16920843720436096`
  - absolute difference `8.940696716308594e-08`

- `A6201-R6201`, pre-update step 4:
  - historical loss `1.2022730112075806`
  - replay loss `1.202272891998291`
  - absolute difference `1.1920928955078125e-07`
  - total preclip grad norm remained exactly equal at
    `0.6824115514755249`.

The step-0 `A6201-R6201` total preclip gradient norm remained exactly equal to
the historical value:

`0.045427411794662476`.

The relevant historical training/runtime code path is unchanged between the
frozen historical execution lineages and the recovery implementation, so the
observed approximately one-float32-ULP scalar drift is treated as numerical
CUDA replay variability, not as evidence of a changed scientific recipe.

No Phase A trajectory science may be interpreted from the failed recovery
runs.

## 3. Why exact tensor identity is corrected

The original authority required, at replay step 20:

- `torch.equal` for `A_theta.weight` and `B_theta.weight`;
- exact tensor SHA256 equality.

That condition is stronger than independent CUDA numerical reproducibility
for this frozen training path.

Exact file SHA256 remains mandatory for all frozen historical checkpoint
files. Exact tensor SHA256 remains recorded for provenance, but replay tensor
SHA equality is no longer a scientific validity requirement.

## 4. Exact invariants that remain exact

The following remain fail-closed exact checks:

1. repository HEAD and authority identity;
2. clean worktree at execution;
3. parent checkpoint file SHA256;
4. frozen historical checkpoint file SHA256;
5. model/tokenizer snapshot revision;
6. 2x Tesla T4 topology and capability contract;
7. train/dev row identity and order;
8. A-init seed and training-RNG seed identity;
9. optimizer = `torch.optim.AdamW`;
10. learning rate = `0.001`;
11. weight decay = `0.0001`;
12. gradient clip norm = `5.0`;
13. no scheduler;
14. final 3-way CE only;
15. exactly 20 optimizer steps;
16. no parent gradients or parent mutation;
17. confirmatory population `9601..9900` not loaded;
18. `B_0` exact zero;
19. reconstructed `A_0` exact identity;
20. `grad_A_0` exact zero;
21. `grad_B_0` finite and nonzero;
22. no Phase B before successful Phase A authentication.

## 5. Numerical replay authentication

Numerical tolerances are pre-registered here and MUST NOT be tuned after
seeing a new replay.

Let float32 machine epsilon be:

`EPS32 = torch.finfo(torch.float32).eps`.

### 5.1 Historical scalar trace gate

For every cell and every pre-update step `0..19`, both of the following must
hold for the historical vs replay values:

- training loss:
  `abs(replay - historical) <= 32 * EPS32 * max(1, abs(historical))`
- total preclip gradient norm:
  `abs(replay - historical) <= 32 * EPS32 * max(1, abs(historical))`

Because all frozen losses and gradient norms in this stage are O(1) or below,
the effective absolute bound is approximately `3.814697265625e-06`.

This gate must be checked for all 180 optimizer steps of the 3x3 replay.

### 5.2 Step-20 parameter gate

For each cell and each of `A_theta.weight`, `B_theta.weight`, compare replay
tensor `x` with frozen historical tensor `x_ref`.

Report:

- tensor SHA256 of replay and reference;
- exact-equality boolean;
- `max_abs = max(abs(x - x_ref))`;
- `l2 = ||x - x_ref||_2`;
- `relative_l2 = ||x - x_ref||_2 / max(||x_ref||_2, EPS32)`.

Require both:

- `max_abs <= 1.0e-5`
- `relative_l2 <= 1.0e-4`.

These bounds are fixed before the corrected scientific replay. They are not
to be enlarged, rescued, swept, or tuned after execution.

### 5.3 Step-20 learned operator gate

For the rank-2 learned write operator:

`W = B_theta @ A_theta`

compute the frozen-vs-replay normalized Frobenius residual without
materializing any unnecessary full hidden-state trajectory:

`||W_replay - W_ref||_F / max(||W_ref||_F, EPS32)`.

Require:

`operator_relative_frobenius_residual <= 1.0e-4`.

This is an additional guard, not a replacement for the parameter gate.

## 6. Scientific-scale separation requirement

The corrected numerical replay tolerances are authentication tolerances only.

They must remain orders of magnitude below the already frozen scientific
representation differences, including the previously observed raw-write
same-RNG/different-A normalized residual near `0.459` and projection-space
A-init separation.

No claim may be made from a difference at or below the authentication
tolerance.

## 7. Failure semantics

If any exact invariant fails:

`TEMPORAL_REPLAY_EXACT_INVARIANT_FAILED`

and stop.

If any scalar trace exceeds the fixed bound:

`TEMPORAL_REPLAY_SCALAR_AUTHENTICATION_FAILED`

and stop.

If any step-20 A/B tensor exceeds either fixed parameter bound:

`TEMPORAL_REPLAY_PARAMETER_AUTHENTICATION_FAILED`

and stop.

If the learned-operator residual exceeds its fixed bound:

`TEMPORAL_REPLAY_OPERATOR_AUTHENTICATION_FAILED`

and stop.

Any such failure blocks Phase A scientific interpretation and Phase B.

No tolerance rescue or post-hoc widening is authorized.

## 8. Corrected success condition

Phase A replay authentication succeeds only if:

- every exact invariant passes;
- every historical scalar trace comparison passes;
- every step-20 A/B parameter comparison passes;
- every step-20 learned-operator comparison passes;
- parent identity is preserved;
- all nine cells finish exactly 20 optimizer steps.

The success marker is:

`GEN5_AINIT_TEMPORAL_BIRTH_PHASE_A_NUMERICAL_REPLAY_AUTHENTICATION_PASS`

Only after this success may Phase A trajectory artifacts be collected,
imported, provenance-authenticated, and scientifically interpreted.

## 9. Implementation scope

Authorized implementation changes are limited to:

- `scripts/audit_reason_router_gen5_ainit_temporal_birth.py`
- `tests/test_reason_router_gen5_ainit_temporal_birth.py`

The implementation must:

- replace replay `torch.equal` / tensor-SHA validity gates with the corrected
  numerical gates above while continuing to report exact-equality and SHA
  diagnostics;
- authenticate the full historical loss and total-preclip-gradient traces for
  every cell;
- compute the step-20 A/B numerical diagnostics;
- compute the step-20 learned-operator residual;
- preserve the post-synchronize instrumentation ordering frozen by the
  recovery implementation;
- preserve all historical runner modes and contracts;
- preserve Phase B as unexecuted until successful Phase A import/freeze.

No other production file may change.

## 10. Validation before execution

Before any corrected scientific replay:

1. targeted unit/static tests must pass;
2. static verify must pass;
3. `cm ship` must show only the authorized implementation files when the
   implementation is frozen;
4. the implementation must be committed and pushed;
5. Kaggle must bootstrap the exact implementation-freeze commit;
6. exact provisioning and 2x T4 CUDA preflight must pass.

## 11. Collection policy

- correction/static validation: `DO_NOT_COLLECT`
- CUDA preflight: `DO_NOT_COLLECT`
- failed numerical replay: `DO_NOT_COLLECT`
- successful corrected Phase A replay: `COLLECT_AND_IMPORT_REQUIRED`

## 12. Phase B boundary

Phase B remains blocked until the corrected Phase A replay:

1. passes numerical replay authentication;
2. is successfully collected;
3. is imported locally;
4. has provenance/hash validation completed;
5. is frozen as Phase A evidence.

No Phase B implementation or execution is authorized before that point.
