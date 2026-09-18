# Gen4 PP3-Excluded Residual Individual-Plane Necessity Localization — Retry1 Execution Freeze

## Status

`RETRY1_SCIENTIFIC_EXECUTION_AUTHORIZED_AFTER_ZERO_FORWARD_COMMAND_ORCHESTRATION_FAILURE`

This document authorizes one retry of the already frozen raw scientific
execution after the first pinned run failed before preflight and before any
scientific model forward.

No scientific design, implementation, static input, endpoint, population,
condition, multiplicity, or interpretation change is authorized.

## Parent execution freeze

Parent execution-freeze commit:

`f6cd2dca4b76e54ba603877d5730001bd078ea2e`

The retry-freeze commit must have this commit as its parent and must change
exactly this retry-freeze document.

The scientific retry execution HEAD is the resulting retry-freeze commit SHA.

## Failed attempt provenance

Failed run name:

`g4k-residual-individual-plane-necessity-xg1-1801-2100-f6cd2dc`

Expected and actual repository commit:

`f6cd2dca4b76e54ba603877d5730001bd078ea2e`

Pinned wrapper command SHA256:

`939da891baea3d0e48ca2abedc84701ce459e4cc8f8625e1c1932ca0864cdfa1`

Started UTC:

`2026-09-18T07:36:28Z`

Finished UTC:

`2026-09-18T07:36:28Z`

Exit code:

`2`

Failure class:

`LOCAL_CLIPBOARD_COMMAND_ORCHESTRATION_ERROR`

The command bytes saved under the run name were PowerShell orchestration
commands rather than the intended Kaggle Bash preflight-and-run command.
Kaggle Bash therefore failed immediately on `Set-Location` and subsequent
PowerShell syntax.

The failure occurred before:

- repository scientific preflight Python;
- model snapshot validation by the intended command;
- checkpoint-backed model load;
- scientific model forwards;
- raw artifact creation;
- primary inference;
- multiplicity correction.

Scientific model forward count for the failed attempt:

`0`

Primary inference executed:

`False`

Multiplicity correction executed:

`False`

Scientific conclusion:

`None`

The failed run identity and its Kaggle run log / metadata / command file must
remain preserved. They must not be reused as a successful run identity.

## Retry identity

Authorized retry run name:

`g4k-residual-individual-plane-necessity-xg1-1801-2100-<retry-execution-short-sha>-retry1`

The exact short SHA is derived from the retry-freeze commit created by this
document.

The retry must use a newly generated pinned command whose command bytes are the
intended Bash preflight-and-run payload, not PowerShell orchestration.

## Scientific contract unchanged

Fresh population:

`xg1_fact_1801..xg1_fact_2100`

Residual planes:

`[P1, P2, P4, P5]`

Conditions:

1. `native`
2. `p1_neutralized`
3. `p1_quarter_turn_control`
4. `p2_neutralized`
5. `p2_quarter_turn_control`
6. `p4_neutralized`
7. `p4_quarter_turn_control`
8. `p5_neutralized`
9. `p5_quarter_turn_control`

Primary per-plane raw endpoint:

`D_k = QC,k - QN,k`

Total scientific model forward budget:

`108000`

Baseline model forwards:

`0`

Two-GPU topology:

- GPU 0: `xg1_fact_1801..xg1_fact_1950`, `54000` forwards
- GPU 1: `xg1_fact_1951..xg1_fact_2100`, `54000` forwards

No DDP.

No NCCL.

## Retry preflight

The retry command must fail closed before any scientific forward unless all of
the following pass:

- exact retry execution HEAD;
- clean repository worktree;
- frozen static-input identities;
- runner Git blob `9cc0f41b24953d94f870aa3e07b993fafd3d83ea`;
- test Git blob `f81c9a97fc4579773e521261708042722e0b1cd6`;
- runner SHA256
  `56b919e54c8cef1d2220abe9652ffa48a77b42d00d28a64890f0ee60807a176d`;
- test SHA256
  `cc935c0366c8b192510159ae65a464ae3eda703581e00659e42178a63fdef410`;
- frozen residual vectors and XG2/XG4 bases;
- exact tokenizer eligibility;
- exact representative checkpoint SHA;
- two accepted T4 GPUs;
- accepted runtime and frozen kernel identities;
- new retry output directory absent.

Preflight must report:

`SCIENTIFIC_MODEL_FORWARD_COUNT=0`

`PRIMARY_INFERENCE_EXECUTED=False`

`MULTIPLICITY_CORRECTION_EXECUTED=False`

## Raw execution boundary

The retry raw runner remains forbidden from computing:

- Student t statistics;
- p-values;
- Holm decisions;
- adjusted p-values;
- supported-plane sets;
- family-level scientific labels;
- plane rankings.

A successful retry must still end with:

`PRIMARY_INFERENCE_EXECUTED=False`

`MULTIPLICITY_CORRECTION_EXECUTED=False`

`SCIENTIFIC_CONCLUSION=None`

## Later inference boundary

Only after successful collect/import and raw artifact validation may the
separate CPU-only confirmatory inference compute exactly four one-sided
Student t-test p-values, one for each of P1/P2/P4/P5, followed by the already
frozen Holm step-down procedure at familywise alpha `0.05`.

No fifth p-value is authorized.

## Failure handling

If retry1 fails before a valid raw artifact is completed:

- preserve retry1 run provenance;
- do not overwrite or reuse retry1;
- do not interpret partial outputs scientifically;
- diagnose the failure before any retry2 authorization.

## Current authorization

Design freeze: `YES`

Static preparation: `YES`

Implementation freeze: `YES`

Original execution freeze: `YES`

Failed original run scientific forwards: `0`

Retry1 raw scientific execution at the exact retry-freeze commit: `YES`

Training: `NO`

Backward: `NO`

Task-head evaluation: `NO`

GPU statistical inference: `NO`

Commit / push of this retry-freeze document: manual only.
