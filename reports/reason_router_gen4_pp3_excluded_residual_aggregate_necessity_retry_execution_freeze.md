# Gen4 PP3-Excluded Residual Aggregate Necessity — Retry Execution Freeze

## Status

`RETRY_RAW_SCIENTIFIC_EXECUTION_AUTHORIZED_AFTER_THIS_FILE_IS_FROZEN`

This document authorizes exactly one retry of the already frozen raw scientific
execution after correction of the endpoint floating-point identity defect.

It does not change the scientific design, population, conditions, intervention,
forward budget, artifact schema, confirmatory inference, or interpretation
boundary.

## Frozen scientific chain

Prospective design:

`c4518d20f4417ca9f057fbd4940c28539e4ffb2c`

Static preparation:

`4a7698264488e811370bdf071c3cde73735757e0`

Implementation authority:

`7356e81d34b2883e74b8fa24b7751f725d9ca1db`

Original implementation:

`f8226d9bc0b075d1ce477389d3a3171e7e7c2470`

Original execution authority:

`02f6f862e2177e0d1e2ee58367bf916be0569094`

Endpoint defect correction:

`d8cd5d63f528fb397cca10acde149b9b96a261cc`

## Failed execution preserved

Failed run:

`g4k-residual-aggregate-necessity-xg1-1501-1800-02f6f86-retry1`

Execution HEAD:

`02f6f862e2177e0d1e2ee58367bf916be0569094`

Command SHA256:

`ce89f204f23a926f2fca83a4741e5c63d2416f45580f60c86413dc29ce841c15`

Run log SHA256:

`ad1ec8ef6358cc8fd2ebf7e29301210fd7c91637f62aaa319754d411b2500dc0`

Run meta SHA256:

`e8727f67a325be416c0f9547dbfd70aa50464592c28a47eb8d2aa9f3f954c3ad`

Imported handoff ZIP SHA256:

`bf1ec2d0df6c16724eb716120ba7e13e7116c518a22f83eb6363db7e05ed1e66`

Exit code:

`1`

Collected scientific artifact files:

`0`

The failure occurred before a pair item could be appended or a shard payload
could be written. No raw scientific artifact or scientific conclusion was
produced by the failed run.

## Defect

The frozen endpoint is mathematically:

`D_RES_NEC = (Q0 - QR) - (Q0 - QC) = QC - QR`

The original runner computed:

`d = (Q0 - QR) - (Q0 - QC)`

and then required bitwise Python-float equality with:

`QC - QR`

That redundant exact-equality assertion is invalid under IEEE-754 because
mathematically equivalent subtraction paths can differ in their final floating
representation.

Both workers failed on this internal assertion:

`D_RES_NEC_INTERNAL`

This is an implementation defect, not a scientific result.

## Corrected implementation

Correction commit:

`d8cd5d63f528fb397cca10acde149b9b96a261cc`

Changed files are exactly:

- `scripts/reason_router_gen4_pp3_excluded_residual_aggregate_necessity_fast_cuda.py`
- `tests/test_reason_router_gen4_pp3_excluded_residual_aggregate_necessity_fast_cuda.py`

Corrected runner blob:

`be859d754fc3d5f42e6e40925a9f83fd64dc48b8`

Corrected test blob:

`42c416746d700ff828d922d50cd796508b844a57`

The corrected endpoint implementation keeps:

- `A_R = Q0 - QR`
- `A_C = Q0 - QC`

and uses the frozen canonical endpoint coordinate:

`D_RES_NEC = QC - QR`

The redundant exact-float equality assertion is removed.

No tolerance is introduced.

No endpoint is changed.

No outcome-dependent correction is introduced.

## Regression validation

Local CPU validation after the correction:

- targeted test suite: `13 passed`
- Python compile validation: PASS
- `git diff --check`: PASS
- scientific model forwards: `0`
- checkpoint loads: `0`
- GPU scientific execution: `0`

A regression test explicitly verifies a cancellation-sensitive input for which:

`(Q0 - QR) - (Q0 - QC) != QC - QR`

at the Python-float bitwise comparison level, while the corrected endpoint
returns the canonical frozen value `QC - QR`.

## Retry execution contract

The retry uses exactly the original frozen scientific contract:

Population:

`xg1_fact_1501..xg1_fact_1800`

Pair count:

`300`

Conditions, in order:

1. `native`
2. `residual_neutralized`
3. `quarter_turn_control`

Residual plane set:

`{P1, P2, P4, P5}`

Excluded plane:

`P3`

Per-item endpoint:

- `Q0`
- `QR`
- `QC`
- `A_R = Q0 - QR`
- `A_C = Q0 - QC`
- `D_RES_NEC = QC - QR`

Exactly two independent GPU workers:

- GPU 0: `xg1_fact_1501..xg1_fact_1650`
- GPU 1: `xg1_fact_1651..xg1_fact_1800`

Forward budget:

- GPU 0: `18000`
- GPU 1: `18000`
- total scientific forwards: `36000`
- baseline forwards: `0`

No DDP.

No NCCL.

No training.

No backward.

No task-head evaluation.

No logits analysis.

No primary statistical inference during the raw run.

No p-value during the raw run.

No scientific conclusion during the raw run.

## Runtime contract

The retry must use the same previously validated runtime contract:

- Python `3.12.13`
- NumPy `2.0.2`
- Torch `2.10.0+cu128`
- Transformers `5.0.0`
- Tokenizers `0.22.2`
- `kernels==0.10.2`
- CUDA runtime `12.8`
- Tesla T4 × 2
- compute capability `7.5`

Representative checkpoint:

`/kaggle/input/datasets/terryterry9/checkpoint/selected_checkpoint.pt`

Checkpoint SHA256:

`1ff3fcf2ebd754ab6f9483d6a9982b9b04b9a4eb3357f9f8cdbe2b30399e7d2f`

Model/tokenizer snapshot revision:

`40e5d2bd7452abb3ca8fadbafe9131ee0e2c2f37`

Exact frozen kernel binary identity must again pass before scientific execution.

## Bootstrap requirement

Because the retry will execute from a new freeze commit, Kaggle must use:

`cm kaggle fresh`

The retry must not execute from the stale `02f6f86` working clone.

The fresh bootstrap must checkout the exact retry-execution-freeze commit,
verify full HEAD identity, and verify a clean worktree.

GPU remains OFF during fresh bootstrap.

GPU is turned ON only for strict runtime/kernel preflight and the raw retry.

## Invocation requirement

The corrected runner must be invoked as a module from repository root:

`python -u -m scripts.reason_router_gen4_pp3_excluded_residual_aggregate_necessity_fast_cuda`

Do not invoke the file directly as:

`python scripts/...py`

because that direct invocation does not establish the repository package import
root required by `from scripts import ...`.

## Retry run identity

Use a new single-use run name ending in:

`retry2`

Do not reuse either prior run identity.

Do not reuse prior output directories.

## Raw success boundary

The retry is successful only if the frozen runner reports:

`PASS_PP3_EXCLUDED_RESIDUAL_AGGREGATE_NECESSITY_RAW_OBSERVATION`

with:

- GPU count `2`
- shard 0 pair range `1501..1650`
- shard 1 pair range `1651..1800`
- total scientific forwards `36000`
- baseline forwards `0`
- primary inference executed `False`
- multiplicity correction executed `False`
- training executed `False`
- backward executed `False`
- task heads executed `False`
- logits read `False`
- scientific conclusion `None`

Raw execution success alone is not the confirmatory scientific conclusion.

## Failure boundary

Any provenance, runtime, kernel, checkpoint, budget, intervention-audit,
endpoint, shard, merge, or artifact failure blocks interpretation.

Do not change parameters or introduce rescue analyses in response to a retry
failure.

Preserve and collect any retry failure before further diagnosis.

## Post-run boundary

After a successful retry:

1. turn GPU OFF;
2. collect the run;
3. import the handoff locally;
4. validate provenance/hash/artifact integrity;
5. only then execute the frozen single confirmatory inference:
   one-sample Student t-test, one-sided greater, `N=300`, `df=299`,
   `alpha=0.05`, exactly one p-value.

## Current authority

Corrected raw retry execution: `YES`

Kaggle two-GPU retry: `YES`

Training/backward: `NO`

Primary inference during raw run: `NO`

Scientific conclusion during raw run: `NO`

Additional p-values/rescue analyses: `NO`

Commit/push of this retry freeze: manual only.
