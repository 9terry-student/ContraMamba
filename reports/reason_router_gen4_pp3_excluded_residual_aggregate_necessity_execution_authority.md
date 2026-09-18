# Gen4 PP3-Excluded Residual Aggregate Necessity — Scientific Execution Authority

## Status

`SCIENTIFIC_EXECUTION_AUTHORIZED_AFTER_THIS_FILE_IS_FROZEN`

This document authorizes exactly one raw scientific execution of the already
frozen PP3-excluded residual aggregate necessity experiment.

It does not authorize confirmatory statistical inference, rescue analysis,
subgroup analysis, alternative controls, additional p-values, training,
backward passes, task-head evaluation, logits analysis, or any scientific
conclusion from the raw run alone.

## Frozen implementation

Implementation commit:

`f8226d9bc0b075d1ce477389d3a3171e7e7c2470`

Implementation files:

- `scripts/reason_router_gen4_pp3_excluded_residual_aggregate_necessity_fast_cuda.py`
- `tests/test_reason_router_gen4_pp3_excluded_residual_aggregate_necessity_fast_cuda.py`

Implementation commit changed exactly those two files.

Observed local implementation validation before freeze:

- targeted test file: `12 passed`
- Python compile validation: PASS
- `git diff --check`: PASS
- scientific model forwards during implementation validation: `0`
- GPU scientific execution during implementation validation: `0`

Runner blob at implementation commit:

`be0e1b6c2a52dbdf2481b7a332d175797f43b471`

Test blob at implementation commit:

`f0ad23ea58ae6cd46a5c70ce922e11a765d09eee`

## Upstream frozen chain

Prospective design:

`c4518d20f4417ca9f057fbd4940c28539e4ffb2c`

Static preparation:

`4a7698264488e811370bdf071c3cde73735757e0`

Implementation authority:

`7356e81d34b2883e74b8fa24b7751f725d9ca1db`

Implementation:

`f8226d9bc0b075d1ce477389d3a3171e7e7c2470`

The execution must use a commit in which all four commits above are ancestors
and the runner/test bytes are unchanged from `f8226d9`.

## Scientific population

Exact source-pair range:

`xg1_fact_1501..xg1_fact_1800`

Pair count:

`300`

Rows:

`1800`

No pair replacement, dropping, filtering, subgrouping, or outcome-dependent
selection is authorized.

## Conditions

Exact condition order:

1. `native`
2. `residual_neutralized`
3. `quarter_turn_control`

Residual plane set:

`{P1, P2, P4, P5}`

PP3 remains excluded from the treatment/control residual subspace and must
remain invariant within the frozen runtime tolerance.

## Primary raw endpoint

Per item:

- `Q0 = Q(native)`
- `QR = Q(residual_neutralized)`
- `QC = Q(quarter_turn_control)`
- `A_R = Q0 - QR`
- `A_C = Q0 - QC`
- `D_RES_NEC = A_R - A_C = QC - QR`

The raw runner records these values only.

The raw runner must not calculate the one-sample t-test, a p-value, positive
label, or final scientific conclusion.

## Exact forward budget

Per direction:

`4` scientific model forwards

Per condition:

`10 × 4 = 40`

Per pair:

`3 × 40 = 120`

Total:

`300 × 120 = 36000`

Baseline model forwards:

`0`

Any observed budget other than exactly 36000 scientific forwards and 0 baseline
forwards invalidates the run.

## Two-GPU contract

Exactly two independent workers.

No DDP.

No NCCL.

GPU 0:

- `xg1_fact_1501..xg1_fact_1650`
- 150 pairs
- 18000 scientific forwards

GPU 1:

- `xg1_fact_1651..xg1_fact_1800`
- 150 pairs
- 18000 scientific forwards

Canonical merged order:

`xg1_fact_1501..xg1_fact_1800`

Each worker loads the same representative model/checkpoint exactly once through
the already validated runtime path.

## Frozen model/checkpoint/runtime

Model:

`state-spaces/mamba-130m-hf`

Model/tokenizer snapshot:

`/kaggle/working/contramamba_runtime/models--state-spaces--mamba-130m-hf/snapshots/40e5d2bd7452abb3ca8fadbafe9131ee0e2c2f37`

Checkpoint:

`/kaggle/input/datasets/terryterry9/checkpoint/selected_checkpoint.pt`

Representative checkpoint SHA256:

`1ff3fcf2ebd754ab6f9483d6a9982b9b04b9a4eb3357f9f8cdbe2b30399e7d2f`

Representative checkpoint bytes:

`518270455`

Expected execution environment:

- Python `3.12.13`
- NumPy `2.0.2`
- Torch `2.10.0+cu128`
- Transformers `5.0.0`
- CUDA runtime `12.8`
- GPU: Tesla T4
- compute capability `7.5`
- `kernels==0.10.2`

Frozen Mamba kernel:

- scientific revision:
  `c8ffc584c147878a6eb978ae0e8db4d116c93a8c`
- binary SHA256:
  `dc4d76a6323b510e77cfb66b5aa7bb0086c8f5cba238002b9c20bc31ea706587`

Frozen causal-conv1d kernel:

- scientific revision:
  `f2651e776f66069cdcf842840db637583def1223`
- binary SHA256:
  `6b013d7b9a033bb9b0a2a714b26470e1aaba4af9bf1b3ec7442c2a53afb6b7b6`

Build variant:

`torch210-cxx11-cu128-x86_64-linux`

The already validated kernel transport/cache compatibility path must be reused.
Any runtime, kernel, checkpoint, snapshot, or device mismatch blocks execution
before scientific forwards.

## Static provenance requirements

The runner must authenticate the frozen static inputs already encoded in the
implementation, including:

- XG1 `1501..1800` source/row identities;
- tokenizer anchor/eligibility identities;
- geometry/preparation manifest identities;
- all P1/P2/P4/P5 vector SHA256 identities;
- frozen PP3 vector identities;
- XG2/XG4 basis plan identities.

No static input may be regenerated or substituted in Kaggle.

## Kaggle bootstrap requirement

The Kaggle notebook currently contains a prior working clone.

Therefore this execution must use **fresh bootstrap mode**:

`cm kaggle fresh`

The bootstrap cell must:

1. remove only the existing `/kaggle/working/ContraMamba` working clone;
2. clone the repository again;
3. fetch the exact execution-freeze commit;
4. checkout that exact commit;
5. verify exact full HEAD identity;
6. verify a clean working tree.

Do not reuse the existing Kaggle working clone for this run.

Do not delete `/kaggle/input` datasets.

Do not delete prior downloaded runtime/kernel caches unless the frozen
compatibility gate itself reports a mismatch.

GPU must remain OFF during fresh bootstrap and CPU/preflight steps.

## GPU/session control

GPU OFF:

- fresh bootstrap;
- repository identity verification;
- CPU-only preflight.

GPU ON:

- only immediately before the authorized two-GPU raw scientific execution.

After the raw run finishes or fails:

- turn GPU OFF promptly;
- preserve logs/artifacts;
- collect the registered run before discarding useful failure state.

## Raw result boundary

A successful raw run must report:

- result:
  `PASS_PP3_EXCLUDED_RESIDUAL_AGGREGATE_NECESSITY_RAW_OBSERVATION`
- GPU count: `2`
- shard 0: `1501..1650`, 18000 forwards
- shard 1: `1651..1800`, 18000 forwards
- total scientific forwards: `36000`
- baseline forwards: `0`
- primary inference executed: `False`
- multiplicity correction executed: `False`
- training executed: `False`
- backward executed: `False`
- task heads executed: `False`
- logits read: `False`
- scientific conclusion: `None`

Raw execution success is not the scientific conclusion.

## Failure boundary

If either worker, runtime gate, kernel gate, checkpoint gate, provenance gate,
artifact validation, budget validation, or merge validation fails:

- do not infer a scientific result;
- do not rerun under modified parameters;
- preserve the failure log;
- collect the registered run if possible;
- diagnose under the existing frozen design before authorizing any retry.

## Post-run sequence

After a raw run:

1. collect the registered run;
2. import the handoff locally;
3. validate provenance/hash/artifact integrity;
4. only then execute the already frozen single confirmatory inference:
   one-sample Student t-test, one-sided greater, `N=300`, `df=299`,
   `alpha=0.05`, exactly one p-value.

No confirmatory inference is authorized inside Kaggle raw execution.

## Current authority

Scientific raw execution: `YES`

Kaggle two-GPU execution: `YES`

Training/backward: `NO`

Primary statistical inference during raw run: `NO`

Scientific conclusion during raw run: `NO`

Additional p-values/rescue analyses: `NO`

Commit/push of this execution authority: manual only.
